"""Traduce un .srt con Claude manteniendo numeración y tiempos intactos.

1. Lee el .srt completo y arma una ficha de la obra (personajes, relaciones, trato de tú o
   usted, glosario, errores de transcripción). Se guarda en nombre.ficha.md para revisarla o
   editarla; si ya existe, se reutiliza.
2. Traduce en bloques, varios a la vez (cada uno con algunas líneas vecinas del original como
   contexto y la ficha en el prompt), y pide la respuesta en JSON con un esquema fijo, así cada
   traducción vuelve con su id.
Requiere la variable de entorno ANTHROPIC_API_KEY.
"""
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import anthropic

MODEL = "claude-opus-5-5"
EFFORT = "medium"     # low/medium/high: más alto = más cuidadoso, más lento y caro
BATCH = 40            # subtítulos por llamada
CONTEXT = 8           # subtítulos vecinos (antes y después, en el original) que se envían como contexto
WORKERS = 6           # bloques que se traducen en paralelo
LINE_CHARS = 42

USO = {"input": 0, "output": 0}   # tokens acumulados en esta sesión
_LOCK = threading.Lock()

BLOCK = re.compile(r"(\d+)\s*\n(\d+:\d+:\d+[,.]\d+\s*-->\s*\d+:\d+:\d+[,.]\d+)[^\n]*\n(.*?)(?=\n\s*\n|\Z)", re.S)

SYSTEM = """Eres un traductor profesional de subtítulos de cine y series.
Traduce cada subtítulo al {idioma}.

Reglas:
- Traduce el sentido y el registro (coloquial, formal, insultos, humor), no palabra por palabra.
- Un subtítulo traducido por cada id recibido, con el mismo id. No unas ni partas subtítulos,
  aunque una frase continúe en el siguiente: tradúcela de forma que se lea bien repartida.
- Sé conciso: el texto debe poder leerse en el tiempo que dura el subtítulo. Máximo 2 líneas
  de unos {chars} caracteres; usa \\n para separar líneas.
- No omitas contenido: todo lo que se dice en el original debe estar en la traducción.
- Si en un mismo subtítulo hablan dos personajes, pon cada uno en su línea con guion
  ("-¿Corriendo?\n-Sí."); en ese caso cada línea puede llegar a ~55 caracteres. Si aun así no
  cabe, abrevia la forma, nunca el contenido. Mantén las etiquetas de formato (<i>, </i>, <b>...).
- Los nombres propios no se traducen.
- Si un subtítulo es solo un sonido o una interjección, adáptalo ("Oh" → "Oh", "Huh?" → "¿Eh?").
- El texto viene de un reconocimiento de voz automático y puede tener errores: si una palabra
  no tiene sentido, usa el contexto para inferir lo que se dijo.
- Respeta la ficha de la obra (si se incluye): nombres, trato de tú/usted y glosario.
{notas}{ficha}"""

FICHA_PROMPT = """Abajo está la transcripción automática completa de los subtítulos de una obra
(película o serie), que luego se traducirá al {idioma} en bloques. Prepara una ficha breve que el
traductor tendrá a la vista en cada bloque, para mantener la coherencia de toda la obra.
{notas}
Incluye, en markdown y sin repetir el diálogo:
- **Obra**: de qué trata, época y lugar, tono (y título si se puede deducir).
- **Personajes**: nombre (con la grafía correcta), quién es, edad aproximada y forma de hablar.
- **Relaciones y trato**: quién se trata de tú y quién de usted (o de "señor/señora") en el
  {idioma}, y cómo cambia si cambia a lo largo de la obra.
- **Glosario**: lugares, apodos, términos recurrentes y cómo traducirlos (o si no se traducen).
- **Errores de transcripción**: palabras que el reconocimiento de voz escuchó mal y lo que
  probablemente se dijo (por ejemplo, un nombre escrito de dos formas).

Sé concreto y breve (máximo ~500 palabras). Si algo no se puede deducir, omítelo.

<subtitulos>
{texto}
</subtitulos>"""

SCHEMA = {
    "type": "object",
    "properties": {
        "subtitulos": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"id": {"type": "integer"}, "texto": {"type": "string"}},
                "required": ["id", "texto"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["subtitulos"],
    "additionalProperties": False,
}


def cliente():
    """Cliente con ANTHROPIC_API_KEY; ANTHROPIC_WORKSPACE_ID solo si la key no tiene workspace."""
    ws = os.environ.get("ANTHROPIC_WORKSPACE_ID")
    # Más reintentos que el default: con varios bloques en paralelo es más fácil topar el rate limit (429).
    return anthropic.Anthropic(default_headers={"anthropic-workspace-id": ws} if ws else None, max_retries=6)


def leer_srt(path):
    raw = open(path, "rb").read()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")
    text = text.replace("\r\n", "\n")
    return [{"id": int(n), "tiempo": t.strip(), "texto": tx.strip()} for n, t, tx in BLOCK.findall(text)]


def _rearmar(texto):
    """Respeta los saltos del modelo, salvo que alguna línea quede demasiado larga."""
    lineas = [l.strip() for l in texto.replace("\\n", "\n").split("\n") if l.strip()]
    if all(len(l) <= LINE_CHARS + 6 for l in lineas) and len(lineas) <= 2:
        return "\n".join(lineas)
    if any(l.startswith("-") for l in lineas):          # diálogo: no mezclar hablantes
        return "\n".join(lineas)
    plano = " ".join(lineas)
    if len(plano) <= LINE_CHARS:
        return plano
    mid = len(plano) // 2
    cut = min((m.start() for m in re.finditer(" ", plano)), key=lambda p: abs(p - mid), default=mid)
    return plano[:cut].strip() + "\n" + plano[cut:].strip()


def escribir_srt(subs, path):
    with open(path, "w", encoding="utf-8") as f:
        for i, s in enumerate(subs, 1):
            f.write(f"{i}\n{s['tiempo']}\n{s['texto']}\n\n")


def _llamar(client, **kw):
    """Una llamada a Claude; acumula tokens y valida stop_reason. Devuelve el texto."""
    kw = dict(model=MODEL, max_tokens=16000, **kw)
    if "haiku" in MODEL:                        # Haiku no tiene modelo de respaldo
        response = client.messages.create(**kw)
    else:
        # Si el modelo rechaza la petición, la API la reintenta con el modelo de respaldo recomendado.
        response = client.beta.messages.create(**kw, betas=["server-side-fallback-2026-07-01"],
                                               fallbacks="default")
    with _LOCK:
        USO["input"] += response.usage.input_tokens
        USO["output"] += response.usage.output_tokens
    if response.stop_reason == "refusal":
        cat = response.stop_details.category if response.stop_details else None
        raise RuntimeError(f"Claude rechazó la petición (categoría: {cat})")
    if response.stop_reason == "max_tokens":
        raise RuntimeError("Respuesta truncada (max_tokens); baja BATCH")
    return next(b.text for b in response.content if b.type == "text")


def preparar_ficha(subs, idioma="español latinoamericano neutro", notas="", client=None):
    """Lee todos los subtítulos y devuelve una ficha (markdown) de personajes, trato y glosario."""
    client = client or cliente()
    texto = "\n".join(f"{s['id']}: {s['texto'].replace(chr(10), ' ')}" for s in subs)
    prompt = FICHA_PROMPT.format(idioma=idioma, texto=texto,
                                 notas=f"\nNotas del usuario sobre la obra: {notas}\n" if notas else "")
    return _llamar(client, messages=[{"role": "user", "content": prompt}],
                   output_config={"effort": EFFORT}).strip()


def _pedir(client, system, antes, despues, bloque):
    partes = []
    if antes:
        partes.append("Contexto: subtítulos anteriores (original; NO los traduzcas):\n"
                      + json.dumps(antes, ensure_ascii=False))
    if despues:
        partes.append("Contexto: subtítulos siguientes (original; NO los traduzcas):\n"
                      + json.dumps(despues, ensure_ascii=False))
    partes.append("Traduce estos subtítulos:\n"
                  + json.dumps([{"id": s["id"], "texto": s["texto"]} for s in bloque], ensure_ascii=False))
    text = _llamar(
        client,
        # El system (reglas + ficha) es igual en todos los bloques: se cachea.
        system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": "\n\n".join(partes)}],
        output_config={"effort": EFFORT, "format": {"type": "json_schema", "schema": SCHEMA}},
    )
    return {d["id"]: d["texto"] for d in json.loads(text)["subtitulos"]}


def _traducir_bloque(client, system, subs, i):
    """Traduce subs[i:i+BATCH] (con 1 reintento de los que falten). Devuelve {id: texto}."""
    bloque = subs[i:i + BATCH]
    ctx = lambda ss: [{"id": s["id"], "texto": s["texto"]} for s in ss]
    antes, despues = ctx(subs[max(0, i - CONTEXT):i]), ctx(subs[i + BATCH:i + BATCH + CONTEXT])
    pendientes, trad = bloque, {}
    for _ in range(2):
        try:
            trad.update(_pedir(client, system, antes, despues, pendientes))
        except (anthropic.AuthenticationError, anthropic.PermissionDeniedError, anthropic.BadRequestError):
            raise                               # key o petición inválida: no tiene sentido seguir
        except (RuntimeError, json.JSONDecodeError, StopIteration, anthropic.APIStatusError) as e:
            print(f"  ! bloque {i // BATCH + 1}: {e}")
        pendientes = [s for s in bloque if s["id"] not in trad]
        if not pendientes:
            break
    if pendientes:
        print(f"  ! {len(pendientes)} subtítulos quedaron sin traducir (ids {[s['id'] for s in pendientes][:10]}...)")
    return trad


def traducir(subs, idioma="español latinoamericano neutro", notas="", client=None, verbose=True, ficha="",
             workers=None):
    """Devuelve una copia de `subs` con el texto traducido. Los tiempos no se tocan."""
    client = client or cliente()
    system = SYSTEM.format(idioma=idioma, chars=LINE_CHARS,
                           notas=f"\nNotas sobre la obra:\n{notas}" if notas else "",
                           ficha=f"\n<ficha_de_la_obra>\n{ficha}\n</ficha_de_la_obra>" if ficha else "")
    out = [dict(s) for s in subs]
    hechos, t0 = 0, time.time()
    workers = workers or WORKERS
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futuros = {pool.submit(_traducir_bloque, client, system, subs, i): i for i in range(0, len(subs), BATCH)}
        if verbose:
            print(f"  {len(subs)} subtítulos → {len(futuros)} bloques de {BATCH}, {workers} en paralelo")
        try:
            for fut in as_completed(futuros):
                i = futuros[fut]
                trad = fut.result()
                for k, s in enumerate(subs[i:i + BATCH]):
                    if s["id"] in trad:
                        out[i + k]["texto"] = _rearmar(trad[s["id"]])
                hechos += 1
                if verbose:
                    print(f"  bloque {i // BATCH + 1:>2} listo (subs {i + 1}-{min(i + BATCH, len(subs))})"
                          f" · {hechos}/{len(futuros)} bloques · {time.time() - t0:.0f}s")
        except BaseException:
            for f in futuros:                   # error de key, Ctrl+C...: no lanzar los bloques pendientes
                f.cancel()
            raise
    return out


def ficha_de(path, subs, idioma, notas, client):
    """Usa nombre.ficha.md si existe (para poder editarla); si no, la prepara y la guarda."""
    fpath = path[:-4] + ".ficha.md"
    if os.path.exists(fpath):
        print("  usando ficha existente:", fpath)
        return open(fpath, encoding="utf-8").read()
    ficha = preparar_ficha(subs, idioma, notas, client)
    with open(fpath, "w", encoding="utf-8") as f:
        f.write(ficha + "\n")
    print("  ficha guardada en", fpath)
    return ficha


def main():
    import argparse
    global MODEL, EFFORT
    ap = argparse.ArgumentParser(description="Traduce .srt con Claude")
    ap.add_argument("srt", nargs="+")
    ap.add_argument("--idioma", default="español latinoamericano neutro")
    ap.add_argument("--sufijo", default="spa", help="archivo de salida: nombre.<sufijo>.srt")
    ap.add_argument("--notas", default="", help="contexto de la obra: título, época, personajes, glosario")
    ap.add_argument("--sin-ficha", action="store_true", help="no preparar la ficha de personajes")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--effort", default=EFFORT)
    a = ap.parse_args()
    MODEL, EFFORT = a.model, a.effort
    client = cliente()
    for path in a.srt:
        out = path[:-4] + f".{a.sufijo}.srt"
        print(path, "→", out)
        subs = leer_srt(path)
        ficha = "" if a.sin_ficha else ficha_de(path, subs, a.idioma, a.notas, client)
        escribir_srt(traducir(subs, a.idioma, a.notas, client, ficha=ficha), out)
    print(f"Tokens ({MODEL}): entrada={USO['input']} salida={USO['output']}")


if __name__ == "__main__":
    main()
