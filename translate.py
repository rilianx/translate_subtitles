"""Traduce un .srt con Claude manteniendo numeración y tiempos intactos.

Envía los subtítulos en bloques (con algunas líneas previas como contexto) y pide la
respuesta en JSON con un esquema fijo, así cada traducción vuelve con su id.
Requiere la variable de entorno ANTHROPIC_API_KEY.
"""
import json
import os
import re

import anthropic

MODEL = "claude-opus-5-5"
EFFORT = "medium"     # low/medium/high: más alto = más cuidadoso, más lento y caro
BATCH = 60            # subtítulos por llamada
CONTEXT = 10          # subtítulos previos (ya traducidos) que se envían como contexto
LINE_CHARS = 42

BLOCK = re.compile(r"(\d+)\s*\n(\d+:\d+:\d+[,.]\d+\s*-->\s*\d+:\d+:\d+[,.]\d+)[^\n]*\n(.*?)(?=\n\s*\n|\Z)", re.S)

SYSTEM = """Eres un traductor profesional de subtítulos de cine y series.
Traduce cada subtítulo al {idioma}.

Reglas:
- Traduce el sentido y el registro (coloquial, formal, insultos, humor), no palabra por palabra.
- Un subtítulo traducido por cada id recibido, con el mismo id. No unas ni partas subtítulos,
  aunque una frase continúe en el siguiente: tradúcela de forma que se lea bien repartida.
- Sé conciso: el texto debe poder leerse en el tiempo que dura el subtítulo. Máximo 2 líneas
  de unos {chars} caracteres; usa \\n para separar líneas.
- Mantén las etiquetas de formato (<i>, </i>, <b>...) y los guiones de diálogo al inicio de línea.
- Los nombres propios no se traducen.
- Si un subtítulo es solo un sonido o una interjección, adáptalo ("Oh" → "Oh", "Huh?" → "¿Eh?").
- El texto viene de un reconocimiento de voz automático y puede tener errores: si una palabra
  no tiene sentido, usa el contexto para inferir lo que se dijo.
{notas}"""

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
    return anthropic.Anthropic(default_headers={"anthropic-workspace-id": ws} if ws else None)


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


def _pedir(client, system, contexto, bloque):
    partes = []
    if contexto:
        partes.append("Contexto (subtítulos anteriores, ya traducidos; NO los traduzcas de nuevo):\n"
                      + json.dumps(contexto, ensure_ascii=False))
    partes.append("Traduce estos subtítulos:\n"
                  + json.dumps([{"id": s["id"], "texto": s["texto"]} for s in bloque], ensure_ascii=False))
    response = client.beta.messages.create(
        model=MODEL,
        max_tokens=16000,
        system=system,
        messages=[{"role": "user", "content": "\n\n".join(partes)}],
        output_config={"effort": EFFORT, "format": {"type": "json_schema", "schema": SCHEMA}},
        # Si el modelo rechaza la petición, la API la reintenta con el modelo de respaldo recomendado.
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    )
    if response.stop_reason == "refusal":
        cat = response.stop_details.category if response.stop_details else None
        raise RuntimeError(f"Claude rechazó el bloque (categoría: {cat})")
    if response.stop_reason == "max_tokens":
        raise RuntimeError("Respuesta truncada (max_tokens); baja BATCH")
    text = next(b.text for b in response.content if b.type == "text")
    return {d["id"]: d["texto"] for d in json.loads(text)["subtitulos"]}


def traducir(subs, idioma="español latinoamericano neutro", notas="", client=None, verbose=True):
    """Devuelve una copia de `subs` con el texto traducido. Los tiempos no se tocan."""
    client = client or cliente()
    system = SYSTEM.format(idioma=idioma, chars=LINE_CHARS,
                           notas=f"\nNotas sobre la obra:\n{notas}" if notas else "")
    out = [dict(s) for s in subs]
    for i in range(0, len(subs), BATCH):
        bloque = subs[i:i + BATCH]
        contexto = [{"id": s["id"], "original": subs[j]["texto"], "traduccion": s["texto"]}
                    for j, s in enumerate(out[max(0, i - CONTEXT):i], start=max(0, i - CONTEXT))]
        pendientes, trad = bloque, {}
        for _ in range(2):                      # 1 reintento solo con los que faltaron
            try:
                trad.update(_pedir(client, system, contexto, pendientes))
            except (anthropic.AuthenticationError, anthropic.PermissionDeniedError, anthropic.BadRequestError):
                raise                           # key o petición inválida: no tiene sentido seguir
            except (RuntimeError, json.JSONDecodeError, anthropic.APIStatusError) as e:
                print(f"  ! bloque {i // BATCH + 1}: {e}")
            pendientes = [s for s in bloque if s["id"] not in trad]
            if not pendientes:
                break
        for k, s in enumerate(bloque):
            if s["id"] in trad:
                out[i + k]["texto"] = _rearmar(trad[s["id"]])
        if pendientes:
            print(f"  ! {len(pendientes)} subtítulos quedaron sin traducir (ids {[s['id'] for s in pendientes][:10]}...)")
        if verbose:
            print(f"  {min(i + BATCH, len(subs))}/{len(subs)}")
    return out


def main():
    import argparse
    global MODEL, EFFORT
    ap = argparse.ArgumentParser(description="Traduce .srt con Claude")
    ap.add_argument("srt", nargs="+")
    ap.add_argument("--idioma", default="español latinoamericano neutro")
    ap.add_argument("--sufijo", default="spa", help="archivo de salida: nombre.<sufijo>.srt")
    ap.add_argument("--notas", default="", help="contexto de la obra: título, época, personajes, glosario")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--effort", default=EFFORT)
    a = ap.parse_args()
    MODEL, EFFORT = a.model, a.effort
    client = cliente()
    for path in a.srt:
        out = path[:-4] + f".{a.sufijo}.srt"
        print(path, "→", out)
        escribir_srt(traducir(leer_srt(path), a.idioma, a.notas, client), out)


if __name__ == "__main__":
    main()
