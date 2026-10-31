"""Transcripción a SRT con tiempos ajustados por palabra.

faster-whisper con timestamps por palabra; los subtítulos se arman a partir de las
palabras (no de los segmentos de Whisper, que vienen redondeados y absorben silencios/música).
"""
import re
from collections import Counter
import subprocess
import numpy as np
from faster_whisper import WhisperModel

# Parámetros de armado de subtítulos
MAX_CHARS = 84        # máx. caracteres por subtítulo (2 líneas de 42)
LINE_CHARS = 42
MAX_DUR = 6.0         # máx. duración de un subtítulo (s)
MIN_DUR = 1.0         # mín. duración en pantalla (s)
GAP_SPLIT = 0.6       # silencio entre palabras que fuerza corte (s)
PAD_END = 0.8         # cola extra al final de cada subtítulo (s)
MAX_WORD = 1.0        # una palabra no "dura" más que esto (corrige palabras estiradas sobre música)


def cargar_audio(path, sr=16000):
    """Decodifica con ffmpeg a 16 kHz mono float32."""
    out = subprocess.run(["ffmpeg", "-nostdin", "-i", path, "-f", "s16le", "-ac", "1", "-ar", str(sr), "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(out, np.int16).astype(np.float32) / 32768.0


def detectar_idiomas(model, audio, ventana=10, sr=16000):
    """Detecta el idioma cada `ventana` segundos y devuelve (idiomas, tramos).

    idiomas: los que aparecen de verdad (≥5 % de las ventanas con confianza alta), del más al menos
    frecuente. tramos: [(inicio, fin, idioma)] contiguos. Las ventanas dudosas (música, ruido) se
    asignan al tramo vecino, y solo se eligen idiomas de la lista, así un "jw" sobre música no
    se transcribe como javanés.
    """
    probs = []
    for t in range(0, len(audio), ventana * sr):
        _, _, todas = model.detect_language(audio=audio[t:t + ventana * sr])
        probs.append(dict(todas))
    seguros = Counter(max(p, key=p.get) for p in probs if max(p.values()) > 0.7)
    total = sum(seguros.values())
    idiomas = [l for l, c in seguros.most_common() if c / max(total, 1) >= 0.05]
    if not idiomas:
        idiomas = [Counter(max(p, key=p.get) for p in probs).most_common(1)[0][0]]
    etiquetas = []
    for p in probs:
        mejor = max(idiomas, key=lambda l: p.get(l, 0))
        etiquetas.append(mejor if sum(p.get(l, 0) for l in idiomas) >= 0.5 else None)
    # Ventanas dudosas: heredan el idioma de la anterior (o de la siguiente, al inicio).
    for i in range(len(etiquetas)):
        if etiquetas[i] is None and i > 0:
            etiquetas[i] = etiquetas[i - 1]
    for i in reversed(range(len(etiquetas))):
        if etiquetas[i] is None:
            etiquetas[i] = etiquetas[i + 1] if i + 1 < len(etiquetas) and etiquetas[i + 1] else idiomas[0]
    tramos, dur = [], len(audio) / sr
    for i, l in enumerate(etiquetas):
        t0, t1 = i * ventana, min((i + 1) * ventana, dur)
        if tramos and tramos[-1][2] == l:
            tramos[-1] = (tramos[-1][0], t1, l)
        else:
            tramos.append((t0, t1, l))
    return idiomas, tramos


def transcribir(model, audio, language=None, **kw):
    """language: código ISO ("en", "fr", "ja"...) o None/"auto" para detectarlo.

    En "auto", si la película mezcla idiomas (p. ej. narración en inglés y entrevistas en chino),
    cada tramo se transcribe con su idioma; forzar uno solo hace que Whisper mezcle o invente texto.
    """
    if isinstance(audio, str):
        audio = cargar_audio(audio)
    opciones = dict(
        word_timestamps=True,
        # Sin VAD: Silero se pierde diálogos mezclados con música/ruido en películas.
        # Este filtro descarta alucinaciones rodeadas de silencio (necesita word_timestamps).
        hallucination_silence_threshold=2.0,
        condition_on_previous_text=False,   # evita alucinaciones/bucles en películas largas
        beam_size=5, **kw)
    if language in (None, "", "auto"):
        idiomas, tramos = detectar_idiomas(model, audio)
        dur = {l: sum(b - a for a, b, x in tramos if x == l) for l in idiomas}
        total = sum(dur.values()) or 1
        print("  idiomas: " + ", ".join(f"{l} {dur[l] / total:.0%}" for l in idiomas)
              + (f" ({len(tramos)} tramos)" if len(idiomas) > 1 else ""))
        partes = [(l, [t for a, b, x in tramos if x == l for t in (a, b)]) for l in idiomas]
    else:
        partes = [(language, "0")]
    words = []
    for lang, clip in partes:
        segments, _ = model.transcribe(audio, language=lang, clip_timestamps=clip, **opciones)
        for seg in segments:
            for w in seg.words or []:
                if w.word.strip():
                    words.append([w.start, w.end, w.word, lang])
    words.sort(key=lambda w: w[0])
    return words


def _limpiar(words):
    """Recorta palabras anormalmente largas (típico: primera palabra tras música)."""
    for w in words:
        if w[1] - w[0] > MAX_WORD:
            if w[2].strip()[-1:] in ".?!,":
                w[1] = w[0] + MAX_WORD     # palabra final de frase: recortar el final
            else:
                w[0] = w[1] - MAX_WORD     # si no: el inicio está adelantado
    return words


def armar_cues(words):
    words = _limpiar(words)
    cues, cur = [], []

    def flush():
        if cur:
            cues.append([cur[0][0], cur[-1][1], "".join(w[2] for w in cur).strip()])
            cur.clear()

    for i, w in enumerate(words):
        if cur:
            text = "".join(x[2] for x in cur).strip()
            gap = w[0] - cur[-1][1]
            prev_end_punct = cur[-1][2].strip()[-1:] in ".?!"
            too_long = len(text) + len(w[2]) > MAX_CHARS or w[1] - cur[0][0] > MAX_DUR
            otro_idioma = len(w) > 3 and w[3] != cur[-1][3]
            if gap > GAP_SPLIT or too_long or otro_idioma or (prev_end_punct and (gap > 0.3 or len(text) > 30)):
                flush()
        cur.append(w)
    flush()

    # Ajustes de duración sin pisar el siguiente subtítulo
    for i, c in enumerate(cues):
        nxt = cues[i + 1][0] if i + 1 < len(cues) else c[1] + 10
        c[1] = min(max(c[1] + PAD_END, c[0] + MIN_DUR), nxt - 0.05)
    return cues


def _partir_lineas(text):
    if len(text) <= LINE_CHARS:
        return text
    mid = len(text) // 2
    cands = [m.start() for m in re.finditer(" ", text)]
    cut = min(cands, key=lambda p: abs(p - mid)) if cands else mid
    return text[:cut].strip() + "\n" + text[cut:].strip()


def _ts(t):
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3600000); m, ms = divmod(ms, 60000); s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def escribir_srt(cues, path):
    with open(path, "w", encoding="utf-8") as f:
        for i, (s, e, t) in enumerate(cues, 1):
            f.write(f"{i}\n{_ts(s)} --> {_ts(e)}\n{_partir_lineas(t)}\n\n")


def main():
    import argparse, time
    ap = argparse.ArgumentParser(description="mp3/mp4 → srt con tiempos por palabra")
    ap.add_argument("audio", nargs="+")
    ap.add_argument("--model", default="medium", help="tiny/base/small/medium/large-v3/large-v3-turbo")
    ap.add_argument("--language", default="auto", help="código ISO del audio (en, fr, ja...) o auto")
    ap.add_argument("--device", default="auto", help="cuda/cpu/auto")
    a = ap.parse_args()
    device = a.device
    if device == "auto":
        import ctranslate2
        device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
    model = WhisperModel(a.model, device=device, compute_type="float16" if device == "cuda" else "int8")
    for audio in a.audio:
        out = audio.rsplit(".", 1)[0] + ".srt"
        t = time.time()
        escribir_srt(armar_cues(transcribir(model, audio, language=a.language)), out)
        print(f"{out} ({time.time() - t:.0f}s)")


if __name__ == "__main__":
    main()
