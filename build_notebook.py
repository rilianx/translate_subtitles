"""Genera mp32srt.ipynb. El notebook descarga el código (.py) desde GitHub al ejecutarse."""
import json

def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src.strip("\n").splitlines(True)}

def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.strip("\n").splitlines(True)}

cells = [
    md("""
# mp3 → srt → traducción
1. **Transcribe** cada `.mp3` sin `.srt` con faster-whisper, usando tiempos por palabra (el idioma del audio se detecta solo).
2. **Traduce** cada `.srt` sin su versión `.spa.srt` con Claude, sin tocar los tiempos.

El código se descarga de GitHub (`rilianx/translate_subtitles`) en cada ejecución, así siempre está al día.

Requisitos: entorno con GPU (*Entorno de ejecución → Cambiar tipo → T4 GPU*) y estos secretos
en el panel 🔑 (con acceso habilitado para este notebook):
- `ANTHROPIC_API_KEY` (y `ANTHROPIC_WORKSPACE_ID` si tu key no pertenece a un workspace).
"""),
    code("""
!pip install -q faster-whisper anthropic
import ctypes, glob, os, subprocess, sys
from google.colab import userdata

# faster-whisper usa CUDA 12; si Colab ya no trae libcublas.so.12, se instala por pip y se precarga.
try:
    ctypes.CDLL("libcublas.so.12")
except OSError:
    !pip install -q nvidia-cublas-cu12 "nvidia-cudnn-cu12==9.*"
    import nvidia
    libs = [f for base in nvidia.__path__
            for pat in ("cublas/lib/libcublasLt.so.12", "cublas/lib/libcublas.so.12", "cudnn/lib/libcudnn*.so.9")
            for f in sorted(glob.glob(os.path.join(base, pat)))]
    for _ in range(3):                    # cuDNN tiene sub-librerías que dependen entre sí
        for f in libs:
            try:
                ctypes.CDLL(f, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass
    ctypes.CDLL("libcublas.so.12")        # si esto falla, el error es claro aquí y no a mitad de película
    print("Librerías CUDA 12 instaladas y cargadas")

# Descarga la última versión del código desde GitHub.
# Si el repo fuera privado, crea el secreto GITHUB_TOKEN (token con lectura del repo).
REPO = "rilianx/translate_subtitles"
try:
    auth = userdata.get("GITHUB_TOKEN") + "@"
except Exception:
    auth = ""
subprocess.run(["rm", "-rf", "/content/translate_subtitles"])
r = subprocess.run(["git", "clone", "-q", "--depth", "1", f"https://{auth}github.com/{REPO}",
                    "/content/translate_subtitles"], capture_output=True, text=True)
if r.returncode:
    raise RuntimeError("No se pudo clonar el repo (¿es privado? crea el secreto GITHUB_TOKEN)")
sys.path.insert(0, "/content/translate_subtitles")
print(subprocess.run(["git", "-C", "/content/translate_subtitles", "log", "-1", "--format=Código: %h %s (%cr)"],
                     capture_output=True, text=True).stdout)
"""),
    code("""
import os
from google.colab import drive
drive.mount('/content/drive')

folder_path = '/content/drive/MyDrive/Multimedia/Translations'
TRANSCRIBIR = True
TRADUCIR = True
WHISPER_MODEL = "medium"           # validado; en GPU puedes probar "large-v3" o "large-v3-turbo"
IDIOMA_AUDIO = "auto"              # "auto" lo detecta; o fíjalo: "en", "fr", "it", "ja"...
IDIOMA_DESTINO = "español latinoamericano neutro"
SUFIJO = "spa"                     # salida: nombre.spa.srt
NOTAS = ""                         # opcional: título, época, personajes, glosario...
USAR_FICHA = True                  # lee todo el srt antes y arma ficha de personajes y trato
"""),
    md("## 1. Transcribir"),
    code("""
import transcribe

archivos = os.listdir(folder_path)
mp3s = sorted(os.path.splitext(f)[0] for f in archivos if f.lower().endswith('.mp3'))
srts = {f[:-4] for f in archivos if f.lower().endswith('.srt')}
pendientes = [m for m in mp3s if m not in srts]
print(f"{len(pendientes)} mp3 sin subtítulos:", *pendientes, sep="\\n- ")

if TRANSCRIBIR and pendientes:
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = transcribe.WhisperModel(WHISPER_MODEL, device=device,
                                    compute_type="float16" if device == "cuda" else "int8")
    for nombre in pendientes:
        print("Transcribiendo:", nombre)
        words = transcribe.transcribir(model, os.path.join(folder_path, nombre + ".mp3"), language=IDIOMA_AUDIO)
        transcribe.escribir_srt(transcribe.armar_cues(words), os.path.join(folder_path, nombre + ".srt"))
"""),
    md("## 2. Traducir con Claude"),
    code("""
import translate
os.environ["ANTHROPIC_API_KEY"] = userdata.get("ANTHROPIC_API_KEY")
try:   # solo si tu key no está asociada a un workspace
    os.environ["ANTHROPIC_WORKSPACE_ID"] = userdata.get("ANTHROPIC_WORKSPACE_ID")
except Exception:
    pass

archivos = os.listdir(folder_path)
originales = [f for f in archivos if f.lower().endswith('.srt') and not f[:-4].endswith(f'.{SUFIJO}')]
pendientes_tr = sorted(f for f in originales if f[:-4] + f'.{SUFIJO}.srt' not in archivos)
print(f"{len(pendientes_tr)} srt sin traducir:", *pendientes_tr, sep="\\n- ")

if TRADUCIR:
    for f in pendientes_tr:
        print("Traduciendo:", f)
        path = os.path.join(folder_path, f)
        subs = translate.leer_srt(path)
        # Ficha de personajes/trato/glosario: se guarda como nombre.ficha.md; si la editas, se reutiliza.
        ficha = translate.ficha_de(path, subs, IDIOMA_DESTINO, NOTAS, translate.cliente()) if USAR_FICHA else ""
        subs = translate.traducir(subs, IDIOMA_DESTINO, NOTAS, ficha=ficha)
        translate.escribir_srt(subs, os.path.join(folder_path, f[:-4] + f'.{SUFIJO}.srt'))
"""),
    md("""
## (Opcional) Validar tiempos contra un SRT de referencia
Compara solo tiempos; la referencia puede estar en otro idioma. F1 = cuánto coincide el tiempo en pantalla.
"""),
    code("""
import evalsrt
ref_srt = '/content/drive/MyDrive/<ruta>/example.srt'
hyp_srt = '/content/drive/MyDrive/<ruta>/example_gen.srt'
TMAX = None   # p.ej. 1200 para evaluar solo los primeros 20 min
tmax = TMAX or max(e for _, e in evalsrt.load(ref_srt))
print(evalsrt.fmt('generado', evalsrt.evaluate(evalsrt.load(ref_srt, tmax), evalsrt.load(hyp_srt, tmax), tmax)))
"""),
]

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
      "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
json.dump(nb, open("mp32srt.ipynb", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("mp32srt.ipynb generado")
