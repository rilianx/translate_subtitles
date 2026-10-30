"""Genera mp32srt.ipynb a partir de los módulos .py (fuente única del código)."""
import json

def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src.strip("\n").splitlines(True)}

def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.strip("\n").splitlines(True)}

def module(name):
    return code(f"%%writefile {name}\n" + open(name, encoding="utf-8").read())

cells = [
    md("""
# mp3 → srt → traducción
1. **Transcribe** cada `.mp3` sin `.srt` con faster-whisper, usando tiempos por palabra.
2. **Traduce** cada `.srt` sin su versión `.spa.srt` con Claude, sin tocar los tiempos.

Requisitos: entorno con GPU (*Entorno de ejecución → Cambiar tipo → T4 GPU*) y, para traducir,
el secreto `ANTHROPIC_API_KEY` en el panel 🔑 con acceso habilitado para este notebook.
"""),
    code("!pip install -q faster-whisper anthropic"),
    code("""
import os
from google.colab import drive
drive.mount('/content/drive')

folder_path = '/content/drive/MyDrive/Multimedia/Translations'
TRANSCRIBIR = True
TRADUCIR = True
WHISPER_MODEL = "medium"           # validado; en GPU puedes probar "large-v3" o "large-v3-turbo"
IDIOMA_AUDIO = "en"
IDIOMA_DESTINO = "español latinoamericano neutro"
SUFIJO = "spa"                     # salida: nombre.spa.srt
NOTAS = ""                         # opcional: título, época, personajes, glosario...
USAR_FICHA = True                  # lee todo el srt antes y arma ficha de personajes y trato
"""),
    md("## Código (se escribe a archivos .py y se importa)"),
    module("transcribe.py"),
    module("translate.py"),
    md("## 1. Transcribir"),
    code("""
import importlib, transcribe
importlib.reload(transcribe)

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
import importlib, translate
from google.colab import userdata
importlib.reload(translate)
os.environ["ANTHROPIC_API_KEY"] = userdata.get("ANTHROPIC_API_KEY")
try:   # solo si tu key no está asociada a un workspace
    os.environ["ANTHROPIC_WORKSPACE_ID"] = userdata.get("ANTHROPIC_WORKSPACE_ID")
except Exception:
    pass

archivos = os.listdir(folder_path)
originales =[f for f in archivos if f.lower().endswith('.srt') and not f[:-4].endswith(f'.{SUFIJO}')]
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
    module("evalsrt.py"),
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
