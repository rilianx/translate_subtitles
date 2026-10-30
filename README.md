# translate_subtitles

Subtítulos automáticos para películas y series:

1. **`transcribe.py`**: audio/video → `.srt` con [faster-whisper](https://github.com/SYSTRAN/faster-whisper) y tiempos **por palabra**.
2. **`translate.py`**: `.srt` → `.spa.srt` (u otro idioma) con Claude, **sin tocar numeración ni tiempos**.
3. **`evalsrt.py`**: compara los tiempos de un `.srt` generado contra uno de referencia, aunque estén en idiomas distintos.

## Uso en Colab (recomendado)

Abre `mp32srt.ipynb` en Colab (*Archivo → Abrir notebook → GitHub*) con un entorno de GPU T4.
- Para traducir, crea el secreto `ANTHROPIC_API_KEY` en el panel 🔑 y habilita el acceso para el notebook. Si tu key no pertenece a un workspace, agrega también `ANTHROPIC_WORKSPACE_ID`.
- En la celda de configuración, ajusta `folder_path`.

El notebook procesa todos los `.mp3` de la carpeta que no tengan `.srt`, y luego traduce todos los `.srt` que no tengan `.spa.srt`.

> Edita los `.py`, no el notebook: el notebook se regenera con `python build_notebook.py`.

## Uso local

```bash
pip install -r requirements.txt
python transcribe.py pelicula.mp3 --model medium          # → pelicula.srt
export ANTHROPIC_API_KEY=...
python translate.py pelicula.srt --notas "Cumbres borrascosas, Inglaterra s. XVIII"   # → pelicula.spa.srt
python evalsrt.py referencia.srt pelicula.srt --max 1200
```

## Por qué los tiempos ahora calzan

Con `openai-whisper`, cada subtítulo es un segmento de Whisper: sus tiempos vienen redondeados y se estiran sobre la música y los silencios (por ejemplo, un subtítulo de 0:00 a 0:28). Ahora:

- Se piden **timestamps por palabra** y los subtítulos se arman a partir de las palabras:
  - se corta en pausas de más de 0,6 s y en fin de frase;
  - máximo 2 líneas de 42 caracteres y entre 1 y 6 s por subtítulo, con 0,8 s extra al final para leer.
- Con `condition_on_previous_text=False` y `hallucination_silence_threshold=2` se evitan los bucles y las frases inventadas sobre música.
- **No se usa VAD (Silero)**: en las pruebas con película se perdía casi todo el diálogo mezclado con música.

Resultado en los primeros 20 minutos de una película, contra subtítulos profesionales, ambos con el modelo `medium`:

| | Segmentos de Whisper | Por palabra (este repo) |
|---|---|---|
| Subtítulos que empiezan a ≤0,5 s de la referencia | 54 % | **79 %** |
| Error mediano al inicio | 0,46 s | **0,19 s** |
| Error mediano al final | 0,53 s | **0,37 s** |
| F1 del tiempo en pantalla | 0,78 | **0,84** |

## Traducción

- Se envían bloques de 60 subtítulos, junto con los 10 anteriores ya traducidos como contexto, para mantener consistentes los nombres y el registro.
- La respuesta vuelve en JSON con un esquema fijo (id → texto); si falta algún id, se reintenta solo con esos.
- `--notas` (o `NOTAS` en el notebook) permite dar contexto de la obra: época, personajes, glosario o tuteo/voseo.
- Modelo por defecto: `claude-opus-5-5` con `effort="medium"`.
  - Si el modelo rechaza un bloque, la API lo reintenta con el modelo de respaldo recomendado (`fallbacks="default"`).
  - Si aun así queda sin traducir, ese bloque se deja en el idioma original y se avisa en la salida.
