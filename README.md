# Segmentación de instancias celulares y clasificación de texto

**Investigación empírica reproducible** · Tarea 3 (Sección 4.8) · Deep Learning · Maestría en Ingeniería Biomédica,
Universidad del Norte · Prof. Dr. Lihki Rubio · Autor: **Andrés Duarte Chaves**

| | |
|---|---|
| 📘 Jupyter Book (artículo científico) | https://aduartech81.github.io/deep-learning-03-cellseg-nlp/ |
| 📊 Tablero interactivo | https://deep-learning-03-cellseg-nlp.onrender.com/ |
| ⚙️ API de inferencia | https://deep-learning-03-cellseg-nlp-api.onrender.com/docs |
| ▶️ Ejecutar en Colab | [`colab/ejecutar_en_colab.ipynb`](colab/ejecutar_en_colab.ipynb) |

## Qué es

Dos problemas con el mismo estándar metodológico (pre-registro, auditoría de datos, presupuesto de búsqueda
equitativo, varias semillas, inferencia estadística, ablaciones, robustez, calibración, equidad y MLOps):

1. **Segmentación de instancias celulares** (Sartorius, contraste de fase; SH-SY5Y, astrocitos, neuronas
   corticales): baseline clásico, Mask R-CNN, modelo estilo Hover-Net, **U-Net++ propia en Keras**, Cellpose y SAM.
2. **Clasificación de hojas de vida** en 24 categorías (Jarvis Calling Hiring Contest): baselines, TF-IDF + XGBoost,
   fastText, CNN-1D, **Word2Vec + BiLSTM propio**, DistilBERT, con análisis de equidad por ser un contexto de
   reclutamiento.

Resultados principales: ver el resumen del libro y la página de conclusiones del tablero.

## Estructura

```
configs/project.yaml        perfiles full (datos reales) y smoke (sintético con el mismo esquema)
src/dl03/                   paquete instalable: métricas, estadística, calibración, deriva, datos, modelos, orquestación
nb_src/*.py                 capítulos (jupytext) -> notebooks/*.ipynb ejecutados
app/                        API FastAPI (/health /ready /predict_text /predict_image /model_info), registro de modelos
tests/                      pytest: utilidades, datos (pandera), fuga, determinismo, API, regresión del modelo
reports/                    model cards y datasheets (generados desde los resultados)
dashboard/                  tablero Dash (solo lee resultados)
colab/                      cuaderno que corre todo en Colab con respaldo en Drive
Dockerfile, Makefile, render.yaml, .github/workflows/{ci,book}.yml
```

## Reproducir

```bash
pip install -r requirements.txt          # en Colab (GPU); ver colab/ejecutar_en_colab.ipynb
export KAGGLE_API_TOKEN=...              # aceptar antes las reglas de ambas competencias en Kaggle
make reproduce                           # = DL03_PROFILE=full scripts/run_chapters.sh
make smoke                               # prueba de punta a punta con datos sintéticos (CPU)
make test lint                           # pruebas, ruff y mypy
make build serve                         # imagen Docker de la API (multi-etapa, usuario no root, HEALTHCHECK)
```

El protocolo se congeló con la etiqueta git `prereg` antes de entrenar los modelos finales. Las versiones exactas
del entorno de Colab quedan en `requirements.lock`; los datos se identifican por su SHA-256 (`MANIFEST.json`) y con
DVC.

## Datos y licencias

* Sartorius – Cell Instance Segmentation (Kaggle; uso según las reglas de la competencia). Las imágenes **no** se
  redistribuyen en este repositorio.
* Jarvis Calling Hiring Contest (Kaggle; CC0). El conjunto dorado de CI incluye 100 hojas de vida del test.

## Uso de asistentes de IA

El código, el diseño experimental y la redacción se desarrollaron con asistencia de un modelo de lenguaje (Claude,
Anthropic). Toda cifra proviene de la ejecución de los notebooks; las funciones metodológicas tienen pruebas
unitarias; el autor revisó y responde por la totalidad del contenido.
