# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 0. Protocolo experimental pre-registrado
#
# Este capítulo se escribió y se congeló **antes de entrenar cualquier modelo final**, con un commit etiquetado
# `prereg` en el repositorio. Toda desviación posterior se declara en la sección 0.6 y en la discusión.
# La celda final imprime el commit de la etiqueta para que quede trazado en la salida ejecutada.

# %%
import subprocess

from dl03.config import environment_report
from dl03.viz import setup

cfg = setup()
env = environment_report()
{k: v for k, v in env.items() if v}

# %% [markdown]
# ## 0.1 Hipótesis refinadas con criterio de refutación numérico
#
# | ID | Hipótesis | Prueba | Se refuta si… |
# |---|---|---|---|
# | H1 | Un modelo preentrenado en microscopía (Cellpose `cyto2` ajustado) supera a U-Net++ entrenada desde cero cuando se usa el 25 % del train, y la brecha se reduce con el 100 %. | Diferencia de AP@[.5:.95] por imagen de test, IC bootstrap 95 % (10 000 remuestreos) a 25 % y 100 %. | el IC 95 % de la diferencia al 25 % incluye 0, **o** la brecha al 100 % no es menor que al 25 %. |
# | H2 | En píxeles, AUC-ROC y balanced accuracy saturan (> 0.95) sin discriminar entre modelos, mientras que AP@[.5:.95] sí discrimina. | Correlación de Spearman entre el ranking por AUC de píxel y el ranking por AP (modelos con mapa de probabilidad). | ρ > 0.9, **o** la AUC media de algún modelo ≤ 0.95. |
# | H3 | El ranking global de modelos cambia al estratificar por tipo celular (paradoja de Simpson). | Orden de modelos por AP medio en shsy5y, astro y cort. | el orden es idéntico en los tres tipos. |
# | H4 | (H0 a contrastar) La diferencia de F1 macro entre el mejor transformer y TF-IDF + XGBoost **no** es significativa tras Holm y varianza entre semillas. | McNemar exacto en test + t de Nadeau–Bengio sobre los 5 folds de la validación anidada (XGBoost) y semillas (transformer), Holm. | p_Holm < 0.05 **y** |ΔF1| ≥ δ_NLP. |
# | H5 | El fine-tuning deja al transformer peor calibrado (ECE) que la regresión logística TF-IDF, y el temperature scaling lo recupera sin cambiar la exactitud. | ECE (15 bins) en test con IC bootstrap; T ajustada solo en validación. | ECE_transformer ≤ ECE_LR, **o** el IC de la reducción de ECE tras escalar incluye 0, **o** cambia algún argmax. |
#
# **Resultados negativos.** Refutar una hipótesis con evidencia sólida se reporta igual que confirmarla.

# %% [markdown]
# ## 0.2 Tabla de pre-registro
#
# | Elemento | Declaración |
# |---|---|
# | Métrica primaria | Imágenes: AP@[0.5:0.95] estilo Sartorius (media sobre umbrales de IoU de TP/(TP+FP+FN), por imagen). NLP: F1 macro. |
# | Métricas secundarias (cerradas) | Imágenes: AP50, AP75, PQ (SQ, RQ), AJI, precisión/recall/F1 por instancia, MAE/RMSE de conteo, Dice, IoU, precisión y recall de píxel, HD95, F1 de borde, AUC-ROC, AUC-PR y balanced accuracy de píxel. NLP: exactitud, precisión/recall/F1 (macro, micro, ponderado), MCC, balanced accuracy, ROC-AUC y PR-AUC (OvR macro), log-loss, Brier, ECE. |
# | Unidad de análisis | Imagen (segmentación); documento en test y fold externo (NLP). Nunca píxeles. |
# | Particiones | 70/15/15 agrupado: imágenes por placa/fecha/muestra (la agrupación más gruesa con ≥ 7 grupos por tipo celular), estratificado por tipo celular; textos agrupando (casi-)duplicados (MinHash, Jaccard ≥ 0.8) y estratificando por categoría. Test sellado, evaluado una sola vez por modelo final. |
# | Validación | 5-fold agrupado sobre train+val para estimar varianza (U-Net++ y baseline clásico en segmentación; los seis modelos ligeros de NLP vía validación anidada). Modelos pesados: HPO en un único split de validación + semillas. |
# | Pruebas estadísticas | α = 0.05, corrección de Holm. Segmentación: Wilcoxon pareado por imagen, Friedman + post hoc, diagrama de diferencia crítica, d de Cohen pareada y δ de Cliff, modelo mixto con intercepto aleatorio por grupo de placa. NLP: McNemar exacto, t corregida de Nadeau–Bengio, Friedman + Holm sobre folds, IC bootstrap. |
# | Efecto mínimo relevante | δ_seg = 0.02 de AP (≈ 6 % relativo a un AP típico de 0.30 en esta tarea; diferencias menores no cambian el conteo celular de forma apreciable). δ_NLP = 0.02 de F1 macro (≈ 1 documento de cada 50 bien clasificado). |
# | Semillas | ≥ 5 en modelos ligeros (NLP ligeros, transformer); 3 en modelos pesados de segmentación. |
# | Presupuesto de HPO | Optuna TPE + MedianPruner, mismo número de trials por familia comparable: 20 en NLP ligero; 12 en segmentación (para los modelos preentrenados, el HPO se aplica a parámetros de inferencia/post-proceso; ver 0.6). |
# | Parada | Early stopping sobre pérdida de validación con paciencia declarada en `configs/project.yaml`. |
# | Reproducibilidad | `configs/project.yaml`, semillas fijas, versiones registradas, MANIFEST con SHA-256 de los datos, MLflow local (si está disponible) y un comando único (`make reproduce`). |

# %% [markdown]
# ## 0.3 Reglas de reporte
# * Toda cifra como media [IC 95 %]; ningún valor único sin incertidumbre.
# * Figuras cualitativas: muestra aleatoria con semilla fija + 5 mejores y 5 peores casos por métrica.
# * Cada métrica con su definición e implementación (`src/dl03/seg_metrics.py`, `src/dl03/nlp_experiments.py`).

# %% [markdown]
# ## 0.4 Modelos y su grado de implementación
#
# | Dominio | Modelo | Implementación |
# |---|---|---|
# | Imágenes | Otsu/adaptativo + watershed | propia (scikit-image) |
# | Imágenes | Mask R-CNN | torchvision 0.2x, **ResNet50-FPN v2 COCO** (no Swin; ver 0.6), anchors del EDA |
# | Imágenes | Estilo Hover-Net (NP + HV) | **reimplementación propia en Keras** con codificador ResNet50 |
# | Imágenes | U-Net++ + watershed | **desde cero en Keras**: arquitectura, pérdidas Dice/Tversky/focal, generador, post-proceso |
# | Imágenes | Cellpose | `cellpose<4`, generalista `cyto2` y ajustado |
# | Imágenes | SAM ViT-B | `segment-anything`: (a) cero-shot, (b) cajas de Mask R-CNN, (c) decodificador ajustado |
# | Texto | Mayoritaria, TF-IDF + LR, NB-SVM | scikit-learn (NB-SVM propio) |
# | Texto | DistilBERT | Hugging Face, bucle propio con warmup, decaimiento lineal, LLRD y precisión mixta |
# | Texto | Word2Vec + BiLSTM | **propio en Keras** (gensim para Word2Vec en train), pooling último/máx./atención |
# | Texto | CNN-1D (Kim, 2014) | propio en Keras |
# | Texto | TF-IDF + XGBoost | xgboost + Optuna |
# | Texto | fastText supervisado | `fasttext` |

# %% [markdown]
# ## 0.5 Verificación de utilidades (pruebas sintéticas de 4.8.24)
# Se ejecuta la batería de pruebas unitarias del paquete; debe pasar completa antes de modelar.

# %%
r = subprocess.run(["python", "-m", "pytest", "-q", "../tests/test_utils.py"], capture_output=True, text=True)
print(r.stdout[-1500:])
assert r.returncode == 0, "las utilidades no pasan sus pruebas"

# %% [markdown]
# ## 0.6 Desviaciones declaradas respecto del enunciado (decididas antes de entrenar)
#
# 1. **Mask R-CNN sin Swin.** Se usa ResNet50-FPN v2 preentrenado en COCO (torchvision). Swin requiere
#    MMDetection/Detectron2, cuya instalación con la versión de PyTorch de Colab no es estable.
# 2. **Hover-Net reimplementado.** Se conservan las ramas NP y HV, sus pérdidas y su post-proceso; no se usa el
#    repositorio oficial (histología, PyTorch antiguo).
# 3. **MedSAM no evaluado**; SAM ViT-B en sus tres modos.
# 4. **HPO de los modelos preentrenados** (Mask R-CNN, Cellpose, SAM, Hover-Net) restringido a los parámetros de
#    inferencia y post-proceso, con el mismo número de trials (12) que U-Net++; los hiperparámetros de
#    entrenamiento se fijan en los valores recomendados por sus autores. Introduce un sesgo a favor de U-Net++,
#    que sí se optimiza en entrenamiento; se discute como amenaza a la validez.
# 5. **Validación anidada con split interno único** (85/15 agrupado) en lugar de k-fold interno.
# 6. **Ablaciones de segmentación a 256 px** (salvo la ablación de resolución) para caber en el presupuesto.
# 7. **5-fold de la mejor configuración** solo para U-Net++ y el baseline clásico; los modelos pesados reportan
#    varianza entre 3 semillas sobre el split fijo.
# 8. **Ablación de preentrenamiento** con Cellpose (desde cero vs `cyto2`, preentrenado en microscopía) y con el
#    codificador del modelo estilo Hover-Net (desde cero vs ImageNet), porque U-Net++ es desde cero por diseño.
# 9. **Texto fuera de distribución**: no existe otra fuente o periodo en el dataset; se reporta como limitación.
# 10. **Anotación manual de errores y de ruido de etiquetas**: un único anotador (el autor), por lo que no se
#     puede calcular κ de Cohen entre anotadores.

# %%
tag = subprocess.run(["git", "rev-list", "-n", "1", "prereg"], capture_output=True, text=True).stdout.strip()
head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
print("commit de la etiqueta prereg:", tag or "(no encontrada)")
print("commit ejecutado:            ", head)
