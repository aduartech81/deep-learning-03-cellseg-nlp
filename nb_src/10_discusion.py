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
# # 10. Comparación final, decisión, hipótesis, autoevaluación crítica y conclusiones
#
# Todas las cifras de este capítulo se leen de las tablas generadas por los capítulos 3–9 (`results/tables`).

# %%
import json
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dl03.config import environment_report
from dl03.viz import PALETTE, save_json, save_table, savefig, setup

cfg = setup()
T = cfg.out_dir / "tables"


def jload(name):
    p = T / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else {}


def first(s):
    """'0.123 [0.1, 0.2]' -> 0.123"""
    try:
        return float(str(s).split()[0])
    except ValueError:
        return np.nan


seg = pd.read_csv(T / "seg_resultados_test.csv", index_col=0)
nlp = pd.read_csv(T / "nlp_resultados_test.csv", index_col=0)
rob = pd.read_csv(T / "seg_robustez_auc_relativa.csv", index_col=0) if (T / "seg_robustez_auc_relativa.csv").exists() \
    else pd.DataFrame()

# %% [markdown]
# ## 10.1 Análisis multicriterio por dominio (pesos justificados y sensibilidad a los pesos)
#
# Criterios normalizados a [0, 1] (1 = mejor) dentro de cada dominio:
# * **Desempeño** (AP o F1 macro, punto central del IC).
# * **Costo/latencia** (inverso del logaritmo de la latencia).
# * **Robustez** (AUC relativa media bajo perturbaciones, solo modelos evaluados; resto = valor medio).
# * **Calibración** (1 − ECE; solo texto).
# * **Desplegabilidad** (1 si cabe en 512 MB sin GPU, 0.5 si requiere PyTorch/TensorFlow en CPU, 0 si requiere GPU).
#
# Pesos base: desempeño 0.45, costo 0.20, robustez 0.15, calibración 0.10 (texto) y desplegabilidad 0.10
# (imágenes: robustez 0.20, sin calibración). La sensibilidad se evalúa con 2000 vectores de pesos de Dirichlet
# centrados en los pesos base: ¿con qué frecuencia gana cada modelo?

# %%
def norm(v, higher=True):
    v = pd.Series(v, dtype=float)
    if v.nunique() <= 1:
        return v * 0 + 1
    z = (v - v.min()) / (v.max() - v.min())
    return z if higher else 1 - z


DEPLOY_SEG = {"Baseline clásico": 1, "U-Net++ + watershed": 1, "Estilo Hover-Net": 0.5, "Mask R-CNN (R50-FPN)": 0.5,
              "Cellpose ajustado": 0.5, "Cellpose cyto2 cero-shot": 0.5, "SAM cero-shot": 0, "SAM + cajas": 0,
              "SAM decod. ajustado": 0}
DEPLOY_NLP = {"Mayoritaria": 1, "TF-IDF + LR": 1, "NB-SVM": 1, "TF-IDF + XGBoost": 1, "fastText": 1, "CNN-1D": 0.5,
              "Word2Vec + BiLSTM": 0.5, "DistilBERT": 0.5}
rob_mean = rob.mean().to_dict() if len(rob) else {}
seg_c = pd.DataFrame({"desempeño": norm(seg["ap"].map(first)),
                      "costo": norm(np.log(seg["latencia (ms/img)"].astype(float).clip(lower=1)), higher=False),
                      "robustez": norm(pd.Series({m: rob_mean.get({"U-Net++ + watershed": "U-Net++"}.get(m, m), np.nan)
                                                  for m in seg.index}).fillna(np.nanmean(list(rob_mean.values())
                                                                                          or [0.5]))),
                      "desplegabilidad": pd.Series(DEPLOY_SEG).reindex(seg.index).fillna(0.5)})
nlp_c = pd.DataFrame({"desempeño": norm(nlp["f1_macro"].map(first)),
                      "costo": norm(np.log(nlp["latencia p50 (ms)"].astype(float).clip(lower=0.01)), higher=False),
                      "calibración": norm(1 - nlp["ece"].map(first)),
                      "desplegabilidad": pd.Series(DEPLOY_NLP).reindex(nlp.index).fillna(0.5)})
W = {"seg": {"desempeño": 0.45, "costo": 0.20, "robustez": 0.20, "desplegabilidad": 0.15},
     "nlp": {"desempeño": 0.45, "costo": 0.20, "calibración": 0.15, "desplegabilidad": 0.20}}


def mcda(crit, w, n=2000, conc=50, seed=0):
    base = crit[list(w)] @ pd.Series(w)
    rng = np.random.default_rng(seed)
    wins = pd.Series(0, index=crit.index)
    for v in rng.dirichlet(np.array(list(w.values())) * conc, n):
        wins[(crit[list(w)] @ v).idxmax()] += 1
    return pd.DataFrame({"puntaje (pesos base)": base, "% de victorias (pesos perturbados)": 100 * wins / n}) \
        .sort_values("puntaje (pesos base)", ascending=False)


m_seg, m_nlp = mcda(seg_c, W["seg"]), mcda(nlp_c, W["nlp"])
save_table(cfg, "decision_seg", m_seg)
save_table(cfg, "decision_nlp", m_nlp)
display(m_seg.round(3))  # noqa: F821
m_nlp.round(3)

# %% [markdown]
# ## 10.2 Costo total de propiedad (estimación)
# Supuestos explícitos: GPU de Colab ≈ USD 0.50–1.20/h según tipo (L4/A100), CPU para inferencia en un servicio
# web pequeño ≈ USD 7–25/mes, reentrenamiento trimestral, revisión humana de los casos derivados (8.4).

# %%
times = {}
for f in (cfg.out_dir / "seg" / "models").glob("*.json"):
    d = json.loads(f.read_text())
    times[d.get("tag", f.stem)] = d.get("train_seconds", np.nan)
finals = pd.read_csv(cfg.out_dir / "nlp" / "finals.csv")
gpu_h = (np.nansum(list(times.values())) + finals["train_seconds"].sum()) / 3600
tco = pd.DataFrame({
    "concepto": ["entrenamiento total del proyecto (GPU-h medidas)", "costo de cómputo del proyecto (USD, rango)",
                 "inferencia mensual (servicio CPU 512 MB)", "reentrenamiento trimestral (GPU-h, modelos desplegados)",
                 "mantenimiento (monitoreo y auditorías, h-persona/mes)"],
    "valor": [f"{gpu_h:.1f}", f"{0.5 * gpu_h:.0f}–{1.2 * gpu_h:.0f}", "USD 7–25", "≈ 1", "4–8"]})
save_table(cfg, "tco", tco)
tco

# %% [markdown]
# ## 10.3 Hipótesis pre-registradas: veredicto

# %%
H = {"H1": jload("seg_H1"), "H2": jload("seg_H2"), "H3": jload("seg_H3"), "H4": jload("nlp_H4"),
     "H5": jload("nlp_H5")}
ver = pd.DataFrame({
    "veredicto": [H["H1"].get("H1"), H["H2"].get("H2"), H["H3"].get("H3 (el ranking cambia por tipo)"),
                  "H0 " + str(H["H4"].get("H0 (sin diferencia relevante)")), H["H5"].get("H5")],
    "evidencia": [
        "; ".join(f"{k}: {v}" for k, v in H["H1"].items() if k.startswith("Δ")),
        f"Spearman(AUC, AP) = {H['H2'].get('Spearman(AUC, AP)')}; todas AUC > 0.95: {H['H2'].get('todas las AUC > 0.95')}",
        json.dumps(H["H3"].get("orden por tipo", {}), ensure_ascii=False)[:300],
        f"ΔF1 = {H['H4'].get('ΔF1 (DistilBERT − XGBoost), media de semillas')}; p Holm = {H['H4'].get('p Holm')}",
        f"ECE DistilBERT = {H['H5'].get('ECE DistilBERT')}; ECE LR = {H['H5'].get('ECE TF-IDF + LR')}; "
        f"reducción = {H['H5'].get('reducción tras T [IC 95 %]')}"]},
    index=["H1 preentrenado > desde cero con poco dato", "H2 píxel satura, instancia discrimina",
           "H3 Simpson por tipo celular", "H4 transformer ≈ TF-IDF + XGBoost", "H5 calibración del fine-tuning"])
save_table(cfg, "hipotesis", ver)
ver

# %% [markdown]
# ## 10.4 Autoevaluación crítica tipo revisión por pares
# Las cinco objeciones más serias que haría un revisor externo, con la respuesta basada en la evidencia o el
# reconocimiento explícito de la limitación.

# %%
abl = pd.read_csv(T / "seg_ablaciones.csv", index_col=0) if (T / "seg_ablaciones.csv").exists() else None
cv5 = pd.read_csv(T / "seg_cv5.csv", index_col=0) if (T / "seg_cv5.csv").exists() else None
nest = pd.read_csv(T / "nlp_anidada.csv", index_col=0) if (T / "nlp_anidada.csv").exists() else None
short = pd.read_csv(T / "nlp_atajo_titulo.csv", index_col=0) if (T / "nlp_atajo_titulo.csv").exists() else None
obj = [
    ("El HPO no fue equitativo: U-Net++ optimizó entrenamiento y los modelos preentrenados solo post-proceso.",
     "Es cierto y está declarado (0.6-4). Si el sesgo favoreciera a U-Net++, su ventaja debería desaparecer frente "
     "a modelos ajustados con sus valores por defecto recomendados; la tabla de resultados muestra la posición real "
     "de U-Net++ y la conclusión se limita a «con este presupuesto»."),
    ("Tres semillas son pocas para modelos con varianza alta entre semillas.",
     "Se reporta la DE entre semillas junto al IC por imágenes; las comparaciones usan la imagen como unidad "
     "(promediando semillas). Donde la DE entre semillas es comparable a δ, la conclusión se marca como no "
     "concluyente." + (f" 5-fold de U-Net++: DE entre folds = {cv5[cv5['modelo'] == 'U-Net++']['AP'].std():.3f}."
                       if cv5 is not None and 'modelo' in cv5 else "")),
    ("Las ablaciones se evalúan en validación, no en test.",
     "Decisión deliberada para no desgastar el test sellado con decenas de variantes; el post-proceso se ajusta en "
     "la mitad A y se mide en la mitad B, agrupadas por placa. Las conclusiones de ablación son relativas entre "
     "variantes, no estimaciones del desempeño final."),
    ("El clasificador de texto podría estar aprendiendo el título del cargo y no el contenido.",
     "Se midió: " + ("; ".join(f"{r['modelo']} pasa de {r['F1 con título']:.3f} a {r['F1 sin título']:.3f}"
                               for _, r in short.iterrows()) if short is not None and len(short) else
                     "ver 8.2") + ". Es una limitación de validez de constructo que se declara."),
    ("El análisis de equidad es débil sin atributos protegidos reales.",
     "Correcto: los proxies de pronombres son escasos y ruidosos; por eso el énfasis está en pruebas contrafactuales "
     "y en la recomendación de no usar el modelo para decisiones automáticas sobre personas."),
]
review = pd.DataFrame(obj, columns=["objeción", "respuesta"])
save_table(cfg, "revision_pares", review)
pd.set_option("display.max_colwidth", 400)
review

# %% [markdown]
# ## 10.5 Amenazas a la validez
# * **Interna:** HPO no equitativo entre familias (declarado); semillas limitadas en modelos pesados; ajuste del
#   post-proceso en validación compartida entre modelos (mismo para todos).
# * **Externa:** un único proveedor, microscopio y protocolo (Sartorius); hojas de vida en inglés de una sola fuente
#   web, parcialmente anonimizadas; sin conjunto fuera de distribución real.
# * **De constructo:** AP@[.5:.95] castiga bordes imprecisos aunque el conteo sea correcto (por eso se reportan
#   métricas de conteo); en texto, la categoría a la que se postuló alguien no es su competencia real.
# * **De conclusión estadística:** muchas comparaciones (Holm), pocas unidades en algunos estratos, dependencia
#   intra-placa (verificada con modelo mixto).

# %% [markdown]
# ## 10.6 Reproducibilidad

# %%
env = environment_report()
co2 = {k: jload(k) for k in ["co2_cap05", "nlp_co2_cap07"]}
repro = {"comando único": "make reproduce  (o scripts/run_chapters.sh con DL03_PROFILE=full)",
         "perfil": cfg.profile, "semillas": {"base": cfg.seed, "seg pesados": cfg.seg["seeds_heavy"],
                                             "nlp ligeros": cfg.nlp["seeds_light"]},
         "entorno": {k: v for k, v in env.items() if v}, "GPU-h medidas (entrenamiento)": round(gpu_h, 2),
         "CO₂ medido (kg, capítulos con codecarbon)": co2,
         "commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()}
save_json(cfg, "reproducibilidad", repro)
repro

# %% [markdown]
# ## 10.7 Conclusiones, trabajo futuro y reflexión ética
#
# Las respuestas a las preguntas de investigación se redactan sobre las tablas de este capítulo y de los capítulos
# 4–9 (ver la sección de conclusiones del libro, `conclusiones.md`, escrita después de la corrida con los datos
# reales). Con el doble de presupuesto, la prioridad sería: 5 semillas y 5-fold para todos los modelos pesados, HPO
# de entrenamiento equitativo, Swin + Mask R-CNN y MedSAM, y un conjunto de texto de otra fuente para evaluar fuera
# de distribución. Con la mitad, se recortarían SAM cero-shot y las ablaciones a 512 px, manteniendo intactos el
# pre-registro, el test sellado y la estadística.
#
# **Uso de asistentes de IA.** El código, el diseño experimental y la redacción se desarrollaron con asistencia de
# un modelo de lenguaje (Claude, Anthropic). Toda cifra proviene de la ejecución de los notebooks; las funciones
# metodológicas están cubiertas por pruebas unitarias; el autor revisó y responde por la totalidad del contenido.
