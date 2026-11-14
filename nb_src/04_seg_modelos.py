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
# # 4. Segmentación: modelos, HPO y evaluación en test
#
# Seis paradigmas evaluados con **el mismo código** sobre el mismo test sellado (15 %, agrupado por placa/fecha):
# baseline clásico, Mask R-CNN, modelo estilo Hover-Net, U-Net++ propia, Cellpose y SAM (tres modos).
# Los hiperparámetros de post-proceso/inferencia se ajustan **solo** en validación (Optuna, 12 trials por modelo).
# El test se evalúa una vez por modelo final. Unidad de análisis: imagen.

# %%
import json
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sst

from dl03 import seg_data as sd
from dl03 import seg_experiments as se
from dl03 import stats as st
from dl03.viz import CELL_COLORS, PALETTE, ci_table, error_map, overlay, save_json, save_table, savefig, setup

cfg = setup()
PRE = not cfg.is_smoke                 # pesos preentrenados (en smoke no hay descargas)
images, instances = sd.build_index(cfg)
splits = sd.Splits.from_json((cfg.out_dir / "seg" / "splits.json").read_text())
design = json.loads((cfg.out_dir / "seg" / "eda_design.json").read_text())
S_HEAVY = cfg.seg["seeds_heavy"]
RES = cfg.seg["img_size"]
NT = cfg.seg["hpo_trials"]
print(f"train {len(splits.train)} · val {len(splits.val)} · test {len(splits.test)} · semillas pesadas {S_HEAVY}")
timings = {}

# %% [markdown]
# ## 4.1 Baseline clásico (umbral + watershed)

# %%
t0 = time.time()
se.run_classical(cfg, images, splits, NT)
timings["classical"] = time.time() - t0

# %% [markdown]
# ## 4.2 U-Net++ (propia): HPO de entrenamiento con Optuna
# Espacio: tasa de aprendizaje, lote, optimizador y *scheduler* (coseno o warmup + coseno ≈ one-cycle). Se entrena
# a la resolución de ablación para caber en el presupuesto; objetivo = AP de validación con post-proceso por
# defecto; poda con MedianPruner sobre la pérdida de validación.

# %%
import optuna  # noqa: E402

from dl03.seg_models.unetpp import probs_to_instances  # noqa: E402

hpo_file = cfg.out_dir / "seg" / "params" / "unetpp_hpo.json"
if not hpo_file.exists():
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    PP0 = dict(thr_fg=0.5, thr_in=0.5, h=1.0, min_size=design["min_size_px"])

    def objective(trial):
        hp = dict(lr=trial.suggest_float("lr", 1e-4, 3e-3, log=True), batch=trial.suggest_categorical("batch", [4, 8]),
                  optimizer=trial.suggest_categorical("optimizer", ["adam", "adamw"]),
                  schedule=trial.suggest_categorical("schedule", ["cosine", "warmup_cosine"]))
        model = se.train_unetpp(cfg, images, splits.train, splits.val, 0, f"hpo_t{trial.number}",
                                res=cfg.seg["ablation_size"], hp=hp, epochs=max(2, cfg.seg["unetpp"]["epochs"] // 2),
                                trial=trial)
        pv = se.unetpp_probs(model, cfg, images, splits.val, cfg.seg["ablation_size"])
        return float(np.mean([se.sm.sartorius_score(sd.load_label(cfg, i), probs_to_instances(pv[i], PP0))
                              for i in splits.val]))
    t0 = time.time()
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=cfg.seed),
                                pruner=optuna.pruners.MedianPruner(n_warmup_steps=3))
    study.optimize(objective, n_trials=NT)
    try:
        imp = optuna.importance.get_param_importances(study)
    except Exception:
        imp = {}
    tr_df = study.trials_dataframe()
    hpo_file.parent.mkdir(parents=True, exist_ok=True)
    hpo_file.write_text(json.dumps({"best": study.best_params, "value": study.best_value, "importance": imp,
                                    "trials": tr_df[[c for c in tr_df.columns if c.startswith(("params_", "value",
                                                                                                "state"))]]
                                    .astype(str).to_dict("list"), "seconds": time.time() - t0}))
hpo = json.loads(hpo_file.read_text())
print("mejores hiperparámetros:", hpo["best"], "| AP val:", round(hpo["value"], 3))
print("importancia (fANOVA):", {k: round(v, 3) for k, v in hpo["importance"].items()})
trials = pd.DataFrame(hpo["trials"])
trials["value"] = pd.to_numeric(trials["value"], errors="coerce")
fig, ax = plt.subplots(1, 4, figsize=(15, 3))
for a, prm in zip(ax, ["params_lr", "params_batch", "params_optimizer", "params_schedule"]):
    if prm == "params_lr":
        a.scatter(pd.to_numeric(trials[prm]), trials["value"])
        a.set_xscale("log")
    else:
        trials.groupby(prm)["value"].mean().plot.bar(ax=a, rot=0, color=PALETTE[0])
    a.set(title=prm.replace("params_", ""), ylabel="AP val")
fig.suptitle("Dependencia parcial (marginal) de los hiperparámetros de U-Net++")
savefig(cfg, "04_unetpp_hpo")
plt.show()

# %% [markdown]
# ## 4.3 Entrenamiento final de los modelos con aprendizaje (3 semillas cada uno)

# %%
best_hp = {k: (int(v) if k == "batch" else v) for k, v in hpo["best"].items()}
runs = {}
for s in S_HEAVY:
    t0 = time.time()
    runs[f"unetpp_s{s}"] = se.unetpp_full(cfg, images, splits, s, f"unetpp_s{s}", res=RES, hp=best_hp)
    runs[f"hovernet_s{s}"] = se.hovernet_full(cfg, images, splits, s, f"hovernet_s{s}", pretrained=PRE, res=RES)
    runs[f"maskrcnn_s{s}"] = se.maskrcnn_full(cfg, images, splits, s, f"maskrcnn_s{s}",
                                              anchors=tuple(design["anchors_maskrcnn"]), pretrained=PRE)
    runs[f"cellpose_ft_s{s}"] = se.cellpose_full(cfg, images, splits, s, f"cellpose_ft_s{s}",
                                                 init=cfg.seg["cellpose"]["model"] if PRE else None)
    se.free_memory()
    timings[f"seed{s}"] = time.time() - t0
if PRE:
    runs["cellpose_zero"] = se.cellpose_full(cfg, images, splits, 0, "cellpose_zero", init="cyto2", zero_shot=True)
se.free_memory()
runs["sam_auto"] = se.sam_full(cfg, images, splits, "auto", "sam_auto", design["sam_max_area"], pretrained=PRE)
se.free_memory()
runs["sam_boxes"] = se.sam_full(cfg, images, splits, "boxes", "sam_boxes", design["sam_max_area"],
                                boxes_tag=f"maskrcnn_s{S_HEAVY[0]}", pretrained=PRE)
se.free_memory()
runs["sam_ft"] = se.sam_full(cfg, images, splits, "finetune", "sam_ft", design["sam_max_area"],
                             boxes_tag=f"maskrcnn_s{S_HEAVY[0]}", pretrained=PRE)
pd.DataFrame({k: {"AP val (post-proceso ajustado)": v[1]["val_ap"]} for k, v in runs.items()}).T.round(3)

# %% [markdown]
# ## 4.4 Evaluación en test (mismo código para todos)

# %%
NAMES = {"classical": "Baseline clásico", "maskrcnn": "Mask R-CNN (R50-FPN)", "hovernet": "Estilo Hover-Net",
         "unetpp": "U-Net++ + watershed", "cellpose_ft": "Cellpose ajustado", "cellpose_zero": "Cellpose cyto2 cero-shot",
         "sam_auto": "SAM cero-shot", "sam_boxes": "SAM + cajas", "sam_ft": "SAM decod. ajustado"}
frames = [se.evaluate(cfg, images, se.load_preds(se.run_dir(cfg, "preds", "classical_s0"), splits.test), "classical")]
for tag in runs:
    base, _, seed = tag.rpartition("_s") if "_s" in tag and tag.split("_s")[-1].isdigit() else (tag, "", "0")
    frames.append(se.evaluate(cfg, images, se.load_preds(se.run_dir(cfg, "preds", tag), splits.test), base,
                              seed=int(seed)))
per_seed = se.add_strata(pd.concat(frames, ignore_index=True), images, instances)
per_seed["count_err"] = per_seed["n_pred"] - per_seed["n_gt"]
per_seed.to_csv(cfg.out_dir / "seg" / "metrics_test_per_seed.csv", index=False)
metric_cols = ["ap", "ap50", "ap75", "pq", "sq", "rq", "aji", "inst_precision", "inst_recall", "inst_f1", "dice",
               "iou", "precision", "recall", "auc_roc", "auc_pr", "balanced_accuracy", "hd95", "boundary_f1",
               "count_err"] + [c for c in per_seed.columns if c.startswith(("p@", "err_"))]
# unidad de análisis = imagen: se promedia sobre semillas por imagen antes de comparar modelos
img = per_seed.groupby(["model", "id", "cell_type", "densidad", "tamano"], observed=True)[metric_cols].mean() \
    .reset_index()
img.to_csv(cfg.out_dir / "seg" / "metrics_test.csv", index=False)
seed_var = per_seed.groupby(["model", "seed"])["ap"].mean().groupby("model").agg(["mean", "std", "count"])
seed_var.rename(columns={"mean": "AP medio entre semillas", "std": "DE entre semillas", "count": "n semillas"}).round(4)

# %% [markdown]
# ### Tabla de resultados (media [IC 95 %], bootstrap sobre imágenes de test, 10 000 remuestreos)

# %%
def latency(tag):
    f = se.run_dir(cfg, "preds", tag) / "_time.json"
    return json.loads(f.read_text())["seconds_per_image"] * 1e3 if f.exists() else np.nan


def n_params(tag):
    f = se.run_dir(cfg, "models") / f"{tag}.json"
    return json.loads(f.read_text()).get("params", np.nan) if f.exists() else np.nan


order = img.groupby("model")["ap"].mean().sort_values(ascending=False).index.tolist()
img["model"] = pd.Categorical(img["model"], order)
table = ci_table(img.sort_values("model"), "model", ["ap", "ap50", "pq", "dice", "iou", "precision", "recall",
                                                    "auc_roc", "hd95", "balanced_accuracy"], n_boot=cfg.n_boot)
first_tag = {m: next((t for t in ["classical_s0"] + list(runs) if t.startswith(m + "_s") or t == m), m) for m in order}
table["latencia (ms/img)"] = [round(latency(first_tag[m]), 1) for m in table.index]
table["parámetros"] = [n_params(first_tag[m]) for m in table.index]
table.index = [NAMES.get(m, m) for m in table.index]
save_table(cfg, "seg_resultados_test", table)
table

# %% [markdown]
# ### Inferencia estadística: Friedman + Wilcoxon pareado con Holm, tamaños de efecto, diferencia crítica

# %%
wide = img.pivot_table(index="id", columns="model", values="ap", observed=True).dropna(axis=1)
cmp = st.compare_models(wide, alpha=cfg.alpha, delta=0.02)
print(f"Friedman χ² = {cmp['friedman_stat']:.2f}, p = {cmp['friedman_p']:.2e}, n = {cmp['n']} imágenes")
pw = cmp["pairwise"].copy()
pw[["A", "B"]] = pw[["A", "B"]].map(lambda m: NAMES.get(m, m))
save_table(cfg, "seg_pairwise", pw)
fig, ax = plt.subplots(figsize=(9, 1.6 + 0.35 * wide.shape[1]))
st.plot_cd_diagram(cmp["mean_ranks"].rename(index=NAMES), cmp["n"],
                   cmp["pairwise"].assign(A=lambda d: d.A.map(NAMES), B=lambda d: d.B.map(NAMES)), ax=ax,
                   title="AP@[.5:.95] por imagen de test")
savefig(cfg, "04_cd_diagram")
plt.show()
pw.sort_values("p_holm").head(15).round(4)

# %% [markdown]
# ### Verificación jerárquica: modelo mixto con intercepto aleatorio por grupo (placa/fecha)

# %%
gcol, groups = sd.choose_group_column(images)
long = img.assign(group=img["id"].map(dict(zip(images.id, groups))))
try:
    tab_mm, icc, _ = st.mixedlm_check(long, "ap", "model", "group")
    print(f"ICC del grupo ({gcol}) sobre AP: {icc:.3f}")
    save_table(cfg, "seg_mixedlm", tab_mm)
    display(tab_mm.round(4))  # noqa: F821
except Exception as e:  # pragma: no cover - pocos grupos en smoke
    print("modelo mixto no estimable:", e)

# %% [markdown]
# ## 4.5 Estratificación (tipo celular, densidad, tamaño) — H3

# %%
heat = img.pivot_table(index="model", columns="cell_type", values="ap", observed=True).loc[order]
fig, ax = plt.subplots(1, 3, figsize=(16, 4))
im_ = ax[0].imshow(heat.values, cmap="viridis", aspect="auto")
ax[0].set_xticks(range(heat.shape[1]), heat.columns)
ax[0].set_yticks(range(heat.shape[0]), [NAMES.get(m, m) for m in heat.index])
for i in range(heat.shape[0]):
    for j in range(heat.shape[1]):
        ax[0].text(j, i, f"{heat.values[i, j]:.2f}", ha="center", va="center", color="w", fontsize=8)
ax[0].set_title("AP por modelo × tipo celular")
for a, col in zip(ax[1:], ["densidad", "tamano"]):
    img.pivot_table(index="model", columns=col, values="ap", observed=True).loc[order].rename(index=NAMES).plot.barh(
        ax=a)
    a.set_title(f"AP por tercil de {col}")
plt.tight_layout()
savefig(cfg, "04_estratificado")
plt.show()
ranks = {ct: heat[ct].rank(ascending=False).astype(int).to_dict() for ct in heat.columns}
same = len({tuple(heat[ct].sort_values(ascending=False).index) for ct in heat.columns}) == 1
h3 = {"orden por tipo": {ct: [NAMES.get(m, m) for m in heat[ct].sort_values(ascending=False).index]
                         for ct in heat.columns},
      "H3 (el ranking cambia por tipo)": "refutada (orden idéntico)" if same else "no refutada (el orden cambia)"}
save_json(cfg, "seg_H3", h3)
h3

# %% [markdown]
# ## 4.6 Píxel vs instancia — H2

# %%
with_prob = img.groupby("model", observed=True)["auc_roc"].mean().dropna()
ap_m = img.groupby("model", observed=True)["ap"].mean().loc[with_prob.index]
rho = sst.spearmanr(with_prob.values, ap_m.values).statistic if len(with_prob) > 2 else np.nan
h2 = {"AUC píxel media por modelo": with_prob.round(3).rename(index=NAMES).to_dict(),
      "AP medio por modelo": ap_m.round(3).rename(index=NAMES).to_dict(), "Spearman(AUC, AP)": rho,
      "todas las AUC > 0.95": bool((with_prob > 0.95).all()),
      "H2": "refutada" if (rho > 0.9 or not (with_prob > 0.95).all()) else "no refutada"}
save_json(cfg, "seg_H2", h2)
print("Nota: la 'probabilidad' del baseline clásico es una intensidad normalizada y la de SAM/Mask R-CNN es "
      "puntaje×máscara: su AUC se reporta pero no es probabilística.")
h2

# %% [markdown]
# ## 4.7 Curvas y distribuciones

# %%
pcols = [c for c in img.columns if c.startswith("p@")]
fig, ax = plt.subplots(1, 3, figsize=(17, 4))
for k, m in enumerate(order):
    sub = img[img.model == m]
    ax[0].plot([float(c[2:]) for c in pcols], sub[pcols].mean().values, "o-", ms=3, label=NAMES.get(m, m),
               color=PALETTE[k % len(PALETTE)])
ax[0].set(title="TP/(TP+FP+FN) vs umbral de IoU", xlabel="umbral IoU")
ax[0].legend(fontsize=7)
ax[1].boxplot([img[img.model == m]["ap"] for m in order], tick_labels=[NAMES.get(m, m) for m in order], vert=False)
ax[1].set_title("AP por imagen (test)")
ax[2].boxplot([img[img.model == m]["dice"] for m in order], tick_labels=[""] * len(order), vert=False)
ax[2].set_title("Dice de píxel por imagen")
plt.tight_layout()
savefig(cfg, "04_curvas_distribuciones")
plt.show()

# %% [markdown]
# Curvas ROC y PR de píxel: ilustración por modelo sobre una imagen representativa por tipo celular (cada curva es
# de **una** imagen; las métricas agregadas de arriba son por imagen y nunca mezclan píxeles de imágenes distintas).

# %%
from sklearn.metrics import precision_recall_curve, roc_curve  # noqa: E402

def median_case(ct):
    a = img[(img.cell_type == ct) & (img.model == order[0])].sort_values("ap")
    return a.id.iloc[len(a) // 2]


rep = {ct: median_case(ct) for ct in heat.columns}
prob_tags = {m: first_tag[m] for m in order if not np.isnan(img[img.model == m]["auc_roc"].mean())}
fig, ax = plt.subplots(2, len(rep), figsize=(5 * len(rep), 8))
ax = np.atleast_2d(ax)
for j, (ct, iid) in enumerate(rep.items()):
    gt = sd.load_label(cfg, iid) > 0
    for k, (m, tag) in enumerate(prob_tags.items()):
        lbl, prob = se.load_preds(se.run_dir(cfg, "preds", tag), [iid])[iid]
        if prob is None:
            continue
        fpr, tpr, _ = roc_curve(gt.ravel()[::7], prob.ravel()[::7])
        pr, rc, _ = precision_recall_curve(gt.ravel()[::7], prob.ravel()[::7])
        ax[0, j].plot(fpr, tpr, label=NAMES.get(m, m), color=PALETTE[k % len(PALETTE)])
        ax[1, j].plot(rc, pr, color=PALETTE[k % len(PALETTE)])
    ax[0, j].set(title=f"ROC píxel · {ct}", xlabel="FPR", ylabel="TPR")
    ax[1, j].set(title=f"PR píxel · {ct} (π = {gt.mean():.2f})", xlabel="recall", ylabel="precisión")
ax[0, 0].legend(fontsize=7)
plt.tight_layout()
savefig(cfg, "04_roc_pr")
plt.show()

# %% [markdown]
# ## 4.8 Conteo celular (utilidad práctica): MAE, RMSE, correlación y Bland–Altman

# %%
cnt = per_seed.groupby(["model", "id"]).agg(n_gt=("n_gt", "first"), n_pred=("n_pred", "mean")).reset_index()
crow = []
for m, g in cnt.groupby("model"):
    e = g.n_pred - g.n_gt
    crow.append({"modelo": NAMES.get(m, m), "MAE": e.abs().mean(), "RMSE": np.sqrt((e ** 2).mean()),
                 "sesgo medio": e.mean(), "r de Pearson": np.corrcoef(g.n_gt, g.n_pred)[0, 1]})
count_tab = pd.DataFrame(crow).set_index("modelo").sort_values("MAE")
save_table(cfg, "seg_conteo", count_tab)
best = order[0]
g = cnt[cnt.model == best]
mean_, diff_ = (g.n_gt + g.n_pred) / 2, g.n_pred - g.n_gt
fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(mean_, diff_, s=10)
for v, ls in [(diff_.mean(), "-"), (diff_.mean() + 1.96 * diff_.std(), "--"), (diff_.mean() - 1.96 * diff_.std(), "--")]:
    ax.axhline(v, color="gray", ls=ls)
ax.set(title=f"Bland–Altman del conteo · {NAMES.get(best, best)}", xlabel="media (ref., pred.)",
       ylabel="pred. − ref.")
savefig(cfg, "04_bland_altman")
plt.show()
count_tab.round(2)

# %% [markdown]
# ## 4.9 Taxonomía de errores (por modelo y tipo celular) y su relación con densidad y tamaño

# %%
ecols = [c for c in img.columns if c.startswith("err_")]
tax = img.groupby(["model", "cell_type"], observed=True)[ecols + []].sum()
tax = tax.div(img.groupby(["model", "cell_type"], observed=True)[ecols].sum().sum(1), axis=0)
tax.columns = [c.replace("err_", "") for c in tax.columns]
fig, ax = plt.subplots(figsize=(12, 0.35 * len(tax) + 1.5))
tax.rename(index=NAMES, level=0).plot.barh(stacked=True, ax=ax, width=0.8)
ax.set(title="Composición de los errores de instancia", xlabel="fracción de errores")
ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left")
savefig(cfg, "04_taxonomia")
plt.show()
corr = []
for m in order:
    sub = img[img.model == m]
    for e in ["err_fusionada", "err_dividida", "err_omitida"]:
        tot = sub[ecols].sum(1).replace(0, np.nan)
        cov = sub["id"].map(images.set_index("id")["coverage"])
        r_ = sst.spearmanr(cov, sub[e] / tot, nan_policy="omit").statistic
        corr.append({"modelo": NAMES.get(m, m), "error": e[4:], "ρ con densidad (cobertura)": r_})
corr = pd.DataFrame(corr).pivot(index="modelo", columns="error", values="ρ con densidad (cobertura)")
save_table(cfg, "seg_taxonomia", tax)
save_table(cfg, "seg_taxonomia_vs_densidad", corr)
corr.round(2)

# %% [markdown]
# ## 4.10 Comparación visual y mapas de error
# Mapa de error a nivel de píxel: **rojo = falso positivo, azul = falso negativo**, gris = acierto. Muestra al azar
# (semilla fija) y los 5 mejores y 5 peores casos del mejor modelo (sin seleccionar solo casos favorables).

# %%
def show_cases(ids, title, fname):
    fig, ax = plt.subplots(len(ids), 4, figsize=(13, 3 * len(ids)))
    ax = np.atleast_2d(ax)
    for r_, iid in enumerate(ids):
        x = sd.load_image(images.set_index("id").loc[iid, "path"])
        gt = sd.load_label(cfg, iid)
        pr = se.load_preds(se.run_dir(cfg, "preds", first_tag[best]), [iid])[iid][0]
        ap_ = img[(img.model == best) & (img.id == iid)]["ap"].iloc[0]
        ax[r_, 0].imshow(x, cmap="gray")
        ax[r_, 0].set_title(f"{iid} · {images.set_index('id').loc[iid, 'cell_type']}", fontsize=8)
        ax[r_, 0].axis("off")
        overlay(x, gt, ax[r_, 1], "referencia")
        overlay(x, pr, ax[r_, 2], f"predicción (AP {ap_:.2f})")
        ax[r_, 3].imshow(error_map(gt, pr))
        ax[r_, 3].set_title("error (rojo FP · azul FN)", fontsize=8)
        ax[r_, 3].axis("off")
    fig.suptitle(title)
    plt.tight_layout()
    savefig(cfg, fname)
    plt.show()


bsub = img[img.model == best].sort_values("ap")
show_cases(list(bsub.sample(min(3, len(bsub)), random_state=cfg.seed).id), f"{NAMES.get(best, best)}: muestra al azar",
           "04_casos_azar")
show_cases(list(bsub.id.tail(5)[::-1]), f"{NAMES.get(best, best)}: 5 mejores", "04_casos_mejores")
show_cases(list(bsub.id.head(5)), f"{NAMES.get(best, best)}: 5 peores", "04_casos_peores")

# %% [markdown]
# Mismas imágenes para todos los modelos (una por tipo celular):

# %%
fig, ax = plt.subplots(len(rep), len(order) + 1, figsize=(3 * (len(order) + 1), 3 * len(rep)))
ax = np.atleast_2d(ax)
for i, iid in enumerate(rep.values()):
    x = sd.load_image(images.set_index("id").loc[iid, "path"])
    overlay(x, sd.load_label(cfg, iid), ax[i, 0], "referencia")
    for j, m in enumerate(order, 1):
        pr = se.load_preds(se.run_dir(cfg, "preds", first_tag[m]), [iid])[iid][0]
        ax[i, j].imshow(error_map(sd.load_label(cfg, iid), pr))
        ax[i, j].set_title(NAMES.get(m, m), fontsize=7)
        ax[i, j].axis("off")
plt.tight_layout()
savefig(cfg, "04_comparacion_modelos")
plt.show()

# %% [markdown]
# ## 4.11 Diagnóstico sesgo–varianza: curvas de entrenamiento y validación (modelos Keras)

# %%
fig, ax = plt.subplots(1, 2, figsize=(12, 3.5))
for a, pref in zip(ax, ["unetpp_s", "hovernet_s"]):
    for s in S_HEAVY:
        f = se.run_dir(cfg, "logs") / f"{pref}{s}.csv"
        if f.exists():
            h = pd.read_csv(f)
            a.plot(h["epoch"], h["loss"], "-", color=PALETTE[s % 8], label=f"train s{s}")
            a.plot(h["epoch"], h["val_loss"], "--", color=PALETTE[s % 8], label=f"val s{s}")
    a.set(title=pref.rstrip("_s"), xlabel="época", ylabel="pérdida")
    a.legend(fontsize=7)
savefig(cfg, "04_curvas_aprendizaje")
plt.show()

# %% [markdown]
# ## 4.12 5-fold de la mejor configuración (U-Net++ y baseline clásico)
# Estimación de la varianza por partición (a la resolución de ablación; desviación declarada 0.6-7). Cada fold usa
# sus propias imágenes de evaluación (nunca el test sellado).

# %%
from dl03.seg_models.postproc import classical_segment  # noqa: E402

cls_params = json.loads((cfg.out_dir / "seg" / "params" / "classical.json").read_text())["params"]
cv_rows = []
for k, (tr_ids, te_ids) in enumerate(splits.folds):
    te_set = set(te_ids)
    val_k = [i for i in splits.val if i not in te_set]          # validación del fold: nunca toca su evaluación
    tr_k = [i for i in tr_ids if i not in set(val_k)]
    sub_split = sd.Splits(splits.group_col, tr_k, val_k, te_ids, [])
    pdir, _ = se.unetpp_full(cfg, images, sub_split, 0, f"cv_unetpp_f{k}", res=cfg.seg["ablation_size"],
                             test_ids=te_ids, hp=best_hp)
    u = se.evaluate(cfg, images, se.load_preds(pdir, te_ids), "unetpp", with_prob=False)
    c = se.evaluate(cfg, images, {i: classical_segment(se.native_image(images, i), **cls_params) for i in te_ids},
                    "classical", with_prob=False)
    cv_rows += [{"fold": k, "modelo": "U-Net++", "AP": u.ap.mean()}, {"fold": k, "modelo": "clásico", "AP": c.ap.mean()}]
cv = pd.DataFrame(cv_rows)
save_table(cfg, "seg_cv5", cv)
cv.groupby("modelo")["AP"].agg(["mean", "std", "min", "max"]).round(4)

# %%
save_json(cfg, "seg_tiempos_cap04", timings)
print({k: f"{v / 60:.1f} min" for k, v in timings.items()})
