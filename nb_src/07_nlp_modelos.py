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
# # 7. Texto: modelos, HPO, validación anidada y comparación estadística
#
# Presupuesto idéntico de HPO (Optuna TPE + poda, mismo número de trials) para los modelos ligeros; validación
# anidada (5 folds externos) para estimar el desempeño **del procedimiento completo** y su sesgo optimista; corridas
# finales con 5 semillas evaluadas una vez en el test sellado; DistilBERT con 5 semillas.

# %%
import json
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dl03 import nlp_experiments as ne
from dl03 import stats as st
from dl03.nlp_models import fit_model
from dl03.viz import PALETTE, save_json, save_table, savefig, setup

cfg = setup(seed_offset=7)
df, y, classes, groups, sp = ne.prepare(cfg)
X, K = df["text"].values, len(classes)
tr, va, te = map(np.array, (sp["train"], sp["val"], sp["test"]))
SEEDS, NT, CAP = cfg.nlp["seeds_light"], cfg.nlp["hpo_trials"], cfg.nlp["bilstm_max_epochs_cap"]
co2 = {}


def tracked(name, fn):
    try:
        from codecarbon import OfflineEmissionsTracker
        t = OfflineEmissionsTracker(country_iso_code="USA", log_level="error", save_to_file=False)
        t.start()
    except Exception:
        t = None
    t0 = time.time()
    out = fn()
    co2[name] = {"kg": (t.stop() if t else None), "segundos": time.time() - t0}
    return out


print(f"train {len(tr)} · val {len(va)} · test {len(te)} · {K} clases · trials {NT} · semillas {SEEDS}")

# %% [markdown]
# ## 7.1 HPO (train → validación) y corridas finales (5 semillas) de los modelos ligeros

# %%
NAMES = {"majority": "Mayoritaria", "tfidf_lr": "TF-IDF + LR", "nbsvm": "NB-SVM", "tfidf_xgb": "TF-IDF + XGBoost",
         "fasttext": "fastText", "cnn1d": "CNN-1D", "bilstm": "Word2Vec + BiLSTM", "transformer": "DistilBERT"}
best = {"majority": {}}
summary = []
for name in ["tfidf_lr", "nbsvm", "tfidf_xgb", "fasttext", "cnn1d", "bilstm"]:
    res = tracked(f"hpo_{name}", lambda n=name: ne.hpo(cfg, n, X, y, tr, va, K, NT, seed=cfg.seed, cap_epochs=CAP))
    best[name] = ne.full_params(name, res["best_params"], CAP)
    summary.append({"modelo": NAMES[name], "F1 val (mejor trial)": res["best_value"],
                    "minutos HPO": res["seconds"] / 60, "mejores hiperparámetros": json.dumps(best[name])})
save_json(cfg, "nlp_best_params", best)
pd.DataFrame(summary).set_index("modelo")

# %%
finals = []
for name in ["majority"] + list(best)[1:]:
    finals.append(tracked(f"final_{name}", lambda n=name: ne.final_runs(cfg, n, best[n], X, y, sp, K, SEEDS,
                                                                       save_model=True)))
finals = pd.concat(finals, ignore_index=True)

# %% [markdown]
# ## 7.2 DistilBERT: tasa de aprendizaje en validación (semilla 0) y 5 semillas
# Truncamiento cabeza + cola (128 + 382 tokens), warmup 10 %, decaimiento lineal, LR decreciente por capa (0.9),
# precisión mixta. Modelo pesado: HPO sobre un único split de validación (declarado).

# %%
tcfg = dict(cfg.nlp["transformer"])
lr_file = cfg.out_dir / "nlp" / "hpo" / "transformer_lr.json"
if not lr_file.exists():
    grid = [2e-5, 3e-5, 5e-5] if tcfg["name"] != "tiny-random" else [tcfg["lr"]]
    scores = {}
    for lr in grid:
        m = ne.fit_transformer(X[tr], y[tr], X[va], y[va], {**tcfg, "lr": lr}, 0, K)
        scores[lr] = ne.f1m(y[va], m.predict_proba(X[va]))
        del m
    lr_file.write_text(json.dumps({str(k): v for k, v in scores.items()}))
lr_scores = {float(k): v for k, v in json.loads(lr_file.read_text()).items()}
tcfg["lr"] = max(lr_scores, key=lr_scores.get)
best["transformer"] = tcfg
print("F1 val por lr:", lr_scores, "→ lr =", tcfg["lr"])
finals = pd.concat([finals, tracked("final_transformer", lambda: ne.final_runs(
    cfg, "transformer", tcfg, X, y, sp, K, cfg.nlp["seeds_transformer"], save_model=True))], ignore_index=True)
finals.to_csv(cfg.out_dir / "nlp" / "finals.csv", index=False)

# %% [markdown]
# ## 7.3 Resultados en test: media [IC 95 %] con bootstrap jerárquico (semillas × documentos)

# %%
def hier_boot(name, metric, n_boot=None, seed=0):
    seeds = sorted(finals[finals.model == name]["seed"].unique())
    P = {s: ne.load_probs(cfg, name, s) for s in seeds}
    yt = y[te]
    rng = np.random.default_rng(seed)
    n_boot = n_boot or (2000 if not cfg.is_smoke else 200)
    vals = []
    for _ in range(n_boot):
        s = seeds[rng.integers(len(seeds))]
        idx = rng.integers(0, len(yt), len(yt))
        vals.append(ne.nlp_metrics(yt[idx], P[s][idx], K)[metric])
    point = np.mean([ne.nlp_metrics(yt, P[s], K)[metric] for s in seeds])
    return point, *np.percentile(vals, [2.5, 97.5])


METRICS = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_micro", "f1_weighted", "mcc",
           "balanced_accuracy", "roc_auc_ovr", "pr_auc_ovr", "log_loss", "brier", "ece"]
rows = {}
for name in finals.model.unique():
    rows[NAMES[name]] = {m: st.fmt_ci(*hier_boot(name, m)) for m in METRICS}
    f = finals[finals.model == name]
    rows[NAMES[name]] |= {"latencia p50 (ms)": round(f["lat_p50_ms"].mean(), 2),
                          "latencia p95 (ms)": round(f["lat_p95_ms"].mean(), 2), "parámetros": f["n_params"].iloc[0],
                          "CO₂ entrenamiento (g)": round(1000 * (co2.get(f"final_{name}", {}).get("kg") or np.nan), 3),
                          "F1 test: DE entre semillas": round(f["test_f1"].std(), 4)}
res_tab = pd.DataFrame(rows).T
res_tab = res_tab.loc[res_tab["f1_macro"].str.split().str[0].astype(float).sort_values(ascending=False).index]
save_table(cfg, "nlp_resultados_test", res_tab)
res_tab[["accuracy", "precision_macro", "recall_macro", "f1_macro", "mcc", "roc_auc_ovr", "pr_auc_ovr", "ece",
         "latencia p50 (ms)", "parámetros", "CO₂ entrenamiento (g)", "F1 test: DE entre semillas"]]

# %% [markdown]
# ### McNemar exacto por pares (predicción por voto de las semillas) con corrección de Holm

# %%
def vote(name):
    seeds = sorted(finals[finals.model == name]["seed"].unique())
    return np.mean([ne.load_probs(cfg, name, s) for s in seeds], 0).argmax(1)


models_ = [m for m in finals.model.unique() if m != "majority"]
preds = {m: vote(m) for m in models_}
from itertools import combinations  # noqa: E402

mc = []
for a, b in combinations(models_, 2):
    mc.append({"A": NAMES[a], "B": NAMES[b], "acc A": (preds[a] == y[te]).mean(), "acc B": (preds[b] == y[te]).mean(),
               "p McNemar": st.mcnemar_exact(y[te], preds[a], preds[b])})
mc = pd.DataFrame(mc)
mc["p Holm"] = st.holm_adjust(mc["p McNemar"])
mc["significativa"] = mc["p Holm"] < cfg.alpha
save_table(cfg, "nlp_mcnemar", mc)
mc.sort_values("p Holm").round(4).head(15)

# %% [markdown]
# ## 7.4 Validación anidada (modelos ligeros): desempeño del procedimiento y sesgo optimista
# Para cada fold externo: HPO completo (mismo número de trials) en un split interno 85/15 agrupado, reentrenar con
# los mejores hiperparámetros y medir en el fold externo. El sesgo optimista = mejor F1 interno − F1 externo.

# %%
nest = pd.concat([tracked(f"nested_{n}", lambda n=n: ne.nested_cv(cfg, n, X, y, groups, sp, K))
                  for n in ["tfidf_lr", "nbsvm", "tfidf_xgb", "fasttext", "cnn1d", "bilstm"]], ignore_index=True)
nest["sesgo optimista"] = nest["inner_best_val_f1"] - nest["outer_f1"]
nest.to_csv(cfg.out_dir / "nlp" / "nested.csv", index=False)
ns = nest.groupby("model").agg(F1_externo_medio=("outer_f1", "mean"), DE_entre_folds=("outer_f1", "std"),
                               F1_interno_medio=("inner_best_val_f1", "mean"), sesgo=("sesgo optimista", "mean"))
ns.index = ns.index.map(NAMES)
save_table(cfg, "nlp_anidada", ns)
wide = nest.pivot(index="fold", columns="model", values="outer_f1").rename(columns=NAMES)
cmp = st.compare_models(wide, alpha=cfg.alpha, delta=0.02) if len(wide) >= 2 and wide.shape[1] >= 3 else None
if cmp:
    print(f"Friedman sobre {len(wide)} folds: χ² = {cmp['friedman_stat']:.2f}, p = {cmp['friedman_p']:.3g}")
    n_tr = int(nest["n_train"].mean())
    n_te = int(nest["n_test"].mean())
    nb = []
    for a, b in combinations(wide.columns, 2):
        t_, p_ = st.corrected_resampled_ttest(wide[a] - wide[b], n_tr, n_te)
        nb.append({"A": a, "B": b, "ΔF1 medio": (wide[a] - wide[b]).mean(), "t corregida": t_, "p": p_})
    nb = pd.DataFrame(nb)
    nb["p Holm"] = st.holm_adjust(nb["p"])
    save_table(cfg, "nlp_nadeau_bengio", nb)
    fig, ax = plt.subplots(figsize=(8, 2.8))
    st.plot_cd_diagram(cmp["mean_ranks"], cmp["n"], cmp["pairwise"], ax=ax, title="F1 macro por fold externo")
    savefig(cfg, "07_cd_anidada")
    plt.show()
ns.round(4)

# %% [markdown]
# ## 7.5 H4: DistilBERT vs TF-IDF + XGBoost

# %%
ft = finals[finals.model == "transformer"].set_index("seed")["test_f1"]
fx = finals[finals.model == "tfidf_xgb"].set_index("seed")["test_f1"]
common = ft.index.intersection(fx.index)
d_seed = (ft.loc[common] - fx.loc[common]) if len(common) else pd.Series(dtype=float)
p_mc = st.mcnemar_exact(y[te], preds["transformer"], preds["tfidf_xgb"])
row = mc[((mc.A == NAMES["transformer"]) & (mc.B == NAMES["tfidf_xgb"])) |
         ((mc.B == NAMES["transformer"]) & (mc.A == NAMES["tfidf_xgb"]))]
p_holm = float(row["p Holm"].iloc[0]) if len(row) else np.nan
dF1 = float(ft.mean() - fx.mean())
boot = st.bootstrap_ci(d_seed.values, n_boot=cfg.n_boot) if len(d_seed) > 1 else (dF1, np.nan, np.nan)
h4 = {"ΔF1 (DistilBERT − XGBoost), media de semillas": dF1, "IC 95 % sobre semillas": boot[1:],
      "p McNemar (voto)": p_mc, "p Holm": p_holm, "δ": 0.02,
      "H0 (sin diferencia relevante)": "rechazada" if (p_holm < cfg.alpha and abs(dF1) >= 0.02) else "no rechazada"}
save_json(cfg, "nlp_H4", h4)
h4

# %% [markdown]
# ## 7.6 Sensibilidad del BiLSTM (rejilla de 576 configuraciones, búsqueda bayesiana)

# %%
hb = json.loads((cfg.out_dir / "nlp" / "hpo" / "main_bilstm.json").read_text())
tb = pd.DataFrame(hb["trials"])
tb = tb[tb["state"] == "COMPLETE"] if "state" in tb else tb
params_ = [c for c in tb.columns if c.startswith("params_")]
fig, ax = plt.subplots(2, 5, figsize=(20, 6))
for a, p_ in zip(ax.ravel(), params_):
    tb.groupby(p_)["value"].agg(["mean", "count"])["mean"].plot.bar(ax=a, rot=0, color=PALETTE[0])
    a.set(title=p_.replace("params_", ""), ylabel="F1 val")
ax.ravel()[-1].barh(list(hb["importance"].keys()), list(hb["importance"].values()), color=PALETTE[1])
ax.ravel()[-1].set_title("importancia fANOVA")
plt.tight_layout()
savefig(cfg, "07_bilstm_sensibilidad")
plt.show()
if {"params_sequence_length", "params_pooling"} <= set(tb.columns):
    inter = tb.pivot_table(index="params_sequence_length", columns="params_pooling", values="value", aggfunc="mean")
    print("Interacción longitud de secuencia × pooling (F1 val medio):")
    display(inter.round(3))  # noqa: F821
print("Importancias:", {k: round(v, 3) for k, v in hb["importance"].items()})

# %% [markdown]
# ## 7.7 Ablación de preprocesamiento por familia (mínimo / estándar / agresivo)
# Clásico (TF-IDF + LR), neuronal (BiLSTM, mejores hiperparámetros) y transformer; 3 semillas; diferencias
# pareadas con bootstrap sobre documentos de test.

# %%
prep_rows = []
fam = {"tfidf_lr": best["tfidf_lr"], "bilstm": best["bilstm"]}
seeds3 = SEEDS[:3]
for name, p in fam.items():
    for prep in ["minimal", "standard", "aggressive"]:
        for s in seeds3:
            f = cfg.out_dir / "nlp" / "prep" / f"{name}_{prep}_s{s}.npy"
            if not f.exists():
                f.parent.mkdir(parents=True, exist_ok=True)
                m = fit_model(name, X[tr], y[tr], X[va], y[va], {**p, "prep": prep}, s, K)
                np.save(f, m.predict_proba(X[te]))
            prep_rows.append({"familia": NAMES[name], "preprocesado": prep, "seed": s,
                              "F1": ne.f1m(y[te], np.load(f))})
for prep in ["minimal", "standard", "aggressive"]:
    for s in cfg.nlp["seeds_transformer"][:3 if not cfg.is_smoke else 1]:
        f = cfg.out_dir / "nlp" / "prep" / f"transformer_{prep}_s{s}.npy"
        if not f.exists():
            from dl03 import nlp_data as nd_
            Xp = np.array([nd_.CLEANERS[prep](t) for t in X])
            m = ne.fit_transformer(Xp[tr], y[tr], Xp[va], y[va], tcfg, s, K)
            np.save(f, m.predict_proba(Xp[te]))
            del m
        prep_rows.append({"familia": NAMES["transformer"], "preprocesado": prep, "seed": s,
                          "F1": ne.f1m(y[te], np.load(f))})
prep_df = pd.DataFrame(prep_rows)
prep_tab = prep_df.pivot_table(index="familia", columns="preprocesado", values="F1", aggfunc=["mean", "std"])
save_table(cfg, "nlp_preprocesado", prep_tab)
prep_tab.round(4)

# %% [markdown]
# ## 7.8 Embeddings: CBOW vs skip-gram vs aleatorios; congelados vs ajustados; evaluación intrínseca

# %%
emb_rows = []
pb = {k: v for k, v in best["bilstm"].items()}
for w2v in ["skipgram", "cbow", "none"]:
    for trainable in [True, False]:
        if w2v == "none" and not trainable:
            continue
        for s in seeds3:
            m = fit_model("bilstm", X[tr], y[tr], X[va], y[va], {**pb, "w2v": w2v, "emb_trainable": trainable}, s, K)
            emb_rows.append({"embeddings": w2v, "ajustados": trainable, "seed": s,
                             "F1 test": ne.f1m(y[te], m.predict_proba(X[te])),
                             "cobertura vocabulario W2V": getattr(m, "w2v_coverage", np.nan)})
emb = pd.DataFrame(emb_rows).groupby(["embeddings", "ajustados"])[["F1 test", "cobertura vocabulario W2V"]] \
    .agg(["mean", "std"])
save_table(cfg, "nlp_embeddings", emb)
from dl03 import nlp_data as nd  # noqa: E402
from dl03.nlp_models import train_word2vec  # noqa: E402

toks = [t.split() for t in df.text.values[tr]]
toks = [nd.clean_standard(" ".join(t)).split() for t in toks]
w_sg = train_word2vec(toks, 100, 1, cfg.seed)
w_cb = train_word2vec(toks, 100, 0, cfg.seed)
probe = [w for w in ["python", "nurse", "chef", "loan", "teacher", "design", "sales", "pilot"] if w in w_sg.wv]
neigh = pd.DataFrame({w: {"skip-gram": ", ".join(x for x, _ in w_sg.wv.most_similar(w, topn=5)),
                          "CBOW": ", ".join(x for x, _ in w_cb.wv.most_similar(w, topn=5))} for w in probe}).T
display(emb.round(4))  # noqa: F821
neigh

# %% [markdown]
# Manejo de OOV: fastText con sub-palabras (minn/maxn > 0) frente a Word2Vec con `<UNK>`. Efecto cuantificado:
# F1 del mejor fastText y del BiLSTM en los documentos de test con tasa de OOV por encima vs por debajo de la
# mediana.

# %%
vocab_tr = {w for t in toks for w in t}
oov_rate = np.array([np.mean([w not in vocab_tr for w in nd.clean_standard(t).split()] or [0]) for t in X[te]])
hi = np.zeros(len(te), bool)
hi[np.argsort(oov_rate, kind="stable")[len(te) // 2:]] = True   # mitad superior por rango de OOV
oov_tab = {}
for name in ["fasttext", "bilstm", "transformer", "tfidf_lr"]:
    P = np.mean([ne.load_probs(cfg, name, s) for s in sorted(finals[finals.model == name].seed)], 0)
    oov_tab[NAMES[name]] = {"F1 OOV alto": ne.f1m(y[te][hi], P[hi]), "F1 OOV bajo": ne.f1m(y[te][~hi], P[~hi])}
oov_tab = pd.DataFrame(oov_tab).T.assign(diferencia=lambda d: d["F1 OOV alto"] - d["F1 OOV bajo"])
print("subpalabras en el mejor fastText:", {k: best["fasttext"].get(k) for k in ("minn", "maxn")})
save_table(cfg, "nlp_oov", oov_tab)
oov_tab.round(4)

# %% [markdown]
# ## 7.9 Curvas de aprendizaje por familia (10–100 % del train)

# %%
from sklearn.model_selection import StratifiedShuffleSplit  # noqa: E402

lc = []
for frac in cfg.nlp["data_fractions"]:
    if frac < 1:
        sub = tr[next(StratifiedShuffleSplit(1, train_size=frac, random_state=0).split(tr, y[tr]))[0]]
    else:
        sub = tr
    for name in ["tfidf_lr", "tfidf_xgb", "fasttext", "cnn1d", "bilstm", "transformer"]:
        f = cfg.out_dir / "nlp" / "lc" / f"{name}_{frac}.json"
        if not f.exists():
            f.parent.mkdir(parents=True, exist_ok=True)
            if name == "transformer":
                m = ne.fit_transformer(X[sub], y[sub], X[va], y[va], tcfg, 0, K)
            else:
                m = fit_model(name, X[sub], y[sub], X[va], y[va], best[name], 0, K)
            f.write_text(json.dumps({"f1": ne.f1m(y[te], m.predict_proba(X[te])), "n": int(len(sub))}))
        lc.append({"modelo": NAMES[name], "fracción": frac, **json.loads(f.read_text())})
lc = pd.DataFrame(lc)
fig, ax = plt.subplots(figsize=(7, 4))
for k, (m, g) in enumerate(lc.groupby("modelo")):
    ax.plot(g["n"], g["f1"], "o-", color=PALETTE[k % 8], label=m)
ax.set(xscale="log", xlabel="documentos de entrenamiento", ylabel="F1 macro (test)", title="Curvas de aprendizaje")
ax.legend(fontsize=8)
savefig(cfg, "07_curvas_aprendizaje")
plt.show()
save_table(cfg, "nlp_curvas_aprendizaje", lc)

# %% [markdown]
# ## 7.10 Matriz de confusión normalizada del mejor modelo

# %%
from sklearn.metrics import confusion_matrix  # noqa: E402

best_name = finals.groupby("model")["test_f1"].mean().idxmax()
cm = confusion_matrix(y[te], preds[best_name], normalize="true", labels=range(K))
fig, ax = plt.subplots(figsize=(10, 8.5))
ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(K), classes, rotation=90, fontsize=7)
ax.set_yticks(range(K), classes, fontsize=7)
for i in range(K):
    for j in range(K):
        if cm[i, j] >= 0.1:
            ax.text(j, i, f"{cm[i, j]:.1f}", ha="center", va="center", fontsize=6,
                    color="white" if cm[i, j] > 0.5 else "black")
ax.set(title=f"Matriz de confusión normalizada (fila = verdadera) · {NAMES[best_name]}", xlabel="predicha")
savefig(cfg, "07_confusion")
plt.show()
save_json(cfg, "nlp_co2_cap07", co2)
print("mejor modelo por F1 medio de test:", NAMES[best_name])
