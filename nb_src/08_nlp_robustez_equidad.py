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
# # 8. Texto: robustez, atajos, calibración, explicabilidad y equidad
#
# Se evalúan los modelos finales del capítulo 7 (semilla 0 reentrenada aquí cuando hace falta el objeto del
# modelo). Nada se ajusta en test: la temperatura y el umbral conforme se calibran en validación.

# %%
import json
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dl03 import calibration as cal
from dl03 import nlp_data as nd
from dl03 import nlp_experiments as ne
from dl03 import stats as st
from dl03.nlp_models import fit_model
from dl03.viz import PALETTE, save_json, save_table, savefig, setup

cfg = setup(seed_offset=8)
df, y, classes, groups, sp = ne.prepare(cfg)
X, K = df["text"].values, len(classes)
tr, va, te = map(np.array, (sp["train"], sp["val"], sp["test"]))
best = json.loads((cfg.out_dir / "tables" / "nlp_best_params.json").read_text())
lr_scores = json.loads((cfg.out_dir / "nlp" / "hpo" / "transformer_lr.json").read_text())
tcfg = dict(cfg.nlp["transformer"], lr=float(max(lr_scores, key=lr_scores.get)))
NAMES = {"tfidf_lr": "TF-IDF + LR", "tfidf_xgb": "TF-IDF + XGBoost", "fasttext": "fastText", "cnn1d": "CNN-1D",
         "bilstm": "Word2Vec + BiLSTM", "transformer": "DistilBERT", "nbsvm": "NB-SVM"}
KEY = ["tfidf_lr", "tfidf_xgb", "fasttext", "bilstm", "transformer"]
models = {}
for name in KEY:
    if name == "transformer":
        models[name] = ne.fit_transformer(X[tr], y[tr], X[va], y[va], tcfg, 0, K)
    else:
        models[name] = fit_model(name, X[tr], y[tr], X[va], y[va], best[name], 0, K)
base_P = {n: m.predict_proba(X[te]) for n, m in models.items()}
base_f1 = {n: ne.f1m(y[te], P) for n, P in base_P.items()}
base_f1

# %% [markdown]
# ## 8.1 Pruebas de invarianza y direccionalidad (CheckList; Ribeiro et al., 2020)
# Invarianza (la categoría no debería cambiar): mayúsculas, errores tipográficos (5 % de palabras), sinónimos,
# cambio de nombres propios y paráfrasis por retro-traducción (inglés → español → inglés, MarianMT, sobre los 400
# primeros caracteres; omitida en smoke). Direccionalidad: añadir una frase fuertemente asociada a otra categoría
# debería **aumentar** la probabilidad de esa categoría. Se reporta la tasa de cambio de predicción y la caída de F1.

# %%
rng = np.random.default_rng(cfg.seed)
SYN = {"managed": "supervised", "developed": "built", "experience": "background", "team": "group",
       "responsible": "accountable", "customer": "client", "skills": "abilities", "project": "initiative"}
NAMES_SWAP = [("John", "Maria"), ("Michael", "Aisha"), ("David", "Mei"), ("James", "Fatima")]


def typos(t, rate=0.05):
    w = t.split()
    for i in rng.choice(len(w), max(1, int(rate * len(w))), replace=False) if w else []:
        if len(w[i]) > 3:
            j = rng.integers(1, len(w[i]) - 1)
            w[i] = w[i][:j] + w[i][j + 1] + w[i][j] + w[i][j + 2:]
    return " ".join(w)


def synonyms(t):
    return re.sub(r"\b(" + "|".join(SYN) + r")\b", lambda m: SYN[m.group(1)], t, flags=re.I)


def swap_names(t):
    for a, b in NAMES_SWAP:
        t = re.sub(rf"\b{a}\b", b, t)
    return "John Smith " + t


PERT = {"MAYÚSCULAS": str.upper, "minúsculas": str.lower, "errores tipográficos": typos, "sinónimos": synonyms,
        "nombres propios": swap_names}
if not cfg.is_smoke:
    from transformers import MarianMTModel, MarianTokenizer

    def _mt(name):
        return MarianTokenizer.from_pretrained(name), MarianMTModel.from_pretrained(name)
    (tk1, m1), (tk2, m2) = _mt("Helsinki-NLP/opus-mt-en-es"), _mt("Helsinki-NLP/opus-mt-es-en")

    def backtrans(t):
        head, tail = t[:400], t[400:]
        es = tk1.decode(m1.generate(**tk1(head, return_tensors="pt", truncation=True))[0], skip_special_tokens=True)
        en = tk2.decode(m2.generate(**tk2(es, return_tensors="pt", truncation=True))[0], skip_special_tokens=True)
        return en + tail
    PERT["paráfrasis (retro-traducción)"] = backtrans
sub = te[: min(len(te), 150 if not cfg.is_smoke else 30)]
inv = []
for pname, f in PERT.items():
    Xp = np.array([f(t) for t in X[sub]])
    for n, m in models.items():
        P0, P1 = m.predict_proba(X[sub]), m.predict_proba(Xp)
        inv.append({"perturbación": pname, "modelo": NAMES[n], "cambio de predicción": float((P0.argmax(1) !=
                                                                                            P1.argmax(1)).mean()),
                    "ΔF1": ne.f1m(y[sub], P1) - ne.f1m(y[sub], P0)})
inv = pd.DataFrame(inv)
save_table(cfg, "nlp_checklist_invarianza", inv)
inv.pivot(index="perturbación", columns="modelo", values="cambio de predicción").round(3)

# %%
# Direccionalidad: se añade una frase típica de otra categoría y se mide si sube su probabilidad.
target = {c: f"Experienced {c.lower().replace('-', ' ')} professional with strong {c.lower().replace('-', ' ')} skills."
          for c in classes}
dirs = []
for n, m in models.items():
    ok = []
    for i in sub[:60]:
        tgt = (y[i] + 1 + rng.integers(K - 1)) % K
        p0 = m.predict_proba([X[i]])[0, tgt]
        p1 = m.predict_proba([X[i] + " " + target[classes[tgt]]])[0, tgt]
        ok.append(p1 >= p0)
    dirs.append({"modelo": NAMES[n], "fracción en la dirección esperada": float(np.mean(ok))})
pd.DataFrame(dirs)

# %% [markdown]
# ## 8.2 Ablación de atajos: enmascarar el título del cargo
# Si la métrica colapsa al quitar la primera línea, el modelo aprendió el atajo del título, no el contenido.

# %%
Xm = np.array([nd.mask_title(t) for t in X[te]])
sc = []
for n, m in models.items():
    Pm = m.predict_proba(Xm)
    d = []
    rng2 = np.random.default_rng(0)
    for _ in range(1000 if not cfg.is_smoke else 100):
        idx = rng2.integers(0, len(te), len(te))
        d.append(ne.f1m(y[te][idx], base_P[n][idx]) - ne.f1m(y[te][idx], Pm[idx]))
    sc.append({"modelo": NAMES[n], "F1 con título": base_f1[n], "F1 sin título": ne.f1m(y[te], Pm),
               "caída [IC 95 %]": st.fmt_ci(np.mean(d), *np.percentile(d, [2.5, 97.5]))})
sc = pd.DataFrame(sc)
save_table(cfg, "nlp_atajo_titulo", sc)
sc

# %% [markdown]
# ## 8.3 Ruido de etiquetas: ¿cambia el ranking al quitar los ejemplos sospechosos?
# Se reentrena sin los documentos señalados por confident learning en train/val (capítulo 6) y se compara el
# ranking en test. Conjunto fuera de distribución: no existe otra fuente o periodo en el dataset (limitación
# declarada en el pre-registro).

# %%
bad = set(np.load(cfg.out_dir / "nlp" / "label_issues.npy").tolist())
tr_c = np.array([i for i in tr if i not in bad])
va_c = np.array([i for i in va if i not in bad])
clean = {}
for n in KEY:
    m = ne.fit_transformer(X[tr_c], y[tr_c], X[va_c], y[va_c], tcfg, 0, K) if n == "transformer" else \
        fit_model(n, X[tr_c], y[tr_c], X[va_c], y[va_c], best[n], 0, K)
    clean[NAMES[n]] = {"F1 original": base_f1[n], "F1 sin sospechosos": ne.f1m(y[te], m.predict_proba(X[te]))}
clean = pd.DataFrame(clean).T
clean["rango original"] = clean["F1 original"].rank(ascending=False)
clean["rango limpio"] = clean["F1 sin sospechosos"].rank(ascending=False)
print(f"{len(bad)} documentos retirados de train/val")
save_table(cfg, "nlp_ruido_ranking", clean)
clean.round(4)

# %% [markdown]
# ## 8.4 Calibración, temperature scaling (H5), predicción conforme y clasificación selectiva

# %%
calib = {}
for n, m in models.items():
    Lva = m.logits(X[va]) if hasattr(m, "logits") else np.log(np.clip(m.predict_proba(X[va]), 1e-9, 1))
    Lte = m.logits(X[te]) if hasattr(m, "logits") else np.log(np.clip(base_P[n], 1e-9, 1))
    T = cal.fit_temperature(Lva, y[va])
    P0, PT = cal.softmax(Lte), cal.softmax(Lte / T)
    calib[n] = dict(T=T, P0=P0, PT=PT)
rows = []
for n, c in calib.items():
    d = []
    rng3 = np.random.default_rng(1)
    for _ in range(2000 if not cfg.is_smoke else 200):
        idx = rng3.integers(0, len(te), len(te))
        d.append(cal.expected_calibration_error(c["P0"][idx], y[te][idx]) -
                 cal.expected_calibration_error(c["PT"][idx], y[te][idx]))
    rows.append({"modelo": NAMES[n], "T": c["T"], "ECE antes": cal.expected_calibration_error(c["P0"], y[te]),
                 "ECE después": cal.expected_calibration_error(c["PT"], y[te]),
                 "reducción de ECE [IC 95 %]": st.fmt_ci(np.mean(d), *np.percentile(d, [2.5, 97.5])),
                 "Brier antes": cal.brier_score(c["P0"], y[te]), "Brier después": cal.brier_score(c["PT"], y[te]),
                 "argmax idéntico": bool((c["P0"].argmax(1) == c["PT"].argmax(1)).all())})
caltab = pd.DataFrame(rows).set_index("modelo")
save_table(cfg, "nlp_calibracion", caltab)
t_red = [r for r in rows if r["modelo"] == "DistilBERT"][0]
lr_ece = caltab.loc["TF-IDF + LR", "ECE antes"]
lo = float(t_red["reducción de ECE [IC 95 %]"].split("[")[1].split(",")[0])
h5 = {"ECE DistilBERT": t_red["ECE antes"], "ECE TF-IDF + LR": lr_ece,
      "reducción tras T [IC 95 %]": t_red["reducción de ECE [IC 95 %]"], "argmax idéntico": t_red["argmax idéntico"],
      "H5": "no refutada" if (t_red["ECE antes"] > lr_ece and lo > 0 and t_red["argmax idéntico"]) else "refutada"}
save_json(cfg, "nlp_H5", h5)
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for k, (n, c) in enumerate(calib.items()):
    rb = cal.reliability_bins(c["P0"], y[te])
    ax[0].plot(rb[:, 2], rb[:, 1], "o-", color=PALETTE[k], label=NAMES[n], ms=3)
    rb = cal.reliability_bins(c["PT"], y[te])
    ax[1].plot(rb[:, 2], rb[:, 1], "o-", color=PALETTE[k], ms=3)
for a, t in zip(ax, ["antes", "después de temperature scaling"]):
    a.plot([0, 1], [0, 1], "--", c="gray")
    a.set(title=f"Confiabilidad ({t})", xlabel="confianza", ylabel="exactitud")
ax[0].legend(fontsize=7)
savefig(cfg, "08_confiabilidad")
plt.show()
display(caltab.round(4))  # noqa: F821
h5

# %%
# Predicción conforme (LAC, α = 0.1) calibrada en validación; cobertura empírica y tamaño medio del conjunto.
conf_rows = []
for n, m in models.items():
    Pva = cal.softmax((m.logits(X[va]) if hasattr(m, "logits") else np.log(np.clip(m.predict_proba(X[va]), 1e-9, 1)))
                      / calib[n]["T"])
    sets, q = cal.conformal_lac(Pva, y[va], calib[n]["PT"], alpha=0.1)
    conf_rows.append({"modelo": NAMES[n], "cobertura objetivo": 0.9,
                      "cobertura empírica": float(sets[np.arange(len(te)), y[te]].mean()),
                      "tamaño medio del conjunto": float(sets.sum(1).mean()), "q̂": q})
conf = pd.DataFrame(conf_rows).set_index("modelo")
save_table(cfg, "nlp_conforme", conf)
# Clasificación selectiva: ¿qué fracción derivar a revisión humana para lograr 95 % de exactitud?
fig, ax = plt.subplots(figsize=(6.5, 4))
sel = {}
for k, (n, c) in enumerate(calib.items()):
    cov, risk, aurc = cal.risk_coverage(c["PT"].max(1), c["PT"].argmax(1) == y[te])
    ax.plot(cov, risk, color=PALETTE[k], label=f"{NAMES[n]} (AURC {aurc:.3f})")
    ok = cov[risk <= 0.05]
    sel[NAMES[n]] = {"cobertura con riesgo ≤ 5 %": float(ok.max()) if ok.size else 0.0,
                     "a revisión humana": 1 - (float(ok.max()) if ok.size else 0.0)}
ax.set(xlabel="cobertura", ylabel="riesgo (1 − exactitud)", title="Riesgo–cobertura (tras temperature scaling)")
ax.legend(fontsize=7)
savefig(cfg, "08_riesgo_cobertura")
plt.show()
save_json(cfg, "nlp_selectiva", sel)
display(conf.round(3))  # noqa: F821
pd.DataFrame(sel).T.round(3)

# %% [markdown]
# ## 8.5 Explicabilidad: SHAP (XGBoost), gradientes integrados (DistilBERT), LIME como contraste y fidelidad
# Fidelidad por *deletion*: se eliminan las k palabras más importantes según cada método y se mide la caída de la
# probabilidad de la clase predicha; un método fiel produce caídas mayores que eliminar k palabras al azar.

# %%
import shap  # noqa: E402

xgb = models["tfidf_xgb"]
Xv = xgb.vec.transform([nd.CLEANERS[xgb.prep](t) for t in X[te[:40]]])
expl = shap.TreeExplainer(xgb.clf)
sv = expl.shap_values(Xv)
sv = np.stack(sv, -1) if isinstance(sv, list) else sv          # (n, features, K)
feat = xgb.vec.get_feature_names_out()
imp = pd.Series(np.abs(sv).mean((0, 2)), feat).nlargest(20)
fig, ax = plt.subplots(figsize=(6, 5))
imp[::-1].plot.barh(ax=ax, color=PALETTE[0])
ax.set_title("SHAP medio |φ| · TF-IDF + XGBoost (40 docs de test)")
savefig(cfg, "08_shap")
plt.show()

# %%
import torch  # noqa: E402

tm = models["transformer"]


def integrated_gradients(text, steps=24):
    """Gradientes integrados sobre los embeddings de palabra (línea base = embedding cero)."""
    enc = tm.encode([text])
    ids = torch.as_tensor(enc["input_ids"], device=next(tm.model.parameters()).device)
    am = torch.as_tensor(enc["attention_mask"], device=ids.device)
    emb_layer = tm.model.get_input_embeddings()
    e = emb_layer(ids).detach()
    with torch.no_grad():
        k = int(tm.model(input_ids=ids, attention_mask=am).logits.argmax(-1))
    total = torch.zeros_like(e)
    for a in torch.linspace(0, 1, steps):
        x = (a * e).requires_grad_(True)
        out = tm.model(inputs_embeds=x, attention_mask=am).logits[0, k]
        total += torch.autograd.grad(out, x)[0]
    attr = (e * total / steps).sum(-1)[0].detach().cpu().numpy()
    toks = tm.tok.convert_ids_to_tokens(enc["input_ids"][0])
    return [(t, a) for t, a, m_ in zip(toks, attr, enc["attention_mask"][0]) if m_ and t not in ("[CLS]", "[SEP]")], k


def word_scores_ig(text):
    toks, _ = integrated_gradients(text)
    scores, cur, val = {}, "", 0.0
    for t, a in toks:
        if t.startswith("##"):
            cur += t[2:]
            val += a
        else:
            if cur:
                scores[cur] = scores.get(cur, 0) + val
            cur, val = t, a
    if cur:
        scores[cur] = scores.get(cur, 0) + val
    return scores


from lime.lime_text import LimeTextExplainer  # noqa: E402

lime = LimeTextExplainer(class_names=classes, random_state=cfg.seed)


def deletion_drop(model, text, words, k=5):
    p = model.predict_proba([text])[0]
    c = int(p.argmax())
    rm = set(w.lower() for w in words[:k])
    t2 = " ".join(w for w in text.split() if re.sub(r"\W", "", w.lower()) not in rm)
    return float(p[c] - model.predict_proba([t2])[0, c])


fid = []
for i in te[: 15 if not cfg.is_smoke else 4]:
    t = nd.clean_minimal(X[i])[:2500]
    ig = word_scores_ig(t)
    ig_words = [w for w, _ in sorted(ig.items(), key=lambda kv: -kv[1])]
    le = lime.explain_instance(t, tm.predict_proba, num_features=10, top_labels=1,
                               num_samples=300 if not cfg.is_smoke else 50)
    lime_words = [w for w, _ in sorted(le.as_list(le.top_labels[0]), key=lambda kv: -kv[1])]
    rand_words = list(np.random.default_rng(int(i)).permutation(list({w.lower() for w in t.split()})))
    fid.append({"IG": deletion_drop(tm, t, ig_words), "LIME": deletion_drop(tm, t, lime_words),
                "azar": deletion_drop(tm, t, rand_words)})
fid = pd.DataFrame(fid)
save_table(cfg, "nlp_fidelidad", fid.describe())
print("caída media de la probabilidad al borrar las 5 palabras más importantes:")
fid.mean().round(4)

# %% [markdown]
# ## 8.6 Análisis de errores: hoja de anotación manual (≥ 50 errores por modelo clave)
# Categorías definidas **a priori**: ambigüedad entre categorías afines, etiqueta dudosa, perfil mixto/multirol,
# documento muy corto o truncado, vocabulario de dominio raro, título engañoso. Se sugiere una categoría automática
# (heurística) como punto de partida; la anotación final la hace el autor (un solo anotador: sin κ).

# %%
AFFINE = [{"HR", "PUBLIC-RELATIONS"}, {"FINANCE", "BANKING", "ACCOUNTANT"}, {"SALES", "BUSINESS-DEVELOPMENT"},
          {"DESIGNER", "ARTS", "DIGITAL-MEDIA", "APPAREL"}, {"ENGINEERING", "INFORMATION-TECHNOLOGY", "CONSTRUCTION"}]
bad_set = set(np.load(cfg.out_dir / "nlp" / "label_issues.npy").tolist())
err_rows = []
for n in ["transformer", "tfidf_xgb", "bilstm"]:
    pred = base_P[n].argmax(1)
    wrong = np.where(pred != y[te])[0][:60]
    for j in wrong:
        i = te[j]
        a, b = classes[y[i]], classes[pred[j]]
        if any({a, b} <= s for s in AFFINE):
            sug = "ambigüedad entre categorías afines"
        elif len(X[i].split()) < 120:
            sug = "documento muy corto"
        elif b.lower().split("-")[0] in nd.first_line_title(X[i]).lower():
            sug = "título engañoso"
        else:
            sug = "perfil mixto / otra"
        err_rows.append({"modelo": NAMES[n], "ID": df.ID.iloc[i], "verdadera": a, "predicha": b,
                         "título": nd.first_line_title(X[i])[:70], "sugerencia automática": sug,
                         "categoría (revisión manual)": "", "comentario": ""})
errs = pd.DataFrame(err_rows)
errs.to_csv(cfg.out_dir / "tables" / "nlp_errores_para_anotar.csv", index=False)
errs.groupby(["modelo", "sugerencia automática"]).size().unstack(fill_value=0)

# %% [markdown]
# ## 8.7 Equidad y riesgo (contexto de reclutamiento)
# El dataset no trae atributos protegidos; los textos están parcialmente anonimizados. Por eso:
# * **Pruebas contrafactuales**: se antepone a cada documento una frase con un marcador de género, edad u origen y
#   se mide el cambio de la distribución predicha (variación total) y del argmax entre las versiones.
# * **Métricas por subgrupo con proxies**: presencia de pronombres masculinos/femeninos (proxy débil y declarado),
#   con tasa de "selección" definida como ser asignado a la categoría verdadera (recall) y razón de impacto
#   dispar (regla 4/5), con IC bootstrap.

# %%
CF = {"género": ("He is a dedicated professional.", "She is a dedicated professional."),
      "edad": ("Recent graduate, 23 years old.", "Senior professional, 58 years old."),
      "origen": ("Born and raised in Ohio.", "Born and raised in Nigeria."),
      "nombre": ("Name: Greg Baker.", "Name: Lakisha Washington.")}
cf_rows = []
for n, m in models.items():
    for attr, (a_, b_) in CF.items():
        Pa = m.predict_proba([a_ + " " + t for t in X[sub]])
        Pb = m.predict_proba([b_ + " " + t for t in X[sub]])
        cf_rows.append({"modelo": NAMES[n], "atributo": attr, "cambio de argmax": float((Pa.argmax(1) !=
                                                                                         Pb.argmax(1)).mean()),
                        "variación total media": float(0.5 * np.abs(Pa - Pb).sum(1).mean())})
cf = pd.DataFrame(cf_rows)
save_table(cfg, "nlp_contrafactual", cf)
cf.pivot(index="atributo", columns="modelo", values="cambio de argmax").round(4)

# %%
male = df.text.str.contains(r"\b(?:he|him|his|mr)\b", case=False, regex=True).values
female = df.text.str.contains(r"\b(?:she|her|hers|mrs|ms)\b", case=False, regex=True).values
grp = np.where(male & ~female, "proxy masculino", np.where(female & ~male, "proxy femenino", "sin proxy/ambos"))
fair = []
for n in KEY:
    hit = base_P[n].argmax(1) == y[te]
    g_te = grp[te]
    r_ = {}
    for g in ["proxy masculino", "proxy femenino", "sin proxy/ambos"]:
        msk = g_te == g
        r_[g] = (hit[msk].mean() if msk.sum() else np.nan, int(msk.sum()))
    rm, rf = r_["proxy masculino"][0], r_["proxy femenino"][0]
    di = (min(rm, rf) / max(rm, rf)) if np.isfinite(rm) and np.isfinite(rf) and max(rm, rf) > 0 else np.nan
    boots = []
    rng4 = np.random.default_rng(2)
    im_, if_ = np.where(g_te == "proxy masculino")[0], np.where(g_te == "proxy femenino")[0]
    if len(im_) > 3 and len(if_) > 3:
        for _ in range(2000):
            a = hit[rng4.choice(im_, len(im_))].mean()
            b = hit[rng4.choice(if_, len(if_))].mean()
            boots.append(a - b)
    fair.append({"modelo": NAMES[n], "recall proxy masc. (n)": f"{rm:.3f} ({r_['proxy masculino'][1]})",
                 "recall proxy fem. (n)": f"{rf:.3f} ({r_['proxy femenino'][1]})",
                 "diferencia [IC 95 %]": st.fmt_ci(np.mean(boots), *np.percentile(boots, [2.5, 97.5]))
                 if boots else "n insuficiente", "razón de impacto (regla 4/5)": di})
fair = pd.DataFrame(fair)
save_table(cfg, "nlp_equidad_subgrupos", fair)
fair

# %% [markdown]
# **Tensiones y marco regulatorio.** Paridad demográfica, igualdad de oportunidades y probabilidades igualadas no
# pueden satisfacerse a la vez cuando las tasas base difieren entre grupos (Kleinberg et al., 2016;
# Chouldechova, 2017). En la Unión Europea, el Reglamento de IA (Reglamento (UE) 2024/1689, anexo III, punto 4)
# clasifica como de **alto riesgo** los sistemas usados para seleccionar o filtrar candidatos, con obligaciones de
# gestión de riesgos, calidad de datos, transparencia y supervisión humana; en Nueva York, la Local Law 144 exige
# auditorías de sesgo de las herramientas automatizadas de decisión de empleo; en Colombia, la Ley 1581 de 2012
# regula el tratamiento de los datos personales que contienen estas hojas de vida.
#
# **¿Debería usarse automáticamente este modelo para decidir sobre personas?** No. Con las pruebas de este
# capítulo, como mucho como **ayuda de enrutamiento** (sugerir la categoría de cargo) con: revisión humana de los
# casos de baja confianza (8.4), auditoría periódica de subgrupos con atributos reales recogidos con
# consentimiento, registro de decisiones, explicación al candidato y prohibición de usarlo para descartar
# candidatos.

# %% [markdown]
# ## 8.8 Eficiencia: cuantización dinámica del transformer (CPU)

# %%
import time  # noqa: E402

cpu_model = tm.model.to("cpu").float().eval()
qmodel = torch.quantization.quantize_dynamic(cpu_model, {torch.nn.Linear}, dtype=torch.qint8)


def bench(mod, n=20):
    enc = tm.encode(list(X[te[:n]]))
    ids, am = torch.as_tensor(enc["input_ids"]), torch.as_tensor(enc["attention_mask"])
    t, out = [], []
    with torch.no_grad():
        for k in range(len(ids)):
            t0 = time.perf_counter()
            out.append(mod(input_ids=ids[k:k + 1], attention_mask=am[k:k + 1]).logits.numpy())
            t.append(1000 * (time.perf_counter() - t0))
    return np.concatenate(out), np.percentile(t, [50, 95])


def size_mb(mod):
    import io
    buf = io.BytesIO()
    torch.save(mod.state_dict(), buf)
    return buf.getbuffer().nbytes / 2 ** 20


n_b = 60 if not cfg.is_smoke else 10
L32, t32 = bench(cpu_model, n_b)
L8, t8 = bench(qmodel, n_b)
qtab = pd.DataFrame({"FP32": {"tamaño (MB)": size_mb(cpu_model), "latencia p50 CPU (ms)": t32[0],
                              "latencia p95 CPU (ms)": t32[1], "F1 (subconjunto)": ne.f1m(y[te[:n_b]], L32)},
                     "INT8 dinámico": {"tamaño (MB)": size_mb(qmodel), "latencia p50 CPU (ms)": t8[0],
                                       "latencia p95 CPU (ms)": t8[1], "F1 (subconjunto)": ne.f1m(y[te[:n_b]], L8)}}).T
qtab["acuerdo de argmax con FP32"] = [1.0, float((L32.argmax(1) == L8.argmax(1)).mean())]
save_table(cfg, "nlp_cuantizacion", qtab)
qtab.round(3)
