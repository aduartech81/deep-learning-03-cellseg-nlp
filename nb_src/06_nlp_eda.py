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
# # 6. Texto: análisis exploratorio y auditoría de datos
#
# Hojas de vida en inglés clasificadas en 24 categorías de cargo (Jarvis Calling Hiring Contest). Al ser un
# contexto de **selección de personal**, la auditoría incluye información personal, atributos protegidos y sus
# proxies, y posibles atajos léxicos antes de cualquier modelado.

# %%
import json
import re
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dl03 import nlp_data as nd
from dl03 import nlp_experiments as ne
from dl03.viz import PALETTE, save_json, save_table, savefig, setup

cfg = setup()
df, y, classes, groups, sp = ne.prepare(cfg)
raw_path = cfg.data_dir / "raw" / "resumes" / "Resume.csv"
raw = pd.read_csv(raw_path)
print(json.loads((raw_path.parent / "MANIFEST.json").read_text()) if (raw_path.parent / "MANIFEST.json").exists()
      else "smoke")
schema = {"filas": len(raw), "columnas": list(raw.columns), "dtypes": raw.dtypes.astype(str).to_dict(),
          "nulos": raw.isna().sum().to_dict(), "IDs únicos": int(raw["ID"].nunique()),
          "categorías": int(raw["Category"].nunique())}
save_json(cfg, "nlp_schema", schema)
schema

# %%
import pandera.pandas as pa  # noqa: E402

pa.DataFrameSchema({"ID": pa.Column(int), "Resume_str": pa.Column(str, pa.Check.str_length(min_value=1)),
                    "Category": pa.Column(str, pa.Check.isin(nd.CATEGORIES))}).validate(raw, lazy=True)
print("integridad OK")

# %% [markdown]
# ## 6.1 Clases

# %%
cnt = df["label"].value_counts()
imb = cnt.max() / cnt.min()
fig, ax = plt.subplots(figsize=(11, 3.6))
cnt.plot.bar(ax=ax, color=PALETTE[0])
ax.set(title=f"Documentos por categoría (razón de desbalance máx/mín = {imb:.1f})", ylabel="n")
savefig(cfg, "06_clases")
plt.show()
save_table(cfg, "nlp_clases", cnt.to_frame("n").assign(prop=lambda d: d.n / d.n.sum()))

# %% [markdown]
# ## 6.2 Longitud (palabras y tokens del tokenizer real)

# %%
if cfg.nlp["transformer"]["name"] == "tiny-random":
    from dl03.nlp_models import tiny_tokenizer
    tok = tiny_tokenizer(df["text"].map(nd.clean_minimal).tolist())
else:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg.nlp["transformer"]["name"])
df["n_words"] = df["text"].str.split().str.len()
df["n_tokens"] = [len(tok(t, add_special_tokens=False, truncation=False)["input_ids"])
                  for t in df["text"].map(nd.clean_minimal)]
df["n_words_std"] = df["text"].map(nd.clean_standard).str.split().str.len()
pct = df[["n_words", "n_words_std", "n_tokens"]].quantile([0.5, 0.9, 0.95, 0.99]).T
trunc = pd.DataFrame({L: (df["n_words_std"] > L).groupby(df["label"]).mean() for L in [50, 100, 300, 500]})
trunc["tokens>510 (transformer)"] = (df["n_tokens"] > 510).groupby(df["label"]).mean()
fig, ax = plt.subplots(1, 2, figsize=(14, 3.8))
ax[0].hist(df["n_tokens"], bins=60, color=PALETTE[0])
for q, v in pct.loc["n_tokens"].items():
    ax[0].axvline(v, ls="--", c="gray")
    ax[0].text(v, ax[0].get_ylim()[1] * 0.9, f"p{int(q * 100)}", fontsize=8)
ax[0].set(title="Tokens WordPiece por documento", xlabel="tokens")
df.boxplot("n_words_std", by="label", ax=ax[1], rot=90, showfliers=False)
ax[1].set(title="Palabras (tras limpieza estándar) por categoría", xlabel="")
plt.suptitle("")
savefig(cfg, "06_longitud")
plt.show()
save_table(cfg, "nlp_percentiles_longitud", pct)
save_table(cfg, "nlp_truncamiento", trunc)
pct.round(0)

# %%
trunc.round(2)

# %% [markdown]
# ## 6.3 Frecuencias, n-gramas y vocabulario (Zipf, Heaps, OOV)

# %%
tr_idx, te_idx = np.array(sp["train"]), np.array(sp["test"])
raw_tokens = [re.findall(r"[a-z]+", t.lower()) for t in df["text"]]
std_tokens = [t.split() for t in df["text"].map(nd.clean_standard)]
c_all = Counter(w for d in raw_tokens for w in d)
c_std = Counter(w for d in std_tokens for w in d)
fig, ax = plt.subplots(1, 4, figsize=(20, 4))
pd.Series(dict(c_all.most_common(20))).plot.barh(ax=ax[0], color=PALETTE[1])
ax[0].set_title("Top 20 con stopwords")
ax[0].invert_yaxis()
pd.Series(dict(c_std.most_common(20))).plot.barh(ax=ax[1], color=PALETTE[0])
ax[1].set_title("Top 20 sin stopwords")
ax[1].invert_yaxis()
freqs = np.array(sorted(c_std.values(), reverse=True))
ranks = np.arange(1, len(freqs) + 1)
zipf_s = -np.polyfit(np.log(ranks[:5000]), np.log(freqs[:5000]), 1)[0]
ax[2].loglog(ranks, freqs)
ax[2].set(title=f"Zipf (pendiente ≈ −{zipf_s:.2f})", xlabel="rango", ylabel="frecuencia")
seen, growth, n_tok = set(), [], 0
for d in std_tokens:
    for w in d:
        n_tok += 1
        seen.add(w)
    growth.append((n_tok, len(seen)))
growth = np.array(growth)
heaps_b = np.polyfit(np.log(growth[:, 0]), np.log(growth[:, 1]), 1)[0]
ax[3].loglog(growth[:, 0], growth[:, 1])
ax[3].set(title=f"Heaps V ∝ N^β, β ≈ {heaps_b:.2f}", xlabel="tokens vistos", ylabel="vocabulario")
savefig(cfg, "06_frecuencias")
plt.show()
vocab_tr = {w for i in tr_idx for w in std_tokens[i]}
te_tok = [w for i in te_idx for w in std_tokens[i]]
oov = {"OOV por token (test vs vocab. train)": float(np.mean([w not in vocab_tr for w in te_tok])),
       "OOV por tipo": float(np.mean([w not in vocab_tr for w in set(te_tok)])), "Zipf s": zipf_s,
       "Heaps β": heaps_b, "vocabulario total": len(c_std)}
save_json(cfg, "nlp_vocabulario", oov)
oov

# %%
from sklearn.feature_extraction.text import CountVectorizer  # noqa: E402

show = cnt.index[:6]
fig, ax = plt.subplots(2, 6, figsize=(24, 7))
for j, c in enumerate(show):
    texts = df.loc[df.label == c, "text"].map(nd.clean_standard)
    for i, n in enumerate([2, 3]):
        cv = CountVectorizer(ngram_range=(n, n), max_features=2000).fit(texts)
        s = pd.Series(np.asarray(cv.transform(texts).sum(0)).ravel(), cv.get_feature_names_out()).nlargest(8)
        s[::-1].plot.barh(ax=ax[i, j], color=PALETTE[i])
        ax[i, j].set_title(f"{c} · {n}-gramas", fontsize=9)
plt.tight_layout()
savefig(cfg, "06_ngramas")
plt.show()

# %% [markdown]
# ## 6.4 Auditoría de calidad
# ### Duplicados y casi-duplicados
# Hash exacto del texto normalizado + MinHash/LSH (Jaccard ≥ 0.8 sobre 5-gramas de palabras). Los grupos se
# mantienen en una sola partición. **Impacto si se dejaran:** se compara el F1 de TF-IDF + LR con una partición
# aleatoria (duplicados pueden cruzar) frente a la agrupada.

# %%
pairs = json.loads((cfg.out_dir / "nlp" / "splits.json").read_text())["pairs"]
dup = {"pares (casi-)duplicados": len(pairs), "de ellos exactos": int(sum(p[3] for p in pairs)),
       "documentos implicados": len({i for p in pairs for i in p[:2]}),
       "grupos con > 1 documento": int((pd.Series(groups).value_counts() > 1).sum()),
       "pares con etiqueta distinta": int(sum(df.label.iloc[p[0]] != df.label.iloc[p[1]] for p in pairs))}
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, cross_val_predict  # noqa: E402

from dl03.nlp_models import fit_tfidf_lr  # noqa: E402


def cv_f1(splitter, use_groups):
    f = []
    X = df["text"].values
    for a, b in splitter.split(X, y, groups if use_groups else None):
        m = fit_tfidf_lr(X[a], y[a], None, None, {}, 0, len(classes))
        f.append(ne.f1m(y[b], m.predict_proba(X[b])))
    return np.mean(f)


dup["F1 CV aleatoria (duplicados pueden cruzar)"] = cv_f1(StratifiedKFold(5, shuffle=True, random_state=0), False)
dup["F1 CV agrupada"] = cv_f1(StratifiedGroupKFold(5, shuffle=True, random_state=0), True)
save_json(cfg, "nlp_duplicados", dup)
dup

# %% [markdown]
# ### Ruido de etiquetas (confident learning)
# Probabilidades fuera de muestra (CV 5-fold agrupada sobre train+val; el test no se toca) con TF-IDF + LR;
# `cleanlab.find_label_issues` señala ejemplos con etiqueta probablemente errónea. Se exporta una muestra de 50
# para revisión manual del autor (una sola persona: sin κ entre anotadores, declarado en el pre-registro).

# %%
from cleanlab.filter import find_label_issues  # noqa: E402

trva = np.array(sp["train"] + sp["val"])
oof = np.zeros((len(trva), len(classes)))
for a, b in StratifiedGroupKFold(5, shuffle=True, random_state=1).split(trva, y[trva], groups[trva]):
    m = fit_tfidf_lr(df.text.values[trva[a]], y[trva[a]], None, None, {}, 0, len(classes))
    oof[b] = m.predict_proba(df.text.values[trva[b]])
issues = find_label_issues(y[trva], oof, return_indices_ranked_by="self_confidence")
noise = {"documentos señalados": int(len(issues)), "tasa estimada de error de etiqueta": len(issues) / len(trva)}
sheet = df.iloc[trva[issues[:50]]][["ID", "label"]].assign(
    sugerida=[classes[k] for k in oof[issues[:50]].argmax(1)], titulo=df.iloc[trva[issues[:50]]]["text"]
    .map(nd.first_line_title).str[:80], etiqueta_correcta_segun_revision="", comentario="")
sheet.to_csv(cfg.out_dir / "tables" / "nlp_revision_manual_etiquetas.csv", index=False)
np.save(cfg.out_dir / "nlp" / "label_issues.npy", trva[issues])
save_json(cfg, "nlp_ruido_etiquetas", noise)
display(noise)  # noqa: F821
sheet.head(10)

# %% [markdown]
# ### Palabras discriminativas: log-odds con prior de Dirichlet informativo (Monroe et al., 2008)
# Para cada categoría frente al resto: $\hat\delta_w = \log\frac{y_w^i + \alpha_w}{n^i + \alpha_0 - y_w^i - \alpha_w}
# - \log\frac{y_w^j + \alpha_w}{n^j + \alpha_0 - y_w^j - \alpha_w}$, varianza $\approx 1/(y_w^i + \alpha_w) +
# 1/(y_w^j + \alpha_w)$, y se reporta $z = \hat\delta/\sigma$ (|z| > 1.96 ≈ IC 95 % que excluye 0).

# %%
cvz = CountVectorizer(min_df=3).fit(df.text.map(nd.clean_standard).values[tr_idx])
Xc = cvz.transform(df.text.map(nd.clean_standard).values[tr_idx])
vocab = cvz.get_feature_names_out()
prior = np.asarray(Xc.sum(0)).ravel() + 0.01
a0 = prior.sum()
ytr = y[tr_idx]
top_rows = []
for k, c in enumerate(classes):
    yi = np.asarray(Xc[ytr == k].sum(0)).ravel()
    yj = np.asarray(Xc[ytr != k].sum(0)).ravel()
    ni, nj = yi.sum(), yj.sum()
    d = np.log((yi + prior) / (ni + a0 - yi - prior)) - np.log((yj + prior) / (nj + a0 - yj - prior))
    z = d / np.sqrt(1 / (yi + prior) + 1 / (yj + prior))
    for w_ in np.argsort(-z)[:8]:
        top_rows.append({"categoría": c, "palabra": vocab[w_], "z": z[w_]})
disc = pd.DataFrame(top_rows)
save_table(cfg, "nlp_palabras_discriminativas", disc)
disc.groupby("categoría").head(4).pivot_table(index="categoría", values="palabra", aggfunc=", ".join).head(24)

# %% [markdown]
# ### Atajos candidatos (*spurious features*)
# * **Título del cargo** en la primera línea: ¿cuánto predice por sí solo la categoría?
# * **Plantillas** repetidas (p. ej., "Company Name", "City, State"), fechas, nombres propios.

# %%
titles = df["text"].map(nd.first_line_title)
kw_hit = np.mean([any(p in t.lower() for p in c.lower().replace("-", " ").split()) for t, c in zip(titles, df.label)])
Xt = titles.values
f_title = []
for a, b in StratifiedGroupKFold(5, shuffle=True, random_state=2).split(trva, y[trva], groups[trva]):
    m = fit_tfidf_lr(Xt[trva[a]], y[trva[a]], None, None, {"ngram": 1}, 0, len(classes))
    f_title.append(ne.f1m(y[trva[b]], m.predict_proba(Xt[trva[b]])))
phr = Counter()
for t in df.text:
    phr.update(set(re.findall(r"\b(?:[A-Z][a-z]+ ){1,2}(?:Name|State|City|Date)\b", t)))
shortcut = {"títulos que contienen una palabra de la categoría": kw_hit,
            "F1 macro usando SOLO el título (CV agrupada)": float(np.mean(f_title)),
            "plantillas frecuentes": dict(phr.most_common(8)),
            "documentos con años (19xx/20xx)": float(df.text.str.contains(r"\b(?:19|20)\d{2}\b").mean())}
save_json(cfg, "nlp_atajos", shortcut)
shortcut

# %% [markdown]
# ### Estructura temática: LDA y proyección de embeddings (TF-IDF → SVD → UMAP)
# ¿Los tópicos se alinean con las clases o con artefactos de origen? Se mide la información mutua normalizada entre
# el tópico dominante y la categoría.

# %%
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.metrics import normalized_mutual_info_score  # noqa: E402

cv_l = CountVectorizer(max_df=0.5, min_df=3, max_features=20000)
Xl = cv_l.fit_transform(df.text.map(nd.clean_standard))
lda = LatentDirichletAllocation(n_components=len(classes), random_state=cfg.seed, learning_method="online",
                                max_iter=10 if not cfg.is_smoke else 3)
theta = lda.fit_transform(Xl)
nmi = normalized_mutual_info_score(y, theta.argmax(1))
terms = cv_l.get_feature_names_out()
topics = {f"tópico {k}": ", ".join(terms[np.argsort(-c)[:8]]) for k, c in enumerate(lda.components_[:8])}
Zs = TruncatedSVD(100 if not cfg.is_smoke else 20, random_state=cfg.seed).fit_transform(
    TfidfVectorizer(min_df=2, sublinear_tf=True).fit_transform(df.text.map(nd.clean_standard)))
import umap  # noqa: E402

U = umap.UMAP(random_state=cfg.seed, n_neighbors=15).fit_transform(Zs)
fig, ax = plt.subplots(figsize=(8, 6))
cmap = plt.get_cmap("tab20", len(classes))
for k, c in enumerate(classes):
    ax.scatter(U[y == k, 0], U[y == k, 1], s=6, color=cmap(k), label=c)
ax.legend(fontsize=6, ncol=2, bbox_to_anchor=(1.01, 1), loc="upper left")
ax.set_title(f"UMAP de TF-IDF (SVD) por categoría · NMI(tópico LDA, clase) = {nmi:.2f}")
savefig(cfg, "06_umap_topicos")
plt.show()
save_json(cfg, "nlp_topicos", {"nmi_topico_clase": nmi, "topicos": topics})
topics

# %% [markdown]
# ### Idioma, codificación, información personal (PII) y atributos protegidos o proxies

# %%
PAT = {"correo": r"\S+@\S+\.\w+", "teléfono": r"\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}", "URL": r"https?://|www\.",
       "pronombres masculinos": r"\b(?:he|him|his|mr)\b", "pronombres femeninos": r"\b(?:she|her|hers|mrs|ms)\b",
       "edad / nacimiento": r"\b(?:\d{2} years old|age:? ?\d{2}|born (?:in|on))\b",
       "nacionalidad / origen": r"\b(?:citizen|nationality|visa|green card|immigrant)\b",
       "estado civil": r"\b(?:married|single|divorced|marital)\b",
       "religión": r"\b(?:church|christian|muslim|jewish|hindu|catholic)\b",
       "militar / veterano": r"\b(?:veteran|army|navy|air force|marine corps)\b",
       "placeholders de anonimización": r"Company Name|City ?, ?State"}
pii = {k: float(df.text.str.contains(v, case=False, regex=True).mean()) for k, v in PAT.items()}
non_ascii = df.text.map(lambda t: np.mean([ord(ch) > 127 for ch in t]) if t else 0)
eng = df.text.map(lambda t: np.mean([w in nd.ENGLISH_STOP_WORDS for w in t.lower().split()]) if t.split() else 0)
pii |= {"fracción media de caracteres no ASCII": float(non_ascii.mean()),
        "docs con < 10 % de stopwords en inglés (posible otro idioma)": float((eng < 0.10).mean())}
save_json(cfg, "nlp_pii", pii)
pd.Series(pii).round(4).to_frame("proporción de documentos")

# %% [markdown]
# ### Diversidad léxica (TTR, MTLD) y legibilidad (Flesch) por clase: posibles fuentes de atajo

# %%
df["ttr"] = [len(set(t)) / max(len(t), 1) for t in std_tokens]
df["mtld"] = [nd.mtld(t) for t in std_tokens]
df["flesch"] = df.text.map(nd.flesch_reading_ease)
lex = df.groupby("label")[["n_words", "ttr", "mtld", "flesch"]].median()
from scipy.stats import kruskal  # noqa: E402

lex_p = {c: kruskal(*[g[c].values for _, g in df.groupby("label")]).pvalue for c in ["n_words", "mtld", "flesch"]}
save_table(cfg, "nlp_lexico", lex)
print("Kruskal–Wallis entre clases:", {k: f"{v:.2e}" for k, v in lex_p.items()})
lex.round(2)

# %% [markdown]
# ## 6.5 Interpretación cuantitativa
#
# Las cifras de este capítulo (en `results/tables/nlp_*`) fijan las decisiones de modelado:
#
# 1. **Desbalance**: la razón máx/mín de 6.1 justifica F1 **macro** como métrica primaria (cada categoría pesa
#    igual) y la partición estratificada.
# 2. **Longitud**: los percentiles de 6.2 muestran que las hojas de vida son largas; con `sequence_length` = 50 o
#    100 (rejilla obligatoria del BiLSTM) la gran mayoría de documentos se trunca, y el transformer (510 tokens)
#    también trunca una fracción importante. Por eso se usa truncamiento cabeza + cola en el transformer y se
#    reporta la tasa de truncamiento por clase.
# 3. **Ruido**: duplicados (6.4) y tasa estimada de error de etiqueta (confident learning) acotan el F1 alcanzable;
#    se reevalúa el ranking tras quitar los ejemplos sospechosos (capítulo 8).
# 4. **Atajos**: el título del cargo por sí solo predice la categoría con el F1 de 6.4; el capítulo 8 enmascara el
#    título y mide la caída de cada modelo.
# 5. **Equidad**: los datos están parcialmente anonimizados (placeholders) y casi no contienen atributos
#    protegidos explícitos; los proxies detectables son escasos (6.4), lo que limita las métricas por subgrupo y
#    obliga a pruebas contrafactuales.
