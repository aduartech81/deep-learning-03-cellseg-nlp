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
# # 3. Segmentación: análisis exploratorio y auditoría de datos
#
# Objetivo: entender estructura, sesgos y fuentes de error de Sartorius **antes** de modelar, y derivar de ello
# decisiones de diseño cuantificadas (anchors, resolución, post-proceso, estratificación y particiones).

# %%
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from skimage import filters, io

from dl03 import seg_data as sd
from dl03.rle import rle_decode
from dl03.viz import CELL_COLORS, ci_values, overlay, save_json, save_table, savefig, setup

cfg = setup()
images, instances = sd.build_index(cfg)
raw = cfg.data_dir / "raw" / "sartorius"
ann = pd.read_csv(raw / "train.csv")
print(json.loads((raw / "MANIFEST.json").read_text()))
print(f"{len(images)} imágenes anotadas, {len(instances)} instancias, {ann['id'].nunique()} ids en train.csv")

# %% [markdown]
# ## 3.1 Esquema real vs supuesto (ficha de datos)

# %%
expected = set(sd.SEG_COLUMNS)
found = set(ann.columns)
img0 = io.imread(images["path"].iloc[0])
schema = {"columnas faltantes": sorted(expected - found), "columnas extra": sorted(found - expected),
          "forma de imagen (alto, ancho)": img0.shape, "dtype": str(img0.dtype),
          "canales": 1 if img0.ndim == 2 else img0.shape[-1],
          "rango de intensidad": (int(img0.min()), int(img0.max())),
          "tamaños distintos (alto×ancho)": images.groupby(["height", "width"]).size().to_dict(),
          "tipos celulares": images["cell_type"].value_counts().to_dict(),
          "anotaciones con solape (píxeles perdidos > 0)": int((images["overlap_px"] > 0).sum())}
save_json(cfg, "seg_schema", schema)
schema

# %% [markdown]
# ## 3.2 Pruebas de integridad (pandera)

# %%
import pandera.pandas as pa  # noqa: E402

h, w = images["height"].iloc[0], images["width"].iloc[0]


def rle_in_bounds(s: pd.Series) -> pd.Series:
    def ok(r):
        v = np.asarray(str(r).split(), dtype=np.int64)
        return v.size % 2 == 0 and (v[0::2] >= 1).all() and (v[0::2] - 1 + v[1::2] <= h * w).all()
    return s.map(ok)


ann_schema = pa.DataFrameSchema({
    "id": pa.Column(str, nullable=False),
    "annotation": pa.Column(str, nullable=False, checks=pa.Check(rle_in_bounds, element_wise=False)),
    "width": pa.Column(int, pa.Check.gt(0)), "height": pa.Column(int, pa.Check.gt(0)),
    "cell_type": pa.Column(str, pa.Check.isin(["shsy5y", "astro", "cort"])),
})
img_schema = pa.DataFrameSchema({"id": pa.Column(str, unique=True), "n_inst": pa.Column(int, pa.Check.ge(1)),
                                 "coverage": pa.Column(float, pa.Check.in_range(0, 1))})
ann_schema.validate(ann, lazy=True)
img_schema.validate(images, lazy=True)
print("integridad OK: sin nulos, RLE dentro de límites, ids únicos, rangos válidos")

# %% [markdown]
# ## 3.3 Verificación visual de la decodificación RLE
# Se superpone el contorno de las instancias decodificadas sobre la imagen. Si el orden de aplanado fuera el
# incorrecto (columnas en vez de filas), los contornos no coincidirían con las células.

# %%
rng = np.random.default_rng(cfg.seed)
pick = [images[images.cell_type == ct].sample(1, random_state=cfg.seed)["id"].iloc[0] for ct in CELL_COLORS
        if (images.cell_type == ct).any()]
fig, axes = plt.subplots(1, len(pick), figsize=(5 * len(pick), 3.8))
for ax, iid in zip(np.atleast_1d(axes), pick):
    ct = images.set_index("id").loc[iid, "cell_type"]
    overlay(sd.load_image(images.set_index("id").loc[iid, "path"]), sd.load_label(cfg, iid), ax, f"{ct} · {iid}")
savefig(cfg, "03_rle_overlay")
plt.show()
r0 = ann.iloc[0]
m = rle_decode(r0["annotation"], (r0["height"], r0["width"]))
print("primera instancia: área", m.sum(), "px; filas ocupadas", np.where(m.any(1))[0][[0, -1]],
      "(compacta en filas ⇒ aplanado en orden C correcto)")

# %% [markdown]
# ## 3.4 Análisis obligatorios (globales y por tipo celular)

# %%
inst = instances.merge(images[["id", "cell_type"]], on="id", suffixes=("", "_img"))
fig, ax = plt.subplots(2, 3, figsize=(15, 8))
for ct, col in CELL_COLORS.items():
    sub = images[images.cell_type == ct]
    if sub.empty:
        continue
    ax[0, 0].scatter(sub.width + rng.normal(0, 2, len(sub)), sub.height + rng.normal(0, 2, len(sub)), s=8, c=col,
                     label=ct)
    ax[0, 1].hist(sub.n_inst, bins=30, alpha=0.6, color=col, label=ct)
    ax[0, 2].hist(np.log10(inst[inst.cell_type == ct].area), bins=50, alpha=0.6, color=col, density=True, label=ct)
ax[0, 0].set(title="Tamaño de imagen (jitter visual)", xlabel="ancho", ylabel="alto")
ax[0, 1].set(title="Células por imagen", xlabel="n instancias")
ax[0, 2].set(title="Área de célula (log10 px)", xlabel="log10 área")
for a in ax[0]:
    a.legend()
types = [t for t in CELL_COLORS if (images.cell_type == t).any()]
ax[1, 0].violinplot([images[images.cell_type == t].n_inst for t in types], showmedians=True)
ax[1, 0].set_xticks(range(1, len(types) + 1), types)
ax[1, 0].set_title("Conteo por tipo celular")
ax[1, 1].boxplot([inst[inst.cell_type == t].area for t in types], tick_labels=types, showfliers=False)
ax[1, 1].set(title="Área por tipo (sin atípicos)", ylabel="px")
ax[1, 2].scatter(images.n_inst, images.coverage, c=images.cell_type.map(CELL_COLORS), s=8)
ax[1, 2].set(title="Densidad: conteo vs fracción cubierta", xlabel="n instancias", ylabel="cobertura")
plt.tight_layout()
savefig(cfg, "03_distribuciones")
plt.show()

# %% [markdown]
# ### Ajuste de distribución del área (log-normal vs gamma) con prueba de bondad de ajuste
# KS sobre una submuestra (n ≤ 3000 por tipo) con parámetros estimados; con muestras grandes cualquier desvío es
# "significativo", así que se reporta también el estadístico D (tamaño del desvío).

# %%
rows = []
for ct in types:
    a = inst[inst.cell_type == ct].area.values.astype(float)
    a = rng.choice(a, min(len(a), 3000), replace=False)
    for name, dist in [("lognorm", stats.lognorm), ("gamma", stats.gamma)]:
        par = dist.fit(a, floc=0)
        ks = stats.kstest(a, dist.name, par)
        rows.append({"tipo": ct, "distribución": name, "KS D": ks.statistic, "p": ks.pvalue,
                     "AIC": 2 * len(par) - 2 * dist.logpdf(a, *par).sum()})
fit = pd.DataFrame(rows)
save_table(cfg, "seg_area_fit", fit)
fit

# %% [markdown]
# ### Intensidad, contraste y SNR por imagen

# %%
def img_stats(row):
    x = sd.load_image(row.path)
    lbl = sd.load_label(cfg, row.id)
    fg, bg = x[lbl > 0], x[lbl == 0]
    lap = filters.laplace(x / 255.0)
    yy, xx = np.mgrid[: x.shape[0], : x.shape[1]]
    A = np.c_[np.ones(bg.size), xx[lbl == 0], yy[lbl == 0]]
    coef = np.linalg.lstsq(A, bg, rcond=None)[0]                 # fondo ~ plano: gradiente de iluminación
    return pd.Series({"mean": x.mean(), "std": x.std(), "contrast": abs(fg.mean() - bg.mean()) / (bg.std() + 1e-6),
                      "snr": fg.mean() / (bg.std() + 1e-6), "focus_lapvar": lap.var(),
                      "illum_grad": float(np.hypot(coef[1], coef[2]) * max(x.shape)),
                      "saturated_frac": float(((x <= 1) | (x >= 254)).mean())})


ist = images.join(images.apply(img_stats, axis=1))
fig, ax = plt.subplots(1, 3, figsize=(15, 3.6))
for ct in types:
    s = ist[ist.cell_type == ct]
    xx_ = sd.load_image(s.path.iloc[0]).ravel()
    ax[0].hist(xx_, bins=80, alpha=0.5, density=True, color=CELL_COLORS[ct], label=ct)
    ax[1].hist(s.contrast, bins=30, alpha=0.6, color=CELL_COLORS[ct], label=ct)
    ax[2].hist(s.focus_lapvar, bins=30, alpha=0.6, color=CELL_COLORS[ct], label=ct)
ax[0].set(title="Histograma de intensidad (una imagen por tipo)")
ax[1].set(title="Contraste célula/fondo (en DE del fondo)")
ax[2].set(title="Enfoque (varianza del Laplaciano)")
for a in ax:
    a.legend()
savefig(cfg, "03_intensidad")
plt.show()
save_table(cfg, "seg_image_stats", ist.drop(columns=["path"]))
ist.groupby("cell_type")[["mean", "contrast", "snr", "focus_lapvar", "illum_grad", "saturated_frac"]].median()

# %% [markdown]
# ### Muestra cualitativa: 5 imágenes al azar (semilla fija) y casos extremos

# %%
rand = images.sample(5, random_state=cfg.seed)["id"].tolist()
ext = [images.loc[images.n_inst.idxmax(), "id"], images.loc[images.n_inst.idxmin(), "id"],
       ist.loc[ist.focus_lapvar.idxmin(), "id"], ist.loc[ist.contrast.idxmin(), "id"],
       ist.loc[ist.illum_grad.idxmax(), "id"]]
labels = ["más células", "menos células", "peor enfoque", "menor contraste", "mayor gradiente de iluminación"]
fig, ax = plt.subplots(2, 5, figsize=(20, 7))
meta = images.set_index("id")
for a, iid in zip(ax[0], rand):
    overlay(sd.load_image(meta.loc[iid, "path"]), sd.load_label(cfg, iid), a, f"azar · {meta.loc[iid, 'cell_type']}")
for a, iid, lab in zip(ax[1], ext, labels):
    overlay(sd.load_image(meta.loc[iid, "path"]), sd.load_label(cfg, iid), a, f"{lab} · {meta.loc[iid, 'cell_type']}")
savefig(cfg, "03_cualitativa")
plt.show()

# %% [markdown]
# ## 3.5 Análisis adicionales (criterio de entrada al modelado)
# ### Densidad y contacto entre células
# Dos células "se tocan" si al dilatar una de ellas 1 px alcanza a la otra. Más contactos ⇒ más riesgo de
# fusiones en el post-proceso; células fragmentadas o finas ⇒ riesgo de divisiones.

# %%
from scipy import ndimage as ndi  # noqa: E402


def contacts(iid):
    lbl = sd.load_label(cfg, iid)
    pairs = set()
    shifts = [((slice(None), slice(0, -1)), (slice(None), slice(1, None))),
              ((slice(0, -1), slice(None)), (slice(1, None), slice(None))),
              ((slice(0, -1), slice(0, -1)), (slice(1, None), slice(1, None))),
              ((slice(0, -1), slice(1, None)), (slice(1, None), slice(0, -1)))]
    for sa, sb in shifts:
        a, b = lbl[sa], lbl[sb]
        m = (a > 0) & (b > 0) & (a != b)
        if m.any():
            pairs |= set(map(tuple, np.sort(np.stack([a[m], b[m]], 1), 1)))
    from skimage.measure import regionprops
    props = regionprops(lbl)
    touching = len({i for p in pairs for i in p})
    frag = sum(ndi.label(p.image)[1] > 1 for p in props)
    return pd.Series({"pares_en_contacto": len(pairs), "frac_celulas_que_tocan": touching / max(len(props), 1),
                      "instancias_fragmentadas": frag})


cont = images.set_index("id").join(pd.DataFrame({i: contacts(i) for i in images.id}).T)
save_table(cfg, "seg_contacts", cont[["cell_type", "pares_en_contacto", "frac_celulas_que_tocan",
                                      "instancias_fragmentadas"]])
cont.groupby("cell_type")[["coverage", "pares_en_contacto", "frac_celulas_que_tocan", "instancias_fragmentadas"]] \
    .agg(["median", "mean"])

# %% [markdown]
# ### Tamaños, cola pesada y resolución
# ¿Qué fracción de instancias queda por debajo de 16 px de área (≈ el *stride* 4 del nivel P2 de un FPN al
# cuadrado) o desaparece al reescalar? Se cuenta cuántas quedan con menos de 10 px tras reescalar el ancho a 512 y
# a 256.

# %%
size_rows = []
W0 = images["width"].iloc[0]
for ct in types + ["global"]:
    a = inst.area if ct == "global" else inst[inst.cell_type == ct].area
    q = a.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    size_rows.append({"tipo": ct, **{f"q{int(k * 100)}": v for k, v in q.items()},
                      "% < 16 px": 100 * (a < 16).mean(),
                      "% < 10 px a 512": 100 * (a * (512 / W0) ** 2 < 10).mean(),
                      "% < 10 px a 256": 100 * (a * (256 / W0) ** 2 < 10).mean()})
sizes = pd.DataFrame(size_rows).set_index("tipo")
save_table(cfg, "seg_sizes", sizes)
sizes.round(2)

# %% [markdown]
# ### Efectos de lote
# Se proyectan las imágenes con un codificador preentrenado (ResNet50 ImageNet; en el perfil smoke, pesos
# aleatorios) y se mide cuán distinguibles son las placas/fechas con un clasificador de dominio (AUC con validación
# cruzada **agrupada** por imagen) y con una prueba de permutación.

# %%
import keras  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import cross_val_predict, StratifiedKFold  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

enc = keras.applications.ResNet50(include_top=False, weights=None if cfg.is_smoke else "imagenet", pooling="avg")
emb = []
for p_ in images.path:
    x = sd.normalize(sd.load_image(p_), "percentile")
    x = sd.Geometry.for_resolution(x.shape, 224).to_model(np.clip(x, 0, 1))
    emb.append(enc.predict(keras.applications.resnet50.preprocess_input(np.repeat(x[None, ..., None] * 255, 3, -1)),
                           verbose=0)[0])
emb = np.array(emb)
Z = PCA(2, random_state=cfg.seed).fit_transform(emb)
group_col, groups = sd.choose_group_column(images)
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
for ct in types:
    m_ = images.cell_type.values == ct
    ax[0].scatter(Z[m_, 0], Z[m_, 1], s=8, c=CELL_COLORS[ct], label=ct)
ax[0].legend()
ax[0].set_title("PCA de embeddings por tipo celular")
codes = pd.factorize(groups)[0]
ax[1].scatter(Z[:, 0], Z[:, 1], s=8, c=codes, cmap="tab20")
ax[1].set_title(f"…coloreado por grupo ({group_col})")
savefig(cfg, "03_batch_effects")
plt.show()

batch_rows = []
for ct in types:
    m_ = (images.cell_type == ct).values
    g_ = pd.factorize(images.loc[m_, "sample_date"])[0]
    if len(np.unique(g_)) < 2:
        continue
    top = pd.Series(g_).value_counts().index[0]
    yb = (g_ == top).astype(int)                       # fecha más frecuente vs resto
    if yb.sum() < 5 or (1 - yb).sum() < 5:
        continue
    pr = cross_val_predict(LogisticRegression(max_iter=2000, C=0.1), emb[m_], yb,
                           cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
    auc = roc_auc_score(yb, pr)
    perm = [roc_auc_score(rng.permutation(yb), pr) for _ in range(200)]
    batch_rows.append({"tipo": ct, "AUC fecha mayoritaria vs resto": auc,
                       "p permutación": (1 + sum(p >= auc for p in perm)) / 201})
batch = pd.DataFrame(batch_rows)
save_table(cfg, "seg_batch_effects", batch)
batch

# %% [markdown]
# ### Calidad de anotación y auditoría de fuga
# * Instancias diminutas (< 10 px), fragmentadas (más de una componente conexa) y píxeles en solape.
# * Casi-duplicados entre grupos con *hash* perceptual (dHash de 64 bits; distancia de Hamming ≤ 4).
# * Muestra de 30 imágenes para revisión manual de células visibles sin anotar (hoja exportada para el autor).

# %%
def dhash(x, n=8):
    from skimage.transform import resize
    r = resize(x, (n, n + 1), anti_aliasing=True)
    return (r[:, 1:] > r[:, :-1]).ravel()


hashes = np.array([dhash(sd.load_image(p_)) for p_ in images.path])
ham = (hashes[:, None, :] != hashes[None, :, :]).sum(-1)
iu = np.triu_indices(len(images), 1)
near = [(images.id.iloc[i], images.id.iloc[j], int(ham[i, j])) for i, j in zip(*iu) if ham[i, j] <= 4]
gmap = dict(zip(images.id, groups))
cross = [p for p in near if gmap[p[0]] != gmap[p[1]]]
qual = {"instancias < 10 px": int((inst.area < 10).sum()),
        "instancias fragmentadas": int(cont["instancias_fragmentadas"].sum()),
        "imágenes con solape de anotaciones": int((images.overlap_px > 0).sum()),
        "píxeles en solape (total)": int(images.overlap_px.sum()),
        "pares casi-duplicados (Hamming ≤ 4)": len(near), "…de ellos entre grupos distintos": len(cross)}
images.sample(min(30, len(images)), random_state=cfg.seed)[["id", "cell_type", "n_inst"]].assign(
    celulas_visibles_sin_anotar="", mascaras_erroneas="", comentario="").to_csv(
    cfg.out_dir / "tables" / "seg_revision_manual_anotaciones.csv", index=False)
save_json(cfg, "seg_quality", qual)
qual

# %% [markdown]
# ## 3.6 Interpretación cuantitativa

# %%
import statsmodels.formula.api as smf  # noqa: E402

icc_rows = []
for ct in types:
    s = ist[ist.cell_type == ct].assign(g=lambda d: d["sample_date"].astype(str))
    if s.g.nunique() < 2:
        continue
    r_ = smf.mixedlm("mean ~ 1", s, groups=s["g"]).fit(reml=True)
    v_g, v_e = float(r_.cov_re.iloc[0, 0]), float(r_.scale)
    icc_rows.append({"tipo": ct, "ICC intensidad (fecha)": v_g / (v_g + v_e),
                     "CV de la intensidad media entre fechas":
                         s.groupby("g")["mean"].mean().std() / s.groupby("g")["mean"].mean().mean()})
counts = images.n_inst.values


def gini(x):
    x = np.sort(np.asarray(x, float))
    n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1) @ x / (n * x.sum()))


summary = {
    "homogeneidad": icc_rows,
    "conteo: Gini": gini(counts), "conteo: CV": float(counts.std() / counts.mean()),
    "conteo medio por tipo [IC 95 %]": {ct: ci_values(images[images.cell_type == ct].n_inst, 2000) for ct in types},
    "proporción de imágenes por tipo": images.cell_type.value_counts(normalize=True).round(3).to_dict(),
    "proporción de instancias por tipo": inst.cell_type.value_counts(normalize=True).round(3).to_dict(),
    "área: cuantiles (px)": sizes.loc["global", ["q5", "q50", "q95"]].to_dict(),
    "imágenes con enfoque bajo (lapvar < p10)": int((ist.focus_lapvar < ist.focus_lapvar.quantile(0.1)).sum()),
    "imágenes con saturación > 1 %": int((ist.saturated_frac > 0.01).sum()),
}
summary

# %% [markdown]
# ## 3.7 Decisiones de diseño derivadas del EDA y particiones
# Se guardan en `results/seg/eda_design.json` y los capítulos de modelado las leen de ahí (no se fijan a mano).

# %%
a_all = inst.area.values
q = np.quantile(np.sqrt(a_all), [0.05, 0.25, 0.5, 0.75, 0.99])
anchors = tuple(int(2 ** np.round(np.log2(max(4, v)))) for v in q)
anchors = tuple(sorted(set(anchors)))
while len(anchors) < 5:
    anchors = anchors + (anchors[-1] * 2,)
design = {
    "anchors_maskrcnn": anchors[:5],
    "sam_max_area": int(np.quantile(a_all, 0.999) * 1.5),
    "min_size_px": int(max(5, np.quantile(a_all, 0.01) * 0.5)),
    "max_detections": int(min(1000, images.n_inst.max() * 1.2)),
    "resolucion_base": "ancho 512 (pérdida de instancias < 10 px: %.2f %%)" % sizes.loc["global", "% < 10 px a 512"],
    "group_col": group_col,
}
splits = sd.make_splits(images, cfg)
sd.assert_no_group_leak(images, splits)
(cfg.out_dir / "seg").mkdir(parents=True, exist_ok=True)
(cfg.out_dir / "seg" / "splits.json").write_text(splits.to_json())
(cfg.out_dir / "seg" / "eda_design.json").write_text(json.dumps(design, indent=1))
tab = pd.DataFrame({k: images.set_index("id").loc[getattr(splits, k), "cell_type"].value_counts()
                    for k in ("train", "val", "test")})
print(design)
tab.assign(total=tab.sum(1))

# %% [markdown]
# **Implicaciones de diseño (cada una apoyada en una cifra de este capítulo):**
#
# 1. *Anchors* de Mask R-CNN a partir de los cuantiles de √área (tabla 3.5), no los de COCO (32–512 px).
# 2. Resolución base ancho 512 y no 256: a 256 se pierde una fracción mayor de instancias por debajo de 10 px
#    (tabla 3.5); 256 queda solo para ablaciones.
# 3. Objetivo de 3 clases (fondo/interior/borde) en U-Net++ y mapas H/V en el modelo estilo Hover-Net, por la
#    fracción de células en contacto (3.5): un objetivo binario fusionaría células vecinas.
# 4. Estratificación por tipo celular y reporte estratificado: conteo, área y contraste difieren entre tipos
#    (3.4), así que la métrica global depende de la mezcla del test.
# 5. Particiones agrupadas por la columna elegida (`group_col`): las fechas/placas son distinguibles en el espacio
#    de embeddings (AUC del clasificador de dominio), lo que confirma el riesgo de fuga si se mezclaran.
# 6. Normalización por percentiles por imagen: la intensidad media varía entre fechas (ICC/CV en 3.6).
# 7. Número máximo de detecciones y `min_size` del post-proceso derivados del conteo máximo y del percentil 1 del
#    área.
