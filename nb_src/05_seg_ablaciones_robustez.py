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
# # 5. Segmentación: ablaciones, eficiencia de datos, robustez, calibración y eficiencia
#
# Las ablaciones usan U-Net++ (modelo propio) con ≥ 3 semillas y la mitad de las épocas del modelo final. Para no
# gastar el test sellado en decenas de variantes, **las ablaciones se evalúan en validación**, dividida en dos
# mitades agrupadas: el post-proceso se ajusta en la mitad A y se mide en la mitad B. La curva de eficiencia de
# datos (H1) sí se evalúa en test, porque es la prueba pre-registrada de H1.

# %%
import json
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import optimize
from scipy import stats as sst

from dl03 import seg_data as sd
from dl03 import seg_experiments as se
from dl03 import stats as st
from dl03.viz import PALETTE, save_json, save_table, savefig, setup

cfg = setup(seed_offset=5)
PRE = not cfg.is_smoke
images, instances = sd.build_index(cfg)
splits = sd.Splits.from_json((cfg.out_dir / "seg" / "splits.json").read_text())
design = json.loads((cfg.out_dir / "seg" / "eda_design.json").read_text())
best_hp = {k: (int(v) if k == "batch" else v) for k, v in
           json.loads((cfg.out_dir / "seg" / "params" / "unetpp_hpo.json").read_text())["best"].items()}
SEEDS = cfg.seg["seeds_heavy"] if len(cfg.seg["seeds_heavy"]) >= 3 else cfg.seg["seeds_light"]
AB = cfg.seg["ablation_size"]
EP = max(2, cfg.seg["unetpp"]["epochs"] // 2)
try:
    from codecarbon import OfflineEmissionsTracker
    tracker = OfflineEmissionsTracker(country_iso_code="USA", log_level="error", save_to_file=False)
    tracker.start()
except Exception:
    tracker = None

# mitades agrupadas de validación: A (ajuste del post-proceso) y B (medición)
_, g = sd.choose_group_column(images)
gmap = dict(zip(images.id, g))
vg = sorted({gmap[i] for i in splits.val})
rng = np.random.default_rng(cfg.seed)
ga = set(rng.choice(vg, len(vg) // 2, replace=False))
VA = [i for i in splits.val if gmap[i] in ga]
VB = [i for i in splits.val if gmap[i] not in ga]
print(f"validación A {len(VA)} imágenes · B {len(VB)} imágenes · semillas {SEEDS} · épocas {EP}")


def ablation(tag, seed, res=AB, loss="ce_dice", aug="geom_photo", norm="percentile", frac=1.0, eval_ids=None,
             train_ids=None):
    """Entrena (o recupera) una variante y devuelve las métricas por imagen en ``eval_ids`` (por defecto VB)."""
    eval_ids = eval_ids if eval_ids is not None else VB
    sub = sd.Splits(splits.group_col, train_ids or splits.train, VA, eval_ids, [])
    pdir, _ = se.unetpp_full(cfg, images, sub, seed, f"ab_{tag}_s{seed}", res=res, loss=loss, aug=aug, norm=norm,
                             frac=frac, hp=best_hp, epochs=EP, test_ids=eval_ids)
    return se.evaluate(cfg, images, se.load_preds(pdir, eval_ids), tag, seed, with_prob=False)


# %% [markdown]
# ## 5.1 Ablación de preprocesamiento: resolución × normalización × augmentation (factorial 2×2×2)

# %%
t0 = time.time()
rows = []
for res in [AB, cfg.seg["img_size"]]:
    for norm in ["div255", "percentile"]:
        for aug in ["none", "geom_photo"]:
            for s in SEEDS:
                d = ablation(f"fact_r{res}_{norm}_{aug}", s, res=res, norm=norm, aug=aug)
                rows.append(d.assign(res=str(res), norm=norm, aug=aug))
fact = pd.concat(rows, ignore_index=True)
fact.to_csv(cfg.out_dir / "seg" / "ablation_factorial.csv", index=False)
fit = smf.mixedlm("ap ~ C(res) * C(norm) * C(aug)", fact, groups=fact["id"]).fit(reml=True)
anova = pd.DataFrame({"coef": fit.fe_params, "p": fit.pvalues.reindex(fit.fe_params.index)}).round(4)
save_table(cfg, "seg_ablacion_factorial", anova)
print(f"{time.time() - t0:.0f} s · modelo mixto con intercepto aleatorio por imagen (medidas repetidas)")
fact.groupby(["res", "norm", "aug"])["ap"].agg(["mean", "std", "count"]).round(4)

# %%
anova

# %% [markdown]
# ## 5.2 Ablaciones de un factor (resto fijo en la configuración base: 256 px, percentiles, geom+foto, BCE+Dice)

# %%
one = {
    "resolución": [("256", dict(res=AB)), (str(cfg.seg["img_size"]), dict(res=cfg.seg["img_size"])),
                   ("nativa+pad", dict(res="native")), ("teselas", dict(res="tiling"))],
    "augmentation": [("ninguna", dict(aug="none")), ("geométrica", dict(aug="geom")),
                     ("geom+fotométrica", dict(aug="geom_photo")), ("+elástica", dict(aug="geom_photo_elastic"))],
    "pérdida": [("BCE", dict(loss="ce")), ("Dice", dict(loss="dice")), ("BCE+Dice", dict(loss="ce_dice")),
                ("Tversky", dict(loss="tversky")), ("focal", dict(loss="focal"))],
}
res_rows = []
for factor, variants in one.items():
    for name, kw in variants:
        tag = "one_" + "_".join(f"{k}{v}" for k, v in kw.items())
        for s in SEEDS:
            res_rows.append(ablation(tag, s, **kw).assign(factor=factor, variante=name))
abl = pd.concat(res_rows, ignore_index=True)
abl.to_csv(cfg.out_dir / "seg" / "ablation_one_factor.csv", index=False)
summ = []
for factor, sub in abl.groupby("factor", sort=False):
    per_img = sub.groupby(["variante", "id"])["ap"].mean().unstack(0)
    fr = sst.friedmanchisquare(*[per_img[c] for c in per_img]) if per_img.shape[1] > 2 else None
    seeds_sd = sub.groupby(["variante", "seed"])["ap"].mean().groupby("variante").std()
    for v in per_img:
        summ.append({"factor": factor, "variante": v, "AP medio": per_img[v].mean(),
                     "DE entre semillas": seeds_sd.get(v, np.nan),
                     "p Friedman (factor)": fr.pvalue if fr is not None else np.nan})
summ = pd.DataFrame(summ)
save_table(cfg, "seg_ablaciones", summ)
fig, ax = plt.subplots(1, 3, figsize=(16, 3.6))
for a, (factor, sub) in zip(ax, summ.groupby("factor", sort=False)):
    a.barh(sub.variante, sub["AP medio"], xerr=sub["DE entre semillas"], color=PALETTE[0])
    a.set(title=f"{factor} (p Friedman {sub['p Friedman (factor)'].iloc[0]:.3g})", xlabel="AP (validación B)")
plt.tight_layout()
savefig(cfg, "05_ablaciones")
plt.show()
summ.round(4)

# %% [markdown]
# ## 5.3 Preentrenamiento: desde cero vs preentrenado
# Cellpose desde cero vs `cyto2` (preentrenado en microscopía) y codificador del modelo estilo Hover-Net desde
# cero vs ImageNet (una semilla; desviación declarada). Evaluado en test, junto con los modelos del capítulo 4.

# %%
pre_rows = []
for s in SEEDS[:3]:
    pd_, _ = se.cellpose_full(cfg, images, splits, s, f"cellpose_scratch_s{s}", init=None)
    pre_rows.append(se.evaluate(cfg, images, se.load_preds(pd_, splits.test), "Cellpose desde cero", s, with_prob=False))
    pre_rows.append(se.evaluate(cfg, images, se.load_preds(se.run_dir(cfg, "preds", f"cellpose_ft_s{s}"), splits.test),
                                "Cellpose cyto2 (microscopía)" if PRE else "Cellpose (smoke)", s, with_prob=False))
pd_, _ = se.hovernet_full(cfg, images, splits, SEEDS[0], f"hovernet_scratch_s{SEEDS[0]}", pretrained=False,
                          res=cfg.seg["img_size"])
pre_rows.append(se.evaluate(cfg, images, se.load_preds(pd_, splits.test), "Hover-Net desde cero", SEEDS[0],
                            with_prob=False))
pre_rows.append(se.evaluate(cfg, images, se.load_preds(se.run_dir(cfg, "preds", f"hovernet_s{SEEDS[0]}"), splits.test),
                            "Hover-Net ImageNet" if PRE else "Hover-Net (smoke)", SEEDS[0], with_prob=False))
pre = pd.concat(pre_rows, ignore_index=True)
pre_tab = pre.groupby("model")["ap"].agg(["mean", "std", "count"])
for a, b in [("Cellpose desde cero", "Cellpose cyto2 (microscopía)"), ("Hover-Net desde cero", "Hover-Net ImageNet")]:
    if a in pre.model.values and b in pre.model.values:
        da = pre[pre.model == a].groupby("id")["ap"].mean()
        db = pre[pre.model == b].groupby("id")["ap"].mean()
        print(f"{b} − {a}: ΔAP = {st.fmt_ci(*st.paired_bootstrap_diff(db, da, n_boot=cfg.n_boot))}")
save_table(cfg, "seg_preentrenamiento", pre_tab)
pre_tab.round(4)

# %% [markdown]
# ## 5.4 Eficiencia de datos (10/25/50/100 %) y prueba de H1
# U-Net++ desde cero frente a Cellpose `cyto2` ajustado, mismas fracciones estratificadas por tipo celular, 3
# semillas, evaluación en test. Ajuste de una ley de potencia $\text{error}(n) = a n^{-b} + c$ con error = 1 − AP.

# %%
fr_rows = []
for frac in cfg.seg["data_fractions"]:
    for s in SEEDS:
        sub = sd.Splits(splits.group_col, splits.train, VA, splits.test, [])
        pdir, _ = se.unetpp_full(cfg, images, sub, s, f"frac_unetpp_{frac}_s{s}", res=AB, frac=frac, hp=best_hp,
                                 epochs=EP)
        fr_rows.append(se.evaluate(cfg, images, se.load_preds(pdir, splits.test), "U-Net++ (cero)", s,
                                   with_prob=False).assign(frac=frac))
        tr_ids = se.keras_arrays(cfg, images, splits.train, AB, frac=frac, seed=s)[3] if frac < 1 else splits.train
        pdir, _ = se.cellpose_full(cfg, images, splits, s, f"frac_cellpose_{frac}_s{s}",
                                   init=cfg.seg["cellpose"]["model"] if PRE else None, train_ids=tr_ids)
        fr_rows.append(se.evaluate(cfg, images, se.load_preds(pdir, splits.test), "Cellpose (preentrenado)", s,
                                   with_prob=False).assign(frac=frac))
fr = pd.concat(fr_rows, ignore_index=True)
fr.to_csv(cfg.out_dir / "seg" / "data_efficiency.csv", index=False)
n_train = len(splits.train)


def power(n, a, b, c):
    return a * np.power(n, -b) + c


fig, ax = plt.subplots(figsize=(7, 4))
fits = {}
for k, (m, sub) in enumerate(fr.groupby("model")):
    curve = sub.groupby("frac")["ap"].agg(["mean", "std"])
    n = curve.index.values * n_train
    ax.errorbar(n, curve["mean"], yerr=curve["std"], fmt="o", color=PALETTE[k], label=m, capsize=3)
    try:
        par, _ = optimize.curve_fit(power, n, 1 - curve["mean"].values, p0=(1, 0.5, 0.3), maxfev=20000,
                                    bounds=([0, 0, 0], [100, 3, 1]))
        xx = np.linspace(n.min(), n.max() * 2, 100)
        ax.plot(xx, 1 - power(xx, *par), "--", color=PALETTE[k])
        fits[m] = {"a": par[0], "b": par[1], "c (error asintótico)": par[2],
                   "AP extrapolado con 2× datos": 1 - power(2 * n.max(), *par)}
    except Exception as e:
        fits[m] = {"error": str(e)}
ax.set(xscale="log", xlabel="imágenes de entrenamiento", ylabel="AP (test)", title="Eficiencia de datos")
ax.legend()
savefig(cfg, "05_eficiencia_datos")
plt.show()
h1 = {}
for frac in [f for f in cfg.seg["data_fractions"] if f in (0.25, 1.0)] or cfg.seg["data_fractions"]:
    a = fr[(fr.frac == frac) & (fr.model == "Cellpose (preentrenado)")].groupby("id")["ap"].mean()
    b = fr[(fr.frac == frac) & (fr.model == "U-Net++ (cero)")].groupby("id")["ap"].mean()
    h1[f"Δ AP (Cellpose − U-Net++) con {int(frac * 100)} %"] = st.paired_bootstrap_diff(a, b, n_boot=cfg.n_boot)
keys = list(h1)
ci25 = h1[keys[0]]
verdict = "refutada" if (ci25[1] <= 0 <= ci25[2]) or (len(keys) > 1 and h1[keys[-1]][0] >= ci25[0]) else \
    "no refutada"
h1_out = {k: st.fmt_ci(*v) for k, v in h1.items()} | {"H1": verdict, "leyes de potencia": fits}
save_json(cfg, "seg_H1", h1_out)
h1_out

# %% [markdown]
# ## 5.5 Robustez: dejar un tipo celular fuera (leave-one-cell-type-out)

# %%
meta = images.set_index("id")
loco = []
for ct in sorted(images.cell_type.unique()):
    tr = [i for i in splits.train if meta.loc[i, "cell_type"] != ct]
    te = [i for i in splits.test if meta.loc[i, "cell_type"] == ct]
    va = [i for i in VA if meta.loc[i, "cell_type"] != ct]
    sub = sd.Splits(splits.group_col, tr, va, te, [])
    pdir, _ = se.unetpp_full(cfg, images, sub, SEEDS[0], f"loco_{ct}", res=AB, hp=best_hp, epochs=EP, test_ids=te)
    out_ = se.evaluate(cfg, images, se.load_preds(pdir, te), "fuera", with_prob=False)
    ref = fr[(fr.frac == 1.0) & (fr.model == "U-Net++ (cero)") & (fr.id.isin(te))].groupby("id")["ap"].mean()
    d = st.paired_bootstrap_diff(out_.set_index("id").loc[ref.index, "ap"], ref, n_boot=cfg.n_boot)
    loco.append({"tipo excluido": ct, "AP con el tipo en train": ref.mean(), "AP sin el tipo": out_.ap.mean(),
                 "caída [IC 95 %]": st.fmt_ci(*d)})
loco = pd.DataFrame(loco)
save_table(cfg, "seg_loco", loco)
loco

# %% [markdown]
# ## 5.6 Robustez a perturbaciones sintéticas y TTA
# Ruido gaussiano, desenfoque, contraste/brillo, gradiente de iluminación y compresión JPEG con severidad creciente,
# sobre el test. Se reporta la curva AP vs severidad y el área relativa bajo la curva (AP medio perturbado / AP
# limpio). Modelos: U-Net++ y Cellpose ajustado (semilla 0).

# %%
import io as _io  # noqa: E402

from PIL import Image  # noqa: E402
from skimage import filters  # noqa: E402

from dl03.seg_models import cellpose_wrap as cw  # noqa: E402
from dl03.seg_models.unetpp import predict_probs, probs_to_instances  # noqa: E402

L = cfg.seg["perturb_levels"]


def perturb(x, kind, lev):
    """x en [0, 255]; lev = 1..L."""
    s = lev / L
    if kind == "ruido":
        return np.clip(x + np.random.default_rng(lev).normal(0, 25 * s, x.shape), 0, 255)
    if kind == "desenfoque":
        return filters.gaussian(x, 3 * s, preserve_range=True)
    if kind == "contraste":
        return np.clip((x - x.mean()) * (1 - 0.7 * s) + x.mean() + 30 * s, 0, 255)
    if kind == "iluminación":
        return np.clip(x * (1 - 0.6 * s * np.linspace(0, 1, x.shape[1])[None, :]), 0, 255)
    buf = _io.BytesIO()
    Image.fromarray(x.astype(np.uint8)).save(buf, "JPEG", quality=int(max(5, 90 - 85 * s)))
    return np.asarray(Image.open(buf), dtype=np.float32)


umodel = se.load_keras(se.run_dir(cfg, "models") / f"unetpp_s{SEEDS[0]}.keras")
u_pp = json.loads((cfg.out_dir / "seg" / "params" / f"unetpp_s{SEEDS[0]}.json").read_text())["params"]
cp_pp = json.loads((cfg.out_dir / "seg" / "params" / f"cellpose_ft_s{SEEDS[0]}.json").read_text())
cp_files = sorted(f for f in se.run_dir(cfg, "models", f"cellpose_ft_s{SEEDS[0]}").glob("**/*") if f.is_file()
                  and "cellpose_ft" in f.name)
cpm = cw.load_finetuned(str(cp_files[-1])) if cp_files else None
RESM = cfg.seg["img_size"]


def unet_label(x255):
    x = np.clip(sd.normalize(x255), 0, 1)
    gm = sd.Geometry.for_resolution(x.shape, RESM)
    p = gm.to_native(predict_probs(umodel, gm.to_model(x)[None])[0])
    return probs_to_instances(p, u_pp), p


def cp_label(x255):
    d = None if cp_pp.get("diameter") is None else cp_pp["diameter"] * cp_pp["params"]["diam_scale"]
    return cw.predict(cpm, x255, d, cp_pp["params"]["flow_threshold"], cp_pp["params"]["cellprob_threshold"])


models_r = {"U-Net++": unet_label} | ({"Cellpose ajustado": cp_label} if cpm is not None else {})
test_r = splits.test if not cfg.is_smoke else splits.test[:4]
rob = []
for kind in ["ruido", "desenfoque", "contraste", "iluminación", "JPEG"]:
    for lev in range(0, L + 1):
        for m, f in models_r.items():
            for iid in test_r:
                x = se.native_image(images, iid)
                lbl = f(perturb(x, kind, lev) if lev else x)[0]
                rob.append({"perturbación": kind, "severidad": lev, "modelo": m, "id": iid,
                            "ap": se.sm.sartorius_score(sd.load_label(cfg, iid), lbl)})
rob = pd.DataFrame(rob)
rob.to_csv(cfg.out_dir / "seg" / "robustness.csv", index=False)
fig, ax = plt.subplots(1, 5, figsize=(20, 3.4), sharey=True)
rauc = []
for a, (kind, sub) in zip(ax, rob.groupby("perturbación", sort=False)):
    for k, (m, s2) in enumerate(sub.groupby("modelo")):
        c = s2.groupby("severidad")["ap"].mean()
        a.plot(c.index, c.values, "o-", color=PALETTE[k], label=m)
        rauc.append({"perturbación": kind, "modelo": m, "AUC relativa": c.iloc[1:].mean() / max(c.iloc[0], 1e-9)})
    a.set(title=kind, xlabel="severidad")
ax[0].set_ylabel("AP")
ax[0].legend()
savefig(cfg, "05_robustez")
plt.show()
rauc = pd.DataFrame(rauc).pivot(index="perturbación", columns="modelo", values="AUC relativa")
save_table(cfg, "seg_robustez_auc_relativa", rauc)
rauc.round(3)

# %%
# TTA (flips): ¿mejora el AP y la estabilidad?
tta_rows = []
for iid in test_r:
    x = np.clip(sd.normalize(se.native_image(images, iid)), 0, 1)
    gm = sd.Geometry.for_resolution(x.shape, RESM)
    xm = gm.to_model(x)
    outs = []
    for fy in (False, True):
        for fx in (False, True):
            v = xm[::-1] if fy else xm
            v = v[:, ::-1] if fx else v
            p = predict_probs(umodel, np.ascontiguousarray(v)[None])[0]
            p = p[::-1] if fy else p
            p = p[:, ::-1] if fx else p
            outs.append(gm.to_native(np.ascontiguousarray(p)))
    gt = sd.load_label(cfg, iid)
    aps = [se.sm.sartorius_score(gt, probs_to_instances(p, u_pp)) for p in outs]
    tta_rows.append({"id": iid, "AP sin TTA": aps[0], "AP con TTA (media de 4 flips)":
                     se.sm.sartorius_score(gt, probs_to_instances(np.mean(outs, 0), u_pp)),
                     "DE del AP entre vistas": np.std(aps)})
tta = pd.DataFrame(tta_rows)
d_tta = st.paired_bootstrap_diff(tta["AP con TTA (media de 4 flips)"], tta["AP sin TTA"], n_boot=cfg.n_boot)
print("ΔAP por TTA:", st.fmt_ci(*d_tta))
save_table(cfg, "seg_tta", tta.describe())
tta.describe().round(4)

# %% [markdown]
# ## 5.7 Calibración e incertidumbre (U-Net++)
# * Calibración de la probabilidad de primer plano por píxel: ECE por imagen y diagrama de confiabilidad (con
#   submuestreo dentro de cada imagen).
# * Incertidumbre: MC-Dropout (T = 10), ensamble profundo de las 3 semillas y varianza bajo TTA. ¿Predice el error?
#   Correlación de Spearman entre incertidumbre media de la imagen y 1 − AP, y curva riesgo–cobertura (predicción
#   selectiva: rechazar primero las imágenes más inciertas).

# %%
from dl03.calibration import binary_ece  # noqa: E402
from dl03.seg_models.unetpp import mc_dropout_probs  # noqa: E402

members = [se.load_keras(se.run_dir(cfg, "models") / f"unetpp_s{s}.keras") for s in cfg.seg["seeds_heavy"]]
unc_rows, rel_p, rel_y = [], [], []
for iid in test_r:
    x = np.clip(sd.normalize(se.native_image(images, iid)), 0, 1)
    gm = sd.Geometry.for_resolution(x.shape, RESM)
    xm = gm.to_model(x)[None]
    gt = sd.load_label(cfg, iid)
    ens = np.stack([gm.to_native(predict_probs(mb, xm)[0]) for mb in members])
    p_ens = ens.mean(0)
    mc_mean, mc_var = mc_dropout_probs(umodel, xm, T=10 if not cfg.is_smoke else 3)
    pf = p_ens[..., 1:].sum(-1)
    sub_idx = np.random.default_rng(0).choice(pf.size, min(20000, pf.size), replace=False)
    rel_p.append(pf.ravel()[sub_idx])
    rel_y.append((gt > 0).ravel()[sub_idx])
    unc_rows.append({"id": iid, "ap_ensamble": se.sm.sartorius_score(gt, probs_to_instances(p_ens, u_pp)),
                     "ap_s0": se.sm.sartorius_score(gt, probs_to_instances(ens[0], u_pp)),
                     "ece_pixel": binary_ece(pf.ravel()[sub_idx], (gt > 0).ravel()[sub_idx]),
                     "unc_ensamble": float(ens[..., 1:].sum(-1).var(0).mean()),
                     "unc_mcdropout": float(mc_var.mean())})
unc = pd.DataFrame(unc_rows)
unc["error"] = 1 - unc["ap_ensamble"]
corr_u = {k: sst.spearmanr(unc[k], unc["error"]).statistic for k in ["unc_ensamble", "unc_mcdropout"]}
print("ECE de píxel (media por imagen):", round(unc.ece_pixel.mean(), 4),
      "| ΔAP ensamble vs una semilla:", st.fmt_ci(*st.paired_bootstrap_diff(unc.ap_ensamble, unc.ap_s0, n_boot=2000)),
      "| ρ(incertidumbre, error):", {k: round(v, 3) for k, v in corr_u.items()})
order_ = np.argsort(unc["unc_ensamble"].values)
cov = np.arange(1, len(unc) + 1) / len(unc)
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
P_, Y_ = np.concatenate(rel_p), np.concatenate(rel_y)
bins = np.linspace(0, 1, 11)
idx = np.digitize(P_, bins[1:-1])
ax[0].plot([0, 1], [0, 1], "--", c="gray")
ax[0].plot([P_[idx == b].mean() for b in range(10) if (idx == b).any()],
           [Y_[idx == b].mean() for b in range(10) if (idx == b).any()], "o-")
ax[0].set(title="Confiabilidad: probabilidad de célula (píxel)", xlabel="confianza", ylabel="frecuencia observada")
ax[1].plot(cov, np.cumsum(unc["ap_ensamble"].values[order_]) / np.arange(1, len(unc) + 1), "o-")
ax[1].set(title="Riesgo–cobertura: AP de las imágenes retenidas", xlabel="cobertura (más seguras primero)",
          ylabel="AP medio retenido")
savefig(cfg, "05_calibracion_incertidumbre")
plt.show()
save_json(cfg, "seg_incertidumbre", {"ece_pixel_media": unc.ece_pixel.mean(), "spearman_unc_error": corr_u,
                                     "ap_ensamble": unc.ap_ensamble.mean(), "ap_una_semilla": unc.ap_s0.mean()})

# %% [markdown]
# ## 5.8 Explicabilidad
# * Grad-CAM de U-Net++ sobre el cuello de botella (X40): ¿dónde mira el modelo para decidir "célula"?
# * Embeddings del codificador de SAM: ¿separa los tipos celulares (información que el prompt no usa)?
# * Hipótesis causal verificable sobre los fallos: *las fallas de U-Net++ se explican por la densidad (fusiones) más
#   que por el tipo celular*. Se contrasta con una regresión de 1 − AP sobre cobertura y tipo celular.

# %%
from dl03.seg_models.unetpp import grad_cam  # noqa: E402

cams = test_r[:3]
fig, ax = plt.subplots(2, len(cams), figsize=(5 * len(cams), 7))
ax = np.atleast_2d(ax)
for j, iid in enumerate(cams):
    x = np.clip(sd.normalize(se.native_image(images, iid)), 0, 1)
    gm = sd.Geometry.for_resolution(x.shape, RESM)
    cam = grad_cam(umodel, gm.to_model(x))
    ax[0, j].imshow(gm.to_model(x), cmap="gray")
    ax[0, j].set_title(f"{iid} · {meta.loc[iid, 'cell_type']}", fontsize=9)
    ax[1, j].imshow(gm.to_model(x), cmap="gray")
    ax[1, j].imshow(cam, cmap="jet", alpha=0.45)
    ax[1, j].set_title("Grad-CAM (X40 → probabilidad de célula)", fontsize=9)
    for a in ax[:, j]:
        a.axis("off")
savefig(cfg, "05_gradcam")
plt.show()

# %%
import torch  # noqa: E402

from dl03.seg_models import sam_wrap as sw  # noqa: E402

sam = sw.load_sam(cfg.seg["sam"]["model_type"], cfg.out_dir / "cache_sam", PRE)
emb = []
ids_e = splits.test if not cfg.is_smoke else splits.test[:6]
from segment_anything import SamPredictor  # noqa: E402

pred_ = SamPredictor(sam)
for iid in ids_e:
    pred_.set_image(sw.to_rgb8(np.clip(sd.normalize(se.native_image(images, iid)), 0, 1)))
    emb.append(pred_.get_image_embedding().mean((2, 3)).cpu().numpy()[0])
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import cross_val_score  # noqa: E402

E = np.array(emb)
yct = meta.loc[ids_e, "cell_type"].values
Z = PCA(2).fit_transform(E)
acc = cross_val_score(LogisticRegression(max_iter=2000), E, yct, cv=min(5, pd.Series(yct).value_counts().min())).mean() \
    if pd.Series(yct).value_counts().min() >= 2 else np.nan
fig, ax = plt.subplots(figsize=(5.5, 4))
for k, ct in enumerate(sorted(set(yct))):
    ax.scatter(Z[yct == ct, 0], Z[yct == ct, 1], s=12, color=PALETTE[k], label=ct)
ax.legend()
ax.set_title(f"Embeddings de SAM (PCA) · exactitud lineal tipo celular = {acc:.2f}")
savefig(cfg, "05_sam_embeddings")
plt.show()

# %%
img4 = pd.read_csv(cfg.out_dir / "seg" / "metrics_test.csv")
u = img4[img4.model == "unetpp"].assign(error=lambda d: 1 - d.ap,
                                        coverage=lambda d: d.id.map(meta["coverage"]))
ols = smf.ols("error ~ coverage + C(cell_type)", u).fit()
merge_r = sst.spearmanr(u.coverage, u["err_fusionada"] / u[[c for c in u if c.startswith("err_")]].sum(1).replace(0, 1))
print(ols.summary().tables[1])
print(f"ρ(cobertura, fracción de fusiones) = {merge_r.statistic:.3f} (p = {merge_r.pvalue:.3g})")
save_json(cfg, "seg_hipotesis_fallos", {"coef": ols.params.to_dict(), "p": ols.pvalues.to_dict(),
                                        "rho_cov_fusiones": merge_r.statistic, "p_rho": merge_r.pvalue})

# %% [markdown]
# ## 5.9 Eficiencia: parámetros, FLOPs, memoria pico, rendimiento, latencia y CO₂; frente de Pareto

# %%
def keras_flops(model, shape):
    """FLOPs de una pasada (perfilador de TensorFlow sobre el grafo concreto para la forma ``shape`` = (H, W, C))."""
    import tensorflow as tf
    conc = tf.function(lambda x: model(x, training=False)).get_concrete_function(
        tf.TensorSpec((1,) + tuple(shape), tf.float32))
    try:
        from tensorflow.python.profiler.model_analyzer import profile
        from tensorflow.python.profiler.option_builder import ProfileOptionBuilder
        return float(profile(conc.graph, options=ProfileOptionBuilder.float_operation()).total_float_ops)
    except Exception:
        return float("nan")


def torch_flops(model, x):
    from torch.utils.flop_counter import FlopCounterMode
    model.eval()
    with FlopCounterMode(display=False) as fc, torch.no_grad():
        model(x)
    return float(fc.get_total_flops())


gm0 = sd.Geometry.for_resolution(se.native_image(images, splits.test[0]).shape, RESM)
eff = []
x01 = np.clip(sd.normalize(se.native_image(images, splits.test[0])), 0, 1)
eff.append({"modelo": "U-Net++", "parámetros": umodel.count_params(),
            "GFLOPs/img": keras_flops(umodel, gm0.padded + (1,)) / 1e9})
from dl03.seg_models import maskrcnn as mr  # noqa: E402

mrm = mr.build_maskrcnn(tuple(design["anchors_maskrcnn"]), False, design["max_detections"], *x01.shape)
mrm.load_state_dict(torch.load(se.run_dir(cfg, "models") / f"maskrcnn_s{SEEDS[0]}.pt", map_location="cpu"))
mrm.to(mr.device())
eff.append({"modelo": "Mask R-CNN", "parámetros": sum(p.numel() for p in mrm.parameters()),
            "GFLOPs/img": torch_flops(mrm, [mr._img_tensor(x01).to(mr.device())]) / 1e9})
if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()
    mr.predict_raw(mrm, x01)
    eff[-1]["memoria pico GPU (MB)"] = torch.cuda.max_memory_allocated() / 2 ** 20
eff.append({"modelo": "SAM ViT-B", "parámetros": sum(p.numel() for p in sam.parameters())})
eff = pd.DataFrame(eff).set_index("modelo")
res_tab = pd.read_csv(cfg.out_dir / "tables" / "seg_resultados_test.csv", index_col=0)
lat = res_tab["latencia (ms/img)"]
eff_all = pd.DataFrame({"AP": img4.groupby("model")["ap"].mean()}).rename(index={
    "classical": "Baseline clásico", "maskrcnn": "Mask R-CNN (R50-FPN)", "hovernet": "Estilo Hover-Net",
    "unetpp": "U-Net++ + watershed", "cellpose_ft": "Cellpose ajustado", "cellpose_zero": "Cellpose cyto2 cero-shot",
    "sam_auto": "SAM cero-shot", "sam_boxes": "SAM + cajas", "sam_ft": "SAM decod. ajustado"}).join(lat)
eff_all["img/s"] = 1000 / eff_all["latencia (ms/img)"]
pareto = []
for m_, r_ in eff_all.sort_values("latencia (ms/img)").iterrows():
    if not pareto or r_["AP"] > eff_all.loc[pareto[-1], "AP"]:
        pareto.append(m_)
eff_all["en frente de Pareto"] = eff_all.index.isin(pareto)
fig, ax = plt.subplots(figsize=(7, 4.2))
ax.scatter(eff_all["latencia (ms/img)"], eff_all["AP"], s=40)
for m_, r_ in eff_all.iterrows():
    ax.annotate(m_, (r_["latencia (ms/img)"], r_["AP"]), fontsize=7, xytext=(4, 2), textcoords="offset points")
pp = eff_all.loc[pareto].sort_values("latencia (ms/img)")
ax.plot(pp["latencia (ms/img)"], pp["AP"], "--", color=PALETTE[1])
ax.set(xscale="log", xlabel="latencia por imagen (ms, incluye post-proceso)", ylabel="AP test",
       title="Frente de Pareto AP vs latencia")
savefig(cfg, "05_pareto")
plt.show()
co2 = tracker.stop() if tracker is not None else None
save_json(cfg, "co2_cap05", {"kg_co2eq": co2})
save_table(cfg, "seg_eficiencia", eff_all)
save_table(cfg, "seg_flops", eff)
display(eff.round(2))  # noqa: F821
eff_all.round(3)
