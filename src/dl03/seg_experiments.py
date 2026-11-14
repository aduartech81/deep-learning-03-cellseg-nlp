"""Orquestación de los experimentos de segmentación (reanudable: cada corrida deja su artefacto y se omite si
ya existe). Todos los modelos producen mapas de etiquetas a resolución nativa y se evalúan con el mismo código.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from . import seg_metrics as sm
from .config import ExperimentConfig, set_global_seed
from .seg_data import Geometry, build_index, load_image, load_label, normalize

# ------------------------------------------------------------- utilidades ---
_IMG_CACHE: dict[str, np.ndarray] = {}


def native_image(images: pd.DataFrame, iid: str) -> np.ndarray:
    if iid not in _IMG_CACHE:
        _IMG_CACHE[iid] = load_image(images.set_index("id").loc[iid, "path"])
    return _IMG_CACHE[iid]


def native01(images, iid, mode: str = "percentile") -> np.ndarray:
    x = normalize(native_image(images, iid), mode)
    return np.clip(x, 0, 1) if mode != "train" else x


def free_memory() -> None:
    """Libera memoria entre modelos (TensorFlow, PyTorch y la caché de imágenes) para no agotar la RAM."""
    import gc
    import sys
    if "keras" in sys.modules:
        import keras
        keras.backend.clear_session()
    if "torch" in sys.modules:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if len(_IMG_CACHE) > 300:
        _IMG_CACHE.clear()
    gc.collect()


def run_dir(cfg: ExperimentConfig, *parts: str) -> Path:
    p = cfg.out_dir.joinpath("seg", *parts)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_preds(path: Path, preds: dict[str, tuple[np.ndarray, np.ndarray | None]]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for iid, (lbl, prob) in preds.items():
        d = {"lbl": lbl.astype(np.int32)}
        if prob is not None:
            d["prob"] = prob.astype(np.float16)
        np.savez_compressed(path / f"{iid}.npz", **d)


def load_preds(path: Path, ids) -> dict[str, tuple[np.ndarray, np.ndarray | None]]:
    out = {}
    for iid in ids:
        z = np.load(path / f"{iid}.npz")
        out[iid] = (z["lbl"], z["prob"].astype(np.float32) if "prob" in z else None)
    return out


def preds_exist(path: Path, ids) -> bool:
    return path.exists() and all((path / f"{i}.npz").exists() for i in ids)


def evaluate(cfg, images: pd.DataFrame, preds: dict, model: str, seed: int | str = 0, extra: dict | None = None,
             with_prob: bool = True) -> pd.DataFrame:
    meta = images.set_index("id")
    rows = []
    for iid, (lbl, prob) in preds.items():
        gt = load_label(cfg, iid)
        r = sm.image_metrics(gt, lbl, prob if with_prob else None)
        r.update(id=iid, model=model, seed=seed, cell_type=meta.loc[iid, "cell_type"],
                 coverage=meta.loc[iid, "coverage"], n_inst=meta.loc[iid, "n_inst"])
        if extra:
            r.update(extra)
        rows.append(r)
    return pd.DataFrame(rows)


def add_strata(df: pd.DataFrame, images: pd.DataFrame, instances: pd.DataFrame) -> pd.DataFrame:
    """Terciles de densidad (cobertura) y de tamaño (área mediana por imagen) definidos con TODO el conjunto."""
    med_area = instances.groupby("id")["area"].median().rename("med_area")
    im = images.set_index("id").join(med_area)
    dens = pd.qcut(im["coverage"], 3, labels=["baja", "media", "alta"])
    size = pd.qcut(im["med_area"], 3, labels=["pequeñas", "medianas", "grandes"])
    return df.assign(densidad=df["id"].map(dens).astype(str), tamano=df["id"].map(size).astype(str))


def mean_ap(preds: dict, cfg) -> float:
    return float(np.mean([sm.sartorius_score(load_label(cfg, i), p[0]) for i, p in preds.items()]))


# ------------------------------------------------- post-proceso con Optuna ---
def tune(objective_ids, build: Callable[[dict, str], np.ndarray], space: Callable, cfg, n_trials: int,
         seed: int = 0) -> tuple[dict, float, pd.DataFrame]:
    """Optuna (TPE) sobre los ids de VALIDACIÓN. ``build(params, iid) -> etiquetas``. Devuelve mejores params."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    gts = {i: load_label(cfg, i) for i in objective_ids}

    def obj(trial):
        p = space(trial)
        return float(np.mean([sm.sartorius_score(gts[i], build(p, i)) for i in objective_ids]))
    st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    st.optimize(obj, n_trials=n_trials)
    return st.best_params, st.best_value, st.trials_dataframe()


# ================================================================= baseline ===
def classical_space(trial):
    return dict(method=trial.suggest_categorical("method", ["otsu", "adaptive"]),
                block=trial.suggest_categorical("block", [31, 51, 101]),
                h=trial.suggest_float("h", 0.5, 4.0), min_size=trial.suggest_int("min_size", 5, 80),
                sigma=trial.suggest_float("sigma", 0.5, 2.5))


def run_classical(cfg, images, splits, n_trials):
    from .seg_models.postproc import classical_segment
    out = run_dir(cfg, "preds", "classical_s0")
    params_f = run_dir(cfg, "params") / "classical.json"
    if not params_f.exists():
        best, val, _ = tune(splits.val, lambda p, i: classical_segment(native_image(images, i), **p)[0],
                            classical_space, cfg, n_trials)
        params_f.write_text(json.dumps({"params": best, "val_ap": val}))
    best = json.loads(params_f.read_text())["params"]
    if not preds_exist(out, splits.test):
        t0 = time.time()
        save_preds(out, {i: classical_segment(native_image(images, i), **best) for i in splits.test})
        (out / "_time.json").write_text(json.dumps({"seconds_per_image": (time.time() - t0) / len(splits.test)}))
    return best


# ================================================================= U-Net++ ===
def keras_arrays(cfg, images, ids, res, norm="percentile", frac: float = 1.0, seed: int = 0):
    """Imágenes y etiquetas a la geometría del modelo. ``frac``: submuestreo estratificado del train."""
    if frac < 1.0:
        sub = images.set_index("id").loc[ids].reset_index()
        ids = list(sub.groupby("cell_type", group_keys=False).apply(
            lambda g: g.sample(max(1, int(round(frac * len(g)))), random_state=seed))["id"])
    X, Y, G = [], [], []
    for iid in ids:
        img, lbl = native01(images, iid, norm), load_label(cfg, iid)
        if res == "tiling":
            X.append(img.astype(np.float32))
            Y.append(lbl)
            G.append(Geometry(img.shape, 1.0, img.shape))
            continue
        g = Geometry.for_resolution(img.shape, res)
        X.append(g.to_model(img).astype(np.float32))
        Y.append(g.to_model(lbl, order=0).astype(np.int32))
        G.append(g)
    return X, Y, G, ids


def train_unetpp(cfg, images, train_ids, val_ids, seed: int, tag: str, res=512, loss="ce_dice", aug="geom_photo",
                 frac: float = 1.0, hp: dict | None = None, norm: str = "percentile", epochs: int | None = None,
                 trial=None):
    """Entrena U-Net++ y devuelve la ruta del modelo (omite si ya existe)."""
    import keras

    from .seg_models import keras_common as kc
    from .seg_models.unetpp import build_unetpp
    path = run_dir(cfg, "models") / f"{tag}.keras"
    if path.exists() and trial is None:
        return path
    hp = {"lr": 1e-3, "batch": cfg.seg["unetpp"]["batch"], "optimizer": "adamw", "schedule": "warmup_cosine",
          **(hp or {})}
    set_global_seed(seed, deterministic=False)
    keras.backend.clear_session()
    kc.maybe_mixed_precision()
    Xtr, Ytr, _, _ = keras_arrays(cfg, images, train_ids, res, norm, frac, seed)
    Xva, Yva, _, _ = keras_arrays(cfg, images, val_ids, res, norm)
    crop = tile_size(Xtr[0].shape) if res == "tiling" else None
    ep = epochs or cfg.seg["unetpp"]["epochs"]
    model = build_unetpp(tuple(cfg.seg["unetpp"]["filters"]))
    n_out = len(model.outputs)
    tr = kc.SegSequence(Xtr, Ytr, hp["batch"], kc.onehot3, kc.build_augment(aug), True, seed, crop, n_out)
    va = kc.SegSequence(Xva, Yva, hp["batch"], kc.onehot3, None, False, seed, crop, n_out)
    sched = kc.lr_schedule(hp["schedule"], hp["lr"], ep * len(tr))
    model.compile(kc.make_optimizer(hp["optimizer"], sched), loss=[kc.LOSSES[loss]] * n_out)
    cbs = [keras.callbacks.EarlyStopping("val_loss", patience=cfg.seg["unetpp"]["patience"],
                                         restore_best_weights=True),
           keras.callbacks.CSVLogger(str(run_dir(cfg, "logs") / f"{tag}.csv"))]
    if trial is not None:
        from .nlp_models import _OptunaPrune
        cbs.append(_OptunaPrune(trial))
    t0 = time.time()
    model.fit(tr, validation_data=va, epochs=ep, callbacks=cbs, verbose=0)
    meta = {"tag": tag, "seed": seed, "res": res, "loss": loss, "aug": aug, "frac": frac, "hp": hp, "norm": norm,
            "train_seconds": time.time() - t0, "n_train": len(Xtr), "params": int(model.count_params())}
    (run_dir(cfg, "models") / f"{tag}.json").write_text(json.dumps(meta, default=str))
    if trial is None:
        model.save(path)
        return path
    return model


def load_keras(path: Path):
    import keras

    from .seg_models.unetpp import MCSpatialDropout  # noqa: F401  (registro de la capa)
    return keras.models.load_model(path, compile=False)


def unetpp_probs(model, cfg, images, ids, res, norm="percentile") -> dict[str, np.ndarray]:
    from .seg_models.unetpp import predict_probs, predict_tiled
    out = {}
    for iid in ids:
        img = native01(images, iid, norm)
        if res == "tiling":
            t = tile_size(img.shape)
            out[iid] = predict_tiled(model, img, tile=t, stride=3 * t // 4)
            continue
        g = Geometry.for_resolution(img.shape, res)
        p = predict_probs(model, g.to_model(img)[None])[0]
        out[iid] = g.to_native(p, order=1)
    return out


def tile_size(shape, tile: int = 256) -> int:
    """Tesela de 256 px (múltiplo de 32), o menor si la imagen es más pequeña."""
    return int(min(tile, (min(shape[:2]) // 32) * 32))


def unetpp_pp_space(trial):
    return dict(thr_fg=trial.suggest_float("thr_fg", 0.3, 0.7), thr_in=trial.suggest_float("thr_in", 0.3, 0.8),
                h=trial.suggest_float("h", 0.5, 4.0), min_size=trial.suggest_int("min_size", 5, 60))


def unetpp_full(cfg, images, splits, seed, tag, res=512, test_ids=None, n_trials=None, **kw):
    """Entrena, ajusta el post-proceso en validación y predice test. Devuelve (dir de predicciones, params)."""
    from .seg_models.unetpp import probs_to_instances
    test_ids = test_ids if test_ids is not None else splits.test
    pdir = run_dir(cfg, "preds", tag)
    pf = run_dir(cfg, "params") / f"{tag}.json"
    if preds_exist(pdir, test_ids) and pf.exists():
        return pdir, json.loads(pf.read_text())
    path = train_unetpp(cfg, images, kw.pop("train_ids", splits.train), splits.val, seed, tag, res=res, **kw)
    model = load_keras(path)
    norm = kw.get("norm", "percentile")
    pv = unetpp_probs(model, cfg, images, splits.val, res, norm)
    best, val, _ = tune(splits.val, lambda p, i: probs_to_instances(pv[i], p), unetpp_pp_space, cfg,
                        n_trials or cfg.seg["hpo_trials"], seed)
    pf.write_text(json.dumps({"params": best, "val_ap": val}))
    t0 = time.time()
    pt = unetpp_probs(model, cfg, images, test_ids, res, norm)
    save_preds(pdir, {i: (probs_to_instances(pt[i], best), pt[i][..., 1:].sum(-1)) for i in test_ids})
    (pdir / "_time.json").write_text(json.dumps({"seconds_per_image": (time.time() - t0) / len(test_ids)}))
    return pdir, {"params": best, "val_ap": val}


# ================================================================ Hover-Net ===
def hovernet_full(cfg, images, splits, seed, tag, pretrained=True, res=512):
    import keras

    from .seg_models import keras_common as kc
    from .seg_models.hovernet import (
        HoverSequenceAdapter,
        build_hovernet,
        hover_target,
        hv_loss,
        np_loss,
        to_resnet_input,
    )
    from .seg_models.postproc import hovernet_postproc
    pdir = run_dir(cfg, "preds", tag)
    pf = run_dir(cfg, "params") / f"{tag}.json"
    if preds_exist(pdir, splits.test) and pf.exists():
        return pdir, json.loads(pf.read_text())
    mpath = run_dir(cfg, "models") / f"{tag}.weights.h5"
    set_global_seed(seed, deterministic=False)
    keras.backend.clear_session()
    kc.maybe_mixed_precision()
    model = build_hovernet(pretrained)
    hp = cfg.seg["hovernet"]
    if not mpath.exists():
        Xtr, Ytr, _, _ = keras_arrays(cfg, images, splits.train, res)
        Xva, Yva, _, _ = keras_arrays(cfg, images, splits.val, res)
        tr = HoverSequenceAdapter(kc.SegSequence(Xtr, Ytr, hp["batch"], hover_target, kc.build_augment("geom_photo"),
                                                 True, seed))
        va = HoverSequenceAdapter(kc.SegSequence(Xva, Yva, hp["batch"], hover_target, None, False, seed))
        sched = kc.lr_schedule("warmup_cosine", 1e-4, hp["epochs"] * len(tr))
        model.compile(kc.make_optimizer("adamw", sched), loss=[np_loss, hv_loss])
        t0 = time.time()
        model.fit(tr, validation_data=va, epochs=hp["epochs"], verbose=0,
                  callbacks=[keras.callbacks.EarlyStopping("val_loss", patience=hp["patience"],
                                                           restore_best_weights=True),
                             keras.callbacks.CSVLogger(str(run_dir(cfg, "logs") / f"{tag}.csv"))])
        model.save_weights(mpath)
        (run_dir(cfg, "models") / f"{tag}.json").write_text(json.dumps(
            {"tag": tag, "seed": seed, "pretrained": pretrained, "train_seconds": time.time() - t0,
             "params": int(model.count_params())}))
    else:
        model.load_weights(mpath)

    def raw(ids):
        out = {}
        for iid in ids:
            img = native01(images, iid)
            g = Geometry.for_resolution(img.shape, res)
            p_np, hv = model.predict(to_resnet_input(g.to_model(img)[None]), verbose=0)
            out[iid] = (g.to_native(p_np[0, ..., 0]), g.to_native(hv[0]))
        return out
    rv = raw(splits.val)

    def space(t):
        return dict(thr=t.suggest_float("thr", 0.3, 0.7), k=t.suggest_float("k", 0.2, 0.7),
                    min_size=t.suggest_int("min_size", 5, 60), ksize=t.suggest_categorical("ksize", [11, 21]))
    best, val, _ = tune(splits.val, lambda p, i: hovernet_postproc(rv[i][0], rv[i][1], **p), space, cfg,
                        cfg.seg["hpo_trials"], seed)
    pf.write_text(json.dumps({"params": best, "val_ap": val}))
    t0 = time.time()
    rt = raw(splits.test)
    save_preds(pdir, {i: (hovernet_postproc(rt[i][0], rt[i][1], **best), rt[i][0]) for i in splits.test})
    (pdir / "_time.json").write_text(json.dumps({"seconds_per_image": (time.time() - t0) / len(splits.test)}))
    return pdir, {"params": best, "val_ap": val}


# ================================================================ Mask R-CNN ===
def maskrcnn_full(cfg, images, splits, seed, tag, anchors=(8, 16, 32, 64, 128), pretrained=True):
    import torch

    from .seg_models import maskrcnn as mr
    pdir = run_dir(cfg, "preds", tag)
    pf = run_dir(cfg, "params") / f"{tag}.json"
    rawdir = run_dir(cfg, "raw", tag)
    if preds_exist(pdir, splits.test) and pf.exists():
        return pdir, json.loads(pf.read_text())
    hp = cfg.seg["maskrcnn"]
    h, w = native_image(images, splits.train[0]).shape
    set_global_seed(seed, deterministic=False)
    model = mr.build_maskrcnn(anchors, pretrained, hp["max_det"], min_size=h, max_size=w)
    mpath = run_dir(cfg, "models") / f"{tag}.pt"
    if mpath.exists():
        model.load_state_dict(torch.load(mpath, map_location="cpu"))
    else:
        t0 = time.time()
        hist = mr.train_maskrcnn(model, [native01(images, i) for i in splits.train],
                                 [load_label(cfg, i) for i in splits.train], hp["epochs"], hp["batch"], seed=seed,
                                 log=run_dir(cfg, "logs") / f"{tag}.log")
        torch.save(model.state_dict(), mpath)
        (run_dir(cfg, "models") / f"{tag}.json").write_text(json.dumps(
            {"tag": tag, "seed": seed, "train_seconds": time.time() - t0, "loss": hist,
             "params": int(sum(p.numel() for p in model.parameters()))}))
    raws = {}
    t_inf = 0.0
    for iid in list(splits.val) + list(splits.test):
        f = rawdir / f"{iid}.npz"
        if not f.exists():
            t0 = time.time()
            r = mr.predict_raw(model, native01(images, iid))
            t_inf += time.time() - t0
            np.savez_compressed(f, **r)
        z = np.load(f)
        raws[iid] = {k: z[k] for k in ("boxes", "scores", "masks")}
    shape = native_image(images, splits.val[0]).shape

    def space(t):
        return dict(score_thr=t.suggest_float("score_thr", 0.2, 0.8), mask_thr=t.suggest_float("mask_thr", 0.3, 0.7),
                    min_size=t.suggest_int("min_size", 5, 60))
    best, val, _ = tune(splits.val, lambda p, i: mr.raw_to_label(raws[i], shape, **p)[0], space, cfg,
                        cfg.seg["hpo_trials"], seed)
    pf.write_text(json.dumps({"params": best, "val_ap": val}))
    save_preds(pdir, {i: mr.raw_to_label(raws[i], shape, **best) for i in splits.test})
    if t_inf:
        (pdir / "_time.json").write_text(json.dumps({"seconds_per_image": t_inf / (len(splits.val) +
                                                                                   len(splits.test))}))
    return pdir, {"params": best, "val_ap": val}


# ================================================================== Cellpose ===
def cellpose_full(cfg, images, splits, seed, tag, init: str | None = "cyto2", zero_shot: bool = False,
                  train_ids=None, test_ids=None):
    from .seg_models import cellpose_wrap as cw
    test_ids = test_ids if test_ids is not None else splits.test
    pdir = run_dir(cfg, "preds", tag)
    pf = run_dir(cfg, "params") / f"{tag}.json"
    if preds_exist(pdir, test_ids) and pf.exists():
        return pdir, json.loads(pf.read_text())
    set_global_seed(seed, deterministic=False)
    imgs = lambda ids: [native_image(images, i) for i in ids]  # noqa: E731  (cellpose normaliza internamente)
    if zero_shot:
        model = cw.generalist(init or "cyto2")
        diam_default = None
    else:
        mdir = run_dir(cfg, "models", tag)
        found = sorted(mdir.glob("**/models/*")) + sorted(mdir.glob(f"**/{tag}"))
        found = [f for f in found if f.is_file()]
        if found:
            path = str(found[-1])
        else:
            tr = train_ids if train_ids is not None else splits.train
            t0 = time.time()
            path = cw.finetune(imgs(tr), [load_label(cfg, i) for i in tr], mdir, tag, cfg.seg["cellpose"]["epochs"],
                               init=init, seed=seed)
            (run_dir(cfg, "models") / f"{tag}.json").write_text(json.dumps(
                {"tag": tag, "seed": seed, "init": init, "train_seconds": time.time() - t0, "path": path}))
        model = cw.load_finetuned(path)
        diam_default = cw.model_diameter(model)

    def space(t):
        return dict(flow_threshold=t.suggest_float("flow_threshold", 0.2, 0.8),
                    cellprob_threshold=t.suggest_float("cellprob_threshold", -2.0, 2.0),
                    diam_scale=t.suggest_float("diam_scale", 0.7, 1.4))
    cache: dict = {}

    def build(p, iid):
        d = None if diam_default is None else diam_default * p["diam_scale"]
        key = (iid, round(p["flow_threshold"], 3), round(p["cellprob_threshold"], 3),
               None if d is None else round(d, 2))
        if key not in cache:
            cache[key] = cw.predict(model, native_image(images, iid), d, p["flow_threshold"],
                                    p["cellprob_threshold"])
        return cache[key][0]
    best, val, _ = tune(splits.val, build, space, cfg, cfg.seg["hpo_trials"], seed)
    pf.write_text(json.dumps({"params": best, "val_ap": val, "diameter": diam_default}))
    t0 = time.time()
    d = None if diam_default is None else diam_default * best["diam_scale"]
    save_preds(pdir, {i: cw.predict(model, native_image(images, i), d, best["flow_threshold"],
                                    best["cellprob_threshold"]) for i in test_ids})
    (pdir / "_time.json").write_text(json.dumps({"seconds_per_image": (time.time() - t0) / len(test_ids)}))
    return pdir, {"params": best, "val_ap": val}


# ======================================================================= SAM ===
def sam_full(cfg, images, splits, mode: str, tag: str, max_area: int, boxes_tag: str | None = None, seed: int = 0,
             pretrained: bool = True):
    """mode: 'auto' (a), 'boxes' (b, cajas de Mask R-CNN) o 'finetune' (c, decodificador ajustado + cajas)."""
    import torch

    from .seg_models import sam_wrap as sw
    pdir = run_dir(cfg, "preds", tag)
    pf = run_dir(cfg, "params") / f"{tag}.json"
    if preds_exist(pdir, splits.test) and pf.exists():
        return pdir, json.loads(pf.read_text())
    set_global_seed(seed, deterministic=False)
    sam = sw.load_sam(cfg.seg["sam"]["model_type"], cfg.out_dir / "cache_sam", pretrained)
    if mode == "finetune":
        dpath = run_dir(cfg, "models") / f"{tag}_decoder.pt"
        if dpath.exists():
            sam.mask_decoder.load_state_dict(torch.load(dpath, map_location="cpu"))
        else:
            t0 = time.time()
            hist = sw.finetune_decoder(sam, [native01(images, i) for i in splits.train],
                                       [load_label(cfg, i) for i in splits.train], epochs=2 if not cfg.is_smoke else 1,
                                       seed=seed)
            torch.save(sam.mask_decoder.state_dict(), dpath)
            (run_dir(cfg, "models") / f"{tag}.json").write_text(json.dumps(
                {"tag": tag, "train_seconds": time.time() - t0, "loss": hist}))
    raw = run_dir(cfg, "raw", boxes_tag) if boxes_tag else None

    def build(p, iid):
        x = native01(images, iid)
        if mode == "auto":
            return sw.automatic(sam, x, cfg.seg["sam"]["points_per_side"], p["pred_iou_thresh"], p["stability_thresh"],
                                max_area)
        z = np.load(raw / f"{iid}.npz")
        return sw.with_boxes(sam, x, z["boxes"], z["scores"], p["score_thr"])

    def space(t):
        if mode == "auto":
            return dict(pred_iou_thresh=t.suggest_float("pred_iou_thresh", 0.6, 0.95),
                        stability_thresh=t.suggest_float("stability_thresh", 0.7, 0.97))
        return dict(score_thr=t.suggest_float("score_thr", 0.2, 0.8))
    cache: dict = {}

    def cached(p, iid):
        key = (iid,) + tuple(round(v, 3) for v in p.values())
        if key not in cache:
            cache[key] = build(p, iid)
        return cache[key][0]
    n_tr = max(2, cfg.seg["hpo_trials"] // (3 if mode == "auto" else 1))   # SAM automático es costoso: declarado
    val_ids = splits.val if mode != "auto" or cfg.is_smoke else splits.val[: max(10, len(splits.val) // 3)]
    best, val, _ = tune(val_ids, cached, space, cfg, n_tr, seed)
    pf.write_text(json.dumps({"params": best, "val_ap": val, "n_trials": n_tr, "n_val": len(val_ids)}))
    t0 = time.time()
    save_preds(pdir, {i: build(best, i) for i in splits.test})
    (pdir / "_time.json").write_text(json.dumps({"seconds_per_image": (time.time() - t0) / len(splits.test)}))
    return pdir, {"params": best, "val_ap": val}


def load_all(cfg, images):
    images_, instances = build_index(cfg)
    return images_, instances
