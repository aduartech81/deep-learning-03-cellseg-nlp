"""Orquestación NLP: preparación, HPO con presupuesto idéntico, validación anidada y corridas finales por semilla.

Reanudable: cada resultado se guarda en ``results/nlp`` y se omite si ya existe.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import metrics as skm
from sklearn.model_selection import StratifiedGroupKFold

from . import nlp_data as nd
from .calibration import brier_score, expected_calibration_error
from .config import ExperimentConfig
from .nlp_models import fit_model, fit_transformer, space

LIGHT = ["tfidf_lr", "nbsvm", "tfidf_xgb", "fasttext", "cnn1d", "bilstm"]


def out(cfg: ExperimentConfig, *parts) -> Path:
    p = cfg.out_dir.joinpath("nlp", *parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def prepare(cfg: ExperimentConfig):
    df = nd.load_resumes(cfg)
    classes = sorted(df["label"].unique())
    y = df["label"].map({c: i for i, c in enumerate(classes)}).values
    f = out(cfg, "splits.json")
    if f.exists():
        d = json.loads(f.read_text())
        groups = np.array(d["groups"])
        sp = d["splits"]
        sp["folds"] = [tuple(x) for x in sp["folds"]]
    else:
        groups, pairs = nd.find_duplicates(df["text"].tolist())
        sp = nd.make_splits(df, groups, cfg)
        f.write_text(json.dumps({"groups": groups.tolist(), "splits": sp,
                                 "pairs": [(int(a), int(b), float(j), bool(e)) for a, b, j, e in pairs]}))
    nd.assert_no_leak(groups, sp)
    return df, y, classes, groups, sp


def nlp_metrics(y, P, K: int | None = None) -> dict[str, float]:
    y = np.asarray(y)
    P = np.asarray(P, float)
    K = K or P.shape[1]
    pred = P.argmax(1)
    r = dict(accuracy=skm.accuracy_score(y, pred), balanced_accuracy=skm.balanced_accuracy_score(y, pred),
             mcc=skm.matthews_corrcoef(y, pred), log_loss=skm.log_loss(y, np.clip(P, 1e-9, 1), labels=range(K)),
             brier=brier_score(P, y), ece=expected_calibration_error(P, y))
    for avg in ("macro", "micro", "weighted"):
        r[f"precision_{avg}"] = skm.precision_score(y, pred, average=avg, zero_division=0)
        r[f"recall_{avg}"] = skm.recall_score(y, pred, average=avg, zero_division=0)
        r[f"f1_{avg}"] = skm.f1_score(y, pred, average=avg, zero_division=0)
    present = np.unique(y)
    Y = np.eye(K)[y][:, present]
    try:
        r["roc_auc_ovr"] = skm.roc_auc_score(Y, P[:, present], average="macro")
        r["pr_auc_ovr"] = skm.average_precision_score(Y, P[:, present], average="macro")
    except ValueError:
        r["roc_auc_ovr"] = r["pr_auc_ovr"] = np.nan
    return r


def f1m(y, P) -> float:
    return float(skm.f1_score(y, np.asarray(P).argmax(1), average="macro", zero_division=0))


# --------------------------------------------------------------------- HPO ---
def hpo(cfg, name, X, y, tr, va, K, n_trials, seed=0, cap_epochs=50, tag="main"):
    """Optuna TPE + MedianPruner, objetivo F1 macro en validación. Mismo n_trials para todos los modelos."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    f = out(cfg, "hpo", f"{tag}_{name}.json")
    if f.exists():
        return json.loads(f.read_text())

    def obj(trial):
        p = space(name, trial, cap_epochs)
        kw = {"trial": trial} if name in ("bilstm", "cnn1d") else {}
        m = fit_model(name, X[tr], y[tr], X[va], y[va], p, seed, K, **kw)
        return f1m(y[va], m.predict_proba(X[va]))
    st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed),
                             pruner=optuna.pruners.MedianPruner(n_warmup_steps=3))
    t0 = time.time()
    st.optimize(obj, n_trials=n_trials, catch=(ValueError, RuntimeError))
    trials = st.trials_dataframe()
    res = {"best_params": st.best_params, "best_value": st.best_value, "seconds": time.time() - t0,
           "trials": trials.drop(columns=[c for c in trials.columns if "datetime" in c or c == "duration"],
                                 errors="ignore").to_dict("list")}
    try:
        res["importance"] = {k: float(v) for k, v in optuna.importance.get_param_importances(st).items()}
    except Exception:
        res["importance"] = {}
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(res, default=float))
    return res


def full_params(name, best, cap_epochs=50):
    """Completa los params de Optuna con los valores fijos del espacio (p. ej. tope de épocas)."""
    p = dict(best)
    if name in ("bilstm", "cnn1d") and "max_epochs" in p:
        p["max_epochs"] = min(cap_epochs, p["max_epochs"])
    if name == "cnn1d":
        p.setdefault("max_epochs", min(cap_epochs, 30))
    return p


# ------------------------------------------------------------ anidada ------
def nested_cv(cfg, name, X, y, groups, sp, K):
    """Validación anidada: 5 folds externos (sobre train+val; test sellado fuera); dentro de cada fold, HPO con un
    único split interno 85/15 agrupado (desviación declarada: no k-fold interno, por presupuesto)."""
    f = out(cfg, "nested", f"{name}.json")
    if f.exists():
        return pd.DataFrame(json.loads(f.read_text()))
    rows = []
    for k, (otr, ote) in enumerate(sp["folds"][: cfg.nlp["nested_outer"]]):
        otr = np.array(otr)
        inner = StratifiedGroupKFold(round(1 / 0.15), shuffle=True, random_state=cfg.seed + 10 + k)
        a, b = next(inner.split(otr, y[otr], groups[otr]))
        res = hpo(cfg, name, X, y, otr[a], otr[b], K, cfg.nlp.get("nested_trials", cfg.nlp["hpo_trials"]), seed=k,
                  cap_epochs=cfg.nlp["bilstm_max_epochs_cap"], tag=f"nested{k}")
        p = full_params(name, res["best_params"], cfg.nlp["bilstm_max_epochs_cap"])
        m = fit_model(name, X[otr[a]], y[otr[a]], X[otr[b]], y[otr[b]], p, k, K)
        P = m.predict_proba(X[np.array(ote)])
        rows.append(dict(model=name, fold=k, inner_best_val_f1=res["best_value"], outer_f1=f1m(y[ote], P),
                         outer_acc=float((P.argmax(1) == y[ote]).mean()), n_train=len(a), n_test=len(ote),
                         params=json.dumps(p)))
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(rows))
    return pd.DataFrame(rows)


# ------------------------------------------------------------- finales ------
def latency_ms(model, texts, n: int = 50) -> dict[str, float]:
    t = []
    for s in texts[:n]:
        t0 = time.perf_counter()
        model.predict_proba([s])
        t.append((time.perf_counter() - t0) * 1e3)
    return {"lat_p50_ms": float(np.percentile(t, 50)), "lat_p95_ms": float(np.percentile(t, 95))}


def final_runs(cfg, name, params, X, y, sp, K, seeds, save_model: bool = False):
    """Entrena con train (validación para early stopping), guarda probabilidades de val y test por semilla."""
    tr, va, te = map(np.array, (sp["train"], sp["val"], sp["test"]))
    rows = []
    for s in seeds:
        fp = out(cfg, "probs", f"{name}_s{s}.npz")
        meta = out(cfg, "probs", f"{name}_s{s}.json")
        need_model = save_model and s == seeds[0] and not any(out(cfg, "models", name, "x").parent.glob("*"))
        if fp.exists() and meta.exists() and not need_model:
            rows.append(json.loads(meta.read_text()))
            continue
        t0 = time.time()
        if name == "transformer":
            m = fit_transformer(X[tr], y[tr], X[va], y[va], params, s, K)
        else:
            m = fit_model(name, X[tr], y[tr], X[va], y[va], params, s, K)
        secs = time.time() - t0
        Pva, Pte = m.predict_proba(X[va]), m.predict_proba(X[te])
        extra = {}
        if name == "transformer":
            extra["logits_val"], extra["logits_test"] = m.logits(X[va]), m.logits(X[te])
        np.savez_compressed(fp, val=Pva, test=Pte, **extra)
        info = dict(model=name, seed=s, train_seconds=secs, val_f1=f1m(y[va], Pva), test_f1=f1m(y[te], Pte),
                    n_params=_n_params(m), **latency_ms(m, list(X[te])))
        meta.write_text(json.dumps(info, default=float))
        rows.append(info)
        if save_model and s == seeds[0]:
            _save(cfg, name, m)
    return pd.DataFrame(rows)


def _n_params(m) -> float:
    try:
        if hasattr(m, "model") and hasattr(m.model, "count_params"):
            return float(m.model.count_params())
        if hasattr(m, "model") and hasattr(m.model, "parameters"):
            return float(sum(p.numel() for p in m.model.parameters()))
        if hasattr(m, "vec"):
            clf = m.clf
            if isinstance(clf, list):
                return float(sum(c.coef_.size for c in clf))
            if hasattr(clf, "coef_"):
                return float(clf.coef_.size + clf.intercept_.size)
            if hasattr(clf, "get_booster"):
                return float(len(clf.get_booster().get_dump()))      # nº de árboles (no comparable)
        if hasattr(m, "model") and hasattr(m.model, "get_input_matrix"):
            return float(m.model.get_input_matrix().size + m.model.get_output_matrix().size)
    except Exception:
        pass
    return float("nan")


def _save(cfg, name, m):
    import joblib
    d = out(cfg, "models", name)
    d.mkdir(parents=True, exist_ok=True)
    if hasattr(m, "vec"):
        joblib.dump(m, d / "model.joblib")
    elif hasattr(m, "model") and hasattr(m.model, "save") and hasattr(m, "vectorizer"):
        m.model.save(d / "model.keras")
    elif name == "transformer":
        m.model.save_pretrained(d)
        m.tok.save_pretrained(d)


def load_probs(cfg, name, seed, part="test"):
    return np.load(out(cfg, "probs", f"{name}_s{seed}.npz"))[part]
