"""Detección de deriva (4.8.24 y 4.8.49.6): PSI, KS, MMD y simulación de flujos con deriva controlada."""
from __future__ import annotations

import numpy as np
from scipy import stats


def psi(ref, cur, bins: int = 10, eps: float = 1e-6) -> float:
    """Population Stability Index con cortes por cuantiles de la referencia."""
    ref, cur = np.asarray(ref, float), np.asarray(cur, float)
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.clip(np.histogram(ref, edges)[0] / len(ref), eps, None)
    c = np.clip(np.histogram(cur, edges)[0] / len(cur), eps, None)
    return float(np.sum((c - r) * np.log(c / r)))


def drift_report(ref, cur) -> dict[str, float]:
    ks = stats.ks_2samp(ref, cur)
    return dict(psi=psi(ref, cur), ks_stat=float(ks.statistic), ks_p=float(ks.pvalue))


def _rbf(x: np.ndarray, y: np.ndarray, gamma: float) -> np.ndarray:
    d = (x ** 2).sum(1)[:, None] + (y ** 2).sum(1)[None, :] - 2 * x @ y.T
    return np.exp(-gamma * np.maximum(d, 0))


def mmd2(x, y, gamma: float | None = None) -> float:
    """MMD² insesgado con núcleo RBF; gamma por la heurística de la mediana si no se da."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.ndim == 1:
        x, y = x[:, None], y[:, None]
    if gamma is None:
        z = np.vstack([x, y])
        sub = z[np.random.default_rng(0).choice(len(z), min(len(z), 500), replace=False)]
        d = ((sub[:, None, :] - sub[None, :, :]) ** 2).sum(-1)
        med = float(np.median(d[d > 0])) if (d > 0).any() else 1.0
        gamma = 1.0 / med
    g = float(gamma)
    kxx, kyy, kxy = _rbf(x, x, g), _rbf(y, y, g), _rbf(x, y, g)
    n, m = len(x), len(y)
    np.fill_diagonal(kxx, 0)
    np.fill_diagonal(kyy, 0)
    return float(kxx.sum() / (n * (n - 1)) + kyy.sum() / (m * (m - 1)) - 2 * kxy.mean())


def mmd_permutation_test(x, y, n_perm: int = 200, seed: int = 0) -> tuple[float, float]:
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.ndim == 1:
        x, y = x[:, None], y[:, None]
    obs = mmd2(x, y)
    z = np.vstack([x, y])
    rng = np.random.default_rng(seed)
    cnt = 0
    for _ in range(n_perm):
        idx = rng.permutation(len(z))
        cnt += mmd2(z[idx[:len(x)]], z[idx[len(x):]]) >= obs
    return obs, (cnt + 1) / (n_perm + 1)


def detection_roc(stat_no_drift, stat_drift):
    """AUC del detector: separa ventanas sin deriva de ventanas con deriva según el estadístico."""
    from sklearn.metrics import roc_auc_score
    y = np.r_[np.zeros(len(stat_no_drift)), np.ones(len(stat_drift))]
    return float(roc_auc_score(y, np.r_[stat_no_drift, stat_drift]))
