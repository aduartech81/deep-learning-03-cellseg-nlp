"""Inferencia estadística (4.8.22.4 y 4.8.24).

Las unidades de análisis son imágenes (segmentación) o ejemplos/folds (NLP); nunca píxeles. Diferencias con la
versión de la guía: ``compare_models`` además devuelve el delta de Cliff y la d de Cohen pareada con su
interpretación respecto del efecto mínimo relevante δ declarado en el pre-registro, y se añaden el diagrama de
diferencia crítica, el bootstrap pareado de diferencias y la verificación con modelo mixto.
"""
from __future__ import annotations

from itertools import combinations
from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats


def bootstrap_ci(x, stat: Callable = np.mean, n_boot: int = 10_000, alpha: float = 0.05,
                 seed: int = 0) -> tuple[float, float, float]:
    """IC bootstrap percentil sobre unidades de análisis. Devuelve (estimador, lo, hi). Ignora NaN."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    chunk = max(1, 2_000_000 // max(x.size, 1))
    for s in range(0, n_boot, chunk):
        e = min(n_boot, s + chunk)
        idx = rng.integers(0, x.size, size=(e - s, x.size))
        boots[s:e] = stat(x[idx], axis=1)
    q = np.asarray(np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)]), dtype=float)
    return float(stat(x)), float(q[0]), float(q[1])


def fmt_ci(est: float, lo: float, hi: float, d: int = 3) -> str:
    return f"{est:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]"


def paired_bootstrap_diff(a, b, n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0):
    """IC bootstrap de la diferencia media pareada a − b (mismas unidades)."""
    d = np.asarray(a, float) - np.asarray(b, float)
    return bootstrap_ci(d, n_boot=n_boot, alpha=alpha, seed=seed)


def cliffs_delta(a, b) -> float:
    a, b = np.asarray(a), np.asarray(b)
    return float(((a[:, None] > b[None, :]).sum() - (a[:, None] < b[None, :]).sum()) / (len(a) * len(b)))


def cliffs_label(d: float) -> str:
    """Umbrales de Romano et al. (2006)."""
    d = abs(d)
    return "despreciable" if d < 0.147 else "pequeño" if d < 0.33 else "mediano" if d < 0.474 else "grande"


def holm_adjust(pvals) -> np.ndarray:
    p = np.asarray(pvals, float)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(np.argsort(p)):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def compare_models(scores: pd.DataFrame, alpha: float = 0.05, delta: float | None = None) -> dict:
    """Friedman + Wilcoxon pareado por pares con Holm + tamaños de efecto.

    scores: filas = unidades pareadas, columnas = modelos (≥ 3). Filas con NaN se eliminan.
    """
    scores = scores.dropna()
    models = list(scores.columns)
    fr = stats.friedmanchisquare(*[scores[m] for m in models])
    rows = []
    for a, b in combinations(models, 2):
        d = scores[a] - scores[b]
        try:
            p = stats.wilcoxon(scores[a], scores[b]).pvalue
        except ValueError:
            p = 1.0
        sd = d.std(ddof=1)
        cd = cliffs_delta(scores[a], scores[b])
        row = dict(A=a, B=b, mean_diff=d.mean(), cohen_dz=(d.mean() / sd) if sd > 0 else 0.0,
                   cliffs_delta=cd, cliffs_mag=cliffs_label(cd), p_raw=p)
        if delta is not None:
            row["supera_delta"] = bool(abs(d.mean()) >= delta)
        rows.append(row)
    out = pd.DataFrame(rows)
    out["p_holm"] = holm_adjust(out["p_raw"])
    out["reject"] = out["p_holm"] < alpha
    ranks = scores.rank(axis=1, ascending=False).mean().sort_values()
    return {"friedman_stat": float(fr.statistic), "friedman_p": float(fr.pvalue), "mean_ranks": ranks,
            "pairwise": out, "n": len(scores)}


def nemenyi_cd(k: int, n: int, alpha: float = 0.05) -> float:
    """Diferencia crítica de Nemenyi (Demšar, 2006) para k modelos y n unidades."""
    q = stats.studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2)
    return float(q * np.sqrt(k * (k + 1) / (6 * n)))


def plot_cd_diagram(mean_ranks: pd.Series, n: int, pairwise: pd.DataFrame | None = None, alpha: float = 0.05,
                    ax=None, title: str = ""):
    """Diagrama de diferencia crítica: rangos medios y barras que unen modelos NO distinguibles (Holm)."""
    import matplotlib.pyplot as plt
    k = len(mean_ranks)
    cd = nemenyi_cd(k, n, alpha)
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 1.2 + 0.32 * k))
    r = mean_ranks.sort_values()
    ax.set_xlim(0.8, k + 0.2)
    ax.set_ylim(-0.5 - 0.35 * k, 1.35)
    ax.axhline(0.5, color="#555", lw=1)
    for i in range(1, k + 1):
        ax.plot([i, i], [0.45, 0.55], color="#555", lw=1)
        ax.text(i, 0.7, str(i), ha="center", fontsize=8)
    for j, (m, v) in enumerate(r.items()):
        y = -0.3 - 0.35 * j
        ax.plot([v, v], [0.5, y], color="#1f6f8b", lw=1)
        ax.plot(v, 0.5, "o", color="#1f6f8b", ms=4)
        ax.text(v + 0.05, y, f"{m} ({v:.2f})", va="center", fontsize=8)
    if pairwise is not None:
        same = pairwise[~pairwise["reject"]]
        for i, (_, row) in enumerate(same.iterrows()):
            xs = sorted([r[row["A"]], r[row["B"]]])
            ax.plot(xs, [0.25 - 0.06 * (i % 4)] * 2, color="#c0392b", lw=3, alpha=0.6)
    ax.plot([1, 1 + cd], [0.95, 0.95], color="k", lw=2)
    ax.text(1 + cd / 2, 0.98, f"CD Nemenyi = {cd:.2f}", ha="center", va="bottom", fontsize=8)
    ax.axis("off")
    ax.set_title(title + "  (barras rojas: diferencias no significativas tras Holm)", fontsize=9, pad=14)
    return ax


def corrected_resampled_ttest(diffs, n_train: int, n_test: int) -> tuple[float, float]:
    """t remuestreada corregida (Nadeau & Bengio, 2003) sobre diferencias por repetición de CV."""
    d = np.asarray(diffs, float)
    k = len(d)
    var = d.var(ddof=1)
    if var == 0:
        return 0.0, 1.0
    t = d.mean() / np.sqrt((1 / k + n_test / n_train) * var)
    return float(t), float(2 * stats.t.sf(abs(t), k - 1))


def mcnemar_exact(y_true, pred_a, pred_b) -> float:
    y_true, pred_a, pred_b = map(np.asarray, (y_true, pred_a, pred_b))
    a_ok, b_ok = pred_a == y_true, pred_b == y_true
    b01, b10 = int((a_ok & ~b_ok).sum()), int((~a_ok & b_ok).sum())
    n = b01 + b10
    return 1.0 if n == 0 else float(stats.binomtest(b01, n, 0.5).pvalue)


def mixedlm_check(long_df: pd.DataFrame, metric: str, model_col: str = "model", group_col: str = "group"):
    """Modelo mixto metric ~ modelo + (1 | grupo): ¿las diferencias sobreviven a la correlación intra-grupo?"""
    import statsmodels.formula.api as smf
    df = long_df[[metric, model_col, group_col]].dropna().rename(columns={metric: "y", model_col: "m",
                                                                          group_col: "g"})
    res = smf.mixedlm("y ~ C(m)", df, groups=df["g"]).fit(reml=True, method="lbfgs")
    tab = pd.DataFrame({"coef": res.fe_params, "p": res.pvalues.reindex(res.fe_params.index)})
    icc = float(res.cov_re.iloc[0, 0] / (res.cov_re.iloc[0, 0] + res.scale))
    return tab, icc, res


def bayes_signed_rank_rope(a, b, rope: float, n_samples: int = 20_000, seed: int = 0) -> dict[str, float]:
    """Prueba de rangos con signo bayesiana (Benavoli et al., 2017) vía Dirichlet; P(a>b), P(rope), P(b>a)."""
    d = np.asarray(a, float) - np.asarray(b, float)
    rng = np.random.default_rng(seed)
    z = np.concatenate([[0.0], d])
    w = rng.dirichlet(np.concatenate([[0.5], np.ones(len(d))]), size=n_samples)
    s = (z[:, None] + z[None, :]) / 2
    # simplificación O(n^2) por muestra: válida para n de cientos
    left = (s < -rope).astype(float)
    mid = (np.abs(s) <= rope).astype(float)
    right = (s > rope).astype(float)
    pl = np.einsum("ki,ij,kj->k", w, left, w)
    pm = np.einsum("ki,ij,kj->k", w, mid, w)
    pr = np.einsum("ki,ij,kj->k", w, right, w)
    win = np.argmax(np.stack([pl, pm, pr], 1), 1)
    return {"p_b_mejor": float((win == 0).mean()), "p_rope": float((win == 1).mean()),
            "p_a_mejor": float((win == 2).mean())}
