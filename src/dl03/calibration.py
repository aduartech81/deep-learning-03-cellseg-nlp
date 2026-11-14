"""Calibración, incertidumbre y predicción conforme (4.8.24, 4.8.31.7, 4.8.46.3)."""
from __future__ import annotations

import numpy as np


def softmax(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, float)
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def expected_calibration_error(probs, y, n_bins: int = 15) -> float:
    """ECE de la clase predicha (top-label), bins de igual ancho."""
    probs, y = np.asarray(probs), np.asarray(y)
    conf, acc = probs.max(1), (probs.argmax(1) == y).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(ece)


def binary_ece(p, y, n_bins: int = 15) -> float:
    """ECE para probabilidades binarias (p. ej., píxel = célula)."""
    p = np.asarray(p, float).ravel()
    return expected_calibration_error(np.stack([1 - p, p], 1), np.asarray(y).ravel().astype(int), n_bins)


def brier_score(probs, y) -> float:
    probs = np.asarray(probs, float)
    onehot = np.eye(probs.shape[1])[np.asarray(y)]
    return float(((probs - onehot) ** 2).sum(1).mean())


def nll(probs, y) -> float:
    probs = np.asarray(probs, float)
    return float(-np.log(probs[np.arange(len(y)), np.asarray(y)] + 1e-12).mean())


def reliability_bins(probs, y, n_bins: int = 10):
    probs, y = np.asarray(probs), np.asarray(y)
    conf, acc = probs.max(1), (probs.argmax(1) == y).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        rows.append((0.5 * (lo + hi), acc[m].mean() if m.any() else np.nan, conf[m].mean() if m.any() else np.nan,
                     int(m.sum())))
    return np.array(rows, dtype=float)


def fit_temperature(logits, y) -> float:
    """Temperature scaling (Guo et al., 2017): minimiza la NLL en validación."""
    from scipy.optimize import minimize_scalar
    logits, y = np.asarray(logits, float), np.asarray(y)

    def f(t):
        p = softmax(logits / t)
        return -np.log(p[np.arange(len(y)), y] + 1e-12).mean()
    return float(minimize_scalar(f, bounds=(0.05, 10), method="bounded").x)


def conformal_lac(cal_probs, cal_y, test_probs, alpha: float = 0.1):
    """Predicción conforme split con puntaje LAC (1 − p_y). Devuelve conjuntos (bool n×K) y el umbral q̂."""
    cal_probs, test_probs = np.asarray(cal_probs), np.asarray(test_probs)
    n = len(cal_y)
    scores = 1 - cal_probs[np.arange(n), np.asarray(cal_y)]
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    qhat = float(np.quantile(scores, level, method="higher"))
    return test_probs >= 1 - qhat, qhat


def risk_coverage(confidence, correct):
    """Curva riesgo–cobertura para clasificación selectiva (orden decreciente de confianza)."""
    order = np.argsort(-np.asarray(confidence))
    c = np.asarray(correct, float)[order]
    k = np.arange(1, len(c) + 1)
    coverage = k / len(c)
    risk = 1 - np.cumsum(c) / k
    aurc = float(np.trapezoid(risk, coverage)) if hasattr(np, "trapezoid") else float(np.trapz(risk, coverage))
    return coverage, risk, aurc
