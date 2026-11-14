"""Estilo de figuras y utilidades de reporte (tablas con IC, guardado de figuras para el libro y el tablero)."""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import ExperimentConfig, load_config, set_global_seed
from .stats import bootstrap_ci, fmt_ci

INK = "#1b2a3a"
PALETTE = ["#1f6f8b", "#e07a1f", "#2a9d8f", "#c0392b", "#7b5ea7", "#8a8a8a", "#d4a017", "#3d5a80"]
CELL_COLORS = {"shsy5y": "#1f6f8b", "astro": "#e07a1f", "cort": "#2a9d8f"}


def setup(seed_offset: int = 0) -> ExperimentConfig:
    """Configura el notebook: perfil, semillas, estilo. Imprime el perfil para que quede en la salida."""
    cfg = load_config()
    set_global_seed(cfg.seed + seed_offset, deterministic=False)
    matplotlib.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 150, "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
        "axes.edgecolor": "#9aa5b1", "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
        "axes.spines.top": False, "axes.spines.right": False, "axes.prop_cycle": matplotlib.cycler(color=PALETTE),
        "axes.grid": True, "grid.color": "#e6ebf0", "grid.linewidth": 0.8, "legend.frameon": False})
    pd.set_option("display.max_columns", 40)
    pd.set_option("display.width", 160)
    pd.set_option("display.precision", 3)
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    print(f"Perfil: {cfg.profile} | datos: {cfg.data_dir.name} | resultados: {cfg.out_dir.name}")
    if cfg.is_smoke:
        print("AVISO: perfil smoke (datos sintéticos). Las cifras de esta ejecución NO son resultados del proyecto.")
    return cfg


def savefig(cfg: ExperimentConfig, name: str, fig=None) -> Path:
    p = cfg.out_dir / "figs" / f"{name}.png"
    p.parent.mkdir(parents=True, exist_ok=True)
    (fig or plt.gcf()).savefig(p, bbox_inches="tight")
    return p


def save_table(cfg: ExperimentConfig, name: str, df: pd.DataFrame) -> Path:
    p = cfg.out_dir / "tables" / f"{name}.csv"
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=True)
    return p


def jsonable(o):
    """Convierte claves no textuales (tuplas, numpy) y valores numpy para poder serializar en JSON."""
    if isinstance(o, dict):
        return {str(k) if not isinstance(k, (str, int, float, bool)) else k: jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if hasattr(o, "item") and getattr(o, "ndim", 1) == 0:
        return o.item()
    if hasattr(o, "tolist"):
        return o.tolist()
    return o


def save_json(cfg: ExperimentConfig, name: str, obj) -> Path:
    p = cfg.out_dir / "tables" / f"{name}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(jsonable(obj), indent=1, default=str))
    return p


def ci_table(df: pd.DataFrame, by: str, metrics: list[str], n_boot: int = 10_000, digits: int = 3) -> pd.DataFrame:
    """Tabla 'media [IC 95 %]' por grupo, bootstrap sobre las filas (unidades de análisis)."""
    out = {}
    for g, sub in df.groupby(by, sort=False):
        out[g] = {m: fmt_ci(*bootstrap_ci(sub[m].values, n_boot=n_boot), d=digits) for m in metrics}
    return pd.DataFrame(out).T


def ci_values(x, n_boot=10_000):
    return dict(zip(("est", "lo", "hi"), bootstrap_ci(np.asarray(x, float), n_boot=n_boot)))


def overlay(img: np.ndarray, lbl: np.ndarray, ax, title: str = "", alpha: float = 0.45):
    from skimage.color import label2rgb
    from skimage.segmentation import find_boundaries
    ax.imshow(img, cmap="gray")
    if lbl.max() > 0:
        ax.imshow(label2rgb(lbl, bg_label=0), alpha=alpha)
        ax.contour(find_boundaries(lbl), levels=[0.5], colors="white", linewidths=0.3)
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def error_map(gt: np.ndarray, pr: np.ndarray) -> np.ndarray:
    """RGB: FP en rojo, FN en azul, TP en gris claro (a nivel de píxel)."""
    g, p = gt > 0, pr > 0
    rgb = np.ones(gt.shape + (3,)) * 0.08
    rgb[g & p] = 0.75
    rgb[p & ~g] = (0.85, 0.15, 0.15)
    rgb[g & ~p] = (0.15, 0.35, 0.9)
    return rgb
