"""Configuración, semillas y reporte de entorno (adaptado de «Utilidades de código reutilizables», 4.8.24)."""
from __future__ import annotations

import os
import platform
import random
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ExperimentConfig:
    profile: str
    seed: int
    alpha: float
    n_boot: int
    val_frac: float
    test_frac: float
    k_folds: int
    data_dir: Path
    out_dir: Path
    seg: dict[str, Any] = field(default_factory=dict)
    nlp: dict[str, Any] = field(default_factory=dict)

    @property
    def is_smoke(self) -> bool:
        return self.profile == "smoke"

    def out(self, *parts: str) -> Path:
        p = self.out_dir.joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p


def load_config(profile: str | None = None, path: Path | None = None) -> ExperimentConfig:
    profile = profile or os.environ.get("DL03_PROFILE", "full")
    raw = yaml.safe_load((path or ROOT / "configs" / "project.yaml").read_text(encoding="utf-8"))
    if profile not in raw["profiles"]:
        raise ValueError(f"Perfil desconocido: {profile}")
    c, p = raw["common"], raw["profiles"][profile]
    cfg = ExperimentConfig(profile=profile, seed=c["seed"], alpha=c["alpha"], n_boot=c["n_boot"],
                           val_frac=c["val_frac"], test_frac=c["test_frac"], k_folds=c["k_folds"],
                           data_dir=ROOT / p["data_dir"], out_dir=ROOT / p["out_dir"], seg=p["seg"], nlp=p["nlp"])
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    return cfg


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    """Fija semillas en Python, NumPy, TensorFlow y PyTorch (si están instalados).

    El determinismo total en GPU no está garantizado (algunas operaciones de cuDNN no tienen versión
    determinista); por eso el protocolo reporta varianza entre semillas en lugar de asumir repetibilidad bit a bit.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.keras.utils.set_random_seed(seed)
        if deterministic:
            try:
                tf.config.experimental.enable_op_determinism()
            except Exception:  # pragma: no cover - depende de la versión
                pass
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.use_deterministic_algorithms(True, warn_only=True)
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


PKGS = ("numpy", "pandas", "scipy", "scikit-learn", "tensorflow", "keras", "torch", "torchvision", "transformers",
        "cellpose", "segment-anything", "xgboost", "fasttext", "gensim", "optuna", "albumentations", "statsmodels")


def environment_report(pkgs: tuple[str, ...] = PKGS) -> dict[str, Any]:
    """Versiones de software y hardware para el reporte de reproducibilidad."""
    rep: dict[str, Any] = {"python": platform.python_version(), "platform": platform.platform()}
    for p in pkgs:
        try:
            rep[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            rep[p] = None
    try:
        import torch
        rep["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        rep["gpu"] = None
    return rep
