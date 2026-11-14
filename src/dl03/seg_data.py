"""Datos de segmentación (Sartorius Cell Instance Segmentation).

Esquema esperado de ``train.csv`` (una fila por instancia): id, annotation (RLE), width, height, cell_type,
plate_time, sample_date, sample_id, elapsed_timedelta. Imágenes en ``train/<id>.png`` (escala de grises, 520×704).
El capítulo de EDA verifica este esquema contra los archivos reales y registra cualquier discrepancia.

Toda evaluación se hace a resolución nativa: cada modelo devuelve un mapa de etiquetas del tamaño original.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from skimage import io, measure
from skimage.segmentation import find_boundaries
from skimage.transform import resize
from sklearn.model_selection import StratifiedGroupKFold

from .config import ExperimentConfig
from .rle import instances_to_label_map, rle_encode

SEG_COLUMNS = ["id", "annotation", "width", "height", "cell_type", "plate_time", "sample_date", "sample_id",
               "elapsed_timedelta"]


# ------------------------------------------------------------- descarga ----
def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def download_sartorius(cfg: ExperimentConfig) -> Path:
    """Descarga con la API de Kaggle (requiere ~/.kaggle/kaggle.json y reglas aceptadas). Solo extrae train/."""
    raw = cfg.data_dir / "raw" / "sartorius"
    if (raw / "train.csv").exists() and (raw / "train").exists():
        return raw
    raw.mkdir(parents=True, exist_ok=True)
    comp = cfg.seg["competition"]
    subprocess.run(["kaggle", "competitions", "download", "-c", comp, "-p", str(raw)], check=True)
    zp = raw / f"{comp}.zip"
    with zipfile.ZipFile(zp) as z:
        members = [m for m in z.namelist() if m == "train.csv" or m.startswith("train/")]
        z.extractall(raw, members)
    manifest = {"source": f"kaggle competitions download -c {comp}", "zip_sha256": sha256(zp),
                "train_csv_sha256": sha256(raw / "train.csv"), "n_png": len(list((raw / "train").glob("*.png")))}
    (raw / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    zp.unlink()  # el zip completo (~2.4 GB) no se conserva
    return raw


# ------------------------------------------------------------ sintético ----
def make_synthetic(cfg: ExperimentConfig) -> Path:
    """Genera un conjunto con el MISMO esquema que Sartorius (solo para el perfil smoke)."""
    raw = cfg.data_dir / "raw" / "sartorius"
    if (raw / "train.csv").exists():
        return raw
    (raw / "train").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg.seed)
    h, w = 96, 128
    spec = {"shsy5y": (3, 6, 18), "astro": (6, 11, 6), "cort": (3, 5, 12)}   # radio min, max, nº células
    rows = []
    n = cfg.seg["n_images"]
    for k in range(n):
        ct = list(spec)[k % 3]
        rmin, rmax, nc = spec[ct]
        lbl = np.zeros((h, w), np.int32)
        yy, xx = np.mgrid[:h, :w]
        inst = 0
        for _ in range(nc + int(rng.integers(-2, 3))):
            cy, cx = rng.integers(rmax, h - rmax), rng.integers(rmax, w - rmax)
            a, b = rng.uniform(rmin, rmax), rng.uniform(rmin, rmax) * (0.5 if ct == "shsy5y" else 1)
            th = rng.uniform(0, np.pi)
            u = ((xx - cx) * np.cos(th) + (yy - cy) * np.sin(th)) / a
            v = (-(xx - cx) * np.sin(th) + (yy - cy) * np.cos(th)) / b
            m = (u ** 2 + v ** 2 <= 1) & (lbl == 0)
            if m.sum() < 8:
                continue
            inst += 1
            lbl[m] = inst
        img = 120 + 25 * (xx / w) + rng.normal(0, 6, (h, w))
        img[lbl > 0] += 35 + rng.normal(0, 4, (lbl > 0).sum())
        img[find_boundaries(lbl, mode="inner")] -= 25
        iid = f"syn{k:04d}"
        io.imsave(raw / "train" / f"{iid}.png", np.clip(img, 0, 255).astype(np.uint8), check_contrast=False)
        date = f"2019-0{1 + (k // 3) % 8}-1{k % 3}"
        for i in range(1, inst + 1):
            rows.append(dict(id=iid, annotation=rle_encode(lbl == i), width=w, height=h, cell_type=ct,
                             plate_time="11h30m00s", sample_date=date,
                             sample_id=f"{ct}_W{k % 5}_Vessel-{700 + (k // 3) % 8}_Ph_{k}",
                             elapsed_timedelta="0 days 11:30:00"))
    pd.DataFrame(rows, columns=SEG_COLUMNS).to_csv(raw / "train.csv", index=False)
    (raw / "MANIFEST.json").write_text(json.dumps({"source": "synthetic (perfil smoke)", "n_png": n}, indent=2))
    return raw


def get_raw(cfg: ExperimentConfig) -> Path:
    return make_synthetic(cfg) if cfg.seg["source"] == "synthetic" else download_sartorius(cfg)


# --------------------------------------------------------- índice/caché ----
def labels_dir(cfg: ExperimentConfig) -> Path:
    d = cfg.data_dir / "processed" / "labels"
    d.mkdir(parents=True, exist_ok=True)
    return d


def build_index(cfg: ExperimentConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """DataFrame por imagen y por instancia; guarda los mapas de etiquetas en caché (npz)."""
    raw = get_raw(cfg)
    proc = cfg.data_dir / "processed"
    fi, fn = proc / "images.parquet", proc / "instances.parquet"
    if fi.exists() and fn.exists():
        return pd.read_parquet(fi), pd.read_parquet(fn)
    ann = pd.read_csv(raw / "train.csv")
    ldir = labels_dir(cfg)
    img_rows, inst_rows = [], []
    for iid, g in ann.groupby("id", sort=True):
        h, w = int(g["height"].iloc[0]), int(g["width"].iloc[0])
        lbl, lost = instances_to_label_map(g["annotation"].tolist(), (h, w), return_overlap=True)
        np.savez_compressed(ldir / f"{iid}.npz", lbl=lbl)
        props = measure.regionprops(lbl)
        for p in props:
            inst_rows.append(dict(id=iid, label=p.label, area=p.area, perimeter=p.perimeter,
                                  eccentricity=p.eccentricity, solidity=p.solidity,
                                  major_axis=p.axis_major_length, cell_type=g["cell_type"].iloc[0]))
        first = g.iloc[0]
        img_rows.append(dict(id=iid, path=str(raw / "train" / f"{iid}.png"), height=h, width=w,
                             cell_type=first["cell_type"], sample_id=first["sample_id"],
                             sample_date=first["sample_date"], plate_time=first["plate_time"],
                             elapsed_timedelta=first["elapsed_timedelta"], n_annot=len(g), n_inst=len(props),
                             overlap_px=lost, coverage=float((lbl > 0).mean())))
    images = pd.DataFrame(img_rows)
    images["vessel"] = images["sample_id"].map(_vessel)
    instances = pd.DataFrame(inst_rows)
    proc.mkdir(parents=True, exist_ok=True)
    images.to_parquet(fi)
    instances.to_parquet(fn)
    return images, instances


def _vessel(s: str) -> str | None:
    m = re.search(r"Vessel-?(\d+)", str(s))
    return m.group(1) if m else None


def load_image(path: str) -> np.ndarray:
    """Imagen en float32 [0, 255], un canal (los PNG de Sartorius son 8 bits en escala de grises)."""
    img = io.imread(path)
    if img.ndim == 3:
        img = img[..., :3].mean(-1)
    return img.astype(np.float32)


def load_label(cfg: ExperimentConfig, iid: str) -> np.ndarray:
    return np.load(labels_dir(cfg) / f"{iid}.npz")["lbl"]


# ------------------------------------------------------------ particiones ---
def choose_group_column(images: pd.DataFrame, min_groups: int = 7) -> tuple[str, pd.Series]:
    """Elige la agrupación MÁS GRUESA que deja ≥ ``min_groups`` grupos por tipo celular.

    Orden de preferencia: placa (Vessel en sample_id) → fecha de muestra → sample_id. Imágenes de una misma placa
    comparten condiciones técnicas; repartirlas entre train y test inflaría el rendimiento.
    """
    for col in ("vessel", "sample_date", "sample_id"):
        if col not in images or images[col].isna().any():
            continue
        g = images["cell_type"] + "|" + images[col].astype(str)
        if g.groupby(images["cell_type"]).nunique().min() >= min_groups:
            return col, g
    return "id", images["id"]


@dataclass
class Splits:
    group_col: str
    train: list[str]
    val: list[str]
    test: list[str]
    folds: list[tuple[list[str], list[str]]]   # 5-fold sobre train+val (test sellado nunca participa)

    def to_json(self) -> str:
        return json.dumps(self.__dict__, indent=1)

    @staticmethod
    def from_json(s: str) -> "Splits":
        d = json.loads(s)
        d["folds"] = [tuple(f) for f in d["folds"]]
        return Splits(**d)


def make_splits(images: pd.DataFrame, cfg: ExperimentConfig) -> Splits:
    """70/15/15 agrupado y estratificado por tipo celular + 5-fold agrupado sobre train+val."""
    col, groups = choose_group_column(images)
    y = images["cell_type"].values
    ids = images["id"].values
    n_test = max(2, round(1 / cfg.test_frac))
    sgkf = StratifiedGroupKFold(n_splits=n_test, shuffle=True, random_state=cfg.seed)
    rest_idx, test_idx = next(sgkf.split(ids, y, groups))
    n_val = max(2, round((1 - cfg.test_frac) / cfg.val_frac))
    sgkf2 = StratifiedGroupKFold(n_splits=n_val, shuffle=True, random_state=cfg.seed + 1)
    tr_i, va_i = next(sgkf2.split(ids[rest_idx], y[rest_idx], groups.values[rest_idx]))
    train, val = ids[rest_idx][tr_i], ids[rest_idx][va_i]
    sgkf3 = StratifiedGroupKFold(n_splits=cfg.k_folds, shuffle=True, random_state=cfg.seed + 2)
    folds = [(list(ids[rest_idx][a]), list(ids[rest_idx][b]))
             for a, b in sgkf3.split(ids[rest_idx], y[rest_idx], groups.values[rest_idx])]
    return Splits(col, list(train), list(val), list(ids[test_idx]), folds)


def assert_no_group_leak(images: pd.DataFrame, splits: Splits) -> None:
    _, g = choose_group_column(images)
    gm = dict(zip(images["id"], g))
    s = {k: {gm[i] for i in getattr(splits, k)} for k in ("train", "val", "test")}
    assert not (s["train"] & s["test"]) and not (s["val"] & s["test"]) and not (s["train"] & s["val"]), \
        "fuga de grupos entre particiones"
    assert not (set(splits.train) & set(splits.test)), "fuga de imágenes"


# ------------------------------------------------------- resolución/pad ----
@dataclass
class Geometry:
    """Lleva una imagen nativa (H, W) a la entrada del modelo y devuelve las predicciones al tamaño nativo."""
    native: tuple[int, int]
    scale: float
    padded: tuple[int, int]

    @staticmethod
    def for_resolution(native: tuple[int, int], res: int | str, mult: int = 32) -> "Geometry":
        h, w = native
        scale = 1.0 if res == "native" else float(res) / w
        sh, sw = int(round(h * scale)), int(round(w * scale))
        return Geometry(native, scale, (int(np.ceil(sh / mult) * mult), int(np.ceil(sw / mult) * mult)))

    @property
    def scaled(self) -> tuple[int, int]:
        return int(round(self.native[0] * self.scale)), int(round(self.native[1] * self.scale))

    def to_model(self, arr: np.ndarray, order: int = 1) -> np.ndarray:
        sh, sw = self.scaled
        out = arr if self.scale == 1.0 else resize(arr, (sh, sw) + arr.shape[2:], order=order,
                                                   preserve_range=True, anti_aliasing=(order > 0))
        pad = [(0, self.padded[0] - sh), (0, self.padded[1] - sw)] + [(0, 0)] * (arr.ndim - 2)
        return np.pad(out, pad, mode="reflect" if order > 0 else "constant")

    def to_native(self, arr: np.ndarray, order: int = 1) -> np.ndarray:
        sh, sw = self.scaled
        crop = arr[:sh, :sw]
        if self.scale == 1.0:
            return crop
        return resize(crop, self.native + crop.shape[2:], order=order, preserve_range=True,
                      anti_aliasing=False).astype(arr.dtype)


def normalize(img: np.ndarray, mode: str = "percentile", stats: tuple[float, float] | None = None) -> np.ndarray:
    """Normalización: 'div255' (baseline), 'percentile' (1–99 por imagen) o 'train' (media/DE del train)."""
    if mode == "div255":
        return img / 255.0
    if mode == "train":
        mu, sd = stats if stats is not None else (128.0, 30.0)
        return (img - mu) / max(sd, 1e-6)
    lo, hi = np.percentile(img, [1, 99])
    return np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1.5)


# --------------------------------------------------------------- objetivos ---
def three_class_target(lbl: np.ndarray, border_px: int = 2) -> np.ndarray:
    """0 = fondo, 1 = interior de célula, 2 = borde/contacto (bordes internos engrosados)."""
    from scipy import ndimage as ndi
    b = find_boundaries(lbl, mode="inner")
    if border_px > 1:
        b = ndi.binary_dilation(b, iterations=border_px - 1) & (lbl > 0)
    t = (lbl > 0).astype(np.uint8)
    t[b] = 2
    return t


def hv_maps(lbl: np.ndarray) -> np.ndarray:
    """Mapas horizontal/vertical de Hover-Net: distancia al centroide normalizada a [-1, 1] por instancia."""
    hv = np.zeros(lbl.shape + (2,), np.float32)
    for p in measure.regionprops(lbl):
        rr, cc = p.coords[:, 0], p.coords[:, 1]
        cy, cx = p.centroid
        dy, dx = rr - cy, cc - cx
        if np.abs(dx).max() > 0:
            hv[rr, cc, 0] = np.where(dx < 0, -dx / dx.min() if dx.min() < 0 else 0, dx / max(dx.max(), 1e-6))
        if np.abs(dy).max() > 0:
            hv[rr, cc, 1] = np.where(dy < 0, -dy / dy.min() if dy.min() < 0 else 0, dy / max(dy.max(), 1e-6))
    return hv


def relabel_resized(lbl: np.ndarray, geom: Geometry) -> np.ndarray:
    """Mapa de etiquetas nativo -> tamaño del modelo (vecino más próximo, sin interpolar etiquetas)."""
    return geom.to_model(lbl.astype(np.int32), order=0).astype(np.int32)
