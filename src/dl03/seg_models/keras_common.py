"""Piezas Keras compartidas: pérdidas propias, generador con augmentation y utilidades de entrenamiento.

Implementación desde cero (exigida por 4.8.28.2): pérdidas Dice/Tversky/focal y el generador de datos.
"""
from __future__ import annotations

from typing import Callable

import keras
import numpy as np
from keras import ops

from ..seg_data import three_class_target

EPS = 1e-6


# ----------------------------------------------------------------- pérdidas ---
def _fg(y_true, y_pred, skip_bg: bool = True):
    if skip_bg:
        return y_true[..., 1:], y_pred[..., 1:]
    return y_true, y_pred


def dice_loss(y_true, y_pred):
    """1 − Dice suave multiclase (sin fondo), sumando sobre el lote: estable con primer plano escaso."""
    t, p = _fg(ops.cast(y_true, "float32"), ops.cast(y_pred, "float32"))
    axes = (0, 1, 2)
    num = 2 * ops.sum(t * p, axis=axes) + EPS
    den = ops.sum(t, axis=axes) + ops.sum(p, axis=axes) + EPS
    return 1 - ops.mean(num / den)


def tversky_loss(y_true, y_pred, alpha: float = 0.3, beta: float = 0.7):
    """Tversky (Salehi et al., 2017): beta > alpha penaliza más los falsos negativos."""
    t, p = _fg(ops.cast(y_true, "float32"), ops.cast(y_pred, "float32"))
    axes = (0, 1, 2)
    tp = ops.sum(t * p, axis=axes)
    fp = ops.sum((1 - t) * p, axis=axes)
    fn = ops.sum(t * (1 - p), axis=axes)
    return 1 - ops.mean((tp + EPS) / (tp + alpha * fp + beta * fn + EPS))


def focal_loss(y_true, y_pred, gamma: float = 2.0):
    t = ops.cast(y_true, "float32")
    p = ops.clip(ops.cast(y_pred, "float32"), EPS, 1 - EPS)
    return ops.mean(ops.sum(-t * ops.power(1 - p, gamma) * ops.log(p), axis=-1))


def ce_loss(y_true, y_pred):
    t = ops.cast(y_true, "float32")
    p = ops.clip(ops.cast(y_pred, "float32"), EPS, 1 - EPS)
    return ops.mean(ops.sum(-t * ops.log(p), axis=-1))


def ce_dice_loss(y_true, y_pred):
    return ce_loss(y_true, y_pred) + dice_loss(y_true, y_pred)


LOSSES: dict[str, Callable] = {"ce": ce_loss, "dice": dice_loss, "ce_dice": ce_dice_loss,
                               "tversky": tversky_loss, "focal": focal_loss}


# ------------------------------------------------------------ augmentation ---
def build_augment(level: str):
    """'none' | 'geom' | 'geom_photo' | 'geom_photo_elastic'. La misma transformación se aplica a imagen y
    etiquetas (vecino más próximo para la máscara: albumentations usa INTER_NEAREST en 'mask')."""
    if level == "none":
        return None
    import albumentations as A
    # sin RandomRotate90: intercambia alto y ancho y las imágenes no son cuadradas (520×704); los dos flips ya
    # cubren la rotación de 180° y Affine las rotaciones pequeñas
    t = [A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5),
         A.Affine(scale=(0.85, 1.15), rotate=(-20, 20), p=0.5, border_mode=0)]
    if "photo" in level:
        t += [A.RandomBrightnessContrast(0.15, 0.15, p=0.5), A.GaussNoise(std_range=(0.01, 0.04), p=0.3),
              A.GaussianBlur(blur_limit=(3, 5), p=0.2)]
    if "elastic" in level:
        t += [A.ElasticTransform(alpha=20, sigma=5, p=0.3, border_mode=0)]
    return A.Compose(t)


class SegSequence(keras.utils.PyDataset):
    """Lote de (imagen, objetivo). Las imágenes y etiquetas llegan ya a la geometría del modelo.

    ``target_fn`` construye el objetivo a partir del mapa de etiquetas AUMENTADO (así los mapas H/V y los bordes
    se recalculan coherentemente tras flips/rotaciones). ``crop``: recorte aleatorio (entrenamiento por teselas).
    """

    def __init__(self, imgs: list[np.ndarray], lbls: list[np.ndarray], batch: int, target_fn: Callable,
                 augment=None, shuffle: bool = True, seed: int = 0, crop: int | None = None, n_out: int = 1,
                 channels: int = 1, **kw):
        super().__init__(**kw)
        self.imgs, self.lbls, self.batch = imgs, lbls, batch
        self.target_fn, self.augment, self.shuffle, self.crop = target_fn, augment, shuffle, crop
        self.n_out, self.channels = n_out, channels
        self.rng = np.random.default_rng(seed)
        self.idx = np.arange(len(imgs))
        if shuffle:
            self.rng.shuffle(self.idx)

    def __len__(self) -> int:
        return int(np.ceil(len(self.imgs) / self.batch))

    def on_epoch_end(self) -> None:
        if self.shuffle:
            self.rng.shuffle(self.idx)

    def _one(self, i: int):
        x, lbl = self.imgs[i], self.lbls[i]
        if self.crop:
            h, w = x.shape[:2]
            c = self.crop
            y0, x0 = self.rng.integers(0, max(1, h - c + 1)), self.rng.integers(0, max(1, w - c + 1))
            x, lbl = x[y0:y0 + c, x0:x0 + c], lbl[y0:y0 + c, x0:x0 + c]
        if self.augment is not None:
            r = self.augment(image=x.astype(np.float32), mask=lbl.astype(np.int32))
            x, lbl = r["image"], r["mask"]
        return x, self.target_fn(lbl)

    def __getitem__(self, b: int):
        ids = self.idx[b * self.batch:(b + 1) * self.batch]
        xs, ys = zip(*(self._one(i) for i in ids))
        x = np.stack(xs)[..., None].astype(np.float32)
        if self.channels == 3:
            x = np.repeat(x, 3, -1)
        if isinstance(ys[0], tuple):
            y = tuple(np.stack(z).astype(np.float32) for z in zip(*ys))
        else:
            y = np.stack(ys).astype(np.float32)
        if self.n_out > 1:
            y = tuple([y] * self.n_out)
        return x, y


def onehot3(lbl: np.ndarray) -> np.ndarray:
    return np.eye(3, dtype=np.float32)[three_class_target(lbl)]


def lr_schedule(name: str, lr: float, steps: int, warmup_frac: float = 0.1):
    if name == "constant":
        return lr
    if name == "cosine":
        return keras.optimizers.schedules.CosineDecay(lr, max(steps, 1))
    # 'warmup_cosine' ≈ one-cycle: rampa lineal hasta lr y decaimiento coseno
    w = max(1, int(warmup_frac * steps))
    return keras.optimizers.schedules.CosineDecay(lr * 0.05, max(steps - w, 1), warmup_target=lr, warmup_steps=w)


def make_optimizer(name: str, schedule):
    if name == "adamw":
        return keras.optimizers.AdamW(schedule, weight_decay=1e-4)
    if name == "sgd":
        return keras.optimizers.SGD(schedule, momentum=0.9, nesterov=True)
    return keras.optimizers.Adam(schedule)


def maybe_mixed_precision() -> bool:
    import tensorflow as tf
    if tf.config.list_physical_devices("GPU"):
        keras.mixed_precision.set_global_policy("mixed_float16")
        return True
    keras.mixed_precision.set_global_policy("float32")
    return False
