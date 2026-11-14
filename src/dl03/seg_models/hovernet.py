"""Modelo estilo Hover-Net (Graham et al., 2019) reimplementado en Keras.

Desviación declarada: no se usa el repositorio oficial (PyTorch, pensado para histología con clasificación de
tipo nuclear). Se conservan las dos ramas que el enunciado pide (NP = píxel de núcleo/célula y HV = mapas
horizontal/vertical), sus pérdidas (BCE + Dice; MSE + MSGE) y su post-proceso, sobre un codificador ResNet50
(ImageNet o desde cero) con decodificadores tipo U-Net.
"""
from __future__ import annotations

import keras
import numpy as np
from keras import layers, ops

from ..seg_data import hv_maps
from .keras_common import EPS

SKIPS = ["conv1_relu", "conv2_block3_out", "conv3_block4_out", "conv4_block6_out"]


def _decoder(feats: list, name: str, widths=(256, 128, 64, 32)):
    x = feats[-1]
    for k, (skip, f) in enumerate(zip(reversed(feats[:-1]), widths)):
        x = layers.UpSampling2D(2, interpolation="bilinear", name=f"{name}_up{k}")(x)
        x = layers.Concatenate(name=f"{name}_cat{k}")([x, skip])
        for r in (1, 2):
            x = layers.Conv2D(f, 3, padding="same", use_bias=False, name=f"{name}_c{k}{r}")(x)
            x = layers.BatchNormalization(name=f"{name}_bn{k}{r}")(x)
            x = layers.Activation("relu", name=f"{name}_a{k}{r}")(x)
    x = layers.UpSampling2D(2, interpolation="bilinear", name=f"{name}_upfin")(x)
    return layers.Conv2D(32, 3, padding="same", activation="relu", name=f"{name}_cfin")(x)


def build_hovernet(pretrained: bool = True) -> keras.Model:
    base = keras.applications.ResNet50(include_top=False, weights="imagenet" if pretrained else None,
                                       input_shape=(None, None, 3))
    feats = [base.get_layer(n).output for n in SKIPS] + [base.output]
    np_head = layers.Conv2D(1, 1, activation="sigmoid", dtype="float32", name="np")(_decoder(feats, "dnp"))
    hv_head = layers.Conv2D(2, 1, activation="tanh", dtype="float32", name="hv")(_decoder(feats, "dhv"))
    return keras.Model(base.input, [np_head, hv_head], name="hovernet_style")


def np_loss(y_true, y_pred):
    t, p = ops.cast(y_true, "float32"), ops.clip(ops.cast(y_pred, "float32"), EPS, 1 - EPS)
    bce = -ops.mean(t * ops.log(p) + (1 - t) * ops.log(1 - p))
    dice = 1 - (2 * ops.sum(t * p) + EPS) / (ops.sum(t) + ops.sum(p) + EPS)
    return bce + dice


def _sobel_kernels(size: int = 5):
    r = np.arange(-(size // 2), size // 2 + 1, dtype=np.float32)
    h, v = np.meshgrid(r, r)
    kh = h / np.maximum(h * h + v * v, 1e-6)
    kv = v / np.maximum(h * h + v * v, 1e-6)
    return kh[..., None, None], kv[..., None, None]


KH, KV = _sobel_kernels()


def hv_loss(y_true, y_pred):
    """y_true: (B, H, W, 3) = [h, v, máscara]; MSE + 2·MSGE (gradientes Sobel dentro de las células)."""
    t = ops.cast(y_true, "float32")
    p = ops.cast(y_pred, "float32")
    hv_t, m = t[..., :2], t[..., 2:3]
    mse = ops.mean(ops.square(p - hv_t))
    gh_p = ops.conv(p[..., 0:1], KH, padding="same")
    gh_t = ops.conv(hv_t[..., 0:1], KH, padding="same")
    gv_p = ops.conv(p[..., 1:2], KV, padding="same")
    gv_t = ops.conv(hv_t[..., 1:2], KV, padding="same")
    msge = ops.sum(m * (ops.square(gh_p - gh_t) + ops.square(gv_p - gv_t))) / (ops.sum(m) + EPS)
    return mse + 2.0 * msge


def hover_target(lbl: np.ndarray):
    hv = hv_maps(lbl)
    m = (lbl > 0).astype(np.float32)[..., None]
    return m, np.concatenate([hv, m], -1)


def to_resnet_input(x01: np.ndarray) -> np.ndarray:
    """(N, H, W) en [0, 1] -> (N, H, W, 3) con el preprocesado 'caffe' de ResNet50."""
    x = np.repeat(x01[..., None] * 255.0, 3, -1).astype(np.float32)
    return keras.applications.resnet50.preprocess_input(x)


class HoverSequenceAdapter(keras.utils.PyDataset):
    """Envuelve SegSequence (1 canal [0,1]) y aplica el preprocesado de ResNet50."""

    def __init__(self, seq, **kw):
        super().__init__(**kw)
        self.seq = seq

    def __len__(self):
        return len(self.seq)

    def on_epoch_end(self):
        self.seq.on_epoch_end()

    def __getitem__(self, i):
        x, y = self.seq[i]
        return to_resnet_input(x[..., 0]), y
