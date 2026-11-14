"""U-Net++ (Zhou et al., 2018) implementada desde cero en Keras, con supervisión profunda.

Objetivo de 3 clases (fondo / interior / borde-contacto) + watershed con marcadores por h-maxima.
Incluye: capas de dropout espacial "MC" (activables en inferencia para MC-Dropout), Grad-CAM sobre el cuello de
botella y predicción por teselas con solape para la variante de resolución "tiling".
"""
from __future__ import annotations

import keras
import numpy as np
from keras import layers

from .postproc import watershed_from_probs


@keras.saving.register_keras_serializable(package="dl03")
class MCSpatialDropout(layers.SpatialDropout2D):
    """Dropout espacial que puede quedar activo en inferencia sin poner BatchNorm en modo entrenamiento."""

    mc_active = False

    def call(self, inputs, training=None):
        return super().call(inputs, training=bool(training) or MCSpatialDropout.mc_active)


def conv_block(x, f: int, name: str, drop: float = 0.0):
    for k in (1, 2):
        x = layers.Conv2D(f, 3, padding="same", use_bias=False, name=f"{name}_c{k}")(x)
        x = layers.BatchNormalization(name=f"{name}_bn{k}")(x)
        x = layers.Activation("relu", name=f"{name}_r{k}" if k == 1 else name)(x)
    if drop > 0:
        x = MCSpatialDropout(drop, name=f"{name}_mcdrop")(x)
    return x


def build_unetpp(filters=(32, 64, 128, 256, 512), n_classes: int = 3, deep_supervision: bool = True,
                 mc_drop: float = 0.1, channels: int = 1) -> keras.Model:
    """X[i][j]: nodo en profundidad i y columna j (j = 0 es el codificador)."""
    inp = keras.Input((None, None, channels), name="image")
    depth = len(filters)
    X: dict[tuple[int, int], keras.KerasTensor] = {}
    x = inp
    for i in range(depth):
        x = conv_block(x if i == 0 else layers.MaxPooling2D(name=f"pool{i}")(X[(i - 1, 0)]), filters[i],
                       f"X{i}0", drop=mc_drop if i == depth - 1 else 0.0)
        X[(i, 0)] = x
    for j in range(1, depth):
        for i in range(depth - j):
            up = layers.Conv2DTranspose(filters[i], 2, strides=2, padding="same", name=f"up{i + 1}{j - 1}")(
                X[(i + 1, j - 1)])
            cat = layers.Concatenate(name=f"cat{i}{j}")([X[(i, k)] for k in range(j)] + [up])
            X[(i, j)] = conv_block(cat, filters[i], f"X{i}{j}", drop=mc_drop if (i == 0 and j == depth - 1) else 0.0)
    heads = []
    cols = range(1, depth) if deep_supervision else [depth - 1]
    for j in cols:
        h = layers.Conv2D(n_classes, 1, name=f"logits{j}")(X[(0, j)])
        heads.append(layers.Activation("softmax", dtype="float32", name=f"out{j}")(h))
    return keras.Model(inp, heads if len(heads) > 1 else heads[0], name="unetpp")


def predict_probs(model: keras.Model, x: np.ndarray, batch: int = 4) -> np.ndarray:
    """x: (N, H, W) normalizada -> probabilidades (N, H, W, 3) promediando las cabezas de supervisión profunda."""
    out = model.predict(x[..., None].astype(np.float32), batch_size=batch, verbose=0)
    if isinstance(out, list):
        out = np.mean(out, axis=0)
    return out


def predict_tiled(model: keras.Model, img: np.ndarray, tile: int = 256, stride: int = 192) -> np.ndarray:
    """Predicción por teselas con solape y promedio ponderado (ventana de Hann) en la zona de solape."""
    h, w = img.shape
    ph, pw = max(h, tile), max(w, tile)
    pad = np.pad(img, ((0, ph - h), (0, pw - w)), mode="reflect")
    acc = np.zeros((ph, pw, 3), np.float32)
    wsum = np.zeros((ph, pw, 1), np.float32)
    win = np.outer(np.hanning(tile) + 1e-3, np.hanning(tile) + 1e-3)[..., None].astype(np.float32)
    ys = list(range(0, ph - tile + 1, stride)) + ([ph - tile] if (ph - tile) % stride else [])
    xs = list(range(0, pw - tile + 1, stride)) + ([pw - tile] if (pw - tile) % stride else [])
    crops, pos = [], []
    for y0 in sorted(set(ys)):
        for x0 in sorted(set(xs)):
            crops.append(pad[y0:y0 + tile, x0:x0 + tile])
            pos.append((y0, x0))
    probs = predict_probs(model, np.stack(crops))
    for (y0, x0), p in zip(pos, probs):
        acc[y0:y0 + tile, x0:x0 + tile] += p * win
        wsum[y0:y0 + tile, x0:x0 + tile] += win
    return (acc / wsum)[:h, :w]


def probs_to_instances(p: np.ndarray, pp: dict) -> np.ndarray:
    """p: (H, W, 3) -> etiquetas. pp: thr_fg, thr_in, h, min_size (ajustados SOLO en validación)."""
    return watershed_from_probs(p[..., 1] + p[..., 2], p[..., 1], **pp)


def mc_dropout_probs(model: keras.Model, x: np.ndarray, T: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Media y varianza (sobre T pasadas con dropout activo) de la probabilidad de primer plano."""
    MCSpatialDropout.mc_active = True
    try:
        s = np.stack([predict_probs(model, x)[..., 1:].sum(-1) for _ in range(T)])
    finally:
        MCSpatialDropout.mc_active = False
    return s.mean(0), s.var(0)


def grad_cam(model: keras.Model, x: np.ndarray, layer: str = "X40", head: int = -1) -> np.ndarray:
    """Grad-CAM (Selvaraju et al., 2017) de la probabilidad media de célula respecto de una capa interna."""
    import tensorflow as tf
    sub = keras.Model(model.input, [model.get_layer(layer).output, model.output])
    xx = tf.convert_to_tensor(x[None, ..., None].astype(np.float32))
    with tf.GradientTape() as tape:
        feat, out = sub(xx, training=False)
        out = out[head] if isinstance(out, (list, tuple)) else out
        score = tf.reduce_mean(out[..., 1] + out[..., 2])
    g = tape.gradient(score, feat)
    w = tf.reduce_mean(g, axis=(1, 2), keepdims=True)
    cam = tf.nn.relu(tf.reduce_sum(w * feat, -1))[0].numpy()
    cam = cam / (cam.max() + 1e-8)
    from skimage.transform import resize
    return resize(cam, x.shape, order=1)
