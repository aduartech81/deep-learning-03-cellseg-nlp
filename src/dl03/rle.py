"""RLE y mapas de etiquetas (4.8.24, con supuestos documentados).

Supuestos y casos borde
-----------------------
* Convención Sartorius: pares «inicio longitud», inicio **1-indexado**, aplanado en orden de filas (C) sobre una
  imagen de forma (alto, ancho) = (520, 704). El capítulo de EDA lo verifica visualmente con superposiciones.
* Un RLE vacío o NaN produce una máscara vacía.
* Si un RLE se sale de la imagen se lanza ``ValueError`` (en la versión original de la guía se truncaba en
  silencio, lo que ocultaba errores de forma).
* Las anotaciones de Sartorius pueden solaparse. ``instances_to_label_map`` resuelve el solape asignando el píxel a
  la primera instancia (como la guía) y devuelve, si se pide, cuántos píxeles se perdieron por solape.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np


def rle_decode(rle: str | float | None, shape: tuple[int, int]) -> np.ndarray:
    """RLE -> máscara binaria uint8 de forma (alto, ancho)."""
    h, w = shape
    mask = np.zeros(h * w, dtype=np.uint8)
    if isinstance(rle, str) and rle.strip():
        s = np.asarray(rle.split(), dtype=np.int64)
        if s.size % 2:
            raise ValueError("RLE con número impar de valores")
        starts, lengths = s[0::2] - 1, s[1::2]
        if (starts < 0).any() or (starts + lengths > h * w).any():
            raise ValueError("RLE fuera de los límites de la imagen")
        for a, ln in zip(starts, lengths):
            mask[a:a + ln] = 1
    return mask.reshape(shape)


def rle_encode(mask: np.ndarray) -> str:
    """Máscara binaria -> RLE 1-indexado (orden de filas)."""
    pixels = np.concatenate([[0], np.asarray(mask, dtype=np.uint8).flatten(), [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(map(str, runs))


def instances_to_label_map(rles: Iterable[str], shape: tuple[int, int], return_overlap: bool = False):
    """Lista de RLE -> mapa de etiquetas int32 (0 = fondo). En solapes prevalece la primera instancia."""
    lbl = np.zeros(shape, dtype=np.int32)
    lost = 0
    for i, r in enumerate(rles, start=1):
        m = rle_decode(r, shape).astype(bool)
        lost += int((m & (lbl > 0)).sum())
        lbl[m & (lbl == 0)] = i
    return (lbl, lost) if return_overlap else lbl


def label_map_to_rles(lbl: np.ndarray) -> list[str]:
    """Mapa de etiquetas -> lista de RLE (una por instancia, en orden de etiqueta)."""
    return [rle_encode(lbl == k) for k in np.unique(lbl) if k != 0]
