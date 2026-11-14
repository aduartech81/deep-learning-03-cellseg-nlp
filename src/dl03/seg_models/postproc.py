"""Post-procesamiento común: watershed con marcadores por h-maxima y post-proceso tipo Hover-Net."""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage import filters, measure, morphology, segmentation


def drop_small_bool(mask: np.ndarray, min_size: int) -> np.ndarray:
    """Quita componentes conexas con área < min_size (independiente de la versión de scikit-image)."""
    return remove_small(measure.label(mask), min_size) > 0


def remove_small(lbl: np.ndarray, min_size: int) -> np.ndarray:
    if min_size <= 0 or lbl.max() == 0:
        return lbl
    areas = np.bincount(lbl.ravel())
    small = np.where(areas < min_size)[0]
    small = small[small > 0]
    if small.size:
        lbl = lbl.copy()
        lbl[np.isin(lbl, small)] = 0
    return segmentation.relabel_sequential(lbl)[0].astype(np.int32)


def watershed_from_probs(p_fg: np.ndarray, p_in: np.ndarray, thr_fg: float = 0.5, thr_in: float = 0.5,
                         h: float = 1.0, min_size: int = 10, sigma: float = 0.0) -> np.ndarray:
    """Instancias a partir de probabilidades de primer plano (célula+borde) e interior.

    Marcadores: h-maxima de la transformada de distancia del interior (separa células en contacto).
    Inundación: watershed sobre −distancia restringido a la máscara de primer plano.
    """
    if sigma > 0:
        p_fg, p_in = filters.gaussian(p_fg, sigma), filters.gaussian(p_in, sigma)
    fg = p_fg > thr_fg
    interior = (p_in > thr_in) & fg
    if not interior.any():
        return np.zeros(p_fg.shape, np.int32)
    dist = ndi.distance_transform_edt(interior)
    markers = measure.label(morphology.h_maxima(dist, h) if h > 0 else interior)
    if markers.max() == 0:
        markers = measure.label(interior)
    lbl = segmentation.watershed(-dist, markers, mask=fg)
    return remove_small(lbl.astype(np.int32), min_size)


def classical_segment(img: np.ndarray, method: str = "otsu", block: int = 51, h: float = 2.0, min_size: int = 30,
                      invert: bool | None = None, sigma: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Baseline sin aprendizaje: umbral (Otsu o adaptativo) + watershed. Devuelve (etiquetas, 'probabilidad').

    En contraste de fase las células pueden ser más claras u oscuras que el fondo; con ``invert=None`` se decide
    por la asimetría de la distribución de intensidades (las células ocupan la cola minoritaria).
    La 'probabilidad' es la intensidad normalizada: un score continuo sin significado probabilístico, que se
    usa solo para AUC de píxel (declarado como tal en el reporte).
    """
    x = filters.gaussian(img, sigma, preserve_range=True)
    # corrección de fondo: se resta un fondo muy suavizado (elimina gradientes de iluminación) y la desviación
    # absoluta respecto de él separa células (halo claro / interior oscuro) del fondo
    bg = filters.gaussian(img, max(img.shape) / 16, preserve_range=True)
    dev = np.abs(x - bg)
    if invert is True:
        dev = bg - x
    elif invert is False:
        dev = x - bg
    score = (dev - dev.min()) / max(np.ptp(dev), 1e-6)
    if method == "otsu":
        fg = score > filters.threshold_otsu(score)
    else:
        fg = score > filters.threshold_local(score, block_size=block | 1, offset=-0.01)
    fg = drop_small_bool(ndi.binary_fill_holes(morphology.opening(fg)), min_size)
    dist = ndi.distance_transform_edt(fg)
    markers = measure.label(morphology.h_maxima(dist, h))
    lbl = segmentation.watershed(-dist, markers, mask=fg).astype(np.int32)
    return remove_small(lbl, min_size), score.astype(np.float32)


def hovernet_postproc(p_np: np.ndarray, hv: np.ndarray, thr: float = 0.5, k: float = 0.4, min_size: int = 10,
                      ksize: int = 21) -> np.ndarray:
    """Post-proceso de Graham et al. (2019): gradientes de los mapas H/V para separar instancias."""
    import cv2
    blb = drop_small_bool(p_np > thr, min_size)
    if not blb.any():
        return np.zeros(p_np.shape, np.int32)
    h_dir = cv2.normalize(hv[..., 0], None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)
    v_dir = cv2.normalize(hv[..., 1], None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)
    sobelh = cv2.Sobel(h_dir, cv2.CV_64F, 1, 0, ksize=ksize)
    sobelv = cv2.Sobel(v_dir, cv2.CV_64F, 0, 1, ksize=ksize)
    sobelh = 1 - cv2.normalize(sobelh, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)
    sobelv = 1 - cv2.normalize(sobelv, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)
    overall = np.maximum(sobelh, sobelv) - (1 - blb)
    overall[overall < 0] = 0
    dist = -cv2.GaussianBlur((1.0 - overall) * blb, (3, 3), 0)
    marker = blb.astype(np.float32) - (overall >= k)
    marker[marker < 0] = 0
    marker = ndi.binary_fill_holes(marker).astype(np.uint8)
    marker = cv2.morphologyEx(marker, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    marker = remove_small(measure.label(marker), max(1, min_size // 2))
    lbl = segmentation.watershed(dist, marker, mask=blb)
    return remove_small(lbl.astype(np.int32), min_size)


def instances_to_label(masks: np.ndarray, scores: np.ndarray, shape: tuple[int, int], score_thr: float = 0.5,
                       mask_thr: float = 0.5, min_size: int = 10, min_keep: float = 0.5):
    """Lista de máscaras (N,H,W) con puntajes -> mapa de etiquetas sin solapes + mapa de 'probabilidad'.

    Los píxeles en disputa se asignan a la instancia de mayor puntaje; se descarta una instancia si conserva
    menos de ``min_keep`` de su área tras resolver solapes (evita fragmentos).
    """
    lbl = np.zeros(shape, np.int32)
    prob = np.zeros(shape, np.float32)
    order = np.argsort(-scores)
    k = 0
    for i in order:
        if scores[i] < score_thr:
            continue
        m = masks[i] > mask_thr if masks[i].dtype != bool else masks[i]
        prob = np.maximum(prob, scores[i] * (masks[i].astype(np.float32) if masks[i].dtype != bool else m))
        free = m & (lbl == 0)
        if free.sum() < max(min_size, min_keep * m.sum()):
            continue
        k += 1
        lbl[free] = k
    return lbl, prob
