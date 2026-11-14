"""Métricas de segmentación: píxel, borde, instancia, conteo y taxonomía de errores.

Todas operan sobre UNA imagen (mapas de etiquetas enteros, 0 = fondo). La agregación sobre imágenes, con IC
bootstrap, se hace en ``seg_eval``. Las funciones de la guía (4.8.24) se conservan; se añaden AJI, F1 de borde,
precisión/recall/F1 por instancia y la clasificación automática de errores (4.8.31.3).

Supuesto clave: las instancias de un mismo mapa son disjuntas. Con IoU > 0.5 cada predicción puede emparejarse
con a lo sumo una referencia, de modo que el emparejamiento por umbral es único y no requiere algoritmo húngaro.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage.segmentation import find_boundaries

THRESHOLDS = np.round(np.arange(0.5, 1.0, 0.05), 2)


# ------------------------------------------------------------------ píxel ----
def dice_iou(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-7) -> tuple[float, float]:
    p, g = pred.astype(bool), gt.astype(bool)
    inter = np.logical_and(p, g).sum()
    union = np.logical_or(p, g).sum()
    return float((2 * inter + eps) / (p.sum() + g.sum() + eps)), float((inter + eps) / (union + eps))


def pixel_confusion_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    p, g = pred.astype(bool), gt.astype(bool)
    tp = np.logical_and(p, g).sum()
    fp = np.logical_and(p, ~g).sum()
    fn = np.logical_and(~p, g).sum()
    tn = np.logical_and(~p, ~g).sum()
    recall = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    return dict(precision=float(tp / max(tp + fp, 1)), recall=float(recall), specificity=float(spec),
                balanced_accuracy=float(0.5 * (recall + spec)))


def pixel_auc(prob: np.ndarray, gt: np.ndarray, max_pixels: int = 200_000, seed: int = 0) -> dict[str, float]:
    """AUC-ROC y AUC-PR de píxel para UNA imagen (submuestreo estratificado para acotar memoria)."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    y = gt.astype(bool).ravel()
    s = np.asarray(prob, float).ravel()
    if y.all() or (~y).all():
        return dict(auc_roc=np.nan, auc_pr=np.nan)
    if y.size > max_pixels:
        idx = np.random.default_rng(seed).choice(y.size, max_pixels, replace=False)
        y, s = y[idx], s[idx]
    return dict(auc_roc=float(roc_auc_score(y, s)), auc_pr=float(average_precision_score(y, s)))


# ------------------------------------------------------------------- borde ---
def hausdorff95(a: np.ndarray, b: np.ndarray) -> float:
    """Distancia de Hausdorff al percentil 95 entre contornos de las máscaras de primer plano (píxeles)."""
    a, b = a.astype(bool), b.astype(bool)
    if not a.any() and not b.any():
        return 0.0
    if not a.any() or not b.any():
        return float("inf")
    sa, sb = a ^ ndi.binary_erosion(a), b ^ ndi.binary_erosion(b)
    da = ndi.distance_transform_edt(~sb)[sa]
    db = ndi.distance_transform_edt(~sa)[sb]
    return float(np.percentile(np.concatenate([da, db]), 95))


def boundary_f1(gt_lbl: np.ndarray, pr_lbl: np.ndarray, tol: int = 2) -> float:
    """F1 de borde de instancias con tolerancia ``tol`` píxeles (Csurka et al., 2013)."""
    bg = find_boundaries(gt_lbl, mode="inner")
    bp = find_boundaries(pr_lbl, mode="inner")
    if not bg.any() and not bp.any():
        return 1.0
    if not bg.any() or not bp.any():
        return 0.0
    dg = ndi.distance_transform_edt(~bg)
    dp = ndi.distance_transform_edt(~bp)
    prec = float((dg[bp] <= tol).mean())
    rec = float((dp[bg] <= tol).mean())
    return 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)


# --------------------------------------------------------------- instancia ---
def _relabel(lbl: np.ndarray) -> np.ndarray:
    u, inv = np.unique(lbl, return_inverse=True)
    inv = inv.reshape(lbl.shape)
    return inv if u[0] == 0 else inv + 1


def intersection_matrix(gt_lbl: np.ndarray, pr_lbl: np.ndarray):
    """Intersecciones (n_gt+1 x n_pr+1, fila/columna 0 = fondo) y áreas por instancia."""
    gt_lbl, pr_lbl = _relabel(gt_lbl), _relabel(pr_lbl)
    n_gt, n_pr = int(gt_lbl.max()), int(pr_lbl.max())
    idx = gt_lbl.ravel().astype(np.int64) * (n_pr + 1) + pr_lbl.ravel()
    inter = np.bincount(idx, minlength=(n_gt + 1) * (n_pr + 1)).reshape(n_gt + 1, n_pr + 1)
    return inter, inter.sum(1), inter.sum(0)


def instance_iou_matrix(gt_lbl: np.ndarray, pr_lbl: np.ndarray) -> np.ndarray:
    """Matriz IoU (n_gt x n_pred) entre mapas de etiquetas."""
    inter, a_gt, a_pr = intersection_matrix(gt_lbl, pr_lbl)
    union = a_gt[:, None] + a_pr[None, :] - inter
    return (inter / np.maximum(union, 1))[1:, 1:]


def _tp_fp_fn(iou: np.ndarray, t: float) -> tuple[int, int, int]:
    n_gt, n_pr = iou.shape
    tp = int((iou > t).sum())
    return tp, n_pr - tp, n_gt - tp


def precision_at(iou: np.ndarray, t: float) -> float:
    """TP/(TP+FP+FN) en un umbral (definición de la competencia Sartorius; no es AP de COCO)."""
    tp, fp, fn = _tp_fp_fn(iou, t)
    d = tp + fp + fn
    return tp / d if d else 1.0


def sartorius_score(gt_lbl: np.ndarray, pr_lbl: np.ndarray, thresholds=THRESHOLDS) -> float:
    """AP@[0.5:0.95] estilo Sartorius: media sobre umbrales de TP/(TP+FP+FN)."""
    iou = instance_iou_matrix(gt_lbl, pr_lbl)
    return float(np.mean([precision_at(iou, t) for t in thresholds]))


def panoptic_quality(gt_lbl: np.ndarray, pr_lbl: np.ndarray) -> tuple[float, float, float]:
    """PQ = SQ * RQ (Kirillov et al., 2019). Devuelve (PQ, SQ, RQ)."""
    iou = instance_iou_matrix(gt_lbl, pr_lbl)
    n_gt, n_pr = iou.shape
    matched = iou > 0.5
    tp = int(matched.sum())
    fp, fn = n_pr - tp, n_gt - tp
    if tp + fp + fn == 0:
        return 1.0, 1.0, 1.0
    sq = float(iou[matched].mean()) if tp else 0.0
    rq = tp / (tp + 0.5 * fp + 0.5 * fn)
    return sq * rq, sq, rq


def aji(gt_lbl: np.ndarray, pr_lbl: np.ndarray) -> float:
    """Aggregated Jaccard Index (Kumar et al., 2017)."""
    inter, a_gt, a_pr = intersection_matrix(gt_lbl, pr_lbl)
    inter, a_gt, a_pr = inter[1:, 1:], a_gt[1:], a_pr[1:]
    if len(a_gt) == 0 and len(a_pr) == 0:
        return 1.0
    if len(a_gt) == 0 or len(a_pr) == 0:
        return 0.0
    union = a_gt[:, None] + a_pr[None, :] - inter
    iou = inter / np.maximum(union, 1)
    c = u = 0.0
    used = np.zeros(len(a_pr), bool)
    for g in range(len(a_gt)):
        k = int(iou[g].argmax())
        if iou[g, k] > 0:
            c += inter[g, k]
            u += union[g, k]
            used[k] = True
        else:                       # referencia sin ninguna predicción que la toque
            u += a_gt[g]
    u += float(a_pr[~used].sum())   # predicciones nunca elegidas penalizan el denominador
    return float(c / u) if u > 0 else 0.0


def instance_prf(gt_lbl: np.ndarray, pr_lbl: np.ndarray, t: float = 0.5) -> dict[str, float]:
    iou = instance_iou_matrix(gt_lbl, pr_lbl)
    tp, fp, fn = _tp_fp_fn(iou, t)
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    return dict(inst_precision=p, inst_recall=r, inst_f1=0.0 if p + r == 0 else 2 * p * r / (p + r))


# ---------------------------------------------------------- taxonomía ---------
ERROR_TYPES = ("omitida", "falsa", "fusionada", "dividida", "desplazada", "borde_impreciso")


def error_taxonomy(gt_lbl: np.ndarray, pr_lbl: np.ndarray, cover: float = 0.5,
                   loose: float = 0.1, tight: float = 0.75) -> dict[str, int]:
    """Clasifica automáticamente los errores de instancia (4.8.31.3).

    Reglas (declaradas a priori):
    * fusionada: referencia no emparejada (IoU ≤ 0.5) que tiene ≥ ``cover`` de su área dentro de una predicción
      que a su vez cubre ≥ 2 referencias en esa misma proporción (se cuentan las referencias).
    * dividida: referencia no emparejada que contiene ≥ 2 predicciones con ≥ ``cover`` de su propia área dentro.
    * omitida: referencia no emparejada, no fusionada ni dividida, con IoU máximo ≤ ``loose``.
    * desplazada: referencia no emparejada restante (IoU máximo en (``loose``, 0.5]).
    * borde_impreciso: referencia emparejada con IoU en (0.5, ``tight``].
    * falsa: predicción no emparejada con IoU máximo ≤ ``loose`` que no participa en fusiones ni divisiones.
    """
    inter, a_gt, a_pr = intersection_matrix(gt_lbl, pr_lbl)
    inter, a_gt, a_pr = inter[1:, 1:], a_gt[1:], a_pr[1:]
    out = dict.fromkeys(ERROR_TYPES, 0)
    n_gt, n_pr = len(a_gt), len(a_pr)
    if n_gt == 0:
        out["falsa"] = n_pr
        return out
    if n_pr == 0:
        out["omitida"] = n_gt
        return out
    union = a_gt[:, None] + a_pr[None, :] - inter
    iou = inter / np.maximum(union, 1)
    gt_in_pr = inter / np.maximum(a_gt[:, None], 1)      # fracción de la referencia dentro de la predicción
    pr_in_gt = inter / np.maximum(a_pr[None, :], 1)      # fracción de la predicción dentro de la referencia
    matched_gt = (iou > 0.5).any(1)
    matched_pr = (iou > 0.5).any(0)
    big_pr = (gt_in_pr >= cover).sum(0) >= 2             # predicciones que engloban ≥ 2 referencias
    merged = ~matched_gt & ((gt_in_pr >= cover) & big_pr[None, :]).any(1)
    split = ~matched_gt & ~merged & ((pr_in_gt >= cover).sum(1) >= 2)
    best = iou.max(1)
    out["fusionada"] = int(merged.sum())
    out["dividida"] = int(split.sum())
    rest = ~matched_gt & ~merged & ~split
    out["omitida"] = int((rest & (best <= loose)).sum())
    out["desplazada"] = int((rest & (best > loose)).sum())
    out["borde_impreciso"] = int((matched_gt & (best <= tight)).sum())
    involved = big_pr | ((pr_in_gt >= cover) & split[:, None]).any(0)
    out["falsa"] = int((~matched_pr & ~involved & (iou.max(0) <= loose)).sum())
    return out


# ------------------------------------------------------------- todo junto ---
def image_metrics(gt_lbl: np.ndarray, pr_lbl: np.ndarray, prob: np.ndarray | None = None) -> dict[str, float]:
    """Todas las métricas para una imagen. ``prob``: mapa de probabilidad de primer plano (o None)."""
    iou = instance_iou_matrix(gt_lbl, pr_lbl)
    pq, sq, rq = panoptic_quality(gt_lbl, pr_lbl)
    g, p = gt_lbl > 0, pr_lbl > 0
    d, j = dice_iou(p, g)
    row: dict[str, float] = dict(
        ap=float(np.mean([precision_at(iou, t) for t in THRESHOLDS])),
        ap50=precision_at(iou, 0.5), ap75=precision_at(iou, 0.75), pq=pq, sq=sq, rq=rq, aji=aji(gt_lbl, pr_lbl),
        dice=d, iou=j, hd95=hausdorff95(p, g), boundary_f1=boundary_f1(gt_lbl, pr_lbl),
        n_gt=int(iou.shape[0]), n_pred=int(iou.shape[1]))
    row.update(instance_prf(gt_lbl, pr_lbl))
    row.update(pixel_confusion_metrics(p, g))
    if prob is not None:
        row.update(pixel_auc(prob, g))
    else:
        row.update(auc_roc=np.nan, auc_pr=np.nan)
    row.update({f"err_{k}": v for k, v in error_taxonomy(gt_lbl, pr_lbl).items()})
    row.update({f"p@{t:.2f}": precision_at(iou, t) for t in THRESHOLDS})
    return row
