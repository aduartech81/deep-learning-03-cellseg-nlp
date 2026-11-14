"""Mask R-CNN (He et al., 2017) con torchvision.

Desviación declarada respecto del enunciado: backbone ResNet50-FPN v2 preentrenado en COCO en lugar de Swin.
Swin + Mask R-CNN requiere MMDetection/Detectron2, cuya instalación en Colab con la versión actual de PyTorch no
es estable; la interfaz de evaluación es la misma y el cambio se registra en el pre-registro.
Anchors derivados del EDA (cuantiles de √área de las células).
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from .postproc import instances_to_label


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_maskrcnn(anchor_sizes=(8, 16, 32, 64, 128), pretrained: bool = True, max_det: int = 600,
                   min_size: int = 520, max_size: int = 704):
    ag = AnchorGenerator(sizes=tuple((s,) for s in anchor_sizes), aspect_ratios=((0.5, 1.0, 2.0),) * 5)
    m = maskrcnn_resnet50_fpn_v2(weights="DEFAULT" if pretrained else None, weights_backbone=None,
                                 min_size=min_size, max_size=max_size,
                                 box_detections_per_img=max_det, rpn_pre_nms_top_n_train=4000,
                                 rpn_post_nms_top_n_train=2000, rpn_pre_nms_top_n_test=4000,
                                 rpn_post_nms_top_n_test=2000, box_batch_size_per_image=512,
                                 box_score_thresh=0.05, image_mean=[0.5] * 3, image_std=[0.25] * 3)
    m.rpn.anchor_generator = ag            # mismas 3 relaciones de aspecto por nivel: la cabeza RPN es compatible
    in_f = m.roi_heads.box_predictor.cls_score.in_features
    m.roi_heads.box_predictor = FastRCNNPredictor(in_f, 2)
    m.roi_heads.mask_predictor = MaskRCNNPredictor(256, 256, 2)
    return m


def _target(lbl: np.ndarray):
    ids = np.unique(lbl)
    ids = ids[ids > 0]
    masks = np.stack([lbl == i for i in ids]).astype(np.uint8) if len(ids) else np.zeros((0,) + lbl.shape, np.uint8)
    boxes = []
    for m in masks:
        ys, xs = np.where(m)
        boxes.append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])
    return {"boxes": torch.as_tensor(np.array(boxes, np.float32).reshape(-1, 4)),
            "labels": torch.ones(len(masks), dtype=torch.int64), "masks": torch.as_tensor(masks)}


def _img_tensor(x01: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(np.repeat(x01[None], 3, 0).astype(np.float32))


def train_maskrcnn(model, imgs: list[np.ndarray], lbls: list[np.ndarray], epochs: int, batch: int = 2,
                   lr: float = 0.01, seed: int = 0, log: Path | None = None) -> list[float]:
    """imgs en [0,1] (H, W) nativas; SGD + warmup lineal + coseno; AMP en GPU; flips como augmentation."""
    dev = device()
    model.to(dev).train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=1e-4)
    steps = epochs * int(np.ceil(len(imgs) / batch))
    warm = max(1, steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + np.cos(np.pi * (s - warm) / max(1, steps - warm))))
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")
    rng = np.random.default_rng(seed)
    hist = []
    for ep in range(epochs):
        order = rng.permutation(len(imgs))
        tot, t0 = 0.0, time.time()
        for b in range(0, len(order), batch):
            xs, ts = [], []
            for i in order[b:b + batch]:
                x, y = imgs[i], lbls[i]
                if rng.random() < 0.5:
                    x, y = x[:, ::-1], y[:, ::-1]
                if rng.random() < 0.5:
                    x, y = x[::-1], y[::-1]
                t = _target(np.ascontiguousarray(y))
                if len(t["boxes"]) == 0:
                    continue
                xs.append(_img_tensor(np.ascontiguousarray(x)).to(dev))
                ts.append({k: v.to(dev) for k, v in t.items()})
            if not xs:
                continue
            with torch.autocast(dev.type, enabled=dev.type == "cuda"):
                loss = sum(model(xs, ts).values())
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += float(loss.detach())
        hist.append(tot / max(1, len(order) / batch))
        if log is not None:
            with open(log, "a") as fh:
                fh.write(f"epoch {ep} loss {hist[-1]:.4f} {time.time() - t0:.0f}s\n")
    return hist


@torch.no_grad()
def predict_raw(model, x01: np.ndarray) -> dict[str, np.ndarray]:
    dev = device()
    model.to(dev).eval()
    out = model([_img_tensor(x01).to(dev)])[0]
    return {"boxes": out["boxes"].cpu().numpy(), "scores": out["scores"].cpu().numpy(),
            "masks": out["masks"][:, 0].cpu().numpy().astype(np.float16)}


def raw_to_label(raw: dict, shape, score_thr: float = 0.5, mask_thr: float = 0.5, min_size: int = 10):
    return instances_to_label(raw["masks"].astype(np.float32), raw["scores"], shape, score_thr, mask_thr, min_size)
