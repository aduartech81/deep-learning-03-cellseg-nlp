"""Segment Anything (Kirillov et al., 2023), tres modos de uso (4.8.28.1, modelo 5).

(a) cero-shot con el generador automático de máscaras;
(b) prompts de caja tomadas de las detecciones de Mask R-CNN (mismas imágenes de test);
(c) ajuste fino solo del decodificador de máscaras (codificadores congelados) con cajas de la referencia.
MedSAM no se evalúa (desviación declarada: presupuesto de cómputo).
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np
import torch

from .postproc import instances_to_label

_EMB_CACHE: dict = {}
CKPT = {"vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"}


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_sam(model_type: str = "vit_b", cache: Path | None = None, pretrained: bool = True):
    from segment_anything import sam_model_registry
    ckpt = None
    if pretrained:
        cache = cache or Path.home() / ".cache" / "sam"
        cache.mkdir(parents=True, exist_ok=True)
        ckpt = cache / Path(CKPT[model_type]).name
        if not ckpt.exists():
            urllib.request.urlretrieve(CKPT[model_type], ckpt)
    sam = sam_model_registry[model_type](checkpoint=str(ckpt) if ckpt else None)
    return sam.to(device()).eval()


def to_rgb8(x01: np.ndarray) -> np.ndarray:
    return (np.clip(x01, 0, 1) * 255).astype(np.uint8)[..., None].repeat(3, -1)


def automatic(sam, x01: np.ndarray, points_per_side: int = 32, pred_iou_thresh: float = 0.8,
              stability_thresh: float = 0.9, max_area: int | None = None, min_size: int = 10):
    """(a) Cero-shot. Se descartan máscaras mayores que ``max_area`` (fondo / grupos) definido con el EDA."""
    from segment_anything import SamAutomaticMaskGenerator
    gen = SamAutomaticMaskGenerator(sam, points_per_side=points_per_side, pred_iou_thresh=pred_iou_thresh,
                                    stability_score_thresh=stability_thresh, min_mask_region_area=min_size,
                                    points_per_batch=64)
    anns = gen.generate(to_rgb8(x01))
    if max_area:
        anns = [a for a in anns if a["area"] <= max_area]
    if not anns:
        z = np.zeros(x01.shape, np.int32)
        return z, z.astype(np.float32)
    masks = np.stack([a["segmentation"] for a in anns])
    scores = np.array([a["predicted_iou"] * a["stability_score"] for a in anns], np.float32)
    return instances_to_label(masks, scores, x01.shape, score_thr=0.0, min_size=min_size)


@torch.no_grad()
def with_boxes(sam, x01: np.ndarray, boxes: np.ndarray, box_scores: np.ndarray, score_thr: float = 0.5,
               batch: int = 64, min_size: int = 10, decoder=None):
    """(b)/(c) Máscaras a partir de cajas (xyxy, coordenadas nativas)."""
    from segment_anything import SamPredictor
    keep = box_scores >= score_thr
    boxes, box_scores = boxes[keep], box_scores[keep]
    if len(boxes) == 0:
        z = np.zeros(x01.shape, np.int32)
        return z, z.astype(np.float32)
    pred = SamPredictor(sam)
    if decoder is not None:
        sam.mask_decoder = decoder
    key = (id(sam), x01.shape, float(x01.sum()))
    if key in _EMB_CACHE:                     # el embedding de la imagen no depende de las cajas: se reutiliza
        pred.features, pred.original_size, pred.input_size = _EMB_CACHE[key]
        pred.is_image_set = True
    else:
        pred.set_image(to_rgb8(x01))
        if len(_EMB_CACHE) > 100:
            _EMB_CACHE.clear()
        _EMB_CACHE[key] = (pred.features, pred.original_size, pred.input_size)
    tb = pred.transform.apply_boxes_torch(torch.as_tensor(boxes, dtype=torch.float32, device=device()),
                                          x01.shape)
    out = []
    for s in range(0, len(tb), batch):
        m, iou, _ = pred.predict_torch(point_coords=None, point_labels=None, boxes=tb[s:s + batch],
                                       multimask_output=False, return_logits=True)
        out.append(torch.sigmoid(m[:, 0]).cpu().numpy())
    probs = np.concatenate(out)
    return instances_to_label(probs, box_scores, x01.shape, score_thr=0.0, min_size=min_size)


def finetune_decoder(sam, imgs: list[np.ndarray], lbls: list[np.ndarray], epochs: int = 2, boxes_per_img: int = 32,
                     lr: float = 1e-4, seed: int = 0, jitter: float = 0.1):
    """(c) Ajuste fino del decodificador: BCE + Dice sobre máscaras de baja resolución, cajas con ruido."""
    import torch.nn.functional as F
    from segment_anything.utils.transforms import ResizeLongestSide
    from skimage import measure
    dev = device()
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    tr = ResizeLongestSide(sam.image_encoder.img_size)
    for p in sam.image_encoder.parameters():
        p.requires_grad = False
    for p in sam.prompt_encoder.parameters():
        p.requires_grad = False
    opt = torch.optim.AdamW(sam.mask_decoder.parameters(), lr=lr, weight_decay=1e-4)
    sam.mask_decoder.train()
    emb_cache: dict[int, torch.Tensor] = {}
    hist = []
    for ep in range(epochs):
        tot = 0.0
        for i in rng.permutation(len(imgs)):
            x, lbl = imgs[i], lbls[i]
            props = measure.regionprops(lbl)
            if not props:
                continue
            if i not in emb_cache:
                im = tr.apply_image(to_rgb8(x))
                t = torch.as_tensor(im, device=dev).permute(2, 0, 1)[None].float()
                with torch.no_grad():
                    emb_cache[i] = sam.image_encoder(sam.preprocess(t)).half().cpu()
            emb = emb_cache[i].to(dev).float()
            sel = rng.choice(len(props), min(boxes_per_img, len(props)), replace=False)
            bx, gt = [], []
            for j in sel:
                y0, x0, y1, x1 = props[j].bbox
                w, h = x1 - x0, y1 - y0
                d = rng.normal(0, jitter, 4) * np.array([w, h, w, h])
                bx.append([x0 + d[0], y0 + d[1], x1 + d[2], y1 + d[3]])
                gt.append(lbl == props[j].label)
            bt = tr.apply_boxes_torch(torch.as_tensor(np.array(bx), dtype=torch.float32, device=dev), x.shape)
            sp, de = sam.prompt_encoder(points=None, boxes=bt, masks=None)
            low, _ = sam.mask_decoder(image_embeddings=emb, image_pe=sam.prompt_encoder.get_dense_pe(),
                                      sparse_prompt_embeddings=sp, dense_prompt_embeddings=de,
                                      multimask_output=False)
            up = sam.postprocess_masks(low, tr.get_preprocess_shape(*x.shape, tr.target_length), x.shape)[:, 0]
            g = torch.as_tensor(np.stack(gt), dtype=torch.float32, device=dev)
            p = torch.sigmoid(up)
            loss = F.binary_cross_entropy_with_logits(up, g) + \
                (1 - (2 * (p * g).sum((1, 2)) + 1) / (p.sum((1, 2)) + g.sum((1, 2)) + 1)).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.detach())
        hist.append(tot / max(1, len(imgs)))
    sam.mask_decoder.eval()
    return hist
