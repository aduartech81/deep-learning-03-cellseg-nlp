"""Cellpose 2.0/3.x (Stringer et al., 2021; Pachitariu & Stringer, 2022): generalista vs ajustado.

Se fija cellpose<4 porque la serie 4 (Cellpose-SAM) cambió de arquitectura y de API. El "humano en el bucle" de
Cellpose 2.0 se aproxima con el ajuste fino sobre las anotaciones de entrenamiento (las anotaciones de
Sartorius hacen de corrección humana); se declara como tal.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def _gpu() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def generalist(model_type: str = "cyto2"):
    from cellpose import models
    return models.Cellpose(gpu=_gpu(), model_type=model_type)


def finetune(imgs: list[np.ndarray], lbls: list[np.ndarray], save_dir: Path, name: str, epochs: int,
             init: str | None = "cyto2", lr: float = 0.005, val=None, seed: int = 0) -> str:
    """Ajuste fino desde ``init`` (modelo de microscopía) o desde cero (init=None). Devuelve la ruta del modelo."""
    import torch
    from cellpose import models, train
    torch.manual_seed(seed)
    np.random.seed(seed)
    if init:
        net = models.CellposeModel(gpu=_gpu(), model_type=init)
    else:
        net = models.CellposeModel(gpu=_gpu(), pretrained_model=False, model_type=None)
    save_dir.mkdir(parents=True, exist_ok=True)
    kw = dict(train_data=imgs, train_labels=lbls, channels=[0, 0], normalize=True, n_epochs=epochs,
              learning_rate=lr, weight_decay=1e-4, save_path=str(save_dir), model_name=name, min_train_masks=1)
    if val is not None:
        kw.update(test_data=val[0], test_labels=val[1])
    out = train.train_seg(net.net, **kw)
    return str(out[0] if isinstance(out, tuple) else out)


def load_finetuned(path: str):
    from cellpose import models
    return models.CellposeModel(gpu=_gpu(), pretrained_model=path)


def predict(model, img: np.ndarray, diameter: float | None, flow_threshold: float = 0.4,
            cellprob_threshold: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve (etiquetas, probabilidad de célula = sigmoide del logit 'cellprob')."""
    out = model.eval(img, channels=[0, 0], diameter=diameter, flow_threshold=flow_threshold,
                     cellprob_threshold=cellprob_threshold)
    masks, flows = out[0], out[1]
    cp = np.asarray(flows[2], np.float32)
    if cp.shape != masks.shape:
        from skimage.transform import resize
        cp = resize(cp, masks.shape, order=1, preserve_range=True)
    return masks.astype(np.int32), (1 / (1 + np.exp(-cp))).astype(np.float32)


def model_diameter(model) -> float | None:
    d = getattr(model, "diam_labels", None)
    try:
        d = float(d)
    except (TypeError, ValueError):
        return None
    return d if d > 0 else None
