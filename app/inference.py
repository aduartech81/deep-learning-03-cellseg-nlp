"""Carga de modelos y predicción. Sin TensorFlow ni PyTorch: texto con scikit-learn (joblib) e imágenes con
U-Net++ exportada a ONNX (onnxruntime) + watershed propio. Pensado para caber en 512 MB."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

MODELS = Path(os.environ.get("MODELS_DIR", Path(__file__).parent / "models"))


class TextModel:
    def __init__(self, d: Path):
        import joblib
        self.model = joblib.load(d / "text_model.joblib")
        meta = json.loads((d / "text_model.json").read_text())
        self.classes, self.version = meta["classes"], meta["version"]

    def predict_proba(self, text: str) -> dict[str, float]:
        p = self.model.predict_proba([text])[0]
        return {c: float(v) for c, v in zip(self.classes, p)}


class ImageModel:
    def __init__(self, d: Path):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(str(d / "unetpp.onnx"), providers=["CPUExecutionProvider"])
        meta = json.loads((d / "image_model.json").read_text())
        self.pp, self.res, self.version = meta["postproc"], meta["resolution"], meta["version"]
        self.inp = self.sess.get_inputs()[0].name

    def probs(self, img: np.ndarray) -> np.ndarray:
        from dl03.seg_data import Geometry, normalize
        x = np.clip(normalize(img.astype(np.float32), "percentile"), 0, 1)
        g = Geometry.for_resolution(x.shape, self.res)
        out = self.sess.run(None, {self.inp: g.to_model(x)[None, ..., None].astype(np.float32)})
        p = np.mean(out, 0)[0] if len(out) > 1 else out[0][0]
        return g.to_native(p)

    def predict(self, img: np.ndarray) -> np.ndarray:
        from dl03.seg_models.postproc import watershed_from_probs
        p = self.probs(img)
        return watershed_from_probs(p[..., 1] + p[..., 2], p[..., 1], **self.pp)


def load_info(d: Path = MODELS) -> dict:
    f = d / "model_info.json"
    return json.loads(f.read_text()) if f.exists() else {}
