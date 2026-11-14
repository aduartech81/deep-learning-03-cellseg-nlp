"""Regresión del modelo desplegado sobre un conjunto dorado: las métricas no pueden caer más de ε respecto de las
registradas al exportar (capítulo 9). Se omite si los artefactos reales aún no existen."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
M = ROOT / "app" / "models"
G = ROOT / "app" / "golden"
EPS = 0.02
need = pytest.mark.skipif(not (G / "golden.json").exists() or not (M / "text_model.joblib").exists(),
                          reason="artefactos reales no exportados todavía")


@need
def test_text_golden():
    import os
    os.environ["MODELS_DIR"] = str(M)
    from sklearn.metrics import f1_score

    from app.inference import TextModel
    ref = json.loads((G / "golden.json").read_text())
    df = pd.read_csv(G / "texts.csv")
    tm = TextModel(M)
    pred = [max(p, key=p.get) for p in map(tm.predict_proba, df["text"])]
    f1 = f1_score(df["label"], pred, average="macro")
    assert f1 >= ref["text_f1_macro"] - EPS, (f1, ref["text_f1_macro"])


@need
def test_image_golden():
    from skimage import io

    from app.inference import ImageModel
    from dl03.seg_metrics import sartorius_score
    ref = json.loads((G / "golden.json").read_text())
    im = ImageModel(M)
    aps = [sartorius_score(np.load(f.with_suffix(".npz"))["lbl"], im.predict(io.imread(f).astype(np.float32)))
           for f in sorted(G.glob("img_*.png"))]
    assert np.mean(aps) >= ref["image_ap"] - EPS, (np.mean(aps), ref["image_ap"])
