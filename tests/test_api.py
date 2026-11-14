"""Pruebas de la API con modelos diminutos construidos en la prueba (sin TensorFlow): esquema, validación de
entradas, endpoints y tiempo límite."""
import base64
import io
import json
import time

import numpy as np
import pytest


def _tiny_onnx(path):
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    w = numpy_helper.from_array(np.array([[[[0.0]]], [[[4.0]]], [[[-2.0]]]], np.float32), "W")
    b = numpy_helper.from_array(np.array([0.0, -2.0, 0.0], np.float32), "B")
    nodes = [helper.make_node("Transpose", ["image"], ["nchw"], perm=[0, 3, 1, 2]),
             helper.make_node("Conv", ["nchw", "W", "B"], ["logits"], kernel_shape=[1, 1]),
             helper.make_node("Softmax", ["logits"], ["prob"], axis=1),
             helper.make_node("Transpose", ["prob"], ["out"], perm=[0, 2, 3, 1])]
    g = helper.make_graph(nodes, "tiny", [helper.make_tensor_value_info("image", TensorProto.FLOAT,
                                                                        [1, None, None, 1])],
                          [helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, None, None, 3])], [w, b])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 8
    onnx.save(m, path)


@pytest.fixture(scope="module")
def client(tmp_path_factory, monkeypatch_module=None):
    import joblib
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    from dl03.nlp_models import SkModel
    d = tmp_path_factory.mktemp("models")
    texts = ["python developer software", "chef kitchen cooking", "nurse patient care", "software engineer code"]
    yy = [0, 1, 2, 0]
    vec = TfidfVectorizer().fit(texts)
    clf = LogisticRegression().fit(vec.transform(texts), yy)
    joblib.dump(SkModel(vec, clf, "standard", 3), d / "text_model.joblib")
    (d / "text_model.json").write_text(json.dumps({"classes": ["IT", "CHEF", "HEALTHCARE"], "version": "test"}))
    _tiny_onnx(str(d / "unetpp.onnx"))
    (d / "image_model.json").write_text(json.dumps({"postproc": {"thr_fg": 0.5, "thr_in": 0.5, "h": 1.0,
                                                                 "min_size": 5}, "resolution": "native",
                                                    "version": "test"}))
    (d / "model_info.json").write_text(json.dumps({"version": "test"}))
    import os
    os.environ["MODELS_DIR"] = str(d)
    import importlib

    import app.inference
    import app.main
    importlib.reload(app.inference)
    importlib.reload(app.main)
    from fastapi.testclient import TestClient
    with TestClient(app.main.app) as c:
        yield c


def _png(arr):
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(arr.astype(np.uint8)).save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


def test_health_ready_info(client):
    assert client.get("/health").json() == {"status": "ok"}
    r = client.get("/ready").json()
    assert r["ready"] and r["text_model"] and r["image_model"]
    assert client.get("/model_info").json()["version"] == "test"


def test_predict_text_schema(client):
    r = client.post("/predict_text", json={"text": "senior python software developer"})
    assert r.status_code == 200
    j = r.json()
    assert j["label"] == "IT" and abs(sum(j["probabilities"].values()) - 1) < 1e-6
    assert set(j) == {"label", "probabilities", "model_version", "request_id", "latency_ms"}
    assert r.headers["x-request-id"] == j["request_id"]


def test_predict_text_validation(client):
    assert client.post("/predict_text", json={"text": ""}).status_code == 422
    assert client.post("/predict_text", json={}).status_code == 422
    assert client.post("/predict_text", json={"text": "x" * 20_001}).status_code == 422


def test_predict_image(client):
    img = np.full((96, 128), 20, np.uint8)
    img[20:50, 20:50] = 220
    img[60:90, 80:120] = 220
    r = client.post("/predict_image", json={"image_png_base64": _png(img)})
    assert r.status_code == 200
    j = r.json()
    assert j["height"] == 96 and j["width"] == 128 and j["n_instances"] == len(j["rles"]) >= 1


def test_predict_image_validation(client):
    assert client.post("/predict_image", json={"image_png_base64": "bm90IGFuIGltYWdlIGF0IGFsbA=="}).status_code == 422
    assert client.post("/predict_image", json={"image_png_base64": _png(np.zeros((8, 8)))}).status_code == 422


def test_timeout(client, monkeypatch):
    import app.main
    monkeypatch.setattr(app.main, "TIMEOUT_S", 0.05)
    monkeypatch.setattr(app.main.app.state.text, "predict_proba", lambda t: time.sleep(0.5) or {"IT": 1.0})
    assert client.post("/predict_text", json={"text": "slow"}).status_code == 504
