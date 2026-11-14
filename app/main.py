"""API de inferencia: /health, /ready, /predict_text, /predict_image, /model_info."""
from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException, Request

from .inference import MODELS, ImageModel, TextModel, load_info
from .schemas import Health, ImageIn, ImageOut, Ready, TextIn, TextOut

TIMEOUT_S = float(os.environ.get("INFERENCE_TIMEOUT_S", "30"))
MAX_PIXELS = int(os.environ.get("MAX_PIXELS", str(1024 * 1024)))


class JsonFormatter(logging.Formatter):
    def format(self, r: logging.LogRecord) -> str:
        d = {"t": self.formatTime(r), "level": r.levelname, "msg": r.getMessage()}
        d.update(getattr(r, "extra_fields", {}))
        return json.dumps(d, ensure_ascii=False)


log = logging.getLogger("dl03.api")
_h = logging.StreamHandler()
_h.setFormatter(JsonFormatter())
log.handlers = [_h]
log.setLevel(logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.text = app.state.image = None
    try:
        app.state.text = TextModel(MODELS)
    except Exception as e:  # el servicio arranca aunque falte un modelo; /ready lo refleja
        log.warning(f"modelo de texto no disponible: {e}")
    try:
        app.state.image = ImageModel(MODELS)
    except Exception as e:
        log.warning(f"modelo de imagen no disponible: {e}")
    app.state.info = load_info()
    yield


app = FastAPI(title="DL03 · Segmentación celular y clasificación de hojas de vida", version="1.0.0",
              lifespan=lifespan)


@app.middleware("http")
async def request_id(request: Request, call_next):
    rid = request.headers.get("x-request-id", str(uuid.uuid4()))
    request.state.rid = rid
    t0 = time.perf_counter()
    resp = await call_next(request)
    ms = (time.perf_counter() - t0) * 1e3
    resp.headers["x-request-id"] = rid
    log.info("request", extra={"extra_fields": {"request_id": rid, "path": request.url.path,
                                                "status": resp.status_code, "latency_ms": round(ms, 2)}})
    return resp


async def _run(fn, *a):
    try:
        return await asyncio.wait_for(asyncio.to_thread(fn, *a), timeout=TIMEOUT_S)
    except asyncio.TimeoutError:
        raise HTTPException(504, "la inferencia excedió el tiempo límite")


@app.get("/health", response_model=Health)
def health():
    return {"status": "ok"}


@app.get("/ready", response_model=Ready)
def ready():
    t, i = app.state.text is not None, app.state.image is not None
    return {"ready": t and i, "text_model": t, "image_model": i}


@app.get("/model_info")
def model_info():
    return app.state.info


@app.post("/predict_text", response_model=TextOut)
async def predict_text(payload: TextIn, request: Request):
    if app.state.text is None:
        raise HTTPException(503, "modelo de texto no cargado")
    t0 = time.perf_counter()
    probs = await _run(app.state.text.predict_proba, payload.text)
    return TextOut(label=max(probs, key=probs.get), probabilities=probs, model_version=app.state.text.version,
                   request_id=request.state.rid, latency_ms=(time.perf_counter() - t0) * 1e3)


@app.post("/predict_image", response_model=ImageOut)
async def predict_image(payload: ImageIn, request: Request):
    if app.state.image is None:
        raise HTTPException(503, "modelo de imagen no cargado")
    from PIL import Image
    try:
        im = Image.open(io.BytesIO(base64.b64decode(payload.image_png_base64)))
        img = np.asarray(im.convert("L"), dtype=np.float32)
    except Exception:
        raise HTTPException(422, "no es una imagen PNG válida en base64")
    if img.size > MAX_PIXELS or min(img.shape) < 32:
        raise HTTPException(422, f"tamaño de imagen fuera de rango: {img.shape}")
    t0 = time.perf_counter()
    lbl = await _run(app.state.image.predict, img)
    from dl03.rle import label_map_to_rles
    rles = label_map_to_rles(lbl)
    return ImageOut(n_instances=len(rles), rles=rles, height=img.shape[0], width=img.shape[1],
                    model_version=app.state.image.version, request_id=request.state.rid,
                    latency_ms=(time.perf_counter() - t0) * 1e3)
