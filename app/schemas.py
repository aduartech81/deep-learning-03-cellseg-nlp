"""Esquemas Pydantic de la API (validación de entradas y salidas)."""
from __future__ import annotations

from pydantic import BaseModel, Field


class TextIn(BaseModel):
    text: str = Field(..., min_length=1, max_length=20_000, description="Texto de la hoja de vida (inglés)")


class TextOut(BaseModel):
    label: str
    probabilities: dict[str, float]
    model_version: str
    request_id: str
    latency_ms: float


class ImageIn(BaseModel):
    image_png_base64: str = Field(..., min_length=16, description="PNG en escala de grises codificado en base64")


class ImageOut(BaseModel):
    n_instances: int
    rles: list[str]
    height: int
    width: int
    model_version: str
    request_id: str
    latency_ms: float


class Health(BaseModel):
    status: str


class Ready(BaseModel):
    ready: bool
    text_model: bool
    image_model: bool
