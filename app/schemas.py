"""Pydantic request/response models for the inference API."""
from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class PredictOptions(BaseModel):
    """Per-request overrides. Omitted fields fall back to server defaults."""

    conf: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    iou: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    max_det: Optional[int] = Field(default=None, ge=1, le=1000)


class PredictRequest(BaseModel):
    """JSON body for /predict. Use the multipart form for raw file uploads."""

    image_b64: Optional[str] = Field(
        default=None, description="Base64-encoded image bytes (data URI prefix allowed)."
    )
    image_url: Optional[str] = Field(default=None, description="HTTP(S) URL of an image.")
    options: PredictOptions = Field(default_factory=PredictOptions)


class BoxOut(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    class_id: int
    class_name: str


class PredictResponse(BaseModel):
    detections: List[BoxOut]
    num_detections: int
    image_width: int
    image_height: int
    backend: str
    model: str
    inference_ms: float = Field(description="Model forward pass only.")
    total_ms: float = Field(description="Decode + preprocess + infer + postprocess.")
    batch_size: int = Field(description="Requests coalesced into this forward pass.")
    request_id: str


class HealthResponse(BaseModel):
    status: str
    backend: str
    model: str
    imgsz: int
    classes: Dict[int, str]
    warm: bool
    uptime_s: float


class ErrorResponse(BaseModel):
    detail: str
    request_id: Optional[str] = None
