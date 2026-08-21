"""FastAPI inference service: /predict, /healthz, /metrics.

Config comes from env vars so the same image serves torch / onnx-fp32 / onnx-int8:

    BACKEND=onnx MODEL_PATH=models/best.onnx uvicorn app.main:app --port 8000
"""
from __future__ import annotations

import base64
import binascii
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import List, Optional

import cv2
import numpy as np
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, Response, UploadFile

from app import metrics
from app.backends.base import Detection, build_backend
from app.batching import BatchedPredictor, QueueOverflow
from app.schemas import (
    BoxOut,
    HealthResponse,
    PredictOptions,
    PredictRequest,
    PredictResponse,
)

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("app")

MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", 10 * 1024 * 1024))


class Settings:
    """Env-driven server settings."""

    def __init__(self) -> None:
        self.backend = os.getenv("BACKEND", "onnx")
        self.model_path = os.getenv("MODEL_PATH", "models/best.onnx")
        self.device = os.getenv("DEVICE", "cpu")
        self.imgsz = int(os.getenv("IMGSZ", 640))
        self.conf = float(os.getenv("CONF", 0.25))
        self.iou = float(os.getenv("IOU", 0.45))
        self.max_det = int(os.getenv("MAX_DET", 300))
        self.max_batch_size = int(os.getenv("MAX_BATCH_SIZE", 8))
        self.max_wait_ms = float(os.getenv("MAX_WAIT_MS", 15))
        self.max_queue_size = int(os.getenv("MAX_QUEUE_SIZE", 256))
        self.warmup_runs = int(os.getenv("WARMUP_RUNS", 3))


SETTINGS = Settings()

# Populated on startup.
STATE: dict = {"backend": None, "predictor": None, "warm": False, "started_at": time.time()}


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("loading %s backend from %s", SETTINGS.backend, SETTINGS.model_path)
    backend = build_backend(
        kind=SETTINGS.backend,
        weights=SETTINGS.model_path,
        imgsz=SETTINGS.imgsz,
        conf=SETTINGS.conf,
        iou=SETTINGS.iou,
        max_det=SETTINGS.max_det,
        device=SETTINGS.device,
    )
    predictor = BatchedPredictor(
        backend,
        max_batch_size=SETTINGS.max_batch_size,
        max_wait_ms=SETTINGS.max_wait_ms,
        max_queue_size=SETTINGS.max_queue_size,
    )
    await predictor.start()

    if SETTINGS.warmup_runs:
        backend.warmup(SETTINGS.warmup_runs)

    STATE.update(backend=backend, predictor=predictor, warm=True, started_at=time.time())
    metrics.set_model_info(backend.name, SETTINGS.model_path, SETTINGS.imgsz)
    log.info("ready: backend=%s classes=%s", backend.name, backend.class_names)

    try:
        yield
    finally:
        STATE["warm"] = False
        await predictor.stop()
        backend.close()
        log.info("shutdown complete")


app = FastAPI(
    title="football-detect-serve",
    description="Football object detection (ball / goalkeeper / player / referee).",
    version="0.1.0",
    lifespan=lifespan,
)


# ---------------- helpers ----------------


def _decode_image(data: bytes) -> np.ndarray:
    if not data:
        raise HTTPException(status_code=400, detail="empty image payload")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="image exceeds MAX_UPLOAD_BYTES")

    img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="could not decode image bytes")
    return img


def _decode_b64(payload: str) -> bytes:
    if "," in payload[:64] and payload.lstrip().startswith("data:"):
        payload = payload.split(",", 1)[1]
    try:
        return base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=400, detail="image_b64 is not valid base64") from exc


async def _fetch_url(url: str) -> bytes:
    import asyncio

    import requests

    def _get() -> bytes:
        resp = requests.get(url, timeout=10, stream=True)
        resp.raise_for_status()
        return resp.raw.read(MAX_UPLOAD_BYTES + 1, decode_content=True)

    try:
        return await asyncio.get_running_loop().run_in_executor(None, _get)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"could not fetch image_url: {exc}") from exc


def _get_predictor() -> BatchedPredictor:
    predictor = STATE.get("predictor")
    if predictor is None or not STATE.get("warm"):
        raise HTTPException(status_code=503, detail="model is not loaded yet")
    return predictor


def _apply_overrides(opts: PredictOptions) -> None:
    """Per-request thresholds.

    Postprocessing happens inside the shared backend object, so overrides are
    applied to it directly. Only safe because the batch worker is single-threaded;
    a mismatched override affects at most the batch it rode in with.
    """
    backend = STATE["backend"]
    if opts.conf is not None:
        backend.conf = opts.conf
    if opts.iou is not None:
        backend.iou = opts.iou
    if opts.max_det is not None:
        backend.max_det = opts.max_det


def _reset_overrides() -> None:
    backend = STATE["backend"]
    backend.conf, backend.iou, backend.max_det = (
        SETTINGS.conf,
        SETTINGS.iou,
        SETTINGS.max_det,
    )


def _to_response(
    detections: List[Detection],
    img: np.ndarray,
    batch_size: int,
    infer_ms: float,
    total_ms: float,
    request_id: str,
) -> PredictResponse:
    return PredictResponse(
        detections=[BoxOut(**vars(d)) for d in detections],
        num_detections=len(detections),
        image_width=int(img.shape[1]),
        image_height=int(img.shape[0]),
        backend=STATE["backend"].name,
        model=SETTINGS.model_path,
        inference_ms=round(infer_ms, 3),
        total_ms=round(total_ms, 3),
        batch_size=batch_size,
        request_id=request_id,
    )


# ---------------- endpoints ----------------


@app.post("/predict", response_model=PredictResponse)
async def predict(
    request: Request,
    file: Optional[UploadFile] = File(default=None, description="Image file upload."),
    conf: Optional[float] = Form(default=None),
    iou: Optional[float] = Form(default=None),
    max_det: Optional[int] = Form(default=None),
    predictor: BatchedPredictor = Depends(_get_predictor),
) -> PredictResponse:
    """Detect objects in one image.

    Accepts either `multipart/form-data` with a `file` field, or a JSON body
    with `image_b64` / `image_url` (see PredictRequest).
    """
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
    t0 = time.perf_counter()
    metrics.INFLIGHT.inc()

    try:
        if file is not None:
            data = await file.read()
            options = PredictOptions(conf=conf, iou=iou, max_det=max_det)
        else:
            try:
                body = PredictRequest(**(await request.json()))
            except HTTPException:
                raise
            except Exception as exc:  # noqa: BLE001 - malformed/absent JSON body
                raise HTTPException(
                    status_code=400,
                    detail="provide a multipart 'file' or a JSON body with image_b64/image_url",
                ) from exc

            options = body.options
            if body.image_b64:
                data = _decode_b64(body.image_b64)
            elif body.image_url:
                data = await _fetch_url(body.image_url)
            else:
                raise HTTPException(
                    status_code=400, detail="one of image_b64 or image_url is required"
                )

        img = _decode_image(data)
        _apply_overrides(options)
        try:
            detections, batch_size, infer_ms = await predictor.predict(img)
        finally:
            _reset_overrides()

        total_ms = (time.perf_counter() - t0) * 1000.0
        metrics.REQUESTS.labels(endpoint="/predict", status="200").inc()
        return _to_response(detections, img, batch_size, infer_ms, total_ms, request_id)

    except QueueOverflow as exc:
        metrics.REQUESTS.labels(endpoint="/predict", status="503").inc()
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except HTTPException as exc:
        metrics.REQUESTS.labels(endpoint="/predict", status=str(exc.status_code)).inc()
        raise
    except Exception as exc:  # noqa: BLE001
        log.exception("predict failed request_id=%s", request_id)
        metrics.REQUESTS.labels(endpoint="/predict", status="500").inc()
        raise HTTPException(status_code=500, detail="inference error") from exc
    finally:
        metrics.INFLIGHT.dec()
        metrics.REQUEST_LATENCY.labels(endpoint="/predict").observe(time.perf_counter() - t0)


@app.get("/healthz", response_model=HealthResponse)
async def healthz() -> HealthResponse:
    backend = STATE.get("backend")
    warm = bool(STATE.get("warm"))
    if backend is None:
        metrics.REQUESTS.labels(endpoint="/healthz", status="503").inc()
        raise HTTPException(status_code=503, detail="model is not loaded yet")

    metrics.REQUESTS.labels(endpoint="/healthz", status="200").inc()
    return HealthResponse(
        status="ok" if warm else "loading",
        backend=backend.name,
        model=SETTINGS.model_path,
        imgsz=SETTINGS.imgsz,
        classes=backend.class_names,
        warm=warm,
        uptime_s=round(time.time() - STATE["started_at"], 1),
    )


@app.get("/metrics")
async def prometheus_metrics() -> Response:
    return Response(content=metrics.render_latest(), media_type=metrics.CONTENT_TYPE_LATEST)


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 8000)),
        workers=1,  # batching state is per-process; scale with replicas, not workers
    )
