"""FastAPI inference service: /predict, /healthz, /metrics.

One process serves several backends at once so a single deploy can demo and
load-test torch / onnx-fp32 / onnx-int8 side by side. Pick per request with
`?backend=<name>`; omit it for DEFAULT_BACKEND. `?batch=false` bypasses the
dynamic batcher (for the batching on/off throughput comparison).

    MODELS="torch=models/football_detection_v3.pt,onnx-fp32=models/football_detection_v3.onnx,onnx-int8=models/football_detection_v3_int8.onnx" \
    DEFAULT_BACKEND=onnx-fp32 uvicorn app.main:app --port 7860
"""
from __future__ import annotations

import asyncio
import base64
import binascii
import ipaddress
import logging
import os
import socket
import time
import uuid
from contextlib import asynccontextmanager
from typing import Dict, List, Optional
from urllib.parse import urlparse

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Query, Request, Response, UploadFile

from app import metrics
from app.backends.base import Detection, DetectorBackend, build_backend
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

_DEFAULT_MODELS = (
    "torch=models/football_detection_v3.pt,"
    "onnx-fp32=models/football_detection_v3.onnx,"
    "onnx-int8=models/football_detection_v3_int8.onnx"
)


def _parse_models(spec: str) -> Dict[str, str]:
    """'name=path,name=path' -> {name: path}, order preserved."""
    out: Dict[str, str] = {}
    for pair in spec.split(","):
        pair = pair.strip()
        if not pair:
            continue
        if "=" not in pair:
            raise ValueError(f"bad MODELS entry {pair!r} (want 'name=path')")
        name, path = pair.split("=", 1)
        out[name.strip()] = path.strip()
    return out


def _kind_of(name: str) -> str:
    return "torch" if name.startswith("torch") else "onnx"


class Settings:
    """Env-driven server settings."""

    def __init__(self) -> None:
        self.models = _parse_models(os.getenv("MODELS", "").strip() or _DEFAULT_MODELS)
        self.default_backend = os.getenv("DEFAULT_BACKEND", next(iter(self.models)))
        if self.default_backend not in self.models:
            raise ValueError(f"DEFAULT_BACKEND={self.default_backend} not in MODELS {list(self.models)}")
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

# Populated on startup: name -> {"backend": DetectorBackend, "predictor": BatchedPredictor}
STATE: dict = {"registry": {}, "default": SETTINGS.default_backend, "warm": False, "started_at": time.time()}


@asynccontextmanager
async def lifespan(app: FastAPI):
    registry: Dict[str, dict] = {}
    for name, path in SETTINGS.models.items():
        log.info("loading backend %s (%s) from %s", name, _kind_of(name), path)
        backend = build_backend(
            kind=_kind_of(name),
            weights=path,
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
        registry[name] = {"backend": backend, "predictor": predictor}
        metrics.set_model_info(backend.name, path, SETTINGS.imgsz)
        log.info("ready: %s -> %s classes=%s", name, backend.name, backend.class_names)

    STATE.update(registry=registry, warm=True, started_at=time.time())
    log.info("service ready: backends=%s default=%s", list(registry), STATE["default"])

    try:
        yield
    finally:
        STATE["warm"] = False
        for entry in registry.values():
            await entry["predictor"].stop()
            entry["backend"].close()
        log.info("shutdown complete")


app = FastAPI(
    title="football-detect-serve",
    description="Football object detection (ball / goalkeeper / player / referee).",
    version="0.2.0",
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


def _is_blocked_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """True for anything that isn't a normal public address - loopback, private
    ranges, link-local (this is what covers cloud metadata endpoints like
    169.254.169.254), reserved, multicast, unspecified.
    """
    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
    )


def _validate_image_url(url: str) -> None:
    """SSRF guard for `image_url`: only http(s), and refuse to fetch a hostname
    that resolves to a private/internal/loopback address. Not exhaustive (DNS
    rebinding between this check and the actual request, or a redirect to an
    internal address, aren't covered) - this service isn't deployed publicly,
    so this is defense in depth on an input that was previously unvalidated,
    not a hardened boundary.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"unsupported URL scheme {parsed.scheme!r} (only http/https allowed)")
    if not parsed.hostname:
        raise ValueError("image_url has no hostname")

    try:
        addr_info = socket.getaddrinfo(parsed.hostname, None)
    except socket.gaierror as exc:
        raise ValueError(f"could not resolve host {parsed.hostname!r}: {exc}") from exc

    for *_, sockaddr in addr_info:
        ip = ipaddress.ip_address(sockaddr[0])
        if _is_blocked_ip(ip):
            raise ValueError(f"refusing to fetch image_url: {ip} is a private/internal address")


async def _fetch_url(url: str) -> bytes:
    import requests

    try:
        _validate_image_url(url)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    def _get() -> bytes:
        # allow_redirects=False: a validated hostname could still redirect
        # somewhere internal - don't follow it rather than re-validate a chain.
        resp = requests.get(url, timeout=10, stream=True, allow_redirects=False)
        resp.raise_for_status()
        return resp.raw.read(MAX_UPLOAD_BYTES + 1, decode_content=True)

    try:
        return await asyncio.get_running_loop().run_in_executor(None, _get)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"could not fetch image_url: {exc}") from exc


def _resolve(name: Optional[str]) -> tuple[str, DetectorBackend, BatchedPredictor]:
    """Map ?backend= to a loaded (name, backend, predictor). Raises 400/503."""
    if not STATE.get("warm"):
        raise HTTPException(status_code=503, detail="models are not loaded yet")
    key = name or STATE["default"]
    entry = STATE["registry"].get(key)
    if entry is None:
        raise HTTPException(
            status_code=400,
            detail=f"unknown backend {key!r}; available: {list(STATE['registry'])}",
        )
    return key, entry["backend"], entry["predictor"]


def _to_response(
    detections: List[Detection],
    img: np.ndarray,
    *,
    backend_name: str,
    model_path: str,
    batch_size: int,
    batched: bool,
    infer_ms: float,
    total_ms: float,
    request_id: str,
) -> PredictResponse:
    return PredictResponse(
        detections=[BoxOut(**vars(d)) for d in detections],
        num_detections=len(detections),
        image_width=int(img.shape[1]),
        image_height=int(img.shape[0]),
        backend=backend_name,
        model=model_path,
        inference_ms=round(infer_ms, 3),
        total_ms=round(total_ms, 3),
        batch_size=batch_size,
        batched=batched,
        request_id=request_id,
    )


def _locked_predict(
    backend: DetectorBackend,
    images: List[np.ndarray],
    conf: Optional[float] = None,
    iou: Optional[float] = None,
    max_det: Optional[int] = None,
) -> List[List[Detection]]:
    """backend.predict() under backend.predict_lock - see app/backends/base.py.

    Runs in an executor thread; holding the lock here keeps concurrent
    ?batch=false requests (and the batch worker, for the same backend) from
    ever calling predict() on this instance at the same time. conf/iou/max_det
    are this request's own overrides, passed straight through as call
    arguments - never written onto the shared backend object (see
    DetectorBackend.predict's docstring for why that used to be a bug).
    """
    with backend.predict_lock:
        return backend.predict(images, conf=conf, iou=iou, max_det=max_det)


async def _infer_unbatched(
    backend: DetectorBackend,
    img: np.ndarray,
    *,
    conf: Optional[float] = None,
    iou: Optional[float] = None,
    max_det: Optional[int] = None,
) -> tuple[List[Detection], float]:
    """Skip the queue: one image straight through backend.predict()."""
    loop = asyncio.get_running_loop()
    t0 = time.perf_counter()
    results = await loop.run_in_executor(None, _locked_predict, backend, [img], conf, iou, max_det)
    infer_ms = (time.perf_counter() - t0) * 1000.0
    dets = results[0] if results else []
    for det in dets:
        metrics.DETECTIONS.labels(class_name=det.class_name).inc()
    metrics.INFERENCE_LATENCY.labels(backend=backend.name).observe(infer_ms / 1000.0)
    return dets, infer_ms


# ---------------- endpoints ----------------


@app.post("/predict", response_model=PredictResponse)
async def predict(
    request: Request,
    file: Optional[UploadFile] = File(default=None, description="Image file upload."),
    conf: Optional[float] = Form(default=None),
    iou: Optional[float] = Form(default=None),
    max_det: Optional[int] = Form(default=None),
    backend: Optional[str] = Query(default=None, description="Backend name; omit for the default."),
    batch: bool = Query(default=True, description="False bypasses the dynamic batcher."),
) -> PredictResponse:
    """Detect objects in one image.

    Accepts `multipart/form-data` with a `file` field, or a JSON body with
    `image_b64` / `image_url` (see PredictRequest).
    """
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
    t0 = time.perf_counter()
    metrics.INFLIGHT.inc()
    bname = backend or STATE.get("default", "-")

    try:
        name, be, predictor = _resolve(backend)
        bname = name

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
                raise HTTPException(status_code=400, detail="one of image_b64 or image_url is required")

        img = _decode_image(data)
        if batch:
            detections, batch_size, infer_ms = await predictor.predict(
                img, conf=options.conf, iou=options.iou, max_det=options.max_det
            )
        else:
            detections, infer_ms = await _infer_unbatched(
                be, img, conf=options.conf, iou=options.iou, max_det=options.max_det
            )
            batch_size = 1

        total_ms = (time.perf_counter() - t0) * 1000.0
        metrics.REQUESTS.labels(endpoint="/predict", backend=bname, status="200").inc()
        return _to_response(
            detections,
            img,
            backend_name=be.name,
            model_path=SETTINGS.models[name],
            batch_size=batch_size,
            batched=batch,
            infer_ms=infer_ms,
            total_ms=total_ms,
            request_id=request_id,
        )

    except QueueOverflow as exc:
        metrics.REQUESTS.labels(endpoint="/predict", backend=bname, status="503").inc()
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except HTTPException as exc:
        metrics.REQUESTS.labels(endpoint="/predict", backend=bname, status=str(exc.status_code)).inc()
        raise
    except Exception as exc:  # noqa: BLE001
        log.exception("predict failed request_id=%s", request_id)
        metrics.REQUESTS.labels(endpoint="/predict", backend=bname, status="500").inc()
        raise HTTPException(status_code=500, detail="inference error") from exc
    finally:
        metrics.INFLIGHT.dec()
        metrics.REQUEST_LATENCY.labels(endpoint="/predict", backend=bname).observe(time.perf_counter() - t0)


@app.get("/healthz", response_model=HealthResponse)
async def healthz() -> HealthResponse:
    registry = STATE.get("registry") or {}
    warm = bool(STATE.get("warm"))
    if not registry:
        metrics.REQUESTS.labels(endpoint="/healthz", backend="-", status="503").inc()
        raise HTTPException(status_code=503, detail="models are not loaded yet")

    default = STATE["default"]
    default_be = registry[default]["backend"]
    metrics.REQUESTS.labels(endpoint="/healthz", backend="-", status="200").inc()
    return HealthResponse(
        status="ok" if warm else "loading",
        backend=default,
        backends={n: SETTINGS.models[n] for n in registry},
        model=SETTINGS.models[default],
        imgsz=SETTINGS.imgsz,
        classes=default_be.class_names,
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
        port=int(os.getenv("PORT", 7860)),
        workers=1,  # batching state is per-process; scale with replicas, not workers
    )
