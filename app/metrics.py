"""Prometheus instrumentation. Import-safe: metrics are module-level singletons."""
from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest
from prometheus_client import CONTENT_TYPE_LATEST  # re-exported for main.py

REGISTRY = CollectorRegistry(auto_describe=True)

# --- request level ---
REQUESTS = Counter(
    "fds_requests_total",
    "Requests received, by endpoint, backend and outcome.",
    ["endpoint", "backend", "status"],
    registry=REGISTRY,
)
REQUEST_LATENCY = Histogram(
    "fds_request_latency_seconds",
    "End-to-end request latency (decode -> response).",
    ["endpoint", "backend"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
    registry=REGISTRY,
)

# --- model level ---
INFERENCE_LATENCY = Histogram(
    "fds_inference_latency_seconds",
    "Model forward pass latency, excluding queueing.",
    ["backend"],
    buckets=(0.002, 0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28),
    registry=REGISTRY,
)
PREPROCESS_LATENCY = Histogram(
    "fds_preprocess_latency_seconds",
    "Letterbox + normalize latency per batch.",
    buckets=(0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1),
    registry=REGISTRY,
)
DETECTIONS = Counter(
    "fds_detections_total",
    "Detections returned, by class.",
    ["class_name"],
    registry=REGISTRY,
)

# --- batching ---
QUEUE_WAIT = Histogram(
    "fds_queue_wait_seconds",
    "Time a request spent waiting to be batched.",
    buckets=(0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25),
    registry=REGISTRY,
)
BATCH_SIZE = Histogram(
    "fds_batch_size",
    "Requests coalesced per forward pass.",
    buckets=(1, 2, 3, 4, 6, 8, 12, 16, 32),
    registry=REGISTRY,
)
QUEUE_DEPTH = Gauge(
    "fds_queue_depth",
    "Requests currently waiting in the batch queue.",
    registry=REGISTRY,
)
INFLIGHT = Gauge(
    "fds_inflight_requests",
    "Requests currently being served.",
    registry=REGISTRY,
)

# --- process info ---
MODEL_INFO = Gauge(
    "fds_model_info",
    "Always 1; labels carry the served model identity.",
    ["backend", "model", "imgsz"],
    registry=REGISTRY,
)


def set_model_info(backend: str, model: str, imgsz: int) -> None:
    MODEL_INFO.labels(backend=backend, model=model, imgsz=str(imgsz)).set(1)


def render_latest() -> bytes:
    return generate_latest(REGISTRY)


__all__ = [
    "REGISTRY",
    "CONTENT_TYPE_LATEST",
    "REQUESTS",
    "REQUEST_LATENCY",
    "INFERENCE_LATENCY",
    "PREPROCESS_LATENCY",
    "DETECTIONS",
    "QUEUE_WAIT",
    "BATCH_SIZE",
    "QUEUE_DEPTH",
    "INFLIGHT",
    "set_model_info",
    "render_latest",
]
