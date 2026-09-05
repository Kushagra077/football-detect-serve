# Local-only serving image for the multi-backend FastAPI service (listens on 7860).
#
# NOT deployed: HF Docker Spaces are paid, so the public demo is an HF Gradio SDK
# app (app/gradio_app.py) instead. This image exists to run the service under the
# Locust load test locally and to produce the latency/throughput table. Port 7860
# is kept only so the local container matches what the Space would have looked like.
#
# Both backends decode through ultralytics (YOLO(...).predict()), so torch IS
# required now -- the old "ONNX graph only, no torch" premise is dead (see
# PROGRESS.md, Step 7). CPU-only torch wheel keeps it to ~1 GB.
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# libGL/libglib: opencv needs them even in the headless build. curl: healthcheck.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgl1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- deps layer: caches independently of app code ---
# torch/torchvision from the CPU index so we don't pull the CUDA build.
RUN pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        "torch" \
        "torchvision" \
    && pip install --no-cache-dir \
        "ultralytics" \
        "onnx" \
        "onnxruntime" \
        "fastapi" \
        "uvicorn[standard]" \
        "pydantic>=2" \
        "python-multipart" \
        "prometheus-client" \
        "numpy<2" \
        "opencv-python-headless" \
        "pyyaml" \
        "requests"

# --- app layer ---
COPY app ./app
COPY configs ./configs

# Bake in only the three files the default registry (main.py _DEFAULT_MODELS)
# loads: torch + onnx-fp32 + onnx-int8 on v1. Mount over /app/models to swap.
COPY models/football_detection_v1.pt ./models/
COPY models/football_detection_v1.onnx ./models/
COPY models/football_detection_v1_int8.onnx ./models/

RUN useradd --create-home --uid 10001 appuser && chown -R appuser:appuser /app
USER appuser

# HF Spaces gives the container a writable home; point ultralytics/matplotlib there.
ENV HOME=/home/appuser \
    YOLO_CONFIG_DIR=/home/appuser/.config/Ultralytics \
    MPLCONFIGDIR=/home/appuser/.config/matplotlib

ENV DEFAULT_BACKEND=onnx-fp32 \
    DEVICE=cpu \
    IMGSZ=640 \
    CONF=0.25 \
    IOU=0.45 \
    MAX_BATCH_SIZE=8 \
    MAX_WAIT_MS=15 \
    PORT=7860 \
    OMP_NUM_THREADS=4

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD curl -fsS http://localhost:7860/healthz || exit 1

# One worker: the batching queue is per-process. Scale with replicas.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --workers 1"]
