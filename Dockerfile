# Serving image: ONNX Runtime CPU only. Torch is NOT installed here -- it triples
# the image size and the served artifact is the ONNX graph. Use requirements.txt
# on the training host for the full toolchain.
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# libGL/libglib are needed by opencv even in the headless build
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- deps layer: only the serving subset, so it caches independently of code ---
RUN pip install --no-cache-dir \
        "fastapi" \
        "uvicorn[standard]" \
        "pydantic>=2" \
        "python-multipart" \
        "prometheus-client" \
        "onnxruntime" \
        "numpy<2" \
        "opencv-python-headless" \
        "pyyaml" \
        "requests"

# --- app layer ---
COPY app ./app
COPY configs ./configs

# Bake the model in for immutable deploys; mount over /app/models to swap it.
COPY models/ ./models/

RUN useradd --create-home --uid 10001 appuser && chown -R appuser:appuser /app
USER appuser

ENV BACKEND=onnx \
    MODEL_PATH=models/best.onnx \
    DEVICE=cpu \
    IMGSZ=640 \
    CONF=0.25 \
    IOU=0.45 \
    MAX_BATCH_SIZE=8 \
    MAX_WAIT_MS=15 \
    PORT=8000 \
    OMP_NUM_THREADS=4

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:8000/healthz || exit 1

# One worker: the batching queue is per-process. Scale with replicas.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --workers 1"]
