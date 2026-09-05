"""Locust load test for /predict.

Closed-model (each simulated user loops request -> response -> next request),
unlike the old k6 open-model ramp. That's the right shape here: the concurrency
1/4/16 axis in the spec is "how many requests are in flight at once", which is
exactly what -u (user count) drives.

    locust -f bench/locustfile.py --headless -u 1  -r 1  -t 30s --host http://localhost:7860 \
        --csv reports/loadtest/c1  --html reports/loadtest/c1.html
    locust -f bench/locustfile.py --headless -u 4  -r 4  -t 30s --host http://localhost:7860 \
        --csv reports/loadtest/c4  --html reports/loadtest/c4.html
    locust -f bench/locustfile.py --headless -u 16 -r 16 -t 30s --host http://localhost:7860 \
        --csv reports/loadtest/c16 --html reports/loadtest/c16.html

Batching on/off delta: rerun each of the three with FDS_BATCH=false and compare
the "img/s" (Requests/s in the CSV) and p95 columns against the batch=true runs.

Env vars:
    FDS_IMAGE    path to a real frame (default: first image found under
                 dataset/images/test/)
    FDS_BACKEND  torch | onnx-fp32 | onnx-int8 (default: server's DEFAULT_BACKEND)
    FDS_BATCH    "true" | "false" (default: true)
"""
from __future__ import annotations

import os
from pathlib import Path

from locust import HttpUser, between, task


def _default_image() -> Path:
    test_dir = Path("dataset/images/test")
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        match = next(test_dir.glob(ext), None)
        if match is not None:
            return match
    raise FileNotFoundError(
        f"no image found under {test_dir} - set FDS_IMAGE=path/to/frame.jpg"
    )


IMAGE_PATH = Path(os.environ["FDS_IMAGE"]) if os.getenv("FDS_IMAGE") else _default_image()
BACKEND = os.getenv("FDS_BACKEND", "").strip()
BATCH = os.getenv("FDS_BATCH", "true").strip().lower() != "false"

if not IMAGE_PATH.exists():
    raise FileNotFoundError(
        f"FDS_IMAGE not found: {IMAGE_PATH} (set FDS_IMAGE=path/to/frame.jpg)"
    )
IMAGE_BYTES = IMAGE_PATH.read_bytes()


class PredictUser(HttpUser):
    # Closed-model: no think time. Each user fires the next request the instant
    # the previous one completes, so -u IS the concurrency level.
    wait_time = between(0, 0)

    @task
    def predict(self) -> None:
        params = {"batch": "true" if BATCH else "false"}
        if BACKEND:
            params["backend"] = BACKEND

        files = {"file": ("frame.jpg", IMAGE_BYTES, "image/jpeg")}
        name = f"/predict[backend={BACKEND or 'default'},batch={BATCH}]"

        with self.client.post(
            "/predict", params=params, files=files, name=name, catch_response=True
        ) as resp:
            if resp.status_code == 503:
                # Queue saturated - expected at high concurrency, not a script bug.
                # Left as a Locust "failure" on purpose so it's visible in the report.
                resp.failure("503 queue overflow")
                return
            if resp.status_code != 200:
                resp.failure(f"unexpected status {resp.status_code}")
                return
            body = resp.json()
            if not isinstance(body.get("detections"), list):
                resp.failure("response missing detections array")
