# football-detect-serve

Football object detection (ball / goalkeeper / player / referee) trained with YOLOv8,
exported to ONNX, quantized to INT8, and served behind a batching FastAPI service.

The point of the repo is the path from checkpoint to endpoint: every optimization step has
a gate, and every backend is measured through the same code.

## Design rules

1. **One inference path.** Preprocess, decode and NMS live in [app/backends/base.py](app/backends/base.py).
   Torch and ONNX subclasses implement only the forward pass. NMS is *not* baked into the ONNX
   graph, because then the graph would have a decode path the torch model doesn't.
2. **Parity is a gate, not a report.** [scripts/check_parity.py](scripts/check_parity.py) exits
   non-zero if the ONNX output drifts from torch. A failed parity check invalidates every mAP and
   latency number downstream.
3. **One evaluator for every backend.** [scripts/eval_map.py](scripts/eval_map.py) computes
   COCO-style AP itself rather than calling the ultralytics validator, which only accepts torch
   models. Comparing an ultralytics number to a hand-rolled number is not a comparison.
4. **Quantize against real data.** Static INT8 calibration samples the val split through the
   serving preprocess ([scripts/quantize_int8.py](scripts/quantize_int8.py)), and the result is
   gated on mAP drop.

## Layout

```
configs/train.yaml        epochs, imgsz, batch, augmentation, gate thresholds
scripts/prepare_data.py   download + verify class map + count instances per class
scripts/train.py          thin wrapper over ultralytics
scripts/export_onnx.py    pt -> onnx (dynamic batch, simplified)
scripts/check_parity.py   GATE: torch vs onnx output diff
scripts/quantize_int8.py  ORT static quantization + calibration
scripts/eval_map.py       per-class mAP for ANY backend, one code path
scripts/benchmark.py      latency harness, writes reports/latency.json
app/main.py               FastAPI: /predict /healthz /metrics
app/batching.py           asyncio queue + max-wait window
app/backends/             base + torch + onnx (fp32 and int8)
bench/load_test.js        k6
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The serving container installs only the ONNX Runtime subset — see the [Dockerfile](Dockerfile).

## Pipeline

```bash
# 1. dataset: verifies the class map, counts instances, flags bad labels
python scripts/prepare_data.py

# 2. train (all hyperparameters in configs/train.yaml)
python scripts/train.py --set epochs=100 batch=16 device=0

# 3. export
python scripts/export_onnx.py

# 4. GATE — stop here if it fails
python scripts/check_parity.py

# 5. fp32 baseline accuracy
python scripts/eval_map.py --backend onnx --weights models/best.onnx
cp reports/accuracy.json reports/accuracy.fp32_baseline.json

# 6. quantize, then gate on mAP drop
python scripts/quantize_int8.py
python scripts/eval_map.py --backend onnx --weights models/best.int8.onnx \
    --compare-to reports/accuracy.fp32_baseline.json

# 7. latency across backends and batch sizes
python scripts/benchmark.py --batch-sizes 1 2 4 8
```

### Gate thresholds

| Gate | Config key | Default |
| --- | --- | --- |
| torch vs onnx max abs diff | `parity.max_abs_diff` | `1e-3` |
| torch vs onnx mean abs diff | `parity.max_mean_diff` | `1e-4` |
| matched-box IoU | `--min-box-iou` | `0.99` |
| int8 mAP50-95 drop | `quantize.max_map_drop` | `0.02` |

INT8 will not pass the fp32 parity thresholds — that is expected. Gate the int8 graph on
accuracy (step 6), and if you want a numerical view run
`check_parity.py --onnx models/best.int8.onnx --max-abs-diff 0.05`.

## Serving

```bash
BACKEND=onnx MODEL_PATH=models/best.onnx \
  uvicorn app.main:app --host 0.0.0.0 --port 8000
```

| Env var | Default | Notes |
| --- | --- | --- |
| `BACKEND` | `onnx` | `torch` or `onnx` |
| `MODEL_PATH` | `models/best.onnx` | int8 graphs load through the same backend |
| `DEVICE` | `cpu` | `cuda` needs `onnxruntime-gpu` |
| `IMGSZ` / `CONF` / `IOU` / `MAX_DET` | `640` / `0.25` / `0.45` / `300` | postprocess defaults |
| `MAX_BATCH_SIZE` | `8` | dynamic batch cap |
| `MAX_WAIT_MS` | `15` | how long a request waits to be batched |
| `MAX_QUEUE_SIZE` | `256` | overflow returns 503 rather than queueing forever |

### Endpoints

```bash
# multipart upload
curl -sS -F file=@frame.jpg localhost:8000/predict | jq

# json (base64 or url), with per-request thresholds
curl -sS localhost:8000/predict -H 'content-type: application/json' \
  -d '{"image_b64":"<...>","options":{"conf":0.4}}' | jq

curl -sS localhost:8000/healthz | jq
curl -sS localhost:8000/metrics | grep fds_
```

`/predict` returns boxes in original image coordinates plus `inference_ms`, `total_ms` and the
`batch_size` the request rode in — the gap between the first two is queue wait plus decode.

### Batching

One worker task drains an asyncio queue and fires a forward pass as soon as either
`MAX_BATCH_SIZE` requests are ready or `MAX_WAIT_MS` has passed since the first one arrived.
Inference runs in a thread so the event loop keeps accepting work. State is per-process, so run
`--workers 1` and scale with replicas.

Tuning: raise `MAX_WAIT_MS` for throughput, lower it for tail latency. `fds_batch_size` tells you
whether batches are actually filling — if it sits at 1 under load, the window is too short to help.

## Docker

```bash
docker build -t football-detect-serve .
docker run --rm -p 8000:8000 \
  -e MODEL_PATH=models/best.int8.onnx \
  -v "$PWD/models:/app/models:ro" \
  football-detect-serve
```

## Load test

```bash
k6 run bench/load_test.js                                    # ramp to 100 rps
BASE_URL=http://localhost:8000 IMAGE=frame.jpg k6 run bench/load_test.js
k6 run -e SCENARIO=constant -e RATE=50 -e DURATION=2m bench/load_test.js
```

Without `IMAGE=`, k6 posts a 1x1 PNG — that measures request overhead, not detection.

## Reports

| File | Written by |
| --- | --- |
| `reports/dataset.json` | `prepare_data.py` — per-class instance counts, label problems |
| `reports/parity.json` | `check_parity.py` |
| `reports/quantization.json` | `quantize_int8.py` — size reduction, calibration settings |
| `reports/accuracy.json`, `accuracy.<backend>.json` | `eval_map.py` |
| `reports/latency.json`, `figures/latency.png` | `benchmark.py` |

## Notes

- `models/` is gitignored; fetch released artifacts with
  [models/download_weights.py](models/download_weights.py).
- Football data is badly imbalanced (one ball, twenty-two players). `prepare_data.py` prints the
  ratio; read per-class AP, not just mAP — a model that ignores the ball entirely can still post a
  respectable mAP.
- `eval_map.py` defaults to `--conf 0.001`. mAP needs the low-confidence tail; evaluating at the
  serving threshold of `0.25` understates it.
