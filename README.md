# football-detect-serve

Football object detection (ball / goalkeeper / player / referee / other) — YOLO26 trained
on broadcast football footage, exported to ONNX, quantized to INT8, and served behind a
batching FastAPI service with a live demo on Hugging Face Spaces.

**[Live demo](https://huggingface.co/spaces/Kushagra77/football-detect-serve)** (Gradio,
ZeroGPU — see the [Live demo](#live-demo) section below for why its numbers won't match
the table below)

The point of the repo is the path from checkpoint to endpoint: every optimization step has
a gate, every backend is measured through the same code, and every number below came from
running that code, not from a spec sheet.

![sample detection](reports/figures/sample_detection.png)

## Results

### Accuracy (mAP50-95, val split, 8,250 images)

| model | backend | all | ball | goalkeeper | player | referee |
|---|---|---|---|---|---|---|
| off-the-shelf COCO yolo26n (remapped classes) | torch | 0.093 | 0.013 | 0.000 | 0.361 | 0.000 |
| **v1 nano** | torch | 0.427 | 0.069 | 0.475 | 0.603 | 0.560 |
| v1 nano | onnx-fp32 | 0.427 | 0.069 | 0.475 | 0.603 | 0.560 |
| v1 nano | onnx-int8 | 0.400 (−0.027) | 0.041 | 0.445 | 0.581 | 0.532 |
| **v2 small** | torch | 0.487 | 0.129 | 0.575 | 0.639 | 0.605 |
| v2 small | onnx-fp32 | 0.487 | 0.129 | 0.575 | 0.639 | 0.605 |
| v2 small | onnx-int8 | 0.423 (−0.064) | 0.069 | 0.497 | 0.577 | 0.549 |

The `other` class (sideline staff etc.) has **0 instances in the val split**, so its AP is
NaN and unvalidatable even though both models still emit ~1-4k `other` predictions per
eval — a silent false-positive source that costs nothing in the current metric. See
[Limitations](#limitations).

- ONNX-fp32 reproduces torch to <0.001 mAP per class — the export path is faithful, not
  approximate.
- INT8 costs 0.027-0.064 mAP50-95 depending on model size, with the ball class taking the
  biggest relative hit (~40-46%) since it already has the fewest instances and smallest
  boxes.
- Both checkpoints are **10-epoch, deliberately under-trained** — the point right now is
  the full pipeline working end-to-end, not a maximized model. `configs/train.yaml` is set
  for 100 epochs; the numbers above will move if you train it out.

### Latency (single-request, CPU, Apple M2 8-core, batch=1)

| backend | p50 ms | p95 ms | p99 ms | img/s |
|---|---|---|---|---|
| torch | 23.7 | 24.5 | 26.3 | 41.9 |
| onnx-fp32 | 16.2 | 16.5 | 19.7 | 61.2 |
| onnx-int8 | 19.1 | 20.7 | 23.9 | 51.6 |

**onnx-fp32 is 1.5x torch, free** — no accuracy cost (see table above), just a different
runtime. **onnx-int8 is *slower* than fp32 on this hardware** — Apple Silicon (ARM) lacks
the x86-VNNI instructions onnxruntime's INT8 kernels are tuned for, so it's paying dequant
overhead for no speed gain. **On x86 this would likely flip**, but there is no x86 run to
confirm it against — see [Limitations](#limitations). FP16 was measured and dropped from
this table: on CPU it ties fp32 on accuracy and is *slower* (onnxruntime has no native
fp16 kernels on CPU, so it upcasts and pays the cast cost) — FP16 only helps on GPU, and
this service targets CPU.

### Load test: does batching help? (Locust, local Docker, `onnx-fp32`, 1 min/run)

![load test chart](reports/figures/loadtest.svg)

| concurrency | batching | RPS | p50 ms | p95 ms | p99 ms |
|---|---|---|---|---|---|
| 1 | on | 13.3 | 71 | 100 | 160 |
| 1 | off | 22.4 | 41 | 63 | 81 |
| 4 | on | 17.2 | 250 | 350 | 470 |
| 4 | off | 25.1 | 150 | 210 | 230 |
| 16 | on | 28.7 | 510 | 850 | 1200 |
| 16 | off | 23.8 | 630 | 1100 | 1500 |

**The answer is "it depends on load," not a flat yes/no.** At concurrency 1 and 4,
batching is pure overhead — most requests never find a batch partner inside the 15ms
`MAX_WAIT_MS` window, so they pay that wait and then run solo anyway (nobatch wins on
every axis). At concurrency 16, batching wins outright — higher throughput (28.7 vs 23.8
RPS) *and* lower latency at every percentile — because enough concurrent requests land
inside the window to be grouped into one forward pass, which beats 16 single-image calls
fighting over the same 4 pinned CPU threads. Zero request failures across all six runs.

### The acceptance sentence

> The ONNX runtime gave **1.5x** for free; INT8 bought **nothing** on this CPU (ARM) and
> cost 0.027-0.064 mAP50-95 — the honest result of measuring rather than assuming, and a
> result that plausibly reverses on x86.

## Design rules

1. **One inference path per backend, no custom decode.** `app/backends/torch_backend.py`
   and `onnx_backend.py` both call ultralytics' own `YOLO(...).predict()`, which knows how
   to decode YOLO26's end-to-end (NMS-free, top-k) head — a hand-rolled decoder assuming
   the classic dense anchor-grid output silently produced garbage on this architecture
   (see git history / old PROGRESS.md entries if curious). `app/backends/base.py` keeps
   `letterbox()`/`preprocess()` as generic utilities, not on the hot path.
2. **One evaluator for every backend.** `scripts/eval_map.py` computes COCO-style AP
   itself for torch and ONNX alike through the shared `DetectorBackend.predict()`
   interface, so comparing backends never means comparing two different measurement
   methods.
3. **Parity via the same evaluator, not a separate raw-tensor diff.** ONNX-vs-torch
   fidelity is checked by running `eval_map.py` on both and comparing per-class mAP to
   3 decimal places, not by diffing raw model output tensors (which YOLO26's end-to-end
   head format makes brittle across backends).
4. **Quantize against real data.** `scripts/export_onnx.py --quantize 8` calibrates INT8
   on a sample of the *train* split (`--split train --fraction 0.01`, ~345 images) run
   through the real preprocessing path, and the result is a measured mAP drop, not an
   assumption.

## Layout

```
configs/train.yaml           epochs, imgsz, batch, augmentation, expected class map
scripts/prepare_data.py      verify class map, count instances, flag bad labels
scripts/convert_mot_to_yolo.py   MOT gt.txt + gameinfo.ini -> YOLO format
scripts/train.py              thin wrapper over ultralytics
scripts/export_onnx.py        pt -> onnx, fp32/fp16/int8, static INT8 calibration
scripts/eval_map.py           per-class mAP for ANY backend, one code path
scripts/benchmark.py          single-request latency harness -> reports/latency.json
app/main.py                   FastAPI: /predict /healthz /metrics, multi-backend registry
app/batching.py                asyncio queue + max-wait window
app/backends/                  base (contract) + torch + onnx (fp32/fp16/int8)
app/gradio_app.py              HF Space demo - calls build_backend() directly, no FastAPI
bench/locustfile.py            load test: concurrency x batching on/off
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Pipeline

```bash
# 1. dataset: convert MOT tracking labels to YOLO format, then verify + count instances
python scripts/convert_mot_to_yolo.py --src <path-to-raw-mot-data>
python scripts/prepare_data.py --skip-download --out dataset

# 2. train (hyperparameters in configs/train.yaml)
# configs/train.yaml targets 100 epochs; the shipped v1/v2 checkpoints are a
# deliberately under-trained 10-epoch run (see Results) - reproduce that with:
python scripts/train.py --weights yolo26n.pt --out models/football_detection_v1.pt --set epochs=10

# 3. accuracy (one code path for every backend)
python scripts/eval_map.py --backend torch --weights models/football_detection_v1.pt

# 4. export: fp32, fp16, int8 (int8 calibrates on the train split)
python scripts/export_onnx.py --weights models/football_detection_v1.pt --quantize none
python scripts/export_onnx.py --weights models/football_detection_v1.pt --quantize 16
python scripts/export_onnx.py --weights models/football_detection_v1.pt --quantize 8 \
    --split train --fraction 0.01

# 5. GATE — onnx-fp32 must match torch mAP to ~0.001/class; int8 drop must be sane, not collapsed
python scripts/eval_map.py --backend onnx --weights models/football_detection_v1.onnx
python scripts/eval_map.py --backend onnx --weights models/football_detection_v1_int8.onnx

# 6. single-request latency across backends and batch sizes
python scripts/benchmark.py
```

## Serving

```bash
MODELS="torch=models/football_detection_v1.pt,onnx-fp32=models/football_detection_v1.onnx,onnx-int8=models/football_detection_v1_int8.onnx" \
DEFAULT_BACKEND=onnx-fp32 uvicorn app.main:app --host 0.0.0.0 --port 7860
```

One process serves several backends at once — `?backend=` picks one per request, so you
can demo and load-test torch/onnx-fp32/onnx-int8 side by side without redeploying.

| Env var | Default | Notes |
| --- | --- | --- |
| `MODELS` | torch+onnx-fp32+onnx-int8 on v1 | `name=path,name=path,...` |
| `DEFAULT_BACKEND` | first entry in `MODELS` | used when `?backend=` is omitted |
| `DEVICE` | `cpu` | |
| `IMGSZ` / `CONF` / `IOU` / `MAX_DET` | `640` / `0.25` / `0.45` / `300` | postprocess defaults, overridable per request |
| `MAX_BATCH_SIZE` | `8` | dynamic batch cap |
| `MAX_WAIT_MS` | `15` | how long a request waits to be batched |
| `MAX_QUEUE_SIZE` | `256` | overflow returns 503 rather than queueing forever |

### Endpoints

```bash
# multipart upload, pick a backend, bypass batching
curl -sS -F file=@frame.jpg "localhost:7860/predict?backend=onnx-fp32&batch=false" | jq

# json (base64 or url), with per-request thresholds
curl -sS localhost:7860/predict -H 'content-type: application/json' \
  -d '{"image_b64":"<...>","options":{"conf":0.4}}' | jq

curl -sS localhost:7860/healthz | jq   # lists every loaded backend
curl -sS localhost:7860/metrics | grep fds_   # per-backend counters/histograms
```

`/predict` returns boxes in original image coordinates plus `inference_ms`, `total_ms`,
`batch_size` and `batched` — the gap between `inference_ms` and `total_ms` is decode plus
(if `batched=true`) queue wait.

### Batching

One worker task per backend drains an asyncio queue and fires a forward pass as soon as
either `MAX_BATCH_SIZE` requests are ready or `MAX_WAIT_MS` has passed since the first one
arrived. `?batch=false` skips the queue entirely for a direct comparison. State is
per-process, so run `--workers 1` and scale with replicas, not worker count.

## Docker

**Local-only.** HF Docker Spaces are a paid tier, so this image is never deployed — it
exists to run the multi-backend service under the Locust load test and produce the
latency/throughput numbers above. The public demo is the separate [Gradio
app](#live-demo) instead.

```bash
docker build -t football-detect-serve:local .
docker run --rm -p 7860:7860 football-detect-serve:local
```

Both backends decode through ultralytics now (see Design rules), so this image includes
torch + ultralytics, not just onnxruntime — about 1GB, comfortably inside a 3-4GB
container memory budget.

## Load test

```bash
locust -f bench/locustfile.py --headless -u 16 -r 16 -t 1m \
    --host http://localhost:7860 --csv reports/loadtest/c16 --html reports/loadtest/c16.html

FDS_BATCH=false locust -f bench/locustfile.py --headless -u 16 -r 16 -t 1m \
    --host http://localhost:7860 --csv reports/loadtest/c16_nobatch --html reports/loadtest/c16_nobatch.html
```

Or drop `--headless` for the interactive web UI at `localhost:8089`. `FDS_IMAGE` /
`FDS_BACKEND` / `FDS_BATCH` env vars control the request; see `bench/locustfile.py` for
details. Six HTML reports (concurrency 1/4/16 x batching on/off) are in
`reports/loadtest/`.

## Live demo

The [HF Space](https://huggingface.co/spaces/Kushagra77/football-detect-serve) is a
Gradio app (`app/gradio_app.py`) that calls `build_backend()` directly — no FastAPI, no
batching queue, since it's a low-traffic demo, not the thing under load test.

HF's free tier only offers **ZeroGPU** hardware (not standalone CPU), so every backend in
the demo runs through one `@spaces.GPU`-wrapped call and *may* execute GPU-accelerated —
including the ONNX backends, since ZeroGPU makes `torch.cuda.is_available()` report `True`
everywhere in the process and ultralytics' ONNX backend touches raw CUDA internals
regardless of which execution provider you asked for. **This means the demo's relative
backend speed will not match the CPU latency table above** — that table is the real
comparison; the Space is for checking correctness and accuracy, not speed.

## Reports

| File | Written by |
| --- | --- |
| `reports/dataset.json` | `prepare_data.py` — per-class instance counts, label problems |
| `reports/accuracy.<backend>.<model>.json` | `eval_map.py` |
| `reports/latency.json`, `figures/latency.png` | `benchmark.py` |
| `reports/loadtest/*.csv`, `*.html` | Locust |

## Limitations

- **10-epoch checkpoints.** `configs/train.yaml` targets 100 epochs; the models here are
  intentionally under-trained placeholders that unblock the pipeline. Every number above
  will move with a full training run.
- **The `other` class is unvalidatable.** Zero instances in the val split means its AP is
  NaN, yet both models emit thousands of `other` predictions per eval run — false
  positives that cost nothing in the current mAP. Worth a 4-class mAP alongside the
  5-class one, or folding `other` into `player`, in a future pass.
- **One degenerate box in the raw source labels**, not introduced by conversion: one
  sequence's `gt.txt` has a `w=0` row. Left as-is rather than patched around.
- **Latency numbers are ARM (Apple M2), not x86.** The Docker service that produced them
  is local-only and never deployed to an x86 host, so the "INT8 is slower than fp32"
  result is plausibly an ARM/x86 kernel-tuning artifact (onnxruntime's INT8 kernels are
  x86-VNNI tuned), not a property of the model. There is no x86 run to confirm or deny
  this.
- **`imgsz=640`, not 1280.** A documented tradeoff, not an oversight — it protects the CPU
  latency story that is the point of this phase. Recovering small/distant ball detections
  with tiled high-res inference is out of scope here.
- **Out of scope for this phase:** tracking, video input, team assignment, TensorRT, a
  fancier frontend than the Gradio demo.
