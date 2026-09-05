"""Gradio demo: football detection, calling build_backend() directly.

This is the public HF Space deliverable, separate from the FastAPI service in
app/main.py. No batching queue, no /metrics - it's a low-traffic demo, not the
thing under load test. See PROGRESS.md's 2026-09-04 deploy-plan split.

Run locally:
    python -m app.gradio_app
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import cv2
import gradio as gr
import numpy as np
import spaces
import torch

from app.backends.base import Detection, DetectorBackend, build_backend

IMGSZ = int(os.getenv("IMGSZ", 640))
CONF = float(os.getenv("CONF", 0.25))
IOU = float(os.getenv("IOU", 0.45))

# Same 3 files the local Docker service bakes in (PROGRESS.md Step 8).
_MODELS: Dict[str, str] = {
    "torch": "models/football_detection_v1.pt",
    "onnx-fp32": "models/football_detection_v1.onnx",
    "onnx-int8": "models/football_detection_v1_int8.onnx",
}

# Loaded lazily per backend name and cached - a Space with 16GB RAM (free CPU
# Basic) comfortably holds all three loaded at once.
_CACHE: Dict[str, DetectorBackend] = {}

_COLORS = {
    "ball": (0, 215, 255),
    "goalkeeper": (255, 0, 255),
    "player": (0, 255, 0),
    "referee": (0, 0, 255),
    "other": (180, 180, 180),
}


def _get_backend(name: str) -> DetectorBackend:
    if name not in _CACHE:
        kind = "torch" if name.startswith("torch") else "onnx"
        _CACHE[name] = build_backend(
            kind=kind,
            weights=_MODELS[name],
            imgsz=IMGSZ,
            conf=CONF,
            iou=IOU,
            device="cpu",
        )
    return _CACHE[name]


def _draw(img_bgr: np.ndarray, detections: List[Detection]) -> np.ndarray:
    out = img_bgr.copy()
    for d in detections:
        color = _COLORS.get(d.class_name, (255, 255, 255))
        p1, p2 = (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2))
        cv2.rectangle(out, p1, p2, color, 2)
        label = f"{d.class_name} {d.score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (p1[0], max(0, p1[1] - th - 6)), (p1[0] + tw + 4, p1[1]), color, -1)
        cv2.putText(
            out, label, (p1[0] + 2, max(12, p1[1] - 4)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1,
        )
    return out


@spaces.GPU(duration=30)
def _run_on_gpu(backend_name: str, img_bgr: np.ndarray) -> List[Detection]:
    """Every backend's inference goes through here, not just torch.

    ZeroGPU makes torch.cuda.is_available() lie (return True) everywhere in the
    process, so libraries initialize CUDA-shaped state even off the GPU path.
    ultralytics' ONNX backend does exactly that - it builds an output-tensor
    cache with a raw torch.empty(...).to(device) call during model *load*, and
    ZeroGPU blocks raw CUDA calls made outside @spaces.GPU outright (no graceful
    fallback, just a crash - that's what killed onnx-int8 the first time this
    ran). Keeping onnx backends outside the decorator to save GPU quota doesn't
    survive contact with how ultralytics actually behaves on this host, so
    everything routes through one real GPU-allocated call instead.
    """
    backend = _get_backend(backend_name)
    # Locally (no ZeroGPU host, no CUDA) @spaces.GPU is a no-op passthrough -
    # fall back to CPU so `python -m app.gradio_app` still works on a Mac.
    backend.device = "cuda" if torch.cuda.is_available() else "cpu"
    return backend.predict([img_bgr])[0]


def predict(image: Optional[np.ndarray], backend_name: str) -> Tuple[Optional[np.ndarray], str]:
    if image is None:
        return None, "Upload an image first."

    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    detections = _run_on_gpu(backend_name, img_bgr)
    annotated = cv2.cvtColor(_draw(img_bgr, detections), cv2.COLOR_BGR2RGB)

    counts: Dict[str, int] = {}
    for d in detections:
        counts[d.class_name] = counts.get(d.class_name, 0) + 1
    summary = (
        f"{len(detections)} detection(s) — " + ", ".join(f"{k}: {v}" for k, v in sorted(counts.items()))
        if detections
        else "0 detections"
    )
    return annotated, summary


with gr.Blocks(title="football-detect-serve") as demo:
    gr.Markdown(
        "# Football Object Detection\n"
        "Detects **ball / goalkeeper / player / referee**. Switch backends to compare "
        "accuracy — full mAP and latency tables (measured on CPU) are in the repo README.\n\n"
        "_Note: this Space runs on ZeroGPU, so every backend here may run GPU-accelerated. "
        "Relative speed on this page won't match the CPU-only benchmark table in the "
        "README — that table is the real comparison; this demo is for correctness/accuracy._"
    )
    with gr.Row():
        with gr.Column():
            image_in = gr.Image(type="numpy", label="Upload a frame")
            backend_dd = gr.Dropdown(
                choices=list(_MODELS), value="onnx-fp32", label="Backend"
            )
            run_btn = gr.Button("Detect", variant="primary")
        with gr.Column():
            image_out = gr.Image(type="numpy", label="Detections")
            summary_out = gr.Textbox(label="Summary")

    run_btn.click(predict, inputs=[image_in, backend_dd], outputs=[image_out, summary_out])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", 7860)))
