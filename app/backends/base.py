"""Backend contract: one predict() interface shared by every runtime.

Everything downstream (serving, eval, benchmark) talks only to this interface, so
torch / onnx-fp32 / onnx-int8 are driven identically. Each backend's predict()
delegates to ultralytics, which owns preprocess + inference + decode for whatever
head format the model uses - there is no manual letterbox/preprocess step here;
an earlier one was removed as dead code once both backends moved to ultralytics.
"""
from __future__ import annotations

import abc
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


@dataclass
class Detection:
    """One box in ORIGINAL image coordinates (xyxy, pixels)."""

    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    class_id: int
    class_name: str


def to_detections(result, class_names: Dict[int, str]) -> List[Detection]:
    """Convert one ultralytics Result -> our Detection list. Shared by every backend
    so torch/onnx-fp32/onnx-int8 can't silently diverge in how they read boxes out.
    """
    dets: List[Detection] = []
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cls_id = int(box.cls[0])
        dets.append(
            Detection(
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                score=float(box.conf[0]),
                class_id=cls_id,
                class_name=class_names.get(cls_id, str(cls_id)),
            )
        )
    return dets


class DetectorBackend(abc.ABC):
    """Abstract detector. Subclasses implement load / name / predict."""

    def __init__(
        self,
        weights: str,
        imgsz: int = 640,
        conf: float = 0.25,
        iou: float = 0.45,
        max_det: int = 300,
        class_names: Optional[Dict[int, str]] = None,
    ) -> None:
        self.weights = weights
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.max_det = max_det
        self.class_names: Dict[int, str] = dict(class_names or {})
        # Guards predict(): ultralytics' YOLO keeps mutable per-call state and is
        # not safe to invoke from two threads at once. The unbatched serving path
        # (app/main.py) and the batch worker (app/batching.py) both hold this
        # lock around their call to predict(), since both can run concurrently
        # against the same backend instance. predict() itself does not lock, so
        # offline callers (scripts/, gradio) that only ever call it from one
        # thread pay nothing extra.
        self.predict_lock = threading.Lock()

    # ---------------- lifecycle ----------------

    @abc.abstractmethod
    def load(self) -> "DetectorBackend":
        """Build the session/module. Must be called before predict()."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Short backend id used in reports, e.g. 'torch' or 'onnx-int8'."""

    def warmup(self, runs: int = 3) -> None:
        dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        for _ in range(runs):
            self.predict([dummy])

    def close(self) -> None:  # pragma: no cover - most backends need nothing
        pass

    @abc.abstractmethod
    def predict(self, images: Sequence[np.ndarray]) -> List[List[Detection]]:
        """BGR uint8 HWC images -> per-image detections in original coordinates."""


def build_backend(
    kind: str,
    weights: str,
    imgsz: int = 640,
    conf: float = 0.25,
    iou: float = 0.45,
    max_det: int = 300,
    device: str = "cpu",
    **kwargs: Any,
) -> DetectorBackend:
    """Factory used by serving, eval and benchmark so they stay in lockstep."""
    kind = kind.lower()
    common = dict(weights=weights, imgsz=imgsz, conf=conf, iou=iou, max_det=max_det)

    if kind in {"torch", "pt", "pytorch"}:
        from app.backends.torch_backend import TorchBackend

        return TorchBackend(device=device, **common, **kwargs).load()

    if kind in {"onnx", "onnxruntime", "int8", "onnx-int8"}:
        from app.backends.onnx_backend import OnnxBackend

        return OnnxBackend(device=device, **common, **kwargs).load()

    raise ValueError(f"unknown backend '{kind}' (expected 'torch' or 'onnx')")
