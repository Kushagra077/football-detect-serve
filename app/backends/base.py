"""Backend contract: one predict() interface shared by every runtime.

Everything downstream (serving, eval, benchmark) talks only to this interface, so
torch / onnx-fp32 / onnx-int8 are driven identically. Each backend's predict()
delegates to ultralytics, which owns preprocess + inference + decode for whatever
head format the model uses. letterbox()/preprocess() below are kept as generic
image utilities (calibration, a future custom decoder) but are not on the hot path.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
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


@dataclass
class LetterboxMeta:
    """Everything needed to map letterboxed coords back to the source image."""

    ratio: float
    pad_w: float
    pad_h: float
    orig_h: int
    orig_w: int


def letterbox(
    img: np.ndarray, imgsz: int = 640, color: Tuple[int, int, int] = (114, 114, 114)
) -> Tuple[np.ndarray, LetterboxMeta]:
    """Resize preserving aspect ratio, pad to a square imgsz x imgsz canvas."""
    h, w = img.shape[:2]
    ratio = min(imgsz / h, imgsz / w)
    new_w, new_h = int(round(w * ratio)), int(round(h * ratio))
    if (new_w, new_h) != (w, h):
        interp = cv2.INTER_LINEAR if ratio > 1 else cv2.INTER_AREA
        img = cv2.resize(img, (new_w, new_h), interpolation=interp)

    pad_w = (imgsz - new_w) / 2
    pad_h = (imgsz - new_h) / 2
    top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
    left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, LetterboxMeta(ratio=ratio, pad_w=pad_w, pad_h=pad_h, orig_h=h, orig_w=w)


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

    # ---------------- generic image utilities (not on the predict path) ----------------

    def preprocess(self, images: Sequence[np.ndarray]) -> Tuple[np.ndarray, List[LetterboxMeta]]:
        """BGR uint8 HWC list -> NCHW float32 [0,1] batch + letterbox metadata.

        Not used by predict() (ultralytics does its own preprocessing); kept for
        calibration and any future raw-tensor tooling.
        """
        tensors, metas = [], []
        for img in images:
            padded, meta = letterbox(img, self.imgsz)
            rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
            tensors.append(rgb.transpose(2, 0, 1).astype(np.float32) / 255.0)
            metas.append(meta)
        return np.ascontiguousarray(np.stack(tensors, axis=0)), metas


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
