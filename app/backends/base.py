"""Backend contract: one preprocess/infer/postprocess path shared by every runtime.

Everything downstream (serving, eval, benchmark) talks only to this interface, so
torch / onnx-fp32 / onnx-int8 are measured with identical pre- and postprocessing.
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


def nms_numpy(boxes: np.ndarray, scores: np.ndarray, iou_thres: float) -> List[int]:
    """Greedy NMS on xyxy boxes. Returns kept indices, score-descending."""
    if boxes.size == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]

    keep: List[int] = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
        order = rest[iou <= iou_thres]
    return keep


class DetectorBackend(abc.ABC):
    """Abstract detector. Subclasses implement only `infer` plus load/meta."""

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

    # ---------------- shared pipeline ----------------

    def preprocess(self, images: Sequence[np.ndarray]) -> Tuple[np.ndarray, List[LetterboxMeta]]:
        """BGR uint8 HWC list -> NCHW float32 [0,1] batch + letterbox metadata."""
        tensors, metas = [], []
        for img in images:
            padded, meta = letterbox(img, self.imgsz)
            rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
            tensors.append(rgb.transpose(2, 0, 1).astype(np.float32) / 255.0)
            metas.append(meta)
        return np.ascontiguousarray(np.stack(tensors, axis=0)), metas

    @abc.abstractmethod
    def infer(self, batch: np.ndarray) -> np.ndarray:
        """Raw forward pass.

        Returns YOLOv8 head output shaped (B, 4 + num_classes, num_anchors) with
        xywh box coords in letterboxed pixel space and per-class sigmoid scores.
        """

    def postprocess(
        self, raw: np.ndarray, metas: Sequence[LetterboxMeta]
    ) -> List[List[Detection]]:
        """Decode raw head output -> per-image detections in original coords."""
        raw = np.asarray(raw, dtype=np.float32)
        if raw.ndim == 2:  # single image, missing batch dim
            raw = raw[None]

        results: List[List[Detection]] = []
        for i, meta in enumerate(metas):
            pred = raw[i]
            # Accept both (4+nc, anchors) and (anchors, 4+nc) layouts.
            if pred.shape[0] < pred.shape[1]:
                pred = pred.transpose(1, 0)  # -> (anchors, 4+nc)

            boxes_xywh = pred[:, :4]
            cls_scores = pred[:, 4:]
            if cls_scores.size == 0:
                results.append([])
                continue

            class_ids = cls_scores.argmax(axis=1)
            scores = cls_scores[np.arange(cls_scores.shape[0]), class_ids]

            mask = scores >= self.conf
            boxes_xywh, scores, class_ids = boxes_xywh[mask], scores[mask], class_ids[mask]
            if boxes_xywh.size == 0:
                results.append([])
                continue

            # xywh (center) -> xyxy, still letterboxed space
            cx, cy, bw, bh = boxes_xywh.T
            xyxy = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)

            # undo letterbox
            xyxy[:, [0, 2]] -= meta.pad_w
            xyxy[:, [1, 3]] -= meta.pad_h
            xyxy /= meta.ratio
            xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clip(0, meta.orig_w)
            xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clip(0, meta.orig_h)

            # class-aware NMS: offset boxes per class so classes never suppress each other
            offsets = class_ids.astype(np.float32) * (max(meta.orig_h, meta.orig_w) + 1.0)
            keep = nms_numpy(xyxy + offsets[:, None], scores, self.iou)[: self.max_det]

            results.append(
                [
                    Detection(
                        x1=float(xyxy[k, 0]),
                        y1=float(xyxy[k, 1]),
                        x2=float(xyxy[k, 2]),
                        y2=float(xyxy[k, 3]),
                        score=float(scores[k]),
                        class_id=int(class_ids[k]),
                        class_name=self.class_names.get(int(class_ids[k]), str(class_ids[k])),
                    )
                    for k in keep
                ]
            )
        return results

    def predict(self, images: Sequence[np.ndarray]) -> List[List[Detection]]:
        """Full path: preprocess -> infer -> postprocess."""
        batch, metas = self.preprocess(images)
        return self.postprocess(self.infer(batch), metas)


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
