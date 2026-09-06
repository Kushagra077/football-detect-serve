"""Shared fixtures. No test in this suite touches real dataset content or real
model weights - everything here is synthetic (zeros arrays, fabricated labels,
a fake DetectorBackend), per the project's MOU constraint on dataset distribution.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from app.backends.base import Detection, DetectorBackend

DEFAULT_CLASS_NAMES = {0: "ball", 1: "goalkeeper", 2: "player", 3: "referee", 4: "other"}


class FakeBackend(DetectorBackend):
    """A DetectorBackend that returns canned detections instead of running a model.

    Records the batch size of every predict() call in `.calls`, so batching tests
    can assert on how requests were grouped.
    """

    def __init__(self, *args, fixed_detections: Optional[List[Detection]] = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.class_names = dict(kwargs.get("class_names") or DEFAULT_CLASS_NAMES)
        self._fixed = fixed_detections if fixed_detections is not None else []
        self.calls: List[int] = []
        # (conf, iou, max_det) actually passed to each predict() call - lets
        # tests assert per-request overrides reached predict() as call
        # arguments instead of being read off self.conf/self.iou/self.max_det.
        self.call_options: List[tuple] = []

    def load(self) -> "FakeBackend":
        return self

    @property
    def name(self) -> str:
        return "fake"

    def predict(
        self,
        images: Sequence[np.ndarray],
        *,
        conf: Optional[float] = None,
        iou: Optional[float] = None,
        max_det: Optional[int] = None,
    ) -> List[List[Detection]]:
        self.calls.append(len(images))
        self.call_options.append((conf, iou, max_det))
        return [list(self._fixed) for _ in images]


def tiny_image() -> np.ndarray:
    """A minimal synthetic BGR image - never a real frame."""
    return np.zeros((4, 4, 3), dtype=np.uint8)


def tiny_jpeg_bytes() -> bytes:
    """A minimal synthetic JPEG, encoded on the fly - never a real frame."""
    import cv2

    ok, buf = cv2.imencode(".jpg", tiny_image())
    assert ok
    return buf.tobytes()
