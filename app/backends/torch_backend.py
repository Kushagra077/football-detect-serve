"""PyTorch backend.

Inference goes through ultralytics' own high-level YOLO.predict(), which decodes
whatever head format the loaded model uses (classic anchor grid + NMS, or YOLO26's
end2end top-k output) correctly. The old raw-tensor infer() has been removed - it
assumed the classic anchor-grid layout that YOLO26 does not use. See PROGRESS.md.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from app.backends.base import DetectorBackend, Detection, to_detections


class TorchBackend(DetectorBackend):
    def __init__(self, *args, device: str = "cpu", half: bool = False, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.device = device
        self.half = half
        self._yolo = None

    @property
    def name(self) -> str:
        return "torch-fp16" if self.half else "torch"

    def load(self) -> "TorchBackend":
        from ultralytics import YOLO

        yolo = YOLO(self.weights)
        self._yolo = yolo
        names = getattr(yolo, "names", None)
        if names and not self.class_names:
            self.class_names = {int(k): str(v) for k, v in dict(names).items()}
        return self

    def predict(
        self,
        images: Sequence[np.ndarray],
        *,
        conf: Optional[float] = None,
        iou: Optional[float] = None,
        max_det: Optional[int] = None,
    ) -> List[List[Detection]]:
        if self._yolo is None:
            raise RuntimeError("TorchBackend.load() was never called")
        results = self._yolo.predict(
            list(images),
            imgsz=self.imgsz,
            conf=self.conf if conf is None else conf,
            iou=self.iou if iou is None else iou,
            max_det=self.max_det if max_det is None else max_det,
            device=self.device,
            quantize=16 if (self.half and self.device != "cpu") else None,
            verbose=False,
        )
        return [to_detections(r, self.class_names) for r in results]

    def close(self) -> None:
        self._yolo = None
