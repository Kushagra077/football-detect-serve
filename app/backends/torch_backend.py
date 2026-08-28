"""PyTorch backend.

predict() goes through ultralytics' own high-level YOLO.predict(), which decodes
whatever head format the loaded model uses (classic anchor grid + NMS, or YOLO26's
end2end top-k output) correctly. infer()/base.postprocess() still assume the classic
raw anchor-grid layout and are kept only for scripts/check_parity.py and
scripts/benchmark.py, which need the raw tensor path for bit-comparison with ONNX.
That path is unused by eval/serve until export is wired up for the actual model
family, at which point the end2end decode needs to move into base.postprocess()
too. See PROGRESS.md "Model family switched to YOLO26".
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from app.backends.base import DetectorBackend, Detection


class TorchBackend(DetectorBackend):
    def __init__(self, *args, device: str = "cpu", half: bool = False, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.device = device
        self.half = half
        self._model = None
        self._yolo = None
        self._torch = None

    @property
    def name(self) -> str:
        return "torch-fp16" if self.half else "torch"

    def load(self) -> "TorchBackend":
        import torch
        from ultralytics import YOLO

        self._torch = torch
        yolo = YOLO(self.weights)
        module = yolo.model.float().eval().to(self.device)
        if self.half and self.device != "cpu":
            module = module.half()
        for p in module.parameters():
            p.requires_grad_(False)

        self._model = module
        self._yolo = yolo
        names = getattr(yolo, "names", None) or getattr(module, "names", None)
        if names and not self.class_names:
            self.class_names = {int(k): str(v) for k, v in dict(names).items()}
        return self

    def predict(self, images: Sequence[np.ndarray]) -> List[List[Detection]]:
        if self._yolo is None:
            raise RuntimeError("TorchBackend.load() was never called")
        results = self._yolo.predict(
            list(images),
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            max_det=self.max_det,
            device=self.device,
            quantize=16 if (self.half and self.device != "cpu") else None,
            verbose=False,
        )
        out: List[List[Detection]] = []
        for r in results:
            dets: List[Detection] = []
            for box in r.boxes:
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
                        class_name=self.class_names.get(cls_id, str(cls_id)),
                    )
                )
            out.append(dets)
        return out

    def infer(self, batch: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("TorchBackend.load() was never called")
        torch = self._torch
        x = torch.from_numpy(batch).to(self.device)
        x = x.half() if self.half and self.device != "cpu" else x.float()

        with torch.inference_mode():
            out = self._model(x)

        # ultralytics detect head returns either a tensor or (tensor, aux)
        if isinstance(out, (list, tuple)):
            out = out[0]
        return out.detach().float().cpu().numpy()

    def close(self) -> None:
        self._model = None
        if self._torch is not None and self.device != "cpu":
            self._torch.cuda.empty_cache()
