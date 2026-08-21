"""PyTorch backend: raw ultralytics head output, decoded by base.postprocess.

We deliberately bypass ultralytics' own NMS/scaling so the torch path is
bit-comparable with the ONNX path (see scripts/check_parity.py).
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from app.backends.base import DetectorBackend


class TorchBackend(DetectorBackend):
    def __init__(self, *args, device: str = "cpu", half: bool = False, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.device = device
        self.half = half
        self._model = None
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
        names = getattr(yolo, "names", None) or getattr(module, "names", None)
        if names and not self.class_names:
            self.class_names = {int(k): str(v) for k, v in dict(names).items()}
        return self

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
