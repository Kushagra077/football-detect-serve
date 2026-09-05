"""ONNX Runtime backend. Handles fp32, fp16 and statically-quantized int8 graphs.

Inference goes through ultralytics' own YOLO("model.onnx").predict(), which loads the
graph into onnxruntime, decodes whatever head format it carries (classic anchor grid
+ NMS, or YOLO26's end2end top-k output), and matches the torch backend exactly. The
old raw-tensor path (infer()/base.postprocess()) has been removed - it assumed the
classic anchor-grid layout, which YOLO26 does not use. See PROGRESS.md.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from app.backends.base import DetectorBackend, Detection, to_detections


class OnnxBackend(DetectorBackend):
    def __init__(
        self,
        *args,
        device: str = "cpu",
        providers: Optional[List[str]] = None,
        intra_op_threads: int = 0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.device = device
        # providers / intra_op_threads are accepted for call-site compatibility;
        # onnxruntime session config is owned by ultralytics' autobackend now.
        self.providers = providers
        self.intra_op_threads = intra_op_threads
        self._yolo = None
        self._precision = "fp32"
        self._static_batch: Optional[int] = None

    @property
    def name(self) -> str:
        return f"onnx-{self._precision}"

    def load(self) -> "OnnxBackend":
        from ultralytics import YOLO

        path = Path(self.weights)
        if not path.exists():
            raise FileNotFoundError(f"onnx model not found: {path}")

        self._precision = _detect_precision(path)
        self._static_batch = _static_batch_of(str(path))
        self._yolo = YOLO(str(path), task="detect")

        names = getattr(self._yolo, "names", None)
        if names and not self.class_names:
            self.class_names = {int(k): str(v) for k, v in dict(names).items()}
        return self

    def predict(self, images: Sequence[np.ndarray]) -> List[List[Detection]]:
        if self._yolo is None:
            raise RuntimeError("OnnxBackend.load() was never called")

        images = list(images)
        # A fixed-batch graph (int8 export is batch-1) can't take the whole list at
        # once - ultralytics feeds it as one tensor and onnxruntime rejects the shape.
        # Fall back to fixed-size chunks; the final short chunk still works at batch 1.
        sb = self._static_batch
        if sb and len(images) > sb:
            out: List[List[Detection]] = []
            for i in range(0, len(images), sb):
                out.extend(self.predict(images[i : i + sb]))
            return out

        results = self._yolo.predict(
            images,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            max_det=self.max_det,
            device=self.device,
            verbose=False,
        )
        return [to_detections(r, self.class_names) for r in results]

    def close(self) -> None:
        self._yolo = None


def _detect_precision(path: Path) -> str:
    """int8 / fp16 / fp32 from the filename or the graph's quantized nodes."""
    stem = path.name.lower()
    if "int8" in stem:
        return "int8"
    if "fp16" in stem or "half" in stem:
        return "fp16"
    for n in _graph_nodes(str(path)):
        if n.op_type.startswith(("QLinear", "QuantizeLinear")):
            return "int8"
    return "fp32"


def _static_batch_of(path: str) -> Optional[int]:
    """Fixed batch size of the graph's first input, or None if the axis is dynamic."""
    try:
        import onnx

        model = onnx.load(path, load_external_data=False)
        dim0 = model.graph.input[0].type.tensor_type.shape.dim[0]
        return dim0.dim_value if dim0.HasField("dim_value") and dim0.dim_value > 0 else None
    except Exception:
        return None


def _graph_nodes(path: str):
    """Load nodes lazily; a missing onnx package should not break serving."""
    try:
        import onnx

        return onnx.load(path, load_external_data=False).graph.node
    except Exception:
        return []
