"""ONNX Runtime backend. Handles fp32, fp16 and statically-quantized int8 graphs.

predict() goes through ultralytics' own YOLO("model.onnx").predict(), which decodes
whatever head format the exported graph carries (classic anchor grid + NMS, or
YOLO26's end2end top-k output) correctly and matches the torch backend path exactly.

infer() keeps a bare onnxruntime session that returns the raw output tensor. It is
only used by scripts/check_parity.py and scripts/benchmark.py's raw-tensor mode,
still assumes the classic anchor-grid layout, and is built lazily so the common
eval/serve path never pays for it. See PROGRESS.md "Model family switched to YOLO26".
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from app.backends.base import DetectorBackend, Detection

_METADATA_NAMES_KEY = "names"


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
        self.providers = providers
        self.intra_op_threads = intra_op_threads
        self._yolo = None
        self._precision = "fp32"
        self._static_batch: Optional[int] = None
        # lazily-built raw session (infer() only)
        self._sess = None
        self._input_name: Optional[str] = None
        self._input_dtype = np.float32

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
        # once — ultralytics feeds it as one tensor and onnxruntime rejects the shape.
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

    # ---------------- raw-tensor path (check_parity / benchmark only) ----------------

    def _ensure_session(self) -> None:
        if self._sess is not None:
            return
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if self.intra_op_threads:
            opts.intra_op_num_threads = self.intra_op_threads

        available = ort.get_available_providers()
        if self.providers:
            providers = self.providers
        elif self.device != "cpu" and "CUDAExecutionProvider" in available:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        self._sess = ort.InferenceSession(self.weights, sess_options=opts, providers=providers)
        inp = self._sess.get_inputs()[0]
        self._input_name = inp.name
        self._input_dtype = np.float16 if "float16" in inp.type else np.float32
        batch_dim = inp.shape[0] if inp.shape else None
        self._static_batch = batch_dim if isinstance(batch_dim, int) else None

    def infer(self, batch: np.ndarray) -> np.ndarray:
        self._ensure_session()
        assert self._input_name is not None
        batch = batch.astype(self._input_dtype, copy=False)

        if self._static_batch is not None and batch.shape[0] != self._static_batch:
            return np.concatenate(
                [
                    self._run(self._pad_to_static(chunk))[: chunk.shape[0]]
                    for chunk in _chunks(batch, self._static_batch)
                ],
                axis=0,
            )
        return self._run(batch)

    def _run(self, batch: np.ndarray) -> np.ndarray:
        out = self._sess.run(None, {self._input_name: batch})[0]
        return np.asarray(out, dtype=np.float32)

    def _pad_to_static(self, chunk: np.ndarray) -> np.ndarray:
        assert self._static_batch is not None
        missing = self._static_batch - chunk.shape[0]
        if missing <= 0:
            return chunk
        pad = np.repeat(chunk[-1:], missing, axis=0)
        return np.concatenate([chunk, pad], axis=0)

    def close(self) -> None:
        self._yolo = None
        self._sess = None


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


def _chunks(arr: np.ndarray, size: int):
    for start in range(0, arr.shape[0], size):
        yield arr[start : start + size]


def _graph_nodes(path: str):
    """Load nodes lazily; a missing onnx package should not break serving."""
    try:
        import onnx

        return onnx.load(path, load_external_data=False).graph.node
    except Exception:
        return []


def _parse_names(raw: str) -> dict:
    """ultralytics stores names as a str(dict) in ONNX metadata."""
    import ast

    try:
        parsed = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return {}
    if isinstance(parsed, dict):
        return {int(k): str(v) for k, v in parsed.items()}
    if isinstance(parsed, (list, tuple)):
        return {i: str(v) for i, v in enumerate(parsed)}
    return {}
