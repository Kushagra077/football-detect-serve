"""ONNX Runtime backend. Handles fp32 and statically-quantized int8 graphs.

Both live in the same class: quantization changes the weights, not the graph
signature, so the only difference visible here is `name` and the input dtype.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np

from app.backends.base import DetectorBackend

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
        self._sess = None
        self._input_name: Optional[str] = None
        self._input_dtype = np.float32
        self._static_batch: Optional[int] = None
        self._is_int8 = False

    @property
    def name(self) -> str:
        return "onnx-int8" if self._is_int8 else "onnx-fp32"

    def _resolve_providers(self) -> List[str]:
        import onnxruntime as ort

        if self.providers:
            return self.providers
        available = ort.get_available_providers()
        if self.device != "cpu" and "CUDAExecutionProvider" in available:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    def load(self) -> "OnnxBackend":
        import onnxruntime as ort

        path = Path(self.weights)
        if not path.exists():
            raise FileNotFoundError(f"onnx model not found: {path}")

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if self.intra_op_threads:
            opts.intra_op_num_threads = self.intra_op_threads

        self._sess = ort.InferenceSession(
            str(path), sess_options=opts, providers=self._resolve_providers()
        )

        inp = self._sess.get_inputs()[0]
        self._input_name = inp.name
        self._input_dtype = np.float16 if "float16" in inp.type else np.float32
        batch_dim = inp.shape[0] if inp.shape else None
        self._static_batch = batch_dim if isinstance(batch_dim, int) else None

        # int8 detection: any quantized weight initializer / QLinear node
        self._is_int8 = "int8" in path.name.lower() or any(
            n.op_type.startswith(("QLinear", "QuantizeLinear"))
            for n in _graph_nodes(str(path))
        )

        meta = self._sess.get_modelmeta().custom_metadata_map or {}
        if not self.class_names and _METADATA_NAMES_KEY in meta:
            self.class_names = _parse_names(meta[_METADATA_NAMES_KEY])
        return self

    def infer(self, batch: np.ndarray) -> np.ndarray:
        if self._sess is None or self._input_name is None:
            raise RuntimeError("OnnxBackend.load() was never called")

        batch = batch.astype(self._input_dtype, copy=False)

        # A fixed-batch export still has to serve whatever the caller sent.
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
        self._sess = None


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
