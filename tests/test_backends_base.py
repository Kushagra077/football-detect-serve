"""app/backends/base.py: to_detections() conversion and the build_backend() factory.

TorchBackend/OnnxBackend are imported (safe - their module-level imports are just
numpy/typing, ultralytics/onnxruntime are only imported lazily inside load()), but
.load() itself is monkeypatched so these tests never touch real weights or ML deps.
"""
from __future__ import annotations

import numpy as np
import pytest

from app.backends.base import build_backend, to_detections


class _FakeBox:
    """Duck-types just enough of ultralytics' Boxes API for to_detections()."""

    def __init__(self, xyxy, cls, conf):
        self.xyxy = [np.array(xyxy, dtype=np.float32)]
        self.cls = [cls]
        self.conf = [conf]


class _FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes


def test_to_detections_maps_known_class():
    result = _FakeResult([_FakeBox([10, 20, 30, 40], 2, 0.9)])
    dets = to_detections(result, {2: "player"})
    assert len(dets) == 1
    d = dets[0]
    assert (d.x1, d.y1, d.x2, d.y2) == (10.0, 20.0, 30.0, 40.0)
    assert d.class_id == 2
    assert d.class_name == "player"
    assert d.score == pytest.approx(0.9)


def test_to_detections_unknown_class_falls_back_to_id_string():
    result = _FakeResult([_FakeBox([0, 0, 1, 1], 99, 0.5)])
    dets = to_detections(result, {2: "player"})
    assert dets[0].class_name == "99"


def test_to_detections_empty_boxes():
    assert to_detections(_FakeResult([]), {0: "ball"}) == []


def test_build_backend_routes_torch(monkeypatch):
    from app.backends.torch_backend import TorchBackend

    monkeypatch.setattr(TorchBackend, "load", lambda self: self)
    backend = build_backend(kind="torch", weights="fake.pt")
    assert isinstance(backend, TorchBackend)


def test_build_backend_routes_onnx(monkeypatch):
    from app.backends.onnx_backend import OnnxBackend

    monkeypatch.setattr(OnnxBackend, "load", lambda self: self)
    backend = build_backend(kind="onnx-int8", weights="fake.onnx")
    assert isinstance(backend, OnnxBackend)


def test_build_backend_unknown_kind_raises():
    with pytest.raises(ValueError, match="unknown backend"):
        build_backend(kind="tensorflow", weights="x")
