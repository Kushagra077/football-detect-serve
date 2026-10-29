"""app/main.py's pure string-parsing logic - no server, no backends."""
from __future__ import annotations

import pytest

from app.main import _parse_models


def test_parse_models_single():
    assert _parse_models("torch=models/a.pt") == {"torch": "models/a.pt"}


def test_parse_models_multiple_preserves_order():
    parsed = _parse_models("torch=a.pt,onnx-fp32=b.onnx,onnx-int8=c.onnx")
    assert list(parsed.items()) == [
        ("torch", "a.pt"),
        ("onnx-fp32", "b.onnx"),
        ("onnx-int8", "c.onnx"),
    ]


def test_parse_models_strips_whitespace():
    assert _parse_models(" torch = a.pt , onnx = b.onnx ") == {"torch": "a.pt", "onnx": "b.onnx"}


def test_parse_models_ignores_empty_segments():
    assert _parse_models("torch=a.pt,,onnx=b.onnx,") == {"torch": "a.pt", "onnx": "b.onnx"}


def test_parse_models_missing_equals_raises():
    with pytest.raises(ValueError, match="bad MODELS entry"):
        _parse_models("torch-a.pt")


def test_parse_models_empty_string():
    assert _parse_models("") == {}


# --- per-model input size and backend type -------------------------------------------------

import json

from app.backends.base import resolve_imgsz
from app.main import _kind_of, _parse_entries


def test_parse_entries_reads_size_suffix():
    out = _parse_entries("fast=models/a.onnx@640, accurate=models/b.onnx@1280,plain=models/c.onnx")
    assert out == {
        "fast": ("models/a.onnx", 640),
        "accurate": ("models/b.onnx", 1280),
        "plain": ("models/c.onnx", None),
    }


def test_parse_models_drops_size_suffix():
    assert _parse_models("a=models/x.onnx@960") == {"a": "models/x.onnx"}


def test_at_sign_without_digits_is_part_of_the_path():
    assert _parse_entries("a=models/we@ird.onnx") == {"a": ("models/we@ird.onnx", None)}


def test_kind_follows_the_file_not_the_name():
    assert _kind_of("models/x.pt") == "torch"
    assert _kind_of("models/x.onnx") == "onnx"
    assert _kind_of("models/x_int8.onnx") == "onnx"


def test_resolve_imgsz_prefers_explicit_then_sidecar_then_default(tmp_path):
    onnx = tmp_path / "m.onnx"
    onnx.write_bytes(b"")
    assert resolve_imgsz(str(onnx), None, 640) == (640, "default")
    (tmp_path / "m.meta.json").write_text(json.dumps({"imgsz": 960}))
    assert resolve_imgsz(str(onnx), None, 640) == (960, "export sidecar")
    assert resolve_imgsz(str(onnx), 1280, 640) == (1280, "explicit")


def test_resolve_imgsz_ignores_a_broken_sidecar(tmp_path):
    onnx = tmp_path / "m.onnx"
    onnx.write_bytes(b"")
    (tmp_path / "m.meta.json").write_text("not json")
    assert resolve_imgsz(str(onnx), None, 640) == (640, "default")
