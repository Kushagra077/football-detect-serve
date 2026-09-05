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
