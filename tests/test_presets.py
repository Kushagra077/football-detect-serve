"""The four serving presets must agree everywhere they are declared.

A preset lives in app/main.py (_DEFAULT_MODELS), app/gradio_app.py (_MODELS / _IMGSZ), the Dockerfile (COPY) and
models/*.meta.json (input size). Forgetting one of them gives a container that crashes at startup or a model run
at the wrong input size (silently less accurate), so keep them in one test.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from app import gradio_app
from app.main import _DEFAULT_MODELS, _parse_entries

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {  # preset -> (weights file, input size it was trained at)
    "fast": ("models/football_detection_v3.onnx", 640),
    "balanced": ("models/football_detection_v7.onnx", 960),
    "accurate": ("models/football_detection_v6.onnx", 1280),
    "max": ("models/football_detection_v13.onnx", 1280),
}


def test_default_registry_has_the_four_presets_in_speed_order():
    entries = _parse_entries(_DEFAULT_MODELS)
    assert list(entries) == list(EXPECTED)
    for name, (path, _) in EXPECTED.items():
        assert entries[name][0] == path


@pytest.mark.parametrize("name", list(EXPECTED))
def test_sidecar_records_the_training_input_size(name):
    path, imgsz = EXPECTED[name]
    meta = ROOT / Path(path).with_suffix(".meta.json")
    assert Path(ROOT / path).exists(), f"{path} is missing (is it whitelisted in .gitignore?)"
    assert json.loads(meta.read_text())["imgsz"] == imgsz


@pytest.mark.parametrize("name", list(EXPECTED))
def test_dockerfile_copies_each_preset_and_its_sidecar(name):
    path, _ = EXPECTED[name]
    dockerfile = (ROOT / "Dockerfile").read_text()
    assert path in dockerfile
    assert str(Path(path).with_suffix(".meta.json")) in dockerfile


@pytest.mark.parametrize("name", list(EXPECTED))
def test_gradio_lists_each_preset_at_its_training_size(name):
    path, imgsz = EXPECTED[name]
    labels = [label for label, p in gradio_app._MODELS.items() if p == path]
    assert labels, f"{path} is not offered in the Gradio dropdown"
    if imgsz != gradio_app.IMGSZ:  # entries at the default size need no override
        for label in labels:
            assert gradio_app._IMGSZ.get(label) == imgsz


def _dockerignored(path: str) -> bool:
    """Evaluate .dockerignore for one repo-relative path: the last matching rule wins, '!' re-includes."""
    import fnmatch

    ignored = False
    for raw in (ROOT / ".dockerignore").read_text().splitlines():
        rule = raw.strip()
        if not rule or rule.startswith("#"):
            continue
        negate = rule.startswith("!")
        pattern = rule[1:] if negate else rule
        if fnmatch.fnmatchcase(path, pattern.rstrip("/")):
            ignored = not negate
    return ignored


@pytest.mark.parametrize("name", list(EXPECTED))
def test_dockerignore_does_not_drop_a_preset_from_the_build_context(name):
    """A too-broad rule (models/..._v1*.onnx also matched v13) made `COPY` fail with 'not found' at build time."""
    path, _ = EXPECTED[name]
    assert not _dockerignored(path), f"{path} is excluded by .dockerignore"
    assert not _dockerignored(str(Path(path).with_suffix(".meta.json")))


def test_dockerignore_still_drops_the_retired_models():
    for retired in ("models/football_detection_v1.onnx", "models/football_detection_v1_int8.onnx",
                    "models/football_detection_v2.onnx", "models/football_detection_v2_fp16.onnx"):
        assert _dockerignored(retired), retired
