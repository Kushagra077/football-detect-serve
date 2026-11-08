"""The benchmark's no-dataset fallback must feed models 16:9 frames, not squares.

A square frame is letterboxed to the model's full square input, a 16:9 frame to a much smaller rectangle,
so square noise inflates every latency number (CI has no dataset and always takes the fallback).
"""
from __future__ import annotations

from scripts.benchmark import SYNTHETIC_FRAME_HW, sample_images


def test_fallback_frames_are_16_by_9(tmp_path):
    cfg = {"dataset": {"path": str(tmp_path / "no_such_dataset")}}  # absolute path: ROOT / abs = abs
    frames, source = sample_images(cfg, 3)

    assert source == "synthetic"
    assert len(frames) == 3
    for f in frames:
        assert f.shape == (*SYNTHETIC_FRAME_HW, 3)
        assert f.dtype.name == "uint8"
    h, w = SYNTHETIC_FRAME_HW
    assert abs(w / h - 16 / 9) < 0.01


def test_fallback_is_deterministic(tmp_path):
    cfg = {"dataset": {"path": str(tmp_path / "no_such_dataset")}}
    a, _ = sample_images(cfg, 1)
    b, _ = sample_images(cfg, 1)
    assert (a[0] == b[0]).all()


def test_real_frames_are_used_when_present(tmp_path):
    import cv2
    import numpy as np

    d = tmp_path / "images" / "val"
    d.mkdir(parents=True)
    cv2.imwrite(str(d / "a.jpg"), np.zeros((72, 128, 3), np.uint8))
    frames, source = sample_images({"dataset": {"path": str(tmp_path)}}, 2)

    assert source == "dataset"
    assert frames[0].shape == (72, 128, 3)
