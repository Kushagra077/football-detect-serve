#!/usr/bin/env python3
"""GATE: torch vs onnx numerical parity. Non-zero exit means DO NOT SHIP.

Compares three things on identical inputs:
  1. raw head tensors        (max abs / mean abs diff vs. thresholds)
  2. decoded detection counts (same boxes survive conf + NMS)
  3. matched box geometry     (IoU of corresponding boxes)

A graph that passes (1) but fails (2) usually means a threshold sits right on a
score boundary -- worth knowing before mAP mysteriously moves.

Usage:
    python scripts/check_parity.py
    python scripts/check_parity.py --onnx models/best.int8.onnx --max-abs-diff 0.05
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.backends.base import Detection, letterbox  # noqa: E402
from app.backends.onnx_backend import OnnxBackend  # noqa: E402
from app.backends.torch_backend import TorchBackend  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def sample_images(cfg: dict, n: int) -> List[np.ndarray]:
    """Real val images if available, deterministic noise otherwise."""
    import cv2

    imgsz = cfg.get("train", {}).get("imgsz", 640)
    root = ROOT / cfg.get("dataset", {}).get("path", "data/football")
    paths = [p for p in sorted(root.rglob("*")) if p.suffix.lower() in IMAGE_EXTS][:n]

    if paths:
        print(f"[info] using {len(paths)} real image(s) from {root}")
        imgs = [cv2.imread(str(p)) for p in paths]
        return [im for im in imgs if im is not None]

    print("[warn] no dataset images found; falling back to synthetic noise")
    rng = np.random.default_rng(0)
    return [rng.integers(0, 256, (imgsz, imgsz, 3), dtype=np.uint8) for _ in range(n)]


def iou(a: Detection, b: Detection) -> float:
    x1, y1 = max(a.x1, b.x1), max(a.y1, b.y1)
    x2, y2 = min(a.x2, b.x2), min(a.y2, b.y2)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a.x2 - a.x1) * max(0.0, a.y2 - a.y1)
    area_b = max(0.0, b.x2 - b.x1) * max(0.0, b.y2 - b.y1)
    return inter / (area_a + area_b - inter + 1e-9)


def match_boxes(ref: List[Detection], test: List[Detection]) -> dict:
    """Greedy same-class matching, ref order (already score-descending)."""
    unused = list(range(len(test)))
    ious, score_deltas = [], []
    for r in ref:
        best_j, best_iou = None, 0.0
        for j in unused:
            if test[j].class_id != r.class_id:
                continue
            v = iou(r, test[j])
            if v > best_iou:
                best_j, best_iou = j, v
        if best_j is not None and best_iou > 0.5:
            unused.remove(best_j)
            ious.append(best_iou)
            score_deltas.append(abs(r.score - test[best_j].score))
    return {
        "matched": len(ious),
        "unmatched_ref": len(ref) - len(ious),
        "unmatched_test": len(unused),
        "min_iou": float(min(ious)) if ious else None,
        "mean_iou": float(np.mean(ious)) if ious else None,
        "max_score_delta": float(max(score_deltas)) if score_deltas else 0.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=ROOT / "configs/train.yaml")
    ap.add_argument("--weights", default=None, help="torch .pt (default models/best.pt)")
    ap.add_argument("--onnx", default=None, help="onnx model (default export.onnx_path)")
    ap.add_argument("--num-samples", type=int, default=None)
    ap.add_argument("--max-abs-diff", type=float, default=None)
    ap.add_argument("--max-mean-diff", type=float, default=None)
    ap.add_argument("--min-box-iou", type=float, default=0.99)
    ap.add_argument("--report", type=Path, default=ROOT / "reports/parity.json")
    args = ap.parse_args()

    with args.config.open() as fh:
        cfg = yaml.safe_load(fh)
    pcfg = cfg.get("parity", {})
    scfg = cfg.get("serve", {})

    pt = Path(args.weights or ROOT / "models/best.pt")
    onnx_path = Path(args.onnx or ROOT / cfg["export"]["onnx_path"])
    for p in (pt, onnx_path):
        if not p.exists():
            print(f"[fail] missing model: {p}", file=sys.stderr)
            return 2

    n = args.num_samples or pcfg.get("num_samples", 8)
    max_abs = args.max_abs_diff if args.max_abs_diff is not None else pcfg.get("max_abs_diff", 1e-3)
    max_mean = (
        args.max_mean_diff if args.max_mean_diff is not None else pcfg.get("max_mean_diff", 1e-4)
    )

    common = dict(
        imgsz=cfg.get("train", {}).get("imgsz", 640),
        conf=scfg.get("conf", 0.25),
        iou=scfg.get("iou", 0.45),
        max_det=scfg.get("max_det", 300),
    )
    torch_be = TorchBackend(weights=str(pt), device="cpu", **common).load()
    onnx_be = OnnxBackend(weights=str(onnx_path), device="cpu", **common).load()
    print(f"[info] comparing {torch_be.name} vs {onnx_be.name} on {n} sample(s)")

    images = sample_images(cfg, n)
    if not images:
        print("[fail] no usable input images", file=sys.stderr)
        return 2

    # One shared preprocess => any diff is purely the graph.
    batch, metas = torch_be.preprocess(images)

    raw_t = torch_be.infer(batch)
    raw_o = onnx_be.infer(batch)

    if raw_t.shape != raw_o.shape:
        print(f"[fail] output shape mismatch: torch={raw_t.shape} onnx={raw_o.shape}", file=sys.stderr)
        return 1

    diff = np.abs(raw_t.astype(np.float64) - raw_o.astype(np.float64))
    abs_diff, mean_diff = float(diff.max()), float(diff.mean())
    denom = np.maximum(np.abs(raw_t), 1e-6)
    rel_diff = float((diff / denom).max())

    dets_t = torch_be.postprocess(raw_t, metas)
    dets_o = onnx_be.postprocess(raw_o, metas)

    per_image = []
    count_mismatch = 0
    worst_iou = 1.0
    for i, (dt, do) in enumerate(zip(dets_t, dets_o)):
        m = match_boxes(dt, do)
        m.update(index=i, torch_boxes=len(dt), onnx_boxes=len(do))
        per_image.append(m)
        if len(dt) != len(do):
            count_mismatch += 1
        if m["min_iou"] is not None:
            worst_iou = min(worst_iou, m["min_iou"])

    print("\n--- raw tensor ---")
    print(f"  shape          : {raw_t.shape}")
    print(f"  max |diff|     : {abs_diff:.3e}   (limit {max_abs:.3e})")
    print(f"  mean |diff|    : {mean_diff:.3e}   (limit {max_mean:.3e})")
    print(f"  max rel diff   : {rel_diff:.3e}")

    print("\n--- decoded detections ---")
    for m in per_image:
        iou_txt = f"{m['min_iou']:.4f}" if m["min_iou"] is not None else "n/a"
        print(
            f"  img {m['index']}: torch={m['torch_boxes']:>3} onnx={m['onnx_boxes']:>3} "
            f"matched={m['matched']:>3} min_iou={iou_txt} "
            f"max_dscore={m['max_score_delta']:.4f}"
        )

    failures = []
    if abs_diff > max_abs:
        failures.append(f"max abs diff {abs_diff:.3e} > {max_abs:.3e}")
    if mean_diff > max_mean:
        failures.append(f"mean abs diff {mean_diff:.3e} > {max_mean:.3e}")
    if count_mismatch:
        failures.append(f"detection count differs on {count_mismatch}/{len(per_image)} image(s)")
    if worst_iou < args.min_box_iou:
        failures.append(f"worst matched-box IoU {worst_iou:.4f} < {args.min_box_iou}")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(
            {
                "torch_weights": str(pt),
                "onnx_model": str(onnx_path),
                "backends": [torch_be.name, onnx_be.name],
                "num_samples": len(images),
                "raw": {
                    "shape": list(raw_t.shape),
                    "max_abs_diff": abs_diff,
                    "mean_abs_diff": mean_diff,
                    "max_rel_diff": rel_diff,
                },
                "thresholds": {"max_abs_diff": max_abs, "max_mean_diff": max_mean,
                               "min_box_iou": args.min_box_iou},
                "per_image": per_image,
                "passed": not failures,
                "failures": failures,
            },
            indent=2,
        )
    )
    print(f"\n[ok  ] wrote {args.report.relative_to(ROOT)}")

    if failures:
        print("\n[FAIL] parity gate FAILED:", file=sys.stderr)
        for f in failures:
            print(f"       - {f}", file=sys.stderr)
        return 1

    print("\n[PASS] parity gate passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
