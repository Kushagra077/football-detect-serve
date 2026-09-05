#!/usr/bin/env python3
"""Cross-check scripts/eval_map.py against ultralytics' own YOLO.val().

eval_map.py rolls its own COCO-style 101-point mAP so torch and onnx score
through one code path. This script runs the reference implementation
(Ultralytics val(), pycocotools-equivalent) on the same split and thresholds
and prints a side-by-side diff, giving the custom evaluator an external witness.

It's a check, not a gate: it always exits 0. Small gaps are expected - NMS
tie-breaking, image- vs instance-level averaging, the empty `other` class. A
gap above --tol on a populated class is what's worth a look.

Usage:
    python scripts/crosscheck_map.py --weights models/football_detection_v3.pt --split val
    python scripts/crosscheck_map.py --weights models/football_detection_v3.pt --split test \
        --device mps
    python scripts/crosscheck_map.py --weights models/football_detection_v3.pt --split val \
        --compare-to reports/accuracy.torch.v3.json --tol 0.02
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def find_baseline(weights_stem: str, split: str) -> Path | None:
    """Locate the eval_map.py report for these weights + split.

    Reports are tagged either by the full weights stem (football_detection_v3)
    or the short model tag (v3); val reports have no suffix, test reports carry
    `.test`. Return the first candidate that exists.
    """
    suffix = "" if split == "val" else f".{split}"
    short = re.sub(r"^football_detection_", "", weights_stem)
    for tag in (weights_stem, short):
        cand = ROOT / f"reports/accuracy.torch.{tag}{suffix}.json"
        if cand.exists():
            return cand
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True, help="e.g. models/football_detection_v3.pt")
    ap.add_argument("--data", default=None, help="default: dataset/data.yaml")
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument(
        "--device",
        default="cpu",
        help="cpu / mps / 0 - mAP is device-independent, mps is just faster",
    )
    ap.add_argument("--conf", type=float, default=0.001, help="match eval_map.py (default 0.001)")
    ap.add_argument("--iou", type=float, default=0.7, help="NMS IoU, match eval_map.py (default 0.7)")
    ap.add_argument("--max-det", type=int, default=300, help="match eval_map.py (default 300)")
    ap.add_argument(
        "--compare-to",
        default=None,
        help="eval_map.py report JSON to diff against "
        "(default: reports/accuracy.torch.<stem>[.test].json if present)",
    )
    ap.add_argument("--tol", type=float, default=0.03, help="per-class gap to flag (default 0.03)")
    args = ap.parse_args()

    weights = resolve(Path(args.weights))
    if not weights.exists():
        print(f"[fail] weights not found: {weights}", file=sys.stderr)
        return 2

    data_yaml = resolve(Path(args.data)) if args.data else ROOT / "dataset/data.yaml"
    if not data_yaml.exists():
        print(f"[fail] data yaml not found: {data_yaml}\n       run: python scripts/prepare_data.py", file=sys.stderr)
        return 2

    # locate the eval_map.py report to compare against
    if args.compare_to:
        baseline_path: Path | None = resolve(Path(args.compare_to))
    else:
        baseline_path = find_baseline(weights.stem, args.split)

    names = yaml.safe_load(data_yaml.read_text()).get("names", {})
    if isinstance(names, list):
        names = dict(enumerate(names))
    name_to_cid = {v: k for k, v in names.items()}

    from ultralytics import YOLO

    print(
        f"[info] YOLO.val()  weights={weights.name} split={args.split} "
        f"conf={args.conf} iou={args.iou} max_det={args.max_det} device={args.device}"
    )
    model = YOLO(str(weights))
    r = model.val(
        data=str(data_yaml),
        split=args.split,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        device=args.device,
        verbose=False,
        plots=False,
    )

    # per-class AP50-95 for the classes ultralytics actually scored (skips 0-instance)
    ref_per_class = {int(cid): float(r.box.maps[int(cid)]) for cid in r.box.ap_class_index}
    ref_map, ref_map50 = float(r.box.map), float(r.box.map50)
    print(
        f"\n[ref ] YOLO.val()   mAP50-95={ref_map:.4f}  mAP50={ref_map50:.4f}  "
        f"(mean over {len(ref_per_class)} populated classes)"
    )

    if baseline_path is None or not baseline_path.exists():
        print("[warn] no eval_map.py report found to compare against - reference numbers only:")
        for cid, val in sorted(ref_per_class.items()):
            print(f"        {names.get(cid, cid):<12} {val:.4f}")
        return 0

    base = json.loads(baseline_path.read_text())
    print(
        f"[base] {baseline_path.relative_to(ROOT)}  "
        f"mAP50-95={base['overall']['map50_95']:.4f}  mAP50={base['overall']['map50']:.4f}"
    )

    row = f"\n{'class':<12}{'eval_map':>10}{'YOLO.val':>10}{'gap':>9}"
    print(row)
    print("-" * (len(row) - 1))
    worst = 0.0
    for cname, m in base["per_class"].items():
        custom = m.get("ap50_95")
        if custom is None or (isinstance(custom, float) and math.isnan(custom)):
            print(f"{cname:<12}{'n/a':>10}{'n/a':>10}{'-':>9}   (0 instances, skipped both sides)")
            continue
        ref = ref_per_class.get(name_to_cid.get(cname))
        if ref is None:
            print(f"{cname:<12}{custom:>10.4f}{'n/a':>10}{'-':>9}   (not scored by YOLO.val)")
            continue
        gap = abs(custom - ref)
        worst = max(worst, gap)
        print(f"{cname:<12}{custom:>10.4f}{ref:>10.4f}{gap:>9.4f}{'   <-- > tol' if gap > args.tol else ''}")
    print("-" * (len(row) - 1))
    mean_gap = abs(base["overall"]["map50_95"] - ref_map)
    print(f"{'MEAN':<12}{base['overall']['map50_95']:>10.4f}{ref_map:>10.4f}{mean_gap:>9.4f}")

    biggest = max(worst, mean_gap)
    print()
    if biggest > args.tol:
        print(
            f"[warn] largest gap {biggest:.4f} exceeds tol {args.tol} - worth a look. "
            f"Expected causes: NMS tie-breaking, image- vs instance-level averaging, "
            f"the empty `other` class."
        )
    else:
        print(
            f"[ok  ] evaluators agree within {args.tol} on every populated class "
            f"(largest gap {biggest:.4f}) - eval_map.py is externally validated."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
