#!/usr/bin/env python3
"""Render one detection frame for the README (reports/figures/sample_detection.png).

Runs a single image through the real serving path (build_backend -> predict, same
code the API uses) and draws labelled boxes. This is a figure, not a metric - pick
a frame that shows the model doing something legible (crowd + ball if you can).

Usage:
    python scripts/render_sample.py --weights models/football_detection_v3.pt
    python scripts/render_sample.py --weights models/football_detection_v3.pt \
        --image dataset/images/test/SNMOT-116_000450.jpg --conf 0.3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# BGR, one per class id (0..4). Distinct hues, readable on grass.
CLASS_BGR = {
    0: (0, 255, 255),    # ball        - yellow
    1: (255, 128, 0),    # goalkeeper  - blue
    2: (0, 200, 0),      # player      - green
    3: (0, 0, 255),      # referee     - red
    4: (200, 0, 200),    # other       - magenta
}


def resolve(p: Path) -> Path:
    return p if p.is_absolute() else ROOT / p


def pick_default_image() -> Path | None:
    for split in ("test", "val", "train"):
        d = ROOT / "dataset/images" / split
        if d.is_dir():
            imgs = sorted(d.glob("*.jpg")) or sorted(d.glob("*.png"))
            if imgs:
                return imgs[len(imgs) // 2]  # a middle frame, not the first
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True, help="e.g. models/football_detection_v3.pt")
    ap.add_argument("--image", default=None, help="default: a middle frame from dataset/images/test")
    ap.add_argument("--out", default="reports/figures/sample_detection.png")
    ap.add_argument("--backend", default="torch", help="torch / onnx (default torch)")
    ap.add_argument("--conf", type=float, default=0.25, help="min score to draw (default 0.25)")
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--max-det", type=int, default=300)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--data", default="dataset/data.yaml", help="for class names")
    args = ap.parse_args()

    weights = resolve(Path(args.weights))
    if not weights.exists():
        print(f"[fail] weights not found: {weights}", file=sys.stderr)
        return 2

    image_path = resolve(Path(args.image)) if args.image else pick_default_image()
    if image_path is None or not image_path.exists():
        print(f"[fail] no image (pass --image); tried default under dataset/images/", file=sys.stderr)
        return 2

    data_yaml = resolve(Path(args.data))
    class_names = {}
    if data_yaml.exists():
        names = yaml.safe_load(data_yaml.read_text()).get("names", {})
        class_names = dict(enumerate(names)) if isinstance(names, list) else {int(k): v for k, v in names.items()}

    img = cv2.imread(str(image_path))
    if img is None:
        print(f"[fail] could not read image: {image_path}", file=sys.stderr)
        return 2

    from app.backends.base import build_backend

    print(f"[info] {args.backend} {weights.name}  <-  {image_path.relative_to(ROOT)}  conf>={args.conf}")
    backend = build_backend(
        args.backend,
        str(weights),
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        device=args.device,
        class_names=class_names,
    )
    dets = backend.predict([img])[0]

    canvas = img.copy()
    counts: dict[str, int] = {}
    for d in sorted(dets, key=lambda d: d.score):  # low scores first, so strong boxes draw on top
        color = CLASS_BGR.get(d.class_id, (255, 255, 255))
        p1, p2 = (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2))
        cv2.rectangle(canvas, p1, p2, color, 2)
        label = f"{d.class_name} {d.score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(canvas, (p1[0], p1[1] - th - 5), (p1[0] + tw + 2, p1[1]), color, -1)
        cv2.putText(canvas, label, (p1[0] + 1, p1[1] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
        counts[d.class_name] = counts.get(d.class_name, 0) + 1

    caption = f"{weights.stem} | {args.backend} | {len(dets)} det: " + ", ".join(
        f"{k} {v}" for k, v in sorted(counts.items())
    )
    cv2.rectangle(canvas, (0, canvas.shape[0] - 22), (canvas.shape[1], canvas.shape[0]), (0, 0, 0), -1)
    cv2.putText(canvas, caption, (6, canvas.shape[0] - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    out = resolve(Path(args.out))
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), canvas)
    print(f"[ok  ] wrote {out.relative_to(ROOT)}  ({len(dets)} boxes: {counts})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
