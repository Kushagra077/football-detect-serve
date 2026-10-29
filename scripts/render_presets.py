#!/usr/bin/env python3
"""Draw one frame through the three presets, stacked, for the README.

Each preset runs at its own input size (read from its export sidecar), through the same
build_backend -> predict path the API uses. Use a frame you are allowed to publish: the
training footage is NDA-gated, so the README figure uses a free-licence clip instead.

    python scripts/render_presets.py --video ~/Downloads/pexel_demo_1.mp4 --frame 450 \
        --crop 40 240 940 500 --out reports/figures/presets_example.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.backends.base import build_backend, resolve_imgsz  # noqa: E402

PRESETS = [
    ("fast", "v3", "models/football_detection_v3.onnx"),
    ("balanced", "v7", "models/football_detection_v7.onnx"),
    ("accurate", "v6", "models/football_detection_v6.onnx"),
]
CLASS_BGR = {"ball": (0, 255, 255), "goalkeeper": (255, 128, 0), "player": (0, 200, 0),
             "referee": (0, 0, 255), "other": (200, 0, 200)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--crop", type=int, nargs=4, metavar=("X1", "Y1", "X2", "Y2"), default=None,
                    help="show only this region (detection still runs on the whole frame)")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--scale", type=float, default=1.6, help="enlarge the crop for legibility")
    ap.add_argument("--out", default="reports/figures/presets_example.png")
    args = ap.parse_args()

    cap = cv2.VideoCapture(str(Path(args.video).expanduser()))
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, frame = cap.read()
    if not ok:
        print(f"[fail] cannot read frame {args.frame} of {args.video}", file=sys.stderr)
        return 2
    x1, y1, x2, y2 = args.crop or (0, 0, frame.shape[1], frame.shape[0])

    panels = []
    for name, version, weights in PRESETS:
        imgsz, _ = resolve_imgsz(str(ROOT / weights))
        backend = build_backend("onnx", str(ROOT / weights), imgsz=imgsz, conf=args.conf)
        dets = backend.predict([frame])[0]
        backend.close()
        img = frame.copy()
        counts: dict[str, int] = {}
        for d in dets:
            counts[d.class_name] = counts.get(d.class_name, 0) + 1
            cv2.rectangle(img, (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2)),
                          CLASS_BGR.get(d.class_name, (255, 255, 255)), 2)
        img = cv2.resize(img[y1:y2, x1:x2], None, fx=args.scale, fy=args.scale, interpolation=cv2.INTER_CUBIC)
        summary = ", ".join(f"{k} {v}" for k, v in sorted(counts.items())) or "nothing found"
        bar = np.full((34, img.shape[1], 3), 32, np.uint8)
        cv2.putText(bar, f"{name} ({version}, {imgsz}px): {summary}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 1, cv2.LINE_AA)
        panels.append(np.vstack([bar, img]))
        print(f"{name:9s} {version} imgsz={imgsz:<5d} {summary}")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), np.vstack(panels))
    print(f"[ok  ] wrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
