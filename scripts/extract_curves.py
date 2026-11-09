#!/usr/bin/env python3
"""Copy validation mAP50-95 per epoch out of training checkpoints into reports/training_curves.json.

    python scripts/extract_curves.py v9=models/football_detection_v9.pt v13=models/football_detection_v13_last.pt

Use a run's *last* checkpoint: it carries the whole curve, while best.pt of a run that was still improving
only carries the curve up to its best epoch. Existing entries for other runs are kept. `planned_epochs` is the
`epochs` the run was started with (ultralytics rewrites it to the reachable count when `time=` is set, so
those runs show no "stopped at" label), `epochs_done` is the number of epochs in the curve.
Needs torch + ultralytics (loading a checkpoint unpickles ultralytics classes).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "training_curves.json"
KEY = "metrics/mAP50-95(B)"


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    data = json.loads(OUT.read_text()) if OUT.exists() else {}
    for item in argv:
        name, _, path = item.partition("=")
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        curve = [float(x) for x in ckpt["train_results"][KEY]]
        args = ckpt.get("train_args") or {}
        data[name] = {
            "imgsz": int(args["imgsz"]),
            "planned_epochs": int(args["epochs"]),
            "epochs_done": len(curve),
            "val_map50_95": curve,
        }
        peak = max(range(len(curve)), key=curve.__getitem__) + 1
        print(f"[ok  ] {name}: {len(curve)} epochs, peak {curve[peak - 1]:.4f} at epoch {peak}")
    OUT.write_text(json.dumps(dict(sorted(data.items(), key=lambda kv: int(kv[0][1:]))), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
