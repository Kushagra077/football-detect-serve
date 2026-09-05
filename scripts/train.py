#!/usr/bin/env python3
"""Thin wrapper over ultralytics YOLO.train(), driven by configs/train.yaml.

Everything under the `train:` key is forwarded verbatim, so any ultralytics
argument can be set in YAML without touching this file. CLI overrides win.

Usage:
    python scripts/train.py --weights yolo26n.pt --out models/football_detection_v3.pt
    python scripts/train.py --config configs/train.yaml --set epochs=50 batch=8
    python scripts/train.py --resume
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def coerce(value: str):
    """Parse a CLI --set value into bool/int/float/str."""
    low = value.lower()
    if low in {"true", "false"}:
        return low == "true"
    if low in {"none", "null"}:
        return None
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            pass
    return value


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=ROOT / "configs/train.yaml")
    ap.add_argument("--weights", default=None, help="override model.weights")
    ap.add_argument("--data", default=None, help="override dataset.data_yaml")
    ap.add_argument(
        "--out",
        default=None,
        help="copy the trained checkpoint here, e.g. models/football_detection_v3.pt "
        "(default: models/<run name>.pt; no copy if omitted and run name is unset)",
    )
    ap.add_argument("--resume", action="store_true")
    ap.add_argument(
        "--set",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="override any train: key, e.g. --set epochs=50 imgsz=960",
    )
    args = ap.parse_args()

    with args.config.open() as fh:
        cfg = yaml.safe_load(fh)

    train_args = dict(cfg.get("train", {}))
    for item in args.set:
        if "=" not in item:
            ap.error(f"--set expects KEY=VALUE, got '{item}'")
        key, raw = item.split("=", 1)
        train_args[key] = coerce(raw)

    data_yaml = args.data or cfg["dataset"]["data_yaml"]
    data_path = Path(data_yaml)
    if not data_path.is_absolute():
        data_path = ROOT / data_path
    if not data_path.exists():
        print(
            f"[fail] dataset yaml not found: {data_path}\n"
            f"       run: python scripts/prepare_data.py",
            file=sys.stderr,
        )
        return 2

    weights = args.weights or cfg["model"]["weights"]

    from ultralytics import YOLO

    print(f"[info] weights={weights} data={data_path}")
    print(f"[info] train args: {json.dumps(train_args, default=str)}")

    model = YOLO(weights)
    results = model.train(data=str(data_path), resume=args.resume, **train_args)

    save_dir = Path(getattr(results, "save_dir", train_args.get("project", "runs/detect")))
    best = save_dir / "weights" / "best.pt"
    print(f"[ok  ] training finished; save_dir={save_dir}")

    if best.exists():
        if args.out:
            target = Path(args.out)
            if not target.is_absolute():
                target = ROOT / target
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(best, target)
            print(f"[ok  ] copied best weights -> {target.relative_to(ROOT)}")
        else:
            print(
                f"[info] best weights at {best.relative_to(ROOT)} - not copied "
                f"(pass --out models/football_detection_vN.pt to copy + name it)"
            )

        metrics = getattr(results, "results_dict", None)
        if metrics:
            out = ROOT / "reports/train_metrics.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps({k: float(v) for k, v in metrics.items()}, indent=2))
            print(f"[ok  ] wrote {out.relative_to(ROOT)}")
    else:
        print(f"[warn] expected best.pt at {best}, not found")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
