#!/usr/bin/env python3
"""Export a .pt checkpoint to ONNX (fp32 / fp16 / int8), simplified graph.

Decode is handled by ultralytics on load (YOLO("model.onnx")), so `nms=False` here
just means "don't add a second NMS op" — YOLO26's end2end head already emits final
boxes. The raw-tensor path in app/backends/base.py is kept only for check_parity.py.

Usage:
    python scripts/export_onnx.py --weights models/football_detection_v1.pt
    python scripts/export_onnx.py --weights models/football_detection_v1.pt --quantize 16
    python scripts/export_onnx.py --weights models/football_detection_v1.pt --quantize 8 \
        --data dataset/data.yaml --split train --fraction 0.05
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=ROOT / "configs/train.yaml")
    ap.add_argument("--weights", default=None, help="default: models/best.pt")
    ap.add_argument("--out", default=None, help="default: export.onnx_path from config")
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--opset", type=int, default=None)
    ap.add_argument("--no-dynamic", action="store_true", help="fixed batch size of 1")
    ap.add_argument("--no-simplify", action="store_true")
    ap.add_argument(
        "--quantize",
        choices=["none", "16", "8"],
        default="none",
        help="none=fp32, 16=fp16, 8=int8 (needs --data for calibration)",
    )
    ap.add_argument("--data", type=Path, default=None, help="calibration data.yaml (int8 only)")
    ap.add_argument("--split", default="train", help="calibration split (int8 only)")
    ap.add_argument(
        "--fraction",
        type=float,
        default=300,
        help="int8 calibration size: >=1 is an image count, <1 is a ratio of --split (int8 only)",
    )
    args = ap.parse_args()

    with args.config.open() as fh:
        cfg = yaml.safe_load(fh)
    exp = cfg.get("export", {})

    weights = Path(args.weights or ROOT / "models/best.pt")
    if not weights.exists():
        print(f"[fail] weights not found: {weights}", file=sys.stderr)
        return 2

    imgsz = args.imgsz or cfg.get("train", {}).get("imgsz", 640)
    opset = args.opset or exp.get("opset", 12)
    dynamic = exp.get("dynamic_batch", True) and not args.no_dynamic
    simplify = exp.get("simplify", True) and not args.no_simplify

    quantize = None if args.quantize == "none" else int(args.quantize)
    suffix = {None: "", 16: "_fp16", 8: "_int8"}[quantize]
    default_out = ROOT / exp.get("onnx_path", "models/best.onnx")
    out_path = Path(args.out) if args.out else default_out.with_name(default_out.stem + suffix + ".onnx")

    export_kwargs = dict(
        format="onnx",
        imgsz=imgsz,
        opset=opset,
        dynamic=dynamic,
        simplify=simplify,
        nms=False,
    )
    if quantize is not None:
        export_kwargs["quantize"] = quantize
    if quantize == 8:
        data = args.data or ROOT / "dataset/data.yaml"
        if not Path(data).exists():
            print(f"[fail] int8 needs a calibration set; --data not found: {data}", file=sys.stderr)
            return 2
        frac = int(args.fraction) if args.fraction >= 1 else args.fraction
        export_kwargs.update(data=str(data), split=args.split, fraction=frac)
        if dynamic:
            print("[warn] int8 + dynamic batch: forcing fixed batch for a stable calibration")
            export_kwargs["dynamic"] = dynamic = False

    from ultralytics import YOLO

    print(f"[info] exporting {weights} imgsz={imgsz} opset={opset} dynamic={dynamic} "
          f"simplify={simplify} quantize={quantize}")
    model = YOLO(str(weights))
    produced = Path(model.export(**export_kwargs))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if produced.resolve() != out_path.resolve():
        shutil.move(str(produced), out_path)
    print(f"[ok  ] wrote {out_path.relative_to(ROOT)} ({out_path.stat().st_size/1e6:.1f} MB)")

    # --- verify the graph is loadable and has the signature the backend expects ---
    import onnx

    graph = onnx.load(str(out_path))
    onnx.checker.check_model(graph)

    def shape_of(vi) -> list:
        return [
            d.dim_param if d.HasField("dim_param") else d.dim_value
            for d in vi.type.tensor_type.shape.dim
        ]

    in_shape = shape_of(graph.graph.input[0])
    out_shape = shape_of(graph.graph.output[0])
    print(f"[ok  ] input  {graph.graph.input[0].name}: {in_shape}")
    print(f"[ok  ] output {graph.graph.output[0].name}: {out_shape}")

    if dynamic and isinstance(in_shape[0], int) and in_shape[0] != 0:
        print(f"[warn] dynamic requested but batch dim is fixed at {in_shape[0]}")

    names = {int(k): str(v) for k, v in dict(getattr(model, "names", {}) or {}).items()}
    meta = {
        "weights": str(weights),
        "onnx": str(out_path),
        "imgsz": imgsz,
        "opset": opset,
        "dynamic_batch": dynamic,
        "simplified": simplify,
        "quantize": quantize,
        "precision": {None: "fp32", 16: "fp16", 8: "int8"}[quantize],
        "input_shape": in_shape,
        "output_shape": out_shape,
        "classes": names,
        "size_mb": round(out_path.stat().st_size / 1e6, 2),
    }
    sidecar = out_path.with_suffix(".meta.json")
    sidecar.write_text(json.dumps(meta, indent=2))
    print(f"[ok  ] wrote {sidecar.relative_to(ROOT)}")
    print("\nnext: python scripts/check_parity.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
