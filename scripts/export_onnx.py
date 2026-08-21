#!/usr/bin/env python3
"""Export a .pt checkpoint to ONNX with dynamic batch, simplified graph.

NMS is deliberately NOT baked in: decoding lives in app/backends/base.py so every
backend shares one postprocess path.

Usage:
    python scripts/export_onnx.py
    python scripts/export_onnx.py --weights models/best.pt --out models/best.onnx
    python scripts/export_onnx.py --no-dynamic --opset 17
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
    ap.add_argument("--half", action="store_true", help="fp16 export (CUDA only)")
    args = ap.parse_args()

    with args.config.open() as fh:
        cfg = yaml.safe_load(fh)
    exp = cfg.get("export", {})

    weights = Path(args.weights or ROOT / "models/best.pt")
    if not weights.exists():
        print(f"[fail] weights not found: {weights}", file=sys.stderr)
        return 2

    out_path = Path(args.out or ROOT / exp.get("onnx_path", "models/best.onnx"))
    imgsz = args.imgsz or cfg.get("train", {}).get("imgsz", 640)
    opset = args.opset or exp.get("opset", 12)
    dynamic = exp.get("dynamic_batch", True) and not args.no_dynamic
    simplify = exp.get("simplify", True) and not args.no_simplify

    from ultralytics import YOLO

    print(f"[info] exporting {weights} imgsz={imgsz} opset={opset} dynamic={dynamic} simplify={simplify}")
    model = YOLO(str(weights))
    produced = Path(
        model.export(
            format="onnx",
            imgsz=imgsz,
            opset=opset,
            dynamic=dynamic,
            simplify=simplify,
            half=args.half,
            nms=False,
        )
    )

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
        "half": args.half,
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
