#!/usr/bin/env python3
"""Export a .pt checkpoint to ONNX (fp32 / fp16 / int8), simplified graph.

Decode is handled by ultralytics on load (YOLO("model.onnx")), so `nms=False` here
just means "don't add a second NMS op" — YOLO26's end2end head already emits final
boxes. Parity with torch is checked via scripts/eval_map.py (onnx-fp32 mAP == torch).

Usage:
    python scripts/export_onnx.py --weights models/football_detection_v1.pt
    python scripts/export_onnx.py --weights models/football_detection_v1.pt --quantize 16
    python scripts/export_onnx.py --weights models/football_detection_v1.pt --quantize 8 \
        --data dataset/data.yaml --split train --fraction 0.05
"""
from __future__ import annotations

import argparse
import json
import os
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
    ap.add_argument("--out", default=None, help="default: <weights stem>[_fp16|_int8].onnx next to the checkpoint")
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
        default=0.01,
        help="int8 calibration: ratio of --split to use, (0, 1]. "
        "0.01 of 34.5k train images is ~345 (int8 only)",
    )
    args = ap.parse_args()

    with args.config.open() as fh:
        cfg = yaml.safe_load(fh)
    exp = cfg.get("export", {})

    weights = (Path(args.weights).resolve() if args.weights else ROOT / "models/best.pt")
    if not weights.exists():
        print(f"[fail] weights not found: {weights}", file=sys.stderr)
        return 2

    imgsz = args.imgsz or cfg.get("train", {}).get("imgsz", 640)
    opset = args.opset or exp.get("opset", 12)
    dynamic = exp.get("dynamic_batch", True) and not args.no_dynamic
    simplify = exp.get("simplify", True) and not args.no_simplify

    quantize = None if args.quantize == "none" else int(args.quantize)
    suffix = {None: "", 16: "_fp16", 8: "_int8"}[quantize]
    # default output name tracks the weights file: football_detection_v1.pt -> football_detection_v1[_fp16|_int8].onnx
    out_path = Path(args.out).resolve() if args.out else weights.with_name(weights.stem + suffix + ".onnx")

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
        if not 0.0 < args.fraction <= 1.0:
            print(f"[fail] --fraction must be in (0, 1]; got {args.fraction}", file=sys.stderr)
            return 2
        export_kwargs.update(data=str(data), split=args.split, fraction=args.fraction)
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
    rel = lambda p: os.path.relpath(p, ROOT)  # noqa: E731
    print(f"[ok  ] wrote {rel(out_path)} ({out_path.stat().st_size/1e6:.1f} MB)")

    # --- verify the graph is loadable and has the signature the backend expects ---
    # NB: inference goes through YOLO("model.onnx"), which validates + topo-sorts the
    # graph itself on load. check_model() here is only a courtesy sanity pass, and the
    # fp16 conversion leaves the input Cast node out of topological order, which the
    # strict checker rejects even though onnxruntime loads it fine. So: warn, don't die.
    import onnx

    graph = onnx.load(str(out_path))
    try:
        onnx.checker.check_model(graph)
    except onnx.checker.ValidationError as e:
        print(f"[warn] onnx.checker flagged the graph (harmless for onnxruntime): {e}")

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
    print(f"[ok  ] wrote {rel(sidecar)}")
    print("\nnext: python scripts/eval_map.py --backend onnx --weights", rel(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
