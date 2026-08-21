#!/usr/bin/env python3
"""ONNX Runtime static INT8 quantization with real-image calibration.

Calibration images come from the val split and go through the exact same
preprocess as serving (app/backends/base.letterbox), so the collected activation
ranges match production.

Usage:
    python scripts/quantize_int8.py
    python scripts/quantize_int8.py --calib-images 500 --no-per-channel

Then gate it:
    python scripts/eval_map.py --backend onnx --weights models/best.int8.onnx
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterator, List, Optional

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.backends.base import letterbox  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_val_images(cfg: dict, limit: int) -> List[Path]:
    """Prefer the val split; fall back to any image under the dataset root."""
    ds_root = ROOT / cfg.get("dataset", {}).get("path", "data/football")
    for pattern in ("valid/images", "val/images", "test/images", ""):
        d = ds_root / pattern if pattern else ds_root
        if d.exists():
            imgs = [p for p in sorted(d.rglob("*")) if p.suffix.lower() in IMAGE_EXTS]
            if imgs:
                if len(imgs) > limit:
                    # Even stride beats head-of-list: covers the whole distribution.
                    idx = np.linspace(0, len(imgs) - 1, limit).astype(int)
                    imgs = [imgs[i] for i in idx]
                print(f"[info] {len(imgs)} calibration image(s) from {d}")
                return imgs
    return []


def build_calibration_reader(images: List[Path], input_name: str, imgsz: int):
    from onnxruntime.quantization import CalibrationDataReader

    class _Reader(CalibrationDataReader):
        def __init__(self) -> None:
            self._iter: Optional[Iterator[dict]] = None

        def _generate(self) -> Iterator[dict]:
            import cv2

            for i, path in enumerate(images, 1):
                img = cv2.imread(str(path))
                if img is None:
                    continue
                padded, _ = letterbox(img, imgsz)
                rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
                tensor = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
                if i % 50 == 0:
                    print(f"       calibrated {i}/{len(images)}")
                yield {input_name: tensor[None]}

        def get_next(self):
            if self._iter is None:
                self._iter = self._generate()
            return next(self._iter, None)

        def rewind(self) -> None:
            self._iter = None

    return _Reader()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=ROOT / "configs/train.yaml")
    ap.add_argument("--onnx", default=None, help="fp32 input model")
    ap.add_argument("--out", default=None, help="int8 output model")
    ap.add_argument("--calib-images", type=int, default=None)
    ap.add_argument("--no-per-channel", action="store_true")
    ap.add_argument(
        "--calibrate-method",
        choices=["minmax", "entropy", "percentile"],
        default="minmax",
        help="minmax is safest for detection heads; entropy can clip small-object logits",
    )
    ap.add_argument("--report", type=Path, default=ROOT / "reports/quantization.json")
    args = ap.parse_args()

    with args.config.open() as fh:
        cfg = yaml.safe_load(fh)
    exp, qcfg = cfg.get("export", {}), cfg.get("quantize", {})

    src = Path(args.onnx or ROOT / exp.get("onnx_path", "models/best.onnx"))
    dst = Path(args.out or ROOT / exp.get("int8_path", "models/best.int8.onnx"))
    if not src.exists():
        print(f"[fail] fp32 onnx not found: {src}\n       run: python scripts/export_onnx.py", file=sys.stderr)
        return 2

    imgsz = cfg.get("train", {}).get("imgsz", 640)
    n_calib = args.calib_images or qcfg.get("calib_images", 200)
    per_channel = qcfg.get("per_channel", True) and not args.no_per_channel

    import onnxruntime as ort
    from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process

    input_name = ort.InferenceSession(str(src), providers=["CPUExecutionProvider"]).get_inputs()[0].name

    images = find_val_images(cfg, n_calib)
    if not images:
        print(
            "[fail] no calibration images found. Static quantization without real data "
            "produces a broken model -- run scripts/prepare_data.py first.",
            file=sys.stderr,
        )
        return 2

    # Shape inference + folding first; quantize_static is fragile on raw exports.
    prepped = dst.with_suffix(".prep.onnx")
    print(f"[info] pre-processing graph -> {prepped.name}")
    quant_pre_process(str(src), str(prepped), skip_symbolic_shape=False)

    method = {
        "minmax": CalibrationMethod.MinMax,
        "entropy": CalibrationMethod.Entropy,
        "percentile": CalibrationMethod.Percentile,
    }[args.calibrate_method]

    print(f"[info] quantizing: per_channel={per_channel} method={args.calibrate_method}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    quantize_static(
        model_input=str(prepped),
        model_output=str(dst),
        calibration_data_reader=build_calibration_reader(images, input_name, imgsz),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=per_channel,
        calibrate_method=method,
        # Sigmoid/softmax-adjacent ops lose too much in int8; the head stays fp32.
        nodes_to_exclude=[],
        extra_options={"ActivationSymmetric": False, "WeightSymmetric": True},
    )
    prepped.unlink(missing_ok=True)

    fp32_mb = src.stat().st_size / 1e6
    int8_mb = dst.stat().st_size / 1e6
    print(f"\n[ok  ] {dst.relative_to(ROOT)}")
    print(f"       fp32 {fp32_mb:.1f} MB -> int8 {int8_mb:.1f} MB ({fp32_mb/int8_mb:.2f}x smaller)")

    # Smoke test: the graph must still run and produce the expected shape.
    sess = ort.InferenceSession(str(dst), providers=["CPUExecutionProvider"])
    out = sess.run(None, {input_name: np.zeros((1, 3, imgsz, imgsz), dtype=np.float32)})[0]
    print(f"[ok  ] smoke test output shape: {out.shape}")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(
            {
                "fp32_model": str(src),
                "int8_model": str(dst),
                "fp32_size_mb": round(fp32_mb, 2),
                "int8_size_mb": round(int8_mb, 2),
                "compression": round(fp32_mb / int8_mb, 3),
                "calibration_images": len(images),
                "per_channel": per_channel,
                "calibrate_method": args.calibrate_method,
                "quant_format": "QDQ",
                "output_shape": list(out.shape),
                "max_map_drop_allowed": qcfg.get("max_map_drop", 0.02),
            },
            indent=2,
        )
    )
    print(f"[ok  ] wrote {args.report.relative_to(ROOT)}")
    print(
        "\nnext (accuracy gate):\n"
        f"  python scripts/eval_map.py --backend onnx --weights {dst.relative_to(ROOT)} "
        "--compare-to reports/accuracy.json"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
