#!/usr/bin/env python3
"""Latency harness. Writes reports/latency.json (+ a comparison figure).

Measures each backend at each batch size and splits the cost into
preprocess / infer / postprocess, so a regression can be attributed.

Usage:
    python scripts/benchmark.py                                  # every model found in models/
    python scripts/benchmark.py --backends torch onnx --batch-sizes 1 4 8
    python scripts/benchmark.py --models models/best.onnx models/best.int8.onnx --runs 200
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.backends.base import DetectorBackend, build_backend  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def sample_images(cfg: dict, n: int, imgsz: int) -> List[np.ndarray]:
    import cv2

    root = ROOT / cfg.get("dataset", {}).get("path", "data/football")
    paths = [p for p in sorted(root.rglob("*")) if p.suffix.lower() in IMAGE_EXTS][:n]
    imgs = [im for im in (cv2.imread(str(p)) for p in paths) if im is not None]
    if imgs:
        return imgs
    rng = np.random.default_rng(0)
    return [rng.integers(0, 256, (imgsz, imgsz, 3), np.uint8) for _ in range(n)]


def percentiles(samples_ms: List[float]) -> Dict[str, float]:
    arr = np.array(samples_ms, np.float64)
    return {
        "mean_ms": float(arr.mean()),
        "std_ms": float(arr.std()),
        "min_ms": float(arr.min()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p90_ms": float(np.percentile(arr, 90)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "max_ms": float(arr.max()),
    }


def bench_one(
    backend: DetectorBackend, images: List[np.ndarray], batch: int, runs: int, warmup: int
) -> dict:
    """Time a single (backend, batch) pair with a full stage breakdown."""
    pool = [images[i % len(images)] for i in range(batch)]

    for _ in range(warmup):
        backend.predict(pool)

    total, pre, inf, post = [], [], [], []
    for _ in range(runs):
        t0 = time.perf_counter()
        tensor, metas = backend.preprocess(pool)
        t1 = time.perf_counter()
        raw = backend.infer(tensor)
        t2 = time.perf_counter()
        backend.postprocess(raw, metas)
        t3 = time.perf_counter()

        pre.append((t1 - t0) * 1e3)
        inf.append((t2 - t1) * 1e3)
        post.append((t3 - t2) * 1e3)
        total.append((t3 - t0) * 1e3)

    stats = percentiles(total)
    stats.update(
        batch_size=batch,
        runs=runs,
        preprocess_mean_ms=float(np.mean(pre)),
        infer_mean_ms=float(np.mean(inf)),
        postprocess_mean_ms=float(np.mean(post)),
        per_image_mean_ms=stats["mean_ms"] / batch,
        throughput_img_s=batch * 1000.0 / stats["mean_ms"],
    )
    return stats


def discover_models(cfg: dict, backends: List[str], explicit: Optional[List[str]]) -> List[Tuple[str, Path]]:
    """Returns [(backend_kind, path)] for whatever is actually on disk."""
    if explicit:
        out = []
        for m in explicit:
            p = Path(m) if Path(m).is_absolute() else ROOT / m
            out.append(("torch" if p.suffix == ".pt" else "onnx", p))
        return out

    exp = cfg.get("export", {})
    candidates = [
        ("torch", ROOT / "models/best.pt"),
        ("onnx", ROOT / exp.get("onnx_path", "models/best.onnx")),
        ("onnx", ROOT / exp.get("int8_path", "models/best.int8.onnx")),
    ]
    return [(k, p) for k, p in candidates if k in backends and p.exists()]


def plot(results: List[dict], out: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[warn] matplotlib unavailable; skipping figure")
        return

    by_backend: Dict[str, List[dict]] = {}
    for r in results:
        by_backend.setdefault(r["backend"], []).append(r)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for name, rows in sorted(by_backend.items()):
        rows = sorted(rows, key=lambda r: r["batch_size"])
        bs = [r["batch_size"] for r in rows]
        ax1.plot(bs, [r["p95_ms"] for r in rows], marker="o", label=name)
        ax2.plot(bs, [r["throughput_img_s"] for r in rows], marker="o", label=name)

    ax1.set(xlabel="batch size", ylabel="p95 latency (ms)", title="Latency vs batch size")
    ax2.set(xlabel="batch size", ylabel="images / s", title="Throughput vs batch size")
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    print(f"[ok  ] wrote {out.relative_to(ROOT)}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=ROOT / "configs/train.yaml")
    ap.add_argument("--backends", nargs="*", default=["torch", "onnx"])
    ap.add_argument("--models", nargs="*", default=None, help="explicit model paths")
    ap.add_argument("--batch-sizes", nargs="*", type=int, default=[1, 2, 4, 8])
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=0, help="ORT intra-op threads (0 = default)")
    ap.add_argument("--report", type=Path, default=ROOT / "reports/latency.json")
    ap.add_argument("--figure", type=Path, default=ROOT / "reports/figures/latency.png")
    args = ap.parse_args()

    with args.config.open() as fh:
        cfg = yaml.safe_load(fh)

    imgsz = cfg.get("train", {}).get("imgsz", 640)
    scfg = cfg.get("serve", {})
    models = discover_models(cfg, args.backends, args.models)
    if not models:
        print(
            "[fail] no models found. Train and export first:\n"
            "       python scripts/train.py && python scripts/export_onnx.py",
            file=sys.stderr,
        )
        return 2

    images = sample_images(cfg, max(args.batch_sizes), imgsz)
    print(f"[info] device={args.device} imgsz={imgsz} runs={args.runs} warmup={args.warmup}")

    results: List[dict] = []
    for kind, path in models:
        print(f"\n[info] loading {kind}: {path.relative_to(ROOT)}")
        kwargs = {"intra_op_threads": args.threads} if kind == "onnx" else {}
        backend = build_backend(
            kind=kind,
            weights=str(path),
            imgsz=imgsz,
            conf=scfg.get("conf", 0.25),
            iou=scfg.get("iou", 0.45),
            max_det=scfg.get("max_det", 300),
            device=args.device,
            **kwargs,
        )
        for batch in args.batch_sizes:
            row = bench_one(backend, images, batch, args.runs, args.warmup)
            row.update(backend=backend.name, model=str(path.relative_to(ROOT)), size_mb=round(path.stat().st_size / 1e6, 2))
            results.append(row)
            print(
                f"       bs={batch:<3} p50={row['p50_ms']:7.2f}ms p95={row['p95_ms']:7.2f}ms "
                f"{row['throughput_img_s']:7.1f} img/s  "
                f"(pre {row['preprocess_mean_ms']:.1f} / inf {row['infer_mean_ms']:.1f} / post {row['postprocess_mean_ms']:.1f})"
            )
        backend.close()

    payload = {
        "environment": {
            "platform": platform.platform(),
            "processor": platform.processor() or platform.machine(),
            "python": platform.python_version(),
            "device": args.device,
            "ort_threads": args.threads or "default",
        },
        "settings": {"imgsz": imgsz, "runs": args.runs, "warmup": args.warmup},
        "results": results,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(payload, indent=2))
    print(f"\n[ok  ] wrote {args.report.relative_to(ROOT)}")

    plot(results, args.figure)

    bs1 = sorted([r for r in results if r["batch_size"] == 1], key=lambda r: r["p95_ms"])
    if len(bs1) > 1:
        best, worst = bs1[0], bs1[-1]
        print(
            f"\n[info] fastest at bs=1: {best['backend']} {best['p95_ms']:.2f}ms p95 "
            f"({worst['p95_ms']/best['p95_ms']:.2f}x faster than {worst['backend']})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
