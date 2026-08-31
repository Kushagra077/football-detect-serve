#!/usr/bin/env python3
"""Step 6 - model latency harness. Writes reports/latency.json (+ a figure).

Measures each backend's end-to-end predict() latency on CPU across a batch-size
sweep. Reports p50/p95/p99, throughput, and the preprocess/inference/postprocess
split that ultralytics tracks internally (predictor.speed).

Concurrency (1/4/16 simultaneous requests) is NOT measured here - that is the
HTTP load test in Step 8. This script isolates "how fast is the model itself".

Usage:
    python scripts/benchmark.py                          # every model in models/
    python scripts/benchmark.py --models models/football_detection_v1.onnx --runs 200
    python scripts/benchmark.py --batch-sizes 1 4 8 --omp-threads 4
"""
from __future__ import annotations

import os
import sys

# --- pin thread env BEFORE numpy / torch / onnxruntime import ---
def _early_omp_threads(default: int = 4) -> int:
    n = default
    argv = sys.argv
    for i, a in enumerate(argv):
        if a == "--omp-threads" and i + 1 < len(argv):
            n = int(argv[i + 1])
        elif a.startswith("--omp-threads="):
            n = int(a.split("=", 1)[1])
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = str(n)
    return n


_OMP_THREADS = _early_omp_threads()

import argparse  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Dict, List, Optional, Tuple  # noqa: E402

import numpy as np  # noqa: E402
import yaml  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.backends.base import DetectorBackend, build_backend  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MODEL_GLOBS = ("football_detection_v*.pt", "football_detection_v*.onnx")


def sample_images(cfg: dict, n: int, imgsz: int) -> List[np.ndarray]:
    import cv2

    root = ROOT / cfg.get("dataset", {}).get("path", "dataset")
    for sub in ("images/val", "images/train", ""):
        d = root / sub
        if d.is_dir():
            paths = [p for p in sorted(d.rglob("*")) if p.suffix.lower() in IMAGE_EXTS][:n]
            if paths:
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


def bench_one(backend: DetectorBackend, images: List[np.ndarray], batch: int, runs: int, warmup: int) -> dict:
    """Time predict() for one (backend, batch) pair."""
    pool = [images[i % len(images)] for i in range(batch)]

    for _ in range(warmup):
        backend.predict(pool)

    wall: List[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        backend.predict(pool)
        wall.append((time.perf_counter() - t0) * 1e3)

    stats = percentiles(wall)
    stats.update(
        batch_size=batch,
        runs=runs,
        per_image_mean_ms=stats["mean_ms"] / batch,
        throughput_img_s=batch * 1000.0 / stats["mean_ms"],
        ultralytics_speed_ms=_ul_speed(backend),  # per-image pre/inf/post from ultralytics
    )
    return stats


def _ul_speed(backend: DetectorBackend) -> Optional[Dict[str, float]]:
    pred = getattr(getattr(backend, "_yolo", None), "predictor", None)
    speed = getattr(pred, "speed", None)
    if isinstance(speed, dict):
        return {k: round(float(v), 3) for k, v in speed.items() if v is not None}
    return None


def discover_models(explicit: Optional[List[str]]) -> List[Tuple[str, Path]]:
    """Returns [(backend_kind, path)] for whatever is on disk."""
    if explicit:
        out = []
        for m in explicit:
            p = Path(m) if Path(m).is_absolute() else ROOT / m
            out.append(("torch" if p.suffix == ".pt" else "onnx", p))
        return out

    found: List[Tuple[str, Path]] = []
    for pat in MODEL_GLOBS:
        for p in sorted((ROOT / "models").glob(pat)):
            found.append(("torch" if p.suffix == ".pt" else "onnx", p))
    return found


def physical_cores() -> Optional[int]:
    try:
        import psutil

        return psutil.cpu_count(logical=False)
    except Exception:
        return None


def versions() -> Dict[str, str]:
    out = {"python": platform.python_version()}
    for mod in ("torch", "onnxruntime", "ultralytics", "numpy"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:
            out[mod] = "n/a"
    return out


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
        by_backend.setdefault(f"{r['backend']} [{r['model']}]", []).append(r)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for name, rows in sorted(by_backend.items()):
        rows = sorted(rows, key=lambda r: r["batch_size"])
        bs = [r["batch_size"] for r in rows]
        ax1.plot(bs, [r["p95_ms"] for r in rows], marker="o", label=name)
        ax2.plot(bs, [r["throughput_img_s"] for r in rows], marker="o", label=name)

    ax1.set(xlabel="batch size", ylabel="p95 latency (ms)", title="Latency vs batch size (CPU)")
    ax2.set(xlabel="batch size", ylabel="images / s", title="Throughput vs batch size (CPU)")
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    print(f"[ok  ] wrote {out.relative_to(ROOT)}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=ROOT / "configs/train.yaml")
    ap.add_argument("--models", nargs="*", default=None, help="explicit model paths (default: models/football_detection_v*)")
    ap.add_argument("--batch-sizes", nargs="*", type=int, default=[1, 2, 4, 8])
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--omp-threads", type=int, default=_OMP_THREADS, help="pinned before import; recorded in the report")
    ap.add_argument("--report", type=Path, default=ROOT / "reports/latency.json")
    ap.add_argument("--figure", type=Path, default=ROOT / "reports/figures/latency.png")
    args = ap.parse_args()

    with args.config.open() as fh:
        cfg = yaml.safe_load(fh)

    imgsz = cfg.get("train", {}).get("imgsz", 640)
    scfg = cfg.get("serve", {})
    models = discover_models(args.models)
    if not models:
        print("[fail] no models found. Export first: python scripts/export_onnx.py --weights models/football_detection_v1.pt", file=sys.stderr)
        return 2

    images = sample_images(cfg, max(args.batch_sizes), imgsz)
    print(f"[info] device={args.device} imgsz={imgsz} omp_threads={args.omp_threads} "
          f"runs={args.runs} warmup={args.warmup}  models={len(models)}")

    results: List[dict] = []
    for kind, path in models:
        rel = path.relative_to(ROOT)
        print(f"\n[info] {kind}: {rel}")
        backend = build_backend(
            kind=kind,
            weights=str(path),
            imgsz=imgsz,
            conf=scfg.get("conf", 0.25),
            iou=scfg.get("iou", 0.45),
            max_det=scfg.get("max_det", 300),
            device=args.device,
        )
        for batch in args.batch_sizes:
            row = bench_one(backend, images, batch, args.runs, args.warmup)
            row.update(backend=backend.name, model=str(rel), size_mb=round(path.stat().st_size / 1e6, 2))
            results.append(row)
            spd = row["ultralytics_speed_ms"] or {}
            print(f"       bs={batch:<2} p50={row['p50_ms']:7.2f} p95={row['p95_ms']:7.2f} p99={row['p99_ms']:7.2f} ms  "
                  f"{row['throughput_img_s']:6.1f} img/s  "
                  f"(pre {spd.get('preprocess', 0):.1f} / inf {spd.get('inference', 0):.1f} / post {spd.get('postprocess', 0):.1f})")
        backend.close()

    payload = {
        "environment": {
            "platform": platform.platform(),
            "processor": platform.processor() or platform.machine(),
            "logical_cores": os.cpu_count(),
            "physical_cores": physical_cores(),
            "omp_threads": args.omp_threads,
            "device": args.device,
            "versions": versions(),
        },
        "settings": {"imgsz": imgsz, "runs": args.runs, "warmup": args.warmup, "batch_sizes": args.batch_sizes},
        "note": "single-request latency; concurrency 1/4/16 is measured by the Step 8 HTTP load test",
        "results": results,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(payload, indent=2))
    print(f"\n[ok  ] wrote {args.report.relative_to(ROOT)}")

    plot(results, args.figure)

    bs1 = sorted([r for r in results if r["batch_size"] == 1], key=lambda r: r["p95_ms"])
    if len(bs1) > 1:
        best, worst = bs1[0], bs1[-1]
        print(f"\n[info] fastest at bs=1: {best['backend']} [{best['model']}] {best['p95_ms']:.2f}ms p95 "
              f"({worst['p95_ms'] / best['p95_ms']:.2f}x faster than {worst['backend']} [{worst['model']}])")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
