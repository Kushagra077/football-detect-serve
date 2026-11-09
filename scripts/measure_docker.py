#!/usr/bin/env python3
"""Time the four presets inside the Docker image and record RAM, so the README's container numbers have a file.

    docker build -t football-detect-serve:local .
    python scripts/measure_docker.py                      # -> reports/docker.presets.json

For each preset it sends one real 1080p dataset frame as a multipart upload with `curl` (the whole request:
upload, JPEG decode, inference, response), 3 untimed warm-ups then 20 timed requests, sequentially, once with the
default dynamic batching and once with `?batch=false`. The server's own `inference_ms` / `total_ms` from the
response are recorded next to the curl time. After all requests it reads the container's memory from
`docker stats`. Needs a running Docker daemon; starts and removes its own container.
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts._paths import display_path  # noqa: E402

PRESETS = ["fast", "balanced", "accurate", "max"]


def sh(*cmd: str) -> str:
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout.strip()


def first_test_frame() -> Path:
    frame = next(iter(sorted((ROOT / "dataset/images/test").glob("*.jpg"))), None)
    if frame is None:
        raise SystemExit("no frame under dataset/images/test; pass --frame")
    return frame


def one_request(url: str, frame: Path) -> tuple[float, dict]:
    body = Path(tempfile.mkstemp(suffix=".json")[1])
    out = sh("curl", "-sS", "-o", str(body), "-w", "%{time_total}", "-F", f"file=@{frame}", url)
    payload = json.loads(body.read_text())
    body.unlink()
    return float(out) * 1000, payload


def pct(values: list[float], q: float) -> float:
    s = sorted(values)
    return s[min(len(s) - 1, round(q * (len(s) - 1)))]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", default="football-detect-serve:local")
    ap.add_argument("--frame", type=Path, default=None)
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--port", type=int, default=7861)
    ap.add_argument("--report", type=Path, default=ROOT / "reports/docker.presets.json")
    args = ap.parse_args()

    frame = args.frame or first_test_frame()
    name = "fds-measure"
    subprocess.run(["docker", "rm", "-f", name], capture_output=True)
    sh("docker", "run", "-d", "--name", name, "-p", f"{args.port}:7860", args.image)
    base = f"http://localhost:{args.port}"
    try:
        for _ in range(120):  # the image's healthcheck allows 90 s of start-up
            if subprocess.run(["curl", "-fsS", f"{base}/healthz"], capture_output=True).returncode == 0:
                break
            time.sleep(2)
        else:
            raise SystemExit("container never became healthy")
        health = json.loads(sh("curl", "-sS", f"{base}/healthz"))

        results, requests_sent = {}, 0
        for preset in PRESETS:
            results[preset] = {}
            for mode, extra in (("batching_on", ""), ("batching_off", "&batch=false")):
                url = f"{base}/predict?backend={preset}{extra}"
                for _ in range(args.warmup):
                    one_request(url, frame)
                runs = [one_request(url, frame) for _ in range(args.runs)]
                requests_sent += args.warmup + args.runs
                wall = [r[0] for r in runs]
                results[preset][mode] = {
                    "curl_ms_p50": statistics.median(wall),
                    "curl_ms_p95": pct(wall, 0.95),
                    "curl_ms_min": min(wall),
                    "server_inference_ms_p50": statistics.median(r[1]["inference_ms"] for r in runs),
                    "server_total_ms_p50": statistics.median(r[1]["total_ms"] for r in runs),
                    "detections_median": statistics.median(len(r[1]["detections"]) for r in runs),
                }
                print(f"[ok  ] {preset:<9} {mode:<12} curl p50 {results[preset][mode]['curl_ms_p50']:7.1f} ms "
                      f"(server inference {results[preset][mode]['server_inference_ms_p50']:.1f} ms)")

        mem = sh("docker", "stats", "--no-stream", "--format", "{{.MemUsage}}", name)
        image_bytes = int(sh("docker", "image", "inspect", "--format", "{{.Size}}", args.image))
        info = json.loads(sh("docker", "info", "--format", "{{json .}}"))
        report = {
            "image": args.image,
            "image_size_gb": round(image_bytes / 1e9, 2),
            "container_memory_after_all_requests": mem,
            "requests_sent_before_memory_reading": requests_sent,
            "docker_vm": {"cpus": info["NCPU"], "memory_gib": round(info["MemTotal"] / 2**30, 2)},
            "host": platform.platform(),
            "frame": {"file": display_path(frame), "source": "dataset test split, 1920x1080 JPEG"},
            "method": f"curl multipart upload, {args.warmup} untimed warm-ups then {args.runs} timed sequential "
                      f"requests per preset and batching mode; whole request incl. upload and decode",
            "healthz": health,
            "results": results,
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2))
        print(f"[ok  ] image {report['image_size_gb']} GB, memory {mem}, wrote {display_path(args.report)}")
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
