#!/usr/bin/env python3
"""Fetch pretrained/released weights into models/ (this dir is gitignored).

Set WEIGHTS_BASE_URL (or --base-url) to wherever your release artifacts live.

Usage:
    python models/download_weights.py                    # all known artifacts
    python models/download_weights.py football_detection_v1.onnx
    python models/download_weights.py --base-url https://example.com/releases/v0.1
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# name -> optional sha256 ("" = skip verification)
ARTIFACTS = {
    "football_detection_v1.pt": "",
    "football_detection_v1.onnx": "",
    "football_detection_v1_int8.onnx": "",
    "football_detection_v2.pt": "",
    "football_detection_v2.onnx": "",
    "football_detection_v2_int8.onnx": "",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path) -> None:
    import requests

    print(f"[get ] {url}")
    with requests.get(url, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        written = 0
        tmp = dest.with_suffix(dest.suffix + ".part")
        with tmp.open("wb") as fh:
            for chunk in resp.iter_content(1 << 20):
                fh.write(chunk)
                written += len(chunk)
                if total:
                    print(f"\r       {written/1e6:.1f}/{total/1e6:.1f} MB", end="")
        print()
        tmp.replace(dest)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="*", default=None, help="artifact names (default: all)")
    ap.add_argument("--base-url", default=os.getenv("WEIGHTS_BASE_URL", ""))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if not args.base_url:
        print(
            "[fail] no base URL. Pass --base-url or set WEIGHTS_BASE_URL.\n"
            "       Known artifacts: " + ", ".join(ARTIFACTS),
            file=sys.stderr,
        )
        return 2

    wanted = args.names or list(ARTIFACTS)
    unknown = [n for n in wanted if n not in ARTIFACTS]
    if unknown:
        print(f"[fail] unknown artifact(s): {unknown}", file=sys.stderr)
        return 2

    for name in wanted:
        dest = HERE / name
        if dest.exists() and not args.force:
            print(f"[skip] {name} already present")
            continue
        download(f"{args.base_url.rstrip('/')}/{name}", dest)

        expected = ARTIFACTS[name]
        if expected:
            actual = sha256(dest)
            if actual != expected:
                print(f"[fail] {name} sha256 mismatch\n       want {expected}\n       got  {actual}", file=sys.stderr)
                dest.unlink()
                return 1
            print(f"[ok  ] {name} checksum verified")
        else:
            print(f"[ok  ] {name} ({dest.stat().st_size/1e6:.1f} MB, checksum not pinned)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
