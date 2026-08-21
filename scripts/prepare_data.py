#!/usr/bin/env python3
"""Download the dataset, verify its class map, and count instances per class.

Fails loudly on the things that silently ruin a detection run:
  * class map that disagrees with configs/train.yaml (wrong ids => garbage mAP)
  * labels referencing class ids outside the map
  * missing image/label pairs
  * out-of-range or degenerate boxes

Usage:
    python scripts/prepare_data.py                       # use configs/train.yaml
    python scripts/prepare_data.py --url <zip-url>
    python scripts/prepare_data.py --skip-download       # verify what is on disk
"""
from __future__ import annotations

import argparse
import collections
import json
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "val", "test")


def load_config(path: Path) -> dict:
    with path.open() as fh:
        return yaml.safe_load(fh)


def download_zip(url: str, dest_dir: Path) -> Path:
    import requests

    dest_dir.mkdir(parents=True, exist_ok=True)
    archive = dest_dir / "dataset.zip"
    if archive.exists():
        print(f"[skip] archive already present: {archive}")
        return archive

    print(f"[get ] {url}")
    with requests.get(url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        written = 0
        with archive.open("wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
                written += len(chunk)
                if total:
                    pct = 100 * written / total
                    print(f"\r       {written/1e6:.1f}/{total/1e6:.1f} MB ({pct:.0f}%)", end="")
    print()
    return archive


def extract(archive: Path, dest: Path) -> None:
    print(f"[unzip] {archive} -> {dest}")
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(dest)


def find_data_yaml(root: Path) -> Path:
    direct = root / "data.yaml"
    if direct.exists():
        return direct
    candidates = sorted(root.rglob("data.yaml"))
    if not candidates:
        raise FileNotFoundError(f"no data.yaml found under {root}")
    return candidates[0]


def parse_names(raw) -> Dict[int, str]:
    if isinstance(raw, dict):
        return {int(k): str(v) for k, v in raw.items()}
    if isinstance(raw, (list, tuple)):
        return {i: str(v) for i, v in enumerate(raw)}
    raise TypeError(f"unsupported 'names' type in data.yaml: {type(raw)}")


def verify_class_map(found: Dict[int, str], expected: Dict[int, str]) -> List[str]:
    """Compare id->name maps exactly. Order and ids both matter."""
    errors: List[str] = []
    if not expected:
        print("[warn] no expected_classes in config; skipping class-map verification")
        return errors

    if set(found) != set(expected):
        errors.append(f"class ids differ: dataset={sorted(found)} expected={sorted(expected)}")
    for cid in sorted(set(found) & set(expected)):
        if found[cid].strip().lower() != expected[cid].strip().lower():
            errors.append(
                f"class {cid} name mismatch: dataset='{found[cid]}' expected='{expected[cid]}'"
            )
    return errors


def resolve_split_dir(data_yaml: Path, cfg: dict, split: str) -> Path | None:
    entry = cfg.get(split)
    if not entry:
        return None
    base = Path(cfg.get("path", ".")) if cfg.get("path") else data_yaml.parent
    if not base.is_absolute():
        base = (data_yaml.parent / base).resolve()
    p = Path(entry)
    return p if p.is_absolute() else (base / p).resolve()


def label_path_for(img: Path) -> Path:
    parts = list(img.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            break
    else:
        return img.with_suffix(".txt")
    return Path(*parts).with_suffix(".txt")


def scan_split(images_dir: Path, num_classes: int) -> Tuple[collections.Counter, dict, List[str]]:
    counts: collections.Counter = collections.Counter()
    problems: List[str] = []
    stats = {"images": 0, "labeled": 0, "empty_labels": 0, "missing_labels": 0, "boxes": 0}

    images = [p for p in sorted(images_dir.rglob("*")) if p.suffix.lower() in IMAGE_EXTS]
    stats["images"] = len(images)

    for img in images:
        lbl = label_path_for(img)
        if not lbl.exists():
            stats["missing_labels"] += 1
            if len(problems) < 25:
                problems.append(f"missing label for {img.name}")
            continue

        lines = [ln.strip() for ln in lbl.read_text().splitlines() if ln.strip()]
        if not lines:
            stats["empty_labels"] += 1  # legitimate background image
            continue

        stats["labeled"] += 1
        for ln_no, line in enumerate(lines, 1):
            fields = line.split()
            if len(fields) < 5:
                problems.append(f"{lbl.name}:{ln_no} expected 5 fields, got {len(fields)}")
                continue
            try:
                cid = int(float(fields[0]))
                x, y, w, h = (float(v) for v in fields[1:5])
            except ValueError:
                problems.append(f"{lbl.name}:{ln_no} non-numeric label values")
                continue

            if not 0 <= cid < num_classes:
                problems.append(f"{lbl.name}:{ln_no} class id {cid} outside 0..{num_classes-1}")
                continue
            if not all(0.0 <= v <= 1.0 for v in (x, y, w, h)):
                problems.append(f"{lbl.name}:{ln_no} coords not normalized: {x} {y} {w} {h}")
            if w <= 0 or h <= 0:
                problems.append(f"{lbl.name}:{ln_no} degenerate box w={w} h={h}")

            counts[cid] += 1
            stats["boxes"] += 1

    return counts, stats, problems


def print_table(names: Dict[int, str], per_split: Dict[str, collections.Counter]) -> None:
    splits = [s for s in SPLITS if s in per_split]
    header = f"{'id':>3}  {'class':<14}" + "".join(f"{s:>10}" for s in splits) + f"{'total':>10}"
    print("\n" + header)
    print("-" * len(header))
    for cid in sorted(names):
        row = f"{cid:>3}  {names[cid]:<14}"
        total = 0
        for s in splits:
            n = per_split[s].get(cid, 0)
            total += n
            row += f"{n:>10,}"
        print(row + f"{total:>10,}")
    print("-" * len(header))
    totals = f"{'':>3}  {'ALL':<14}"
    grand = 0
    for s in splits:
        n = sum(per_split[s].values())
        grand += n
        totals += f"{n:>10,}"
    print(totals + f"{grand:>10,}")

    # Class imbalance is the norm here (one ball vs. twenty-two players); surface it
    # rather than treating it as an error.
    all_counts = {cid: sum(per_split[s].get(cid, 0) for s in splits) for cid in names}
    nonzero = {c: n for c, n in all_counts.items() if n}
    if nonzero:
        ratio = max(nonzero.values()) / min(nonzero.values())
        print(f"\nimbalance ratio (max/min class): {ratio:.1f}x")
    for cid, n in all_counts.items():
        if n == 0:
            print(f"[warn] class {cid} '{names[cid]}' has ZERO instances")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=ROOT / "configs/train.yaml")
    ap.add_argument("--url", default=None, help="override dataset.url")
    ap.add_argument("--out", type=Path, default=None, help="override dataset.path")
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--strict", action="store_true", help="exit non-zero on any label problem")
    ap.add_argument("--report", type=Path, default=ROOT / "reports/dataset.json")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ds_cfg = cfg.get("dataset", {})
    url = args.url or ds_cfg.get("url") or ""
    out_dir = (args.out or ROOT / ds_cfg.get("path", "data/football")).resolve()

    if url and not args.skip_download:
        archive = download_zip(url, out_dir)
        if not (out_dir / "data.yaml").exists():
            extract(archive, out_dir)
    elif not out_dir.exists():
        print(f"[fail] {out_dir} does not exist and no dataset.url was given", file=sys.stderr)
        return 2

    data_yaml = find_data_yaml(out_dir)
    data_cfg = load_config(data_yaml)
    print(f"[ok  ] data.yaml: {data_yaml}")

    names = parse_names(data_cfg.get("names", {}))
    declared_nc = int(data_cfg.get("nc", len(names)))
    if declared_nc != len(names):
        print(f"[fail] nc={declared_nc} but names has {len(names)} entries", file=sys.stderr)
        return 2

    expected = {int(k): str(v) for k, v in (ds_cfg.get("expected_classes") or {}).items()}
    errors = verify_class_map(names, expected)
    if errors:
        print("[fail] class map verification failed:", file=sys.stderr)
        for e in errors:
            print(f"       - {e}", file=sys.stderr)
        return 2
    print(f"[ok  ] class map verified: {names}")

    per_split: Dict[str, collections.Counter] = {}
    all_stats: Dict[str, dict] = {}
    all_problems: Dict[str, List[str]] = {}

    for split in SPLITS:
        images_dir = resolve_split_dir(data_yaml, data_cfg, split)
        if images_dir is None:
            continue
        if not images_dir.exists():
            print(f"[warn] {split} path does not exist: {images_dir}")
            continue
        counts, stats, problems = scan_split(images_dir, len(names))
        per_split[split] = counts
        all_stats[split] = stats
        all_problems[split] = problems
        print(
            f"[ok  ] {split:<5} images={stats['images']:,} boxes={stats['boxes']:,} "
            f"empty={stats['empty_labels']:,} missing_labels={stats['missing_labels']:,}"
        )

    if not per_split:
        print("[fail] no usable splits found", file=sys.stderr)
        return 2

    print_table(names, per_split)

    problem_count = sum(len(v) for v in all_problems.values())
    if problem_count:
        print(f"\n[warn] {problem_count} label problem(s); first few:")
        for split, problems in all_problems.items():
            for p in problems[:5]:
                print(f"       {split}: {p}")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(
            {
                "data_yaml": str(data_yaml),
                "classes": names,
                "instances_per_class": {
                    s: {str(k): v for k, v in c.items()} for s, c in per_split.items()
                },
                "split_stats": all_stats,
                "problems": all_problems,
            },
            indent=2,
        )
    )
    print(f"\n[ok  ] wrote {args.report.relative_to(ROOT)}")

    if args.strict and problem_count:
        print("[fail] --strict and label problems present", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
