#!/usr/bin/env python3
"""Build a shuffled, cross-sequence INT8 calibration list.

Why this exists: ultralytics' own dataset loader (ultralytics/data/base.py)
sorts every image path alphabetically, then - when export()'s `fraction` < 1 -
just slices off the first N. Our filenames are `<sequence>_<frame:06d>.jpg`
(e.g. SNMOT-060_000001.jpg), so a sorted-then-sliced 1% is the first ~345
frames of whichever sequence sorts first: one ~14-second clip, one stadium,
one lighting condition, two kits. That's a bad calibration set - narrow and
repetitive, not representative of the dataset the model actually serves.

This script picks calibration images itself - stratified across every
training sequence, randomly (not consecutively) within each one - and writes
a manifest + a tiny data.yaml. Point scripts/export_onnx.py at that yaml with
`--split train --fraction 1.0`: fraction=1.0 means ultralytics' sort-then-
slice never removes anything, so our selection is exactly what gets used.

Usage:
    python scripts/build_calibration_set.py
    python scripts/build_calibration_set.py --n 500 --seed 1
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path
from random import Random

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SEQ_RE = re.compile(r"^(?P<seq>.+)_(?P<frame>\d{6})\.(?:jpg|jpeg|png)$", re.IGNORECASE)


def group_by_sequence(images_dir: Path) -> dict[str, list[Path]]:
    """{sequence_name: [image paths]}, e.g. 'SNMOT-060' -> its 750 frames.

    Files that don't match `<sequence>_<frame:06d>.<ext>` are skipped with a
    warning rather than crashing - same warn-and-skip posture as
    convert_mot_to_yolo.py uses for unknown roles.
    """
    groups: dict[str, list[Path]] = defaultdict(list)
    skipped = 0
    for path in sorted(images_dir.iterdir()):
        if not path.is_file():
            continue
        m = SEQ_RE.match(path.name)
        if not m:
            skipped += 1
            continue
        groups[m.group("seq")].append(path)
    if skipped:
        print(f"[warn] {skipped} file(s) in {images_dir} didn't match <sequence>_<frame>.<ext>, skipped")
    return dict(groups)


def stratified_sample(groups: dict[str, list[Path]], n: int, seed: int) -> tuple[list[Path], dict[str, int]]:
    """Pick ~n images total, spread evenly across every sequence, chosen
    randomly (not consecutively) within each one.

    Sequences with fewer frames than their even share contribute everything
    they have; the shortfall is redistributed round-robin over the remaining
    sequences (largest-first) so the total still lands at n whenever enough
    images exist overall. Returns (picked images, {sequence: count taken}).
    """
    rng = Random(seed)
    seq_names = sorted(groups)  # deterministic order before shuffling picks
    remaining = {name: list(groups[name]) for name in seq_names}
    for pool in remaining.values():
        rng.shuffle(pool)

    quota = {name: 0 for name in seq_names}
    want = n
    active = list(seq_names)
    while want > 0 and active:
        share = max(1, want // len(active))
        progressed = False
        for name in list(active):
            available = len(remaining[name]) - quota[name]
            take = min(share, available, want)
            if take > 0:
                quota[name] += take
                want -= take
                progressed = True
            if quota[name] >= len(remaining[name]):
                active.remove(name)
            if want <= 0:
                break
        if not progressed:
            break  # every remaining sequence is exhausted

    picked: list[Path] = []
    for name in seq_names:
        picked.extend(remaining[name][: quota[name]])
    rng.shuffle(picked)  # don't leave them grouped by sequence in the manifest
    return picked, quota


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images-dir", type=Path, default=ROOT / "dataset/images/train")
    ap.add_argument("--data-yaml", type=Path, default=ROOT / "dataset/data.yaml", help="source of nc/names/val")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "dataset")
    ap.add_argument("--n", type=int, default=345, help="target calibration set size (default matches the old 1%% count)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not args.images_dir.exists():
        print(f"[fail] images dir not found: {args.images_dir}", file=sys.stderr)
        return 2

    groups = group_by_sequence(args.images_dir)
    if not groups:
        print(f"[fail] no <sequence>_<frame>.<ext> images found in {args.images_dir}", file=sys.stderr)
        return 2

    total_available = sum(len(v) for v in groups.values())
    n = min(args.n, total_available)
    picked, quota = stratified_sample(groups, n, args.seed)

    manifest_path = args.out_dir / "calib_manifest.txt"
    manifest_path.write_text("\n".join(str(p.resolve()) for p in picked) + "\n")

    src = yaml.safe_load(args.data_yaml.read_text())
    calib_yaml = {
        "path": src.get("path", str(args.images_dir.parents[1].resolve())),
        "train": str(manifest_path.resolve()),  # what --split train actually reads
        "val": src.get("val", "images/val"),  # required key, unused by export's calibration pass
        "nc": src["nc"],
        "names": src["names"],
    }
    calib_yaml_path = args.out_dir / "calib_data.yaml"
    calib_yaml_path.write_text(yaml.safe_dump(calib_yaml, sort_keys=False))

    per_seq = ", ".join(f"{name}={count}" for name, count in sorted(quota.items()) if count)
    print(f"[ok  ] picked {len(picked)}/{total_available} images across {len(groups)} sequences")
    print(f"       {per_seq}")
    print(f"[ok  ] wrote {manifest_path.relative_to(ROOT)}")
    print(f"[ok  ] wrote {calib_yaml_path.relative_to(ROOT)}")
    print(
        "\nNext: python scripts/export_onnx.py --weights <ckpt> --quantize 8 "
        f"--data {calib_yaml_path.relative_to(ROOT)} --split train --fraction 1.0"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
