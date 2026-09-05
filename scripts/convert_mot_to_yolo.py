#!/usr/bin/env python3
"""Convert MOT-format tracking data into a YOLO detection dataset.

Reads <src>/{train,test}/<sequence>/{gt/gt.txt, gameinfo.ini, seqinfo.ini, img1/}
and writes dataset/{images,labels}/{train,val,test}/ + dataset/data.yaml.

Class map (must match configs/train.yaml expected_classes):
    0 ball
    1 goalkeeper
    2 player
    3 referee
    4 other

Split: the source ships train/ and test/ only, no val/. Splitting by frame would
leak near-duplicate consecutive frames across the split, so every 5th train
sequence (by sorted name) becomes val; the rest stay train. The source's test/
becomes YOLO test.

Images are copied (not symlinked) so the dataset/ folder is self-contained and
zippable for upload elsewhere (e.g. Kaggle). This duplicates ~17GB on disk.

Usage:
    python scripts/convert_mot_to_yolo.py --src <path-to-raw-mot-data> --out dataset
"""
from __future__ import annotations

import argparse
import configparser
import re
import shutil
import sys
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]

CLASS_IDS = {"ball": 0, "goalkeeper": 1, "player": 2, "referee": 3, "other": 4}

# gameinfo.ini role strings -> canonical class name
ROLE_MAP = {
    "player team left": "player",
    "player team right": "player",
    "goalkeeper team left": "goalkeeper",
    "goalkeeper team right": "goalkeeper",
    "goalkeepers team left": "goalkeeper",  # dataset has this typo variant too
    "goalkeepers team right": "goalkeeper",
    "referee": "referee",
    "ball": "ball",
    "other": "other",
}

TRACKLET_RE = re.compile(r"^trackletID_(\d+)$")


def parse_gameinfo(path: Path) -> dict[int, str]:
    """Map trackID -> canonical class name from gameinfo.ini."""
    cfg = configparser.ConfigParser()
    cfg.optionxform = str  # preserve key case; default lowercases "trackletID_18" -> "trackletid_18"
    cfg.read(path)
    track_class: dict[int, str] = {}
    for key, value in cfg["Sequence"].items():
        m = TRACKLET_RE.match(key)
        if not m:
            continue
        track_id = int(m.group(1))
        role = value.split(";", 1)[0].strip().lower()
        cls = ROLE_MAP.get(role)
        if cls is None:
            print(f"[warn] {path}: unknown role '{role}' for track {track_id}, skipping")
            continue
        track_class[track_id] = cls
    return track_class


def parse_seqinfo(path: Path) -> tuple[int, int]:
    cfg = configparser.ConfigParser()
    cfg.read(path)
    sec = cfg["Sequence"]
    return int(sec["imWidth"]), int(sec["imHeight"])


def convert_sequence(seq_dir: Path, images_out: Path, labels_out: Path) -> tuple[int, int]:
    """Returns (num_frames_labeled, num_boxes)."""
    track_class = parse_gameinfo(seq_dir / "gameinfo.ini")
    img_w, img_h = parse_seqinfo(seq_dir / "seqinfo.ini")

    frame_boxes: dict[int, list[str]] = {}
    gt_path = seq_dir / "gt" / "gt.txt"
    with gt_path.open() as fh:
        for line in fh:
            fields = line.strip().split(",")
            if len(fields) < 6:
                continue
            frame, track_id, x, y, w, h = (
                int(fields[0]),
                int(fields[1]),
                float(fields[2]),
                float(fields[3]),
                float(fields[4]),
                float(fields[5]),
            )
            cls = track_class.get(track_id)
            if cls is None:
                continue
            cid = CLASS_IDS[cls]
            cx = (x + w / 2) / img_w
            cy = (y + h / 2) / img_h
            nw = w / img_w
            nh = h / img_h
            frame_boxes.setdefault(frame, []).append(f"{cid} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

    seq_name = seq_dir.name
    img_dir = seq_dir / "img1"
    num_boxes = 0
    images = sorted(img_dir.glob("*.jpg"))
    for img_path in tqdm(images, desc=seq_name, leave=False):
        frame = int(img_path.stem)
        lines = frame_boxes.get(frame, [])
        num_boxes += len(lines)

        stem = f"{seq_name}_{frame:06d}"
        label_path = labels_out / f"{stem}.txt"
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""))

        dst_path = images_out / f"{stem}.jpg"
        if dst_path.exists() or dst_path.is_symlink():
            dst_path.unlink()
        shutil.copy2(img_path, dst_path)

    return len(images), num_boxes


def write_data_yaml(out_dir: Path) -> None:
    names_block = "\n".join(f"  {cid}: {name}" for name, cid in sorted(CLASS_IDS.items(), key=lambda kv: kv[1]))
    (out_dir / "data.yaml").write_text(
        f"path: {out_dir.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n"
        f"nc: {len(CLASS_IDS)}\n"
        f"names:\n{names_block}\n"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True, help="raw MOT data root (train/, test/)")
    ap.add_argument("--out", type=Path, default=ROOT / "dataset")
    ap.add_argument("--val-every", type=int, default=5, help="hold out every Nth train sequence as val")
    args = ap.parse_args()

    if not args.src.exists():
        print(f"[fail] source dir not found: {args.src}", file=sys.stderr)
        return 2

    train_seqs = sorted(p for p in (args.src / "train").iterdir() if p.is_dir())
    test_seqs = sorted(p for p in (args.src / "test").iterdir() if p.is_dir())

    val_seqs = train_seqs[args.val_every - 1 :: args.val_every]
    val_names = {p.name for p in val_seqs}
    train_seqs = [p for p in train_seqs if p.name not in val_names]

    splits = {"train": train_seqs, "val": val_seqs, "test": test_seqs}

    for split, seqs in splits.items():
        images_out = args.out / "images" / split
        labels_out = args.out / "labels" / split
        images_out.mkdir(parents=True, exist_ok=True)
        labels_out.mkdir(parents=True, exist_ok=True)

        total_frames = 0
        total_boxes = 0
        for seq_dir in tqdm(seqs, desc=f"{split} sequences"):
            n_frames, n_boxes = convert_sequence(seq_dir, images_out, labels_out)
            total_frames += n_frames
            total_boxes += n_boxes
        print(f"[ok  ] {split:<5} sequences={len(seqs):<3} frames={total_frames:,} boxes={total_boxes:,}")

    write_data_yaml(args.out)
    print(f"[ok  ] wrote {args.out / 'data.yaml'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
