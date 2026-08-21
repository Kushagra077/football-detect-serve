#!/usr/bin/env python3
"""Per-class mAP for ANY backend through ONE code path.

The metric is computed here (COCO-style 101-point AP over IoU 0.50:0.95) rather
than delegated to ultralytics' validator, because the validator only accepts
torch models. Running torch / onnx-fp32 / onnx-int8 through the same evaluator is
the only way the numbers are comparable.

Usage:
    python scripts/eval_map.py --backend torch --weights models/best.pt
    python scripts/eval_map.py --backend onnx  --weights models/best.onnx
    python scripts/eval_map.py --backend onnx  --weights models/best.int8.onnx \
        --compare-to reports/accuracy.json --max-map-drop 0.02
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.backends.base import Detection, build_backend  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IOU_THRESHOLDS = np.arange(0.5, 1.0, 0.05)  # COCO 0.50:0.95


# ---------------- dataset ----------------


def resolve_split(data_yaml: Path, data_cfg: dict, split: str) -> Optional[Path]:
    entry = data_cfg.get(split)
    if not entry:
        return None
    base = Path(data_cfg.get("path", "")) if data_cfg.get("path") else data_yaml.parent
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


def load_ground_truth(img: Path, w: int, h: int) -> Tuple[np.ndarray, np.ndarray]:
    """YOLO normalized xywh -> absolute xyxy. Returns (boxes Nx4, class_ids N)."""
    lbl = label_path_for(img)
    if not lbl.exists():
        return np.zeros((0, 4), np.float32), np.zeros((0,), np.int64)

    boxes, cls = [], []
    for line in lbl.read_text().splitlines():
        fields = line.split()
        if len(fields) < 5:
            continue
        cid = int(float(fields[0]))
        cx, cy, bw, bh = (float(v) for v in fields[1:5])
        boxes.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
        cls.append(cid)
    return np.array(boxes, np.float32).reshape(-1, 4), np.array(cls, np.int64)


# ---------------- metric ----------------


def iou_matrix(det: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """(N,4) x (M,4) xyxy -> (N,M) IoU."""
    if det.size == 0 or gt.size == 0:
        return np.zeros((det.shape[0], gt.shape[0]), np.float32)
    lt = np.maximum(det[:, None, :2], gt[None, :, :2])
    rb = np.minimum(det[:, None, 2:], gt[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_d = np.clip(det[:, 2] - det[:, 0], 0, None) * np.clip(det[:, 3] - det[:, 1], 0, None)
    area_g = np.clip(gt[:, 2] - gt[:, 0], 0, None) * np.clip(gt[:, 3] - gt[:, 1], 0, None)
    return inter / (area_d[:, None] + area_g[None, :] - inter + 1e-9)


def average_precision(tp: np.ndarray, conf: np.ndarray, n_gt: int) -> Tuple[float, float, float]:
    """101-point interpolated AP for one class at one IoU. Returns (ap, p, r)."""
    if n_gt == 0:
        return float("nan"), float("nan"), float("nan")
    if tp.size == 0:
        return 0.0, 0.0, 0.0

    order = conf.argsort()[::-1]
    tp = tp[order]
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(1 - tp)

    recall = tp_cum / n_gt
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)

    # monotone-decreasing precision envelope
    mpre = np.concatenate([[1.0], precision, [0.0]])
    mrec = np.concatenate([[0.0], recall, [1.0]])
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]

    ap = float(np.trapz(np.interp(np.linspace(0, 1, 101), mrec, mpre), np.linspace(0, 1, 101)))
    return ap, float(precision[-1]), float(recall[-1])


def evaluate(
    records: List[dict], class_ids: List[int]
) -> Tuple[Dict[int, dict], Dict[str, float]]:
    """records: [{det_boxes, det_scores, det_classes, gt_boxes, gt_classes}, ...]"""
    per_class: Dict[int, dict] = {}

    for cid in class_ids:
        n_gt = sum(int((r["gt_classes"] == cid).sum()) for r in records)
        # tp flags per IoU threshold, accumulated across images
        tp_by_iou = [[] for _ in IOU_THRESHOLDS]
        confs: List[float] = []

        for r in records:
            dmask = r["det_classes"] == cid
            gmask = r["gt_classes"] == cid
            dboxes, dscores = r["det_boxes"][dmask], r["det_scores"][dmask]
            gboxes = r["gt_boxes"][gmask]

            order = dscores.argsort()[::-1]
            dboxes, dscores = dboxes[order], dscores[order]
            confs.extend(dscores.tolist())

            ious = iou_matrix(dboxes, gboxes)
            for t_idx, thr in enumerate(IOU_THRESHOLDS):
                matched = np.zeros(gboxes.shape[0], bool)
                flags = np.zeros(dboxes.shape[0], np.float32)
                for d in range(dboxes.shape[0]):
                    if gboxes.shape[0] == 0:
                        break
                    cand = np.where((ious[d] >= thr) & ~matched)[0]
                    if cand.size:
                        best = cand[ious[d, cand].argmax()]
                        matched[best] = True
                        flags[d] = 1.0
                tp_by_iou[t_idx].extend(flags.tolist())

        conf_arr = np.array(confs, np.float32)
        aps, p50, r50 = [], float("nan"), float("nan")
        for t_idx, thr in enumerate(IOU_THRESHOLDS):
            ap, p, r = average_precision(np.array(tp_by_iou[t_idx], np.float32), conf_arr, n_gt)
            aps.append(ap)
            if t_idx == 0:
                p50, r50 = p, r

        per_class[cid] = {
            "instances": n_gt,
            "predictions": int(conf_arr.size),
            "precision_50": p50,
            "recall_50": r50,
            "ap50": aps[0],
            "ap75": aps[5],
            "ap50_95": float(np.nanmean(aps)) if not np.all(np.isnan(aps)) else float("nan"),
        }

    present = [v for v in per_class.values() if v["instances"] > 0]
    overall = {
        "map50": float(np.mean([v["ap50"] for v in present])) if present else 0.0,
        "map75": float(np.mean([v["ap75"] for v in present])) if present else 0.0,
        "map50_95": float(np.mean([v["ap50_95"] for v in present])) if present else 0.0,
        "mean_precision_50": float(np.mean([v["precision_50"] for v in present])) if present else 0.0,
        "mean_recall_50": float(np.mean([v["recall_50"] for v in present])) if present else 0.0,
    }
    return per_class, overall


# ---------------- driver ----------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=ROOT / "configs/train.yaml")
    ap.add_argument("--backend", choices=["torch", "onnx"], default="onnx")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="evaluate only N images (0 = all)")
    ap.add_argument("--conf", type=float, default=0.001, help="low conf: mAP needs the tail")
    ap.add_argument("--iou", type=float, default=0.7, help="NMS IoU")
    ap.add_argument("--report", type=Path, default=ROOT / "reports/accuracy.json")
    ap.add_argument("--compare-to", type=Path, default=None, help="baseline accuracy.json to gate against")
    ap.add_argument("--max-map-drop", type=float, default=None)
    args = ap.parse_args()

    import cv2

    with args.config.open() as fh:
        cfg = yaml.safe_load(fh)

    data_yaml = Path(cfg["dataset"]["data_yaml"])
    if not data_yaml.is_absolute():
        data_yaml = ROOT / data_yaml
    if not data_yaml.exists():
        print(f"[fail] data yaml not found: {data_yaml}", file=sys.stderr)
        return 2
    with data_yaml.open() as fh:
        data_cfg = yaml.safe_load(fh)

    # ultralytics uses 'val' as the key for the validation split
    split_key = "val" if args.split == "val" else args.split
    images_dir = resolve_split(data_yaml, data_cfg, split_key)
    if images_dir is None or not images_dir.exists():
        print(f"[fail] split '{args.split}' not found (looked at {images_dir})", file=sys.stderr)
        return 2

    image_paths = [p for p in sorted(images_dir.rglob("*")) if p.suffix.lower() in IMAGE_EXTS]
    if args.limit:
        image_paths = image_paths[: args.limit]
    if not image_paths:
        print(f"[fail] no images under {images_dir}", file=sys.stderr)
        return 2

    default_weights = {"torch": "models/best.pt", "onnx": cfg["export"]["onnx_path"]}[args.backend]
    weights = Path(args.weights or ROOT / default_weights)
    if not weights.exists():
        print(f"[fail] weights not found: {weights}", file=sys.stderr)
        return 2

    backend = build_backend(
        kind=args.backend,
        weights=str(weights),
        imgsz=cfg.get("train", {}).get("imgsz", 640),
        conf=args.conf,
        iou=args.iou,
        max_det=cfg.get("serve", {}).get("max_det", 300),
        device=args.device,
    )
    names = backend.class_names or {
        int(k): str(v) for k, v in (cfg["dataset"].get("expected_classes") or {}).items()
    }
    class_ids = sorted(names)

    print(f"[info] backend={backend.name} weights={weights.name} images={len(image_paths):,}")
    backend.warmup(2)

    records: List[dict] = []
    t0 = time.perf_counter()
    for start in range(0, len(image_paths), args.batch):
        chunk = image_paths[start : start + args.batch]
        imgs, valid = [], []
        for p in chunk:
            im = cv2.imread(str(p))
            if im is not None:
                imgs.append(im)
                valid.append(p)
        if not imgs:
            continue

        batched: List[List[Detection]] = backend.predict(imgs)
        for path, img, dets in zip(valid, imgs, batched):
            h, w = img.shape[:2]
            gt_boxes, gt_classes = load_ground_truth(path, w, h)
            records.append(
                {
                    "det_boxes": np.array([[d.x1, d.y1, d.x2, d.y2] for d in dets], np.float32).reshape(-1, 4),
                    "det_scores": np.array([d.score for d in dets], np.float32),
                    "det_classes": np.array([d.class_id for d in dets], np.int64),
                    "gt_boxes": gt_boxes,
                    "gt_classes": gt_classes,
                }
            )
        done = start + len(chunk)
        print(f"\r       {done}/{len(image_paths)} images", end="", flush=True)
    elapsed = time.perf_counter() - t0
    print(f"\n[ok  ] inference done in {elapsed:.1f}s ({len(records)/elapsed:.1f} img/s)")

    per_class, overall = evaluate(records, class_ids)

    header = f"{'class':<14}{'instances':>11}{'preds':>9}{'P@50':>8}{'R@50':>8}{'AP50':>8}{'AP75':>8}{'AP50-95':>9}"
    print("\n" + header)
    print("-" * len(header))
    for cid in class_ids:
        m = per_class[cid]
        print(
            f"{names.get(cid, str(cid)):<14}{m['instances']:>11,}{m['predictions']:>9,}"
            f"{m['precision_50']:>8.3f}{m['recall_50']:>8.3f}"
            f"{m['ap50']:>8.3f}{m['ap75']:>8.3f}{m['ap50_95']:>9.3f}"
        )
    print("-" * len(header))
    print(
        f"{'ALL':<14}{sum(m['instances'] for m in per_class.values()):>11,}"
        f"{sum(m['predictions'] for m in per_class.values()):>9,}"
        f"{overall['mean_precision_50']:>8.3f}{overall['mean_recall_50']:>8.3f}"
        f"{overall['map50']:>8.3f}{overall['map75']:>8.3f}{overall['map50_95']:>9.3f}"
    )

    payload = {
        "backend": backend.name,
        "weights": str(weights),
        "split": args.split,
        "num_images": len(records),
        "conf": args.conf,
        "nms_iou": args.iou,
        "eval_seconds": round(elapsed, 2),
        "overall": overall,
        "per_class": {names.get(cid, str(cid)): per_class[cid] for cid in class_ids},
    }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    # Keep one file per backend as well as the canonical accuracy.json.
    args.report.write_text(json.dumps(payload, indent=2))
    tagged = args.report.with_name(f"accuracy.{backend.name}.json")
    tagged.write_text(json.dumps(payload, indent=2))
    print(f"\n[ok  ] wrote {args.report.relative_to(ROOT)} and {tagged.name}")

    if args.compare_to:
        baseline_path = args.compare_to if args.compare_to.is_absolute() else ROOT / args.compare_to
        if not baseline_path.exists():
            print(f"[warn] baseline {baseline_path} missing; skipping accuracy gate")
            return 0
        baseline = json.loads(baseline_path.read_text())
        limit = (
            args.max_map_drop
            if args.max_map_drop is not None
            else cfg.get("quantize", {}).get("max_map_drop", 0.02)
        )
        drop = baseline["overall"]["map50_95"] - overall["map50_95"]
        print(
            f"\n[gate] baseline({baseline['backend']}) mAP50-95={baseline['overall']['map50_95']:.4f} "
            f"-> {backend.name} {overall['map50_95']:.4f}  drop={drop:+.4f} (limit {limit})"
        )
        if drop > limit:
            print("[FAIL] accuracy gate FAILED", file=sys.stderr)
            return 1
        print("[PASS] accuracy gate passed")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
