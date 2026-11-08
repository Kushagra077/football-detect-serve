#!/usr/bin/env python3
"""Tiled (SAHI-style) vs full-frame inference on a val subset, same AP code as scripts/eval_map.py.

Experiment record, not part of the pipeline: result in reports/tiling_experiment.val.json (README, v9-v13 notes).
Usage: python scripts/tile_eval.py --stride 10 --out reports/tiling_experiment.val.json

Configs scored on identical images:
  v7_full      v7 whole frame at 960 (baseline)
  v6_full      v6 whole frame at 1280
  v7_t960      v7 on 6 native-res 960x600 tiles (scale 1.0), class-wise NMS merge
  v7_t1280     v7 on 4 native-res 1280x720 tiles fed at 960 (scale 0.75), merge
  v7_full+ball960 / v7_full+ball1280   full-frame boxes, plus ball-only boxes from the tiles
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import cv2  # noqa: E402
import eval_map as em  # noqa: E402
from app.backends.base import build_backend  # noqa: E402

BALL = 0


def nms(boxes, scores, thr):
    if len(boxes) == 0:
        return np.zeros(0, np.int64)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        ious = em.iou_matrix(boxes[i:i + 1], boxes[order[1:]])[0]
        order = order[1:][ious < thr]
    return np.array(keep, np.int64)


def to_arrays(dets):
    b = np.array([[d.x1, d.y1, d.x2, d.y2] for d in dets], np.float32).reshape(-1, 4)
    s = np.array([d.score for d in dets], np.float32)
    c = np.array([d.class_id for d in dets], np.int64)
    return b, s, c


def merge(b, s, c, thr=0.5):
    keep = []
    for cid in np.unique(c):
        idx = np.where(c == cid)[0]
        keep.extend(idx[nms(b[idx], s[idx], thr)].tolist())
    keep = np.array(sorted(keep), np.int64)
    return b[keep], s[keep], c[keep]


def tile_origins(W, H, tw, th, cols, rows):
    xs = np.linspace(0, W - tw, cols).round().astype(int)
    ys = np.linspace(0, H - th, rows).round().astype(int)
    return [(int(x), int(y)) for y in ys for x in xs]


def tiled(backend, img, tw, th, cols, rows):
    H, W = img.shape[:2]
    org = tile_origins(W, H, tw, th, cols, rows)
    tiles = [img[y:y + th, x:x + tw] for x, y in org]
    out = backend.predict(tiles)
    bs, ss, cs = [], [], []
    for (x, y), dets in zip(org, out):
        b, s, c = to_arrays(dets)
        if len(b):
            b = b + np.array([x, y, x, y], np.float32)
        bs.append(b); ss.append(s); cs.append(c)
    return np.concatenate(bs), np.concatenate(ss), np.concatenate(cs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    paths = sorted((ROOT / "dataset/images/val").rglob("*.jpg"))[:: args.stride]
    if args.limit:
        paths = paths[: args.limit]
    print(f"[info] {len(paths)} val images", flush=True)

    mk = lambda w, sz: build_backend(kind="onnx", weights=str(ROOT / "models" / w), imgsz=sz,
                                     conf=0.001, iou=0.7, max_det=300, device="cpu")
    v7, v6 = mk("football_detection_v7.onnx", 960), mk("football_detection_v6.onnx", 1280)
    v7.warmup(2); v6.warmup(2)

    cfgs = ["v7_full", "v6_full", "v7_t960", "v7_t1280", "v7_full+ball960", "v7_full+ball1280"]
    recs = {k: [] for k in cfgs}
    secs = {k: 0.0 for k in cfgs}

    for n, p in enumerate(paths, 1):
        img = cv2.imread(str(p))
        H, W = img.shape[:2]
        gb, gc = em.load_ground_truth(p, W, H)

        def add(k, b, s, c):
            recs[k].append({"det_boxes": b, "det_scores": s, "det_classes": c, "gt_boxes": gb, "gt_classes": gc})

        t = time.perf_counter(); f7 = to_arrays(v7.predict([img])[0]); secs["v7_full"] += time.perf_counter() - t
        t = time.perf_counter(); f6 = to_arrays(v6.predict([img])[0]); secs["v6_full"] += time.perf_counter() - t
        add("v7_full", *f7); add("v6_full", *f6)

        t = time.perf_counter(); a = tiled(v7, img, 960, 600, 3, 2); secs["v7_t960"] += time.perf_counter() - t
        t = time.perf_counter(); b = tiled(v7, img, 1280, 720, 2, 2); secs["v7_t1280"] += time.perf_counter() - t
        add("v7_t960", *merge(*a)); add("v7_t1280", *merge(*b))

        for k, tl in (("v7_full+ball960", a), ("v7_full+ball1280", b)):
            m = tl[2] == BALL
            fb, fs, fc = f7
            nb = np.concatenate([fb, tl[0][m]]); ns = np.concatenate([fs, tl[1][m]]); nc = np.concatenate([fc, tl[2][m]])
            add(k, *merge(nb, ns, nc))
            secs[k] = secs["v7_full"] + secs["v7_t960" if "960" in k else "v7_t1280"]
        if n % 25 == 0:
            print(f"\r{n}/{len(paths)}", end="", flush=True)

    print("\n[info] scoring", flush=True)
    res = {}
    for k in cfgs:
        per, overall = em.evaluate(recs[k], [0, 1, 2, 3])
        res[k] = {"ms_per_img_inference": 1000 * secs[k] / len(paths),
                  "mean4": float(np.mean([per[c]["ap50_95"] for c in range(4)])),
                  "ap": {c: round(per[c]["ap50_95"], 4) for c in range(4)},
                  "ball_ap50": round(per[0]["ap50"], 4), "ball_recall50": round(per[0]["recall_50"], 4),
                  "ball_precision50": round(per[0]["precision_50"], 4), "ball_preds": per[0]["predictions"]}
        print(k, json.dumps(res[k]), flush=True)
    args.out.write_text(json.dumps({"n_images": len(paths), "results": res}, indent=1))


if __name__ == "__main__":
    main()
