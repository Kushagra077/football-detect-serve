#!/usr/bin/env python3
"""Charts for the speed/accuracy presets, drawn only from files in reports/.

    python scripts/plot_presets.py

Reads   reports/accuracy.torch.<v>.test.json   (test split, torch, one evaluator for every model)
        reports/latency.presets.json           (scripts/benchmark.py on the ONNX fp32 presets)
        reports/training_curves.json           (training-time val mAP copied out of the checkpoints)
Writes  reports/figures/presets_tradeoff.png, presets_per_class.png, training_curves.png

Nothing is typed in by hand: change a report, re-run, and the charts follow.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
FIG = REPORTS / "figures"
CLASSES = ["ball", "goalkeeper", "player", "referee"]  # `other` has no val instances: left out of every mean

# version -> (preset name, input size, colour). v8 is an unfinished checkpoint, drawn hollow.
PRESETS = {
    "v3": ("fast", 640, "#6b7a8f"),
    "v7": ("balanced", 960, "#1f9d8b"),
    "v6": ("accurate", 1280, "#e07b39"),
}
V8_COLOR = "#8a5cc2"
TEXT = "#2b2b2b"


def _load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def per_class_ap(version: str, split: str = "test") -> dict[str, float] | None:
    rep = _load(REPORTS / f"accuracy.torch.{version}.{split}.json")
    if rep is None:
        return None
    return {c: float(rep["per_class"][c]["ap50_95"]) for c in CLASSES}


def mean4(ap: dict[str, float]) -> float:
    return sum(ap.values()) / len(ap)


def latency_p50(version: str) -> float | None:
    """Batch-1 p50 of the ONNX fp32 file for this version, from reports/latency.presets.json."""
    rep = _load(REPORTS / "latency.presets.json")
    if rep is None:
        return None
    for r in rep["results"]:
        if r["batch_size"] == 1 and r["model"].endswith(f"football_detection_{version}.onnx"):
            return float(r["p50_ms"])
    return None


def style(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=TEXT)


def tradeoff() -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    pts = []
    for v, (name, size, color) in PRESETS.items():
        ap, ms = per_class_ap(v), latency_p50(v)
        if ap is None or ms is None:
            print(f"[warn] {v}: missing accuracy or latency report, skipped", file=sys.stderr)
            continue
        pts.append((ms, mean4(ap)))
        ax.scatter(ms, mean4(ap), s=150, color=color, zorder=3)
        offset = (12, 8) if v == "v3" else (10, -26)  # keep v3's label clear of the axis
        ax.annotate(f"{name}\n{v} @ {size}px", (ms, mean4(ap)), textcoords="offset points",
                    xytext=offset, fontsize=10, color=TEXT)
    if len(pts) > 1:
        pts.sort()
        ax.plot(*zip(*pts), color="#999999", linestyle="--", linewidth=1, zorder=2)

    ap8, ms7 = per_class_ap("v8"), latency_p50("v7")
    if ap8 is not None and ms7 is not None:  # same network and input size as v7, so v7's speed applies
        ax.scatter(ms7, mean4(ap8), s=150, facecolors="none", edgecolors=V8_COLOR, linewidths=2, zorder=3)
        ax.annotate("v8 @ 960px\nunfinished: epoch 8 of 16", (ms7, mean4(ap8)), textcoords="offset points",
                    xytext=(10, -34), fontsize=9, color=V8_COLOR)

    ax.set_xlim(left=5)
    ax.set_xlabel("latency per image, ms (ONNX fp32, CPU, batch 1, Apple M2) — lower is faster", color=TEXT)
    ax.set_ylabel("test mAP50-95, mean of 4 classes — higher is better", color=TEXT)
    ax.set_title("Speed vs accuracy: three presets", loc="left", fontsize=13, color=TEXT)
    style(ax)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "presets_tradeoff.png", dpi=150)
    plt.close(fig)


def per_class() -> None:
    models = [(v, *PRESETS[v][:2], PRESETS[v][2]) for v in PRESETS] + [("v8", "epoch 8", 960, V8_COLOR)]
    rows = [(v, label, size, color, per_class_ap(v)) for v, label, size, color in models]
    rows = [r for r in rows if r[4] is not None]
    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.8 / max(1, len(rows))
    for i, (v, label, size, color, ap) in enumerate(rows):
        xs = [k + (i - (len(rows) - 1) / 2) * width for k in range(len(CLASSES))]
        ax.bar(xs, [ap[c] for c in CLASSES], width * 0.92, color=color, alpha=0.55 if v == "v8" else 1.0,
               label=f"{v} ({label}, {size}px)")
    ax.set_xticks(range(len(CLASSES)), CLASSES)
    ax.set_ylabel("test AP50-95", color=TEXT)
    ax.set_title("Per-class accuracy: the ball is the weak class in every model", loc="left", fontsize=13,
                 color=TEXT)
    ax.legend(frameon=False, fontsize=9)
    style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "presets_per_class.png", dpi=150)
    plt.close(fig)


def curves() -> None:
    data = _load(REPORTS / "training_curves.json")
    if data is None:
        print("[warn] reports/training_curves.json missing, skipped", file=sys.stderr)
        return
    colors = {**{v: PRESETS[v][2] for v in PRESETS}, "v8": V8_COLOR}
    fig, ax = plt.subplots(figsize=(8, 5))
    for v, d in data.items():
        ys = d["val_map50_95"]
        label = f"{v} @ {d['imgsz']}px" + (f" (stopped at {d['epochs_done']}/{d['planned_epochs']})"
                                           if d["epochs_done"] < d["planned_epochs"] else "")
        ax.plot(range(1, len(ys) + 1), ys, color=colors.get(v, "#888"), linewidth=2, label=label)
    ax.set_xlabel("epoch", color=TEXT)
    ax.set_ylabel("validation mAP50-95 during training (5 classes)", color=TEXT)
    ax.set_title("Training curves: accuracy flattens after about 15 epochs", loc="left", fontsize=13, color=TEXT)
    ax.legend(frameon=False, fontsize=9)
    style(ax)
    fig.text(0.01, 0.005, "Monitoring metric only; it counts the empty 'other' class, so it is not comparable "
             "to the test numbers elsewhere.", fontsize=7.5, color="#777")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(FIG / "training_curves.png", dpi=150)
    plt.close(fig)


def main() -> int:
    tradeoff()
    per_class()
    curves()
    print("[ok  ] wrote", ", ".join(sorted(p.name for p in FIG.glob("presets_*.png"))), "and training_curves.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
