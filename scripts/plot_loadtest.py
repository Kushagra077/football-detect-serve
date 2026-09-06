"""Regenerate reports/figures/loadtest.svg from the Locust HTML reports.

The chart used to be hand-authored - someone typed pixel coordinates straight
into the SVG from a one-off look at the numbers, so it could (and did) drift
out of sync with reports/loadtest/*.html after a re-run. This script reads the
same six HTML reports the README's load-test table is built from and draws
the chart from them directly, so the figure can never say something the data
doesn't.

Usage:
    python scripts/plot_loadtest.py
    python scripts/plot_loadtest.py --loadtest-dir reports/loadtest --out reports/figures/loadtest.svg
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List

CONCURRENCIES = [1, 4, 16]
RUN_STEMS = {c: {"on": f"c{c}_batch", "off": f"c{c}_nobatch"} for c in CONCURRENCIES}

BLUE = "#2563eb"  # batching on
RED = "#dc2626"  # batching off


def _extract_aggregated(html_path: Path) -> dict:
    """Pull the 'Aggregated' stats row out of a Locust --html report.

    Locust embeds the run's stats as a plain JSON object inside a <script>
    block (no separate CSV needed) - find the object containing
    "name": "Aggregated" by brace-matching outward from that key.
    """
    html = html_path.read_text(encoding="utf-8")
    idx = html.find('"name": "Aggregated"')
    if idx == -1:
        raise ValueError(f"no Aggregated stats found in {html_path}")
    start = html.rfind("{", 0, idx)
    depth, i = 0, start
    while True:
        if html[i] == "{":
            depth += 1
        elif html[i] == "}":
            depth -= 1
            if depth == 0:
                break
        i += 1
    return json.loads(html[start : i + 1])


def load_series(loadtest_dir: Path) -> Dict[str, Dict[int, dict]]:
    """{'on': {1: stats, 4: stats, 16: stats}, 'off': {...}}"""
    series: Dict[str, Dict[int, dict]] = {"on": {}, "off": {}}
    for c in CONCURRENCIES:
        for mode, stem in RUN_STEMS[c].items():
            path = loadtest_dir / f"{stem}.html"
            if not path.exists():
                raise FileNotFoundError(f"missing {path} - run the 6 locust scenarios first (see README)")
            series[mode][c] = _extract_aggregated(path)
    return series


def _nice_step(raw_step: float) -> float:
    """Round a step up to the nearest 1/2/5/10 x 10^n - standard 'nice axis' rule."""
    if raw_step <= 0:
        return 1.0
    exponent = math.floor(math.log10(raw_step))
    fraction = raw_step / (10**exponent)
    nice_fraction = 1 if fraction <= 1 else 2 if fraction <= 2 else 5 if fraction <= 5 else 10
    return nice_fraction * (10**exponent)


def nice_ticks(data_max: float, target_ticks: int = 7) -> List[float]:
    """0..N axis ticks at a human-friendly step, always starting at 0."""
    raw_step = (data_max or 1) / (target_ticks - 1)
    step = _nice_step(raw_step)
    top = math.ceil(data_max / step) * step
    ticks, v = [], 0.0
    while v <= top + step * 1e-9:
        ticks.append(round(v, 6))
        v += step
    return ticks


def _panel(x0: float, title: str, on_vals: List[float], off_vals: List[float]) -> str:
    """One axes + two polylines, in the same column layout as the original chart."""
    top, bottom = 40.0, 330.0
    x_left, x_right = x0 + 60.0, x0 + 425.0
    ticks = nice_ticks(max(on_vals + off_vals))
    axis_max = ticks[-1]

    def y_of(v: float) -> float:
        return bottom - (v / axis_max) * (bottom - top)

    def x_of(i: int) -> float:
        return x_left + (x_right - x_left) * i / (len(CONCURRENCIES) - 1)

    parts = [
        f'<text x="{(x_left + x_right) / 2:.1f}" y="26" text-anchor="middle" '
        f'font-size="14" font-weight="600" fill="var(--fg,#1a1a1a)">{title}</text>',
        f'<line x1="{x_left}" y1="{top}" x2="{x_left}" y2="{bottom}" stroke="#888" stroke-width="1"/>',
        f'<line x1="{x_left}" y1="{bottom}" x2="{x_right}" y2="{bottom}" stroke="#888" stroke-width="1"/>',
    ]
    for t in ticks:
        y = y_of(t)
        parts.append(f'<line x1="{x_left}" y1="{y:.1f}" x2="{x_right}" y2="{y:.1f}" stroke="#e0e0e0" stroke-width="1"/>')
        parts.append(f'<text x="{x_left - 8}" y="{y + 4:.1f}" text-anchor="end" font-size="10" fill="#666">{t:g}</text>')
    for i, c in enumerate(CONCURRENCIES):
        parts.append(f'<text x="{x_of(i):.1f}" y="348" text-anchor="middle" font-size="11" fill="#666">{c}</text>')
    parts.append(
        f'<text x="{(x_left + x_right) / 2:.1f}" y="366" text-anchor="middle" '
        f'font-size="11" fill="#888">concurrency (users)</text>'
    )

    for vals, color in ((on_vals, BLUE), (off_vals, RED)):
        pts = " ".join(f"{x_of(i):.1f},{y_of(v):.1f}" for i, v in enumerate(vals))
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        for i, v in enumerate(vals):
            parts.append(f'<circle cx="{x_of(i):.1f}" cy="{y_of(v):.1f}" r="3.5" fill="{color}"/>')

    legend_x = x_left + 12
    parts.append(f'<circle cx="{legend_x:.1f}" cy="386" r="4" fill="{BLUE}"/>')
    parts.append(f'<text x="{legend_x + 10:.1f}" y="390" font-size="10" fill="#444">batching on</text>')
    parts.append(f'<circle cx="{legend_x:.1f}" cy="400" r="4" fill="{RED}"/>')
    parts.append(f'<text x="{legend_x + 10:.1f}" y="404" font-size="10" fill="#444">batching off</text>')
    return "\n".join(parts)


def render(series: Dict[str, Dict[int, dict]]) -> str:
    on_rps = [series["on"][c]["total_rps"] for c in CONCURRENCIES]
    off_rps = [series["off"][c]["total_rps"] for c in CONCURRENCIES]
    on_p95 = [series["on"][c]["response_time_percentile_0.95"] for c in CONCURRENCIES]
    off_p95 = [series["off"][c]["response_time_percentile_0.95"] for c in CONCURRENCIES]

    left = _panel(0.0, "Throughput (RPS)", on_rps, off_rps)
    right = _panel(405.0, "p95 latency (ms)", on_p95, off_p95)
    caption = (
        "Locust load test, local Docker (M2 CPU), 1 min per run, real frame - "
        "generated by scripts/plot_loadtest.py from reports/loadtest/*.html"
    )
    return (
        '<svg viewBox="0 0 860 380" xmlns="http://www.w3.org/2000/svg" '
        'font-family="Helvetica,Arial,sans-serif">\n'
        '<rect x="0" y="0" width="860" height="380" fill="#ffffff"/>\n'
        f"{left}\n{right}\n"
        f'<text x="430.0" y="374" text-anchor="middle" font-size="10" fill="#999">{caption}</text>\n'
        "</svg>\n"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--loadtest-dir", type=Path, default=Path("reports/loadtest"))
    ap.add_argument("--out", type=Path, default=Path("reports/figures/loadtest.svg"))
    args = ap.parse_args()

    series = load_series(args.loadtest_dir)
    svg = render(series)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(svg, encoding="utf-8")

    print(f"wrote {args.out}")
    for mode, label in (("on", "batching on "), ("off", "batching off")):
        row = ", ".join(
            f"c{c}={series[mode][c]['total_rps']:.1f} rps / {series[mode][c]['response_time_percentile_0.95']:.0f}ms p95"
            for c in CONCURRENCIES
        )
        print(f"  {label}: {row}")


if __name__ == "__main__":
    main()
