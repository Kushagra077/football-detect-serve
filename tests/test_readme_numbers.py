"""Every number in the README's result tables must match the report file it comes from.

The README quotes accuracy and latency from reports/*.json. A hand-typed table drifts the moment a report is
regenerated (a v3 val score of 0.4695 was once printed as 0.470), so these tests parse the tables and compare
them with the JSON, rounded the way the README rounds. They read only tracked files, so they run in CI.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text().splitlines()
CLASSES = ["ball", "goalkeeper", "player", "referee"]  # `other` is left out of every mean


def _report(name: str) -> dict:
    return json.loads((ROOT / "reports" / name).read_text())


def _mean4(rep: dict) -> float:
    return sum(rep["per_class"][c]["ap50_95"] for c in CLASSES) / 4


def _section(start: str, stop: str) -> list[str]:
    """README lines from the heading that starts with `start` up to the one that starts with `stop`."""
    i = next(n for n, line in enumerate(README) if line.startswith(start))
    j = next(n for n, line in enumerate(README) if n > i and line.startswith(stop))
    return README[i:j]


def _rows(lines: list[str]) -> list[list[str]]:
    rows = []
    for line in lines:
        if line.startswith("|") and not line.startswith("|---") and not line.startswith("| model") \
                and not line.startswith("| preset") and not line.startswith("| backend"):
            rows.append([c.strip().replace("*", "") for c in line.strip("|").split("|")])
    return rows


def _nums(cell: str) -> list[float]:
    return [float(x) for x in re.findall(r"-?\d+\.\d+|-?\d+", cell.replace(",", ""))]


def _version(label: str) -> str:
    return re.search(r"\bv(\d+)\b", label).group(1)


def _close(readme_value: float, true_value: float, places: int = 3) -> bool:
    return abs(round(true_value, places) - readme_value) < 10 ** -places * 0.6


def test_test_split_table_matches_reports():
    rows = _rows(_section("#### test split", "#### val split"))
    assert len(rows) >= 10
    for row in rows:
        rep = _report(f"accuracy.torch.v{_version(row[0])}.test.json")
        expected = [_mean4(rep)] + [rep["per_class"][c]["ap50_95"] for c in CLASSES] + [rep["per_class"]["other"]["ap50_95"]]
        got = [_nums(c)[0] for c in row[1:7]]
        assert all(_close(g, e) for g, e in zip(got, expected)), (row[0], got, [round(e, 3) for e in expected])


def test_val_split_table_matches_reports():
    rows = _rows(_section("#### val split", "#### val vs test"))
    assert len(rows) >= 15
    for row in rows:
        label, backend = row[0], row[1]
        if "COCO" in label:
            rep = _report("accuracy.baseline.json")
        elif backend == "torch":
            rep = _report(f"accuracy.torch.v{_version(label)}.json")
        else:
            rep = _report(f"accuracy.onnx.v{_version(label)}.{backend.split('-')[1]}.json")
        expected = [_mean4(rep)] + [rep["per_class"][c]["ap50_95"] for c in CLASSES]
        got = [_nums(c)[0] for c in row[2:7]]
        assert all(_close(g, e) for g, e in zip(got, expected)), (label, backend, got, [round(e, 3) for e in expected])


PRESETS = {"fast": "v3", "balanced": "v7", "accurate": "v6", "max": "v13"}


def _latency(rep: dict, version: str, batch: int) -> dict:
    return next(r for r in rep["results"] if r["model"].endswith(f"football_detection_{version}.onnx")
                and r["batch_size"] == batch)


def test_preset_table_matches_reports():
    rows = _rows(_section("| preset | model | input size", "mAP is the 4-class mean"))
    # the section starts at the header row, which _rows skips
    assert [r[0] for r in rows] == list(PRESETS)
    m2 = _report("latency.presets.json")
    for row in rows:
        v = PRESETS[row[0]]
        test, val = _report(f"accuracy.torch.{v}.test.json"), _report(f"accuracy.torch.{v}.json")
        assert _close(_nums(row[3])[0], _mean4(test)), (row[0], "test mAP")
        assert _close(_nums(row[4])[0], _mean4(val)), (row[0], "val mAP")
        assert _close(_nums(row[5])[0], test["per_class"]["ball"]["ap50_95"]), (row[0], "ball AP")
        p50, p95 = _nums(row[6])
        b1 = _latency(m2, v, 1)
        assert _close(p50, b1["p50_ms"], 1) and _close(p95, b1["p95_ms"], 1), (row[0], "latency")
        assert round(_latency(m2, v, 8)["throughput_img_s"]) == _nums(row[7])[0], (row[0], "batch-8 img/s")


def test_x86_presets_table_matches_the_ci_report():
    rows = _rows(_section("### Latency of the four presets", "The frames are synthetic"))
    assert [r[0] for r in rows] == list(PRESETS)
    ci = _report("latency.x86.presets.json")
    for row in rows:
        b1 = _latency(ci, PRESETS[row[0]], 1)
        got = [_nums(c)[0] for c in row[3:7]]
        want = [b1["p50_ms"], b1["p95_ms"], b1["p99_ms"], b1["throughput_img_s"]]
        assert all(_close(g, w, 1) for g, w in zip(got, want)), (row[0], got, [round(w, 1) for w in want])


@pytest.mark.parametrize("report,heading,stop", [
    ("latency.json", "### Latency (single-request, CPU, Apple M2", "(Latency tracks the architecture"),
    ("latency.x86.json", "### Latency (single-request, CPU, x86_64", "**The hypothesis was wrong.**"),
])
def test_v3_backend_latency_tables_match_reports(report, heading, stop):
    rows = _rows(_section(heading, stop))
    assert [r[0] for r in rows] == ["torch", "onnx-fp32", "onnx-int8"]
    rep = _report(report)
    for row in rows:
        b1 = next(r for r in rep["results"] if r["backend"] == row[0] and r["batch_size"] == 1)
        got = [_nums(c)[0] for c in row[1:5]]
        want = [b1["p50_ms"], b1["p95_ms"], b1["p99_ms"], b1["throughput_img_s"]]
        assert all(_close(g, w, 1) for g, w in zip(got, want)), (report, row[0], got, [round(w, 1) for w in want])


def test_load_test_table_matches_the_locust_reports():
    import sys

    sys.path.insert(0, str(ROOT))
    from scripts.plot_loadtest import load_series

    series = load_series(ROOT / "reports" / "loadtest")
    rows = [r for r in _rows(_section("| concurrency | batching", "These six runs")) if r[0].isdigit()]
    assert len(rows) == 6
    for row in rows:
        stats = series["on" if row[1] == "on" else "off"][int(row[0])]
        want = [stats["total_rps"], stats["median_response_time"], stats["response_time_percentile_0.95"],
                stats["response_time_percentile_0.99"]]
        got = [_nums(c)[0] for c in row[2:6]]
        assert _close(got[0], want[0], 1) and all(_close(g, w, 0) for g, w in zip(got[1:], want[1:])), (row, want)


def test_dataset_counts_quoted_in_the_readme_match_dataset_json():
    ds = _report("dataset.json")
    text = "\n".join(README)
    assert f"{ds['split_stats']['test']['images']:,} images" in text
    assert f"{ds['split_stats']['val']['images']:,} images" in text
    assert f"{ds['instances_per_class']['test']['4']:,} in test" in text


def test_docker_numbers_quoted_in_the_readme_match_the_report():
    rep = _report("docker.presets.json")
    text = " ".join(" ".join(README).split())  # a number list may wrap across README lines
    for mode in ("batching_on", "batching_off"):
        p50 = [round(rep["results"][p][mode]["curl_ms_p50"]) for p in PRESETS]
        assert " / ".join(str(v) for v in p50) + " ms" in text, (mode, p50)
    server = [round(rep["results"][p]["batching_off"]["server_inference_ms_p50"]) for p in PRESETS]
    assert " / ".join(str(v) for v in server) + " ms server-side" in text, server
    gib = float(re.match(r"([\d.]+)GiB", rep["container_memory_after_all_requests"]).group(1))
    assert f"{gib:.2f} GiB of RAM" in text and f"after {rep['requests_sent_before_memory_reading']} requests" in text
    assert f"{rep['image_size_gb']} GB on disk" in text


def test_v13_last_checkpoint_figures_quoted_in_the_readme_match_the_reports():
    """The README gives v13's last-epoch checkpoint next to the val-selected one: test 0.490, val 0.549, ball 0.203,
    goalkeeper 0.558."""
    test, val = _report("accuracy.torch.v13_last.test.json"), _report("accuracy.torch.v13_last.json")
    text = " ".join(" ".join(README).split())
    assert f"scores higher on test, {_mean4(test):.3f} (ball {test['per_class']['ball']['ap50_95']:.3f}, " \
           f"goalkeeper {test['per_class']['goalkeeper']['ap50_95']:.3f})" in text
    assert f"(0.549 against 0.550)" in text and round(_mean4(val), 3) == 0.549
