"""scripts/prepare_data.py's label-validation logic, against fabricated label
files written to pytest's tmp_path - never real dataset content."""
from __future__ import annotations

from scripts.prepare_data import scan_split, verify_class_map


def test_verify_class_map_matches():
    assert verify_class_map({0: "ball", 1: "player"}, {0: "ball", 1: "player"}) == []


def test_verify_class_map_no_expected_is_a_noop():
    assert verify_class_map({0: "ball"}, {}) == []


def test_verify_class_map_id_set_differs():
    errors = verify_class_map({0: "ball"}, {0: "ball", 1: "player"})
    assert any("class ids differ" in e for e in errors)


def test_verify_class_map_name_mismatch_case_insensitive_ok():
    # names are compared case-insensitively, so this should NOT be an error
    assert verify_class_map({0: "Ball"}, {0: "ball"}) == []


def test_verify_class_map_name_mismatch_flagged():
    errors = verify_class_map({0: "Referee"}, {0: "player"})
    assert any("name mismatch" in e for e in errors)


def _write_pair(tmp_path, name: str, label_lines: str):
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir(exist_ok=True)
    labels_dir.mkdir(exist_ok=True)
    (images_dir / f"{name}.jpg").write_bytes(b"not a real image, content is irrelevant here")
    (labels_dir / f"{name}.txt").write_text(label_lines)
    return images_dir


def test_scan_split_counts_valid_box(tmp_path):
    images_dir = _write_pair(tmp_path, "a", "2 0.5 0.5 0.2 0.3\n")
    counts, stats, problems = scan_split(images_dir, num_classes=5)
    assert stats["images"] == 1
    assert stats["boxes"] == 1
    assert counts[2] == 1
    assert problems == []


def test_scan_split_flags_degenerate_box(tmp_path):
    images_dir = _write_pair(tmp_path, "b", "0 0.5 0.5 0.0 0.2\n")
    _, stats, problems = scan_split(images_dir, num_classes=5)
    assert stats["boxes"] == 1  # still counted, just flagged
    assert any("degenerate box" in p for p in problems)


def test_scan_split_flags_out_of_range_class_id(tmp_path):
    images_dir = _write_pair(tmp_path, "c", "9 0.5 0.5 0.1 0.1\n")
    _, stats, problems = scan_split(images_dir, num_classes=5)
    assert any("outside 0..4" in p for p in problems)
    assert stats["boxes"] == 0  # out-of-range rows are not counted


def test_scan_split_flags_unnormalized_coords(tmp_path):
    images_dir = _write_pair(tmp_path, "d", "0 1.5 0.5 0.1 0.1\n")
    _, _, problems = scan_split(images_dir, num_classes=5)
    assert any("not normalized" in p for p in problems)


def test_scan_split_empty_label_is_a_legitimate_background_image(tmp_path):
    images_dir = _write_pair(tmp_path, "e", "")
    counts, stats, problems = scan_split(images_dir, num_classes=5)
    assert stats["empty_labels"] == 1
    assert stats["boxes"] == 0
    assert problems == []


def test_scan_split_missing_label_file(tmp_path):
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    (images_dir / "f.jpg").write_bytes(b"x")
    _, stats, problems = scan_split(images_dir, num_classes=5)
    assert stats["missing_labels"] == 1
    assert any("missing label" in p for p in problems)
