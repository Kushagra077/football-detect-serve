"""scripts/build_calibration_set.py's sequence grouping and stratified sampling,
against fabricated (empty, zero-byte) files written to pytest's tmp_path - never
real dataset content.

Guards the actual bug: ultralytics sorts an image list alphabetically before
slicing --fraction, so a naive fraction of one sorted split becomes the first
N frames of whichever sequence sorts first (one clip, not a representative
sample). group_by_sequence + stratified_sample exist to hand ultralytics an
already-curated, already-complete (fraction=1.0) list instead.
"""
from __future__ import annotations

from pathlib import Path

from scripts.build_calibration_set import group_by_sequence, stratified_sample


def _touch_sequence(images_dir: Path, seq: str, n_frames: int) -> None:
    for i in range(1, n_frames + 1):
        (images_dir / f"{seq}_{i:06d}.jpg").touch()


def test_group_by_sequence_splits_by_prefix(tmp_path):
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    _touch_sequence(images_dir, "SNMOT-060", 5)
    _touch_sequence(images_dir, "SNMOT-061", 3)

    groups = group_by_sequence(images_dir)

    assert set(groups) == {"SNMOT-060", "SNMOT-061"}
    assert len(groups["SNMOT-060"]) == 5
    assert len(groups["SNMOT-061"]) == 3


def test_group_by_sequence_skips_unmatched_filenames(tmp_path, capsys):
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    _touch_sequence(images_dir, "SNMOT-060", 2)
    (images_dir / "not_a_sequence_frame.jpg").touch()
    (images_dir / "README.txt").touch()

    groups = group_by_sequence(images_dir)

    assert set(groups) == {"SNMOT-060"}
    assert len(groups["SNMOT-060"]) == 2
    assert "skipped" in capsys.readouterr().out


def test_stratified_sample_spreads_across_every_sequence():
    """The actual regression this guards: a naive sorted-then-sliced fraction
    would take everything from the first sequence and none from the rest.
    Every sequence must contribute at least one image to a large-enough pick.
    """
    images_dir_groups = {
        f"SEQ-{i}": [Path(f"/fake/SEQ-{i}_{j:06d}.jpg") for j in range(100)] for i in range(10)
    }

    picked, quota = stratified_sample(images_dir_groups, n=50, seed=0)

    assert len(picked) == 50
    assert set(quota) == set(images_dir_groups)
    assert all(count > 0 for count in quota.values())
    assert all(count == 5 for count in quota.values())  # 50 / 10 sequences, evenly divisible


def test_stratified_sample_does_not_pick_consecutive_frames_only():
    """Regression for 'shuffled, not sequential': within one sequence, a
    sample of half its frames should not just be frames 1..N in order.
    """
    groups = {"SEQ-1": [Path(f"/fake/SEQ-1_{j:06d}.jpg") for j in range(1, 101)]}

    picked, _ = stratified_sample(groups, n=20, seed=0)
    picked_frame_numbers = sorted(int(p.stem.split("_")[-1]) for p in picked)

    assert len(picked) == 20
    # If sampling were sequential (old ultralytics behavior), this would be
    # [1..20]. A random sample of 20 of 100 landing on exactly that range by
    # chance is effectively impossible.
    assert picked_frame_numbers != list(range(1, 21))


def test_stratified_sample_redistributes_when_a_sequence_is_short():
    """A sequence with fewer frames than its even share contributes everything
    it has; the shortfall goes to the other sequences instead of shrinking
    the total below what's actually available.
    """
    groups = {
        "SHORT": [Path(f"/fake/SHORT_{j:06d}.jpg") for j in range(3)],
        "LONG-A": [Path(f"/fake/LONG-A_{j:06d}.jpg") for j in range(100)],
        "LONG-B": [Path(f"/fake/LONG-B_{j:06d}.jpg") for j in range(100)],
    }

    picked, quota = stratified_sample(groups, n=30, seed=0)

    assert quota["SHORT"] == 3  # capped at what's available
    assert len(picked) == 30  # shortfall (10 - 3 = 7) redistributed to LONG-A/LONG-B
    assert quota["LONG-A"] + quota["LONG-B"] == 27


def test_stratified_sample_is_deterministic_given_a_seed():
    groups = {f"SEQ-{i}": [Path(f"/fake/SEQ-{i}_{j:06d}.jpg") for j in range(20)] for i in range(5)}

    picked_a, _ = stratified_sample(groups, n=25, seed=7)
    picked_b, _ = stratified_sample(groups, n=25, seed=7)

    assert picked_a == picked_b
