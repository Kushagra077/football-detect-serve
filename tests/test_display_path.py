"""Log lines must not crash when --report / --figure / --models point outside the repo.

eval_map.py and benchmark.py (and later train, prepare_data, render_presets, crosscheck_map and
build_calibration_set) used to print `path.relative_to(ROOT)`, which raises ValueError for a path like
/tmp/report.json - *after* the report was written, and in eval_map.py before the --compare-to accuracy gate ran,
so the gate was silently skipped and the exit code was non-zero.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from scripts import (benchmark, build_calibration_set, crosscheck_map, eval_map, prepare_data, render_presets,
                     train)


ALL = [eval_map, benchmark, build_calibration_set, crosscheck_map, prepare_data, render_presets, train]


@pytest.mark.parametrize("mod", ALL)
def test_path_inside_repo_is_shown_relative(mod):
    p = mod.ROOT / "reports" / "accuracy.json"
    assert mod._display_path(p) == str(Path("reports") / "accuracy.json")


@pytest.mark.parametrize("mod", ALL)
def test_path_outside_repo_is_shown_absolute_not_an_error(mod, tmp_path):
    p = tmp_path / "report.json"
    assert mod._display_path(p) == str(p)
