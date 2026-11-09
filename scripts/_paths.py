"""Shared path helper for the scripts' log lines."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def display_path(p: Path) -> str:
    """Repo-relative path for log lines; the absolute path if it is outside the repo (e.g. --report /tmp/x.json).

    `p.relative_to(ROOT)` raises ValueError for such a path, and these lines run after the output was already
    written (and, in eval_map.py, before the accuracy gate), so a crash there loses work or skips the gate.
    """
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)
