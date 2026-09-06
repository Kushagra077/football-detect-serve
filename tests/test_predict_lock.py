"""backend.predict_lock actually serializes concurrent predict() calls.

Guards against the ?batch=false thread-safety bug: two threads calling
backend.predict() at once on a shared instance (e.g. ultralytics YOLO, which
keeps mutable per-call state) can corrupt each other's results silently - no
exception, just wrong detections. app/main.py._locked_predict and
app/batching.py._infer_batch both hold backend.predict_lock around their call
to predict(); this test proves that lock actually prevents overlap, and that
the test harness itself is capable of detecting overlap (so the first
assertion isn't just timing luck).
"""
from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

from app.backends.base import DetectorBackend
from app.main import _locked_predict


class _SlowBackend(DetectorBackend):
    """Tracks how many predict() calls were ever in flight at the same time."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.active = 0
        self.max_concurrent = 0
        self._state_lock = threading.Lock()

    def load(self) -> "_SlowBackend":
        return self

    @property
    def name(self) -> str:
        return "slow-fake"

    def predict(self, images, *, conf=None, iou=None, max_det=None):
        with self._state_lock:
            self.active += 1
            self.max_concurrent = max(self.max_concurrent, self.active)
        time.sleep(0.02)  # wide enough that overlapping calls reliably collide
        with self._state_lock:
            self.active -= 1
        return [[] for _ in images]


def test_locked_predict_serializes_concurrent_calls():
    backend = _SlowBackend(weights="fake").load()

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(_locked_predict, backend, [None]) for _ in range(8)]
        for f in futures:
            f.result()

    assert backend.max_concurrent == 1


def test_harness_detects_overlap_when_unlocked():
    """Sanity check: the same 8 threads calling predict() directly (no lock) DO
    overlap, proving the assertion above is a real guarantee, not an artifact
    of the test being too fast to ever race.
    """
    backend = _SlowBackend(weights="fake").load()

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(backend.predict, [None]) for _ in range(8)]
        for f in futures:
            f.result()

    assert backend.max_concurrent > 1
