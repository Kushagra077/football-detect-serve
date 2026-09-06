"""app/batching.py's BatchedPredictor, against a FakeBackend - no real model."""
from __future__ import annotations

import asyncio

import pytest

from app.batching import BatchedPredictor, QueueOverflow
from tests.conftest import FakeBackend, tiny_image


async def test_concurrent_requests_coalesce_into_one_batch():
    backend = FakeBackend(weights="fake")
    predictor = BatchedPredictor(backend, max_batch_size=8, max_wait_ms=50, max_queue_size=10)
    await predictor.start()
    try:
        img = tiny_image()
        results = await asyncio.gather(*(predictor.predict(img) for _ in range(4)))
        batch_sizes = [batch_size for _, batch_size, _ in results]
        assert batch_sizes == [4, 4, 4, 4]
        assert backend.calls == [4]  # one predict() call handled all four
    finally:
        await predictor.stop()


async def test_batch_size_capped_at_max_batch_size():
    backend = FakeBackend(weights="fake")
    predictor = BatchedPredictor(backend, max_batch_size=2, max_wait_ms=50, max_queue_size=10)
    await predictor.start()
    try:
        img = tiny_image()
        results = await asyncio.gather(*(predictor.predict(img) for _ in range(4)))
        batch_sizes = sorted(batch_size for _, batch_size, _ in results)
        # 4 requests, cap of 2 -> two batches of 2, never a batch of 4
        assert max(batch_sizes) <= 2
        assert sum(backend.calls) == 4
    finally:
        await predictor.stop()


async def test_requests_with_different_overrides_never_share_a_threshold():
    """Guards the shared-mutable-state bug: conf/iou/max_det used to be written
    onto the shared backend object, so two requests coalesced into one batch
    could end up inferred under whichever request's override landed last.
    Overrides now travel with the job and are passed as call arguments, so a
    batch containing different thresholds must split into separate calls -
    never blend them.
    """
    backend = FakeBackend(weights="fake")
    predictor = BatchedPredictor(backend, max_batch_size=8, max_wait_ms=50, max_queue_size=10)
    await predictor.start()
    try:
        img = tiny_image()
        await asyncio.gather(
            predictor.predict(img, conf=0.1),
            predictor.predict(img, conf=0.9),
        )
        # Two distinct forward passes, one per threshold - never merged.
        assert sorted(backend.calls) == [1, 1]
        assert sorted(backend.call_options) == [(0.1, None, None), (0.9, None, None)]
    finally:
        await predictor.stop()


async def test_requests_with_matching_overrides_still_coalesce():
    """The common case - everyone using the same (or no) override - must still
    become exactly one forward pass; correctness for the mixed case shouldn't
    cost throughput in the normal case.
    """
    backend = FakeBackend(weights="fake")
    predictor = BatchedPredictor(backend, max_batch_size=8, max_wait_ms=50, max_queue_size=10)
    await predictor.start()
    try:
        img = tiny_image()
        results = await asyncio.gather(*(predictor.predict(img, conf=0.4) for _ in range(4)))
        assert backend.calls == [4]  # still one predict() call for all four
        assert backend.call_options == [(0.4, None, None)]
        assert [batch_size for _, batch_size, _ in results] == [4, 4, 4, 4]
    finally:
        await predictor.stop()


async def test_batching_metrics_are_labeled_per_backend():
    """Guards against the bug where QUEUE_WAIT/BATCH_SIZE/QUEUE_DEPTH had no
    labels at all: with 3 backends each running their own BatchedPredictor,
    an unlabeled metric silently blends all of them into one meaningless
    series. Two predictors with distinct backend names must produce distinct
    label values, not one shared bucket.
    """
    from app import metrics as metrics_module

    backend_a = FakeBackend(weights="fake", name="test-backend-a")
    backend_b = FakeBackend(weights="fake", name="test-backend-b")
    predictor_a = BatchedPredictor(backend_a, max_batch_size=8, max_wait_ms=20, max_queue_size=10)
    predictor_b = BatchedPredictor(backend_b, max_batch_size=8, max_wait_ms=20, max_queue_size=10)
    await predictor_a.start()
    await predictor_b.start()
    try:
        img = tiny_image()
        await predictor_a.predict(img)
        await predictor_b.predict(img)

        batch_size_labels = {
            s.labels["backend"]
            for s in metrics_module.BATCH_SIZE.collect()[0].samples
            if s.name.endswith("_count")
        }
        queue_wait_labels = {
            s.labels["backend"]
            for s in metrics_module.QUEUE_WAIT.collect()[0].samples
            if s.name.endswith("_count")
        }
        assert {"test-backend-a", "test-backend-b"} <= batch_size_labels
        assert {"test-backend-a", "test-backend-b"} <= queue_wait_labels
    finally:
        await predictor_a.stop()
        await predictor_b.stop()


async def test_queue_overflow_raises_when_full():
    backend = FakeBackend(weights="fake")
    # No .start() - nothing drains the queue, so it fills up deterministically.
    predictor = BatchedPredictor(backend, max_batch_size=8, max_wait_ms=1000, max_queue_size=1)
    img = tiny_image()

    first = asyncio.ensure_future(predictor.predict(img))
    await asyncio.sleep(0)  # let the first job's put_nowait execute

    with pytest.raises(QueueOverflow):
        await predictor.predict(img)

    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
