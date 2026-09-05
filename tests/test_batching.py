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
