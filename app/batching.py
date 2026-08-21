"""Dynamic batching: asyncio queue + max-wait window.

A single worker task drains the queue, forming a batch as soon as either
`max_batch_size` items are available or `max_wait_ms` has elapsed since the
first item arrived. Inference runs in a thread so the event loop keeps accepting
requests (ONNX Runtime / torch release the GIL during the forward pass).
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

from app import metrics
from app.backends.base import Detection, DetectorBackend

log = logging.getLogger(__name__)


@dataclass
class _Job:
    image: np.ndarray
    future: "asyncio.Future[List[Detection]]"
    enqueued_at: float = field(default_factory=time.perf_counter)
    batch_size: int = 0
    infer_ms: float = 0.0


class BatchedPredictor:
    """Coalesces concurrent single-image requests into one forward pass."""

    def __init__(
        self,
        backend: DetectorBackend,
        max_batch_size: int = 8,
        max_wait_ms: float = 15.0,
        max_queue_size: int = 256,
    ) -> None:
        self.backend = backend
        self.max_batch_size = max(1, max_batch_size)
        self.max_wait_s = max(0.0, max_wait_ms / 1000.0)
        self.max_queue_size = max_queue_size

        self._queue: "asyncio.Queue[_Job]" = asyncio.Queue(maxsize=max_queue_size)
        self._worker: Optional[asyncio.Task] = None
        self._closing = False

    # ---------------- lifecycle ----------------

    async def start(self) -> None:
        if self._worker is None:
            self._closing = False
            self._worker = asyncio.create_task(self._run(), name="batch-worker")

    async def stop(self) -> None:
        self._closing = True
        if self._worker is not None:
            self._worker.cancel()
            try:
                await self._worker
            except asyncio.CancelledError:
                pass
            self._worker = None

    # ---------------- public API ----------------

    async def predict(self, image: np.ndarray) -> tuple[List[Detection], int, float]:
        """Submit one image. Returns (detections, batch_size, infer_ms)."""
        if self._closing:
            raise RuntimeError("predictor is shutting down")

        loop = asyncio.get_running_loop()
        job = _Job(image=image, future=loop.create_future())
        try:
            self._queue.put_nowait(job)
        except asyncio.QueueFull as exc:
            raise QueueOverflow("inference queue is full") from exc

        metrics.QUEUE_DEPTH.set(self._queue.qsize())
        detections = await job.future
        return detections, job.batch_size, job.infer_ms

    # ---------------- worker ----------------

    async def _collect_batch(self) -> List[_Job]:
        first = await self._queue.get()
        batch = [first]
        deadline = time.perf_counter() + self.max_wait_s

        while len(batch) < self.max_batch_size:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                break
            try:
                batch.append(await asyncio.wait_for(self._queue.get(), timeout=remaining))
            except asyncio.TimeoutError:
                break
        return batch

    async def _run(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            try:
                batch = await self._collect_batch()
            except asyncio.CancelledError:
                self._drain_cancelled()
                raise

            metrics.QUEUE_DEPTH.set(self._queue.qsize())
            metrics.BATCH_SIZE.observe(len(batch))
            now = time.perf_counter()
            for job in batch:
                metrics.QUEUE_WAIT.observe(now - job.enqueued_at)

            try:
                results, infer_ms = await loop.run_in_executor(
                    None, self._infer_batch, [j.image for j in batch]
                )
            except asyncio.CancelledError:
                for job in batch:
                    if not job.future.done():
                        job.future.cancel()
                raise
            except Exception as exc:  # noqa: BLE001 - one bad batch must not kill the worker
                log.exception("batch inference failed")
                for job in batch:
                    if not job.future.done():
                        job.future.set_exception(exc)
                continue

            for job, dets in zip(batch, results):
                job.batch_size = len(batch)
                job.infer_ms = infer_ms
                if not job.future.done():
                    job.future.set_result(dets)

    def _infer_batch(self, images: Sequence[np.ndarray]) -> tuple[List[List[Detection]], float]:
        """Runs in a worker thread. Times pre/infer separately for metrics."""
        t0 = time.perf_counter()
        tensor, metas = self.backend.preprocess(images)
        t1 = time.perf_counter()
        raw = self.backend.infer(tensor)
        t2 = time.perf_counter()
        results = self.backend.postprocess(raw, metas)

        metrics.PREPROCESS_LATENCY.observe(t1 - t0)
        metrics.INFERENCE_LATENCY.labels(backend=self.backend.name).observe(t2 - t1)
        for dets in results:
            for det in dets:
                metrics.DETECTIONS.labels(class_name=det.class_name).inc()
        return results, (t2 - t1) * 1000.0

    def _drain_cancelled(self) -> None:
        while not self._queue.empty():
            job = self._queue.get_nowait()
            if not job.future.done():
                job.future.cancel()


class QueueOverflow(RuntimeError):
    """Raised when the batch queue is saturated; maps to HTTP 503."""
