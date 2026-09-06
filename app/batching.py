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
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from app import metrics
from app.backends.base import Detection, DetectorBackend

log = logging.getLogger(__name__)


@dataclass
class _Job:
    image: np.ndarray
    future: "asyncio.Future[List[Detection]]"
    # Per-request threshold overrides (None = backend default). These travel
    # with the job and are passed to predict() as call arguments - never
    # written onto the shared backend object, which was the bug: two
    # concurrent requests writing to backend.conf could stomp each other's
    # value, and a coalesced batch would run every image under whichever
    # request's override happened to land last.
    conf: Optional[float] = None
    iou: Optional[float] = None
    max_det: Optional[int] = None
    enqueued_at: float = field(default_factory=time.perf_counter)
    batch_size: int = 0
    infer_ms: float = 0.0


def _group_by_options(batch: Sequence[_Job]) -> List[List["_Job"]]:
    """Split a collected batch into runs of jobs sharing identical
    (conf, iou, max_det) - a single predict() call can only apply one
    threshold to the whole call, so jobs that disagree can't be inferred
    together. Jobs are grouped by key regardless of position (not just
    adjacent ones), so the common case - everyone using backend defaults -
    still becomes exactly one forward pass even if a single overridden
    request is queued in between them; a request with its own override
    always gets its own group, so it always gets exactly the threshold it
    asked for.
    """
    order: List[Tuple[Optional[float], Optional[float], Optional[int]]] = []
    groups: Dict[Tuple[Optional[float], Optional[float], Optional[int]], List[_Job]] = {}
    for job in batch:
        key = (job.conf, job.iou, job.max_det)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(job)
    return [groups[key] for key in order]


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

    async def predict(
        self,
        image: np.ndarray,
        *,
        conf: Optional[float] = None,
        iou: Optional[float] = None,
        max_det: Optional[int] = None,
    ) -> tuple[List[Detection], int, float]:
        """Submit one image. Returns (detections, batch_size, infer_ms).

        conf/iou/max_det are this request's own overrides (None = backend
        default); they travel with the job and never touch shared state.
        """
        if self._closing:
            raise RuntimeError("predictor is shutting down")

        loop = asyncio.get_running_loop()
        job = _Job(image=image, future=loop.create_future(), conf=conf, iou=iou, max_det=max_det)
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
            now = time.perf_counter()
            for job in batch:
                metrics.QUEUE_WAIT.observe(now - job.enqueued_at)

            try:
                # Same threshold -> one forward pass, same as before. Different
                # thresholds -> one forward pass per distinct (conf, iou,
                # max_det); see _group_by_options. Either way, every job gets
                # inferred with exactly the threshold it asked for.
                for group in _group_by_options(batch):
                    metrics.BATCH_SIZE.observe(len(group))
                    results, infer_ms = await loop.run_in_executor(None, self._infer_batch, group)
                    for job, dets in zip(group, results):
                        job.batch_size = len(group)
                        job.infer_ms = infer_ms
                        if not job.future.done():
                            job.future.set_result(dets)
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

    def _infer_batch(self, jobs: Sequence[_Job]) -> tuple[List[List[Detection]], float]:
        """Runs in a worker thread. All jobs in `jobs` share identical
        conf/iou/max_det (see _group_by_options), so one predict() call
        covers preprocess + inference + decode for the whole group.

        Holds backend.predict_lock so this never overlaps with an unbatched
        (?batch=false) request hitting the same backend instance concurrently.
        """
        images = [j.image for j in jobs]
        conf, iou, max_det = jobs[0].conf, jobs[0].iou, jobs[0].max_det
        t0 = time.perf_counter()
        with self.backend.predict_lock:
            results = self.backend.predict(images, conf=conf, iou=iou, max_det=max_det)
        infer_s = time.perf_counter() - t0

        metrics.INFERENCE_LATENCY.labels(backend=self.backend.name).observe(infer_s)
        for dets in results:
            for det in dets:
                metrics.DETECTIONS.labels(class_name=det.class_name).inc()
        return results, infer_s * 1000.0

    def _drain_cancelled(self) -> None:
        while not self._queue.empty():
            job = self._queue.get_nowait()
            if not job.future.done():
                job.future.cancel()


class QueueOverflow(RuntimeError):
    """Raised when the batch queue is saturated; maps to HTTP 503."""
