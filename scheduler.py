from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass

from model_runner import ModelRunner
from schemas import GenerationRequest, ScheduleResult

logger = logging.getLogger("minibatch.scheduler")


class QueueFull(Exception):
    """Raised by submit when the pending queue is at capacity."""


@dataclass
class _Item:
    req: GenerationRequest
    future: asyncio.Future
    enqueue_t: float


class Scheduler:
    """Size-or-timeout static batcher.

    A batch closes when the queue reaches max_batch_size, or when the oldest
    waiting request has waited max_wait_ms, whichever comes first. Pending
    requests live in a plain list and are only ever removed by _take, so the
    timeout wait (on an Event) can be cancelled without dropping a request.
    """

    def __init__(
        self,
        runner: ModelRunner,
        max_batch_size: int,
        max_wait_ms: float,
        max_queue_depth: int = 1024,
    ) -> None:
        if max_batch_size <= 0:
            raise ValueError(f"max_batch_size must be positive, got {max_batch_size}")
        if max_wait_ms < 0:
            raise ValueError(f"max_wait_ms must be non-negative, got {max_wait_ms}")
        if max_queue_depth < max_batch_size:
            raise ValueError(
                f"max_queue_depth ({max_queue_depth}) must be >= "
                f"max_batch_size ({max_batch_size})"
            )
        self._runner = runner
        self._max_batch_size = max_batch_size
        self._max_wait_s = max_wait_ms / 1000.0
        self._max_queue_depth = max_queue_depth
        self._pending: list[_Item] = []
        self._event = asyncio.Event()
        self._task: asyncio.Task | None = None

    def submit(self, req: GenerationRequest) -> asyncio.Future:
        if len(self._pending) >= self._max_queue_depth:
            raise QueueFull()
        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        self._pending.append(_Item(req, fut, time.perf_counter()))
        self._event.set()
        return fut

    async def start(self) -> None:
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass

    async def _run(self) -> None:
        while True:
            if not self._pending:
                self._event.clear()
                await self._event.wait()
                continue
            if len(self._pending) >= self._max_batch_size:
                await self._dispatch(self._take(self._max_batch_size), "full")
                continue
            wait = self._pending[0].enqueue_t + self._max_wait_s - time.perf_counter()
            if wait <= 0:
                await self._dispatch(self._take(len(self._pending)), "timeout")
                continue
            self._event.clear()
            try:
                await asyncio.wait_for(self._event.wait(), wait)
            except asyncio.TimeoutError:
                pass

    def _take(self, n: int) -> list[_Item]:
        batch = self._pending[:n]
        self._pending = self._pending[n:]
        return batch

    @staticmethod
    def _abort(batch: list[_Item], exc: Exception) -> None:
        for item in batch:
            item.future.set_exception(exc)

    async def _dispatch(self, batch: list[_Item], reason: str) -> None:
        loop = asyncio.get_running_loop()
        flush_t = time.perf_counter()
        reqs = [item.req for item in batch]
        try:
            outputs = await loop.run_in_executor(None, self._runner.run_batch, reqs)
        except Exception as exc:
            # One bad batch must not kill the loop or leave callers hanging.
            logger.exception("run_batch failed (size=%d)", len(batch))
            self._abort(batch, exc)
            return
        if len(outputs) != len(batch):
            # A wrong output count would otherwise drop the tail futures via zip
            # and hang those requests forever.
            logger.error(
                "runner returned %d outputs for %d requests", len(outputs), len(batch)
            )
            self._abort(
                batch,
                RuntimeError(
                    f"runner returned {len(outputs)} outputs for {len(batch)} requests"
                ),
            )
            return
        generate_ms = (time.perf_counter() - flush_t) * 1000.0
        logger.info(
            "batch closed size=%d reason=%s generate_ms=%.1f queue_depth=%d",
            len(batch),
            reason,
            generate_ms,
            len(self._pending),
        )
        # Outputs map to reqs by position (the ModelRunner contract), so a
        # duplicate request_id in the batch cannot cross-talk.
        for item, output in zip(batch, outputs):
            queue_wait_ms = (flush_t - item.enqueue_t) * 1000.0
            item.future.set_result(
                ScheduleResult(
                    output=output,
                    queue_wait_ms=queue_wait_ms,
                    generate_ms=generate_ms,
                )
            )
