from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass

from engine import InferenceEngine
from schemas import GenerationOutput, GenerationRequest, ScheduleResult

logger = logging.getLogger("minibatch.scheduler")


class QueueFull(Exception):
    """Raised by submit when the pending queue is at capacity."""


@dataclass
class _Item:
    req: GenerationRequest
    future: asyncio.Future
    enqueue_t: float


@dataclass
class _Active:
    item: _Item
    admit_t: float


class Scheduler:
    """Continuous (iteration-level) batcher.

    Pumps an InferenceEngine: each loop iteration admits as many queued requests
    as the engine has free slots (FCFS), runs one decode step off the event loop
    in an executor, and resolves the future of every sequence that step evicts. A
    request's future resolves the moment its own sequence finishes, independent
    of its batchmates, so a short request no longer waits out a long neighbor.

    max_batch_size is the concurrency cap: the engine's active-sequence (KV slot)
    limit. There is no batch-close timeout; the loop steps whenever any sequence
    is active and refills freed slots from the queue between steps.
    """

    def __init__(
        self,
        engine: InferenceEngine,
        max_batch_size: int,
        max_queue_depth: int = 1024,
    ) -> None:
        if max_batch_size <= 0:
            raise ValueError(f"max_batch_size must be positive, got {max_batch_size}")
        if max_queue_depth < max_batch_size:
            raise ValueError(
                f"max_queue_depth ({max_queue_depth}) must be >= "
                f"max_batch_size ({max_batch_size})"
            )
        self._engine = engine
        # The concurrency cap lives on the engine as its KV slot limit; the
        # scheduler is the single knob that sets it.
        self._engine.capacity = max_batch_size
        self._max_queue_depth = max_queue_depth
        self._pending: list[_Item] = []
        self._active: dict[int, _Active] = {}
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
        loop = asyncio.get_running_loop()
        while True:
            if self._engine.num_active() == 0 and not self._pending:
                # No await between this check and clear(), so submit (which
                # appends then sets) can never have its wakeup lost here.
                self._event.clear()
                await self._event.wait()
                continue
            await self._admit(loop)
            if self._engine.num_active() == 0:
                # admit raised and aborted its newcomers; nothing to step.
                continue
            await self._step(loop)

    async def _admit(self, loop: asyncio.AbstractEventLoop) -> None:
        free = self._engine.capacity - self._engine.num_active()
        n = min(free, len(self._pending))
        if n <= 0:
            return
        newcomers = self._take(n)
        # admit_t is taken before prefill, so queue_wait_ms ends at admission and
        # generate_ms (admit_t -> finish) includes prefill plus decode.
        admit_t = time.perf_counter()
        reqs = [item.req for item in newcomers]
        try:
            seq_ids = await loop.run_in_executor(None, self._engine.admit, reqs)
        except Exception as exc:
            # admit is atomic: only these newcomers fail, the active set is intact.
            logger.exception("admit failed (n=%d)", len(newcomers))
            self._abort(newcomers, exc)
            return
        if len(seq_ids) != len(newcomers):
            # admit must return one seq_id per request. A mismatch is a broken
            # contract: the seq_id-to-request mapping is unusable and the engine
            # may hold orphan active sequences we can never resolve or evict (a
            # short return would otherwise hang the tail futures via zip). Fail
            # safe like a step failure: abort the newcomers and the active set,
            # then reset, so nothing hangs and no later step returns an unknown
            # seq_id. The pending queue is untouched and still served.
            logger.error(
                "admit returned %d seq_ids for %d requests", len(seq_ids), len(newcomers)
            )
            exc = RuntimeError(
                f"admit returned {len(seq_ids)} seq_ids for {len(newcomers)} requests"
            )
            self._abort(newcomers, exc)
            self._abort_active(exc)
            self._engine.reset()
            return
        for seq_id, item in zip(seq_ids, newcomers):
            self._active[seq_id] = _Active(item, admit_t)

    async def _step(self, loop: asyncio.AbstractEventLoop) -> None:
        try:
            results = await loop.run_in_executor(None, self._engine.step)
        except Exception as exc:
            # A step failure can corrupt the shared KV reassembly, so abort every
            # active sequence and reset the engine. The pending queue is untouched
            # and the loop keeps serving it.
            logger.exception("step failed (active=%d)", len(self._active))
            self._abort_active(exc)
            self._engine.reset()
            return
        finish_t = time.perf_counter()
        for res in results:
            # Resolve by seq_id, never request_id, so a duplicate request_id in
            # flight cannot cross-talk.
            active = self._active.pop(res.seq_id)
            queue_wait_ms = (active.admit_t - active.item.enqueue_t) * 1000.0
            generate_ms = (finish_t - active.admit_t) * 1000.0
            active.item.future.set_result(
                ScheduleResult(
                    output=GenerationOutput(
                        request_id=active.item.req.request_id, text=res.text
                    ),
                    queue_wait_ms=queue_wait_ms,
                    generate_ms=generate_ms,
                )
            )

    def _take(self, n: int) -> list[_Item]:
        batch = self._pending[:n]
        self._pending = self._pending[n:]
        return batch

    @staticmethod
    def _abort(items: list[_Item], exc: Exception) -> None:
        for item in items:
            if not item.future.done():
                item.future.set_exception(exc)

    def _abort_active(self, exc: Exception) -> None:
        for active in self._active.values():
            if not active.item.future.done():
                active.item.future.set_exception(exc)
        self._active = {}
