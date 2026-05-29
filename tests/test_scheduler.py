from __future__ import annotations

import asyncio
import time

import pytest

from model_runner import FakeRunner
from schemas import GenerationRequest
from scheduler import Scheduler


def _req(i: int) -> GenerationRequest:
    return GenerationRequest(
        request_id=f"r{i}", prompt=f"p{i}", max_new_tokens=8, temperature=0.0
    )


def test_flush_on_full_batch():
    # max_wait is huge, so a returned result can only mean a full batch closed.
    async def body():
        runner = FakeRunner()
        sched = Scheduler(runner, max_batch_size=4, max_wait_ms=10000)
        await sched.start()
        try:
            futs = [sched.submit(_req(i)) for i in range(4)]
            results = await asyncio.gather(*futs)
        finally:
            await sched.stop()
        assert runner.call_count == 1
        assert runner.batch_sizes == [4]
        assert sorted(r.output.request_id for r in results) == [f"r{i}" for i in range(4)]

    asyncio.run(body())


def test_flush_on_timeout():
    # Fewer than max_batch_size: the batch can only close on the timeout, and
    # it must actually wait that long, not flush early.
    max_wait_ms = 20

    async def body():
        runner = FakeRunner()
        sched = Scheduler(runner, max_batch_size=8, max_wait_ms=max_wait_ms)
        await sched.start()
        t0 = time.perf_counter()
        try:
            futs = [sched.submit(_req(i)) for i in range(3)]
            results = await asyncio.gather(*futs)
        finally:
            await sched.stop()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        assert runner.call_count == 1
        assert runner.batch_sizes == [3]
        assert elapsed_ms >= max_wait_ms * 0.9
        assert all(r.queue_wait_ms >= max_wait_ms * 0.9 for r in results)

    asyncio.run(body())


def test_no_loss_no_duplication():
    async def body():
        runner = FakeRunner()
        sched = Scheduler(runner, max_batch_size=4, max_wait_ms=20)
        await sched.start()
        n = 10
        try:
            futs = [sched.submit(_req(i)) for i in range(n)]
            results = await asyncio.gather(*futs)
        finally:
            await sched.stop()
        got = sorted(r.output.request_id for r in results)
        assert got == sorted(f"r{i}" for i in range(n))
        assert sum(runner.batch_sizes) == n

    asyncio.run(body())


def test_batch_is_fcfs():
    # The first closed batch must be the earliest max_batch_size requests, in
    # arrival order.
    async def body():
        runner = FakeRunner()
        sched = Scheduler(runner, max_batch_size=4, max_wait_ms=20)
        await sched.start()
        try:
            futs = [sched.submit(_req(i)) for i in range(6)]
            await asyncio.gather(*futs)
        finally:
            await sched.stop()
        assert runner.batches[0] == [f"r{i}" for i in range(4)]

    asyncio.run(body())


def test_fills_during_wait():
    # An underfull batch that fills before the timeout must close immediately on
    # reaching max_batch_size, not wait out max_wait_ms.
    max_wait_ms = 500

    async def body():
        runner = FakeRunner()
        sched = Scheduler(runner, max_batch_size=4, max_wait_ms=max_wait_ms)
        await sched.start()
        t0 = time.perf_counter()
        try:
            f0 = sched.submit(_req(0))
            await asyncio.sleep(0.02)
            assert runner.call_count == 0  # underfull and not yet timed out
            rest = [sched.submit(_req(i)) for i in range(1, 4)]
            await asyncio.gather(f0, *rest)
        finally:
            await sched.stop()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        assert runner.batch_sizes == [4]
        assert elapsed_ms < max_wait_ms * 0.5  # closed on fill, not on timeout

    asyncio.run(body())


def test_metrics_are_sane():
    async def body():
        runner = FakeRunner()
        sched = Scheduler(runner, max_batch_size=4, max_wait_ms=10)
        await sched.start()
        try:
            results = await asyncio.gather(*[sched.submit(_req(i)) for i in range(4)])
        finally:
            await sched.stop()
        for r in results:
            assert r.generate_ms > 0
            assert r.queue_wait_ms >= 0

    asyncio.run(body())


def test_rejects_invalid_config():
    runner = FakeRunner()
    with pytest.raises(ValueError):
        Scheduler(runner, max_batch_size=0, max_wait_ms=10)
    with pytest.raises(ValueError):
        Scheduler(runner, max_batch_size=4, max_wait_ms=-1)
