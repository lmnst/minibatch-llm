from __future__ import annotations

import asyncio

import pytest

from engine import EngineResult, FakeEngine
from schemas import GenerationRequest
from scheduler import QueueFull, Scheduler


def _req(rid: str, max_new: int = 4) -> GenerationRequest:
    return GenerationRequest(
        request_id=rid, prompt=f"p-{rid}", max_new_tokens=max_new, temperature=0.0
    )


class ProbeEngine(FakeEngine):
    """FakeEngine that records admission and eviction so the continuous-loop
    behaviour (capacity, FCFS, refill, staggered eviction) can be asserted."""

    def __init__(self) -> None:
        super().__init__()
        self.max_active = 0
        self.step_no = 0
        self.admit_batches: list[tuple[list[str], set[str]]] = []
        self.evict_step: dict[str, int] = {}
        self._seqid_req: dict[int, str] = {}

    def admit(self, reqs):
        before = {s.request.request_id for s in self._active.values()}
        seq_ids = super().admit(reqs)
        for sid, req in zip(seq_ids, reqs):
            self._seqid_req[sid] = req.request_id
        self.admit_batches.append(([r.request_id for r in reqs], before))
        self.max_active = max(self.max_active, len(self._active))
        return seq_ids

    def step(self):
        self.step_no += 1
        self.max_active = max(self.max_active, len(self._active))
        results = super().step()
        for res in results:
            self.evict_step[self._seqid_req[res.seq_id]] = self.step_no
        return results


def test_capacity_never_exceeded():
    # N well above capacity with mixed budgets: the active set fills to the cap
    # and never exceeds it.
    async def body():
        engine = ProbeEngine()
        sched = Scheduler(engine, max_batch_size=3)
        await sched.start()
        try:
            budgets = [2, 3, 4]
            futs = [sched.submit(_req(f"r{i}", budgets[i % 3])) for i in range(15)]
            await asyncio.gather(*futs)
        finally:
            await sched.stop()
        assert engine.capacity == 3
        assert engine.max_active == 3
        assert engine.max_active <= 3

    asyncio.run(body())


def test_fcfs_admission():
    # All submitted before the loop drains anything, so admission is strictly in
    # arrival order.
    async def body():
        engine = ProbeEngine()
        sched = Scheduler(engine, max_batch_size=2)
        await sched.start()
        try:
            futs = [sched.submit(_req(f"r{i}", max_new=2)) for i in range(6)]
            await asyncio.gather(*futs)
        finally:
            await sched.stop()
        admitted = [rid for batch, _ in engine.admit_batches for rid in batch]
        assert admitted == [f"r{i}" for i in range(6)]

    asyncio.run(body())


def test_eviction_frees_slot_admits_next():
    # One long sequence holds a slot while short ones cycle through the other:
    # every refill admit happens with the long sequence still active.
    async def body():
        engine = ProbeEngine()
        sched = Scheduler(engine, max_batch_size=2)
        await sched.start()
        try:
            futs = [
                sched.submit(_req("long", max_new=8)),
                sched.submit(_req("s0", max_new=2)),
                sched.submit(_req("s1", max_new=2)),
                sched.submit(_req("s2", max_new=2)),
            ]
            await asyncio.gather(*futs)
        finally:
            await sched.stop()
        refills = [(b, before) for b, before in engine.admit_batches if "long" not in b]
        assert refills, "expected refill admits after eviction"
        for batch, before in refills:
            assert "long" in before

    asyncio.run(body())


def test_staggered_max_new_evicts_in_budget_order():
    # Sequences leave at their own budgets, at different steps; max_new=1 drains
    # at the first step with no decode.
    async def body():
        engine = ProbeEngine()
        sched = Scheduler(engine, max_batch_size=4)
        await sched.start()
        try:
            futs = [
                sched.submit(_req("a", max_new=1)),
                sched.submit(_req("b", max_new=3)),
                sched.submit(_req("c", max_new=5)),
            ]
            await asyncio.gather(*futs)
        finally:
            await sched.stop()
        assert engine.evict_step["a"] < engine.evict_step["b"] < engine.evict_step["c"]

    asyncio.run(body())


def test_no_loss_no_duplication_under_churn():
    # N well above capacity with varied budgets: every request resolves exactly
    # once, none lost, none duplicated.
    async def body():
        engine = FakeEngine()
        sched = Scheduler(engine, max_batch_size=4)
        await sched.start()
        n = 30
        budgets = [1, 2, 3, 5]
        try:
            futs = [sched.submit(_req(f"r{i}", budgets[i % 4])) for i in range(n)]
            results = await asyncio.gather(*futs)
        finally:
            await sched.stop()
        got = sorted(r.output.request_id for r in results)
        assert got == sorted(f"r{i}" for i in range(n))
        assert len(results) == n

    asyncio.run(body())


def test_duplicate_request_id_no_crosstalk():
    # Two in-flight requests share a request_id but carry different prompts and
    # budgets. Resolving by seq_id must bind each future to its own sequence, so
    # each text comes back matching its own prompt (FakeEngine text embeds the
    # prompt) rather than collapsing or swapping.
    async def body():
        engine = FakeEngine()
        sched = Scheduler(engine, max_batch_size=4)
        await sched.start()
        try:
            fa = sched.submit(GenerationRequest("dup", "a", max_new_tokens=3, temperature=0.0))
            fb = sched.submit(GenerationRequest("dup", "b", max_new_tokens=7, temperature=0.0))
            ra, rb = await asyncio.gather(fa, fb)
        finally:
            await sched.stop()
        assert ra.output.request_id == "dup" and ra.output.text == "a|dup"
        assert rb.output.request_id == "dup" and rb.output.text == "b|dup"

    asyncio.run(body())


def test_metrics_are_sane():
    async def body():
        engine = FakeEngine()
        sched = Scheduler(engine, max_batch_size=4)
        await sched.start()
        try:
            results = await asyncio.gather(
                *[sched.submit(_req(f"r{i}", max_new=3)) for i in range(4)]
            )
        finally:
            await sched.stop()
        for r in results:
            assert r.generate_ms > 0
            assert r.queue_wait_ms >= 0

    asyncio.run(body())


def test_admit_exception_is_isolated():
    # An admit that raises aborts only its own newcomers (atomically) and the
    # loop keeps running, so a later batch is still served.
    class FlakyAdmitEngine(FakeEngine):
        def __init__(self) -> None:
            super().__init__()
            self.admit_calls = 0

        def admit(self, reqs):
            self.admit_calls += 1
            if any(r.request_id.startswith("bad") for r in reqs):
                raise RuntimeError("prefill boom")
            return super().admit(reqs)

    async def body():
        engine = FlakyAdmitEngine()
        sched = Scheduler(engine, max_batch_size=2)
        await sched.start()
        try:
            bad = [sched.submit(_req(f"bad{i}", max_new=3)) for i in range(2)]
            for f in bad:
                with pytest.raises(RuntimeError):
                    await f
            good = [sched.submit(_req(f"g{i}", max_new=3)) for i in range(2)]
            results = await asyncio.wait_for(asyncio.gather(*good), timeout=5)
            assert sorted(r.output.request_id for r in results) == ["g0", "g1"]
            assert engine.admit_calls == 2
        finally:
            await sched.stop()

    asyncio.run(body())


def test_step_exception_aborts_active_and_resets():
    # A step that raises aborts every active sequence and resets the engine; the
    # loop survives and serves later requests.
    class BoomStepEngine(FakeEngine):
        def __init__(self) -> None:
            super().__init__()
            self.boom = True
            self.reset_calls = 0

        def step(self):
            if self.boom:
                raise RuntimeError("step boom")
            return super().step()

        def reset(self):
            self.reset_calls += 1
            super().reset()

    async def body():
        engine = BoomStepEngine()
        sched = Scheduler(engine, max_batch_size=2)
        await sched.start()
        try:
            f1, f2 = sched.submit(_req("r1")), sched.submit(_req("r2"))
            with pytest.raises(RuntimeError):
                await asyncio.wait_for(asyncio.gather(f1, f2), timeout=5)
            assert engine.reset_calls >= 1
            engine.boom = False
            good = [sched.submit(_req(f"g{i}", max_new=3)) for i in range(2)]
            results = await asyncio.wait_for(asyncio.gather(*good), timeout=5)
            assert sorted(r.output.request_id for r in results) == ["g0", "g1"]
        finally:
            await sched.stop()

    asyncio.run(body())


def test_admit_count_mismatch_is_isolated():
    # An engine that returns the wrong number of seq_ids breaks the seq_id-to-
    # request mapping. The loop must fail those requests (not hang the tail) and
    # keep serving later ones, the continuous analogue of the P1 output-count
    # guard.
    class ShortAdmitOnceEngine(FakeEngine):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def admit(self, reqs):
            self.calls += 1
            seq_ids = super().admit(reqs)
            return seq_ids[:-1] if self.calls == 1 else seq_ids

    async def body():
        engine = ShortAdmitOnceEngine()
        sched = Scheduler(engine, max_batch_size=2)
        await sched.start()
        try:
            bad = [sched.submit(_req("b0", 3)), sched.submit(_req("b1", 3))]
            with pytest.raises(RuntimeError):
                await asyncio.wait_for(asyncio.gather(*bad), timeout=5)
            good = [sched.submit(_req(f"g{i}", 3)) for i in range(2)]
            results = await asyncio.wait_for(asyncio.gather(*good), timeout=5)
            assert sorted(r.output.request_id for r in results) == ["g0", "g1"]
        finally:
            await sched.stop()

    asyncio.run(body())


def test_queue_full_backpressure():
    # submit beyond max_queue_depth raises QueueFull. The loop is never started,
    # so nothing drains the queue.
    async def body():
        engine = FakeEngine()
        sched = Scheduler(engine, max_batch_size=2, max_queue_depth=3)
        sched.submit(_req("r0"))
        sched.submit(_req("r1"))
        sched.submit(_req("r2"))
        with pytest.raises(QueueFull):
            sched.submit(_req("r3"))

    asyncio.run(body())


def test_rejects_invalid_config():
    engine = FakeEngine()
    with pytest.raises(ValueError):
        Scheduler(engine, max_batch_size=0)
    with pytest.raises(ValueError):
        Scheduler(engine, max_batch_size=4, max_queue_depth=2)


def test_fake_engine_admit_is_atomic_on_overflow():
    # The engine contract the scheduler relies on: an admit that would exceed
    # capacity raises and leaves the active set untouched.
    engine = FakeEngine()
    engine.capacity = 2
    engine.admit([_req("a"), _req("b")])
    assert engine.num_active() == 2
    with pytest.raises(RuntimeError):
        engine.admit([_req("c")])
    assert engine.num_active() == 2


def test_fake_engine_drains_finished():
    # A max_new=1 sequence finishes at admit and is returned by the next step
    # with no further generation; one EngineResult per sequence.
    engine = FakeEngine()
    engine.capacity = 4
    engine.admit([_req("a", max_new=1), _req("b", max_new=3)])
    first = engine.step()
    assert isinstance(first[0], EngineResult)
    assert {r.text for r in first} == {"p-a|a"}  # only the max_new=1 sequence
    assert engine.num_active() == 1  # b still generating
    second = engine.step()  # b finishes on its third token
    assert {r.text for r in second} == {"p-b|b"}
    assert engine.num_active() == 0
