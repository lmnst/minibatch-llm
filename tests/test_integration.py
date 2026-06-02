from __future__ import annotations

import asyncio

import pytest

# torch and transformers are imported lazily inside the test body (guarded by
# importorskip), never at module top level, so this stays model-free at
# collection time.

from schemas import GenerationRequest

# (request_id, prompt, max_new_tokens): varied prompt lengths and budgets.
WORKLOAD = [
    ("a", "The capital of France is", 12),
    ("b", "import numpy as np", 6),
    ("c", "Once upon a time, there was", 16),
    ("d", "Hello there, my friend. How are", 8),
    ("e", "def add(a, b):\n    return", 10),
]


@pytest.mark.model
def test_scheduler_hfengine_integration():
    # The only test that drives the real HFEngine through the real async
    # Scheduler (submit -> queue -> run_in_executor -> resolve), the path the
    # synchronous parity tests and the FakeEngine scheduler tests each only cover
    # half of.
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from engine import HFEngine
    from model_runner import HFModelRunner
    from scheduler import Scheduler

    engine = HFEngine("distilgpt2")
    runner = HFModelRunner("distilgpt2")
    reqs = [
        GenerationRequest(request_id=rid, prompt=p, max_new_tokens=m, temperature=0.0)
        for rid, p, m in WORKLOAD
    ]
    # Trusted baseline: the P1 static path, itself proven token-for-token against
    # standalone greedy by the parity gate. All requests use default EOS.
    baseline = {o.request_id: o.text for o in runner.run_batch(reqs)}

    async def body():
        # capacity 3 with 5 requests, so the queue backs up and the loop exercises
        # admission, eviction, and refill across real executor steps.
        sched = Scheduler(engine, max_batch_size=3)
        await sched.start()
        try:
            futs = [sched.submit(r) for r in reqs]
            return await asyncio.gather(*futs)
        finally:
            await sched.stop()

    results = asyncio.run(body())
    by_id = {r.output.request_id: r for r in results}
    assert sorted(by_id) == sorted(r.request_id for r in reqs)  # nothing lost
    for r in reqs:
        res = by_id[r.request_id]
        # Continuous text matches the static baseline: no neighbour bled into it.
        assert res.output.text == baseline[r.request_id], r.request_id
        assert res.queue_wait_ms >= 0
        assert res.generate_ms > 0
