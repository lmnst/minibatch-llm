from __future__ import annotations

import pytest

# torch and transformers are imported lazily inside the fixtures and test
# bodies, never at module top level, so this file stays model-free at collection
# time (pytest imports every test module even under -m "not model", and a
# top-level ML import would break tests/test_server.py::test_no_ml_imports).

from schemas import GenerationRequest

PROMPTS = {
    "france": "The capital of France is",
    "village": "Once upon a time, in a small village near the mountains, there lived",
    "numpy": "import numpy as np",
    "hello": "Hello there, my friend. How are",
}


def _req(rid: str, max_new: int) -> GenerationRequest:
    return GenerationRequest(
        request_id=rid, prompt=PROMPTS[rid], max_new_tokens=max_new, temperature=0.0
    )


@pytest.fixture(scope="module")
def engine():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from engine import HFEngine

    eng = HFEngine("distilgpt2")
    eng.capacity = 8
    return eng


@pytest.fixture(scope="module")
def runner():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from model_runner import HFModelRunner

    return HFModelRunner("distilgpt2")


def _standalone(model, tokenizer, prompt, max_new, stop_id, pad_id):
    """Greedy token ids for one prompt run entirely on its own, the ground truth
    a continuous-batched sequence must reproduce regardless of its neighbours."""
    import torch

    enc = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            max_new_tokens=max_new,
            do_sample=False,
            eos_token_id=stop_id,
            pad_token_id=pad_id,
        )
    return out[0, enc["input_ids"].shape[1] :].tolist()


def _drive(engine, waves):
    """Pump the engine the way the scheduler does, but synchronously for
    determinism: admit each wave at its tick, step once per tick, collect every
    evicted sequence. Returns {request_id: (EngineResult, evict_tick)}."""
    out = {}
    seqid_req = {}
    waves = sorted(waves, key=lambda w: w[0])
    wi = 0
    tick = 0
    while True:
        while wi < len(waves) and waves[wi][0] <= tick:
            reqs = waves[wi][1]
            for sid, req in zip(engine.admit(reqs), reqs):
                seqid_req[sid] = req.request_id
            wi += 1
        if engine.num_active() == 0 and wi >= len(waves):
            break
        for res in engine.step():
            out[seqid_req[res.seq_id]] = (res, tick)
        tick += 1
    return out


@pytest.mark.model
def test_continuous_matches_generate_default_eos(engine, runner):
    # Core gate: each sequence's continuous token ids equal its standalone greedy
    # ids, and the text equals the P1 static run_batch path. distilgpt2 greedy
    # never emits real EOS, so every sequence runs its full budget.
    engine.reset()
    eos = engine._tokenizer.eos_token_id
    engine.stop_id = eos
    budgets = {"france": 16, "village": 12, "numpy": 20, "hello": 10}
    reqs = [_req(rid, m) for rid, m in budgets.items()]
    out = _drive(engine, [(0, reqs)])

    for rid, m in budgets.items():
        ref = _standalone(engine._model, engine._tokenizer, PROMPTS[rid], m, eos, eos)
        assert out[rid][0].token_ids == ref, rid
        assert len(out[rid][0].token_ids) == m

    static = {o.request_id: o.text for o in runner.run_batch(reqs)}
    for rid in budgets:
        assert out[rid][0].text == static[rid], rid


@pytest.mark.model
def test_per_row_max_new_eviction(engine):
    # Varied budgets: sequences leave at their own max_new_tokens, on different
    # steps, and each still matches standalone greedy.
    engine.reset()
    eos = engine._tokenizer.eos_token_id
    engine.stop_id = eos
    budgets = {"france": 4, "village": 8, "numpy": 12, "hello": 16}
    reqs = [_req(rid, m) for rid, m in budgets.items()]
    out = _drive(engine, [(0, reqs)])

    for rid, m in budgets.items():
        ref = _standalone(engine._model, engine._tokenizer, PROMPTS[rid], m, eos, eos)
        assert out[rid][0].token_ids == ref, rid
    # Eviction ticks are distinct and increase with the budget.
    ticks = [out[rid][1] for rid in ("france", "village", "numpy", "hello")]
    assert ticks == sorted(ticks)
    assert len(set(ticks)) == len(ticks)


@pytest.mark.model
def test_pinned_stop_eviction(engine):
    # Pin the stop token to 198 (newline), which distilgpt2 actually emits, so
    # sequences evict on the stop token at different steps. Token ids must match
    # standalone generate run with the same eos.
    engine.reset()
    eos = engine._tokenizer.eos_token_id
    engine.stop_id = 198
    budget = 24
    reqs = [_req(rid, budget) for rid in PROMPTS]
    out = _drive(engine, [(0, reqs)])

    stop_ticks = []
    for rid in PROMPTS:
        res, tick = out[rid]
        ref = _standalone(engine._model, engine._tokenizer, PROMPTS[rid], budget, 198, eos)
        assert res.token_ids == ref, rid
        # A sequence evicted on the pinned stop ends on 198 (generate keeps the
        # EOS it stops on); record the step it left on.
        if res.token_ids and res.token_ids[-1] == 198:
            stop_ticks.append(tick)
    # Several sequences really stop on 198, and they leave on different steps (not
    # one global stop), so this catches a global-step bug, not just a uniform stop.
    assert len(stop_ticks) >= 2
    assert len(set(stop_ticks)) >= 2


@pytest.mark.model
def test_mid_stream_join_and_leave(engine):
    # Admit two long sequences, step a few times, then admit a short one that
    # evicts while the long ones keep going. All token ids still match greedy.
    engine.reset()
    eos = engine._tokenizer.eos_token_id
    engine.stop_id = eos
    budgets = {"village": 18, "numpy": 18, "france": 4}
    waves = [
        (0, [_req("village", 18), _req("numpy", 18)]),
        (5, [_req("france", 4)]),
    ]
    out = _drive(engine, waves)

    for rid, m in budgets.items():
        ref = _standalone(engine._model, engine._tokenizer, PROMPTS[rid], m, eos, eos)
        assert out[rid][0].token_ids == ref, rid
    # The mid-stream short sequence joined late yet left before the long ones.
    assert out["france"][1] > 0
    assert out["france"][1] < out["village"][1]
    assert out["france"][1] < out["numpy"][1]


@pytest.mark.model
def test_admit_rejects_empty_prompt_atomically(engine):
    # An empty prompt would crash prefill; admit must reject it up front and
    # leave the already-active sequence untouched (admit is atomic).
    engine.reset()
    engine.stop_id = engine._tokenizer.eos_token_id
    engine.admit([_req("france", 4)])
    before = engine.num_active()
    with pytest.raises(ValueError):
        engine.admit([GenerationRequest("empty", "", max_new_tokens=4, temperature=0.0)])
    assert engine.num_active() == before


@pytest.mark.model
def test_admit_rejects_over_context_request_atomically(engine):
    # prompt_tokens + max_new_tokens past the model context (distilgpt2 = 1024)
    # would overflow the position table; reject before decode so a step failure
    # cannot abort innocent batchmates. The active set stays untouched.
    engine.reset()
    engine.stop_id = engine._tokenizer.eos_token_id
    assert engine.context_limit == 1024
    engine.admit([_req("france", 4)])
    before = engine.num_active()
    over = GenerationRequest(
        "toolong", PROMPTS["france"], max_new_tokens=engine.context_limit, temperature=0.0
    )
    with pytest.raises(ValueError):
        engine.admit([over])
    assert engine.num_active() == before
