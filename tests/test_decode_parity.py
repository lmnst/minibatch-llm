from __future__ import annotations

import pytest

# torch and transformers are imported inside the fixture and the test bodies,
# never at module top level. pytest imports every test module during collection
# even under -m "not model", so a top-level ML import would land in sys.modules
# and break tests/test_server.py::test_no_ml_imports_on_server_path. Keeping the
# imports lazy leaves this model test model-free at collection time.


@pytest.fixture(scope="module")
def runner():
    # importorskip lets a torch-free environment, such as the model-free CI box,
    # skip these tests instead of erroring when the runner imports torch.
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from model_runner import HFModelRunner

    return HFModelRunner("distilgpt2")


def _reference_grid(model, enc, max_new, eos_id, pad_id):
    import torch

    with torch.no_grad():
        out = model.generate(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            max_new_tokens=max_new,
            do_sample=False,
            eos_token_id=eos_id,
            pad_token_id=pad_id,
        )
    return out[:, enc["input_ids"].shape[1] :]


# Varied prompt lengths stress per-row position_ids under left padding.
PROMPTS = [
    "The capital of France is",
    "Once upon a time, in a small village near the mountains, there lived",
    "import numpy as np",
    "Hello there, my friend. How are",
]


@pytest.mark.model
def test_uniform_max_new_grid_matches_generate(runner):
    import torch

    tokenizer = runner._tokenizer
    enc = tokenizer(PROMPTS, return_tensors="pt", padding=True)
    eos_id = tokenizer.eos_token_id
    batch_max = 24
    # distilgpt2 greedy never emits its real EOS, so this case is pure raw-greedy
    # parity: prefill, per-row positions, cache round-trip, and the argmax
    # tie-break must match generate over the full [B, batch_max] grid.
    reference = _reference_grid(runner._model, enc, batch_max, eos_id, eos_id)
    manual = runner._greedy_decode(
        enc["input_ids"], enc["attention_mask"], batch_max, eos_id, eos_id
    )
    assert torch.equal(manual, reference)


@pytest.mark.model
def test_eos_to_pad_grid_matches_generate(runner):
    import torch

    tokenizer = runner._tokenizer
    enc = tokenizer(PROMPTS, return_tensors="pt", padding=True)
    batch_max = 16
    # distilgpt2 greedy never emits its real EOS, so pin the stop token to one it
    # does emit (newline, 198) to exercise per-row stop and post-EOS pad. pad
    # stays the real pad (50256), distinct from the model argmax at those steps,
    # so the forced pad actually changes the grid and the check is discriminating.
    # One prompt never emits 198, so generate runs the full batch_max and the
    # grids stay [B, batch_max]. The same stop/pad ids go through both paths, so
    # this is still a token-for-token parity check against generate.
    stop_id = 198
    pad_id = tokenizer.eos_token_id
    reference = _reference_grid(runner._model, enc, batch_max, stop_id, pad_id)
    manual = runner._greedy_decode(
        enc["input_ids"], enc["attention_mask"], batch_max, stop_id, pad_id
    )
    assert torch.equal(manual, reference)
    # Guard the intent: the case really does exercise stop and post-EOS pad, and
    # finishing rows do so at different steps (so this catches a global-step bug,
    # not just a single uniform stop).
    assert (reference == stop_id).any()
    assert (reference == pad_id).any()
    finish_steps = [
        int((row == pad_id).long().argmax()) for row in reference if (row == pad_id).any()
    ]
    assert len(set(finish_steps)) > 1


@pytest.mark.model
def test_run_batch_applies_per_row_max_new(runner):
    import torch

    from schemas import GenerationRequest

    tokenizer = runner._tokenizer
    eos_id = tokenizer.eos_token_id
    # Varied per-row max_new: run_batch runs the whole batch to the largest, then
    # truncates each row to its own budget. This is the only test that drives
    # run_batch end to end, so it covers tokenize -> decode -> per-row truncate ->
    # text, the part the grid-level parity tests do not touch.
    per_row_max_new = [16, 12, 8, 4]
    reqs = [
        GenerationRequest(request_id=f"r{i}", prompt=p, max_new_tokens=m, temperature=0.0)
        for i, (p, m) in enumerate(zip(PROMPTS, per_row_max_new))
    ]
    batch_max = max(per_row_max_new)
    enc = tokenizer(PROMPTS, return_tensors="pt", padding=True)
    reference = _reference_grid(runner._model, enc, batch_max, eos_id, eos_id)
    # Expected text replays run_batch's post-processing on the reference grid:
    # truncate each row to its own max_new_tokens, cut at the first EOS, decode.
    expected = []
    for row, budget in zip(reference, per_row_max_new):
        row = row[:budget]
        eos_hits = (row == eos_id).nonzero()
        if eos_hits.numel() > 0:
            row = row[: eos_hits[0].item()]
        expected.append(tokenizer.decode(row, skip_special_tokens=True))
    outputs = runner.run_batch(reqs)
    assert [o.request_id for o in outputs] == [r.request_id for r in reqs]
    assert [o.text for o in outputs] == expected
