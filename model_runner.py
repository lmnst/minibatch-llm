from __future__ import annotations

from abc import ABC, abstractmethod

from schemas import GenerationOutput, GenerationRequest


class ModelRunner(ABC):
    """P1 static (request-level) batching baseline.

    Once the scheduler's seam, now superseded by engine.InferenceEngine for
    P2 continuous batching. Kept as the static comparison point for the parity
    guard (tests/test_decode_parity.py) and the benchmark: one run_batch runs the
    whole batch to its largest max_new_tokens."""

    @abstractmethod
    def run_batch(self, reqs: list[GenerationRequest]) -> list[GenerationOutput]:
        """Run one batch. Outputs are returned in the same order as reqs."""
        ...


class HFModelRunner(ModelRunner):
    """Static batching over a HuggingFace causal LM with a hand-written greedy
    KV-cache decode loop.

    run_batch drives past_key_values by hand instead of calling generate(): one
    prefill over the left-padded batch, then a greedy argmax per step up to the
    batch's largest max_new_tokens, growing the attention mask and advancing
    per-row position_ids and the shared cache_position each step. A row that
    emits EOS has its later tokens forced to pad, matching generate. Decoding is
    greedy only; the server still rejects temperature != 0, so this runner never
    has to apply per-request sampling.
    """

    def __init__(self, model_id: str) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

        self._torch = torch
        self._cache_cls = DynamicCache
        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        self._tokenizer.padding_side = "left"
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._model = AutoModelForCausalLM.from_pretrained(model_id)
        self._model.eval()

    def _greedy_decode(self, input_ids, attention_mask, max_new, eos_id, pad_id):
        """Greedy-decode the left-padded batch for max_new steps, driving the KV
        cache by hand, and return the [B, max_new] grid of generated token ids.

        Two things keep this in lock-step with generate() instead of silently
        desyncing. First, position_ids are built from the mask and advanced per
        row: the model default does not account for left padding, and rows have
        different real prompt lengths, so there is no single step scalar. Second,
        once a row emits eos_id its later tokens are forced to pad_id, so a
        finished row stops following the model argmax exactly as generate() does.
        cache_position is the shared time index into the cache, distinct from
        position_ids.
        """
        torch = self._torch
        batch_size, prompt_len = input_ids.shape
        # Real position of each token under left padding; padded slots fold to 0.
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 0)
        # Absolute position of each row's first generated token (its real length).
        next_position = position_ids[:, -1] + 1
        with torch.no_grad():
            out = self._model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                cache_position=torch.arange(prompt_len),
                past_key_values=self._cache_cls(),
                use_cache=True,
            )
            cache = out.past_key_values
            # Left padding keeps the last column valid for every row.
            logits = out.logits[:, -1, :]
            mask = attention_mask
            cache_len = prompt_len
            unfinished = torch.ones(batch_size, dtype=torch.bool)
            generated = []
            for step in range(max_new):
                # argmax takes the lowest index on ties, matching HF greedy.
                next_tokens = torch.argmax(logits, dim=-1)
                # A finished row emits pad, not the model argmax.
                next_tokens = torch.where(
                    unfinished, next_tokens, torch.full_like(next_tokens, pad_id)
                )
                generated.append(next_tokens)
                # The emitted EOS is kept; the row counts as finished from here on.
                unfinished = unfinished & (next_tokens != eos_id)
                if step == max_new - 1:
                    break
                mask = torch.cat(
                    [mask, torch.ones((batch_size, 1), dtype=mask.dtype)], dim=-1
                )
                out = self._model(
                    input_ids=next_tokens.unsqueeze(-1),
                    attention_mask=mask,
                    position_ids=(next_position + step).unsqueeze(-1),
                    cache_position=torch.tensor([cache_len]),
                    past_key_values=cache,
                    use_cache=True,
                )
                logits = out.logits[:, -1, :]
                cache = out.past_key_values
                cache_len += 1
        return torch.stack(generated, dim=1)

    def run_batch(self, reqs: list[GenerationRequest]) -> list[GenerationOutput]:
        prompts = [r.prompt for r in reqs]
        enc = self._tokenizer(prompts, return_tensors="pt", padding=True)
        max_new = max(r.max_new_tokens for r in reqs)
        eos_id = self._tokenizer.eos_token_id
        # distilgpt2 has pad == eos, so the same id serves both roles.
        generated = self._greedy_decode(
            enc["input_ids"], enc["attention_mask"], max_new, eos_id, eos_id
        )
        results: list[GenerationOutput] = []
        for r, row in zip(reqs, generated):
            row = row[: r.max_new_tokens]
            eos_hits = (row == eos_id).nonzero()
            if eos_hits.numel() > 0:
                row = row[: eos_hits[0].item()]
            text = self._tokenizer.decode(row, skip_special_tokens=True)
            results.append(GenerationOutput(request_id=r.request_id, text=text))
        return results
