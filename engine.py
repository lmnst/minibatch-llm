from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from schemas import GenerationRequest


@dataclass
class EngineResult:
    """A finished sequence handed back by step(). Identified by the engine's own
    seq_id, never by request_id, so a duplicate request_id never cross-talks."""

    seq_id: int
    text: str
    token_ids: list[int]


class InferenceEngine(ABC):
    """Iteration-level seam the scheduler pumps.

    The engine owns active-sequence state and the KV cache; the scheduler and
    server stay torch-free. Each active sequence advances one token per step and
    is evicted the moment it emits the stop token or reaches its own
    max_new_tokens, so its result resolves independently of its batchmates.

    capacity is the maximum number of concurrently active sequences (the KV slot
    cap); the scheduler sets it to its concurrency cap.
    """

    capacity: int

    @abstractmethod
    def num_active(self) -> int:
        """Number of sequences currently in the active set."""
        ...

    @abstractmethod
    def admit(self, reqs: list[GenerationRequest]) -> list[int]:
        """Prefill and admit newcomers, returning one seq_id per request in
        order. Atomic: if it raises, the active set and KV state are untouched."""
        ...

    @abstractmethod
    def step(self) -> list[EngineResult]:
        """Advance every active sequence one token, evict the finished ones, and
        return them. A sequence already finished at prefill is drained here
        without a wasted decode forward."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """Drop all active sequences and KV state after an engine failure."""
        ...


@dataclass
class _FakeSeq:
    seq_id: int
    request: GenerationRequest
    n_generated: int
    finished: bool


class FakeEngine(InferenceEngine):
    """Deterministic, torch-free engine for scheduler tests.

    Mirrors HFEngine's lifecycle so the scheduler tests are faithful: admit
    emits one token (the prefill token), each step emits one more, and a
    sequence is evicted once its generated count reaches its own
    max_new_tokens. A max_new_tokens of 1 finishes at admit and is drained on
    the next step with no decode, exactly as the real engine drains a
    prefill-finished sequence.
    """

    def __init__(self) -> None:
        self.capacity = 8
        self._active: dict[int, _FakeSeq] = {}
        self._next_seq_id = 0

    def num_active(self) -> int:
        return len(self._active)

    def reset(self) -> None:
        self._active = {}

    def admit(self, reqs: list[GenerationRequest]) -> list[int]:
        if not reqs:
            return []
        if len(self._active) + len(reqs) > self.capacity:
            raise RuntimeError(
                f"admit of {len(reqs)} would exceed capacity {self.capacity} "
                f"(active={len(self._active)})"
            )
        seq_ids: list[int] = []
        for req in reqs:
            sid = self._next_seq_id
            self._next_seq_id += 1
            self._active[sid] = _FakeSeq(
                seq_id=sid,
                request=req,
                n_generated=1,
                finished=req.max_new_tokens <= 1,
            )
            seq_ids.append(sid)
        return seq_ids

    def step(self) -> list[EngineResult]:
        if not self._active:
            return []
        for seq in self._active.values():
            if not seq.finished:
                seq.n_generated += 1
                if seq.n_generated >= seq.request.max_new_tokens:
                    seq.finished = True
        finished = [seq for seq in self._active.values() if seq.finished]
        for seq in finished:
            del self._active[seq.seq_id]
        return [
            EngineResult(
                seq_id=seq.seq_id,
                text=f"{seq.request.prompt}|{seq.request.request_id}",
                token_ids=list(range(seq.n_generated)),
            )
            for seq in finished
        ]


@dataclass
class _Seq:
    """Per-sequence decode state.

    kv holds this sequence's own unpadded cache, batch dim 1, length L_i, as a
    per-layer list of (key, value). generated_ids already includes last_token,
    which has NOT been fed yet: its KV lands in the cache on the next decode
    step, at absolute position length (= L_i).
    """

    seq_id: int
    request: GenerationRequest
    generated_ids: list[int]
    last_token: int
    kv: list  # per layer (key, value), each tensor [1, H, L_i, D]
    length: int
    finished: bool


class HFEngine(InferenceEngine):
    """Continuous (iteration-level) engine over a HuggingFace causal LM.

    Each active sequence keeps its own unpadded KV cache (batch dim 1, length
    L_i). A decode step reassembles a left-padded batched cache from the
    per-sequence caches, runs one forward over the active rows, splits the
    updated cache back into per-sequence caches, and evicts whoever finished.
    This is correctness-first: copying per step, no shared mutable batched cache
    and no PagedAttention. CPU and distilgpt2 (pad == eos) only.
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
        # Default stop is real EOS, but distilgpt2 greedy almost never emits it,
        # so tests override this (e.g. 198) to exercise eviction. Injectable, not
        # a hardcoded 50256.
        self.stop_id = self._tokenizer.eos_token_id
        # Largest absolute position the model can encode. Used to reject requests
        # whose prompt plus max_new_tokens would overflow the position table.
        self.context_limit = getattr(
            self._model.config, "n_positions", None
        ) or getattr(self._model.config, "max_position_embeddings", None)
        self.capacity = 8
        self._active: dict[int, _Seq] = {}
        self._next_seq_id = 0

    def num_active(self) -> int:
        return len(self._active)

    def reset(self) -> None:
        # Drop active sequences and their KV; the model and tokenizer survive.
        self._active = {}

    def admit(self, reqs: list[GenerationRequest]) -> list[int]:
        if not reqs:
            return []
        if len(self._active) + len(reqs) > self.capacity:
            raise RuntimeError(
                f"admit of {len(reqs)} would exceed capacity {self.capacity} "
                f"(active={len(self._active)})"
            )
        torch = self._torch
        prompts = [r.prompt for r in reqs]
        enc = self._tokenizer(prompts, return_tensors="pt", padding=True)
        input_ids = enc["input_ids"]
        mask = enc["attention_mask"]
        prompt_len = input_ids.shape[1]
        # Reject impossible requests before any forward, while the active set is
        # still untouched (admit stays atomic). An empty prompt would crash
        # prefill, and a prompt at or past the context limit would IndexError now
        # or blow up a later decode step and take its innocent batchmates down.
        for k, req in enumerate(reqs):
            real_len = int(mask[k].sum().item())
            if real_len <= 0:
                raise ValueError(f"request {req.request_id!r} has an empty prompt")
            if self.context_limit is not None and (
                real_len + req.max_new_tokens > self.context_limit
            ):
                raise ValueError(
                    f"request {req.request_id!r} needs {real_len} prompt + "
                    f"{req.max_new_tokens} new tokens, over the model context "
                    f"limit of {self.context_limit}"
                )
        # Real per-token positions under left padding; padded slots fold to 0.
        position_ids = mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(mask == 0, 0)
        with torch.no_grad():
            out = self._model(
                input_ids=input_ids,
                attention_mask=mask,
                position_ids=position_ids,
                cache_position=torch.arange(prompt_len),
                past_key_values=self._cache_cls(),
                use_cache=True,
            )
        first_tokens = torch.argmax(out.logits[:, -1, :], dim=-1)
        legacy = out.past_key_values.to_legacy_cache()  # per layer (k, v) [K,H,P,D]
        # Build everything in locals first; commit to _active only at the end so
        # a failure above leaves the active set and KV untouched (atomic admit).
        new_seqs: list[_Seq] = []
        for k, req in enumerate(reqs):
            real_len = int(mask[k].sum().item())
            # This row's real KV is the right-aligned real_len columns of the
            # left-padded prefill batch. contiguous() drops the view onto the big
            # batched tensor so it can be freed.
            kv = [
                (
                    key[k : k + 1, :, prompt_len - real_len :, :].contiguous(),
                    val[k : k + 1, :, prompt_len - real_len :, :].contiguous(),
                )
                for key, val in legacy
            ]
            tok = int(first_tokens[k].item())
            new_seqs.append(
                _Seq(
                    seq_id=-1,
                    request=req,
                    generated_ids=[tok],
                    last_token=tok,
                    kv=kv,
                    length=real_len,
                    # First token may already finish the sequence (it is the stop
                    # token, or the budget was 1). step() then drains it with no
                    # decode forward.
                    finished=(tok == self.stop_id or req.max_new_tokens <= 1),
                )
            )
        seq_ids: list[int] = []
        for seq in new_seqs:
            seq.seq_id = self._next_seq_id
            self._next_seq_id += 1
            self._active[seq.seq_id] = seq
            seq_ids.append(seq.seq_id)
        return seq_ids

    def step(self) -> list[EngineResult]:
        if not self._active:
            return []
        decode_seqs = [s for s in self._active.values() if not s.finished]
        if decode_seqs:
            self._decode(decode_seqs)
        # Collect prefill-finished and freshly-finished sequences in active order.
        finished = [s for s in self._active.values() if s.finished]
        for seq in finished:
            del self._active[seq.seq_id]
        return [self._result(seq) for seq in finished]

    def _decode(self, seqs: list[_Seq]) -> None:
        """One greedy decode step over the active rows: reassemble a left-padded
        batched KV cache, forward, split the updated cache back per sequence."""
        torch = self._torch
        lengths = [s.length for s in seqs]
        l_max = max(lengths)
        batch = len(seqs)
        input_ids = torch.tensor([[s.last_token] for s in seqs], dtype=torch.long)
        n_layers = len(seqs[0].kv)
        layers = []
        for layer in range(n_layers):
            sample_key = seqs[0].kv[layer][0]
            _, n_heads, _, head_dim = sample_key.shape
            k_batch = torch.zeros(
                (batch, n_heads, l_max, head_dim), dtype=sample_key.dtype
            )
            v_batch = torch.zeros(
                (batch, n_heads, l_max, head_dim), dtype=sample_key.dtype
            )
            # Right-align each row's real KV; the left columns stay zero and are
            # masked out, so a shorter neighbor never leaks into attention.
            for b, seq in enumerate(seqs):
                key, val = seq.kv[layer]
                k_batch[b : b + 1, :, l_max - seq.length :, :] = key
                v_batch[b : b + 1, :, l_max - seq.length :, :] = val
            layers.append((k_batch, v_batch))
        cache = self._cache_cls.from_legacy_cache(tuple(layers))
        # Row b: (l_max - L_b) leading pad zeros, then L_b + 1 ones (the +1 is the
        # token fed this step). position_ids is the row's own absolute length;
        # cache_position is the shared padded cache length where the new KV lands.
        mask = torch.zeros((batch, l_max + 1), dtype=torch.long)
        for b, seq in enumerate(seqs):
            mask[b, l_max - seq.length :] = 1
        position_ids = torch.tensor([[s.length] for s in seqs], dtype=torch.long)
        with torch.no_grad():
            out = self._model(
                input_ids=input_ids,
                attention_mask=mask,
                position_ids=position_ids,
                cache_position=torch.tensor([l_max]),
                past_key_values=cache,
                use_cache=True,
            )
        next_tokens = torch.argmax(out.logits[:, -1, :], dim=-1)
        updated = out.past_key_values.to_legacy_cache()  # per layer [B,H,l_max+1,D]
        for b, seq in enumerate(seqs):
            tok = int(next_tokens[b].item())
            old_len = lengths[b]
            # The new token's KV sits at index l_max; this row's real KV is the
            # right-aligned old_len + 1 columns. contiguous() frees the batch view.
            seq.kv = [
                (
                    key[b : b + 1, :, l_max - old_len :, :].contiguous(),
                    val[b : b + 1, :, l_max - old_len :, :].contiguous(),
                )
                for key, val in updated
            ]
            seq.generated_ids.append(tok)
            seq.last_token = tok
            seq.length = old_len + 1
            if tok == self.stop_id or len(seq.generated_ids) >= seq.request.max_new_tokens:
                seq.finished = True

    def _result(self, seq: _Seq) -> EngineResult:
        # token_ids keep the trailing stop (generate includes the EOS it stops
        # on); text cuts at the first stop and drops specials, matching the P1
        # static run_batch text.
        ids = seq.generated_ids
        cut = ids
        for i, tok in enumerate(ids):
            if tok == self.stop_id:
                cut = ids[:i]
                break
        text = self._tokenizer.decode(cut, skip_special_tokens=True)
        return EngineResult(seq_id=seq.seq_id, text=text, token_ids=list(ids))


def build_engine(model_id: str) -> InferenceEngine:
    """Factory used by the server. 'fake' yields the torch-free FakeEngine so the
    scheduler and HTTP layers can run without a model."""
    if model_id == "fake":
        return FakeEngine()
    return HFEngine(model_id)
