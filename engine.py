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


def build_engine(model_id: str) -> InferenceEngine:
    """Factory used by the server. 'fake' yields the torch-free FakeEngine so the
    scheduler and HTTP layers can run without a model. HFEngine lands next."""
    if model_id == "fake":
        return FakeEngine()
    raise NotImplementedError(
        "HFEngine is not wired yet; only 'fake' is available at this stage"
    )
