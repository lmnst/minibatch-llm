from __future__ import annotations

from abc import ABC, abstractmethod

from schemas import GenerationOutput, GenerationRequest


class ModelRunner(ABC):
    """Single seam the scheduler talks to. Swappable so P1 can replace the
    body of run_batch with a hand-written past_key_values decode loop without
    touching server.py or scheduler.py."""

    @abstractmethod
    def run_batch(self, reqs: list[GenerationRequest]) -> list[GenerationOutput]:
        """Run one batch. Outputs are returned in the same order as reqs."""
        ...


class FakeRunner(ModelRunner):
    """Deterministic runner for scheduler tests. Loads no model, no torch."""

    def __init__(self) -> None:
        self.call_count = 0
        self.batch_sizes: list[int] = []
        self.batches: list[list[str]] = []

    def run_batch(self, reqs: list[GenerationRequest]) -> list[GenerationOutput]:
        self.call_count += 1
        self.batch_sizes.append(len(reqs))
        self.batches.append([r.request_id for r in reqs])
        return [
            GenerationOutput(request_id=r.request_id, text=f"{r.prompt}|{r.request_id}")
            for r in reqs
        ]
