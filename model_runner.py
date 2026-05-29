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


class HFModelRunner(ModelRunner):
    """Static batching over a HuggingFace causal LM.

    P0 is greedy only and always runs do_sample=False. A single HF generate()
    call takes one batch-level temperature, so per-request sampling is not
    expressible here. The server enforces that contract by rejecting
    temperature != 0, so this runner never has to apply it. Per-request
    sampling lands in P1 with the hand-written decode loop.
    """

    def __init__(self, model_id: str) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        self._tokenizer.padding_side = "left"
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._model = AutoModelForCausalLM.from_pretrained(model_id)
        self._model.eval()

    def run_batch(self, reqs: list[GenerationRequest]) -> list[GenerationOutput]:
        torch = self._torch
        prompts = [r.prompt for r in reqs]
        enc = self._tokenizer(prompts, return_tensors="pt", padding=True)
        max_new = max(r.max_new_tokens for r in reqs)
        with torch.no_grad():
            out = self._model.generate(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                max_new_tokens=max_new,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        input_len = enc["input_ids"].shape[1]
        generated = out[:, input_len:]
        eos_id = self._tokenizer.eos_token_id
        results: list[GenerationOutput] = []
        for r, row in zip(reqs, generated):
            row = row[: r.max_new_tokens]
            eos_hits = (row == eos_id).nonzero()
            if eos_hits.numel() > 0:
                row = row[: eos_hits[0].item()]
            text = self._tokenizer.decode(row, skip_special_tokens=True)
            results.append(GenerationOutput(request_id=r.request_id, text=text))
        return results


def build_runner(model_id: str) -> ModelRunner:
    """Factory used by the server. A model_id of 'fake' yields FakeRunner, so
    the HTTP layer can be exercised without loading a model."""
    if model_id == "fake":
        return FakeRunner()
    return HFModelRunner(model_id)
