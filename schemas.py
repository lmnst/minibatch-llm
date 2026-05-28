from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GenerationRequest:
    request_id: str
    prompt: str
    max_new_tokens: int
    temperature: float


@dataclass
class GenerationOutput:
    request_id: str
    text: str


@dataclass
class ScheduleResult:
    output: GenerationOutput
    queue_wait_ms: float
    generate_ms: float


@dataclass
class Metrics:
    queue_wait_ms: float
    generate_ms: float
    e2e_ms: float
