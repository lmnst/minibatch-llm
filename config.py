from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class Config:
    max_batch_size: int
    max_wait_ms: float
    model_id: str

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            max_batch_size=int(os.getenv("MAX_BATCH_SIZE", "8")),
            max_wait_ms=float(os.getenv("MAX_WAIT_MS", "10")),
            model_id=os.getenv("MODEL_ID", "distilgpt2"),
        )
