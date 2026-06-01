from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class Config:
    max_batch_size: int
    model_id: str
    max_queue_depth: int
    max_new_tokens_limit: int

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            max_batch_size=int(os.getenv("MAX_BATCH_SIZE", "8")),
            model_id=os.getenv("MODEL_ID", "distilgpt2"),
            max_queue_depth=int(os.getenv("MAX_QUEUE_DEPTH", "1024")),
            max_new_tokens_limit=int(os.getenv("MAX_NEW_TOKENS_LIMIT", "512")),
        )
