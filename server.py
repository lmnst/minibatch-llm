from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from config import Config
from engine import build_engine
from schemas import GenerationRequest, Metrics
from scheduler import QueueFull, Scheduler

config = Config.from_env()
scheduler: Scheduler | None = None


class GenerateBody(BaseModel):
    request_id: str
    prompt: str = Field(min_length=1)
    max_new_tokens: int = Field(default=64, gt=0)
    temperature: float = 0.0


@asynccontextmanager
async def lifespan(app: FastAPI):
    global scheduler
    logging.basicConfig(level=logging.INFO)
    engine = build_engine(config.model_id)
    scheduler = Scheduler(
        engine, config.max_batch_size, config.max_queue_depth
    )
    await scheduler.start()
    yield
    await scheduler.stop()


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/generate")
async def generate(body: GenerateBody):
    if body.temperature != 0.0:
        raise HTTPException(
            status_code=400,
            detail="This server supports greedy decoding only; temperature must be 0.0.",
        )
    if body.max_new_tokens > config.max_new_tokens_limit:
        raise HTTPException(
            status_code=422,
            detail=f"max_new_tokens exceeds the limit of {config.max_new_tokens_limit}",
        )
    if scheduler is None:
        raise HTTPException(status_code=503, detail="server not ready")
    t0 = time.perf_counter()
    req = GenerationRequest(
        request_id=body.request_id,
        prompt=body.prompt,
        max_new_tokens=body.max_new_tokens,
        temperature=body.temperature,
    )
    try:
        result = await scheduler.submit(req)
    except QueueFull:
        raise HTTPException(status_code=503, detail="server overloaded; queue full")
    except ValueError as exc:
        # The engine rejects a prompt whose token length plus max_new_tokens
        # would overflow the model context; that is a bad request, not a 500.
        raise HTTPException(status_code=422, detail=str(exc))
    e2e_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "request_id": result.output.request_id,
        "text": result.output.text,
        "metrics": Metrics(
            queue_wait_ms=result.queue_wait_ms,
            generate_ms=result.generate_ms,
            e2e_ms=e2e_ms,
        ),
    }
