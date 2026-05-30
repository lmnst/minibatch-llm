# MiniBatch-LLM (Phase 0)

A minimal LLM inference server with static (request-level) batching, running
on CPU with distilgpt2. This is the scaffold that later phases swap real
batching logic into. The scheduler talks to the model through one interface
(`ModelRunner`), so P1 can replace `generate()` with a hand-written
past_key_values decode loop without touching `server.py` or `scheduler.py`.

Requires Python 3.9+.

## Layout

| File | Role |
| --- | --- |
| `server.py` | FastAPI app. Submits a request, awaits the result, returns text plus metrics. |
| `scheduler.py` | Async size-or-timeout batcher. Closes a batch and dispatches it to the runner off the event loop. |
| `model_runner.py` | `ModelRunner` interface, `FakeRunner` (tests), `HFModelRunner` (distilgpt2), and a `build_runner` factory. |
| `schemas.py` | Plain dataclasses for requests, outputs, metrics. |
| `config.py` | Reads config from environment. |
| `tests/test_scheduler.py` | Scheduler unit tests against `FakeRunner`, no model loaded. |
| `scripts/smoke.py` | End-to-end concurrency check against the real model. |

## Run

```
pip install -r requirements.txt
uvicorn server:app --host 0.0.0.0 --port 8000
```

First start downloads distilgpt2 (~350 MB). Health probe: `GET /health`.

Example request (curl):

```
curl -X POST http://localhost:8000/generate -H "Content-Type: application/json" -d '{"request_id":"a","prompt":"The capital of France is","max_new_tokens":16}'
```

## Config (environment variables)

| Variable | Default | Meaning |
| --- | --- | --- |
| `MAX_BATCH_SIZE` | `8` | Close a batch once this many requests are queued. |
| `MAX_WAIT_MS` | `10` | Close a batch once the oldest request has waited this long. |
| `MODEL_ID` | `distilgpt2` | HuggingFace model id. The value `fake` selects a no-model runner. |

## Metrics

Each response carries three timings, in milliseconds:

- `queue_wait_ms`: enqueue until the request was picked into a batch.
- `generate_ms`: the `run_batch` call itself (pure inference, shared by a batch).
- `e2e_ms`: handler receive until response sent.

## Tests

```
python -m pytest
```

For real-model concurrency:

```
python scripts/smoke.py
```

## P0 notes

- Greedy only. The server rejects any `temperature` other than 0.0 (a static
  batch shares one decode setting; per-request sampling lands in P1).
- A batch runs to the largest `max_new_tokens` in it. Shorter requests keep
  generating until the whole batch is done, then get truncated. That wasted
  work is what P2 continuous batching removes.
