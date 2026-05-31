# MiniBatch-LLM (Phase 1)

A minimal LLM inference server with static (request-level) batching, running
on CPU with distilgpt2. This is the scaffold that later phases swap real
batching logic into. The scheduler talks to the model through one interface
(`ModelRunner`), which let P1 replace `generate()` with a hand-written
past_key_values decode loop without touching `server.py` or `scheduler.py`.

Requires Python 3.9+.

## Layout

| File | Role |
| --- | --- |
| `server.py` | FastAPI app. Validates the request, submits it, awaits the result, returns text plus metrics. |
| `scheduler.py` | Async size-or-timeout batcher. Closes a batch and dispatches it to the runner off the event loop. |
| `model_runner.py` | `ModelRunner` interface, `FakeRunner` (tests), `HFModelRunner` (distilgpt2), and a `build_runner` factory. |
| `schemas.py` | Plain dataclasses for requests, outputs, metrics. |
| `config.py` | Reads config from environment. |
| `tests/test_scheduler.py` | Scheduler unit tests against `FakeRunner`, no model loaded. |
| `tests/test_server.py` | HTTP-layer tests via `MODEL_ID=fake`, also no model loaded. |
| `scripts/smoke.py` | End-to-end concurrency check against the real model. |

## Run

```
pip install -r requirements.txt
uvicorn server:app --host 0.0.0.0 --port 8000
```

First start downloads distilgpt2 (~350 MB). Health probe: `GET /health`.

Example request (PowerShell):

```
$body = @{ request_id = "a"; prompt = "The capital of France is"; max_new_tokens = 16 } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://localhost:8000/generate -ContentType application/json -Body $body
```

Example request (curl):

```
curl -X POST http://localhost:8000/generate -H "Content-Type: application/json" -d '{"request_id":"a","prompt":"The capital of France is","max_new_tokens":16}'
```

Response:

```
{
  "request_id": "a",
  "text": " the city of Paris...",
  "metrics": {"queue_wait_ms": 3.1, "generate_ms": 412.0, "e2e_ms": 415.4}
}
```

## Config (environment variables)

| Variable | Default | Meaning |
| --- | --- | --- |
| `MAX_BATCH_SIZE` | `8` | Close a batch once this many requests are queued. |
| `MAX_WAIT_MS` | `10` | Close a batch once the oldest request has waited this long. |
| `MODEL_ID` | `distilgpt2` | HuggingFace model id. The value `fake` selects a no-model runner for tests. |
| `MAX_QUEUE_DEPTH` | `1024` | Pending-queue cap. Submitting past it returns 503. |
| `MAX_NEW_TOKENS_LIMIT` | `512` | Per-request `max_new_tokens` ceiling. Past it returns 422. |

## Metrics

Each response carries three timings, in milliseconds:

- `queue_wait_ms`: enqueue until the request was picked into a batch.
- `generate_ms`: the `run_batch` call itself (pure inference, shared by a batch).
- `e2e_ms`: handler receive until response sent.

The scheduler also logs each batch close at INFO with its size, reason
(`full` or `timeout`), and `generate_ms`.

## Tests

The default suite is model-free: it loads neither torch nor transformers, so it
runs anywhere.

```
python -m pytest -m "not model"
```

`tests/test_scheduler.py` drives the scheduler with `FakeRunner`: full-batch
close, timeout close (and that it actually waits), FCFS batch composition,
fill-before-timeout, no loss or duplication, runner-exception isolation,
output-count mismatch isolation, metric sanity, config validation, and queue
backpressure. `tests/test_server.py` exercises the HTTP layer with
`MODEL_ID=fake`: health, metrics shape, the 400 and 422 contracts, and a guard
that importing the server stack pulls neither torch nor transformers. Neither
file loads a model.

The parity gate loads distilgpt2 and is marked `model`:

```
python -m pytest -m model
```

`tests/test_decode_parity.py` checks the hand-written greedy decode loop token
for token against `model.generate(do_sample=False)`: one batch with a uniform
`max_new_tokens` over varied-length prompts, and one batch that pins the stop
token to a value the model actually emits so per-row EOS-to-pad is exercised.
Because this test is marked `model`, bare `python -m pytest` now loads a model
and is no longer model-free; use `-m "not model"` for the model-free suite.

For real-model concurrency:

```
python scripts/smoke.py
```

It starts the server, fires six different prompts at once, and checks that
every response keeps its own request_id and a full set of metrics.

## P0 notes and known limits

- Greedy only. A single HF `generate()` call takes one batch-level
  temperature, so per-request sampling is not expressible under static
  batching. P0 makes that explicit: the server rejects any `temperature` other
  than 0.0 with a 400 rather than silently ignoring it. P1's hand-written loop
  makes per-request sampling expressible but leaves it unwired and greedy.
- Inputs are validated up front. `max_new_tokens` must be in `(0, MAX_NEW_TOKENS_LIMIT]`
  (else 422), `MAX_BATCH_SIZE` must be > 0 and `MAX_WAIT_MS` >= 0 (else the
  server refuses to start), and the queue is bounded (else 503).
- A batch runs to the largest `max_new_tokens` in it. Shorter requests keep
  generating until the whole batch is done, then get truncated. That wasted
  work is the cost of static batching, and is exactly what P2 continuous
  batching removes by evicting a sequence as soon as it hits EOS.

## What P1 changed

`HFModelRunner.run_batch` no longer calls `generate()`. It runs a hand-written
greedy decode loop that drives `past_key_values` directly: one prefill over the
left-padded batch, then an `argmax` per step to the batch's largest
`max_new_tokens`, advancing per-row `position_ids` and the shared
`cache_position` and growing the attention mask each step. A row that emits EOS
has its remaining tokens forced to pad, mirroring `generate`. The loop agrees
with `model.generate(do_sample=False)` token for token, which is the parity gate
above.

The loop stays greedy. It makes per-request sampling expressible, but P1 does
not wire it, so the server still rejects `temperature != 0`. Only
`model_runner.py` and the new parity test changed; because the scheduler depends
only on the `ModelRunner` interface, `server.py` and `scheduler.py` stayed
untouched.
