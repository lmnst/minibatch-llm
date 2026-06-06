# MiniBatch-LLM

A minimal LLM inference server with continuous (iteration-level) batching,
running on CPU with distilgpt2. The scheduler advances a running set of
sequences one token at a time, evicts each sequence the moment it hits its stop
token or its own `max_new_tokens`, and refills the freed slots from the queue
between steps. All torch and KV-cache state lives behind one interface
(`InferenceEngine`), so `server.py` and `scheduler.py` never import transformers
and a `fake` engine drives the scheduler tests with no model.

Requires Python 3.9+.

## Status

Work in progress, milestone 3 of 4.

- P0 (done): static request-level batching scaffold on CPU, the swappable
  runner seam, scheduler and HTTP unit tests.
- P1 (done): hand-written greedy KV-cache decode loop replacing `generate()`,
  proven token for token against `generate(do_sample=False)`.
- P2 (done, this milestone): continuous (iteration-level) batching. Each active
  sequence advances one token per step, is evicted at its stop token or its own
  `max_new_tokens`, and freed slots are refilled from the queue. Greedy output
  is unchanged, and a variable-output-length benchmark shows it beats static on
  throughput without worsening tail latency.
- P3 (later): GPU benchmark with throughput and p50/p95/p99/tokens-per-sec
  curves plus a roofline writeup that explains them.

## Layout

| File | Role |
| --- | --- |
| `server.py` | FastAPI app. Validates the request, submits it, awaits the result, returns text plus metrics. |
| `scheduler.py` | Async continuous batcher. Pumps the engine: admit from the queue up to the slot cap, step once off the event loop, resolve each evicted sequence's future. |
| `engine.py` | `InferenceEngine` seam, `FakeEngine` (torch-free, for tests), `HFEngine` (distilgpt2, owns torch and the KV cache), and a `build_engine` factory. |
| `model_runner.py` | P1 static `HFModelRunner` (one `run_batch` to the batch max). Kept as the static baseline for the parity guard and the benchmark. |
| `schemas.py` | Plain dataclasses for requests, outputs, metrics. |
| `config.py` | Reads config from environment. |
| `tests/test_scheduler.py` | Continuous-loop unit tests against `FakeEngine`, no model loaded. |
| `tests/test_engine_parity.py` | Correctness gate (`model`): continuous token ids equal standalone greedy. |
| `tests/test_decode_parity.py` | P1 static decode parity (`model`), kept as the baseline guard. |
| `tests/test_server.py` | HTTP-layer tests via `MODEL_ID=fake`, no model loaded. |
| `scripts/benchmark.py` | Static vs continuous throughput and p50/p95 on a variable-length workload. |
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
  "metrics": {"queue_wait_ms": 3.1, "generate_ms": 180.0, "e2e_ms": 184.4}
}
```

## How continuous batching works

The scheduler talks to an `InferenceEngine` with a tiny iteration-level seam:

- `capacity` / `num_active()`: the active-sequence (KV slot) cap and current
  occupancy.
- `admit(reqs) -> [seq_id]`: prefill newcomers and add them to the active set.
  Atomic: if it raises, the active set and KV state are untouched.
- `step() -> [EngineResult]`: advance every active sequence one token, evict the
  finished ones, and return them (by `seq_id`, never `request_id`).
- `reset()`: drop all active sequences and KV state after a failure.

`HFEngine` owns all the torch. Each active sequence keeps its **own unpadded KV
cache** (batch dim 1, length `L_i`) plus its `last_token` and absolute position.
A decode step:

1. Reassembles a **left-padded** batched KV cache from the per-sequence caches,
   right-aligning each row's real `L_i` columns and zeroing the rest.
2. Runs one forward over the active rows. The attention mask gives row `i`
   exactly `L_i + 1` ones (its real cache plus the token fed this step) and
   `L_max - L_i` leading zeros, so a shorter neighbor never leaks into attention;
   `position_ids` is the row's own length and `cache_position` is the shared
   padded length.
3. Splits the updated cache back into per-sequence caches, appends each new
   token, and marks a sequence finished on its stop token or its own
   `max_new_tokens`.

This is correctness-first: it copies the cache every step rather than mutating a
shared batched cache, and there is no PagedAttention (an explicit project
non-goal). A newcomer is prefilled as a left-padded batch in `admit`; if its
first token already finishes it (stop token, or `max_new_tokens == 1`), `step`
drains it with no wasted decode forward. `FakeEngine` mirrors this lifecycle
with no torch so the scheduler tests stay deterministic and model-free.

The scheduler loop admits as many queued requests as there are free slots (FCFS),
runs `step` in an executor (keeping torch off the event loop), and resolves each
returned sequence's future immediately. A short request no longer waits out a
long batchmate. `admit` failures abort only their newcomers; a `step` failure
aborts the active set, calls `reset`, and the loop keeps serving the queue.

## Config (environment variables)

| Variable | Default | Meaning |
| --- | --- | --- |
| `MAX_BATCH_SIZE` | `8` | Maximum active sequences (the KV slot cap / concurrency cap). |
| `MODEL_ID` | `distilgpt2` | HuggingFace model id. The value `fake` selects a no-model engine for tests. |
| `MAX_QUEUE_DEPTH` | `1024` | Pending-queue cap. Submitting past it returns 503. |
| `MAX_NEW_TOKENS_LIMIT` | `512` | Per-request `max_new_tokens` ceiling. Past it returns 422. |

There is no batch-close timeout: the continuous loop steps whenever any sequence
is active and refills freed slots immediately, so the size-or-timeout
`MAX_WAIT_MS` from P0/P1 is gone.

## Input validation

A request is rejected up front: empty `prompt` or `max_new_tokens` outside
`(0, MAX_NEW_TOKENS_LIMIT]` returns 422, `temperature != 0` returns 400 (greedy
only), and a full queue returns 503. `HFEngine.admit` adds one more check that
needs the tokenizer: a request whose real prompt token length plus
`max_new_tokens` would overflow the model context window (distilgpt2 has 1024
positions) is rejected before it enters decode, which over HTTP surfaces as 422.
Rejecting in `admit` rather than mid-decode keeps a too-long request from failing
a decode step and aborting its innocent batchmates. `admit` stays atomic, so a
rejected newcomer never disturbs the active set.

## Metrics

Each response carries three timings, in milliseconds:

- `queue_wait_ms`: enqueue until the request was admitted (prefilled).
- `generate_ms`: admission until the sequence finished. Per request now, not a
  batch-shared value: it covers this request's prefill plus its own decode span.
- `e2e_ms`: handler receive until response sent.

The scheduler logs admit and step failures at ERROR with the affected counts.

## Tests

The default suite is model-free: it loads neither torch nor transformers, so it
runs anywhere.

```
python -m pytest -m "not model"
```

`tests/test_scheduler.py` drives the continuous loop with `FakeEngine`: capacity
never exceeded, FCFS admission, eviction frees a slot and admits the next queued
request, no loss or duplication under churn with N well above capacity, staggered
`max_new_tokens` evicting in budget order, queue backpressure, admit-exception
isolation, step-exception isolation with reset, duplicate-`request_id`
non-crosstalk, metric sanity, config validation, and the engine's atomic-admit
contract. `tests/test_server.py` exercises the HTTP layer with `MODEL_ID=fake`:
health, metrics shape, the 400 and 422 contracts, and a guard (in a fresh
interpreter) that importing the server stack pulls neither torch nor
transformers. Neither file loads a model.

The gates load distilgpt2 and are marked `model`:

```
python -m pytest -m model
```

`tests/test_engine_parity.py` is the P2 correctness gate. It drives `HFEngine`
directly (synchronously, for determinism) and checks that each sequence's
continuous-batched token ids equal its **standalone** `generate(do_sample=False)`
ids, across four cases: default EOS with the static text also matching
`HFModelRunner`; per-row `max_new_tokens` eviction at different steps; a pinned
stop token (`198`, a newline distilgpt2 actually emits) so eviction fires
mid-batch; and a mid-stream join where a short sequence is admitted late and
leaves before the long ones. `tests/test_decode_parity.py` keeps the P1 static
decode parity as the baseline guard. Because these are marked `model`, bare
`python -m pytest` loads a model; use `-m "not model"` for the model-free suite.

For real-model concurrency over HTTP:

```
python scripts/smoke.py
```

## Benchmark

`scripts/benchmark.py` compares static and continuous batching in process on the
same fixed, variable-output-length workload:

```
python scripts/benchmark.py
```

**Methodology.** All N requests are available at `t0` (a saturated queue), with
prompt lengths varied and `max_new_tokens` drawn from `[8, 16, 32, 48]` under a
fixed seed, so short and long requests are interleaved across batches. distilgpt2
greedy never emits EOS here, so each request delivers exactly its
`max_new_tokens`; the total useful tokens are identical for both paths and the
throughput difference is pure wall-clock. Static runs FCFS chunks of `capacity`
through `HFModelRunner.run_batch` (each chunk runs to the largest
`max_new_tokens` in it). Continuous feeds the same requests through `HFEngine`,
evicting and timing each sequence at its own finish. Latency is per request from
`t0`; throughput is total delivered tokens over wall time; both are reported as
the median of 3 runs after an untimed warmup.

**Machine.** Intel Core (Family 6 Model 166), CPU only. Python 3.9.12, torch
2.3.0, transformers 4.57.3. `capacity = 8`, `N = 32`, 3 runs.

**Results (representative run on this machine):**

| metric | static | continuous |
| --- | --- | --- |
| tokens/sec | 48.8 | 54.8 |
| p50 latency (ms) | 8319.7 | 7615.3 |
| p95 latency (ms) | 13446.0 | 11791.2 |

Continuous wins throughput (about 1.12x here) and lowers p95, so the gate
(`continuous tokens/sec > static` and `continuous p95 <= static p95`) passes.

**Why continuous wins.** A static chunk runs every row to the chunk's largest
`max_new_tokens`, so a short request keeps doing forward passes (forced to pad)
until its longest batchmate finishes. Continuous evicts a sequence the step it
finishes and immediately admits a waiting one, so no slot burns compute on an
already-done row; it does strictly fewer forward-row-steps. p95 improves for a
related reason: static resolves a whole chunk at once, so the last chunk's
requests all wait out the entire run, whereas continuous streams results out as
each sequence finishes. The margin here is modest because the per-step KV
reassembly (the correctness-first copying scheme) eats into the saved compute on
CPU at distilgpt2 scale; it widens with higher output-length variance and on GPU,
which is the P3 story.

## Deferred to P3

GPU and real 7B/8B models, the roofline writeup and optional vLLM reference line,
device portability (tensors still assume CPU) and `pad != eos` support (this code
relies on distilgpt2's `pad == eos == 50256`), non-greedy sampling, SSE
streaming, auth, and multi-tenancy. The decode loop stays greedy; the server
still rejects `temperature != 0` with a 400.

## License

MIT. See [LICENSE](LICENSE).
