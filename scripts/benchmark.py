"""Static vs continuous batching benchmark on a variable-output-length workload.

Both paths run the same fixed requests in process on CPU with distilgpt2:

- static: FCFS chunks of `capacity` through HFModelRunner.run_batch (the P1
  baseline). A chunk runs to the largest max_new_tokens in it, so short requests
  spin until their batchmates finish.
- continuous: the same requests through HFEngine, admitting up to `capacity`,
  stepping once, evicting each sequence at its own finish, and refilling freed
  slots.

All N requests are available at t0 (a saturated queue), so latency is measured
from a common start. distilgpt2 greedy never emits EOS here, so each request
delivers exactly its max_new_tokens; total useful tokens are identical across
both paths and the throughput difference is pure wall-clock.

The gate: continuous throughput must beat static AND continuous p95 latency must
be no worse than static, on the median of several runs. Run from the repo root:

    python scripts/benchmark.py
"""

from __future__ import annotations

import os
import platform
import random
import statistics
import sys
import time
from dataclasses import dataclass

# Run directly (python scripts/benchmark.py): put the repo root on the path so
# the engine and runner modules import.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import HFEngine  # noqa: E402
from model_runner import HFModelRunner  # noqa: E402
from schemas import GenerationRequest  # noqa: E402

CAPACITY = 8
N = 32
RUNS = 3
MAX_NEW_CHOICES = [8, 16, 32, 48]
SEED = 1234

# Fixed prompts of deliberately varied length (short one-liners through a long
# narrative and a code block), so prompt length is also ragged.
BASE_PROMPTS = [
    "The capital of France is",
    "import numpy as np",
    "Once upon a time, in a small village near the mountains, there lived",
    "In computer science, a hash table is a data structure that maps",
    "The quick brown fox jumps over the lazy dog and then it",
    "def fibonacci(n):\n    if n < 2:\n        return n\n    return",
]


@dataclass
class RunStats:
    tokens_per_sec: float
    p50_ms: float
    p95_ms: float


def build_workload() -> list[tuple[str, str, int]]:
    rng = random.Random(SEED)
    work = []
    for i in range(N):
        work.append((f"req{i}", rng.choice(BASE_PROMPTS), rng.choice(MAX_NEW_CHOICES)))
    return work


def _reqs(items: list[tuple[str, str, int]]) -> list[GenerationRequest]:
    return [
        GenerationRequest(request_id=rid, prompt=p, max_new_tokens=m, temperature=0.0)
        for rid, p, m in items
    ]


def percentile(values: list[float], q: float) -> float:
    s = sorted(values)
    if not s:
        return 0.0
    k = (len(s) - 1) * q / 100.0
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def run_static(runner: HFModelRunner, workload, capacity) -> RunStats:
    t0 = time.perf_counter()
    latencies = []
    total_tokens = 0
    for i in range(0, len(workload), capacity):
        chunk = workload[i : i + capacity]
        runner.run_batch(_reqs(chunk))
        done = time.perf_counter()
        for _, _, m in chunk:
            latencies.append((done - t0) * 1000.0)
            total_tokens += m  # no EOS for distilgpt2 greedy, so delivered == max_new
    wall = time.perf_counter() - t0
    return RunStats(total_tokens / wall, percentile(latencies, 50), percentile(latencies, 95))


def run_continuous(engine: HFEngine, workload, capacity) -> RunStats:
    engine.reset()
    engine.capacity = capacity
    t0 = time.perf_counter()
    latencies = []
    total_tokens = 0
    pending = list(workload)
    seqid_req = {}
    while pending or engine.num_active() > 0:
        free = capacity - engine.num_active()
        if free > 0 and pending:
            take = pending[:free]
            pending = pending[free:]
            for sid, item in zip(engine.admit(_reqs(take)), take):
                seqid_req[sid] = item[0]
        for res in engine.step():
            done = time.perf_counter()
            latencies.append((done - t0) * 1000.0)
            total_tokens += len(res.token_ids)
    wall = time.perf_counter() - t0
    return RunStats(total_tokens / wall, percentile(latencies, 50), percentile(latencies, 95))


def median_stats(runs: list[RunStats]) -> RunStats:
    return RunStats(
        statistics.median(r.tokens_per_sec for r in runs),
        statistics.median(r.p50_ms for r in runs),
        statistics.median(r.p95_ms for r in runs),
    )


def main() -> None:
    import torch
    import transformers

    workload = build_workload()
    runner = HFModelRunner("distilgpt2")
    engine = HFEngine("distilgpt2")

    # Untimed warmup so model lazy-init and allocator state do not skew run 1.
    run_static(runner, workload, CAPACITY)
    run_continuous(engine, workload, CAPACITY)

    static_runs, cont_runs = [], []
    for _ in range(RUNS):
        static_runs.append(run_static(runner, workload, CAPACITY))
        cont_runs.append(run_continuous(engine, workload, CAPACITY))
    static = median_stats(static_runs)
    cont = median_stats(cont_runs)

    print("MiniBatch-LLM benchmark: static vs continuous batching")
    print(
        f"machine: {platform.processor() or platform.machine()} | "
        f"python {platform.python_version()} | torch {torch.__version__} | "
        f"transformers {transformers.__version__}"
    )
    print(
        f"capacity={CAPACITY} N={N} runs={RUNS} (median-of-{RUNS}) "
        f"max_new in {MAX_NEW_CHOICES}, model=distilgpt2, device=cpu"
    )
    print()
    header = f"{'metric':<20}{'static':>14}{'continuous':>14}"
    print(header)
    print("-" * len(header))
    print(f"{'tokens/sec':<20}{static.tokens_per_sec:>14.1f}{cont.tokens_per_sec:>14.1f}")
    print(f"{'p50 latency (ms)':<20}{static.p50_ms:>14.1f}{cont.p50_ms:>14.1f}")
    print(f"{'p95 latency (ms)':<20}{static.p95_ms:>14.1f}{cont.p95_ms:>14.1f}")
    print()
    gain = cont.tokens_per_sec / static.tokens_per_sec if static.tokens_per_sec else 0.0
    print(f"throughput gain: {gain:.2f}x")

    pass_tp = cont.tokens_per_sec > static.tokens_per_sec
    pass_p95 = cont.p95_ms <= static.p95_ms
    ok = pass_tp and pass_p95
    print(
        f"gate: continuous throughput > static ({pass_tp}) AND "
        f"continuous p95 <= static p95 ({pass_p95})"
    )
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
