"""Concurrent smoke test for the P0 server.

Starts uvicorn in a subprocess, fires several different requests at once,
checks that each response carries its own request_id (no cross-talk) and a
full set of metrics, then shuts the server down. Run from the repo root:

    python scripts/smoke.py
"""

from __future__ import annotations

import concurrent.futures
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:8000"
URL = BASE + "/generate"

# (request_id, prompt, max_new_tokens)
REQUESTS = [
    ("q0", "The capital of France is", 16),
    ("q1", "Once upon a time, there was", 24),
    ("q2", "Water boils at a temperature of", 12),
    ("q3", "The first president of the United States was", 16),
    ("q4", "def add(a, b):\n    return", 12),
    ("q5", "Roses are red, violets are", 10),
]


def wait_ready(proc: subprocess.Popen, timeout: float = 300.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise SystemExit(
                f"server exited before becoming ready (code {proc.returncode})"
            )
        try:
            with urllib.request.urlopen(BASE + "/health", timeout=5) as r:
                if r.status == 200:
                    return
        except (urllib.error.URLError, ConnectionError, OSError):
            time.sleep(2.0)
    raise SystemExit("server did not become ready in time")


def call(req_id: str, prompt: str, max_new_tokens: int) -> dict:
    body = json.dumps(
        {
            "request_id": req_id,
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
            "temperature": 0.0,
        }
    ).encode()
    req = urllib.request.Request(
        URL, data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())


def run_checks() -> bool:
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(REQUESTS)) as ex:
        futs = {ex.submit(call, rid, p, n): rid for rid, p, n in REQUESTS}
        results: dict[str, dict] = {}
        for fut in concurrent.futures.as_completed(futs):
            results[futs[fut]] = fut.result()

    ok = True
    for rid, prompt, _ in REQUESTS:
        res = results[rid]
        m = res.get("metrics", {})
        id_match = res.get("request_id") == rid
        metrics_ok = all(
            isinstance(m.get(k), (int, float))
            for k in ("queue_wait_ms", "generate_ms", "e2e_ms")
        )
        text_ok = isinstance(res.get("text"), str)
        ok = ok and id_match and metrics_ok and text_ok
        print(f"[{rid}] id_match={id_match} metrics_ok={metrics_ok}")
        print(f"      prompt={prompt!r}")
        print(f"      text={res.get('text')!r}")
        if metrics_ok:
            print(
                f"      queue_wait_ms={m['queue_wait_ms']:.2f} "
                f"generate_ms={m['generate_ms']:.2f} e2e_ms={m['e2e_ms']:.2f}"
            )
    ok = ok and sorted(results.keys()) == sorted(r[0] for r in REQUESTS)
    return ok


def main() -> None:
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn", "server:app",
            "--host", "127.0.0.1", "--port", "8000", "--log-level", "warning",
        ]
    )
    try:
        wait_ready(proc)
        ok = run_checks()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
    print("SMOKE_OK" if ok else "SMOKE_FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
