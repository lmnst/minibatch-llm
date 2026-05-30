from __future__ import annotations

import os
import subprocess
import sys

# Must run before importing server, which reads config at import time.
os.environ["MODEL_ID"] = "fake"

from fastapi.testclient import TestClient  # noqa: E402

from server import app  # noqa: E402


def test_health():
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}


def test_generate_returns_metrics_shape():
    with TestClient(app) as client:
        r = client.post(
            "/generate",
            json={"request_id": "a", "prompt": "hello", "max_new_tokens": 8},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["request_id"] == "a"
        assert isinstance(body["text"], str)
        metrics = body["metrics"]
        assert set(metrics) == {"queue_wait_ms", "generate_ms", "e2e_ms"}
        assert all(isinstance(metrics[k], (int, float)) for k in metrics)


def test_rejects_nonzero_temperature():
    with TestClient(app) as client:
        r = client.post(
            "/generate",
            json={"request_id": "a", "prompt": "x", "temperature": 0.7},
        )
        assert r.status_code == 400


def test_rejects_nonpositive_max_new_tokens():
    with TestClient(app) as client:
        r = client.post(
            "/generate",
            json={"request_id": "a", "prompt": "x", "max_new_tokens": 0},
        )
        assert r.status_code == 422


def test_no_ml_imports_on_server_path():
    # Invariant #1 guard: importing the server stack must not pull torch or
    # transformers. Runs in a fresh interpreter so the check does not depend on
    # what the current pytest process has already imported; a model-marked test
    # in the same run loads torch, which would poison an in-process assertion.
    # Locks the swappable-runner architecture against regression.
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    probe = (
        "import sys, scheduler, server; "
        "assert 'torch' not in sys.modules, 'server path imported torch'; "
        "assert 'transformers' not in sys.modules, "
        "'server path imported transformers'"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=repo_root,
        env={**os.environ, "PYTHONPATH": repo_root, "MODEL_ID": "fake"},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
