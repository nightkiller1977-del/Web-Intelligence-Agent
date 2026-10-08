"""
End-to-end behavior tests for Web Intelligence Agent.

These tests start a real uvicorn server backed by local Ollama inference
and DuckDuckGo search, then exercise the full research lifecycle against
live external services. They are NOT meant for CI — run them locally with:

    pytest tests/test_behavior_e2e.py -m e2e -v -s

Prerequisites:
  - Ollama running at http://127.0.0.1:11434 with 'mistral' and 'nomic-embed-text' pulled
  - Network access for DuckDuckGo search and web scraping
  - pytest-timeout installed (in requirements-test.txt)
"""

import asyncio
import os
import secrets
import signal
import socket
import subprocess
import sys
import tempfile
import time
import uuid

import httpx
import pytest

E2E_AUTH_TOKEN = secrets.token_hex(32)
OLLAMA_BASE_URL = "http://127.0.0.1:11434"
RESEARCH_TIMEOUT = 1500


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _server_env(port: int) -> dict:
    env = os.environ.copy()
    env.update({
        "DEPLOYMENT_MODE": "local",
        "STORAGE_BACKEND": "local",
        "WEB_INTELLIGENCE_AUTH_TOKEN": E2E_AUTH_TOKEN,
        "HOST": "127.0.0.1",
        "PORT": str(port),
        "RETRIEVER": "duckduckgo",
        "LOCAL_DEFAULT_MODEL": "mistral",
        "LOCAL_EMBEDDING_MODEL": "nomic-embed-text",
        "OLLAMA_BASE_URL": OLLAMA_BASE_URL,
        "AI_OPENROUTER_ENABLED": "false",
        "MAX_CONCURRENT_OPS": "3",
        "MAX_MEMORY_MB": "512",
        # Isolate from ambient Brain Memory so tests never read/write the real store
        "BRAIN_MEMORY_ENABLED": "false",
        "BRAIN_MEMORY_CONTEXT_ENABLED": "false",
        "BRAIN_MEMORY_URL": "",
        "BRAIN_MEMORY_KEY_ID": "",
        "BRAIN_MEMORY_SECRET": "",
    })
    return env


@pytest.fixture(scope="module")
def e2e_server():
    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"
    log_file = tempfile.NamedTemporaryFile(
        prefix="wia_e2e_", suffix=".log", delete=False,
    )
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn",
            "app.main:app",
            "--host", "127.0.0.1",
            "--port", str(port),
        ],
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        env=_server_env(port),
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )

    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        try:
            resp = httpx.get(f"{base_url}/health/live", timeout=2)
            if resp.status_code == 200:
                break
        except httpx.ConnectError:
            pass
        time.sleep(0.5)
    else:
        proc.kill()
        proc.wait(timeout=5)
        log_file.seek(0)
        out = log_file.read().decode(errors="replace")
        log_file.close()
        os.unlink(log_file.name)
        pytest.fail(f"Server did not start within 30s. Output:\n{out}")

    yield base_url

    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    log_file.close()
    os.unlink(log_file.name)


def _auth_headers() -> dict:
    return {"Authorization": f"Bearer {E2E_AUTH_TOKEN}"}


def _make_request_body(op_id: str, query: str, *, mode: str = "quick", timeout_s: int = 600) -> dict:
    return {
        "operationId": op_id,
        "attemptId": str(uuid.uuid4()),
        "query": query,
        "mode": mode,
        "limits": {
            "maximumDurationSeconds": timeout_s,
            "maximumSearches": 5,
            "maximumPages": 10,
            "maximumSources": 5,
        },
    }


def _cancel_and_wait(base_url: str, op_id: str, timeout: int = 30) -> None:
    """Cancel an operation and wait for it to reach a terminal state."""
    try:
        httpx.post(
            f"{base_url}/v1/research/{op_id}/cancel",
            headers=_auth_headers(),
            timeout=timeout,
        )
    except httpx.HTTPError:
        pass
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            r = httpx.get(
                f"{base_url}/v1/research/{op_id}/result",
                headers=_auth_headers(),
                timeout=10,
            )
            if r.status_code == 200 and r.json().get("status") in (
                "completed", "partial", "failed", "cancelled",
            ):
                return
        except httpx.HTTPError:
            pass
        time.sleep(2)


# ---------------------------------------------------------------------------
# Fast path tests — no Ollama needed, exercise service plumbing
# ---------------------------------------------------------------------------


@pytest.mark.e2e
class TestHealthEndpoints:
    def test_liveness(self, e2e_server):
        resp = httpx.get(f"{e2e_server}/health/live", timeout=5)
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_readiness(self, e2e_server):
        resp = httpx.get(f"{e2e_server}/health/ready", timeout=5)
        body = resp.json()
        assert resp.status_code == 200
        assert body["gpt_researcher"] is True
        assert body["storage"] is True
        assert body["auth"] is True

    def test_version(self, e2e_server):
        resp = httpx.get(f"{e2e_server}/version", timeout=5)
        assert resp.status_code == 200
        body = resp.json()
        assert body["service"] == "web-intelligence-agent"
        assert "engine" in body

    def test_capabilities(self, e2e_server):
        resp = httpx.get(f"{e2e_server}/capabilities", timeout=5)
        assert resp.status_code == 200
        caps = resp.json()["capabilities"]
        assert caps["quick_search"] is True
        assert caps["cancellations"] is True
        assert caps["structured_evidence"] is True


@pytest.mark.e2e
class TestAuth:
    def test_missing_token_rejected(self, e2e_server):
        resp = httpx.post(
            f"{e2e_server}/v1/research",
            json=_make_request_body("auth-test-1", "test"),
        )
        assert resp.status_code == 401

    def test_wrong_token_rejected(self, e2e_server):
        resp = httpx.post(
            f"{e2e_server}/v1/research",
            json=_make_request_body("auth-test-2", "test"),
            headers={"Authorization": "Bearer wrong-token"},
        )
        assert resp.status_code == 401

    def test_valid_token_passes_auth(self, e2e_server):
        op_id = f"auth-pass-{uuid.uuid4().hex[:8]}"
        body = _make_request_body(op_id, "test query")
        body["idempotencyKey"] = str(uuid.uuid4())
        resp = httpx.post(
            f"{e2e_server}/v1/research",
            json=body,
            headers=_auth_headers(),
        )
        assert resp.status_code in (202, 409)
        if resp.status_code == 202:
            _cancel_and_wait(e2e_server, op_id)


@pytest.mark.e2e
class TestIdempotency:
    def test_missing_idempotency_key_rejected(self, e2e_server):
        body = _make_request_body("idemp-test-1", "test")
        resp = httpx.post(
            f"{e2e_server}/v1/research",
            json=body,
            headers=_auth_headers(),
        )
        assert resp.status_code == 400
        assert "Idempotency" in resp.json()["detail"]

    def test_duplicate_idempotency_key_returns_same_op(self, e2e_server):
        op_id = f"idemp-dup-{uuid.uuid4().hex[:8]}"
        idem_key = str(uuid.uuid4())
        body = _make_request_body(op_id, "test query")
        body["idempotencyKey"] = idem_key

        resp1 = httpx.post(
            f"{e2e_server}/v1/research",
            json=body,
            headers=_auth_headers(),
        )
        assert resp1.status_code == 202

        time.sleep(0.5)

        resp2 = httpx.post(
            f"{e2e_server}/v1/research",
            json=body,
            headers=_auth_headers(),
        )
        # 200 = idempotent hit returning existing state
        # 202 = local backend re-admitted the same operation (benign with in-memory storage)
        assert resp2.status_code in (200, 202)
        assert resp2.json()["operationId"] == op_id

        _cancel_and_wait(e2e_server, op_id)


@pytest.mark.e2e
class TestRequestValidation:
    def test_empty_query_rejected(self, e2e_server):
        body = _make_request_body("val-empty-1", "   ")
        body["idempotencyKey"] = str(uuid.uuid4())
        resp = httpx.post(
            f"{e2e_server}/v1/research",
            json=body,
            headers=_auth_headers(),
        )
        assert resp.status_code == 422

    def test_invalid_mode_rejected(self, e2e_server):
        body = _make_request_body("val-mode-1", "test query")
        body["idempotencyKey"] = str(uuid.uuid4())
        body["mode"] = "turbo"
        resp = httpx.post(
            f"{e2e_server}/v1/research",
            json=body,
            headers=_auth_headers(),
        )
        assert resp.status_code == 422

    def test_raw_llm_key_header_rejected(self, e2e_server):
        body = _make_request_body("val-llm-1", "test query")
        body["idempotencyKey"] = str(uuid.uuid4())
        headers = _auth_headers()
        headers["X-LLM-Key"] = "sk-fake"
        resp = httpx.post(
            f"{e2e_server}/v1/research",
            json=body,
            headers=headers,
        )
        assert resp.status_code == 400
        assert "Raw LLM credentials" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# Cancellation test — submits then immediately cancels
# ---------------------------------------------------------------------------


@pytest.mark.e2e
class TestCancellation:
    def test_cancel_running_operation(self, e2e_server):
        op_id = f"cancel-test-{uuid.uuid4().hex[:8]}"
        body = _make_request_body(op_id, "Explain quantum computing in detail")
        body["idempotencyKey"] = str(uuid.uuid4())
        body["limits"]["maximumDurationSeconds"] = 300

        resp = httpx.post(
            f"{e2e_server}/v1/research",
            json=body,
            headers=_auth_headers(),
        )
        assert resp.status_code == 202

        time.sleep(2)

        cancel_resp = httpx.post(
            f"{e2e_server}/v1/research/{op_id}/cancel",
            headers=_auth_headers(),
            timeout=30,
        )
        assert cancel_resp.status_code == 200
        assert cancel_resp.json()["status"] in ("cancelled", "completed", "partial", "failed")

        # Wait for the cancelled task to fully release Ollama before the next test
        time.sleep(10)

        result_resp = httpx.get(
            f"{e2e_server}/v1/research/{op_id}/result",
            headers=_auth_headers(),
            timeout=30,
        )
        assert result_resp.status_code == 200
        assert result_resp.json()["status"] in ("cancelled", "completed", "partial", "failed")


# ---------------------------------------------------------------------------
# Full research lifecycle — real Ollama + DuckDuckGo
# ---------------------------------------------------------------------------


@pytest.mark.e2e
class TestResearchLifecycle:
    @pytest.mark.timeout(RESEARCH_TIMEOUT)
    def test_full_research_produces_structured_result(self, e2e_server):
        op_id = f"e2e-lifecycle-{uuid.uuid4().hex[:8]}"
        body = _make_request_body(
            op_id,
            "What is the Python GIL?",
            timeout_s=600,
        )
        body["idempotencyKey"] = str(uuid.uuid4())
        body["requireClaimVerification"] = True

        submit_resp = httpx.post(
            f"{e2e_server}/v1/research",
            json=body,
            headers=_auth_headers(),
        )
        assert submit_resp.status_code == 202
        assert submit_resp.json()["operationId"] == op_id
        assert submit_resp.json()["status"] == "queued"

        # Poll until terminal — use a generous per-request timeout because
        # the server's event loop is slow while Ollama saturates CPU.
        deadline = time.monotonic() + RESEARCH_TIMEOUT
        start = time.monotonic()
        terminal = None
        last_status = None
        while time.monotonic() < deadline:
            try:
                poll = httpx.get(
                    f"{e2e_server}/v1/research/{op_id}/result",
                    headers=_auth_headers(),
                    timeout=60,
                )
            except httpx.ReadTimeout:
                elapsed = int(time.monotonic() - start)
                print(f"  [{elapsed}s] poll timeout, retrying...")
                time.sleep(5)
                continue
            assert poll.status_code == 200
            data = poll.json()
            status = data["status"]
            if status != last_status:
                elapsed = int(time.monotonic() - start)
                print(f"  [{elapsed}s] status={status}")
                last_status = status
            if status in ("completed", "partial", "failed", "cancelled"):
                terminal = data
                break
            time.sleep(10)

        if terminal is None:
            elapsed = int(time.monotonic() - start)
            pytest.fail(
                f"Research did not complete within {RESEARCH_TIMEOUT}s. "
                f"Last status={last_status} after {elapsed}s"
            )
        assert terminal["status"] in ("completed", "partial"), (
            f"Expected completed/partial but got {terminal['status']}: "
            f"{terminal.get('error', {}).get('message', 'no error detail')}"
        )

        # --- Structural validation ---
        assert terminal["operationId"] == op_id
        assert terminal["mode"] == "quick"

        # Answer
        assert terminal["answer"] is not None
        assert len(terminal["answer"]) > 50

        # Sources
        sources = terminal.get("sources") or []
        assert len(sources) >= 1
        for src in sources:
            assert "id" in src
            assert "url" in src
            assert src["url"].startswith("http")
            assert "title" in src
            assert "retrievedAt" in src

        # Evidence
        evidence = terminal.get("evidence") or []
        assert len(evidence) >= 1
        for ev in evidence:
            assert "id" in ev
            assert "sourceId" in ev
            assert "passage" in ev
            assert len(ev["passage"]) > 0
            assert ev["sourceId"] in {s["id"] for s in sources}

        # Claims (requested via requireClaimVerification)
        claims = terminal.get("claims") or []
        assert len(claims) >= 1
        for claim in claims:
            assert "id" in claim
            assert "text" in claim
            assert "verificationStatus" in claim

        # Citations
        citations = terminal.get("citations") or []
        assert len(citations) >= 1
        source_ids = {s["id"] for s in sources}
        for cit in citations:
            assert "id" in cit
            assert "sourceId" in cit
            assert cit["sourceId"] in source_ids

        # Metrics
        metrics = terminal.get("metrics")
        assert metrics is not None
        assert metrics["durationMs"] > 0
        assert metrics["pagesRead"] >= 1
        assert metrics["sourcesUsed"] >= 1

        # Degraded flag
        assert terminal.get("degraded") is False or terminal.get("degraded") is None

    def test_result_not_found_for_unknown_op(self, e2e_server):
        resp = httpx.get(
            f"{e2e_server}/v1/research/nonexistent-op/result",
            headers=_auth_headers(),
            timeout=30,
        )
        assert resp.status_code == 404
