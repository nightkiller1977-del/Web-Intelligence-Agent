import asyncio
import json
import time

import httpx
import pytest
from fastapi.testclient import TestClient

import app.api as api
from app.config import settings
from app.main import app


def _payload(operation_id="op-api", idempotency_key="idem-api"):
    return {
        "operationId": operation_id,
        "idempotencyKey": idempotency_key,
        "attemptId": f"attempt-{operation_id}",
        "query": "What changed in this adapter?",
        "mode": "quick",
        "profile": "general",
        "limits": {
            "maximumDurationSeconds": 30,
            "maximumSearches": 1,
            "maximumPages": 1,
            "maximumSources": 1,
            "maximumMemoryMb": 256,
        },
    }


def _auth_headers():
    return {"Authorization": f"Bearer {settings.AUTH_TOKEN}"}


def _allow_test_gateway(monkeypatch):
    """Keep tests for unrelated gateway behavior independent of DNS."""
    monkeypatch.setattr(api, "is_gateway_destination_allowed", lambda _gateway: True)


@pytest.fixture(autouse=True)
def reset_storage_state():
    for attr in ("operations", "events", "idempotency_keys", "operation_claims", "operation_owners"):
        if hasattr(api.storage, attr):
            getattr(api.storage, attr).clear()
    if hasattr(api.storage, "_concurrency_slots"):
        api.storage._concurrency_slots.clear()
    api.cancellation_manager.active_tasks.clear()


def _wait_for_terminal_result(client, operation_id):
    for _ in range(50):
        response = client.get(f"/v1/research/{operation_id}/result", headers=_auth_headers())
        assert response.status_code == 200
        body = response.json()
        if body["status"] in api.TERMINAL_STATUSES:
            return body
        time.sleep(0.02)
    raise AssertionError("operation did not finish")


async def _collect_sse_events(client, operation_id, headers):
    """Reads the SSE stream for operation_id to completion and returns the
    decoded `data:` payloads in the order they were received."""
    events = []
    async with client.stream("GET", f"/v1/research/{operation_id}/events", headers=headers) as response:
        assert response.status_code == 200
        async for line in response.aiter_lines():
            if not line.startswith("data:"):
                continue
            payload = line[len("data:"):].strip()
            if payload:
                events.append(json.loads(payload))
    return events


def test_capabilities_do_not_advertise_unimplemented_contract_features():
    with TestClient(app) as client:
        response = client.get("/capabilities")

    assert response.status_code == 200
    capabilities = response.json()["capabilities"]
    assert capabilities["source_level_citations"] is True
    assert capabilities["citations"] is True
    assert capabilities["structured_evidence"] is True
    assert capabilities["claim_verification"] is True
    assert capabilities["source_policy"] is True
    assert capabilities["model_budget_limits"] is True
    assert capabilities["model_preferences"] is True


def test_docs_require_authentication_by_default():
    with TestClient(app) as client:
        response = client.get("/docs")

    assert response.status_code == 401


def test_docs_can_be_exposed_for_local_browser_testing(monkeypatch):
    monkeypatch.setattr(settings, "DEPLOYMENT_MODE", "local")
    monkeypatch.setattr(settings, "ALLOW_UNAUTHENTICATED_DOCS", True)

    with TestClient(app) as client:
        docs = client.get("/docs")
        openapi = client.get("/openapi.json")

    assert docs.status_code == 200
    assert openapi.status_code == 200


def test_research_submission_completes_with_mocked_adapter(monkeypatch):
    async def fake_conduct_web_research(**kwargs):
        await kwargs["reporter"].report("completed", "Mock research completed.")
        return {
            "operationId": kwargs["op_id"],
            "status": "completed",
            "mode": kwargs["mode"],
            "profile": kwargs["profile"],
            "answer": "Mock answer",
            "sources": [],
            "evidence": [],
            "claims": [],
            "citations": [],
            "searchesPerformed": [],
            "metrics": {
                "startedAt": "2026-01-01T00:00:00+00:00",
                "completedAt": "2026-01-01T00:00:01+00:00",
                "durationMs": 1,
                "searchesPerformed": 0,
                "pagesRead": 0,
                "sourcesConsidered": 0,
                "sourcesUsed": 0,
            },
        }

    monkeypatch.setattr(api, "conduct_web_research", fake_conduct_web_research)
    operation_id = "op-api-complete"

    with TestClient(app) as client:
        response = client.post("/v1/research", json=_payload(operation_id), headers=_auth_headers())
        assert response.status_code == 202
        assert response.json() == {"operationId": operation_id, "status": "queued"}

        result = _wait_for_terminal_result(client, operation_id)
        events = client.get(f"/v1/research/{operation_id}/events", headers=_auth_headers())

    assert result["status"] == "completed"
    assert result["answer"] == "Mock answer"
    assert events.status_code == 200


def test_research_submission_returns_passage_backed_claims(monkeypatch):
    async def fake_conduct_web_research(**kwargs):
        return {
            "operationId": kwargs["op_id"],
            "status": "completed",
            "mode": kwargs["mode"],
            "profile": kwargs["profile"],
            "answer": "Mock answer",
            "sources": [
                {
                    "id": "src-op-passage-0",
                    "url": "https://example.com/a",
                    "title": "Example A",
                    "retrievedAt": 1,
                    "sourceType": "web",
                }
            ],
            "evidence": [
                {
                    "id": "ev-op-passage-0",
                    "sourceId": "src-op-passage-0",
                    "passage": "This source passage directly supports the returned claim.",
                    "relevanceScore": 0.9,
                }
            ],
            "claims": [
                {
                    "id": "claim-op-passage-0",
                    "text": "This source passage directly supports the returned claim.",
                    "evidenceIds": ["ev-op-passage-0"],
                    "confidence": 0.85,
                    "verificationStatus": "supported",
                }
            ],
            "citations": [
                {
                    "id": "cite-op-passage-0",
                    "sourceId": "src-op-passage-0",
                    "evidenceIds": ["ev-op-passage-0"],
                    "claimIds": ["claim-op-passage-0"],
                }
            ],
            "searchesPerformed": [],
            "metrics": {
                "startedAt": "2026-01-01T00:00:00+00:00",
                "completedAt": "2026-01-01T00:00:01+00:00",
                "durationMs": 1,
                "searchesPerformed": 0,
                "pagesRead": 1,
                "sourcesConsidered": 1,
                "sourcesUsed": 1,
            },
        }

    monkeypatch.setattr(api, "conduct_web_research", fake_conduct_web_research)
    operation_id = "op-api-passage"

    with TestClient(app) as client:
        response = client.post("/v1/research", json=_payload(operation_id), headers=_auth_headers())
        result = _wait_for_terminal_result(client, operation_id)

    assert response.status_code == 202
    assert result["claims"][0]["verificationStatus"] == "supported"
    assert result["claims"][0]["evidenceIds"] == [result["evidence"][0]["id"]]
    assert result["citations"][0]["claimIds"] == [result["claims"][0]["id"]]


def test_research_submission_reuses_idempotency_key(monkeypatch):
    async def fake_conduct_web_research(**kwargs):
        return {
            "operationId": kwargs["op_id"],
            "status": "completed",
            "mode": kwargs["mode"],
            "profile": kwargs["profile"],
            "answer": "Mock answer",
            "sources": [],
            "evidence": [],
            "claims": [],
            "citations": [],
            "searchesPerformed": [],
            "metrics": {
                "startedAt": "2026-01-01T00:00:00+00:00",
                "completedAt": "2026-01-01T00:00:01+00:00",
                "durationMs": 1,
                "searchesPerformed": 0,
                "pagesRead": 0,
                "sourcesConsidered": 0,
                "sourcesUsed": 0,
            },
        }

    monkeypatch.setattr(api, "conduct_web_research", fake_conduct_web_research)
    operation_id = "op-api-idem"

    with TestClient(app) as client:
        first = client.post("/v1/research", json=_payload(operation_id, "same-key"), headers=_auth_headers())
        second = client.post("/v1/research", json=_payload("different-op", "same-key"), headers=_auth_headers())

    assert first.status_code == 202
    assert second.status_code == 202
    assert second.json()["operationId"] == operation_id


def test_idempotency_retry_does_not_report_unpersisted_admission_as_accepted():
    lookup_key = "pending-admission-key"
    operation_id = "op-pending-admission"
    asyncio.run(
        api.storage.claim_idempotency_key(
            lookup_key, operation_id, "pending-admission"
        )
    )

    with TestClient(app) as client:
        retry = client.post(
            "/v1/research",
            json=_payload("different-op-id", lookup_key),
            headers=_auth_headers(),
        )

    assert retry.status_code == 409
    assert retry.headers["retry-after"] == "1"
    assert "pending" in retry.json()["detail"].lower()


def test_submission_survives_initial_progress_event_failure(monkeypatch):
    async def unavailable_progress_stream(*_args, **_kwargs):
        raise RuntimeError("progress stream unavailable")

    monkeypatch.setattr(api.ProgressReporter, "report", unavailable_progress_stream)

    with TestClient(app) as client:
        response = client.post(
            "/v1/research",
            json=_payload("op-progress-unavailable", "idem-progress-unavailable"),
            headers=_auth_headers(),
        )

    assert response.status_code == 202


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("remaining_owner", "expected_calls"),
    [
        ("op-stale", ["idempotency"]),
        (
            "op-winner",
            ["idempotency", "owner", "operation", "slot", "spend"],
        ),
    ],
)
async def test_rollback_cleans_only_resources_not_owned_by_the_key_winner(
    monkeypatch, remaining_owner, expected_calls
):
    calls = []

    def recorded(name, result=None):
        async def call(*_args):
            calls.append(name)
            return result

        return call

    monkeypatch.setattr(
        api.storage,
        "release_idempotency_key",
        recorded("idempotency", remaining_owner),
    )
    monkeypatch.setattr(
        api.storage, "release_operation_lease", recorded("owner")
    )
    monkeypatch.setattr(
        api.storage, "release_operation_id", recorded("operation")
    )
    monkeypatch.setattr(
        api.storage, "release_concurrency_slot", recorded("slot")
    )
    monkeypatch.setattr(
        api.storage, "release_daily_spend", recorded("spend")
    )

    await api._rollback_admission(
        "op-stale",
        "idem-reclaimed",
        "stale-admission-token",
        True,
        True,
        True,
        True,
        0.25,
    )

    assert calls == expected_calls


def test_idempotency_retry_resolves_before_gateway_admission_checks(monkeypatch):
    # A retry of already-accepted work must resolve to its existing operation
    # even when a later admission check (here, gateway configuration that
    # became invalid after the first request -- e.g. a rolling restart or a
    # replica missing its secret) would now reject a *new* request. Those
    # checks only matter for new work; applying them to a retry would turn an
    # ambiguous submission into a spurious rejection instead of letting the
    # client reconcile with its persisted operation.
    async def fake_conduct_web_research(**kwargs):
        return {
            "operationId": kwargs["op_id"],
            "status": "completed",
            "mode": kwargs["mode"],
            "profile": kwargs["profile"],
            "answer": "Mock answer",
            "sources": [], "evidence": [], "claims": [], "citations": [], "searchesPerformed": [],
            "metrics": {
                "startedAt": "2026-01-01T00:00:00+00:00",
                "completedAt": "2026-01-01T00:00:01+00:00",
                "durationMs": 1,
                "searchesPerformed": 0, "pagesRead": 0, "sourcesConsidered": 0, "sourcesUsed": 0,
            },
        }

    monkeypatch.setattr(api, "conduct_web_research", fake_conduct_web_research)
    monkeypatch.setattr(settings, "AI_OPENROUTER_ENABLED", True)
    monkeypatch.setattr(settings, "AI_OPENROUTER_BASE_URL", "https://gateway.example")
    monkeypatch.setattr(settings, "AI_OPENROUTER_API_KEY", "test-gateway-key")
    _allow_test_gateway(monkeypatch)

    operation_id = "op-idem-retry-gateway"
    payload = _payload(operation_id, "same-retry-key")
    payload["model_provider"] = "openai"
    payload["model_name"] = "gpt-4o-mini"

    with TestClient(app) as client:
        first = client.post("/v1/research", json=payload, headers=_auth_headers())
        assert first.status_code == 202

        # Simulate the gateway secret becoming unavailable (e.g. a rolling
        # restart hit a replica missing it) before the client retries.
        monkeypatch.setattr(settings, "AI_OPENROUTER_API_KEY", "")

        retry_payload = _payload("different-op-id", "same-retry-key")
        retry_payload["model_provider"] = "openai"
        retry_payload["model_name"] = "gpt-4o-mini"
        retry = client.post("/v1/research", json=retry_payload, headers=_auth_headers())

    assert retry.status_code == 202
    assert retry.json()["operationId"] == operation_id


def test_source_policy_allowed_domains_is_passed_to_adapter(monkeypatch):
    observed_source_policy = None

    async def fake_conduct_web_research(**kwargs):
        nonlocal observed_source_policy
        observed_source_policy = kwargs["source_policy"]
        return {
            "operationId": kwargs["op_id"],
            "status": "completed",
            "mode": kwargs["mode"],
            "profile": kwargs["profile"],
            "answer": "Mock answer",
            "sources": [],
            "evidence": [],
            "claims": [],
            "citations": [],
            "searchesPerformed": [],
            "metrics": {
                "startedAt": "2026-01-01T00:00:00+00:00",
                "completedAt": "2026-01-01T00:00:01+00:00",
                "durationMs": 1,
                "searchesPerformed": 0,
                "pagesRead": 0,
                "sourcesConsidered": 0,
                "sourcesUsed": 0,
            },
        }

    monkeypatch.setattr(api, "conduct_web_research", fake_conduct_web_research)
    payload = _payload("op-source-policy")
    payload["sourcePolicy"] = {"allowedDomains": ["example.com"]}

    with TestClient(app) as client:
        response = client.post("/v1/research", json=payload, headers=_auth_headers())
        result = _wait_for_terminal_result(client, "op-source-policy")

    assert response.status_code == 202
    assert result["status"] == "completed"
    assert observed_source_policy == {"allowedDomains": ["example.com"]}


def test_freshness_and_inputs_are_passed_to_adapter(monkeypatch, tmp_path):
    observed = {}

    async def fake_conduct_web_research(**kwargs):
        observed["freshness"] = kwargs["freshness"]
        observed["inputs"] = kwargs["inputs"]
        return {
            "operationId": kwargs["op_id"],
            "status": "completed",
            "mode": kwargs["mode"],
            "profile": kwargs["profile"],
            "answer": "Mock answer",
            "sources": [],
            "evidence": [],
            "claims": [],
            "citations": [],
            "searchesPerformed": [],
            "metrics": {
                "startedAt": "2026-01-01T00:00:00+00:00",
                "completedAt": "2026-01-01T00:00:01+00:00",
                "durationMs": 1,
                "searchesPerformed": 0,
                "pagesRead": 0,
                "sourcesConsidered": 0,
                "sourcesUsed": 0,
            },
        }

    monkeypatch.setattr(api, "conduct_web_research", fake_conduct_web_research)
    document = tmp_path / "notes.md"
    document.write_text("local context", encoding="utf-8")
    payload = _payload("op-fresh-inputs")
    payload["freshness"] = {"since": "2026-08-01"}
    payload["inputs"] = {"documents": [{"path": str(document), "displayName": "Notes"}]}

    with TestClient(app) as client:
        response = client.post("/v1/research", json=payload, headers=_auth_headers())
        result = _wait_for_terminal_result(client, "op-fresh-inputs")

    assert response.status_code == 202
    assert result["status"] == "completed"
    assert observed["freshness"] == {"since": "2026-08-01"}
    assert observed["inputs"]["documents"][0]["displayName"] == "Notes"


def test_unknown_source_policy_fields_are_rejected():
    payload = _payload("op-source-policy-bad")
    payload["sourcePolicy"] = {"deniedDomains": ["example.com"]}

    with TestClient(app) as client:
        response = client.post("/v1/research", json=payload, headers=_auth_headers())

    assert response.status_code == 400
    assert "Unsupported sourcePolicy fields" in response.json()["detail"]


def test_model_budget_limits_are_passed_to_adapter(monkeypatch):
    observed_limits = None

    async def fake_conduct_web_research(**kwargs):
        nonlocal observed_limits
        observed_limits = kwargs["limits"]
        return {
            "operationId": kwargs["op_id"],
            "status": "completed",
            "mode": kwargs["mode"],
            "profile": kwargs["profile"],
            "answer": "Mock answer",
            "sources": [],
            "evidence": [],
            "claims": [],
            "citations": [],
            "searchesPerformed": [],
            "metrics": {
                "startedAt": "2026-01-01T00:00:00+00:00",
                "completedAt": "2026-01-01T00:00:01+00:00",
                "durationMs": 1,
                "searchesPerformed": 0,
                "pagesRead": 0,
                "sourcesConsidered": 0,
                "sourcesUsed": 0,
            },
        }

    monkeypatch.setattr(api, "conduct_web_research", fake_conduct_web_research)
    payload = _payload("op-model-budget")
    payload["limits"]["maximumModelTokens"] = 1000
    payload["limits"]["maximumModelCostUsd"] = 0.25

    with TestClient(app) as client:
        response = client.post("/v1/research", json=payload, headers=_auth_headers())
        result = _wait_for_terminal_result(client, "op-model-budget")

    assert response.status_code == 202
    assert result["status"] == "completed"
    assert observed_limits["maximumModelTokens"] == 1000
    assert observed_limits["maximumModelCostUsd"] == 0.25


def test_model_preferences_are_passed_to_adapter(monkeypatch):
    observed_model = None

    async def fake_conduct_web_research(**kwargs):
        nonlocal observed_model
        observed_model = (kwargs["model_provider"], kwargs["model_name"])
        return {
            "operationId": kwargs["op_id"],
            "status": "completed",
            "mode": kwargs["mode"],
            "profile": kwargs["profile"],
            "answer": "Mock answer",
            "sources": [],
            "evidence": [],
            "claims": [],
            "citations": [],
            "searchesPerformed": [],
            "metrics": {
                "startedAt": "2026-01-01T00:00:00+00:00",
                "completedAt": "2026-01-01T00:00:01+00:00",
                "durationMs": 1,
                "searchesPerformed": 0,
                "pagesRead": 0,
                "sourcesConsidered": 0,
                "sourcesUsed": 0,
            },
        }

    monkeypatch.setattr(api, "conduct_web_research", fake_conduct_web_research)
    monkeypatch.setattr(settings, "AI_OPENROUTER_ENABLED", True)
    monkeypatch.setattr(settings, "AI_OPENROUTER_BASE_URL", "https://gateway.example")
    monkeypatch.setattr(settings, "AI_OPENROUTER_API_KEY", "test-gateway-key")
    _allow_test_gateway(monkeypatch)
    payload = _payload("op-model-preferences")
    payload["model_provider"] = "openai"
    payload["model_name"] = "gpt-4o-mini"

    with TestClient(app) as client:
        response = client.post("/v1/research", json=payload, headers=_auth_headers())
        result = _wait_for_terminal_result(client, "op-model-preferences")

    assert response.status_code == 202
    assert result["status"] == "completed"
    assert observed_model == ("openai", "gpt-4o-mini")


def test_incomplete_model_preferences_are_rejected():
    payload = _payload("op-model-preferences-bad")
    payload["model_provider"] = "openai"

    with TestClient(app) as client:
        response = client.post("/v1/research", json=payload, headers=_auth_headers())

    assert response.status_code == 400
    assert "must be provided together" in response.json()["detail"]


def test_external_model_selection_requires_the_shared_gateway(monkeypatch):
    monkeypatch.setattr(settings, "AI_OPENROUTER_ENABLED", False)
    payload = _payload("op-external-model-without-gateway")
    payload["model_provider"] = "openai"
    payload["model_name"] = "gpt-4o-mini"

    with TestClient(app) as client:
        response = client.post("/v1/research", json=payload, headers=_auth_headers())

    assert response.status_code == 503
    assert "AI-OpenRouter" in response.json()["detail"]


def test_raw_llm_credentials_are_rejected_even_in_local_mode(monkeypatch):
    monkeypatch.setattr(settings, "DEPLOYMENT_MODE", "local")
    payload = _payload("op-raw-llm-credential")
    headers = _auth_headers() | {"X-LLM-Key": "direct-provider-key"}

    with TestClient(app) as client:
        response = client.post("/v1/research", json=payload, headers=headers)

    assert response.status_code == 400
    assert "AI-OpenRouter" in response.json()["detail"]


def test_invalid_gateway_configuration_fails_closed(monkeypatch):
    monkeypatch.setattr(settings, "AI_OPENROUTER_ENABLED", True)
    monkeypatch.setattr(settings, "AI_OPENROUTER_BASE_URL", "http://gateway.example")
    monkeypatch.setattr(settings, "AI_OPENROUTER_API_KEY", "test-gateway-key")

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post("/v1/research", json=_payload("op-invalid-gateway"), headers=_auth_headers())

    assert response.status_code == 503
    assert "AI-OpenRouter" in response.json()["detail"]


def test_gateway_destination_blocked_by_egress_policy_fails_closed(monkeypatch):
    monkeypatch.setattr(settings, "AI_OPENROUTER_ENABLED", True)
    monkeypatch.setattr(settings, "AI_OPENROUTER_BASE_URL", "https://127.0.0.1")
    monkeypatch.setattr(settings, "AI_OPENROUTER_API_KEY", "test-gateway-key")
    payload = _payload("op-blocked-gateway")
    payload["model_provider"] = "openai"
    payload["model_name"] = "gpt-4o-mini"

    with TestClient(app) as client:
        response = client.post("/v1/research", json=payload, headers=_auth_headers())

    assert response.status_code == 503
    assert "AI-OpenRouter" in response.json()["detail"]


def test_invalid_gateway_configuration_does_not_block_local_requests(monkeypatch):
    # A broken cloud gateway config must not disable local-first operation: a
    # local model provider needs no external egress, so it must still be admitted.
    monkeypatch.setattr(settings, "AI_OPENROUTER_ENABLED", True)
    monkeypatch.setattr(settings, "AI_OPENROUTER_BASE_URL", "http://gateway.example")
    monkeypatch.setattr(settings, "AI_OPENROUTER_API_KEY", "test-gateway-key")
    payload = _payload("op-invalid-gateway-local-model")
    payload["model_provider"] = "ollama"
    payload["model_name"] = "llama3"

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post("/v1/research", json=payload, headers=_auth_headers())

    assert response.status_code == 202


def test_cancel_unknown_operation_returns_not_found():
    with TestClient(app) as client:
        response = client.post("/v1/research/op-missing/cancel", headers=_auth_headers())

    assert response.status_code == 404


def test_metrics_endpoint_is_public():
    with TestClient(app) as client:
        response = client.get("/metrics")

    assert response.status_code == 200
    assert "research_operations_total" in response.text


@pytest.mark.anyio
async def test_sse_stream_emits_events_in_pushed_order():
    """The SSE endpoint must replay progress events in the exact order they
    were pushed to storage, not just prove the endpoint responds."""
    operation_id = "op-sse-order"
    stages = ["planning", "searching", "reading", "synthesizing", "completed"]

    await api.storage.save_operation(operation_id, {"operationId": operation_id, "status": "completed"})
    for stage in stages:
        await api.storage.push_progress_event(operation_id, {"stage": stage, "message": f"{stage}-message"})

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        events = await asyncio.wait_for(
            _collect_sse_events(client, operation_id, _auth_headers()),
            timeout=2,
        )

    assert [e["stage"] for e in events] == stages


@pytest.mark.anyio
async def test_sse_stream_closes_after_terminal_status():
    """The generator has a `while True` loop gated on the operation reaching
    a terminal status. This wraps the read in asyncio.wait_for specifically
    so a broken terminal-status check (infinite loop) fails the test instead
    of hanging the whole suite."""
    operation_id = "op-sse-closes"
    await api.storage.save_operation(operation_id, {"operationId": operation_id, "status": "running"})
    await api.storage.push_progress_event(operation_id, {"stage": "planning", "message": "start"})

    async def flip_to_terminal_after_delay():
        await asyncio.sleep(0.6)  # let the stream observe one non-terminal poll first
        await api.storage.save_operation(operation_id, {"status": "completed"})

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        flipper = asyncio.create_task(flip_to_terminal_after_delay())
        try:
            events = await asyncio.wait_for(
                _collect_sse_events(client, operation_id, _auth_headers()),
                timeout=2,
            )
        finally:
            await flipper

    assert any(e["stage"] == "planning" for e in events)


@pytest.mark.anyio
async def test_sse_stream_flushes_residual_events_before_close(monkeypatch):
    """Exercises the generator's post-terminal-status event fetch - the one
    made *after* it observes a terminal status, specifically to flush any
    event that landed in the gap between the loop's first fetch and its
    terminal-status check. A fake storage.get_progress_events_after with a
    call-counting side effect deterministically reproduces that gap instead
    of relying on a real timing race."""
    operation_id = "op-sse-residual"
    await api.storage.save_operation(operation_id, {"operationId": operation_id, "status": "completed"})

    first_batch = [{"stage": "planning", "message": "first"}]
    residual_batch = first_batch + [{"stage": "completed", "message": "residual"}]
    call_count = {"n": 0}

    async def fake_get_progress_events_after(op_id, cursor):
        call_count["n"] += 1
        events = first_batch if call_count["n"] == 1 else residual_batch
        return [(str(index), event) for index, event in enumerate(events, 1) if cursor is None or index > int(cursor)]

    monkeypatch.setattr(api.storage, "get_progress_events_after", fake_get_progress_events_after)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        events = await asyncio.wait_for(
            _collect_sse_events(client, operation_id, _auth_headers()),
            timeout=2,
        )

    assert [e["message"] for e in events] == ["first", "residual"]
    assert call_count["n"] >= 2, "the terminal-status branch must re-fetch events to flush residual ones"


@pytest.mark.anyio
async def test_cancel_actually_cancels_running_background_task(monkeypatch):
    """Proves cancellation isn't just "endpoint doesn't 401": the background
    research task must actually receive CancelledError and the operation
    must actually transition to status=cancelled."""
    started = asyncio.Event()
    cancelled_seen = asyncio.Event()

    async def fake_conduct_web_research(**kwargs):
        started.set()
        try:
            await asyncio.Event().wait()  # blocks until this task is cancelled
        except asyncio.CancelledError:
            cancelled_seen.set()
            raise

    monkeypatch.setattr(api, "conduct_web_research", fake_conduct_web_research)
    operation_id = "op-cancel-real"

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/v1/research", json=_payload(operation_id), headers=_auth_headers()
        )
        assert response.status_code == 202

        await asyncio.wait_for(started.wait(), timeout=2)

        cancel_response = await client.post(
            f"/v1/research/{operation_id}/cancel", headers=_auth_headers()
        )
        assert cancel_response.status_code == 200
        assert cancel_response.json() == {"operationId": operation_id, "status": "cancelled"}

        await asyncio.wait_for(cancelled_seen.wait(), timeout=2)

        result_response = await client.get(
            f"/v1/research/{operation_id}/result", headers=_auth_headers()
        )

    result = result_response.json()
    assert result["status"] == "cancelled"
    assert result["answer"] == "Operation was cancelled by the client."


@pytest.mark.anyio
async def test_cancel_refuses_already_terminal_operation(monkeypatch):
    """Cancelling an operation that already finished must not flip its
    status to "cancelled" - it should just report the existing terminal
    status untouched."""

    async def fake_conduct_web_research(**kwargs):
        return {
            "operationId": kwargs["op_id"],
            "status": "completed",
            "mode": kwargs["mode"],
            "profile": kwargs["profile"],
            "answer": "Already done",
            "sources": [],
            "evidence": [],
            "claims": [],
            "citations": [],
            "searchesPerformed": [],
            "metrics": {
                "startedAt": "2026-01-01T00:00:00+00:00",
                "completedAt": "2026-01-01T00:00:01+00:00",
                "durationMs": 1,
                "searchesPerformed": 0,
                "pagesRead": 0,
                "sourcesConsidered": 0,
                "sourcesUsed": 0,
            },
        }

    monkeypatch.setattr(api, "conduct_web_research", fake_conduct_web_research)
    operation_id = "op-cancel-terminal"

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/v1/research", json=_payload(operation_id), headers=_auth_headers()
        )
        assert response.status_code == 202

        result = None
        for _ in range(50):
            r = await client.get(f"/v1/research/{operation_id}/result", headers=_auth_headers())
            body = r.json()
            if body["status"] in api.TERMINAL_STATUSES:
                result = body
                break
            await asyncio.sleep(0.02)
        assert result is not None and result["status"] == "completed"

        cancel_response = await client.post(
            f"/v1/research/{operation_id}/cancel", headers=_auth_headers()
        )

    assert cancel_response.status_code == 200
    assert cancel_response.json() == {"operationId": operation_id, "status": "completed"}


def _completed_result(op_id, **overrides):
    result = {
        "operationId": op_id,
        "status": "completed",
        "mode": "quick",
        "profile": "general",
        "answer": "Mock answer",
        "sources": [], "evidence": [], "claims": [], "citations": [],
        "searchesPerformed": [],
        "metrics": {
            "startedAt": "2026-01-01T00:00:00+00:00",
            "completedAt": "2026-01-01T00:00:01+00:00",
            "durationMs": 1,
            "searchesPerformed": 0, "pagesRead": 0,
            "sourcesConsidered": 0, "sourcesUsed": 0,
        },
    }
    result.update(overrides)
    return result


def test_result_reports_requires_reconciliation_for_failed_operation(monkeypatch):
    async def failing_research(**kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(api, "conduct_web_research", failing_research)

    with TestClient(app) as client:
        assert client.post(
            "/v1/research", json=_payload("op-reconcile"), headers=_auth_headers()
        ).status_code == 202
        result = _wait_for_terminal_result(client, "op-reconcile")

    assert result["status"] == "failed"
    assert result["requiresReconciliation"] is True


def test_result_reports_no_reconciliation_for_completed_operation(monkeypatch):
    async def ok_research(**kwargs):
        return _completed_result(kwargs["op_id"])

    monkeypatch.setattr(api, "conduct_web_research", ok_research)

    with TestClient(app) as client:
        assert client.post(
            "/v1/research", json=_payload("op-no-reconcile"), headers=_auth_headers()
        ).status_code == 202
        result = _wait_for_terminal_result(client, "op-no-reconcile")

    assert result["status"] == "completed"
    assert result["requiresReconciliation"] is False


def test_result_reports_degraded_when_adapter_marks_it(monkeypatch):
    async def degraded_research(**kwargs):
        return _completed_result(
            kwargs["op_id"],
            degraded=True,
            degradedReasons=["No source passage text was available."],
        )

    monkeypatch.setattr(api, "conduct_web_research", degraded_research)

    with TestClient(app) as client:
        assert client.post(
            "/v1/research", json=_payload("op-degraded"), headers=_auth_headers()
        ).status_code == 202
        result = _wait_for_terminal_result(client, "op-degraded")

    assert result["degraded"] is True
    assert result["degradedReasons"]


def test_new_operations_rejected_once_daily_spend_limit_reached(monkeypatch):
    async def ok_research(**kwargs):
        return _completed_result(kwargs["op_id"])

    monkeypatch.setattr(api, "conduct_web_research", ok_research)
    # Drive the authoritative shared ceiling: a limit below the per-operation
    # reserve makes the atomic admission reservation fail without real research.
    monkeypatch.setattr(settings, "DAILY_SPEND_LIMIT_USD", 0.01)

    with TestClient(app) as client:
        response = client.post(
            "/v1/research", json=_payload("op-spend"), headers=_auth_headers()
        )

    assert response.status_code == 429
    assert "spend limit" in response.json()["detail"].lower()


def test_failed_operation_still_reports_input_limitations(monkeypatch):
    async def failing_research(**kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(api, "conduct_web_research", failing_research)
    payload = _payload("op-fail-limits")
    payload["inputs"] = {"documents": [{"path": "/tmp/notes.md", "displayName": "Notes"}]}

    with TestClient(app) as client:
        assert client.post(
            "/v1/research", json=payload, headers=_auth_headers()
        ).status_code == 202
        result = _wait_for_terminal_result(client, "op-fail-limits")

    assert result["status"] == "failed"
    assert any("Local inputs" in item for item in result["limitations"])



@pytest.mark.parametrize(
    ("gateway_enabled", "gateway_url"),
    [
        (False, ""),
        (True, "http://gateway.example"),
        (True, "https://gateway.example"),
    ],
)
def test_unsupported_model_provider_is_rejected_before_gateway_validation(
    monkeypatch, gateway_enabled, gateway_url
):
    monkeypatch.setattr(settings, "AI_OPENROUTER_ENABLED", gateway_enabled)
    monkeypatch.setattr(settings, "AI_OPENROUTER_BASE_URL", gateway_url)
    monkeypatch.setattr(settings, "AI_OPENROUTER_API_KEY", "test-gateway-key")
    payload = _payload("op-gateway-provider")
    payload["model_provider"] = "anthropic"
    payload["model_name"] = "claude-3"

    with TestClient(app) as client:
        response = client.post("/v1/research", json=payload, headers=_auth_headers())

    assert response.status_code == 400
    assert "OpenAI-compatible" in response.json()["detail"]


def test_search_credentials_are_rejected_in_remote_mode(monkeypatch):
    # A remote caller must not be told its X-Search-Key was honored when remote
    # mode silently ignores request-scoped credentials.
    monkeypatch.setattr(settings, "DEPLOYMENT_MODE", "remote")
    monkeypatch.setattr(settings, "AUTH_TOKEN", "test-token")
    headers = _auth_headers() | {"X-Search-Key": "tavily-key"}
    payload = _payload("op-remote-search-key")

    with TestClient(app) as client:
        response = client.post("/v1/research", json=payload, headers=headers)

    assert response.status_code == 400
    assert "local deployment mode" in response.json()["detail"]


def test_search_credentials_are_accepted_in_local_mode_with_gateway_enabled(monkeypatch):
    # Enabling the model gateway must not disable the independent local search
    # credential path; the request should be admitted.
    monkeypatch.setattr(settings, "DEPLOYMENT_MODE", "local")
    monkeypatch.setattr(settings, "AI_OPENROUTER_ENABLED", True)
    monkeypatch.setattr(settings, "AI_OPENROUTER_BASE_URL", "https://gateway.example")
    monkeypatch.setattr(settings, "AI_OPENROUTER_API_KEY", "test-gateway-key")
    _allow_test_gateway(monkeypatch)
    headers = _auth_headers() | {"X-Search-Key": "tavily-key"}
    payload = _payload("op-local-search-key-gateway")

    with TestClient(app) as client:
        response = client.post("/v1/research", json=payload, headers=headers)

    assert response.status_code == 202


def test_gateway_mode_allows_local_model_provider(monkeypatch):
    # A local provider needs no external egress, so the gateway's OpenAI-only
    # restriction must not reject it.
    monkeypatch.setattr(settings, "AI_OPENROUTER_ENABLED", True)
    monkeypatch.setattr(settings, "AI_OPENROUTER_BASE_URL", "https://gateway.example")
    monkeypatch.setattr(settings, "AI_OPENROUTER_API_KEY", "test-gateway-key")
    payload = _payload("op-gateway-ollama")
    payload["model_provider"] = "ollama"
    payload["model_name"] = "llama3"

    with TestClient(app) as client:
        response = client.post("/v1/research", json=payload, headers=_auth_headers())

    # Accepted for processing (a local provider is admitted, not rejected).
    assert response.status_code == 202


def test_gateway_request_does_not_reserve_local_budget(monkeypatch):
    # AI-OpenRouter is the budget authority for gateway-served work, so this
    # service must not apply its own daily ceiling (which could independently
    # reject work the gateway would allow).
    async def ok_research(**kwargs):
        return _completed_result(kwargs["op_id"])

    monkeypatch.setattr(api, "conduct_web_research", ok_research)
    # A ceiling below the per-operation reserve would normally reject admission.
    monkeypatch.setattr(settings, "DAILY_SPEND_LIMIT_USD", 0.01)
    monkeypatch.setattr(settings, "AI_OPENROUTER_ENABLED", True)
    monkeypatch.setattr(settings, "AI_OPENROUTER_BASE_URL", "https://gateway.example")
    monkeypatch.setattr(settings, "AI_OPENROUTER_API_KEY", "test-gateway-key")
    _allow_test_gateway(monkeypatch)
    payload = _payload("op-gateway-budget")
    payload["model_provider"] = "openai"
    payload["model_name"] = "gpt-4o-mini"

    with TestClient(app) as client:
        response = client.post("/v1/research", json=payload, headers=_auth_headers())
        assert response.status_code == 202
        result = _wait_for_terminal_result(client, "op-gateway-budget")

    assert result["status"] == "completed"


def test_local_request_still_enforces_local_spend_ceiling(monkeypatch):
    # The local ceiling remains an independent guard for local-model work (the
    # gateway is not authoritative for it), so it is still enforced.
    async def ok_research(**kwargs):
        return _completed_result(kwargs["op_id"])

    monkeypatch.setattr(api, "conduct_web_research", ok_research)
    monkeypatch.setattr(settings, "DAILY_SPEND_LIMIT_USD", 0.01)
    monkeypatch.setattr(settings, "AI_OPENROUTER_ENABLED", False)
    monkeypatch.setattr(settings, "OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    payload = _payload("op-local-budget")
    payload["model_provider"] = "ollama"
    payload["model_name"] = "llama3"

    with TestClient(app) as client:
        response = client.post("/v1/research", json=payload, headers=_auth_headers())

    assert response.status_code == 429
    assert "spend limit" in response.json()["detail"].lower()


def test_local_requests_do_not_exhaust_daily_budget_on_repeated_runs(monkeypatch):
    # Local-model work has no real external cost: the admission reservation
    # must be reconciled down to zero on completion, not to the
    # provider-agnostic token-cost heuristic reported in estimatedModelCostUsd,
    # or repeated credential-free local runs would eventually exhaust
    # DAILY_SPEND_LIMIT_USD on their own.
    async def ok_research(**kwargs):
        result = _completed_result(kwargs["op_id"])
        result["metrics"]["estimatedModelCostUsd"] = 0.2
        return result

    monkeypatch.setattr(api, "conduct_web_research", ok_research)
    # Larger than one admission reserve (0.25 default) but smaller than two
    # reserves plus the heuristic cost, so a second run only succeeds if the
    # first run's reservation was released back to zero, not to 0.2.
    monkeypatch.setattr(settings, "DAILY_SPEND_LIMIT_USD", 0.3)
    monkeypatch.setattr(settings, "AI_OPENROUTER_ENABLED", False)
    monkeypatch.setattr(settings, "OLLAMA_BASE_URL", "http://127.0.0.1:11434")

    with TestClient(app) as client:
        for i in range(3):
            payload = _payload(f"op-local-repeat-{i}", f"idem-local-repeat-{i}")
            payload["model_provider"] = "ollama"
            payload["model_name"] = "llama3"
            response = client.post("/v1/research", json=payload, headers=_auth_headers())
            assert response.status_code == 202, f"run {i} was rejected: {response.json()}"
            result = _wait_for_terminal_result(client, f"op-local-repeat-{i}")
            assert result["status"] == "completed"


def test_public_local_model_endpoint_is_rejected(monkeypatch):
    # A provider name alone does not establish locality: a public OLLAMA_BASE_URL
    # must not be admitted as a local-only request.
    monkeypatch.setattr(settings, "OLLAMA_BASE_URL", "https://ollama.example.com")
    payload = _payload("op-public-ollama")
    payload["model_provider"] = "ollama"
    payload["model_name"] = "llama3"

    with TestClient(app) as client:
        response = client.post("/v1/research", json=payload, headers=_auth_headers())

    assert response.status_code == 400
    assert "local address" in response.json()["detail"]


def test_no_gateway_with_public_local_endpoint_is_rejected(monkeypatch):
    monkeypatch.setattr(settings, "AI_OPENROUTER_ENABLED", False)
    monkeypatch.setattr(settings, "OLLAMA_BASE_URL", "https://ollama.example.com")

    with TestClient(app) as client:
        response = client.post("/v1/research", json=_payload("op-no-gateway-public-ollama"), headers=_auth_headers())

    assert response.status_code == 503
    assert "local model endpoint" in response.json()["detail"]
