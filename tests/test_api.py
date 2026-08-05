import time

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


@pytest.fixture(autouse=True)
def reset_storage_state():
    for attr in ("operations", "events", "idempotency_keys", "operation_claims"):
        if hasattr(api.storage, attr):
            getattr(api.storage, attr).clear()
    api.cancellation_manager.active_tasks.clear()
    api._active_ops_count = 0


def _wait_for_terminal_result(client, operation_id):
    for _ in range(50):
        response = client.get(f"/v1/research/{operation_id}/result", headers=_auth_headers())
        assert response.status_code == 200
        body = response.json()
        if body["status"] in api.TERMINAL_STATUSES:
            return body
        time.sleep(0.02)
    raise AssertionError("operation did not finish")


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


def test_cancel_unknown_operation_returns_not_found():
    with TestClient(app) as client:
        response = client.post("/v1/research/op-missing/cancel", headers=_auth_headers())

    assert response.status_code == 404


def test_metrics_endpoint_is_public():
    with TestClient(app) as client:
        response = client.get("/metrics")

    assert response.status_code == 200
    assert "research_operations_total" in response.text
