import time

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
    assert capabilities["citations"] is False
    assert capabilities["structured_evidence"] is False
    assert capabilities["claim_verification"] is False
    assert capabilities["source_policy"] is False
    assert capabilities["model_budget_limits"] is False
    assert capabilities["model_preferences"] is False


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


def test_source_policy_is_rejected_as_unsupported():
    payload = _payload("op-source-policy")
    payload["sourcePolicy"] = {"allowedDomains": ["example.com"]}

    with TestClient(app) as client:
        response = client.post("/v1/research", json=payload, headers=_auth_headers())

    assert response.status_code == 400
    assert "sourcePolicy is not supported" in response.json()["detail"]


def test_model_budget_limits_are_rejected_as_unsupported():
    payload = _payload("op-model-budget")
    payload["limits"]["maximumModelTokens"] = 1000

    with TestClient(app) as client:
        response = client.post("/v1/research", json=payload, headers=_auth_headers())

    assert response.status_code == 400
    assert "Unsupported model budget limits" in response.json()["detail"]


def test_model_preferences_are_rejected_as_unsupported():
    payload = _payload("op-model-preferences")
    payload["model_provider"] = "openai"

    with TestClient(app) as client:
        response = client.post("/v1/research", json=payload, headers=_auth_headers())

    assert response.status_code == 400
    assert "Unsupported model preference fields" in response.json()["detail"]


def test_cancel_unknown_operation_returns_not_found():
    with TestClient(app) as client:
        response = client.post("/v1/research/op-missing/cancel", headers=_auth_headers())

    assert response.status_code == 404
