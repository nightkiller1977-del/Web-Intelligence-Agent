import hashlib
import hmac
import json


def test_search_signs_exact_body_and_bounds_untrusted_context(monkeypatch):
    from app.brain_memory import BrainMemoryClient

    client = BrainMemoryClient("https://brain.example", "web-agent-1", "test-secret")
    captured = {}

    def fake_request(path, body, domain):
        captured.update(path=path, body=body, domain=domain)
        return {
            "results": [
                {"text": "Historical evidence one", "provenance": {"sourceType": "web"}},
                {"text": "Historical evidence two", "provenance": {"sourceType": "web"}},
                {"text": "x" * 8000, "provenance": {"sourceType": "web"}},
            ]
        }

    monkeypatch.setattr(client, "_request", fake_request)

    context = client.recall_context("new research question")

    assert captured["path"] == "/v1/memories/search"
    assert captured["domain"] == "brain-memory-http-search-v1"
    assert json.loads(captured["body"]) == {
        "text": "new research question",
        "scope": "web-intelligence",
        "topK": 3,
        "minimumScore": 0.6,
    }
    assert "UNTRUSTED HISTORICAL EVIDENCE" in context
    assert len(context) <= 6000
    assert "Historical evidence one" in context


def test_recall_retains_stable_artifact_provenance(monkeypatch):
    from app.brain_memory import BrainMemoryClient

    client = BrainMemoryClient("https://brain.example", "web-agent-1", "test-secret")
    monkeypatch.setattr(client, "_request", lambda *_args: {
        "results": [
            {
                "text": "Historical evidence one",
                "provenance": {"sourceType": "web", "artifactId": "artifact-abc"},
            },
            {
                "text": "Historical evidence two",
                "provenance": {"sourceType": "web", "sourceId": "source-xyz"},
            },
        ]
    })

    context = client.recall_context("new research question")

    assert "artifact:artifact-abc" in context
    assert "artifact:source-xyz" in context


def test_ingest_uses_route_specific_signature_and_deidentified_outcome(monkeypatch):
    from app.brain_memory import BrainMemoryClient

    client = BrainMemoryClient("https://brain.example", "web-agent-1", "test-secret")
    captured = {}

    def fake_request(path, body, domain):
        captured.update(path=path, body=body, domain=domain)
        return {"accepted": True}

    monkeypatch.setattr(client, "_request", fake_request)

    assert client.ingest_verified_outcome(
        operation_id="op-1",
        status="completed",
        mode="standard",
        source_count=2,
        verified_claim_count=1,
        source_types=["web", "web"],
    ) is True

    envelope = json.loads(captured["body"])
    payload = json.loads(envelope["sourceText"])
    assert captured["path"] == "/v1/artifacts/ingest"
    assert captured["domain"] == "brain-memory-http-ingest-v1"
    assert payload == {
        "kind": "outcome",
        "mode": "standard",
        "status": "completed",
        "sourceCount": 2,
        "verifiedClaimCount": 1,
        "sourceTypes": ["web"],
    }
    assert "op-1" not in captured["body"]


def test_signature_is_bound_to_exact_body():
    from app.brain_memory import BrainMemoryClient

    client = BrainMemoryClient("https://brain.example", "web-agent-1", "test-secret")
    body = '{"text":"a"}'
    signature = client.sign(body, "/v1/memories/search", "brain-memory-http-search-v1", "request-1", "2026-01-01T00:00:00.000Z")
    canonical = "\n".join([
        "brain-memory-http-search-v1", "POST", "/v1/memories/search", "web-agent-1",
        "request-1", "2026-01-01T00:00:00.000Z", hashlib.sha256(body.encode()).hexdigest(),
    ])
    assert signature == hmac.new(b"test-secret", canonical.encode(), hashlib.sha256).hexdigest()


def test_request_rejects_private_or_rebound_brain_endpoint(monkeypatch):
    from app.brain_memory import BrainMemoryClient

    client = BrainMemoryClient("https://brain.example", "web-agent-1", "test-secret")
    monkeypatch.setattr("app.brain_memory.is_safe_egress_url", lambda url: False)

    try:
        client._request("/v1/memories/search", "{}", "brain-memory-http-search-v1")
    except ValueError as exc:
        assert "approved egress" in str(exc)
    else:
        raise AssertionError("unsafe Brain endpoint must be rejected before a request is sent")


def test_recall_ignores_malformed_results_payload(monkeypatch):
    from app.brain_memory import BrainMemoryClient

    client = BrainMemoryClient("https://brain.example", "web-agent-1", "test-secret")
    monkeypatch.setattr(client, "_request", lambda *_args: {"results": None})

    assert client.recall_context("question") == ""
