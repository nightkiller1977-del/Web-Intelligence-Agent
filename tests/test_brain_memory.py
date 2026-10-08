import hashlib
import hmac
import json
from urllib.request import ProxyHandler


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
    text = envelope["sourceText"]
    assert captured["path"] == "/v1/artifacts/ingest"
    assert captured["domain"] == "brain-memory-http-ingest-v1"
    # Prose, not JSON: a .json fileName or application/json mimeType routes the
    # artifact through Brain's provider-export importers, which this is not.
    assert envelope["descriptor"]["fileName"] == "outcome.txt"
    assert envelope["descriptor"]["mimeType"] == "text/plain"
    assert "Web research outcome: completed (mode=standard)." in text
    assert "Sources consulted: 2 (web). Verified claims: 1." in text
    assert "op-1" not in captured["body"]


def test_ingest_retains_verified_findings_with_locator_and_as_of_stamp(monkeypatch):
    """Counters alone embed to nothing; the claim text is what makes it recallable."""
    from app.brain_memory import BrainMemoryClient

    client = BrainMemoryClient("https://brain.example", "web-agent-1", "test-secret")
    captured = {}

    def fake_request(path, body, domain):
        captured.update(body=body)
        return {"acceptedChunks": 1, "duplicates": 0, "rejected": 0, "quarantined": 0}

    monkeypatch.setattr(client, "_request", fake_request)

    assert client.ingest_verified_outcome(
        operation_id="op-1",
        status="completed",
        mode="quick",
        source_count=1,
        verified_claim_count=2,
        source_types=["web"],
        findings=[
            {"text": "urllib3 v2 requires OpenSSL 1.1.1 or newer", "url": "https://example.test/a", "sourceType": "web"},
            {"text": "A claim from a local design note", "url": "", "sourceType": "document"},
            {"text": "A claim whose evidence resolved to nothing", "url": "", "sourceType": ""},
        ],
    ) is True

    text = json.loads(captured["body"])["sourceText"]
    assert "urllib3 v2 requires OpenSSL 1.1.1 or newer [web: https://example.test/a]" in text
    # A redacted local locator still carries its type, so first-party material
    # is not recalled later as if it came from the web.
    assert "A claim from a local design note [document]" in text
    # Retained even with no provenance at all — dropping it would discard
    # verified knowledge.
    assert "- A claim whose evidence resolved to nothing" in text
    assert "Source-supported findings (passage-matched, not fact-checked), retrieved " in text
    # Verification is token overlap plus a negation check. It must never be
    # persisted as a claim of truth.
    assert "true as of" not in text


def test_ingest_bounds_retained_findings(monkeypatch):
    from app.brain_memory import BrainMemoryClient

    client = BrainMemoryClient("https://brain.example", "web-agent-1", "test-secret")
    captured = {}
    monkeypatch.setattr(client, "_request", lambda path, body, domain: captured.update(body=body) or {"acceptedChunks": 1})

    client.ingest_verified_outcome(
        operation_id="op-1",
        status="completed",
        mode="quick",
        source_count=40,
        verified_claim_count=40,
        source_types=["web"],
        findings=[{"text": f"short claim {index}", "url": "https://example.test/a", "sourceType": "web"} for index in range(40)],
    )

    text = json.loads(captured["body"])["sourceText"]
    assert len(text.encode("utf-8")) <= 8000
    # Exactly the cap, not merely "at most": a retention regression that dropped
    # every finding would also satisfy an upper bound.
    assert len([line for line in text.splitlines() if line.startswith("- ")]) == 10
    assert "Verified claims: 40." in text


def test_ingest_omits_overlong_claims_rather_than_truncating(monkeypatch):
    """Cutting a claim can strip a trailing negation and invert what the source
    supported, so an overlong claim is dropped and counted, never shortened."""
    from app.brain_memory import BrainMemoryClient

    client = BrainMemoryClient("https://brain.example", "web-agent-1", "test-secret")
    captured = {}
    monkeypatch.setattr(client, "_request", lambda path, body, domain: captured.update(body=body) or {"acceptedChunks": 1})

    long_claim = "The adapter supports streaming " + "x" * 5000 + " however, this is not approved."
    client.ingest_verified_outcome(
        operation_id="op-1", status="completed", mode="quick",
        source_count=1, verified_claim_count=2, source_types=["web"],
        findings=[
            {"text": long_claim, "url": "https://example.test/a", "sourceType": "web"},
            {"text": "a short verified claim", "url": "https://example.test/b", "sourceType": "web"},
        ],
    )

    text = json.loads(captured["body"])["sourceText"]
    assert "The adapter supports streaming" not in text
    assert "a short verified claim [web: https://example.test/b]" in text
    # The omission is recorded, not silent.
    assert "1 finding(s) omitted rather than truncated" in text
    assert len(text.encode("utf-8")) <= 8000


def test_ingest_omits_an_overlong_locator_but_keeps_the_claim(monkeypatch):
    """A cut path segment or percent-escape yields an invalid URL, or a valid
    one for a different resource — but the claim itself is still knowledge."""
    from app.brain_memory import BrainMemoryClient

    client = BrainMemoryClient("https://brain.example", "web-agent-1", "test-secret")
    captured = {}
    monkeypatch.setattr(client, "_request", lambda path, body, domain: captured.update(body=body) or {"acceptedChunks": 1})

    client.ingest_verified_outcome(
        operation_id="op-1", status="completed", mode="quick",
        source_count=1, verified_claim_count=1, source_types=["web"],
        findings=[{"text": "a verified claim", "url": "https://example.test/" + "y" * 400, "sourceType": "web"}],
    )

    text = json.loads(captured["body"])["sourceText"]
    assert "a verified claim [web]" in text
    assert "yyy" not in text


def test_ingest_without_findings_keeps_counter_only_record(monkeypatch):
    from app.brain_memory import BrainMemoryClient

    client = BrainMemoryClient("https://brain.example", "web-agent-1", "test-secret")
    captured = {}
    monkeypatch.setattr(client, "_request", lambda path, body, domain: captured.update(body=body) or {"acceptedChunks": 1})

    client.ingest_verified_outcome(
        operation_id="op-1", status="partial", mode="quick",
        source_count=0, verified_claim_count=0, source_types=[],
    )

    text = json.loads(captured["body"])["sourceText"]
    assert "Source-supported findings" not in text
    assert "Sources consulted: 0 (none). Verified claims: 0." in text


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


def test_request_disables_ambient_proxies(monkeypatch):
    """Signed Brain credentials must only be sent to the configured endpoint."""
    from app.brain_memory import BrainMemoryClient

    client = BrainMemoryClient("https://brain.example", "web-agent-1", "test-secret")
    handlers = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _limit):
            return b'{"accepted": true}'

    class Opener:
        def open(self, _request, timeout):
            assert timeout == 5
            return Response()

    def fake_build_opener(*configured_handlers):
        handlers.extend(configured_handlers)
        return Opener()

    monkeypatch.setattr("app.brain_memory.is_safe_egress_url", lambda _url: True)
    monkeypatch.setattr("app.brain_memory.build_opener", fake_build_opener)

    assert client._request("/v1/memories/search", "{}", "brain-memory-http-search-v1") == {"accepted": True}
    proxy_handlers = [handler for handler in handlers if isinstance(handler, ProxyHandler)]
    assert len(proxy_handlers) == 1
    assert proxy_handlers[0].proxies == {}


def test_recall_ignores_malformed_results_payload(monkeypatch):
    from app.brain_memory import BrainMemoryClient

    client = BrainMemoryClient("https://brain.example", "web-agent-1", "test-secret")
    monkeypatch.setattr(client, "_request", lambda *_args: {"results": None})

    assert client.recall_context("question") == ""


def test_ingest_rejects_negative_or_non_accepting_receipt(monkeypatch):
    from app.brain_memory import BrainMemoryClient

    for receipt in (
        {"accepted": False},
        {"acceptedChunks": 0, "duplicates": 0, "rejected": 1},
        {"acceptedChunks": 0, "duplicates": 0, "quarantined": 1},
        {"acceptedChunks": 0, "duplicates": 0},
        {},
        "not-an-object",
    ):
        client = BrainMemoryClient("https://brain.example", "web-agent-1", "test-secret")
        monkeypatch.setattr(client, "_request", lambda *_args, _r=receipt: _r)
        assert client.ingest_verified_outcome(
            operation_id="op-1",
            status="completed",
            mode="standard",
            source_count=1,
            verified_claim_count=1,
            source_types=["web"],
        ) is False, receipt


def test_ingest_accepts_chunk_receipt_with_accepted_chunks(monkeypatch):
    from app.brain_memory import BrainMemoryClient

    client = BrainMemoryClient("https://brain.example", "web-agent-1", "test-secret")
    monkeypatch.setattr(client, "_request", lambda *_args: {"acceptedChunks": 2, "duplicates": 0, "rejected": 0, "quarantined": 0})
    assert client.ingest_verified_outcome(
        operation_id="op-1",
        status="completed",
        mode="standard",
        source_count=1,
        verified_claim_count=1,
        source_types=["web"],
    ) is True

def test_ingest_rejects_malformed_receipt_counters_without_raising(monkeypatch):
    from app.brain_memory import BrainMemoryClient

    for receipt in (
        {"acceptedChunks": "unknown"},
        {"rejected": "not-a-number"},
        {"quarantined": ["x"], "acceptedChunks": 2},
    ):
        client = BrainMemoryClient("https://brain.example", "web-agent-1", "test-secret")
        monkeypatch.setattr(client, "_request", lambda *_args, _r=receipt: _r)
        assert client.ingest_verified_outcome(
            operation_id="op-1",
            status="completed",
            mode="standard",
            source_count=1,
            verified_claim_count=1,
            source_types=["web"],
        ) is False, receipt


def test_recall_keeps_whole_findings_when_bounding(monkeypatch):
    """The retained artifact is multi-line prose, one finding per line. Cutting
    it at a byte offset would slice the last finding before its qualifier — the
    same inversion the ingest side avoids by dropping whole findings."""
    from app.brain_memory import BrainMemoryClient

    client = BrainMemoryClient("https://brain.example", "web-agent-1", "test-secret")
    findings = [
        f"- finding {index} " + "x" * 300 + " however, this is not approved."
        for index in range(12)
    ]
    artifact = "Web research outcome: completed (mode=quick).\n" + "\n".join(findings)
    monkeypatch.setattr(client, "_request", lambda *_args: {
        "results": [{"text": artifact, "provenance": {"sourceType": "web-intelligence-outcome"}}]
    })

    context = client.recall_context("question")

    recalled = [line.strip() for line in context.splitlines() if line.strip().startswith("- finding")]
    assert recalled, "bounding dropped every finding"
    assert len(recalled) < len(findings), "fixture must exceed the budget or this proves nothing"
    for line in recalled:
        # Whole, never a prefix — a sliced line would not match any original.
        assert line in findings
    assert len(context.encode("utf-8")) <= 6000


def test_recall_context_is_bounded_in_utf8_bytes(monkeypatch):
    from app.brain_memory import BrainMemoryClient

    client = BrainMemoryClient("https://brain.example", "web-agent-1", "test-secret")
    monkeypatch.setattr(client, "_request", lambda *_args: {"results": [{"text": "😀" * 8000}]})

    assert len(client.recall_context("question").encode("utf-8")) <= 6000
