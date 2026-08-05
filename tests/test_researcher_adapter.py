import os

import pytest

from app.model_adapter import RequestEnvironmentManager
import app.model_adapter as model_adapter
from app.researcher_adapter import (
    build_structured_findings,
    estimate_model_calls,
    estimate_model_cost_usd,
    trim_to_token_budget,
)


def test_build_structured_findings_links_claims_evidence_and_citations():
    sources = [
        {
            "id": "src-op-1-0",
            "url": "https://example.com/a",
            "title": "A",
            "retrievedAt": 1,
            "sourceType": "web",
        }
    ]
    report = "The product now supports source policies. It also records operation metrics."

    evidence, claims, citations = build_structured_findings("op-1", report, sources)

    assert evidence
    assert claims
    assert citations
    assert claims[0]["evidenceIds"] == [evidence[0]["id"]]
    assert claims[0]["verificationStatus"] == "partially-supported"
    assert evidence[0]["id"] in citations[0]["evidenceIds"]
    assert claims[0]["id"] in citations[0]["claimIds"]
    assert len(citations[0]["evidenceIds"]) == len(evidence)
    assert len(citations[0]["claimIds"]) == len(claims)


def test_trim_to_token_budget_truncates_long_text():
    text = " ".join(f"word{i}" for i in range(100))

    trimmed = trim_to_token_budget(text, 12)

    assert len(trimmed.split()) < len(text.split())
    assert "Truncated to satisfy maximumModelTokens" in trimmed


def test_budget_estimators_are_deterministic():
    assert estimate_model_calls("quick") == 2
    assert estimate_model_calls("standard") == 2
    assert estimate_model_calls("deep") == 4
    assert estimate_model_cost_usd(1000, 1000) == pytest.approx(0.0125)


def test_request_environment_manager_sets_request_scoped_model_preferences(monkeypatch):
    monkeypatch.delenv("FAST_LLM", raising=False)
    monkeypatch.delenv("SMART_LLM", raising=False)

    assert os.environ.get("FAST_LLM") != "openai:gpt-4o-mini"


def test_model_preferences_apply_when_raw_headers_are_disallowed(monkeypatch):
    monkeypatch.delenv("FAST_LLM", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(model_adapter, "raw_header_credentials_allowed", lambda: False)

    with RequestEnvironmentManager(
        {"X-LLM-Key": "should-not-apply"},
        model_provider="openai",
        model_name="gpt-4o-mini"
    ).apply_keys():
        assert os.environ["FAST_LLM"] == "openai:gpt-4o-mini"
        assert "OPENAI_API_KEY" not in os.environ

    with RequestEnvironmentManager({}, model_provider="openai", model_name="gpt-4o-mini").apply_keys():
        assert os.environ["FAST_LLM"] == "openai:gpt-4o-mini"
        assert os.environ["SMART_LLM"] == "openai:gpt-4o-mini"

    assert os.environ.get("FAST_LLM") != "openai:gpt-4o-mini"
