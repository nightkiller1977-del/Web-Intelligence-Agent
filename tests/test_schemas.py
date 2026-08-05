import pytest
from pydantic import ValidationError

from app.schemas import LimitConfig, ResearchRequestInput


def _limits():
    return LimitConfig(
        maximumDurationSeconds=30,
        maximumSearches=1,
        maximumPages=1,
        maximumSources=1
    )


def test_unknown_research_profile_is_rejected():
    with pytest.raises(ValidationError):
        ResearchRequestInput(
            operationId="op-profile",
            attemptId="attempt-profile",
            query="test",
            profile="security-v2",
            limits=_limits()
        )


def test_unknown_research_mode_is_rejected():
    with pytest.raises(ValidationError):
        ResearchRequestInput(
            operationId="op-mode",
            attemptId="attempt-mode",
            query="test",
            mode="Deep",
            limits=_limits()
        )


def test_non_positive_limits_are_rejected():
    with pytest.raises(ValidationError):
        LimitConfig(
            maximumDurationSeconds=0,
            maximumSearches=1,
            maximumPages=1,
            maximumSources=1
        )


def test_blank_query_is_rejected():
    with pytest.raises(ValidationError):
        ResearchRequestInput(
            operationId="op-blank",
            attemptId="attempt-blank",
            query="   ",
            limits=_limits()
        )


def test_unknown_freshness_field_is_rejected():
    with pytest.raises(ValidationError):
        ResearchRequestInput(
            operationId="op-freshness",
            attemptId="attempt-freshness",
            query="test",
            freshness={"before": "2026-08-01"},
            limits=_limits()
        )


def test_unknown_inputs_field_is_rejected():
    with pytest.raises(ValidationError):
        ResearchRequestInput(
            operationId="op-inputs",
            attemptId="attempt-inputs",
            query="test",
            inputs={"attachments": []},
            limits=_limits()
        )
