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


def test_input_alias_fields_are_rejected():
    with pytest.raises(ValidationError):
        ResearchRequestInput(
            operationId="op-input-alias",
            attemptId="attempt-input-alias",
            query="test",
            inputs={"documentInputs": []},
            limits=_limits()
        )


def test_allow_external_use_must_be_boolean():
    with pytest.raises(ValidationError):
        ResearchRequestInput(
            operationId="op-input-bool",
            attemptId="attempt-input-bool",
            query="test",
            inputs={"allowExternalUse": "false"},
            limits=_limits()
        )
