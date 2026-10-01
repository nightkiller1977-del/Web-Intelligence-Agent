import pytest

from app.config import Settings, external_openai_gateway_config, local_model_endpoint, settings


def test_local_model_endpoint_treats_malformed_port_as_invalid(monkeypatch):
    # Same class of bug as the gateway port check: .port is lazily parsed and
    # raises ValueError only when accessed, so a caller (admission or the
    # no-gateway default) that doesn't expect that exception would see an
    # internal 500 instead of the documented 400/503 configuration response.
    monkeypatch.setattr(settings, "OLLAMA_BASE_URL", "http://127.0.0.1:abc")
    assert local_model_endpoint() is None

    monkeypatch.setattr(settings, "OLLAMA_BASE_URL", "http://127.0.0.1:99999")
    assert local_model_endpoint() is None

    monkeypatch.setattr(settings, "OLLAMA_BASE_URL", "http://127.0.0.1:0")
    assert local_model_endpoint() is None

    # A well-formed port is still accepted.
    monkeypatch.setattr(settings, "OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    assert local_model_endpoint() == ("127.0.0.1", 11434)


@pytest.mark.parametrize("host", [
    "169.254.169.254",
    "0.0.0.0",
    "[fe80::1]",
])
def test_local_model_endpoint_rejects_special_use_addresses(monkeypatch, host):
    monkeypatch.setattr(settings, "OLLAMA_BASE_URL", f"http://{host}:11434")

    assert local_model_endpoint() is None


def test_local_model_endpoint_still_accepts_rfc1918(monkeypatch):
    monkeypatch.setattr(settings, "OLLAMA_BASE_URL", "http://10.0.0.2:11434")

    assert local_model_endpoint() == ("10.0.0.2", 11434)


def test_gateway_config_rejects_malformed_port(monkeypatch):
    # urlsplit().port is a lazily-parsed property: a nonnumeric or
    # out-of-range port is not raised by urlsplit() itself and .hostname
    # stays populated, so a check that never accesses .port would admit the
    # config here and only fail later, asynchronously, when the HTTP client
    # parses the URL.
    monkeypatch.setattr(settings, "AI_OPENROUTER_ENABLED", True)
    monkeypatch.setattr(settings, "AI_OPENROUTER_API_KEY", "test-key")

    monkeypatch.setattr(settings, "AI_OPENROUTER_BASE_URL", "https://gateway.example:abc")
    with pytest.raises(ValueError):
        external_openai_gateway_config()

    monkeypatch.setattr(settings, "AI_OPENROUTER_BASE_URL", "https://gateway.example:99999")
    with pytest.raises(ValueError):
        external_openai_gateway_config()

    monkeypatch.setattr(settings, "AI_OPENROUTER_BASE_URL", "https://gateway.example:0")
    with pytest.raises(ValueError):
        external_openai_gateway_config()

    # A well-formed port is still accepted.
    monkeypatch.setattr(settings, "AI_OPENROUTER_BASE_URL", "https://gateway.example:8443")
    gateway = external_openai_gateway_config()
    assert gateway is not None
    assert gateway.base_url == "https://gateway.example:8443"


def test_auth_token_loads_from_documented_dotenv_name(tmp_path, monkeypatch):
    monkeypatch.delenv("WEB_INTELLIGENCE_AUTH_TOKEN", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("WEB_INTELLIGENCE_AUTH_TOKEN=dotenv-token\n")

    settings = Settings(_env_file=env_file)

    assert settings.AUTH_TOKEN == "dotenv-token"


def test_brain_memory_settings_load_from_dotenv(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "BRAIN_MEMORY_ENABLED=true\n"
        "BRAIN_MEMORY_CONTEXT_ENABLED=true\n"
        "BRAIN_MEMORY_URL=https://brain.example\n"
        "BRAIN_MEMORY_KEY_ID=web-agent-1\n"
        "BRAIN_MEMORY_SECRET=test-secret\n"
    )

    settings = Settings(_env_file=env_file)

    assert settings.BRAIN_MEMORY_ENABLED is True
    assert settings.BRAIN_MEMORY_CONTEXT_ENABLED is True
    assert settings.BRAIN_MEMORY_URL == "https://brain.example"


def test_gateway_settings_load_from_dotenv(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("AI_OPENROUTER_ENABLED=true\nAI_OPENROUTER_BASE_URL=https://gateway.example\nAI_OPENROUTER_API_KEY=test-key\n")
    settings = Settings(_env_file=env_file)
    assert settings.AI_OPENROUTER_ENABLED is True
    assert settings.AI_OPENROUTER_BASE_URL == "https://gateway.example"


def test_non_positive_operational_limits_are_rejected(monkeypatch, tmp_path):
    import pytest

    monkeypatch.setenv("CONCURRENCY_LEASE_TTL_SECONDS", "0")
    with pytest.raises(Exception):
        Settings(_env_file=tmp_path / "missing.env")

    monkeypatch.setenv("CONCURRENCY_LEASE_TTL_SECONDS", "3600")
    monkeypatch.setenv("DEFAULT_OPERATION_COST_RESERVE_USD", "0")
    with pytest.raises(Exception):
        Settings(_env_file=tmp_path / "missing.env")


def test_default_reserve_exceeding_daily_limit_is_rejected(monkeypatch, tmp_path):
    import pytest

    # A ceiling below the documented default reserve would reject every request
    # that has no smaller explicit maximumModelCostUsd, so fail at startup.
    monkeypatch.setenv("DAILY_SPEND_LIMIT_USD", "0.10")
    with pytest.raises(Exception):
        Settings(_env_file=tmp_path / "missing.env")


def test_explicit_reserve_may_exceed_daily_limit(monkeypatch, tmp_path):
    # An operator who deliberately sets both values is not overridden by the
    # documented-default guard.
    monkeypatch.setenv("DAILY_SPEND_LIMIT_USD", "0.10")
    monkeypatch.setenv("DEFAULT_OPERATION_COST_RESERVE_USD", "0.50")
    settings = Settings(_env_file=tmp_path / "missing.env")
    assert settings.DAILY_SPEND_LIMIT_USD == 0.10


def test_reconcile_interval_must_be_shorter_than_lease_ttl(monkeypatch, tmp_path):
    import pytest

    monkeypatch.setenv("STALE_RECONCILE_INTERVAL_SECONDS", "3600")
    monkeypatch.setenv("CONCURRENCY_LEASE_TTL_SECONDS", "3600")
    with pytest.raises(Exception):
        Settings(_env_file=tmp_path / "missing.env")


def test_lease_ttl_below_heartbeat_margin_is_rejected(monkeypatch, tmp_path):
    import pytest

    monkeypatch.setenv("CONCURRENCY_LEASE_TTL_SECONDS", "1")
    with pytest.raises(Exception):
        Settings(_env_file=tmp_path / "missing.env")
