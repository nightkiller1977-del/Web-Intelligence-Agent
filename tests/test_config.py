from app.config import Settings


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


def test_non_positive_operational_limits_are_rejected(monkeypatch, tmp_path):
    import pytest

    monkeypatch.setenv("CONCURRENCY_LEASE_TTL_SECONDS", "0")
    with pytest.raises(Exception):
        Settings(_env_file=tmp_path / "missing.env")

    monkeypatch.setenv("CONCURRENCY_LEASE_TTL_SECONDS", "3600")
    monkeypatch.setenv("DEFAULT_OPERATION_COST_RESERVE_USD", "0")
    with pytest.raises(Exception):
        Settings(_env_file=tmp_path / "missing.env")
