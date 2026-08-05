from app.config import Settings


def test_auth_token_loads_from_documented_dotenv_name(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("WEB_INTELLIGENCE_AUTH_TOKEN=dotenv-token\n")

    settings = Settings(_env_file=env_file)

    assert settings.AUTH_TOKEN == "dotenv-token"
