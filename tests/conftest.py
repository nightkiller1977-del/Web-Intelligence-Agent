"""
tests/conftest.py

Pytest configuration for Web Intelligence Agent tests.
"""

import os
import sys
import pytest

# Add parent directory to path so app module is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Ensure remote mode for test environment
os.environ.setdefault('DEPLOYMENT_MODE', 'remote')
os.environ.setdefault('STORAGE_BACKEND', 'local')  # Use local storage for unit tests
os.environ.setdefault('MAX_CONCURRENT_OPS', '3')
os.environ.setdefault('WEB_INTELLIGENCE_AUTH_TOKEN', 'test-token-12345')
os.environ.setdefault('MAX_MEMORY_MB', '512')


@pytest.fixture
def anyio_backend():
    """Specify asyncio backend for anyio."""
    return 'asyncio'


@pytest.fixture(scope="session")
def event_loop():
    """Provide asyncio event loop for async tests."""
    import asyncio
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_gpt_researcher(monkeypatch):
    """Mock GPT Researcher to avoid actual LLM calls during tests."""
    from unittest.mock import MagicMock, AsyncMock

    mock_researcher_class = MagicMock()
    mock_instance = MagicMock()

    async def mock_conduct_research(*args, **kwargs):
        return {
            "status": "completed",
            "answer": "Test research answer for mocked query.",
            "sources": [
                {
                    "id": "src-1",
                    "url": "https://example.com",
                    "title": "Example Source",
                    "retrievedAt": 1693737600000
                }
            ],
            "evidence": [
                {
                    "id": "ev-1",
                    "sourceId": "src-1",
                    "passage": "This is test evidence text.",
                    "retrievedAt": 1693737600000
                }
            ],
            "claims": [
                {
                    "id": "claim-1",
                    "text": "Test claim supported by evidence.",
                    "confidenceScore": 0.95,
                    "status": "confirmed"
                }
            ]
        }

    monkeypatch.setattr(
        "app.researcher_adapter.conduct_web_research",
        AsyncMock(side_effect=mock_conduct_research)
    )

    return mock_researcher_class


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test (requires live service)"
    )
    config.addinivalue_line(
        "markers", "asyncio: mark test as async"
    )
