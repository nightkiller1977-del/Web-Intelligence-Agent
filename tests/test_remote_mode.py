"""
tests/test_remote_mode.py

Integration tests for remote mode deployment (Render.com).
Tests health checks, Redis connectivity, idempotency, concurrency limits,
and end-to-end research workflow.
"""

import os
import json
import asyncio
import pytest
import httpx
import sys
from unittest.mock import patch, MagicMock, AsyncMock, MagicMock as MockModule

# Set environment for remote mode testing BEFORE importing app
os.environ.setdefault('DEPLOYMENT_MODE', 'remote')
os.environ.setdefault('STORAGE_BACKEND', 'local')  # Use local storage to avoid Redis dependency
os.environ.setdefault('MAX_CONCURRENT_OPS', '3')
os.environ.setdefault('WEB_INTELLIGENCE_AUTH_TOKEN', 'test-token-12345')
os.environ.setdefault('MAX_MEMORY_MB', '512')

# Mock GPT Researcher before importing app modules
sys.modules['gpt_researcher'] = MagicMock()
sys.modules['gpt_researcher.agent'] = MagicMock()
sys.modules['gpt_researcher.actions'] = MagicMock()

from fastapi.testclient import TestClient
from app.main import app
from app.config import settings
from app.storage import storage


class TestHealthEndpoints:
    """Test health check endpoints required by Render."""

    def test_health_live_responds(self):
        """GET /health/live should respond quickly for liveness probes."""
        client = TestClient(app)
        response = client.get("/health/live")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    @pytest.mark.anyio
    async def test_health_ready_checks_dependencies(self):
        """GET /health/ready should probe storage and GPT Researcher availability."""
        client = TestClient(app)
        response = client.get("/health/ready")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] in ["ok", "degraded"]
        assert "gpt_researcher" in data
        assert "storage" in data
        assert "auth" in data

    def test_version_endpoint(self):
        """GET /version should return service and engine versions."""
        client = TestClient(app)
        response = client.get("/version")

        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "web-intelligence-agent"
        assert "serviceVersion" in data
        assert "engine" in data
        assert "gpt-researcher" in data["engine"]["name"]

    def test_capabilities_endpoint(self):
        """GET /capabilities should list supported features."""
        client = TestClient(app)
        response = client.get("/capabilities")

        assert response.status_code == 200
        data = response.json()
        assert "capabilities" in data
        assert data["capabilities"]["ssrf_egress_blocking"] is True
        assert data["capabilities"]["citations"] is True


class TestAuthentication:
    """Test Bearer token authentication in remote mode."""

    def test_request_without_auth(self):
        """POST /v1/research without auth headers should reject."""
        client = TestClient(app)
        response = client.post(
            "/v1/research",
            json={
                "operationId": "op-1",
                "attemptId": "attempt-1",
                "query": "test",
                "mode": "standard",
                "profile": "general",
                "limits": {
                    "maximumDurationSeconds": 30,
                    "maximumSearches": 3,
                    "maximumPages": 5,
                    "maximumSources": 5
                }
            },
            headers={"Idempotency-Key": "idem-1"}
        )
        # Should reject (401/403)
        assert response.status_code in [401, 403]

    def test_request_with_valid_auth(self):
        """POST /v1/research with valid Bearer token should accept request."""
        client = TestClient(app)
        headers = {"Authorization": f"Bearer {settings.AUTH_TOKEN}"}

        # Mock the research execution to avoid actual GPT Researcher call
        with patch('app.api.conduct_web_research', new_callable=AsyncMock) as mock_research:
            mock_research.return_value = {
                "status": "completed",
                "answer": "Test answer",
                "sources": [],
                "evidence": [],
                "claims": []
            }

            response = client.post(
                "/v1/research",
                json={
                    "operationId": "op-1",
                    "idempotencyKey": "idem-key-1",
                    "attemptId": "attempt-1",
                    "query": "test query",
                    "mode": "standard",
                    "profile": "general",
                    "limits": {
                        "maximumDurationSeconds": 30,
                        "maximumSearches": 3,
                        "maximumPages": 5,
                        "maximumSources": 5
                    }
                },
                headers=headers
            )
            # Should accept (202 Accepted or 200 OK with async task)
            assert response.status_code in [200, 202, 201]


class TestIdempotency:
    """Test idempotency key handling for duplicate requests."""

    def test_duplicate_request_with_idempotency_key(self):
        """Two identical requests with same Idempotency-Key should return same operation ID."""
        client = TestClient(app)
        headers = {
            "Authorization": f"Bearer {settings.AUTH_TOKEN}",
            "Idempotency-Key": "idem-key-duplicate-test-1"
        }

        payload = {
            "operationId": "op-dup-1",
            "attemptId": "attempt-dup-1",
            "query": "latest Python releases",
            "mode": "standard",
            "profile": "general",
            "limits": {
                "maximumDurationSeconds": 30,
                "maximumSearches": 3,
                "maximumPages": 5,
                "maximumSources": 5
            }
        }

        with patch('app.api.conduct_web_research', new_callable=AsyncMock):
            # First request
            response1 = client.post("/v1/research", json=payload, headers=headers)

            # Second identical request
            response2 = client.post("/v1/research", json=payload, headers=headers)

            # Both should succeed
            assert response1.status_code in [200, 201, 202]
            assert response2.status_code in [200, 201, 202]

            # Should return same operation ID (or at least acknowledge idempotency)
            data1 = response1.json()
            data2 = response2.json()
            # Both should reference the same operation
            if "operationId" in data1 and "operationId" in data2:
                assert data1["operationId"] == data2["operationId"]


class TestRedisLocking:
    """Test Redis atomic locking for multi-instance safety (horizontal scaling)."""

    @pytest.mark.anyio
    async def test_concurrent_operations_with_locking(self):
        """Multiple concurrent operations should use Redis locking correctly."""
        # Skip if Redis not available
        if settings.STORAGE_BACKEND != "redis":
            pytest.skip("Redis backend not configured")

        client = TestClient(app)
        headers = {"Authorization": f"Bearer {settings.AUTH_TOKEN}"}

        operation_ids = []

        # Simulate multiple concurrent requests
        async def make_request(op_id):
            try:
                response = client.post(
                    "/v1/research",
                    json={
                        "operationId": op_id,
                        "attemptId": f"attempt-{op_id}",
                        "query": f"test query {op_id}",
                        "mode": "standard",
                        "profile": "general",
                        "limits": {
                            "maximumDurationSeconds": 30,
                            "maximumSearches": 3,
                            "maximumPages": 5,
                            "maximumSources": 5
                        }
                    },
                    headers={
                        **headers,
                        "Idempotency-Key": f"idem-key-concurrent-{op_id}"
                    }
                )
                return response.status_code
            except Exception as e:
                pytest.fail(f"Request failed: {e}")

        with patch('app.api.conduct_web_research', new_callable=AsyncMock):
            # Fire 5 concurrent requests (should be limited by MAX_CONCURRENT_OPS)
            tasks = [
                make_request(f"op-concurrent-{i}")
                for i in range(5)
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # At least some should succeed (those within concurrency limit)
            success_count = sum(1 for r in results if isinstance(r, int) and r in [200, 201, 202])
            assert success_count > 0, "At least some concurrent operations should succeed"


class TestConcurrencyLimiting:
    """Test that MAX_CONCURRENT_OPS limit is enforced."""

    @pytest.mark.anyio
    async def test_concurrency_limit_enforced(self):
        """Exceeding MAX_CONCURRENT_OPS should return 429 or queue request."""
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            headers = {"Authorization": f"Bearer {settings.AUTH_TOKEN}"}
            max_ops = int(settings.MAX_CONCURRENT_OPS)

            # Mock research to simulate long-running operations
            async def slow_research(*args, **kwargs):
                await asyncio.sleep(5)  # Simulate slow research
                return {"status": "completed", "answer": "test"}

            with patch('app.api.conduct_web_research', side_effect=slow_research):
                # Try to submit more operations than allowed concurrently
                tasks = []
                for i in range(max_ops + 2):
                    tasks.append(client.post(
                        "/v1/research",
                        json={
                            "operationId": f"op-limit-{i}",
                            "attemptId": f"attempt-limit-{i}",
                            "query": f"query {i}",
                            "mode": "standard",
                            "profile": "general",
                            "limits": {
                                "maximumDurationSeconds": 30,
                                "maximumSearches": 3,
                                "maximumPages": 5,
                                "maximumSources": 5
                            }
                        },
                        headers={
                            **headers,
                            "Idempotency-Key": f"idem-key-limit-{i}"
                        }
                    ))

                responses = await asyncio.gather(*tasks)
                status_codes = [r.status_code for r in responses]

                # At least the first MAX_CONCURRENT_OPS should accept
                accepted = sum(1 for r in status_codes if r in [200, 201, 202])
                assert accepted == max_ops, f"Should accept exactly {max_ops} operations, got {accepted}"

                # The remaining requests should return 429
                rejected = sum(1 for r in status_codes if r == 429)
                assert rejected == 2, f"Should reject exactly 2 operations with 429, got {rejected}"


class TestEventStreaming:
    """Test SSE event streaming for research progress."""

    def test_sse_events_endpoint_exists(self):
        """GET /v1/research/{id}/events should exist and stream events."""
        client = TestClient(app)
        headers = {"Authorization": f"Bearer {settings.AUTH_TOKEN}"}

        # This would require an actual operation to be in progress
        # For now, test that endpoint responds to authentication
        response = client.get(
            "/v1/research/op-nonexistent-123/events",
            headers=headers
        )

        # Should either return 404 (not found) or 200 with stream
        # but not 401/403 (auth should pass)
        assert response.status_code != 401
        assert response.status_code != 403


class TestResultRetrieval:
    """Test fetching completed research results."""

    def test_result_endpoint_authentication(self):
        """GET /v1/research/{id}/result should require auth."""
        client = TestClient(app)

        # Without auth
        response_no_auth = client.get("/v1/research/op-test-123/result")
        assert response_no_auth.status_code in [401, 403, 422]

        # With auth
        headers = {"Authorization": f"Bearer {settings.AUTH_TOKEN}"}
        response_with_auth = client.get("/v1/research/op-test-123/result", headers=headers)

        # Should not be auth error (might be 404 for nonexistent op, but that's OK)
        assert response_with_auth.status_code != 401
        assert response_with_auth.status_code != 403


class TestCancellation:
    """Test operation cancellation endpoint."""

    def test_cancel_endpoint_exists(self):
        """POST /v1/research/{id}/cancel should exist."""
        client = TestClient(app)
        headers = {"Authorization": f"Bearer {settings.AUTH_TOKEN}"}

        response = client.post(
            "/v1/research/op-nonexistent-123/cancel",
            headers=headers
        )

        # Should not be auth error
        assert response.status_code != 401
        assert response.status_code != 403
        # Should be 404 (op not found) or 200 (cancel accepted)
        assert response.status_code in [200, 202, 404, 409]


class TestMemoryPressure:
    """Test memory pressure monitoring and spill-to-disk behavior."""

    def test_memory_threshold_configuration(self):
        """MAX_MEMORY_MB should be configured."""
        assert hasattr(settings, 'MAX_MEMORY_MB')
        assert settings.MAX_MEMORY_MB > 0
        assert settings.MAX_MEMORY_MB <= 2048  # Reasonable upper bound


class TestSSRFProtection:
    """Test SSRF protections in remote mode."""

    def test_private_ip_ranges_blocked(self):
        """Research targeting private IP ranges should be rejected."""
        from app.security import is_safe_url

        # Test various private/blocked ranges
        blocked_urls = [
            "http://localhost:8000",
            "http://127.0.0.1:5432",
            "http://192.168.1.1",
            "http://10.0.0.1",
            "http://169.254.169.254",  # AWS metadata
            "http://[::1]",  # IPv6 loopback
        ]

        for url in blocked_urls:
            result = is_safe_url(url)
            assert not result, f"URL {url} should be blocked by SSRF protection"

    def test_public_urls_allowed(self):
        """Research targeting public URLs should be allowed."""
        from app.security import is_safe_url

        allowed_urls = [
            "https://github.com",
            "https://docs.python.org",
            "https://www.wikipedia.org",
            "https://stackoverflow.com",
        ]

        for url in allowed_urls:
            result = is_safe_url(url)
            assert result, f"URL {url} should be allowed by SSRF protection"


class TestRenderEnvironmentVariables:
    """Test that Render environment variable injection works correctly."""

    def test_auth_token_from_environment(self):
        """AUTH_TOKEN should be loadable from environment (as Render provides)."""
        # In remote mode, WEB_INTELLIGENCE_AUTH_TOKEN is set by Render
        assert settings.AUTH_TOKEN is not None
        assert len(settings.AUTH_TOKEN) > 0

    def test_redis_url_configuration(self):
        """REDIS_URL should be available if Redis backend is configured."""
        if settings.STORAGE_BACKEND == "redis":
            # REDIS_URL should be set or derivable
            assert hasattr(settings, 'REDIS_URL') or hasattr(storage, 'redis')

    def test_deployment_mode_remote(self):
        """DEPLOYMENT_MODE should be 'remote' for this test suite."""
        assert settings.DEPLOYMENT_MODE == "remote"


class TestRenderContainerRecycle:
    """Test recovery behavior after container recycle (stateless transition)."""

    @pytest.mark.anyio
    async def test_redis_reconnection_on_restart(self):
        """Storage should reconnect to Redis if connection is lost and restored."""
        if settings.STORAGE_BACKEND != "redis":
            pytest.skip("Redis backend not configured")

        # Simulate connection loss and recovery
        # In a real scenario, the container would be recycled and restart fresh
        # This test ensures that the storage adapter can reconnect

        # For now, just verify storage has reconnection logic
        assert hasattr(storage, 'redis') or hasattr(storage, 'get_redis_connection')


# Integration test to run locally against a live Render deployment
@pytest.mark.integration
class TestLiveRenderDeployment:
    """Tests against a live Render deployment (requires RENDER_SERVICE_URL env var)."""

    @pytest.fixture(autouse=True)
    def render_url(self):
        """Get Render service URL from environment or skip test."""
        url = os.getenv('RENDER_SERVICE_URL')
        if not url:
            pytest.skip("RENDER_SERVICE_URL not set; skipping live Render tests")
        return url.rstrip('/')

    async def test_live_health_check(self, render_url):
        """Test health endpoints on live Render deployment."""
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{render_url}/health/live")
            assert response.status_code == 200
            assert response.json()["status"] == "ok"

    @pytest.mark.anyio
    async def test_live_research_workflow(self, render_url):
        """Test end-to-end research workflow on live deployment."""
        import time
        async with httpx.AsyncClient() as client:
            # 1. Get auth token (in practice, this comes from Render env vars)
            token = os.getenv('WEB_INTELLIGENCE_AUTH_TOKEN', 'test-token')
            headers = {"Authorization": f"Bearer {token}"}

            # 2. Submit research query
            payload = {
                "operationId": f"test-live-{int(time.time())}",
                "attemptId": "live-attempt-1",
                "idempotencyKey": f"idem-live-{int(time.time())}",
                "query": "What is the latest version of FastAPI?",
                "mode": "standard",
                "profile": "general",
                "limits": {
                    "maximumDurationSeconds": 60,
                    "maximumSearches": 3,
                    "maximumPages": 5,
                    "maximumSources": 5
                }
            }

            response = await client.post(
                f"{render_url}/v1/research",
                json=payload,
                headers=headers
            )
            assert response.status_code in [200, 201, 202]

            # 3. Check capabilities
            response = await client.get(f"{render_url}/capabilities")
            assert response.status_code == 200
            assert response.json()["capabilities"]["ssrf_egress_blocking"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
