import pytest
import requests
import httpx
import socket

import app.security as security
from app.security import enforce_egress_protection


def test_requests_private_url_blocked_when_egress_guard_enabled():
    with enforce_egress_protection():
        with pytest.raises(requests.exceptions.ConnectionError):
            requests.get("http://127.0.0.1/latest/meta-data", timeout=0.1)


def test_httpx_private_url_blocked_when_egress_guard_enabled():
    with enforce_egress_protection():
        with pytest.raises(httpx.ConnectError):
            httpx.get("http://169.254.169.254/latest/meta-data", timeout=0.1)


def test_direct_socket_connect_private_host_blocked_when_guard_enabled():
    with enforce_egress_protection():
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            with pytest.raises(PermissionError):
                sock.connect(("127.0.0.1", 80))
        finally:
            sock.close()


def test_actual_resolver_private_address_is_blocked():
    resolved_hosts = [{"hostname": "example.test", "host": "169.254.169.254", "port": 80}]

    with enforce_egress_protection():
        with pytest.raises(PermissionError):
            security._validate_resolved_hosts("example.test", resolved_hosts)


def test_profile_policy_blocks_disallowed_public_hosts(monkeypatch):
    monkeypatch.setattr(security, "resolve_and_verify_host", lambda _host: True)

    with enforce_egress_protection("security"):
        with pytest.raises(requests.exceptions.ConnectionError):
            requests.get("https://pastebin.com/raw/example", timeout=0.1)


def test_profile_policy_allows_provider_api_hosts(monkeypatch):
    monkeypatch.setattr(security, "resolve_and_verify_host", lambda _host: True)

    with enforce_egress_protection("security"):
        security._ensure_safe_url("https://api.openai.com/v1/chat/completions")


def test_search_provider_budget_is_enforced(monkeypatch):
    monkeypatch.setattr(security, "resolve_and_verify_host", lambda _host: True)

    with enforce_egress_protection("general", maximum_searches=1):
        security._ensure_safe_url("https://api.tavily.com/search")
        with pytest.raises(PermissionError):
            security._ensure_safe_url("https://api.tavily.com/search")


def test_profile_policy_is_preserved_for_thread_fallback(monkeypatch):
    monkeypatch.setattr(security, "resolve_and_verify_host", lambda _host: True)

    with enforce_egress_protection("security"):
        token = security.egress_protection_enabled.set(False)
        try:
            with pytest.raises(PermissionError):
                security._ensure_safe_url("https://pastebin.com/raw/example")
        finally:
            security.egress_protection_enabled.reset(token)


def test_overlapping_thread_fallback_profiles_fail_closed(monkeypatch):
    monkeypatch.setattr(security, "resolve_and_verify_host", lambda _host: True)

    with enforce_egress_protection("security"):
        with enforce_egress_protection("technical"):
            token = security.egress_protection_enabled.set(False)
            try:
                with pytest.raises(PermissionError):
                    security._ensure_safe_url("https://github.com/example/project")
            finally:
                security.egress_protection_enabled.reset(token)


def test_guard_does_not_affect_http_clients_when_disabled(monkeypatch):
    sent_urls = []

    def fake_send(self, request, **kwargs):
        sent_urls.append(request.url)
        response = requests.Response()
        response.status_code = 204
        response.url = request.url
        return response

    monkeypatch.setattr(security, "_original_requests_send", fake_send)

    response = requests.get("http://127.0.0.1/health", timeout=0.1)

    assert response.status_code == 204


def test_direct_socket_connect_public_ip_allowed_under_profiled_egress():
    """Regression test for the profiled-research SSRF bug: when httpx/httpcore
    resolves api.openai.com and calls socket.connect with the resulting public IP,
    _ensure_safe_host must allow it instead of rejecting it for not matching the
    profile's domain allowlist (e.g. "technical" only allows github.com etc.)."""
    with enforce_egress_protection("technical"):
        # A public IP (1.1.1.1 = Cloudflare DNS) must not be blocked by
        # profile-level domain rules — it should pass the public-IP check.
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            # We expect a real connection attempt (which will be refused/timeout),
            # NOT a PermissionError from the SSRF guard.
            try:
                sock.connect(("1.1.1.1", 9))  # port 9 = discard, almost always refused
            except PermissionError:
                raise  # re-raise so the test fails clearly
            except OSError:
                pass  # connection refused or timed out — correct, guard didn't block
        finally:
            sock.close()


def test_direct_socket_connect_private_ip_still_blocked_under_profiled_egress():
    """Private IPs must remain blocked even after the public-IP early-return
    in _ensure_safe_host — the fix must not weaken that defence."""
    with enforce_egress_protection("technical"):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            with pytest.raises(PermissionError):
                sock.connect(("10.0.0.1", 80))
        finally:
            sock.close()
