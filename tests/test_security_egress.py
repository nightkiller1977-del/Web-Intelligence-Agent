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
    assert sent_urls == ["http://127.0.0.1/health"]
