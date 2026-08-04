import pytest
import requests
import httpx

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
