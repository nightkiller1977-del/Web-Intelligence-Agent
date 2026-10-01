import pytest
import requests
import httpx
import socket

import app.security as security
from app.security import enforce_egress_protection


def test_search_budget_exhaustion_is_observable_and_distinguishable():
    with enforce_egress_protection(maximum_searches=1):
        security._consume_search_budget("https://api.tavily.com/search")
        assert security.search_budget_exhausted() is False

        with pytest.raises(security.SearchBudgetExhausted):
            security._consume_search_budget("https://api.tavily.com/search")

        # The flag lives on the shared budget object so the research loop can
        # detect the stop even when a client wrapper masks the exception type.
        assert security.search_budget_exhausted() is True

    # Scoped to the run: the next operation starts with a fresh budget.
    assert security.search_budget_exhausted() is False


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


def test_provider_redaction_allows_public_pages_on_provider_domains():
    from app.security import is_provider_host

    # Public documentation pages are not API endpoints and must not be redacted.
    assert is_provider_host("https://openrouter.ai/docs/quickstart") is False
    assert is_provider_host("https://serpapi.com/blog/how-it-works") is False
    # Machine API endpoints on the same domains/machine hosts are redacted.
    assert is_provider_host("https://api.openai.com/v1/chat/completions") is True
    assert is_provider_host("https://openrouter.ai/api/v1/models") is True
    assert is_provider_host("https://api.tavily.com/search") is True
    assert is_provider_host("https://google.serper.dev/search") is True


def test_gateway_destination_rejects_private_addresses():
    from app.config import GatewayConfig

    gateway = GatewayConfig("https://127.0.0.1", "test-key")

    assert security.is_gateway_destination_allowed(gateway) is False


def test_provider_redaction_includes_the_configured_gateway(monkeypatch):
    monkeypatch.setattr(
        security,
        "external_gateway_hosts",
        lambda: ("gateway.example", "openrouter.ai"),
    )

    assert security.is_provider_host("https://gateway.example/v1/chat/completions") is True
    assert security.is_provider_host("https://attacker.gateway.example/v1/chat/completions") is False
    assert security.is_provider_host("https://openrouter.ai/docs/quickstart") is False


def test_profile_policy_allows_configured_gateway_host(monkeypatch):
    """A configured AI-OpenRouter gateway must be treated as a provider host.

    Regression test: the gateway is not in the static provider list, so profiled
    research applied the public-domain allowlist and refused the configured
    gateway before inference could run.
    """
    monkeypatch.setattr(security, "resolve_and_verify_host", lambda _host: True)
    monkeypatch.setattr(security, "external_gateway_hosts", lambda: ("gateway.example",))

    with enforce_egress_protection("technical"):
        security._ensure_safe_url("https://gateway.example/v1/chat/completions")


def test_profile_policy_still_blocks_unconfigured_hosts(monkeypatch):
    monkeypatch.setattr(security, "resolve_and_verify_host", lambda _host: True)
    monkeypatch.setattr(security, "external_gateway_hosts", lambda: ("gateway.example",))

    with enforce_egress_protection("technical"):
        with pytest.raises(requests.exceptions.ConnectionError):
            requests.get("https://not-the-gateway.example/v1/chat/completions", timeout=0.1)


def test_profile_policy_blocks_subdomains_of_the_configured_gateway(monkeypatch):
    """The configured gateway is a single endpoint, not a domain wildcard.

    Regression test: suffix-matching the configured gateway would exempt an
    attacker-controlled subdomain (``attacker.gateway.example``) from the
    profile allowlist, leaving only the public-IP check.
    """
    monkeypatch.setattr(security, "resolve_and_verify_host", lambda _host: True)
    monkeypatch.setattr(security, "external_gateway_hosts", lambda: ("gateway.example",))

    assert security._is_provider_api_host("gateway.example") is True
    assert security._is_provider_api_host("attacker.gateway.example") is False

    with enforce_egress_protection("technical"):
        with pytest.raises(requests.exceptions.ConnectionError):
            requests.get("https://attacker.gateway.example/v1/chat/completions", timeout=0.1)


def test_local_model_endpoint_is_allowed_under_profiled_egress(monkeypatch):
    """Default local inference must not be blocked by the egress guard.

    Regression test: the no-gateway default points every model tier at Ollama on
    a loopback address, which the private-address check rejected, so local
    research failed before inference. The exemption is scoped to the model
    client's own transport (``model_client=True``, which the patched httpx
    send functions pass automatically) -- see
    test_local_model_endpoint_url_exemption_requires_model_client_transport
    for why a non-model-client caller must not get it.
    """
    monkeypatch.setattr(security, "local_model_endpoint", lambda: ("127.0.0.1", 11434))

    with enforce_egress_protection("technical"):
        security._ensure_safe_url("http://127.0.0.1:11434/api/chat", model_client=True)
        security._ensure_safe_host("127.0.0.1", 11434)


def test_local_model_endpoint_exemption_is_scoped_to_the_configured_endpoint(monkeypatch):
    """Only the configured local endpoint is exempt; other private hosts stay blocked."""
    monkeypatch.setattr(security, "local_model_endpoint", lambda: ("127.0.0.1", 11434))

    with enforce_egress_protection("technical"):
        with pytest.raises(requests.exceptions.ConnectionError):
            requests.get("http://127.0.0.1:6379/", timeout=0.1)
        with pytest.raises(PermissionError):
            security._ensure_safe_host("10.0.0.1", 11434)


def test_local_model_endpoint_url_exemption_is_scoped_to_inference_api_paths(monkeypatch):
    """The URL-level exemption must not become a general SSRF pivot.

    Regression test: an earlier fix exempted the configured local model
    host/port unconditionally in is_safe_url/is_safe_egress_url, so any
    URL-shaped request to that host/port -- including a webpage fetched during
    research whose content links back to it -- was let through the profile
    allowlist and SSRF checks. The exemption must be scoped to the model
    server's own inference API paths (/api/*, /v1/*), and (see the next test)
    to the model-client transport itself.
    """
    monkeypatch.setattr(security, "local_model_endpoint", lambda: ("127.0.0.1", 11434))

    assert security.is_safe_url("http://127.0.0.1:11434/api/chat", "technical", allow_local_model_endpoint=True) is True
    assert security.is_safe_url("http://127.0.0.1:11434/v1/chat/completions", "technical", allow_local_model_endpoint=True) is True
    assert security.is_safe_egress_url("http://127.0.0.1:11434/api/generate", allow_local_model_endpoint=True) is True

    # A non-API path at the same host/port (e.g. a page a scraper was tricked
    # into fetching) is not inference traffic and must not be exempted.
    assert security.is_safe_url("http://127.0.0.1:11434/", "technical", allow_local_model_endpoint=True) is False
    assert security.is_safe_egress_url("http://127.0.0.1:11434/", allow_local_model_endpoint=True) is False


def test_local_model_endpoint_url_exemption_requires_model_client_transport(monkeypatch):
    """The exemption defaults off; only an explicit model-client caller gets it.

    Regression test: is_safe_url()/is_safe_egress_url() must never exempt the
    local model host/port for a caller that did not explicitly ask for the
    model-client exemption -- citation/source-list filtering, a caller-supplied
    query URL check, or a webpage fetched over requests/aiohttp during
    research must all see this host/port as an ordinary private address, not
    as the service's own inference endpoint.
    """
    monkeypatch.setattr(security, "local_model_endpoint", lambda: ("127.0.0.1", 11434))

    assert security.is_safe_url("http://127.0.0.1:11434/api/chat", "technical") is False
    assert security.is_safe_egress_url("http://127.0.0.1:11434/api/chat") is False

    # The egress guard itself defaults to no model-client exemption, matching
    # what the requests/aiohttp transport patches pass (they never set
    # model_client=True -- only the httpx patches do, since httpx is the
    # model SDKs' exclusive transport in this stack).
    with enforce_egress_protection("technical"):
        with pytest.raises(requests.exceptions.ConnectionError):
            requests.get("http://127.0.0.1:11434/api/chat", timeout=0.1)
