# app/security.py
import ipaddress
import contextlib
import contextvars
import logging
import socket
import threading
from urllib.parse import urlparse

from app.config import GatewayConfig, external_gateway_hosts, local_model_endpoint

logger = logging.getLogger("web-intelligence")

BLOCKED_NETWORKS = [
    ipaddress.ip_network("127.0.0.0/8"),       # Loopback
    ipaddress.ip_network("10.0.0.0/8"),          # Private Class A
    ipaddress.ip_network("172.16.0.0/12"),       # Private Class B
    ipaddress.ip_network("192.168.0.0/16"),      # Private Class C
    ipaddress.ip_network("169.254.0.0/16"),      # Link-Local
    ipaddress.ip_network("100.64.0.0/10"),       # Carrier-Grade NAT (RFC 6598)
    ipaddress.ip_network("224.0.0.0/4"),         # Multicast
    ipaddress.ip_network("0.0.0.0/8"),           # Current network

    # IPv6 Ranges
    ipaddress.ip_network("::1/128"),             # Loopback
    ipaddress.ip_network("fc00::/7"),            # Unique Local Address (ULA)
    ipaddress.ip_network("fe80::/10"),           # Link-Local
    ipaddress.ip_network("ff00::/8"),            # Multicast
    ipaddress.ip_network("::/128")               # Unspecified
]

PROFILE_DOMAINS = {
    "technical": {
        "allowed": ["github.com", "docs.github.com", "raw.githubusercontent.com", "developer.mozilla.org", "nodejs.org", "npmjs.com", "pkg.go.dev"],
        "denied": ["pastebin.com", "hastebin.com", "0bin.net"]
    },
    "repair": {
        "allowed": ["github.com", "docs.github.com", "raw.githubusercontent.com", "developer.mozilla.org", "nodejs.org", "npmjs.com", "pkg.go.dev", "stackoverflow.com"],
        "denied": ["pastebin.com", "hastebin.com", "0bin.net"]
    },
    "code-review": {
        "allowed": ["github.com", "gitlab.com", "bitbucket.org", "docs.github.com", "npmjs.com", "pkg.go.dev", "cve.mitre.org", "nvd.nist.gov"],
        "denied": ["pastebin.com", "hastebin.com", "0bin.net"]
    },
    "security": {
        "allowed": ["cve.mitre.org", "nvd.nist.gov", "security.snyk.io", "github.com", "us-cert.cisa.gov", "kb.cert.org"],
        "denied": ["pastebin.com", "hastebin.com", "0bin.net"]
    }
}

PROVIDER_API_HOSTS = {
    "api.openai.com",
    "api.anthropic.com",
    "api.tavily.com",
    "generativelanguage.googleapis.com",
    "openrouter.ai"
}

SEARCH_PROVIDER_HOSTS = {
    "api.tavily.com",
    "google.serper.dev",
    "serpapi.com",
    "api.search.brave.com"
}

# Model and search provider endpoints the service itself must reach. Kept as a
# union so callers (e.g. result redaction) can recognize provider URLs without
# importing both sets.
PROVIDER_HOSTS = PROVIDER_API_HOSTS | SEARCH_PROVIDER_HOSTS


class SearchBudgetExhausted(PermissionError):
    """Raised when a research run exhausts its outbound search-provider budget.

    Subclasses PermissionError so the SSRF egress guards that already catch
    PermissionError keep working, but is distinguishable so the research loop
    can degrade to a bounded "partial" result instead of surfacing a search
    budget stop as an unhandled execution failure.
    """


def _match_domain(host: str, candidates) -> bool:
    return any(host == cand or host.endswith(f".{cand}") for cand in candidates)

def _is_provider_api_host(host: str) -> bool:
    """True when host is a provider API host, matching the configured gateway
    exactly.

    Static provider domains may match subdomains (``api.openai.com``), but the
    operator-configured gateway is a single endpoint: suffix-matching it would
    also exempt an attacker-controlled subdomain of the gateway (for example
    ``attacker.gateway.example.com``) from profile allowlists.
    """
    if _match_domain(host, PROVIDER_API_HOSTS):
        return True
    return host.lower().rstrip(".") in external_gateway_hosts()

# Path prefixes that mark the local model server's own inference API surface.
# Ollama exposes its native API under /api/* and an OpenAI-compatible surface
# under /v1/*; nothing else at that host/port is inference traffic.
_LOCAL_MODEL_API_PATH_PREFIXES = ("/api/", "/v1/")

def _is_local_model_endpoint(host: str, port: int | None = None, path: str | None = None) -> bool:
    """True when host/port is the validated local model endpoint.

    Only the exact operator-configured local model endpoint is exempted, so
    model inference can reach a loopback/private Ollama while every other
    private-address request stays blocked. The port is enforced when known so a
    different service on the same local host is not exposed.

    When ``path`` is given (the URL-level checks in ``is_safe_url``/
    ``is_safe_egress_url``), the exemption is further scoped to the model
    server's own inference API paths. Without this, any URL-shaped request —
    including a webpage fetched during research whose content happens to link
    to this loopback host/port — would be exempted from the profile allowlist
    and SSRF checks, turning a narrow inference-client exception into a general
    SSRF pivot against the local model service. ``path`` is intentionally
    omitted by the socket-level host checks (``_ensure_safe_host`` and
    resolved-address validation): those run only after the URL-level check
    already gated on path for HTTP-client traffic, so they stay permissive for
    the same already-authorized host/port.
    """
    endpoint = local_model_endpoint()
    if not endpoint:
        return False
    endpoint_host, endpoint_port = endpoint
    if port is not None and port != endpoint_port:
        return False
    if path is not None and not any(path.startswith(prefix) for prefix in _LOCAL_MODEL_API_PATH_PREFIXES):
        return False
    candidate = host.lower().rstrip(".")
    if candidate == endpoint_host:
        return True
    # ``localhost`` resolves to either loopback family; accept both spellings so
    # the socket-level guard (which sees the resolved IP) still recognizes it.
    if endpoint_host == "localhost" and candidate in ("127.0.0.1", "::1"):
        return True
    return False

def _hostname_from_url(url: str) -> str | None:
    try:
        parsed = urlparse(url)
        return parsed.hostname.lower().rstrip(".") if parsed.hostname else None
    except Exception:
        return None

def is_safe_ip(ip_str: str) -> bool:
    try:
        ip = ipaddress.ip_address(ip_str)
        if getattr(ip, "ipv4_mapped", None):
            ip = ip.ipv4_mapped

        if (
            ip.is_loopback
            or ip.is_private
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
            or ip.is_unspecified
        ):
            return False

        # Check against explicitly blocked networks
        for network in BLOCKED_NETWORKS:
            if ip in network:
                return False
        return True
    except ValueError:
        return False

def resolve_and_verify_host(hostname: str) -> bool:
    try:
        # Perform DNS lookup close to connection execution (mitigate DNS rebinding)
        addr_info = socket.getaddrinfo(hostname, None)
        ips = [info[4][0] for info in addr_info]

        if not ips:
            return False

        for ip in ips:
            # Strip IPv6 zone index if present
            clean_ip = ip.split('%')[0]
            if not is_safe_ip(clean_ip):
                logger.warning(f"SSRF validation blocked host {hostname} resolving to private IP: {clean_ip}")
                return False
        return True
    except Exception as e:
        logger.error(f"Failed to resolve host {hostname} during SSRF validation: {e}")
        return False

def is_safe_url(url: str, profile: str = "general", *, allow_local_model_endpoint: bool = False) -> bool:
    if not url:
        return False

    try:
        parsed = urlparse(url)
        # Enforce scheme validation: block file://, gopher://, etc.
        if parsed.scheme not in ("http", "https"):
            logger.warning(f"SSRF validation blocked invalid scheme in URL: {url}")
            return False

        hostname = parsed.hostname.lower().rstrip(".") if parsed.hostname else None
        if not hostname:
            return False

        # The operator-configured local model endpoint is a service-owned loopback
        # target, not a public web request; allow it without weakening the profile
        # allowlist for every other private address -- but only for the model
        # client's own transport (see _ensure_safe_url's `model_client` flag).
        # Callers outside the egress guard (citation/source-list filtering,
        # validating a caller-supplied query string) must never treat the
        # internal model endpoint as a legitimate web target, so this
        # exemption defaults off and is scoped to the endpoint's own inference
        # API path even when enabled, so it cannot become a general SSRF pivot
        # for arbitrary webpage-fetch traffic aimed at that host/port.
        if allow_local_model_endpoint and _is_local_model_endpoint(hostname, parsed.port, parsed.path):
            return True

        # Enforce domain allowlist/denylist by profile
        profile_rules = PROFILE_DOMAINS.get(profile)
        if profile_rules:
            allowed = profile_rules.get("allowed", [])
            denied = profile_rules.get("denied", [])

            if allowed and not _match_domain(hostname, allowed):
                logger.warning(f"Domain validation blocked host {hostname} outside allowed list for profile {profile}")
                return False

            if denied and _match_domain(hostname, denied):
                logger.warning(f"Domain validation blocked host {hostname} inside denied list for profile {profile}")
                return False

        # DNS resolution check
        return resolve_and_verify_host(hostname)
    except Exception as e:
        logger.error(f"Error during SSRF check for URL {url}: {e}")
        return False

def is_safe_egress_url(url: str, *, allow_local_model_endpoint: bool = False) -> bool:
    if not url:
        return False

    try:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            logger.warning(f"SSRF egress validation blocked invalid scheme in URL: {url}")
            return False

        hostname = parsed.hostname.lower().rstrip(".") if parsed.hostname else None
        if not hostname:
            return False

        # See is_safe_url()'s matching comment: defaults off so non-model-client
        # callers (Brain Memory, Grafana observability, and this guard's own
        # provider-API branch for non-local providers) never get the local
        # model exemption; only the model client's own transport enables it.
        if allow_local_model_endpoint and _is_local_model_endpoint(hostname, parsed.port, parsed.path):
            return True

        return resolve_and_verify_host(hostname)
    except Exception as e:
        logger.error(f"Error during SSRF egress check for URL {url}: {e}")
        return False


def is_gateway_destination_allowed(gateway: GatewayConfig | None) -> bool:
    """Whether a validated gateway config can pass the service egress policy."""
    return bool(gateway and is_safe_egress_url(gateway.base_url))


active_profile: contextvars.ContextVar[str] = contextvars.ContextVar("active_profile", default="general")
egress_protection_enabled: contextvars.ContextVar[bool] = contextvars.ContextVar("egress_protection_enabled", default=False)
active_search_budget: contextvars.ContextVar[dict | None] = contextvars.ContextVar("active_search_budget", default=None)
_protection_lock = threading.RLock()
_protection_depth = 0
_fallback_profile_stack: list[str] = []
_fallback_search_budget_stack: list[dict | None] = []
_DENY_ALL_PROFILE = "__deny_all__"

def _egress_protection_active() -> bool:
    if egress_protection_enabled.get():
        return True
    with _protection_lock:
        return _protection_depth > 0

def _active_egress_profile() -> str:
    if egress_protection_enabled.get():
        return active_profile.get()

    with _protection_lock:
        if not _fallback_profile_stack:
            return "general"
        unique_profiles = set(_fallback_profile_stack)
        if len(unique_profiles) == 1:
            return _fallback_profile_stack[-1]
        logger.error("SSRF egress guard found overlapping profile contexts; failing closed")
        return _DENY_ALL_PROFILE

def _active_search_budget() -> dict | None:
    budget = active_search_budget.get()
    if budget is not None:
        return budget

    with _protection_lock:
        active_budgets = [budget for budget in _fallback_search_budget_stack if budget is not None]
        if not active_budgets:
            return None
        if len(active_budgets) == 1:
            return active_budgets[0]
        logger.error("SSRF egress guard found overlapping search budgets; failing closed")
        return {"remaining": 0}

# Path prefixes that mark a provider-owned API surface rather than a public
# page. Some providers publish documentation and other public content on the
# same domain as their API, so the host alone is not enough to decide redaction.
_PROVIDER_API_PATH_PREFIXES = ("/v1/", "/v1beta/", "/v1internal/", "/api/v1/")

# Per-provider machine endpoints on mixed content domains. SerpApi's search API
# lives at /search.json, which the generic versioned-API prefixes above do not
# match, so list it explicitly rather than falling back to the public page rule.
_MIXED_PROVIDER_API_PATHS = {
    "serpapi.com": ("/search", "/account", "/locations"),
}


def _is_provider_api_url(url: str) -> bool:
    """True when url targets a model provider's API host the service itself calls.

    Used for outbound egress/DNS decisions, where any request to a provider host
    is treated as the provider API.
    """
    hostname = _hostname_from_url(str(url))
    return bool(hostname and _is_provider_api_host(hostname))


# Provider domains that also serve public web content (docs, blogs, marketing
# pages). On these, only API paths are the service's own endpoints; other pages
# are legitimate public sources and must not be redacted.
_MIXED_PROVIDER_DOMAINS = ("openrouter.ai", "serpapi.com")


def is_provider_host(url: str) -> bool:
    """True when url is one of the service's own provider API endpoints and must
    not be cited as a public web source.

    A public page hosted on a provider domain (for example a docs page on
    ``openrouter.ai`` or a help article on ``serpapi.com``) is not an API
    endpoint and is not redacted; only the machine API surfaces are.
    """
    hostname = _hostname_from_url(str(url))
    if not hostname:
        return False
    configured_gateway = hostname in external_gateway_hosts()
    static_provider = _match_domain(hostname, PROVIDER_HOSTS)
    if not configured_gateway and not static_provider:
        return False
    if static_provider and _match_domain(hostname, _MIXED_PROVIDER_DOMAINS):
        path = urlparse(str(url)).path or ""
        if any(path.startswith(prefix) for prefix in _PROVIDER_API_PATH_PREFIXES):
            return True
        for domain, extra_prefixes in _MIXED_PROVIDER_API_PATHS.items():
            if _match_domain(hostname, (domain,)) and any(path.startswith(pfx) for pfx in extra_prefixes):
                return True
        return False
    return True


def search_budget_exhausted() -> bool:
    """True when the active research run has exhausted its search-provider budget."""
    budget = _active_search_budget()
    return bool(budget and budget.get("exhausted"))

def _is_search_provider_url(url: str) -> bool:
    hostname = _hostname_from_url(str(url))
    return bool(hostname and _match_domain(hostname, SEARCH_PROVIDER_HOSTS))

def _consume_search_budget(url: str):
    if not _is_search_provider_url(url):
        return

    budget = _active_search_budget()
    if budget is None:
        return

    with _protection_lock:
        remaining = int(budget.get("remaining", 0))
        if remaining <= 0:
            # Record exhaustion on the shared budget object before raising. The
            # HTTP client wrappers translate PermissionError into their own
            # connection errors, so the research loop detects a budget stop by
            # observing this flag rather than by catching the exception type.
            budget["exhausted"] = True
            logger.error("Search provider budget exhausted before an outbound request.")
            raise SearchBudgetExhausted("Search budget exhausted before outbound request.")
        budget["remaining"] = remaining - 1

def _ensure_safe_url(url: str, *, model_client: bool = False):
    """Validate an outbound URL under the active egress guard.

    ``model_client`` must be True only when the caller is the patched httpx
    transport (openai/ollama SDKs use httpx exclusively in this stack, while
    GPT Researcher's own page scraping and search retrievers use requests, and
    aiohttp is otherwise unused for model traffic here). Only that transport
    may benefit from the local-model-endpoint exemption, so a webpage fetched
    over requests/aiohttp during profiled research cannot use a link back to
    the loopback model host/port as an SSRF pivot.
    """
    if not _egress_protection_active():
        return
    profile = _active_egress_profile()
    is_safe = False if profile == _DENY_ALL_PROFILE else (
        is_safe_egress_url(str(url), allow_local_model_endpoint=model_client) if _is_provider_api_url(str(url))
        else is_safe_url(str(url), profile, allow_local_model_endpoint=model_client) if profile != "general"
        else is_safe_egress_url(str(url), allow_local_model_endpoint=model_client)
    )
    if not is_safe:
        # Log only the scheme+host, not the full URL: query strings can carry
        # provider credentials or sensitive parameters.
        parsed = urlparse(str(url))
        logger.error("SSRF egress guard denied request to %s://%s", parsed.scheme, parsed.hostname)
        raise PermissionError(f"SSRF blocked outbound request: {url}")
    _consume_search_budget(str(url))

def _ensure_safe_host(host: str, port: int | None = None):
    if not _egress_protection_active() or not host:
        return

    # When httpx/httpcore resolves a hostname and calls socket.connect with the
    # resulting IP, we only need to verify it's a public address — profile-level
    # domain rules already fired at the URL layer.  Feeding a raw IP into the
    # domain-allowlist path would always fail for profiled research because IPs
    # never match strings like "github.com".
    clean_host = host.split('%')[0]  # strip IPv6 zone index before parsing
    try:
        ipaddress.ip_address(clean_host)
        if _is_local_model_endpoint(clean_host, port):
            return
        if not is_safe_ip(clean_host):
            logger.error("SSRF egress guard denied connection to private/reserved IP %s", host)
            raise PermissionError(f"SSRF blocked private IP connection: {host}")
        return
    except ValueError:
        pass  # not an IP address — fall through to hostname checks

    profile = _active_egress_profile()
    # Mirror the provider bypass from _ensure_safe_url so profiled research can
    # reach model/search APIs, and exempt the validated local model endpoint so
    # local inference is not mistaken for a public-address violation.
    if profile == _DENY_ALL_PROFILE:
        is_safe = False
    elif _is_local_model_endpoint(str(host), port):
        is_safe = True
    elif _is_provider_api_host(str(host)):
        is_safe = resolve_and_verify_host(str(host))
    elif profile != "general":
        is_safe = is_safe_url(f"https://{host}", profile)
    else:
        is_safe = resolve_and_verify_host(str(host))
    if not is_safe:
        logger.error("SSRF egress guard denied connection to host %s", host)
        raise PermissionError(f"SSRF blocked outbound host: {host}")

def _validate_resolved_hosts(host: str, resolved_hosts, port: int | None = None):
    if not _egress_protection_active():
        return

    for resolved in resolved_hosts or []:
        resolved_host = None
        if isinstance(resolved, dict):
            resolved_host = resolved.get("host") or resolved.get("hostname")
        elif isinstance(resolved, tuple) and len(resolved) >= 5:
            resolved_host = resolved[4][0]

        if not resolved_host:
            continue

        clean_host = str(resolved_host).split("%")[0]
        try:
            ipaddress.ip_address(clean_host)
            is_safe = _is_local_model_endpoint(host, port) or is_safe_ip(clean_host)
        except ValueError:
            is_safe = resolve_and_verify_host(clean_host)

        if not is_safe:
            logger.error(
                "SSRF egress guard denied resolved address %s for host %s",
                clean_host,
                host
            )
            raise PermissionError(f"SSRF blocked resolved address {clean_host} for host: {host}")

@contextlib.contextmanager
def enforce_egress_protection(profile: str = "general", maximum_searches: int | None = None):
    """Enable outbound URL/host validation for this research operation.

    The context variable covers normal asyncio execution. The process-level
    depth gives synchronous worker threads a conservative general-profile guard
    when libraries move blocking fetch work out of the event loop.
    """
    search_budget = {"remaining": maximum_searches} if maximum_searches is not None else None
    global _protection_depth
    profile_token = active_profile.set(profile)
    enabled_token = egress_protection_enabled.set(True)
    search_budget_token = active_search_budget.set(search_budget)
    with _protection_lock:
        _protection_depth += 1
        _fallback_profile_stack.append(profile)
        _fallback_search_budget_stack.append(search_budget)
    try:
        yield
    finally:
        with _protection_lock:
            _protection_depth = max(0, _protection_depth - 1)
            for index in range(len(_fallback_profile_stack) - 1, -1, -1):
                if _fallback_profile_stack[index] == profile:
                    _fallback_profile_stack.pop(index)
                    break
            for index in range(len(_fallback_search_budget_stack) - 1, -1, -1):
                if _fallback_search_budget_stack[index] is search_budget:
                    _fallback_search_budget_stack.pop(index)
                    break
        active_search_budget.reset(search_budget_token)
        egress_protection_enabled.reset(enabled_token)
        active_profile.reset(profile_token)


try:
    import aiohttp
except ImportError:  # pragma: no cover - optional import during partial installs
    aiohttp = None

if aiohttp:
    _original_aiohttp_request = aiohttp.ClientSession._request
    _original_aiohttp_resolve_host = aiohttp.TCPConnector._resolve_host

    async def patched_aiohttp_request(self, method, url, *args, **kwargs):
        try:
            _ensure_safe_url(str(url))
        except PermissionError as exc:
            raise aiohttp.ClientConnectorError(
                connection_key=None,
                os_error=exc
            )
        return await _original_aiohttp_request(self, method, url, *args, **kwargs)

    async def patched_aiohttp_resolve_host(self, host, port, *args, **kwargs):
        try:
            _ensure_safe_host(host, port)
        except PermissionError as exc:
            raise aiohttp.ClientConnectorError(
                connection_key=None,
                os_error=exc
            )
        resolved_hosts = await _original_aiohttp_resolve_host(self, host, port, *args, **kwargs)
        try:
            _validate_resolved_hosts(host, resolved_hosts, port)
        except PermissionError as exc:
            raise aiohttp.ClientConnectorError(
                connection_key=None,
                os_error=exc
            )
        return resolved_hosts

    aiohttp.ClientSession._request = patched_aiohttp_request
    aiohttp.TCPConnector._resolve_host = patched_aiohttp_resolve_host


try:
    import requests
except ImportError:  # pragma: no cover
    requests = None

if requests:
    _original_requests_send = requests.sessions.Session.send

    def patched_requests_send(self, request, **kwargs):
        try:
            _ensure_safe_url(request.url)
        except PermissionError as exc:
            raise requests.exceptions.ConnectionError(str(exc)) from exc
        return _original_requests_send(self, request, **kwargs)

    requests.sessions.Session.send = patched_requests_send


try:
    import httpx
except ImportError:  # pragma: no cover
    httpx = None

if httpx:
    _original_httpx_client_send = httpx.Client.send
    _original_httpx_async_client_send = httpx.AsyncClient.send

    def patched_httpx_client_send(self, request, *args, **kwargs):
        try:
            # httpx is exclusively the transport openai/ollama SDKs use in
            # this stack (GPT Researcher's own page scraping and search
            # retrievers use requests), so it is the model client's transport
            # and may benefit from the local-model-endpoint exemption.
            _ensure_safe_url(str(request.url), model_client=True)
        except PermissionError as exc:
            raise httpx.ConnectError(str(exc), request=request) from exc
        return _original_httpx_client_send(self, request, *args, **kwargs)

    async def patched_httpx_async_client_send(self, request, *args, **kwargs):
        try:
            _ensure_safe_url(str(request.url), model_client=True)
        except PermissionError as exc:
            raise httpx.ConnectError(str(exc), request=request) from exc
        return await _original_httpx_async_client_send(self, request, *args, **kwargs)

    httpx.Client.send = patched_httpx_client_send
    httpx.AsyncClient.send = patched_httpx_async_client_send


_original_socket_create_connection = socket.create_connection
_original_socket_connect = socket.socket.connect

def patched_socket_create_connection(address, timeout=None, source_address=None, *args, **kwargs):
    host = address[0] if isinstance(address, tuple) and address else None
    port = address[1] if isinstance(address, tuple) and len(address) > 1 else None
    _ensure_safe_host(host, port)
    return _original_socket_create_connection(address, timeout, source_address, *args, **kwargs)

def patched_socket_connect(self, address):
    host = address[0] if isinstance(address, tuple) and address else None
    port = address[1] if isinstance(address, tuple) and len(address) > 1 else None
    _ensure_safe_host(host, port)
    return _original_socket_connect(self, address)

socket.create_connection = patched_socket_create_connection
socket.socket.connect = patched_socket_connect
