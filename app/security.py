# app/security.py
import ipaddress
import contextlib
import contextvars
import logging
import socket
import threading
from urllib.parse import urlparse

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

def _match_domain(host: str, candidates) -> bool:
    return any(host == cand or host.endswith(f".{cand}") for cand in candidates)

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

def is_safe_url(url: str, profile: str = "general") -> bool:
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

def is_safe_egress_url(url: str) -> bool:
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

        return resolve_and_verify_host(hostname)
    except Exception as e:
        logger.error(f"Error during SSRF egress check for URL {url}: {e}")
        return False

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

def _is_provider_api_url(url: str) -> bool:
    hostname = _hostname_from_url(str(url))
    return bool(hostname and _match_domain(hostname, PROVIDER_API_HOSTS))

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
            logger.error("Search provider budget exhausted before request to %s", url)
            raise PermissionError(f"Search budget exhausted before outbound request: {url}")
        budget["remaining"] = remaining - 1

def _ensure_safe_url(url: str):
    if not _egress_protection_active():
        return
    profile = _active_egress_profile()
    is_safe = False if profile == _DENY_ALL_PROFILE else (
        is_safe_egress_url(str(url)) if _is_provider_api_url(str(url))
        else is_safe_url(str(url), profile) if profile != "general"
        else is_safe_egress_url(str(url))
    )
    if not is_safe:
        logger.error("SSRF egress guard denied request to %s", url)
        raise PermissionError(f"SSRF blocked outbound request: {url}")
    _consume_search_budget(str(url))

def _ensure_safe_host(host: str):
    if not _egress_protection_active() or not host:
        return
    profile = _active_egress_profile()
    is_safe = False if profile == _DENY_ALL_PROFILE else (
        is_safe_url(f"https://{host}", profile) if profile != "general" else resolve_and_verify_host(str(host))
    )
    if not is_safe:
        logger.error("SSRF egress guard denied connection to host %s", host)
        raise PermissionError(f"SSRF blocked outbound host: {host}")

def _validate_resolved_hosts(host: str, resolved_hosts):
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
            is_safe = is_safe_ip(clean_host)
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
            _ensure_safe_host(host)
        except PermissionError as exc:
            raise aiohttp.ClientConnectorError(
                connection_key=None,
                os_error=exc
            )
        resolved_hosts = await _original_aiohttp_resolve_host(self, host, port, *args, **kwargs)
        try:
            _validate_resolved_hosts(host, resolved_hosts)
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
            _ensure_safe_url(str(request.url))
        except PermissionError as exc:
            raise httpx.ConnectError(str(exc), request=request) from exc
        return _original_httpx_client_send(self, request, *args, **kwargs)

    async def patched_httpx_async_client_send(self, request, *args, **kwargs):
        try:
            _ensure_safe_url(str(request.url))
        except PermissionError as exc:
            raise httpx.ConnectError(str(exc), request=request) from exc
        return await _original_httpx_async_client_send(self, request, *args, **kwargs)

    httpx.Client.send = patched_httpx_client_send
    httpx.AsyncClient.send = patched_httpx_async_client_send


_original_socket_create_connection = socket.create_connection
_original_socket_connect = socket.socket.connect

def patched_socket_create_connection(address, timeout=None, source_address=None, *args, **kwargs):
    host = address[0] if isinstance(address, tuple) and address else None
    _ensure_safe_host(host)
    return _original_socket_create_connection(address, timeout, source_address, *args, **kwargs)

def patched_socket_connect(self, address):
    host = address[0] if isinstance(address, tuple) and address else None
    _ensure_safe_host(host)
    return _original_socket_connect(self, address)

socket.create_connection = patched_socket_create_connection
socket.socket.connect = patched_socket_connect
