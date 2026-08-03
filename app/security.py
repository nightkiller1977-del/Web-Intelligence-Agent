# app/security.py
import ipaddress
import logging
import socket
from urllib.parse import urlparse
from app.config import settings

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

            # Domain suffix matching
            def match_domain(host, candidates):
                return any(host == cand or host.endswith(f".{cand}") for cand in candidates)

            if allowed and not match_domain(hostname, allowed):
                logger.warning(f"Domain validation blocked host {hostname} outside allowed list for profile {profile}")
                return False

            if denied and match_domain(hostname, denied):
                logger.warning(f"Domain validation blocked host {hostname} inside denied list for profile {profile}")
                return False

        # DNS resolution check
        return resolve_and_verify_host(hostname)
    except Exception as e:
        logger.error(f"Error during SSRF check for URL {url}: {e}")
        return False

# Task-local profile for SSRF domain filtering in the interceptor
import contextvars
import aiohttp

active_profile: contextvars.ContextVar[str] = contextvars.ContextVar("active_profile", default="general")

_original_request = aiohttp.ClientSession._request

async def patched_request(self, method, url, *args, **kwargs):
    url_str = str(url)
    profile = active_profile.get()
    if not is_safe_url(url_str, profile):
        logger.error("SSRF Interceptor: Denied request to %s (profile=%s)", url_str, profile)
        raise aiohttp.ClientConnectorError(
            connection_key=None,
            os_error=PermissionError(f"SSRF blocked connection to private or loopback target: {url_str}")
        )
    return await _original_request(self, method, url, *args, **kwargs)

aiohttp.ClientSession._request = patched_request
