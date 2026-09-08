"""Bounded, metadata-only Loki delivery and pure ASGI request/lifecycle observation.

Destination/credential resolution follows the fleet observability contract
(ACES-293), one implementation per repo:

* ``resolve_loki_config()`` — atomic ``LOKI_URL_REMOTE`` + ``LOKI_REMOTE_AUTH``
  pair from the process environment; a one-sided pair disables export with a
  single warning (values are never logged).
* ``validate_basic_auth()`` — auth must be ``Basic <base64(user:password)>``
  with non-empty user/password and no control characters; anything else
  disables export before a worker or queue entry exists, never a retry loop.
* Default policy — remote export auto-enables in production and is off in
  dev/test unless ``OBSERVABILITY_REMOTE=1``; ``OBSERVABILITY_REMOTE=0`` opts
  out even in production. Production means ``DEPLOYMENT_MODE=remote`` (set by
  ``render.yaml``, mirroring ``app/config.py``) or the ``RENDER`` env var that
  Render sets on every service. Local Python logging is independent.
"""
from __future__ import annotations

import asyncio
import base64
import binascii
import json
import logging
import math
import os
import queue
import threading
import time
import urllib.request
from typing import NamedTuple, Optional
from urllib.parse import urlsplit

_ALLOWED = {"method", "route", "status", "duration_ms", "error_type", "reason", "storage_degraded", "redis_enabled"}

_log = logging.getLogger("grafana-observability")
_warned: set[str] = set()  # reason codes already warned about — never values


# NamedTuple, not a dataclass: this module is also loaded by path in tests
# (spec_from_file_location) where dataclass field resolution needs sys.modules.
class LokiConfig(NamedTuple):
    """Validated, atomic remote-export destination."""
    enabled: bool
    url: Optional[str] = None
    auth: Optional[str] = None
    source: Optional[str] = None  # authority the pair came from ("env")
    reason: Optional[str] = None  # why export is disabled (never a value)


def validate_basic_auth(value):
    """True only for ``Basic <base64>`` decoding to non-empty ``user:pass``."""
    if not isinstance(value, str) or not value:
        return False
    if any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in value):
        return False
    scheme, _, payload = value.partition(" ")
    if scheme.lower() != "basic" or not payload:
        return False
    try:
        decoded = base64.b64decode(payload, validate=True).decode("utf-8")
    except (ValueError, UnicodeDecodeError, binascii.Error):
        return False
    if any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in decoded):
        return False
    user, sep, password = decoded.partition(":")
    return bool(sep) and bool(user) and bool(password)


def _valid_push_url(url):
    try:
        target = urlsplit(url)
    except ValueError:
        return False
    return bool(
        target.scheme == "https" and target.hostname
        and not target.username and not target.password
        and not target.query and not target.fragment
    )


def is_production(env=None):
    env = os.environ if env is None else env
    return env.get("DEPLOYMENT_MODE", "local") == "remote" or bool(env.get("RENDER"))


def _disabled(reason, *, warn=True):
    if warn and reason not in _warned:
        _warned.add(reason)
        _log.warning("loki remote export disabled: %s (values are never logged)", reason)
    return LokiConfig(enabled=False, reason=reason)


def resolve_loki_config(env=None):
    """Resolve the atomic remote pair + policy. Never raises, never logs values."""
    env = os.environ if env is None else env
    url = (env.get("LOKI_URL_REMOTE") or "").strip()
    auth = (env.get("LOKI_REMOTE_AUTH") or "").strip()

    opt = (env.get("OBSERVABILITY_REMOTE") or "").strip()
    if opt == "0":
        return _disabled("opted_out", warn=False)
    if opt != "1" and not is_production(env):
        return _disabled("non_production_default_off", warn=False)

    if not url and not auth:
        return _disabled("unconfigured", warn=False)
    if bool(url) != bool(auth):
        return _disabled("partial_pair")
    if not _valid_push_url(url):
        return _disabled("invalid_url")
    if not validate_basic_auth(auth):
        return _disabled("invalid_auth")
    return LokiConfig(enabled=True, url=url, auth=auth, source="env")


def reset_warnings():
    """Test hook: allow one-shot warnings to fire again."""
    _warned.clear()

class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _send_http(url, auth, body):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())
    request = urllib.request.Request(url, data=body, method="POST", headers={"Content-Type": "application/json", "Authorization": auth})
    with opener.open(request, timeout=1.5) as response:
        return 200 <= response.status < 300


class LokiEmitter:
    """One worker and a finite queue per runtime, never one thread per event."""
    def __init__(self, service, *, sender=None, capacity=64):
        if not 1 <= capacity <= 128:
            raise ValueError("queue capacity must be between 1 and 128")
        self.service = service
        self._sender = sender or _send_http
        self._queue = queue.Queue(maxsize=capacity)
        self._lock = threading.Lock()
        self._worker = None
        self._stopped = threading.Event()
        self.stats = {"accepted": 0, "dropped": 0, "sent": 0, "failed": 0}

    def emit(self, event, **fields):
        # Atomic pair + policy + Basic-auth validation live in one resolver;
        # invalid or partial config never starts the worker, so a malformed
        # credential can never become a retry loop.
        config = resolve_loki_config()
        if not config.enabled:
            return False
        url, auth = config.url, config.auth
        try:
            safe = {"service": self.service, "event": event}
            for key, value in fields.items():
                if key not in _ALLOWED:
                    continue
                if isinstance(value, str):
                    safe[key] = value[:160]
                elif isinstance(value, (bool, int)) or isinstance(value, float) and math.isfinite(value):
                    safe[key] = value
            body = json.dumps({"streams": [{"stream": {
                "application": "ai-agents", "agent": self.service,
                "environment": os.getenv("ENVIRONMENT", "production"),
            }, "values": [[str(time.time_ns()), json.dumps(safe, separators=(",", ":"))]]}]}).encode()
            if len(body) > 4096:
                with self._lock:
                    self.stats["dropped"] += 1
                return False
        except (TypeError, ValueError):
            return False
        with self._lock:
            if self._stopped.is_set():
                return False
            if self._worker is None:
                self._worker = threading.Thread(target=self._run, name="loki-export", daemon=True)
                try:
                    self._worker.start()
                except RuntimeError:
                    self._worker = None
                    self.stats["dropped"] += 1
                    return False
            try:
                self._queue.put_nowait((url, auth, body))
                self.stats["accepted"] += 1
                return True
            except queue.Full:
                self.stats["dropped"] += 1
                return False

    def _run(self):
        while not self._stopped.is_set() or not self._queue.empty():
            try:
                item = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                ok = self._sender(*item)
            except Exception:
                ok = False  # Never recursively log secrets or exporter errors.
            with self._lock:
                self.stats["sent" if ok else "failed"] += 1
            self._queue.task_done()

    def flush(self, timeout=1.75):
        deadline = time.monotonic() + timeout
        with self._queue.all_tasks_done:
            while self._queue.unfinished_tasks:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._queue.all_tasks_done.wait(remaining)
        return True

    async def flush_async(self, timeout=1.75):
        deadline = time.monotonic() + timeout
        while self._queue.unfinished_tasks and time.monotonic() < deadline:
            await asyncio.sleep(0.01)
        return not self._queue.unfinished_tasks

    def stop(self):
        self._stopped.set()


class ObserveASGI:
    """Observe auth rejections, failures and lifespan without changing responses."""
    def __init__(self, app, emitter):
        self.app = app
        self.emitter = emitter

    async def __call__(self, scope, receive, send):
        if scope["type"] == "lifespan":
            async def lifecycle_send(message):
                kind = message["type"]
                events = {"lifespan.startup.complete": "service_started", "lifespan.startup.failed": "startup_failed",
                          "lifespan.shutdown.complete": "service_stopped", "lifespan.shutdown.failed": "shutdown_failed"}
                if kind in events:
                    self.emitter.emit(events[kind])
                if kind in {"lifespan.startup.failed", "lifespan.shutdown.complete", "lifespan.shutdown.failed"}:
                    await self.emitter.flush_async()
                    self.emitter.stop()
                await send(message)
            return await self.app(scope, receive, lifecycle_send)
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        started = time.perf_counter()
        status = 500
        error_type = None
        async def observed_send(message):
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
            await send(message)
        try:
            return await self.app(scope, receive, observed_send)
        except Exception as exc:
            error_type = type(exc).__name__
            raise
        finally:
            fields = {"method": scope.get("method", "UNKNOWN"),
                      "route": getattr(scope.get("route"), "path", None) or "unmatched",
                      "status": status, "duration_ms": round((time.perf_counter() - started) * 1000, 1)}
            if error_type:
                fields["error_type"] = error_type
            self.emitter.emit("http_request", **fields)


def _send_trusted(url, auth, body):
    """Use a worker-local general profile for one validated Grafana destination.

    Do not disable the global guard or mutate research allowlists. Public-IP
    verification and the socket-level SSRF checks remain active in this worker.
    """
    target = urlsplit(url)
    if (target.scheme != "https" or not target.hostname or not target.hostname.endswith(".grafana.net")
            or target.port not in (None, 443) or target.path != "/loki/api/v1/push"
            or target.username or target.password or target.query or target.fragment):
        return False
    from app import security
    enabled = security.egress_protection_enabled.set(True)
    profile = security.active_profile.set("general")
    try:
        if not security.is_safe_egress_url(url):
            return False
        return _send_http(url, auth, body)
    finally:
        security.active_profile.reset(profile)
        security.egress_protection_enabled.reset(enabled)


emitter = LokiEmitter("web-intelligence-agent", sender=_send_trusted)
emit = emitter.emit
