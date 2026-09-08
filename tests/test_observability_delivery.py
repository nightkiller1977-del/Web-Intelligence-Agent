import asyncio
import importlib.util
import json
import os
from pathlib import Path
import threading
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1] / "app/grafana_observability.py"
spec = importlib.util.spec_from_file_location("observability_under_test", SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class TransportTests(unittest.TestCase):
    def test_one_worker_bounded_queue_and_redaction(self):
        release = threading.Event()
        received = []
        def sender(url, auth, body):
            release.wait(1)
            received.append(json.loads(body))
            return True
        emitter = module.LokiEmitter("test-agent", sender=sender, capacity=3)
        with patch.dict(os.environ, {"LOKI_URL_REMOTE": "https://logs.example.grafana.net/loki/api/v1/push", "LOKI_REMOTE_AUTH": "Basic dGVzdDp0ZXN0", "OBSERVABILITY_REMOTE": "1"}):
            try:
                accepted = sum(emitter.emit("test", status=200, body="PRIVATE", token="PRIVATE") for _ in range(100))
                self.assertLessEqual(accepted, 4)
                self.assertLessEqual(emitter._queue.qsize(), 3)
                self.assertTrue(emitter._worker.is_alive())
                self.assertFalse(emitter.flush(0.01))
            finally:
                release.set()
                self.assertTrue(emitter.flush(2))
                emitter.stop()
                emitter._worker.join(1)
        self.assertGreater(emitter.stats["dropped"], 0)
        self.assertNotIn("PRIVATE", json.dumps(received))

    def test_unconfigured_and_unsafe_target_start_no_thread(self):
        emitter = module.LokiEmitter("test-agent")
        with patch.dict(os.environ, {"LOKI_URL_REMOTE": "", "LOKI_REMOTE_AUTH": "", "OBSERVABILITY_REMOTE": "1"}):
            self.assertFalse(emitter.emit("test"))
        with patch.dict(os.environ, {"LOKI_URL_REMOTE": "http://example.com", "LOKI_REMOTE_AUTH": "Basic dGVzdDp0ZXN0", "OBSERVABILITY_REMOTE": "1"}):
            self.assertFalse(emitter.emit("test"))
        self.assertIsNone(emitter._worker)

    def test_malformed_auth_never_starts_worker(self):
        emitter = module.LokiEmitter("test-agent")
        url = "https://logs.example.grafana.net/loki/api/v1/push"
        for bad in ["Bearer dGVzdDp0ZXN0", "Basic !!!notbase64!!!", "Basic dGVzdA==",  # no colon
                    "Basic dGVzdDo=", "Basic OnRlc3Q=",  # empty password / user
                    "Basic dGVzdDp0ZXN0\r\nX: y", "******", "basic"]:
            with patch.dict(os.environ, {"LOKI_URL_REMOTE": url, "LOKI_REMOTE_AUTH": bad, "OBSERVABILITY_REMOTE": "1"}):
                self.assertFalse(emitter.emit("test"), bad)
            self.assertFalse(module.validate_basic_auth(bad), bad)
        self.assertIsNone(emitter._worker)
        self.assertEqual(emitter.stats["accepted"], 0)

    def test_partial_pair_disables_export(self):
        emitter = module.LokiEmitter("test-agent")
        url = "https://logs.example.grafana.net/loki/api/v1/push"
        with patch.dict(os.environ, {"LOKI_URL_REMOTE": url, "LOKI_REMOTE_AUTH": "", "OBSERVABILITY_REMOTE": "1"}):
            self.assertFalse(emitter.emit("test"))
            self.assertEqual(module.resolve_loki_config().reason, "partial_pair")
        with patch.dict(os.environ, {"LOKI_URL_REMOTE": "", "LOKI_REMOTE_AUTH": "Basic dGVzdDp0ZXN0", "OBSERVABILITY_REMOTE": "1"}):
            self.assertFalse(emitter.emit("test"))
            self.assertEqual(module.resolve_loki_config().reason, "partial_pair")
        self.assertIsNone(emitter._worker)

    def test_oversized_body_increments_dropped_counter(self):
        emitter = module.LokiEmitter("test-agent", sender=lambda *a: True)
        with patch.dict(os.environ, {"LOKI_URL_REMOTE": "https://logs.example.grafana.net/loki/api/v1/push", "LOKI_REMOTE_AUTH": "Basic dGVzdDp0ZXN0", "OBSERVABILITY_REMOTE": "1"}):
            self.assertFalse(emitter.emit("x" * 5000))
        self.assertEqual(emitter.stats["dropped"], 1)

class PolicyTests(unittest.TestCase):
    URL = "https://logs.example.grafana.net/loki/api/v1/push"
    AUTH = "Basic dGVzdDp0ZXN0"

    def test_dev_test_default_off_even_with_valid_pair(self):
        env = {"LOKI_URL_REMOTE": self.URL, "LOKI_REMOTE_AUTH": self.AUTH}
        with patch.dict(os.environ, env, clear=True):
            config = module.resolve_loki_config()
            self.assertFalse(config.enabled)
            self.assertEqual(config.reason, "non_production_default_off")
            emitter = module.LokiEmitter("test-agent")
            self.assertFalse(emitter.emit("test"))
            self.assertIsNone(emitter._worker)

    def test_production_render_deployment_mode_auto_on(self):
        with patch.dict(os.environ, {"LOKI_URL_REMOTE": self.URL, "LOKI_REMOTE_AUTH": self.AUTH, "DEPLOYMENT_MODE": "remote"}, clear=True):
            config = module.resolve_loki_config()
            self.assertTrue(config.enabled)
            self.assertEqual((config.url, config.auth, config.source), (self.URL, self.AUTH, "env"))
        with patch.dict(os.environ, {"LOKI_URL_REMOTE": self.URL, "LOKI_REMOTE_AUTH": self.AUTH, "RENDER": "true"}, clear=True):
            self.assertTrue(module.resolve_loki_config().enabled)

    def test_opt_out_wins_even_in_production(self):
        with patch.dict(os.environ, {"LOKI_URL_REMOTE": self.URL, "LOKI_REMOTE_AUTH": self.AUTH, "DEPLOYMENT_MODE": "remote", "OBSERVABILITY_REMOTE": "0"}, clear=True):
            config = module.resolve_loki_config()
            self.assertFalse(config.enabled)
            self.assertEqual(config.reason, "opted_out")

    def test_explicit_opt_in_enables_outside_production(self):
        with patch.dict(os.environ, {"LOKI_URL_REMOTE": self.URL, "LOKI_REMOTE_AUTH": self.AUTH, "OBSERVABILITY_REMOTE": "1"}, clear=True):
            self.assertTrue(module.resolve_loki_config().enabled)

    def test_partial_pair_warns_once_and_never_logs_values(self):
        module.reset_warnings()
        with patch.dict(os.environ, {"LOKI_URL_REMOTE": "https://SECRETVALUE.example/push", "DEPLOYMENT_MODE": "remote"}, clear=True):
            with self.assertLogs("grafana-observability", level="WARNING") as captured:
                module.resolve_loki_config()
                module.resolve_loki_config()
        self.assertEqual(len(captured.records), 1)
        self.assertNotIn("SECRETVALUE", "\n".join(captured.output))

class ASGITests(unittest.IsolatedAsyncioTestCase):
    async def test_exception_is_observed_and_reraised_without_raw_path(self):
        events = []
        class Emitter:
            def emit(self, event, **fields): events.append((event, fields))
        async def broken(scope, receive, send): raise ValueError("PRIVATE")
        app = module.ObserveASGI(broken, Emitter())
        with self.assertRaises(ValueError):
            await app({"type": "http", "method": "GET", "path": "/private@example.com"}, None, None)
        self.assertEqual(events[0][1]["status"], 500)
        self.assertEqual(events[0][1]["route"], "unmatched")
        self.assertNotIn("PRIVATE", repr(events))
        self.assertNotIn("private@example.com", repr(events))

    async def test_auth_rejection_and_lifespan_are_observed(self):
        events = []
        class Emitter:
            def emit(self, event, **fields): events.append((event, fields))
            async def flush_async(self): events.append(("flushed", {}))
            def stop(self): events.append(("stopped", {}))
        async def rejected(scope, receive, send): await send({"type": "http.response.start", "status": 401})
        async def output(message): pass
        await module.ObserveASGI(rejected, Emitter())({"type": "http", "method": "GET"}, None, output)
        self.assertEqual(events[0][1]["status"], 401)
        async def lifecycle(scope, receive, send):
            await send({"type": "lifespan.startup.complete"})
            await send({"type": "lifespan.shutdown.complete"})
        await module.ObserveASGI(lifecycle, Emitter())({"type": "lifespan"}, None, output)
        self.assertEqual([event for event, _ in events][1:], ["service_started", "service_stopped", "flushed", "stopped"])

    async def test_matched_route_template_is_emitted_not_raw_path(self):
        events = []
        class Emitter:
            def emit(self, event, **fields): events.append((event, fields))
        async def ok(scope, receive, send):
            await send({"type": "http.response.start", "status": 204})
        class Route:
            path = "/operations/{operation_id}"
        app = module.ObserveASGI(ok, Emitter())
        async def output(message): pass
        await app({"type": "http", "method": "GET", "path": "/operations/private@example.com", "route": Route()}, None, output)
        self.assertEqual(events[0][1]["route"], "/operations/{operation_id}")
        self.assertNotIn("private@example.com", repr(events))

class EgressTests(unittest.TestCase):
    def test_worker_context_is_scoped_and_still_rejects_private_targets(self):
        import contextvars
        import sys
        import types
        security = types.ModuleType("app.security")
        security.active_profile = contextvars.ContextVar("profile", default="technical")
        security.egress_protection_enabled = contextvars.ContextVar("enabled", default=False)
        security.is_safe_egress_url = lambda url: True
        app = types.ModuleType("app"); app.security = security
        calls = []
        def send(*args):
            calls.append(args)
            self.assertEqual(security.active_profile.get(), "general")
            self.assertTrue(security.egress_protection_enabled.get())
            return True
        url = "https://logs.example.grafana.net/loki/api/v1/push"
        with patch.dict(sys.modules, {"app": app, "app.security": security}), patch.object(module, "_send_http", send):
            self.assertTrue(module._send_trusted(url, "test-auth", b"{}"))
            self.assertEqual(security.active_profile.get(), "technical")
            self.assertFalse(security.egress_protection_enabled.get())
            security.is_safe_egress_url = lambda url: False
            self.assertFalse(module._send_trusted(url, "test-auth", b"{}"))
            self.assertFalse(module._send_trusted("https://evil.example/loki/api/v1/push", "test-auth", b"{}"))
        self.assertEqual(len(calls), 1)

if __name__ == "__main__": unittest.main()
