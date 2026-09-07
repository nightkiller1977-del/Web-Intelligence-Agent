import json
import os
import threading
import time
import urllib.request

SERVICE = "web-intelligence-agent"


def emit(event: str, **fields) -> None:
    url = os.getenv("LOKI_URL_REMOTE", "").strip()
    auth = os.getenv("LOKI_REMOTE_AUTH", "").strip()
    if not url or not auth:
        return

    safe = {"service": SERVICE, "event": event, "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    blocked = {"token", "password", "prompt", "content", "body", "query", "headers"}
    for key, value in fields.items():
        if key.lower() in blocked:
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            safe[key] = value

    payload = {
        "streams": [{
            "stream": {
                "application": "ai-agents",
                "agent": SERVICE,
                "environment": os.getenv("ENVIRONMENT", "production"),
            },
            "values": [[str(time.time_ns()), json.dumps(safe, separators=(",", ":"))]],
        }]
    }

    def _send() -> None:
        try:
            request = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                method="POST",
                headers={"Content-Type": "application/json", "Authorization": auth},
            )
            with urllib.request.urlopen(request, timeout=1.5):
                pass
        except Exception:
            pass

    threading.Thread(target=_send, daemon=True).start()
