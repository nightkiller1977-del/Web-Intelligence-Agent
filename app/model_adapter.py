# app/model_adapter.py
import contextlib
import os
import contextvars
from typing import Dict, Optional
from app.config import raw_header_credentials_allowed

# ContextVar storing a dict of request-scoped environment overrides
request_env = contextvars.ContextVar("request_env", default=None)

# Backup original os.environ methods to delegate back
_orig_getitem = os._Environ.__getitem__
_orig_get = os._Environ.get
_orig_contains = os._Environ.__contains__

def patched_getitem(self, key):
    overrides = request_env.get()
    if overrides and key in overrides:
        return overrides[key]
    return _orig_getitem(self, key)

def patched_get(self, key, default=None):
    overrides = request_env.get()
    if overrides and key in overrides:
        return overrides[key]
    return _orig_get(self, key, default)

def patched_contains(self, key):
    overrides = request_env.get()
    if overrides and key in overrides:
        return True
    return _orig_contains(self, key)

# Apply process-wide safe monkey patches to os.environ mapping
os._Environ.__getitem__ = patched_getitem
os._Environ.get = patched_get
os._Environ.__contains__ = patched_contains

class RequestEnvironmentManager:
    """
    Safely applies request-specific LLM and Search credentials to the task-local context
    without mutating the process-wide os.environ, preventing credential leaks or race conditions.
    """
    def __init__(self, headers: Dict[str, str]):
        self.headers = headers
        self.key_map = {
            "X-LLM-Key": "OPENAI_API_KEY",
            "X-Search-Key": "TAVILY_API_KEY"
        }

    @contextlib.contextmanager
    def apply_keys(self):
        if not raw_header_credentials_allowed():
            yield
            return

        # 1. Read loopback headers
        overrides = {}
        for header_name, env_name in self.key_map.items():
            val = self.headers.get(header_name) or self.headers.get(header_name.lower())
            if val:
                overrides[env_name] = val

        # 2. Inherit current overrides and merge
        current = request_env.get()
        new_env = dict(current) if current else {}
        new_env.update(overrides)

        # 3. Apply context variable token
        token = request_env.set(new_env)
        try:
            yield
        finally:
            # 4. Restore original task context state
            request_env.reset(token)
