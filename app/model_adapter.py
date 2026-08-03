# app/model_adapter.py
import contextlib
import os
from typing import Dict, Optional

class RequestEnvironmentManager:
    """
    Safely applies request-specific LLM and Search credentials to the process environment
    and restores the original state after execution, preventing race conditions or leaks.
    """
    def __init__(self, headers: Dict[str, str]):
        self.headers = headers
        self.original_env: Dict[str, Optional[str]] = {}
        
        # Mapping incoming loopback headers to expected library environment keys
        self.key_map = {
            "X-LLM-Key": "OPENAI_API_KEY",
            "X-Search-Key": "TAVILY_API_KEY"
        }

    @contextlib.contextmanager
    def apply_keys(self):
        try:
            # 1. Record original state and apply new keys
            for header_name, env_name in self.key_map.items():
                # Read from request headers
                val = self.headers.get(header_name) or self.headers.get(header_name.lower())
                if val:
                    self.original_env[env_name] = os.environ.get(env_name)
                    os.environ[env_name] = val
            
            yield
        finally:
            # 2. Restore environment to original state
            for env_name, original_val in self.original_env.items():
                if original_val is None:
                    os.environ.pop(env_name, None)
                else:
                    os.environ[env_name] = original_val
