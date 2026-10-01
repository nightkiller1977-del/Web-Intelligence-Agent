# app/model_adapter.py
import contextlib
import os
import contextvars
from typing import Dict, Optional
from app.config import (
    EXTERNAL_PROVIDER_CREDENTIALS,
    LOCAL_MODEL_PROVIDERS,
    external_openai_gateway_config,
    local_model_endpoint,
    raw_header_credentials_allowed,
    settings,
)

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
    Safely applies request-specific model and search credentials to the task-local context
    without mutating the process-wide os.environ, preventing credential leaks or race conditions.
    """
    def __init__(self, headers: Dict[str, str], model_provider: Optional[str] = None, model_name: Optional[str] = None):
        self.headers = headers
        self.model_provider = model_provider
        self.model_name = model_name
        self.key_map = {
            "X-Search-Key": "TAVILY_API_KEY"
        }

    @contextlib.contextmanager
    def apply_keys(self):
        overrides = {}
        try:
            gateway = external_openai_gateway_config()
        except ValueError:
            # A malformed gateway config is rejected at admission; treat it as
            # absent here so a request context is never built around it.
            gateway = None
        requested_provider = (self.model_provider or "").lower()
        # A provider name alone does not establish locality: only treat a "local"
        # selection as local when the configured endpoint really is local, so it
        # cannot become an unmetered external egress path.
        local_model = requested_provider in LOCAL_MODEL_PROVIDERS and local_model_endpoint() is not None
        # GPT Researcher reads model tiers from the environment and will use
        # whatever provider each tier names with that provider's ambient
        # credential. Every branch below therefore pins *all three* tiers and
        # masks every external-provider credential, so no ambient default can
        # reach a provider outside the approved path.
        if gateway and not local_model:
            overrides["OPENAI_BASE_URL"] = gateway.base_url
            overrides["OPENAI_API_KEY"] = gateway.api_key
            # The gateway is the only permitted external-model egress, so pin the
            # model selection to it. Otherwise an ambient FAST_LLM/SMART_LLM
            # naming another provider would let GPT Researcher call that provider
            # directly with an ambient credential, bypassing the gateway.
            model = self.model_name if (self.model_provider and self.model_name) else settings.AI_OPENROUTER_DEFAULT_MODEL
            llm = f"openai:{model}"
            overrides["FAST_LLM"] = llm
            overrides["SMART_LLM"] = llm
            # Keep the strategic tier on the gateway too; embeddings follow
            # OPENAI_BASE_URL when they are OpenAI-backed and stay local otherwise.
            overrides["STRATEGIC_LLM"] = llm
            # EMBEDDING is an independent setting defaulting to an OpenAI
            # embedding; pin it so it cannot use an unmasked ambient credential.
            overrides["EMBEDDING"] = f"openai:{settings.OPENAI_EMBEDDING_MODEL}"
            # The deprecated EMBEDDING_PROVIDER, when set, overrides EMBEDDING, so
            # pin it to the same gateway-backed provider.
            overrides["EMBEDDING_PROVIDER"] = "openai"
        elif self.model_provider and self.model_name:
            # Explicit selection with no external gateway: the only provider that
            # needs no egress is a local one. Pin all tiers to it so an ambient
            # external tier cannot still be used. Use the normalized provider:
            # admission lowercases it, and GPT Researcher's provider lookup is
            # case-sensitive, so the original casing could fail in the background.
            llm = f"{requested_provider}:{self.model_name}"
            overrides["FAST_LLM"] = llm
            overrides["SMART_LLM"] = llm
            overrides["STRATEGIC_LLM"] = llm
            if local_model:
                overrides["OLLAMA_BASE_URL"] = settings.OLLAMA_BASE_URL
                overrides["EMBEDDING"] = f"ollama:{settings.LOCAL_EMBEDDING_MODEL}"
                overrides["EMBEDDING_PROVIDER"] = "ollama"
                overrides["OLLAMA_EMBEDDING_MODEL"] = settings.LOCAL_EMBEDDING_MODEL
        else:
            # No explicit selection and no gateway: fall back to the local model
            # rather than any ambient external default.
            llm = f"ollama:{settings.LOCAL_DEFAULT_MODEL}"
            overrides["FAST_LLM"] = llm
            overrides["SMART_LLM"] = llm
            overrides["STRATEGIC_LLM"] = llm
            overrides["OLLAMA_BASE_URL"] = settings.OLLAMA_BASE_URL
            overrides["EMBEDDING"] = f"ollama:{settings.LOCAL_EMBEDDING_MODEL}"
            overrides["EMBEDDING_PROVIDER"] = "ollama"
            overrides["OLLAMA_EMBEDDING_MODEL"] = settings.LOCAL_EMBEDDING_MODEL
        # Mask every external-provider credential not already set by this branch.
        for credential in EXTERNAL_PROVIDER_CREDENTIALS:
            if credential not in overrides:
                overrides[credential] = ""
        if raw_header_credentials_allowed():
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
