import asyncio
import sys
import types

sys.modules.setdefault(
    "app.config",
    types.SimpleNamespace(settings=types.SimpleNamespace(STORAGE_BACKEND="memory", REDIS_URL=None))
)

from app.storage import InMemoryStorage


def test_operation_id_claim_rejects_different_idempotency_key():
    async def run():
        storage = InMemoryStorage()

        assert await storage.claim_operation_id("op-1", "idem-a") is True
        assert await storage.claim_operation_id("op-1", "idem-a") is True
        assert await storage.claim_operation_id("op-1", "idem-b") is False

    asyncio.run(run())
