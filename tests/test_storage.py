import asyncio

import app.storage as storage_module
from app.storage import InMemoryStorage, OPERATION_CACHE_LIMIT


def test_operation_id_claim_rejects_different_idempotency_key():
    async def run():
        storage = InMemoryStorage()

        assert await storage.claim_operation_id("op-1", "idem-a") is True
        assert await storage.claim_operation_id("op-1", "idem-a") is True
        assert await storage.claim_operation_id("op-1", "idem-b") is False

    asyncio.run(run())


def test_operation_cache_is_bounded():
    async def run():
        storage = InMemoryStorage()
        for index in range(OPERATION_CACHE_LIMIT + 25):
            await storage.save_operation(f"op-{index}", {"status": "completed"})

        assert len(storage.operations) <= OPERATION_CACHE_LIMIT
        # Oldest entries are evicted; the newest remain queryable.
        assert await storage.get_operation("op-0") is None
        assert await storage.get_operation(f"op-{OPERATION_CACHE_LIMIT + 24}") is not None

    asyncio.run(run())


def test_mark_stale_operations_ignores_internal_metadata_entries():
    async def run():
        storage = InMemoryStorage()
        await storage.save_operation("op-live", {"status": "running"})
        await storage.begin_operation("op-owned")

        await storage.mark_stale_operations()

        assert (await storage.get_operation("op-live"))["status"] == "failed"
        # The ownership metadata entry is not an operation and must be skipped.
        assert not any(
            key.startswith(storage_module.OPERATION_METADATA_PREFIX) and value.get("status")
            for key, value in storage.operations.items()
        )

    asyncio.run(run())


def test_inmemory_concurrency_slot_is_bounded():
    async def run():
        storage = InMemoryStorage()
        acquired = [await storage.acquire_concurrency_slot() for _ in range(5)]

        assert acquired[:3] == [True, True, True]
        assert acquired[3:] == [False, False]

        await storage.release_concurrency_slot()
        assert await storage.acquire_concurrency_slot() is True

    asyncio.run(run())


def test_begin_operation_then_release_lease():
    async def run():
        storage = InMemoryStorage()
        assert await storage.begin_operation("op-lease") is True
        await storage.release_operation_lease("op-lease")
        assert storage.operations.get(f"{storage_module.OPERATION_METADATA_PREFIX}op-lease") is None

    asyncio.run(run())
