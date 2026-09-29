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


def test_mark_stale_operations_marks_queued_and_running_failed():
    async def run():
        storage = InMemoryStorage()
        await storage.save_operation("op-live", {"status": "running"})

        await storage.mark_stale_operations()

        assert (await storage.get_operation("op-live"))["status"] == "failed"

    asyncio.run(run())


def test_operation_id_cannot_collide_with_internal_metadata():
    async def run():
        storage = InMemoryStorage()
        # A caller-supplied id that looks like internal metadata is a normal
        # operation, not a namespace collision.
        await storage.save_operation("__meta:foo", {"status": "completed"})
        assert await storage.begin_operation("foo") is True
        assert (await storage.get_operation("__meta:foo"))["status"] == "completed"
        assert storage.operation_owners["foo"] == storage.instance.instance_id

    asyncio.run(run())


def test_inmemory_concurrency_slot_is_bounded():
    async def run():
        storage = InMemoryStorage()
        acquired = [await storage.acquire_concurrency_slot(f"op-{i}") for i in range(5)]

        assert acquired[:3] == [True, True, True]
        assert acquired[3:] == [False, False]

        # Each release frees exactly its own slot; releasing an unheld slot is a
        # no-op and must not steal another operation's slot.
        await storage.release_concurrency_slot("op-not-held")
        assert await storage.acquire_concurrency_slot("op-4") is False
        await storage.release_concurrency_slot("op-0")
        assert await storage.acquire_concurrency_slot("op-4") is True

    asyncio.run(run())


def test_begin_operation_then_release_lease():
    async def run():
        storage = InMemoryStorage()
        assert await storage.begin_operation("op-lease") is True
        assert storage.operation_owners["op-lease"] == storage.instance.instance_id
        await storage.release_operation_lease("op-lease")
        assert "op-lease" not in storage.operation_owners

    asyncio.run(run())


def test_eviction_keeps_idempotency_reservation():
    async def run():
        storage = InMemoryStorage()
        await storage.claim_idempotency_key("key-1", "op-old")
        # Force the result payload out of the cache while the reservation stays.
        for i in range(storage_module.OPERATION_CACHE_LIMIT + 5):
            await storage.save_operation(f"op-{i}", {"status": "completed"})
        assert await storage.get_operation("op-old") is None
        assert await storage.claim_idempotency_key("key-1", "op-new") == "op-old"

    asyncio.run(run())


def test_eviction_preserves_live_operations():
    async def run():
        storage = InMemoryStorage()
        await storage.save_operation("op-live", {"status": "running"})
        for i in range(storage_module.OPERATION_CACHE_LIMIT + 20):
            await storage.save_operation(f"op-done-{i}", {"status": "completed"})

        # A running operation must never be dropped while it is still executing.
        assert (await storage.get_operation("op-live"))["status"] == "running"

    asyncio.run(run())
