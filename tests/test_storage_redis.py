"""
tests/test_storage_redis.py

Real behavioral coverage for app.storage.RedisStorage, which previously had
zero exercised coverage (the only Redis-locking test in test_remote_mode.py
is gated on STORAGE_BACKEND=redis and always skips locally).

These tests run RedisStorage against fakeredis's redis.asyncio-compatible
FakeRedis client instead of a real Redis/Valkey server. fakeredis's EVAL
support (needed for RedisStorage.claim_operation_id's atomic Lua script,
including its cjson.decode call) requires the optional `lupa` dependency -
see requirements-test.txt. This was verified directly against fakeredis
before writing these tests: without lupa installed, fakeredis raises
"unknown command 'eval'" for any EVAL call. With lupa installed, the exact
Lua source from app/storage.py runs unmodified and produces the same
results it would against real Redis.
"""

import asyncio

import fakeredis.aioredis as fakeredis_aioredis
import pytest
import redis.asyncio as redis_asyncio_module

from app.config import settings
from app.storage import RedisStorage


@pytest.fixture
async def redis_storage(monkeypatch):
    """A RedisStorage instance backed by a real fakeredis FakeRedis client.

    Monkeypatches redis.asyncio.from_url (the exact call RedisStorage.init
    makes) so RedisStorage's production code path - including the real
    EVAL-based Lua script - runs unmodified against fakeredis instead of a
    real server.
    """
    monkeypatch.setattr(settings, "REDIS_URL", "redis://fake-host:6379/0")

    def fake_from_url(url, decode_responses=True, **kwargs):
        # redis.asyncio.from_url is a synchronous factory (it does not return
        # a coroutine) - app/storage.py calls it without awaiting, so the
        # stand-in must match that exact calling convention.
        return fakeredis_aioredis.FakeRedis(decode_responses=decode_responses)

    monkeypatch.setattr(redis_asyncio_module, "from_url", fake_from_url)

    storage = RedisStorage()
    await storage.init()
    assert storage.degraded is False, "fixture setup: fakeredis should connect successfully"

    yield storage

    await storage.redis.aclose()


@pytest.mark.anyio
async def test_claim_operation_id_lua_script_rejects_conflicting_idempotency_key(redis_storage):
    """The Lua script's cjson.decode branch must reject a claim whose
    idempotency key conflicts with the idempotency_key already recorded on
    the operation's saved hash entry - even before any operation_claims key
    has ever been set for that operation id."""
    await redis_storage.save_operation("op-1", {"idempotency_key": "idem-a", "status": "queued"})

    assert await redis_storage.claim_operation_id("op-1", "idem-b") is False

    # The matching key is still accepted afterwards.
    assert await redis_storage.claim_operation_id("op-1", "idem-a") is True


@pytest.mark.anyio
async def test_claim_operation_id_is_reentrant_for_same_key(redis_storage):
    assert await redis_storage.claim_operation_id("op-2", "idem-x") is True
    # Reentrant: repeating the same (op_id, idempotency_key) pair succeeds.
    assert await redis_storage.claim_operation_id("op-2", "idem-x") is True
    # A different idempotency key is rejected once a claim is held.
    assert await redis_storage.claim_operation_id("op-2", "idem-y") is False

    claim_key = "research:operation_claims:op-2"
    stored_value = await redis_storage.redis.get(claim_key)
    assert stored_value == "idem-x"
    ttl = await redis_storage.redis.ttl(claim_key)
    assert 0 < ttl <= 86400, "claim key must carry the script's 86400s EX TTL"


@pytest.mark.anyio
async def test_claim_idempotency_key_uses_atomic_set_nx(redis_storage):
    # First claim wins and signals success via a None return.
    assert await redis_storage.claim_idempotency_key("idem-key-1", "op-a") is None

    # A second claim for the same key from a different operation must not
    # overwrite the winner - this is exactly what SET ... NX guarantees.
    assert await redis_storage.claim_idempotency_key("idem-key-1", "op-b") == "op-a"

    idem_key = "research:idempotency:idem-key-1"
    assert await redis_storage.redis.get(idem_key) == "op-a"
    ttl = await redis_storage.redis.ttl(idem_key)
    assert 0 < ttl <= 86400, "idempotency key must carry the 86400s EX TTL"


@pytest.mark.anyio
async def test_push_progress_event_uses_xadd_with_maxlen_cap(redis_storage):
    op_id = "op-stream"
    total_events = 1200  # deliberately > the hardcoded maxlen=1000 cap

    for i in range(total_events):
        await redis_storage.push_progress_event(op_id, {"stage": "searching", "message": f"event-{i}"})

    stream_key = f"research:events:{op_id}"
    raw_length = await redis_storage.redis.xlen(stream_key)
    assert raw_length == 1000, "XADD MAXLEN=1000 should cap the stream length"

    events = await redis_storage.get_progress_events(op_id)
    assert len(events) == 1000
    # FIFO eviction: the oldest 200 events are gone, newest event is retained.
    assert events[0]["message"] == "event-200"
    assert events[-1]["message"] == f"event-{total_events - 1}"


@pytest.mark.anyio
async def test_redis_storage_degrades_to_inmemory_when_ping_fails(monkeypatch):
    monkeypatch.setattr(settings, "REDIS_URL", "redis://fake-host:6379/0")

    class FailingRedisClient:
        async def ping(self):
            raise ConnectionError("simulated connection failure")

    def fake_from_url(url, decode_responses=True, **kwargs):
        return FailingRedisClient()

    monkeypatch.setattr(redis_asyncio_module, "from_url", fake_from_url)

    storage = RedisStorage()
    await storage.init()

    assert storage.degraded is True

    # Confirm degradation is not just a flag: reads/writes actually go
    # through the in-memory fallback and round-trip correctly.
    await storage.save_operation("op-fallback", {"status": "queued"})
    result = await storage.get_operation("op-fallback")
    assert result == {"status": "queued"}
    assert storage.fallback.operations.get("op-fallback") == {"status": "queued"}
