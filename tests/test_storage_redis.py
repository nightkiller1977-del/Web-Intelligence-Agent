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
import json

import fakeredis.aioredis as fakeredis_aioredis
import pytest
import redis.asyncio as redis_asyncio_module

from app.config import settings
from app.storage import _FAIL_IF_UNOWNED_LUA, RedisStorage, StorageUnavailable


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


@pytest.mark.anyio
async def test_redis_storage_fails_closed_when_required_and_unreachable(monkeypatch):
    monkeypatch.setattr(settings, "REDIS_REQUIRED", True)
    monkeypatch.setattr(settings, "REDIS_URL", "")

    storage = RedisStorage()
    with pytest.raises(StorageUnavailable):
        await storage.init()

    assert storage.degraded is False


@pytest.mark.anyio
async def test_mark_stale_operations_respects_live_owner_lease(redis_storage, monkeypatch):
    # An operation owned by another live instance must not be marked stale.
    await redis_storage.save_operation("op-owned", {"status": "running"})
    await redis_storage.redis.set("research:owners:op-owned", "other-instance:1:1", ex=60)

    # An operation with no owner lease is genuinely abandoned.
    await redis_storage.save_operation("op-abandoned", {"status": "running"})

    await redis_storage.mark_stale_operations()

    assert (await redis_storage.get_operation("op-owned"))["status"] == "running"
    assert (await redis_storage.get_operation("op-abandoned"))["status"] == "failed"


@pytest.mark.anyio
async def test_redis_concurrency_slot_is_service_wide(redis_storage, monkeypatch):
    monkeypatch.setattr(settings, "MAX_CONCURRENT_OPS", 2)

    assert await redis_storage.acquire_concurrency_slot("op-a") is True
    assert await redis_storage.acquire_concurrency_slot("op-b") is True
    assert await redis_storage.acquire_concurrency_slot("op-c") is False

    await redis_storage.release_concurrency_slot("op-a")
    assert await redis_storage.acquire_concurrency_slot("op-c") is True


@pytest.mark.anyio
async def test_redis_begin_operation_lease_is_exclusive(redis_storage):
    assert await redis_storage.begin_operation("op-exclusive") is True

    # A second acquisition attempt (as a peer instance would make) must fail
    # while the first lease is live.
    peer = RedisStorage()
    peer.redis = redis_storage.redis
    assert await peer.begin_operation("op-exclusive") is False

    await redis_storage.release_operation_lease("op-exclusive")
    assert await peer.begin_operation("op-exclusive") is True


@pytest.mark.anyio
async def test_redis_stale_scan_does_not_fail_live_owned_operation(redis_storage):
    # A queued operation with a live owner lease must survive reconciliation:
    # the owner-check and state-write are atomic, so a peer cannot fail work
    # that is actively owned.
    await redis_storage.save_operation("op-live", {"status": "queued"})
    assert await redis_storage.begin_operation("op-live") is True

    await redis_storage.mark_stale_operations()

    assert (await redis_storage.get_operation("op-live"))["status"] == "queued"


@pytest.mark.anyio
async def test_redis_stale_scan_fails_ownerless_operation(redis_storage):
    await redis_storage.save_operation("op-orphan", {"status": "queued"})

    await redis_storage.mark_stale_operations()

    assert (await redis_storage.get_operation("op-orphan"))["status"] == "failed"


@pytest.mark.anyio
async def test_redis_release_lease_is_compare_and_delete(redis_storage):
    peer = RedisStorage()
    peer.redis = redis_storage.redis

    assert await redis_storage.begin_operation("op-cad") is True
    # A peer that does not own the lease must not clear it.
    await peer.release_operation_lease("op-cad")
    assert await redis_storage.redis.get("research:owners:op-cad") is not None

    # The owner can clear its own lease and free it for the peer.
    await redis_storage.release_operation_lease("op-cad")
    assert await redis_storage.redis.get("research:owners:op-cad") is None
    assert await peer.begin_operation("op-cad") is True


@pytest.mark.anyio
async def test_redis_release_slot_is_identity_scoped(redis_storage, monkeypatch):
    monkeypatch.setattr(settings, "MAX_CONCURRENT_OPS", 1)

    assert await redis_storage.acquire_concurrency_slot("op-1") is True
    assert await redis_storage.acquire_concurrency_slot("op-2") is False

    # Releasing a slot this instance never held must not free op-1's slot.
    await redis_storage.release_concurrency_slot("op-not-held")
    assert await redis_storage.acquire_concurrency_slot("op-2") is False

    await redis_storage.release_concurrency_slot("op-1")
    assert await redis_storage.acquire_concurrency_slot("op-2") is True


@pytest.mark.anyio
async def test_redis_expired_slot_is_reclaimed(redis_storage, monkeypatch):
    monkeypatch.setattr(settings, "MAX_CONCURRENT_OPS", 1)
    monkeypatch.setattr(settings, "CONCURRENCY_LEASE_TTL_SECONDS", -1)

    # A slot whose lease has already expired is reclaimed by the next acquire,
    # so a crashed instance cannot pin the ceiling forever.
    assert await redis_storage.acquire_concurrency_slot("op-dead") is True
    assert await redis_storage.acquire_concurrency_slot("op-new") is True


@pytest.mark.anyio
async def test_redis_stale_scan_does_not_overwrite_completed_operation(redis_storage):
    # Simulate the snapshot race: an operation is snapshotted as running, then
    # its worker completes it and releases the lease before the stale scan
    # evaluates it. The atomic recheck must not clobber the completed result.
    await redis_storage.save_operation("op-race", {"status": "running"})
    snapshot = await redis_storage.get_operation("op-race")
    await redis_storage.save_operation("op-race", {"status": "completed", "answer": "done"})

    # Directly exercise the guard: no owner lease, but state is now terminal.
    transitioned = await redis_storage.redis.eval(
        _FAIL_IF_UNOWNED_LUA,
        2,
        "research:owners:op-race",
        "research:operations",
        "op-race",
        json.dumps({**snapshot, "status": "failed"}),
    )
    assert transitioned == 0
    assert (await redis_storage.get_operation("op-race"))["status"] == "completed"


@pytest.mark.anyio
async def test_redis_heartbeat_renews_concurrency_slot(redis_storage, monkeypatch):
    monkeypatch.setattr(settings, "MAX_CONCURRENT_OPS", 1)
    monkeypatch.setattr(settings, "CONCURRENCY_LEASE_TTL_SECONDS", 1)

    assert await redis_storage.acquire_concurrency_slot("op-long") is True
    assert await redis_storage.begin_operation("op-long") is True

    # Force the slot's score into the past to stand in for elapsed time, then
    # heartbeat: the slot must be renewed, not reaped by the next admission.
    await redis_storage.redis.zadd("research:concurrency:slots", {"dead": 0})
    await redis_storage.touch_operation("op-long")

    score = await redis_storage.redis.zscore(
        "research:concurrency:slots", redis_storage.instance.lease_id("op-long")
    )
    assert score is not None and score > 0
@pytest.mark.anyio
async def test_redis_shared_daily_spend_is_shared_and_expiring(redis_storage):
    await redis_storage.add_daily_spend(2.5)
    peer = RedisStorage()
    peer.redis = redis_storage.redis
    assert await peer.get_daily_spend() == 2.5
    # The window key must carry a TTL so the total resets without a local timer.
    assert await redis_storage.redis.ttl("research:spend:daily") > 0


@pytest.mark.anyio
async def test_redis_renew_does_not_resurrect_reaped_slot(redis_storage):
    assert await redis_storage.acquire_concurrency_slot("op-reap") is True
    # Simulate the slot being reaped as expired (e.g. by another admission).
    await redis_storage.redis.zrem("research:concurrency:slots", redis_storage.instance.lease_id("op-reap"))

    await redis_storage.touch_operation("op-reap")

    # The heartbeat must not recreate a slot that was already reclaimed.
    assert await redis_storage.redis.zscore("research:concurrency:slots", redis_storage.instance.lease_id("op-reap")) is None


@pytest.mark.anyio
async def test_redis_reserve_daily_spend_is_atomic(redis_storage):
    assert await redis_storage.reserve_daily_spend(4.0, 5.0) is True
    # Concurrent replicas must not both pass a check that would cross the limit.
    assert await redis_storage.reserve_daily_spend(4.0, 5.0) is False
    assert await redis_storage.get_daily_spend() == 4.0
    await redis_storage.release_daily_spend(4.0)
    assert await redis_storage.get_daily_spend() == 0.0


@pytest.mark.anyio
async def test_redis_reconcile_spend_preserves_window_ttl(redis_storage):
    assert await redis_storage.reserve_daily_spend(4.0, 50.0) is True
    ttl_before = await redis_storage.redis.ttl("research:spend:daily")

    await redis_storage.reconcile_daily_spend(4.0, 1.5, 50.0)

    assert await redis_storage.get_daily_spend() == 1.5
    # Releasing/reconciling must not reset the daily window to a fresh 24h.
    assert await redis_storage.redis.ttl("research:spend:daily") <= ttl_before


@pytest.mark.anyio
async def test_redis_heartbeat_reports_lost_ownership(redis_storage):
    assert await redis_storage.begin_operation("op-hb") is True
    assert await redis_storage.acquire_concurrency_slot("op-hb") is True
    assert await redis_storage.touch_operation("op-hb") is True
    # Simulate another instance taking over the lease.
    await redis_storage.redis.delete("research:owners:op-hb")
    assert await redis_storage.touch_operation("op-hb") is False


@pytest.mark.anyio
async def test_redis_active_index_is_maintained_by_state(redis_storage):
    await redis_storage.save_operation("op-active", {"status": "queued"})
    assert bool(await redis_storage.redis.sismember("research:active_ops", "op-active")) is True

    await redis_storage.save_operation("op-active", {"status": "completed", "answer": "done"})
    # A terminal transition drops the index membership so reconciliation does
    # not keep scanning finished work.
    assert bool(await redis_storage.redis.sismember("research:active_ops", "op-active")) is False


@pytest.mark.anyio
async def test_redis_stale_scan_only_walks_active_operations(redis_storage):
    # A completed historical operation is not in the active index, so the scan
    # never transfers or parses it.
    await redis_storage.save_operation("op-history", {"status": "completed", "answer": "x" * 1000})
    await redis_storage.redis.srem("research:active_ops", "op-history")

    await redis_storage.mark_stale_operations()

    assert (await redis_storage.get_operation("op-history"))["status"] == "completed"


@pytest.mark.anyio
async def test_redis_reconcile_spend_is_retry_safe(redis_storage):
    assert await redis_storage.reserve_daily_spend(4.0, 50.0, "op-retry") is True
    await redis_storage.reconcile_daily_spend(4.0, 1.5, 50.0, "op-retry")
    assert await redis_storage.get_daily_spend() == 1.5

    # A retried reconciliation (for example after a lost response) must not
    # subtract or charge the completed operation a second time.
    await redis_storage.reconcile_daily_spend(4.0, 1.5, 50.0, "op-retry")
    assert await redis_storage.get_daily_spend() == 1.5


@pytest.mark.anyio
async def test_redis_reserve_spend_is_idempotent_per_operation(redis_storage):
    assert await redis_storage.reserve_daily_spend(4.0, 50.0, "op-once") is True
    # A retried admission reservation for the same operation must not double.
    assert await redis_storage.reserve_daily_spend(4.0, 50.0, "op-once") is True
    assert await redis_storage.get_daily_spend() == 4.0

    # A release after reconciliation (reservation already consumed) is a no-op.
    await redis_storage.reconcile_daily_spend(4.0, 2.0, 50.0, "op-once")
    await redis_storage.release_daily_spend(4.0, "op-once")
    assert await redis_storage.get_daily_spend() == 2.0


@pytest.mark.anyio
async def test_redis_slot_expiry_uses_server_time(redis_storage, monkeypatch):
    monkeypatch.setattr(settings, "MAX_CONCURRENT_OPS", 1)
    monkeypatch.setattr(settings, "CONCURRENCY_LEASE_TTL_SECONDS", 3600)
    # Slot expiry must be derived from Redis server TIME rather than the
    # application clock, so replicas with skewed clocks share one timebase. The
    # stored score is therefore near server time (plus the TTL), not the
    # application's own time.time().
    monkeypatch.setattr("app.storage.time.time", lambda: 10_000_000_000.0)
    assert await redis_storage.acquire_concurrency_slot("op-skew-a") is True
    assert await redis_storage.acquire_concurrency_slot("op-skew-b") is False

    server_now = await redis_storage.redis.execute_command("TIME")
    server_ts = float(server_now[0]) + float(server_now[1]) / 1_000_000
    score = float(
        await redis_storage.redis.zscore(
            "research:concurrency:slots", redis_storage.instance.lease_id("op-skew-a")
        )
    )
    assert abs(score - (server_ts + 3600)) < 60


@pytest.mark.anyio
async def test_redis_stale_window_reservation_is_not_subtracted(redis_storage):
    # Reserve in the current window, then simulate the window rolling over: the
    # old hold must be dropped without touching the new window's total, or it
    # would erase another operation's fresh reservation.
    assert await redis_storage.reserve_daily_spend(4.0, 50.0, "op-old-window") is True
    await redis_storage.redis.set("research:spend:window", "9999999999.0")
    await redis_storage.redis.set("research:spend:daily", "4.0", keepttl=True)

    await redis_storage.reconcile_daily_spend(4.0, 1.5, 50.0, "op-old-window")
    assert await redis_storage.get_daily_spend() == 4.0

    # The stale reservation entry is dropped so a later release is also a no-op.
    await redis_storage.release_daily_spend(4.0, "op-old-window")
    assert await redis_storage.get_daily_spend() == 4.0


@pytest.mark.anyio
async def test_redis_reservations_hash_carries_expiry(redis_storage):
    assert await redis_storage.reserve_daily_spend(4.0, 50.0, "op-ttl") is True
    ttl = await redis_storage.redis.ttl("research:spend:reservations")
    assert 0 < ttl <= 86400


@pytest.mark.anyio
async def test_redis_save_operation_tracks_active_index(redis_storage):
    await redis_storage.save_operation("op-active", {"status": "running"})
    assert "op-active" in await redis_storage.redis.smembers("research:active_ops")

    await redis_storage.save_operation("op-active", {"status": "completed"})
    assert "op-active" not in await redis_storage.redis.smembers("research:active_ops")


@pytest.mark.anyio
async def test_redis_stale_scan_backfills_preindex_active_operations(redis_storage):
    # Simulate a pre-index record written by an older version: only the
    # operations hash holds it, and it has no live owner.
    await redis_storage.redis.hset(
        "research:operations", "op-legacy", json.dumps({"status": "running"})
    )
    assert "op-legacy" not in await redis_storage.redis.smembers("research:active_ops")

    await redis_storage.mark_stale_operations()

    assert (await redis_storage.get_operation("op-legacy"))["status"] == "failed"
