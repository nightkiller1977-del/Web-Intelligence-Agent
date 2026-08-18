"""
tests/test_cancellation.py

Behavioral coverage for app.cancellation.CancellationManager, in particular
its cross-instance cancel broadcast over Redis pub/sub - previously
completely untested (test_remote_mode.py only checks that the cancel
endpoint doesn't return 401/403).

The pub/sub broadcast test below uses fakeredis, whose FakeRedis client
implements PUBLISH/SUBSCRIBE against an in-process channel registry, so two
independent CancellationManager instances sharing one FakeRedis client
behave like two separate service instances sharing a real Redis/Valkey
pub/sub channel.
"""

import asyncio

import fakeredis.aioredis as fakeredis_aioredis
import pytest

from app.cancellation import CancellationManager


@pytest.mark.anyio
async def test_cross_instance_cancel_broadcast_via_redis_pubsub():
    """Simulates two service instances sharing Redis: instance B receives a
    cancel request for an operation whose task actually lives on instance A.
    B has no local task to cancel, so it must publish on CANCEL_CHANNEL; A's
    background listener must pick that up and cancel its local task."""
    fake_redis = fakeredis_aioredis.FakeRedis(decode_responses=True)

    manager_a = CancellationManager()  # holds the real running task
    manager_b = CancellationManager()  # receives the cancel call, no local task

    await manager_a.init(fake_redis)
    await manager_b.init(fake_redis)

    started = asyncio.Event()
    was_cancelled = asyncio.Event()

    async def long_running_work():
        started.set()
        try:
            await asyncio.Event().wait()  # blocks forever until cancelled
        except asyncio.CancelledError:
            was_cancelled.set()
            raise

    task = asyncio.create_task(long_running_work())
    manager_a.register_task("op-cross-instance", task)

    try:
        await asyncio.wait_for(started.wait(), timeout=2)

        async def lookup_running(op_id):
            return {"status": "running"}

        result = await manager_b.cancel_task("op-cross-instance", operation_lookup=lookup_running)
        assert result is True, "instance B should broadcast the cancel since it has no local task"

        await asyncio.wait_for(was_cancelled.wait(), timeout=2)
        assert task.cancelled() or task.done()
    finally:
        await manager_a.shutdown()
        await manager_b.shutdown()
        await fake_redis.aclose()


@pytest.mark.anyio
async def test_cross_instance_cancel_refuses_unknown_operation():
    """If operation_lookup can't find the operation at all, instance B must
    refuse to broadcast a cancel for it rather than publishing blindly."""
    fake_redis = fakeredis_aioredis.FakeRedis(decode_responses=True)
    manager_b = CancellationManager()
    await manager_b.init(fake_redis)

    try:
        async def lookup_missing(op_id):
            return None

        result = await manager_b.cancel_task("op-does-not-exist", operation_lookup=lookup_missing)
        assert result is False
    finally:
        await manager_b.shutdown()
        await fake_redis.aclose()


@pytest.mark.anyio
async def test_cross_instance_cancel_refuses_already_terminal_operation():
    """If the looked-up operation is already in a terminal state, instance B
    must not broadcast a cancel for it."""
    fake_redis = fakeredis_aioredis.FakeRedis(decode_responses=True)
    manager_b = CancellationManager()
    await manager_b.init(fake_redis)

    try:
        async def lookup_completed(op_id):
            return {"status": "completed"}

        result = await manager_b.cancel_task("op-already-done", operation_lookup=lookup_completed)
        assert result is False
    finally:
        await manager_b.shutdown()
        await fake_redis.aclose()


@pytest.mark.anyio
async def test_cancel_task_with_no_redis_and_no_local_task_returns_false():
    """Without Redis wired up and no locally-registered task, there's
    nothing this instance can do about the cancel request."""
    manager = CancellationManager()

    result = await manager.cancel_task("op-nowhere", operation_lookup=None)
    assert result is False
