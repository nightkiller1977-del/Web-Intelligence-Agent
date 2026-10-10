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

import app.cancellation as cancellation
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
async def test_local_cancel_refuses_already_terminal_operation():
    """A task that has already produced a durable terminal result can stay
    registered in active_tasks for a while afterward (its own post-result
    cleanup, deliberately - see app/api.py's background_research_task,
    which keeps itself registered through cleanup so shutdown's
    quiesce_tasks() can still find it). Without a stored-status check, the
    local branch below would cancel and report "cancelled" for any op_id
    still registered, with no regard for what was actually stored - unlike
    the cross-instance branch above, which already checks this. A client
    that polls the result, sees it terminal, and then calls /cancel while
    the task is merely still finishing cleanup must get the real stored
    status back, not a spurious "cancelled"."""
    manager = CancellationManager()

    started = asyncio.Event()

    async def still_registered_but_already_done():
        started.set()
        await asyncio.Event().wait()  # stands in for post-result cleanup still in flight

    task = asyncio.create_task(still_registered_but_already_done())
    manager.register_task("op-already-terminal", task)
    await started.wait()

    try:
        async def lookup_completed(op_id):
            return {"status": "completed"}

        result = await manager.cancel_task("op-already-terminal", operation_lookup=lookup_completed)
        assert result is False
        assert not task.cancelled()
        assert "op-already-terminal" in manager.active_tasks
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


@pytest.mark.anyio
async def test_local_cancel_still_works_for_a_genuinely_running_operation():
    """Companion to the refusal test above: the new stored-status check must
    not block a legitimate cancel of an operation that is actually still
    running."""
    manager = CancellationManager()

    started = asyncio.Event()
    was_cancelled = asyncio.Event()

    async def long_running_work():
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            was_cancelled.set()
            raise

    task = asyncio.create_task(long_running_work())
    manager.register_task("op-still-running", task)
    await started.wait()

    async def lookup_running(op_id):
        return {"status": "running"}

    result = await manager.cancel_task("op-still-running", operation_lookup=lookup_running)
    assert result is True
    await asyncio.wait_for(was_cancelled.wait(), timeout=2)


@pytest.mark.anyio
async def test_local_cancel_reports_real_status_when_task_finishes_before_cancellation_takes_effect():
    """The stored-status check above is read before task.cancel(), so it
    can be stale by the time cancel_task() actually acts - the task can
    race ahead and persist its own terminal result in between.
    task.cancel() is then either a no-op (the task already finished) or a
    cancellation its own cancellation-safe cleanup absorbs, and either way
    the task completes normally rather than raising CancelledError - so
    without a second, race-free check after `task` has actually finished,
    cancel_task() would return True unconditionally, and /cancel would
    report "cancelled" for an operation that is really "completed"."""
    manager = CancellationManager()
    stored_status = {"value": "running"}

    async def finishes_on_its_own_right_after_being_cancelled():
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            # Stands in for background_research_task's own cancellation-safe
            # cleanup absorbing a cancellation that arrived after its real
            # result was already persisted - it does not re-raise, so this
            # task completes normally, not with CancelledError.
            stored_status["value"] = "completed"

    task = asyncio.create_task(finishes_on_its_own_right_after_being_cancelled())
    manager.register_task("op-race", task)
    await asyncio.sleep(0)

    async def lookup(op_id):
        return {"status": stored_status["value"]}

    result = await manager.cancel_task("op-race", operation_lookup=lookup)
    assert result is False, (
        "cancel_task() must report the operation's real terminal status, not a blanket "
        "cancelled, when the task actually finished on its own instead of being genuinely cancelled"
    )


@pytest.mark.anyio
async def test_cancel_task_with_no_redis_and_no_local_task_returns_false():
    """Without Redis wired up and no locally-registered task, there's
    nothing this instance can do about the cancel request."""
    manager = CancellationManager()

    result = await manager.cancel_task("op-nowhere", operation_lookup=None)
    assert result is False


@pytest.mark.anyio
async def test_quiesce_tasks_cancels_and_awaits_active_tasks():
    """Shutdown quiescing must cancel and await in-flight research tasks so a
    late finisher cannot schedule work into a closing event loop."""
    manager = CancellationManager()

    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def long_running():
        started.set()
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    task = asyncio.create_task(long_running())
    manager.register_task("op-active", task)
    await started.wait()

    quiesced = await manager.quiesce_tasks()

    assert quiesced == 1
    assert task.done()
    assert cancelled.is_set()
    assert manager.active_tasks["op-active"] is task


@pytest.mark.anyio
async def test_quiesce_tasks_returns_zero_when_idle():
    manager = CancellationManager()
    assert await manager.quiesce_tasks() == 0


@pytest.mark.anyio
async def test_quiesce_tasks_default_timeout_tracks_three_times_the_ingest_constant(monkeypatch):
    """A flat 5s default here is shorter than a single bounded cleanup step
    (DEFAULT_INGEST_WAIT_TIMEOUT_S = 6s), let alone the three such steps
    background_research_task's own `finally` block (app/api.py) can run
    through serially on either path it takes - ingest, lease, concurrency
    slot on success, or spend, lease, concurrency slot on failure/
    cancellation. Shutdown could call quiesce_tasks(), get back control
    after its timeout, and go on to drain ingests and close storage
    connections while a task was still mid-cleanup - abandoning its
    owner-lease/concurrency-slot/spend-hold release exactly as this cleanup
    path otherwise protects for the outcome ingest. Raising the default to
    3 * DEFAULT_INGEST_WAIT_TIMEOUT_S is what covers that worst case.

    Timing how long quiesce_tasks() takes to return against a task driven
    through three serial cleanup steps scaled to a tiny monkeypatched
    DEFAULT_INGEST_WAIT_TIMEOUT_S does not actually discriminate this
    relationship: a total of ~0.12s fits comfortably within *both* the new
    3x default (0.15s) and the old flat 5.0s default it should catch a
    regression of, so it would pass unchanged either way and prove nothing
    - reproducible standalone with a fixed timeout=5.0, yielding the
    identical done=True/steps=3 result. Racing real elapsed time against a
    5-second baseline isn't practical at test speed, so this instead spies
    on the exact timeout value quiesce_tasks() passes to asyncio.wait() and
    asserts it against the monkeypatched constant directly - deterministic,
    fast, and immune to how long any task's own cleanup happens to take."""
    monkeypatch.setattr(cancellation, "DEFAULT_INGEST_WAIT_TIMEOUT_S", 0.05)

    captured = {}
    real_wait = asyncio.wait

    async def capturing_wait(tasks, timeout=None):
        captured["timeout"] = timeout
        return await real_wait(tasks, timeout=timeout)

    monkeypatch.setattr(cancellation.asyncio, "wait", capturing_wait)

    manager = CancellationManager()
    started = asyncio.Event()

    async def long_running():
        started.set()
        await asyncio.Event().wait()

    task = asyncio.create_task(long_running())
    manager.register_task("op-capture-quiesce-timeout", task)
    await started.wait()

    await manager.quiesce_tasks()

    assert captured["timeout"] == pytest.approx(0.15), (
        "quiesce_tasks()'s default must track 3x DEFAULT_INGEST_WAIT_TIMEOUT_S "
        f"(0.15 here), not a flat number unrelated to it - got {captured.get('timeout')!r}"
    )

