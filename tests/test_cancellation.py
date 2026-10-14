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
    stored_status = {"value": "running"}

    async def long_running_work():
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            # Mirrors background_research_task's own cancellation handler,
            # which persists "cancelled" before this task returns.
            stored_status["value"] = "cancelled"
            was_cancelled.set()
            raise

    task = asyncio.create_task(long_running_work())
    manager.register_task("op-still-running", task)
    await started.wait()

    async def lookup_running(op_id):
        return {"status": stored_status["value"]}

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
async def test_local_cancel_does_not_report_success_after_a_lost_lease_exit():
    """A lost-lease exit (_LeaseLostError in app/api.py's
    background_research_task) deliberately persists no status at all, so
    the task its lease moved away from can be done - and absorb a
    cancellation exactly the way a genuinely cancelled task does - while
    storage still shows whatever status the new owner left (here,
    "running"), never "cancelled". A check that only looks for an
    unexpected *terminal* status after awaiting the task would miss this:
    "running" is neither terminal nor "cancelled", so it would fall through
    to a false cancel_task() success, telling the caller "cancelled" for an
    operation that is really still running under a different owner."""
    manager = CancellationManager()
    stored_status = {"value": "running"}

    async def absorbs_cancellation_without_persisting_anything():
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            # Stands in for a lost-lease exit: the task's own cancellation-
            # safe cleanup absorbs the cancellation and returns normally,
            # but nothing here writes a terminal status - the new owner's
            # write (or lack of one yet) is what storage still reflects.
            pass

    task = asyncio.create_task(absorbs_cancellation_without_persisting_anything())
    manager.register_task("op-lease-lost", task)
    await asyncio.sleep(0)

    async def lookup(op_id):
        return {"status": stored_status["value"]}

    result = await manager.cancel_task("op-lease-lost", operation_lookup=lookup)
    assert result is False, (
        "cancel_task() must not report success when the operation it awaited is still "
        "non-terminal after a lost-lease exit that persisted nothing - the real status "
        "belongs to whichever worker now owns it"
    )


@pytest.mark.timeout(2)
@pytest.mark.anyio
async def test_local_cancel_proceeds_when_the_pre_check_status_lookup_hangs(monkeypatch):
    """The Redis client has no socket timeout (app/storage.py's
    aioredis.from_url call), so a connection that accepts but never answers
    could stall the pre-cancel status check forever - and cancelling a task
    this replica already owns needs nothing from Redis at all, so a hung
    status check must never be the reason task.cancel() never runs. The fix
    bounds that lookup and, on timeout, proceeds exactly as if nothing was
    found to refuse on.

    Monkeypatches STATUS_LOOKUP_TIMEOUT_S (the dedicated, small bound this
    lookup uses) to a tiny value and makes operation_lookup never finish on
    its own, so the test is fast and still proves the bound: without the
    fix, this would hang rather than merely run slowly (the class 2s
    timeout above is the backstop in case something regresses the fix
    entirely)."""
    monkeypatch.setattr(cancellation, "STATUS_LOOKUP_TIMEOUT_S", 0.05)
    manager = CancellationManager()

    started = asyncio.Event()
    stored_status = {"value": "running"}

    async def long_running_work():
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            # Mirrors background_research_task's own handler, which
            # persists "cancelled" before returning.
            stored_status["value"] = "cancelled"
            raise

    task = asyncio.create_task(long_running_work())
    manager.register_task("op-pre-check-hangs", task)
    await started.wait()

    # Only the first call (the pre-check) hangs; the second (the post-check,
    # after the real cancellation above) must resolve normally so this test
    # isolates the pre-check's own fallback rather than also exercising the
    # post-check's separate one.
    calls = 0

    async def lookup(op_id):
        nonlocal calls
        calls += 1
        if calls == 1:
            await asyncio.Event().wait()
        return {"status": stored_status["value"]}

    result = await manager.cancel_task("op-pre-check-hangs", operation_lookup=lookup)
    assert result is True, "a hung pre-cancel status lookup must not prevent a legitimate local cancel"
    assert stored_status["value"] == "cancelled"


@pytest.mark.timeout(2)
@pytest.mark.anyio
async def test_local_cancel_proceeds_when_the_pre_check_status_lookup_fails_fast(monkeypatch):
    """Companion to the hang test above: a hang is not the only way this
    lookup can fail. A fast connection/reset error from Redis must get the
    same fallback - proceed exactly as if nothing was found to refuse on -
    not propagate out of cancel_task() uncaught, which would keep
    task.cancel() from ever running and turn /cancel into a 500 for an
    operation this replica could have cancelled immediately."""
    manager = CancellationManager()

    started = asyncio.Event()
    stored_status = {"value": "running"}

    async def long_running_work():
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            stored_status["value"] = "cancelled"
            raise

    task = asyncio.create_task(long_running_work())
    manager.register_task("op-pre-check-fails-fast", task)
    await started.wait()

    # Only the first call (the pre-check) fails; the second (the
    # post-check) must resolve normally so this test isolates the
    # pre-check's own fallback, same reasoning as the hang test above.
    calls = 0

    async def lookup(op_id):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ConnectionError("connection reset by peer")
        return {"status": stored_status["value"]}

    result = await manager.cancel_task("op-pre-check-fails-fast", operation_lookup=lookup)
    assert result is True, "a fast-failing pre-cancel status lookup must not prevent a legitimate local cancel"
    assert stored_status["value"] == "cancelled"


@pytest.mark.timeout(2)
@pytest.mark.anyio
async def test_local_cancel_reports_failure_when_the_post_check_status_lookup_hangs(monkeypatch):
    """Companion to the pre-check test above, but with the opposite
    fallback: a hung Redis read in the second, post-await status check
    cannot fall back to "proceed" the way the pre-check can, because the
    cancellation has already happened by this point - all that is left is
    what to report. Claiming "cancelled" without being able to confirm it
    risks the exact spurious flip the whole terminal-status check exists
    to prevent, for an operation that may already be durably completed.
    The fix reports failure outward instead of guessing success, so the
    caller (the /cancel endpoint) falls back to its own bounded lookup."""
    monkeypatch.setattr(cancellation, "STATUS_LOOKUP_TIMEOUT_S", 0.05)
    manager = CancellationManager()

    started = asyncio.Event()

    async def long_running_work():
        started.set()
        await asyncio.Event().wait()

    task = asyncio.create_task(long_running_work())
    manager.register_task("op-post-check-hangs", task)
    await started.wait()

    async def hangs_forever(op_id):
        await asyncio.Event().wait()

    result = await manager.cancel_task("op-post-check-hangs", operation_lookup=hangs_forever)
    assert result is False, (
        "a hung post-cancel status lookup must not be reported as a confirmed cancellation"
    )


@pytest.mark.timeout(2)
@pytest.mark.anyio
async def test_local_cancel_reports_failure_when_the_post_check_status_lookup_fails_fast(monkeypatch):
    """Companion to the hang test above: a fast connection/reset error in
    the post-check lookup must get the same fallback (report failure, not
    a guessed success) rather than propagate out of cancel_task() uncaught."""
    manager = CancellationManager()

    started = asyncio.Event()

    async def long_running_work():
        started.set()
        await asyncio.Event().wait()

    task = asyncio.create_task(long_running_work())
    manager.register_task("op-post-check-fails-fast", task)
    await started.wait()

    async def fails_fast(op_id):
        raise ConnectionError("connection reset by peer")

    result = await manager.cancel_task("op-post-check-fails-fast", operation_lookup=fails_fast)
    assert result is False, (
        "a fast-failing post-cancel status lookup must not be reported as a confirmed cancellation"
    )


@pytest.mark.timeout(2)
@pytest.mark.anyio
async def test_cross_instance_cancel_broadcasts_when_the_status_lookup_hangs(monkeypatch):
    """Same bound as the local branch, but a hung lookup here can't fall
    back to the local branch's "proceed as if nothing was found" answer -
    that already means something specific (refuse) for a lookup that
    genuinely ran. On timeout we don't know either way, so the fix skips
    the refusal and broadcasts anyway: this instance has no task of its own
    at stake, and a redundant broadcast is harmless, unlike silently
    dropping a cancel request because Redis happened to be slow."""
    monkeypatch.setattr(cancellation, "STATUS_LOOKUP_TIMEOUT_S", 0.05)
    fake_redis = fakeredis_aioredis.FakeRedis(decode_responses=True)
    manager = CancellationManager()
    await manager.init(fake_redis)

    try:
        async def hangs_forever(op_id):
            await asyncio.Event().wait()

        result = await manager.cancel_task("op-cross-instance-lookup-hangs", operation_lookup=hangs_forever)
        assert result is True, "a hung pre-broadcast status lookup must not prevent the cross-instance broadcast"
    finally:
        await manager.shutdown()
        await fake_redis.aclose()


@pytest.mark.timeout(2)
@pytest.mark.anyio
async def test_cross_instance_cancel_broadcasts_when_the_status_lookup_fails_fast(monkeypatch):
    """Companion to the hang test above: a fast connection/reset error
    before broadcasting must get the same fallback (broadcast anyway)
    rather than propagate out of cancel_task() uncaught."""
    fake_redis = fakeredis_aioredis.FakeRedis(decode_responses=True)
    manager = CancellationManager()
    await manager.init(fake_redis)

    try:
        async def fails_fast(op_id):
            raise ConnectionError("connection reset by peer")

        result = await manager.cancel_task("op-cross-instance-lookup-fails-fast", operation_lookup=fails_fast)
        assert result is True, "a fast-failing pre-broadcast status lookup must not prevent the cross-instance broadcast"
    finally:
        await manager.shutdown()
        await fake_redis.aclose()


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
async def test_quiesce_tasks_default_timeout_tracks_ingest_plus_four_release_steps(monkeypatch):
    """A flat 5s default here is shorter than a single bounded cleanup step
    (DEFAULT_INGEST_WAIT_TIMEOUT_S = 6s), let alone that step plus the three
    release steps background_research_task's own `finally` block
    (app/api.py) can run through serially after it on either path it takes
    - lease, concurrency slot on success, or spend, lease, concurrency slot
    on failure/cancellation. Shutdown could call quiesce_tasks(), get back
    control after its timeout, and go on to drain ingests and close storage
    connections while a task was still mid-cleanup - abandoning its
    owner-lease/concurrency-slot/spend-hold release exactly as this cleanup
    path otherwise protects for the outcome ingest.

    Only the ingest wait is bounded by DEFAULT_INGEST_WAIT_TIMEOUT_S; the
    three release steps are each bounded by the much smaller
    STATUS_LOOKUP_TIMEOUT_S (see its own comment in app/cancellation.py -
    they are plain Redis writes, not HTTP calls, so they do not need the
    ingest wait's larger allowance). The default here is that real sum plus
    one extra STATUS_LOOKUP_TIMEOUT_S step of headroom, not exactly the sum
    itself: this wait's own deadline starts counting before each task's own
    cancellation is actually delivered and its cleanup loop computes its
    own per-step deadlines, and scheduling/logging overhead across four
    sequential waits adds more on top of that - so the bare sum leaves zero
    margin against both effects, and real headroom (one more step's worth)
    is needed to reliably cover a task's full serial cleanup rather than
    racing it almost exactly.

    Timing how long quiesce_tasks() takes to return against a task driven
    through its serial cleanup steps scaled to tiny monkeypatched constants
    does not actually discriminate this relationship: a short total elapsed
    time fits comfortably within both the correct default and a regressed
    flat one, so it would pass unchanged either way and prove nothing.
    Racing real elapsed time against a multi-second baseline isn't
    practical at test speed either, so this instead spies on the exact
    timeout value quiesce_tasks() passes to asyncio.wait() and asserts it
    against the monkeypatched constants directly - deterministic, fast, and
    immune to how long any task's own cleanup happens to take."""
    monkeypatch.setattr(cancellation, "DEFAULT_INGEST_WAIT_TIMEOUT_S", 0.1)
    monkeypatch.setattr(cancellation, "STATUS_LOOKUP_TIMEOUT_S", 0.02)

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

    assert captured["timeout"] == pytest.approx(0.18), (
        "quiesce_tasks()'s default must track DEFAULT_INGEST_WAIT_TIMEOUT_S + "
        "4 * STATUS_LOOKUP_TIMEOUT_S (0.18 here), not a flat number or a multiple "
        f"of only one of the two constants - got {captured.get('timeout')!r}"
    )

