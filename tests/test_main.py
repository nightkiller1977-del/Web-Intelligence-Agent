"""
tests/test_main.py

Coverage for app.main's lifespan shutdown sequence.
"""

import asyncio
import time

import pytest

import app.main as main_module
from app.cancellation import cancellation_manager


@pytest.mark.timeout(2)
@pytest.mark.anyio
async def test_shutdown_drain_is_bounded_so_a_stuck_task_cannot_block_teardown(monkeypatch):
    """quiesce_tasks()'s own wait and flush_pending_ingest_tasks()'s backstop
    are each sized independently for their own purpose, with nothing keeping
    their *sum* under the platform's actual SIGTERM-to-SIGKILL grace period -
    raising quiesce_tasks()'s multiplier to give it real headroom over a
    single task's worst-case serial cleanup made that sum equal Azure
    Container Apps' default grace period exactly, leaving zero margin.
    Wrapping the whole drain sequence in one hard ceiling (SHUTDOWN_DRAIN_BUDGET_S)
    is what keeps shutdown from ever running past the platform's own patience,
    even when an individual task is genuinely stuck.

    The registered task must resist quiesce_tasks()'s one cancellation and
    keep running (mirroring how a real cancellation-safe cleanup step
    elsewhere in this codebase survives a single cancellation without
    finishing), or quiesce_tasks()'s own asyncio.wait() returns quickly on
    its own and the outer asyncio.wait_for ceiling is never actually
    exercised - a task that simply finishes on the first cancellation would
    make this test pass whether or not that ceiling exists at all. It
    resists exactly once, so this test's own teardown can still cancel it
    a second time to clean up."""
    monkeypatch.setattr(main_module, "SHUTDOWN_DRAIN_BUDGET_S", 0.1)

    started = asyncio.Event()
    absorbed_once = False

    async def resists_one_cancellation():
        nonlocal absorbed_once
        started.set()
        while True:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                if not absorbed_once:
                    absorbed_once = True
                    continue
                raise

    task = None
    start = None
    try:
        async with main_module.lifespan(main_module.app):
            task = asyncio.create_task(resists_one_cancellation())
            cancellation_manager.register_task("op-stuck-at-shutdown", task)
            await started.wait()
            start = time.monotonic()
        elapsed = time.monotonic() - start

        assert elapsed < 1.0, (
            f"shutdown drain took {elapsed:.3f}s - it must be bounded by "
            "SHUTDOWN_DRAIN_BUDGET_S, not left to quiesce_tasks()'s own larger timeout"
        )
    finally:
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        cancellation_manager.unregister_task("op-stuck-at-shutdown")


@pytest.mark.timeout(2)
@pytest.mark.anyio
async def test_shutdown_awaits_a_task_still_running_after_the_budget_expires(monkeypatch):
    """quiesce_tasks() cancels every registered task exactly once, before
    its own (much larger) wait begins; asyncio.wait() does not cancel its
    members just because the wait around it is later cancelled by the
    outer SHUTDOWN_DRAIN_BUDGET_S ceiling. A task still running when that
    ceiling expires has therefore only ever been cancelled once, with
    nothing further asking it to stop. Merely re-cancelling it without
    also awaiting it would let this function return while that task's own
    cleanup might still be genuinely, briefly in flight - its own per-step
    deadline can still have a little real time left even after this outer
    ceiling gives up - leaving whatever happens next (the ASGI server's
    own teardown, or just the process exiting) outside this function's
    control. Calling quiesce_tasks() again, bounded to a short fixed
    extra step rather than an unbounded or repeated extension, both
    re-cancels and actually awaits it.

    The registered task resists its first cancellation (standing in for a
    cancellation-safe cleanup step genuinely still in flight) and only
    actually finishes on a second one, which must both arrive and be
    awaited to completion before this function returns."""
    monkeypatch.setattr(main_module, "SHUTDOWN_DRAIN_BUDGET_S", 0.1)

    started = asyncio.Event()
    cancellations_received = 0

    async def finishes_only_on_its_second_cancellation():
        nonlocal cancellations_received
        started.set()
        while True:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancellations_received += 1
                if cancellations_received < 2:
                    continue
                raise

    task = None
    try:
        async with main_module.lifespan(main_module.app):
            task = asyncio.create_task(finishes_only_on_its_second_cancellation())
            cancellation_manager.register_task("op-needs-second-cancel", task)
            await started.wait()

        # Checked immediately on exiting lifespan's shutdown phase, not
        # after an additional grace wait: this is what distinguishes
        # actually awaiting the task from merely firing a second
        # cancellation at it and returning regardless.
        assert task.done(), (
            "a task still running when the shutdown budget expires must be awaited to completion, "
            "not just re-cancelled and left running past this function's own return"
        )
        assert cancellations_received == 2
    finally:
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        cancellation_manager.unregister_task("op-needs-second-cancel")


@pytest.mark.timeout(2)
@pytest.mark.anyio
async def test_shutdown_fallback_timeout_is_computed_against_a_single_deadline(monkeypatch):
    """An OpenHands review on this PR found that an earlier version of the
    fallback below - a second quiesce_tasks() call bounded to a flat extra
    DEFAULT_INGEST_WAIT_TIMEOUT_S - did not actually bound a stuck task's
    real remaining cleanup time: that constant only reflects one of the
    (up to four) serial cleanup steps background_research_task's own
    `finally` block can still be running through, so a task that was only
    partway into its cleanup when SHUTDOWN_DRAIN_BUDGET_S first expired
    could still exceed the documented total by multiples of that constant.

    The fix anchors the fallback's own timeout to whatever time genuinely
    remains until one single shutdown deadline (ASSUMED_PLATFORM_GRACE_PERIOD_S
    minus SHUTDOWN_FINAL_STEPS_MARGIN_S, computed once before the first
    attempt even starts), rather than to a second, independently-sized
    constant borrowed from an unrelated part of the system. This spies on
    cancellation_manager.quiesce_tasks() to capture the exact timeout each
    of the two calls actually receives, rather than inferring it from
    elapsed wall-clock time (too imprecise at this resolution to catch a
    regression back to a fixed constant)."""
    monkeypatch.setattr(main_module, "SHUTDOWN_DRAIN_BUDGET_S", 0.05)
    monkeypatch.setattr(main_module, "ASSUMED_PLATFORM_GRACE_PERIOD_S", 0.5)
    monkeypatch.setattr(main_module, "SHUTDOWN_FINAL_STEPS_MARGIN_S", 0.1)

    started = asyncio.Event()
    absorbed_once = False

    async def resists_one_cancellation():
        nonlocal absorbed_once
        started.set()
        while True:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                if not absorbed_once:
                    absorbed_once = True
                    continue
                raise

    calls = []
    real_quiesce = cancellation_manager.quiesce_tasks

    async def spy_quiesce(timeout=None):
        calls.append(timeout)
        return await real_quiesce(timeout=timeout)

    monkeypatch.setattr(cancellation_manager, "quiesce_tasks", spy_quiesce)

    task = None
    try:
        async with main_module.lifespan(main_module.app):
            task = asyncio.create_task(resists_one_cancellation())
            cancellation_manager.register_task("op-fallback-timeout-probe", task)
            await started.wait()
    finally:
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        cancellation_manager.unregister_task("op-fallback-timeout-probe")

    assert len(calls) == 2, (
        "expected both the primary quiesce_tasks() call and the bounded fallback to run, "
        f"got {len(calls)} call(s) with timeouts {calls!r}"
    )
    assert calls[0] is None, "the primary attempt must use quiesce_tasks()'s own default, not a fixed override"
    # Remaining budget at the fallback call is (0.5 - 0.1) minus however long the
    # primary attempt actually ran (bounded by SHUTDOWN_DRAIN_BUDGET_S=0.05) - a
    # fixed unrelated constant (e.g. the real DEFAULT_INGEST_WAIT_TIMEOUT_S, 6.0)
    # would fall well outside this range.
    assert 0.2 < calls[1] < 0.4, (
        f"fallback timeout was {calls[1]!r}; expected it computed from the time actually "
        "remaining until the single shutdown deadline, not a fixed constant"
    )
