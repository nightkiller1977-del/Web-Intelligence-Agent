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
async def test_shutdown_re_cancels_a_task_still_running_after_the_budget_expires(monkeypatch):
    """quiesce_tasks() cancels every registered task exactly once, before
    its own (much larger) wait begins; asyncio.wait() does not cancel its
    members just because the wait around it is later cancelled by the
    outer SHUTDOWN_DRAIN_BUDGET_S ceiling. A task still running when that
    ceiling expires has therefore only ever been cancelled once, with
    nothing further asking it to stop - whatever happens to it afterward
    (the ASGI server's own teardown, or just the process exiting) is then
    outside this function's control. Re-cancelling it explicitly here, in
    this function's own code, removes that dependency entirely.

    The registered task resists its first cancellation (standing in for a
    cancellation-safe cleanup step genuinely still in flight) and only
    actually finishes on a second one, which must arrive before this
    function returns."""
    monkeypatch.setattr(main_module, "SHUTDOWN_DRAIN_BUDGET_S", 0.1)

    started = asyncio.Event()
    finished = asyncio.Event()
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
                finished.set()
                raise

    task = None
    try:
        async with main_module.lifespan(main_module.app):
            task = asyncio.create_task(finishes_only_on_its_second_cancellation())
            cancellation_manager.register_task("op-needs-second-cancel", task)
            await started.wait()

        await asyncio.wait_for(finished.wait(), timeout=1.0)
        assert cancellations_received == 2, (
            "a task still running when the shutdown budget expires must be re-cancelled, "
            "not left to whatever happens to it outside this function's control"
        )
    finally:
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        cancellation_manager.unregister_task("op-needs-second-cancel")
