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

    A first version of this test registered a task that simply never
    finishes on its own and monkeypatched only SHUTDOWN_DRAIN_BUDGET_S -
    but quiesce_tasks() cancels that task exactly once, and a task that
    does nothing to resist that single cancellation leaves
    asyncio.Event().wait() and finishes immediately, so quiesce_tasks()'s
    own asyncio.wait() returns quickly on its own. That made the test pass
    even with the asyncio.wait_for ceiling removed entirely - it only
    discriminated whether SHUTDOWN_DRAIN_BUDGET_S the *constant* existed,
    not whether the ceiling it names actually bounds anything.

    This version absorbs quiesce_tasks()'s one cancellation (mirroring how
    a real cancellation-safe cleanup step elsewhere in this codebase
    survives a single cancellation without finishing) and keeps running,
    so quiesce_tasks()'s own wait cannot return on its own within the
    monkeypatched ceiling - only the outer asyncio.wait_for actually bounds
    this test. It resists exactly once, so this test's own teardown can
    still cancel it a second time to clean up."""
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
