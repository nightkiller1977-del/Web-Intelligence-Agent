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

    Monkeypatches that ceiling to a tiny value and registers a task that
    never finishes on its own, so the test is fast and still proves the
    bound: without the fix, lifespan's shutdown phase would hang on
    quiesce_tasks()'s own (much larger) timeout instead of respecting this
    outer ceiling."""
    monkeypatch.setattr(main_module, "SHUTDOWN_DRAIN_BUDGET_S", 0.1)

    started = asyncio.Event()

    async def never_finishes():
        started.set()
        await asyncio.Event().wait()

    task = None
    start = None
    try:
        async with main_module.lifespan(main_module.app):
            task = asyncio.create_task(never_finishes())
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
