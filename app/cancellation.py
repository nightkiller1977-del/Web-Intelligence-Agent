# app/cancellation.py
import asyncio
import logging
from typing import Awaitable, Callable, Dict, Optional

logger = logging.getLogger("web-intelligence")

CANCEL_CHANNEL = "research:cancel"
TERMINAL_STATUSES = {"completed", "partial", "failed", "cancelled"}


class CancellationManager:
    def __init__(self):
        self.active_tasks: Dict[str, asyncio.Task] = {}
        self._redis = None
        self._pubsub = None
        self._listener_task: Optional[asyncio.Task] = None

    async def init(self, redis_client=None):
        if redis_client is None:
            return
        self._redis = redis_client
        self._pubsub = redis_client.pubsub()
        await self._pubsub.subscribe(CANCEL_CHANNEL)
        self._listener_task = asyncio.create_task(self._listen())

    async def _listen(self):
        try:
            async for msg in self._pubsub.listen():
                if msg["type"] != "message":
                    continue
                op_id = msg["data"]
                if isinstance(op_id, bytes):
                    op_id = op_id.decode()
                task = self.active_tasks.get(op_id)
                if task and not task.done():
                    logger.info("Received cross-instance cancel for operation %s", op_id)
                    task.cancel()
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("Cancellation listener died")

    def register_task(self, op_id: str, task: asyncio.Task):
        self.active_tasks[op_id] = task
        logger.info("Registered active task for operation: %s", op_id)

    def unregister_task(self, op_id: str):
        self.active_tasks.pop(op_id, None)
        logger.debug("Unregistered task for operation: %s", op_id)

    async def cancel_task(
        self,
        op_id: str,
        operation_lookup: Optional[Callable[[str], Awaitable[Optional[dict]]]] = None
    ) -> bool:
        task = self.active_tasks.get(op_id)
        if task and not task.done():
            # A task stays registered for the whole of its own cleanup
            # (ingest-await, then spend/lease/concurrency-slot release) so
            # quiesce_tasks() can still find and await it if shutdown lands
            # during that window - it only unregisters once that cleanup is
            # actually done. That means a result can already be durably
            # terminal (completed/partial/failed/cancelled) while the task
            # is still registered here, so this local branch must check the
            # stored status itself before cancelling, exactly as the
            # cross-instance branch below already does - otherwise a client
            # that polls the result, sees it terminal, and then calls cancel
            # during that cleanup window flips an immutable, already-
            # persisted result to a spurious "cancelled" response.
            if operation_lookup:
                op = await operation_lookup(op_id)
                if op and op.get("status") in TERMINAL_STATUSES:
                    logger.info(
                        "Refusing local cancel for already finalized operation %s with status %s",
                        op_id,
                        op.get("status")
                    )
                    return False
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                logger.info("Task successfully cancelled for operation: %s", op_id)
            self.unregister_task(op_id)
            # The check above can be stale by the time it matters: the task
            # can race ahead and persist its own terminal result in the gap
            # between that await and this point, making task.cancel() either
            # a no-op (already done) or a cancellation its own cancellation-
            # safe cleanup absorbs - either way it finishes normally, not
            # with CancelledError, so the except above never fires. There is
            # no further race to check against here, though: `task` has now
            # fully finished (this await only returns once it has), and
            # nothing else writes this operation's status once its owning
            # task has returned - so this result is authoritative. A
            # non-"cancelled" terminal status here means the cancellation
            # simply arrived too late to take effect.
            if operation_lookup:
                op = await operation_lookup(op_id)
                if op and op.get("status") in TERMINAL_STATUSES and op.get("status") != "cancelled":
                    logger.info(
                        "Operation %s reached %s on its own before cancellation took effect; reporting that instead",
                        op_id,
                        op.get("status")
                    )
                    return False
            return True

        # Task not on this instance — broadcast via Redis pub/sub
        if self._redis:
            if operation_lookup:
                op = await operation_lookup(op_id)
                if not op:
                    logger.warning("Refusing cross-instance cancel for unknown operation: %s", op_id)
                    return False
                if op.get("status") in TERMINAL_STATUSES:
                    logger.info(
                        "Refusing cross-instance cancel for already finalized operation %s with status %s",
                        op_id,
                        op.get("status")
                    )
                    return False
            logger.info("Broadcasting cancel for operation %s to other instances", op_id)
            await self._redis.publish(CANCEL_CHANNEL, op_id)
            return True

        logger.warning("No active task found to cancel for operation: %s", op_id)
        return False

    async def quiesce_tasks(self, timeout: float = 5.0) -> int:
        """Cancel and await in-flight research tasks before shutdown proceeds.

        Shutdown must not let a still-running research task finish after the
        pending-ingest snapshot was taken, or its outcome ingest would be
        scheduled into a closing event loop. Cancelling and awaiting the active
        tasks first makes that snapshot complete.
        """
        pending = [task for task in self.active_tasks.values() if not task.done()]
        if not pending:
            return 0
        for task in pending:
            task.cancel()
        done, _still_pending = await asyncio.wait(pending, timeout=timeout)
        for task in done:
            # Retrieve the outcome so a cancelled/failed task is not reported as
            # an unhandled exception while the loop is tearing down.
            if not task.cancelled():
                task.exception()
        return len(pending)

    async def shutdown(self):
        if self._listener_task:
            self._listener_task.cancel()
            try:
                await self._listener_task
            except asyncio.CancelledError:
                pass
        if self._pubsub:
            await self._pubsub.unsubscribe(CANCEL_CHANNEL)


cancellation_manager = CancellationManager()
