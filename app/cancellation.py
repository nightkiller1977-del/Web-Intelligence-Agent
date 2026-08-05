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
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                logger.info("Task successfully cancelled for operation: %s", op_id)
            self.unregister_task(op_id)
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
