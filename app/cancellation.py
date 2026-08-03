# app/cancellation.py
import asyncio
import logging
from typing import Dict

logger = logging.getLogger("web-intelligence")

class CancellationManager:
    def __init__(self):
        self.active_tasks: Dict[str, asyncio.Task] = {}

    def register_task(self, op_id: str, task: asyncio.Task):
        self.active_tasks[op_id] = task
        logger.info(f"Registered active task for operation: {op_id}")

    def unregister_task(self, op_id: str):
        self.active_tasks.pop(op_id, None)
        logger.debug(f"Unregistered task for operation: {op_id}")

    async def cancel_task(self, op_id: str) -> bool:
        task = self.active_tasks.get(op_id)
        if not task:
            logger.warning(f"No active task found to cancel for operation: {op_id}")
            return False
            
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            logger.info(f"Task successfully cancelled for operation: {op_id}")
        
        self.unregister_task(op_id)
        return True

cancellation_manager = CancellationManager()
