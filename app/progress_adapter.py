# app/progress_adapter.py
import datetime
import logging
from typing import Optional
from app.storage import storage

logger = logging.getLogger("web-intelligence")

class ProgressReporter:
    def __init__(self, op_id: str):
        self.op_id = op_id

    async def report(
        self,
        stage: str,
        message: str,
        completed_units: Optional[int] = None,
        total_units: Optional[int] = None,
        source_title: Optional[str] = None,
        source_url: Optional[str] = None
    ):
        event = {
            "operationId": self.op_id,
            "stage": stage,
            "message": message,
            "completedUnits": completed_units,
            "totalUnits": total_units,
            "sourceTitle": source_title,
            "sourceUrl": source_url,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
        }
        
        logger.debug(f"[{self.op_id}] Progress: {stage} - {message}")
        await storage.push_progress_event(self.op_id, event)

# Helper to hook into gpt-researcher websocket or direct callback loops
class GPTResearcherCallbackHandler:
    def __init__(self, reporter: ProgressReporter):
        self.reporter = reporter

    async def on_planning(self, message: str):
        await self.reporter.report("planning", message)

    async def on_search(self, query: str, index: int, total: int):
        await self.reporter.report(
            "searching",
            f"Searching for: '{query}'",
            completed_units=index,
            total_units=total
        )

    async def on_read(self, url: str, title: str, index: int, total: int):
        await self.reporter.report(
            "reading",
            f"Reading source {index} of {total}: {title}",
            completed_units=index,
            total_units=total,
            source_title=title,
            source_url=url
        )

    async def on_synthesize(self, message: str):
        await self.reporter.report("synthesizing", message)
