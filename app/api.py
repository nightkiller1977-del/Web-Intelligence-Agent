# app/api.py
import asyncio
import json
import logging
from fastapi import APIRouter, Header, HTTPException, Request, status
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from app.config import raw_header_credentials_allowed, settings
from app.storage import storage
from app.schemas import ResearchRequestInput, ResearchResultResponse, CapabilitiesInfo
from app.cancellation import cancellation_manager
from app.progress_adapter import ProgressReporter
from app.researcher_adapter import conduct_web_research
from app.security import is_safe_url

logger = logging.getLogger("web-intelligence")
router = APIRouter()

# Global counter to enforce process concurrency limits
_active_ops_count = 0
_concurrency_lock = asyncio.Lock()
RAW_CREDENTIAL_HEADERS = ("X-LLM-Key", "X-Search-Key")

def client_safe_error() -> dict:
    return {
        "code": "EXECUTION_ERROR",
        "message": "Research execution failed. Check server logs for the redacted diagnostic details.",
        "retryable": True
    }

@router.get("/health/live")
async def health_live():
    """Liveness check: process is up and responsive."""
    return {"status": "ok"}

@router.get("/health/ready")
async def health_ready():
    """Readiness check: verifies storage connectivity and authorization validation."""
    gpt_researcher_ready = True
    try:
        from gpt_researcher import GPTResearcher
    except ImportError:
        gpt_researcher_ready = False

    storage_ready = True
    try:
        # Perform a fast check on the storage backend
        if settings.STORAGE_BACKEND == "redis" and getattr(storage, "degraded", False):
            storage_ready = False
        elif settings.STORAGE_BACKEND == "redis":
            await storage.redis.ping()
    except Exception:
        storage_ready = False

    auth_ready = bool(settings.AUTH_TOKEN)

    status = "ok" if (gpt_researcher_ready and storage_ready) else "degraded"

    return {
        "status": status,
        "gpt_researcher": gpt_researcher_ready,
        "storage": storage_ready,
        "auth": auth_ready
    }

@router.get("/version")
async def version():
    import gpt_researcher
    return {
        "service": "web-intelligence-agent",
        "serviceVersion": "1.0.0",
        "protocolVersion": "1.0.0",
        "engine": {
            "name": "gpt-researcher",
            "version": getattr(gpt_researcher, "__version__", "unknown")
        }
    }

@router.get("/capabilities", response_model=CapabilitiesInfo)
async def capabilities():
    return {
        "service": "web-intelligence-agent",
        "version": "1.0.0",
        "protocol_version": "1.0.0",
        "capabilities": {
            "quick_search": True,
            "standard_research": True,
            "deep_research": True,
            "cancellations": True,
            "citations": True,
            "ssrf_egress_blocking": True
        }
    }

async def background_research_task(req: ResearchRequestInput, reporter: ProgressReporter, headers: dict):
    global _active_ops_count
    op_id = req.operationId

    try:
        # Run execution loop
        result = await conduct_web_research(
            op_id=op_id,
            query=req.query,
            mode=req.mode,
            profile=req.profile,
            limits=req.limits.dict(),
            reporter=reporter,
            headers=headers
        )

        # Save output result
        await storage.save_operation(op_id, result)

    except asyncio.CancelledError:
        logger.warning(f"Operation {op_id} was cancelled during execution.")
        cancelled_state = {
            "operationId": op_id,
            "status": "cancelled",
            "mode": req.mode,
            "profile": req.profile,
            "answer": "Operation was cancelled by the client.",
            "sources": [], "evidence": [], "claims": [], "citations": [], "searchesPerformed": [],
            "metrics": {"startedAt": "", "durationMs": 0, "searchesPerformed": 0, "pagesRead": 0, "sourcesConsidered": 0, "sourcesUsed": 0}
        }
        await storage.save_operation(op_id, cancelled_state)
        await reporter.report("cancelled", "Research task cancelled.")

    except Exception:
        logger.exception("Execution failed for operation %s", op_id)
        failed_state = {
            "operationId": op_id,
            "status": "failed",
            "mode": req.mode,
            "profile": req.profile,
            "answer": "Research execution failed.",
            "sources": [], "evidence": [], "claims": [], "citations": [], "searchesPerformed": [],
            "metrics": {"startedAt": "", "durationMs": 0, "searchesPerformed": 0, "pagesRead": 0, "sourcesConsidered": 0, "sourcesUsed": 0},
            "error": client_safe_error()
        }
        await storage.save_operation(op_id, failed_state)
        await reporter.report("failed", "Research task failed. Check server logs for redacted diagnostics.")

    finally:
        cancellation_manager.unregister_task(op_id)
        async with _concurrency_lock:
            _active_ops_count = max(0, _active_ops_count - 1)

@router.post("/v1/research", status_code=status.HTTP_202_ACCEPTED)
async def start_research(
    req: ResearchRequestInput,
    request: Request,
    idempotency_key: str = Header(None, alias="Idempotency-Key")
):
    global _active_ops_count

    # 1. Enforce and reserve Concurrency slot atomically
    async with _concurrency_lock:
        if _active_ops_count >= settings.MAX_CONCURRENT_OPS:
            raise HTTPException(status_code=429, detail="Concurrency limit reached. Too many active operations.")
        _active_ops_count += 1

    try:
        # 2. Enforce Idempotency Lookups
        lookup_key = idempotency_key or req.idempotencyKey
        if lookup_key:
            ops = await storage.list_operations()
            for existing_op_id, op_state in ops.items():
                if op_state.get("idempotency_key") == lookup_key:
                    logger.info(f"Idempotency hit! Returning existing operation: {existing_op_id}")
                    # Releasing reserved slot since we hit cache
                    async with _concurrency_lock:
                        _active_ops_count = max(0, _active_ops_count - 1)
                    return {"operationId": existing_op_id, "status": op_state.get("status")}

        # 3. Secure initial query validation (check secrets, SSRF URLs if query is an explicit URL)
        query_str = req.query.strip()
        if query_str.lower().startswith(("http://", "https://")):
            if not is_safe_url(query_str, req.profile):
                raise HTTPException(status_code=400, detail="SSRF Validation Error: Target query address is blocked.")

        # 4. Reject raw credential headers outside local loopback mode.
        if not raw_header_credentials_allowed():
            if any(request.headers.get(header) for header in RAW_CREDENTIAL_HEADERS):
                raise HTTPException(
                    status_code=400,
                    detail="Raw provider credentials are accepted only in local deployment mode."
                )
    except Exception as e:
        # Rollback reserved slot if validation fails
        async with _concurrency_lock:
            _active_ops_count = max(0, _active_ops_count - 1)
        raise e

    op_id = req.operationId

    # Register operation shell
    await storage.save_operation(op_id, {
        "operationId": op_id,
        "idempotency_key": lookup_key,
        "attempt_id": req.attemptId,
        "status": "queued",
        "query": req.query,
        "mode": req.mode,
        "profile": req.profile
    })

    # Extract loopback credentials headers
    headers = {
        "X-LLM-Key": request.headers.get("X-LLM-Key", ""),
        "X-Search-Key": request.headers.get("X-Search-Key", "")
    }

    reporter = ProgressReporter(op_id)
    await reporter.report("planning", "Request received. Research task queued.")

    # Spawn research execution task in background
    task = asyncio.create_task(background_research_task(req, reporter, headers))
    cancellation_manager.register_task(op_id, task)

    return {"operationId": op_id, "status": "queued"}

@router.get("/v1/research/{operation_id}/events")
async def get_research_events(operation_id: str):
    """Streams research progress updates as Server-Sent Events."""
    async def event_generator():
        last_idx = 0
        while True:
            events = await storage.get_progress_events(operation_id)
            if len(events) > last_idx:
                for ev in events[last_idx:]:
                    yield {"data": json.dumps(ev)}
                last_idx = len(events)

            op = await storage.get_operation(operation_id)
            if op and op.get("status") in ("completed", "partial", "failed", "cancelled"):
                # Yield any last residual events
                events = await storage.get_progress_events(operation_id)
                for ev in events[last_idx:]:
                    yield {"data": json.dumps(ev)}
                break

            await asyncio.sleep(0.5)

    return EventSourceResponse(event_generator())

@router.get("/v1/research/{operation_id}/result", response_model=ResearchResultResponse)
async def get_research_result(operation_id: str):
    op = await storage.get_operation(operation_id)
    if not op:
        raise HTTPException(status_code=404, detail="Research operation not found")

    return op

@router.post("/v1/research/{operation_id}/cancel")
async def cancel_research(operation_id: str):
    success = await cancellation_manager.cancel_task(operation_id)
    if not success:
        # Check if operation was already finalized
        op = await storage.get_operation(operation_id)
        if op:
            return {"operationId": operation_id, "status": op.get("status")}
        raise HTTPException(status_code=404, detail="Operation task not running or found")

    return {"operationId": operation_id, "status": "cancelled"}
