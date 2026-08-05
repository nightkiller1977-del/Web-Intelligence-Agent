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
from app.metrics import research_duration, sources_fetched

logger = logging.getLogger("web-intelligence")
router = APIRouter()

# Global counter to enforce process concurrency limits
_active_ops_count = 0
_concurrency_lock = asyncio.Lock()
RAW_CREDENTIAL_HEADERS = ("X-LLM-Key", "X-Search-Key")
UNSUPPORTED_MODEL_LIMIT_FIELDS = (
    "maximumModelCalls",
    "maximumModelTokens",
    "maximumModelCostUsd",
)
UNSUPPORTED_MODEL_PREFERENCE_FIELDS = (
    "model_provider",
    "model_name",
)
TERMINAL_STATUSES = ("completed", "partial", "failed", "cancelled")

def client_safe_error() -> dict:
    return {
        "code": "EXECUTION_ERROR",
        "message": "Research execution failed. Check server logs for the redacted diagnostic details.",
        "retryable": True
    }

def unsupported_model_limit_fields(limits) -> list[str]:
    return [
        field
        for field in UNSUPPORTED_MODEL_LIMIT_FIELDS
        if getattr(limits, field, None) is not None
    ]

def unsupported_model_preference_fields(req: ResearchRequestInput) -> list[str]:
    return [
        field
        for field in UNSUPPORTED_MODEL_PREFERENCE_FIELDS
        if getattr(req, field, None) is not None
    ]

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

    status = "ok" if (gpt_researcher_ready and storage_ready and auth_ready) else "degraded"

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
            "ssrf_egress_blocking": True,
            "ssrf_url_query_validation": True,
            "ssrf_source_result_redaction": True,
            "ssrf_prefetch_http_guard": True
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
        metrics = result.get("metrics") or {}
        research_duration.labels(
            agent_profile=req.profile,
            mode=req.mode
        ).observe(metrics.get("durationMs", 0))
        sources_fetched.labels(agent_profile=req.profile).observe(metrics.get("sourcesConsidered", 0))

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

    lookup_key = idempotency_key or req.idempotencyKey
    if not lookup_key:
        raise HTTPException(
            status_code=400,
            detail="Idempotency-Key header or idempotencyKey body field is required."
        )

    claimed_lookup_key = None
    slot_reserved = False
    try:
        # 1. Secure initial query validation (check secrets, SSRF URLs if query is an explicit URL)
        query_str = req.query.strip()
        if query_str.lower().startswith(("http://", "https://")):
            if not is_safe_url(query_str, req.profile):
                raise HTTPException(status_code=400, detail="SSRF Validation Error: Target query address is blocked.")

        # 2. Reject raw credential headers outside local loopback mode.
        if not raw_header_credentials_allowed():
            if any(request.headers.get(header) for header in RAW_CREDENTIAL_HEADERS):
                raise HTTPException(
                    status_code=400,
                    detail="Raw provider credentials are accepted only in local deployment mode."
                )

        unsupported_limits = unsupported_model_limit_fields(req.limits)
        if unsupported_limits:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Unsupported model budget limits: "
                    f"{', '.join(unsupported_limits)}. "
                    "Use duration, search, page, source, and memory limits for this adapter."
                )
            )

        unsupported_preferences = unsupported_model_preference_fields(req)
        if unsupported_preferences:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Unsupported model preference fields: "
                    f"{', '.join(unsupported_preferences)}. "
                    "Configure model/provider defaults through deployment environment variables."
                )
            )

        if req.sourcePolicy:
            raise HTTPException(
                status_code=400,
                detail="sourcePolicy is not supported by this adapter yet. Omit it or enforce source constraints before submitting the request."
            )

        # 3. Atomic idempotency check-and-reserve before new-work concurrency limiting.
        existing_op_id = await storage.claim_idempotency_key(lookup_key, req.operationId)
        if existing_op_id:
            logger.info("Idempotency hit for key %s, returning existing operation: %s", lookup_key, existing_op_id)
            op_state = await storage.get_operation(existing_op_id)
            return {"operationId": existing_op_id, "status": op_state.get("status") if op_state else "unknown"}
        claimed_lookup_key = lookup_key

        # 4. Enforce and reserve a concurrency slot only for new operations.
        async with _concurrency_lock:
            if _active_ops_count >= settings.MAX_CONCURRENT_OPS:
                raise HTTPException(status_code=429, detail="Concurrency limit reached. Too many active operations.")
            _active_ops_count += 1
            slot_reserved = True
    except Exception as e:
        # Rollback reserved slot if validation fails
        if claimed_lookup_key:
            await storage.release_idempotency_key(claimed_lookup_key, req.operationId)
        if slot_reserved:
            async with _concurrency_lock:
                _active_ops_count = max(0, _active_ops_count - 1)
        raise e

    op_id = req.operationId
    task_started = False

    try:
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
        task_started = True
    except Exception as e:
        if not task_started:
            if claimed_lookup_key:
                await storage.release_idempotency_key(claimed_lookup_key, req.operationId)
            if slot_reserved:
                async with _concurrency_lock:
                    _active_ops_count = max(0, _active_ops_count - 1)
        raise e

    return {"operationId": op_id, "status": "queued"}

@router.get("/v1/research/{operation_id}/events")
async def get_research_events(operation_id: str):
    """Streams research progress updates as Server-Sent Events."""
    op = await storage.get_operation(operation_id)
    if not op:
        raise HTTPException(status_code=404, detail="Research operation not found")

    async def event_generator():
        last_idx = 0
        while True:
            events = await storage.get_progress_events(operation_id)
            if len(events) > last_idx:
                for ev in events[last_idx:]:
                    yield {"data": json.dumps(ev)}
                last_idx = len(events)

            op = await storage.get_operation(operation_id)
            if op and op.get("status") in TERMINAL_STATUSES:
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

    if op.get("status") not in TERMINAL_STATUSES:
        return {
            "operationId": operation_id,
            "status": op.get("status", "queued"),
            "mode": op.get("mode", "standard"),
            "profile": op.get("profile", "general"),
            "answer": None,
            "sources": [],
            "evidence": [],
            "claims": [],
            "citations": [],
            "searchesPerformed": [],
            "metrics": None
        }

    return op

@router.post("/v1/research/{operation_id}/cancel")
async def cancel_research(operation_id: str):
    success = await cancellation_manager.cancel_task(operation_id, storage.get_operation)
    if not success:
        # Check if operation was already finalized
        op = await storage.get_operation(operation_id)
        if op:
            return {"operationId": operation_id, "status": op.get("status")}
        raise HTTPException(status_code=404, detail="Operation task not running or found")

    return {"operationId": operation_id, "status": "cancelled"}
