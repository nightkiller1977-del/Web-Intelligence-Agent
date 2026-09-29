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
from app.researcher_adapter import conduct_web_research, _LeaseLostError, schedule_outcome_ingest
from app.security import is_safe_url
from app.metrics import observed_result_cost, observe_research_result, record_operation_spend

logger = logging.getLogger("web-intelligence")
router = APIRouter()

RAW_CREDENTIAL_HEADERS = ("X-LLM-Key", "X-Search-Key")
TERMINAL_STATUSES = ("completed", "partial", "failed", "cancelled")

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
            "source_level_citations": True,
            "citations": True,
            "structured_evidence": True,
            "claim_verification": True,
            "source_policy": True,
            "model_budget_limits": True,
            "model_preferences": True,
            "ssrf_egress_blocking": True,
            "ssrf_url_query_validation": True,
            "ssrf_source_result_redaction": True,
            "ssrf_prefetch_http_guard": True
        }
    }

async def background_research_task(req: ResearchRequestInput, reporter: ProgressReporter, headers: dict, spend_reserved: float = 0.0):
    op_id = req.operationId
    spend_reconciled = False

    try:
        # Ownership was claimed at admission time (before any queued state was
        # published), so this task is the sole owner and may proceed.
        await storage.save_operation(op_id, {"status": "running"})
        await reporter.report("planning", "Research task started.")

        # Run execution loop
        result = await conduct_web_research(
            op_id=op_id,
            query=req.query,
            mode=req.mode,
            profile=req.profile,
            limits=req.limits.model_dump(),
            source_policy=req.sourcePolicy,
            freshness=req.freshness,
            inputs=req.inputs,
            model_provider=req.model_provider,
            model_name=req.model_name,
            require_claim_verification=bool(req.requireClaimVerification),
            reporter=reporter,
            headers=headers
        )

        # Save output result
        await storage.save_operation(op_id, result)
        observe_research_result(result)
        # Reconcile the admission reservation to the observed cost in one
        # atomic step, so another replica cannot reserve the temporarily freed
        # capacity in between. Isolated from the result path: a transient
        # accounting failure must not overwrite the completed result with a
        # failure, and the operation is left fail-closed (spend_reconciled stays
        # True) so the finally block does not release the hold and undercharge.
        actual = observed_result_cost(result)
        if spend_reserved:
            try:
                await storage.reconcile_daily_spend(spend_reserved, actual, settings.DAILY_SPEND_LIMIT_USD, op_id)
                spend_reconciled = True
            except Exception:
                spend_reconciled = True
                logger.warning("Failed to reconcile spend for operation %s; retaining the reservation.", op_id, exc_info=True)
        else:
            try:
                await record_operation_spend(storage, result)
            except Exception:
                logger.warning("Failed to record observed spend for operation %s.", op_id, exc_info=True)
        # Only after the result is durable may the optional Brain outcome ingest
        # be scheduled; otherwise Brain could record a completed/partial outcome
        # for a result that was never stored.
        schedule_outcome_ingest(result, op_id, req.mode)
        await reporter.report(result["status"], f"Research task {result['status']}.")

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
        observe_research_result(cancelled_state)
        await reporter.report("cancelled", "Research task cancelled.")

    except _LeaseLostError:
        # Ownership was lost mid-run: skip persistence so this worker cannot
        # overwrite the terminal state reconciliation (or the new owner) wrote.
        logger.warning("Operation %s lost its ownership lease; not persisting a failed state.", op_id)
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
            # Input-context limitations are computed before research runs, so a
            # failure after that point must still report them to the caller.
            "limitations": req.limitations_context(),
            "error": client_safe_error()
        }
        await storage.save_operation(op_id, failed_state)
        observe_research_result(failed_state)
        await reporter.report("failed", "Research task failed. Check server logs for redacted diagnostics.")

    finally:
        # Cleanup steps are independent: a failure in one (for example a
        # transient Redis error releasing the spend hold) must not skip the
        # others, or the owner lease and concurrency slot would stay pinned for
        # the full lease TTL and the task would leak in the local registry.
        if spend_reserved and not spend_reconciled:
            try:
                await storage.release_daily_spend(spend_reserved, op_id)
            except Exception:
                logger.warning("Failed to release spend hold for operation %s.", op_id, exc_info=True)
        try:
            cancellation_manager.unregister_task(op_id)
        except Exception:
            logger.warning("Failed to unregister task for operation %s.", op_id, exc_info=True)
        try:
            await storage.release_operation_lease(op_id)
        except Exception:
            logger.warning("Failed to release owner lease for operation %s.", op_id, exc_info=True)
        try:
            await storage.release_concurrency_slot(op_id)
        except Exception:
            logger.warning("Failed to release concurrency slot for operation %s.", op_id, exc_info=True)

@router.post("/v1/research", status_code=status.HTTP_202_ACCEPTED)
async def start_research(
    req: ResearchRequestInput,
    request: Request,
    idempotency_key: str = Header(None, alias="Idempotency-Key")
):
    claimed_lookup_key = None
    claimed_operation_id = False
    slot_reserved = False
    operation_owned = False
    spend_reserved = 0.0

    lookup_key = idempotency_key or req.idempotencyKey
    if not lookup_key:
        raise HTTPException(
            status_code=400,
            detail="Idempotency-Key header or idempotencyKey body field is required."
        )

    claimed_lookup_key = None
    claimed_operation_id = False
    slot_reserved = False
    operation_owned = False
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

        if bool(req.model_provider) != bool(req.model_name):
            raise HTTPException(
                status_code=400,
                detail="model_provider and model_name must be provided together."
            )

        unsupported_source_policy_keys = set((req.sourcePolicy or {}).keys()) - {"allowedDomains"}
        if unsupported_source_policy_keys:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Unsupported sourcePolicy fields: "
                    f"{', '.join(sorted(unsupported_source_policy_keys))}. "
                    "Only allowedDomains is supported."
                )
            )

        # 3. Atomic idempotency check-and-reserve before new-work concurrency limiting.
        existing_op_id = await storage.claim_idempotency_key(lookup_key, req.operationId)
        if existing_op_id:
            logger.info("Idempotency hit for key %s, returning existing operation: %s", lookup_key, existing_op_id)
            op_state = await storage.get_operation(existing_op_id)
            return {"operationId": existing_op_id, "status": op_state.get("status") if op_state else "unknown"}
        claimed_lookup_key = lookup_key

        # 3b. Atomically reserve budget for this new operation. The check-and-
        # reserve happens in one step so concurrent replicas cannot all pass a
        # non-atomic read-then-admit and collectively exceed the ceiling. The
        # reservation is reconciled to the observed cost on completion, or
        # released on cancel/failure. Placed after the idempotency-hit return
        # so a retry of already-accepted work still resolves to its operation.
        reserve_amount = (
            req.limits.maximumModelCostUsd
            if req.limits and req.limits.maximumModelCostUsd is not None
            else settings.DEFAULT_OPERATION_COST_RESERVE_USD
        )
        if not await storage.reserve_daily_spend(reserve_amount, settings.DAILY_SPEND_LIMIT_USD, req.operationId):
            raise HTTPException(
                status_code=429,
                detail="Daily spend limit reached. New research operations are paused until the limit resets."
            )
        spend_reserved = reserve_amount

        operation_claimed = await storage.claim_operation_id(req.operationId, lookup_key)
        if not operation_claimed:
            raise HTTPException(
                status_code=409,
                detail="operationId is already associated with a different idempotency key."
            )
        claimed_operation_id = True

        # 4. Enforce a service-wide concurrency slot only for new operations.
        # The reservation lives in storage so the limit is shared across
        # instances/workers rather than being per-process.
        if not await storage.acquire_concurrency_slot(req.operationId):
            raise HTTPException(status_code=429, detail="Concurrency limit reached. Too many active operations.")
        slot_reserved = True

        # 5. Claim exclusive ownership *before* any queued state is published.
        # Publishing queued state first would let a concurrent startup
        # reconciliation observe an ownerless queued operation and fail it out
        # from under the worker that is about to run it.
        if not await storage.begin_operation(req.operationId):
            raise HTTPException(
                status_code=409,
                detail="operationId is already being processed by another live instance."
            )
        operation_owned = True
    except Exception as e:
        # Rollback reserved slot if validation fails
        if operation_owned:
            await storage.release_operation_lease(req.operationId)
        if claimed_operation_id:
            await storage.release_operation_id(req.operationId, lookup_key)
        if claimed_lookup_key:
            await storage.release_idempotency_key(claimed_lookup_key, req.operationId)
        if slot_reserved:
            await storage.release_concurrency_slot(req.operationId)
        if spend_reserved:
            try:
                await storage.release_daily_spend(spend_reserved, req.operationId)
            except Exception:
                logger.warning("Failed to release admission spend hold during rollback.", exc_info=True)
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
        task = asyncio.create_task(background_research_task(req, reporter, headers, spend_reserved))
        cancellation_manager.register_task(op_id, task)
        task_started = True
    except Exception as e:
        if not task_started:
            if operation_owned:
                await storage.release_operation_lease(req.operationId)
            if claimed_operation_id:
                await storage.release_operation_id(req.operationId, lookup_key)
            if claimed_lookup_key:
                await storage.release_idempotency_key(claimed_lookup_key, req.operationId)
            if slot_reserved:
                await storage.release_concurrency_slot(req.operationId)
            if spend_reserved:
                try:
                    await storage.release_daily_spend(spend_reserved, req.operationId)
                except Exception:
                    logger.warning("Failed to release admission spend hold during rollback.", exc_info=True)
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

    # A failed or cancelled operation's side effects are not guaranteed to have
    # settled; callers must reconcile rather than assume a clean stop.
    op["requiresReconciliation"] = op.get("status") in ("failed", "cancelled")
    op["degraded"] = bool(op.get("degraded") or getattr(storage, "degraded", False))
    if op["degraded"] and not op.get("degradedReasons"):
        op["degradedReasons"] = ["Durable storage is degraded; results may not survive a restart."]
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
