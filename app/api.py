# app/api.py
import asyncio
import json
import logging
import secrets
from fastapi import APIRouter, Header, HTTPException, Request, status
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from app.config import (
    LOCAL_MODEL_PROVIDERS,
    external_openai_gateway_config,
    local_model_endpoint,
    raw_header_credentials_allowed,
    settings,
)
from app.storage import storage
from app.schemas import ResearchRequestInput, ResearchResultResponse, CapabilitiesInfo
from app.cancellation import cancellation_manager
from app.progress_adapter import ProgressReporter
from app.researcher_adapter import conduct_web_research, _LeaseLostError, schedule_outcome_ingest_task, DEFAULT_INGEST_WAIT_TIMEOUT_S
from app.security import is_gateway_destination_allowed, is_safe_url
from app.metrics import observe_research_result, record_operation_spend

logger = logging.getLogger("web-intelligence")
router = APIRouter()

# External model egress must go through the configured gateway, so a raw LLM
# key header is never accepted. The search key remains a local-mode-only input.
RAW_LLM_CREDENTIAL_HEADER = "X-LLM-Key"
RAW_SEARCH_CREDENTIAL_HEADER = "X-Search-Key"
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
    """Readiness check for dependencies required to accept new work."""
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

    gateway_ready = True
    inference_ready = local_model_endpoint() is not None
    if settings.AI_OPENROUTER_ENABLED:
        try:
            gateway_ready = is_gateway_destination_allowed(
                external_openai_gateway_config()
            )
        except ValueError:
            gateway_ready = False
        inference_ready = gateway_ready

    ready = gpt_researcher_ready and storage_ready and auth_ready and inference_ready

    return JSONResponse(status_code=200 if ready else 503, content={
        "status": "ok" if ready else "degraded",
        "gpt_researcher": gpt_researcher_ready,
        "storage": storage_ready,
        "auth": auth_ready,
        "gateway": gateway_ready,
        "inference": inference_ready,
    })

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

async def _rollback_admission(
    operation_id: str,
    lookup_key: str,
    admission_token: str,
    claimed_lookup_key: bool,
    claimed_operation_id: bool,
    slot_reserved: bool,
    operation_owned: bool,
    spend_reserved: float,
) -> None:
    """Release admission state after a failed request.

    The idempotency lease is authoritative. A retry for the same operation
    shares its slot and budget state, while a retry with a different operation
    owns separate resources that this stale request can safely release. Once
    that distinction is known, the remaining cleanups are isolated so one
    backend error cannot skip the others.
    """
    async def _run(label: str, coro_factory):
        try:
            return await coro_factory()
        except Exception:
            logger.warning("Failed to %s for operation %s during rollback.", label, operation_id, exc_info=True)
            return None

    if claimed_lookup_key:
        try:
            remaining_owner = await storage.release_idempotency_key(
                lookup_key, operation_id, admission_token
            )
        except Exception:
            logger.warning(
                "Failed to release idempotency key for operation %s during rollback.",
                operation_id,
                exc_info=True,
            )
            return
        if remaining_owner == operation_id:
            return
    if operation_owned:
        await _run("release owner lease", lambda: storage.release_operation_lease(operation_id))
    if claimed_operation_id:
        await _run("release operation claim", lambda: storage.release_operation_id(operation_id, lookup_key))
    if slot_reserved:
        await _run("release concurrency slot", lambda: storage.release_concurrency_slot(operation_id))
    if spend_reserved:
        await _run("release spend hold", lambda: storage.release_daily_spend(spend_reserved, operation_id))


async def _report_progress_best_effort(
    reporter: ProgressReporter, stage: str, message: str
) -> None:
    """Publish a progress event without changing durable operation state.

    Callers persist the corresponding state first, so a transient event-stream
    failure must not turn a successfully queued or completed operation into an
    API or execution failure.
    """
    try:
        await reporter.report(stage, message)
    except Exception:
        logger.warning("Failed to publish progress event for stage %s.", stage, exc_info=True)


async def _release_cancellation_safe(awaitable, op_id: str, description: str, timeout=None) -> None:
    """Run a single idempotent release call to completion, surviving any
    number of cancellations of the calling task without ever cancelling the
    release itself.

    A bare `try/except CancelledError: log and move on` leaves the release's
    own outcome ambiguous - the call could have been aborted before it ever
    reached Redis, or it could have completed there with the response simply
    never observed here - and never retries or reconciles it, potentially
    leaving the spend reservation, owner lease, or concurrency slot pinned
    until its TTL expires. Shielding and re-awaiting the same call (rather
    than issuing a fresh one on every retry) is safe here specifically
    because every caller of this helper releases something idempotent:
    removing an already-removed reservation/lease/slot is a no-op, not an
    error, so settling on the one in-flight call is enough.

    Bounded by `timeout` (defaulting to DEFAULT_INGEST_WAIT_TIMEOUT_S, looked
    up fresh on each call so tests can monkeypatch it the same way they do
    for the ingest wait): the Redis client is created without a socket
    timeout (see app/storage.py's `aioredis.from_url` call), so a connection
    that accepts but never answers would otherwise let this loop absorb
    cancellations forever, pinning the task mid-cleanup so it never
    unregisters. Giving up after the deadline leaves this release exactly as
    ambiguous as any other failure case here, falling back on the same TTL
    expiry already relied on elsewhere.

    While waiting, `task` itself is never cancelled by anything in this
    loop - only the outer `wait_for`/`shield` wrapper is, which is what lets
    it survive the calling task being cancelled repeatedly. On giving up
    (wall-clock timeout), though, `task` *is* cancelled rather than left
    running: nothing else tracks it (unlike the per-operation outcome
    ingest, which stays in the module-global pending set for a later
    shutdown-time drain), so an abandoned-but-still-pending release would
    otherwise leak its in-flight Redis call indefinitely. Cancelling it here
    is safe for the same reason re-awaiting it was: the release is
    idempotent, so losing this one attempt is no worse than any other
    failure case this helper already treats as ambiguous and falls back on
    the reservation's TTL for.

    If `task` nonetheless ends up cancelled some other way (an external,
    direct `task.cancel()` on this exact object, from outside this
    function), `while not task.done()` would otherwise exit silently and
    this would return as if the release had settled normally, when it
    never actually ran to completion. The check after the loop catches that
    too.
    """
    if timeout is None:
        timeout = DEFAULT_INGEST_WAIT_TIMEOUT_S
    task = asyncio.ensure_future(awaitable)
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout

    def _abandon():
        task.cancel()

        def _reap(finished_task):
            if not finished_task.cancelled():
                finished_task.exception()  # retrieve and discard so it isn't logged as unhandled

        task.add_done_callback(_reap)

    while not task.done():
        remaining = deadline - loop.time()
        if remaining <= 0:
            logger.warning("Timed out waiting to %s for operation %s.", description, op_id)
            _abandon()
            return
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=remaining)
        except asyncio.TimeoutError:
            logger.warning("Timed out waiting to %s for operation %s.", description, op_id)
            _abandon()
            return
        except asyncio.CancelledError:
            continue
        except Exception:
            logger.warning("Failed to %s for operation %s.", description, op_id, exc_info=True)
            return
    if task.cancelled():
        logger.warning("Release task was cancelled rather than completed to %s for operation %s.", description, op_id)


async def background_research_task(
    req: ResearchRequestInput,
    reporter: ProgressReporter,
    headers: dict,
    spend_reserved: float = 0.0,
    accounting_delegated: bool = False,
):
    op_id = req.operationId
    spend_reconciled = False
    ingest_task = None

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

        # Persist the terminal state first so a crash before the progress event
        # cannot let an SSE consumer observe a terminal announcement for a result
        # that was never stored. The stream waits (bounded) for the terminal
        # event after it sees this state, so the event is still delivered.
        await storage.save_operation(op_id, result)
        await _report_progress_best_effort(
            reporter, result["status"], f"Research task {result['status']}."
        )
        observe_research_result(result)
        # Reconcile the admission reservation to the observed cost in one
        # atomic step, so another replica cannot reserve the temporarily freed
        # capacity in between. Isolated from the result path: a transient
        # accounting failure must not overwrite the completed result with a
        # failure, and the operation is left fail-closed (spend_reconciled stays
        # True) so the finally block does not release the hold and undercharge.
        # When AI-OpenRouter is authoritative, this service holds no reservation
        # and records nothing, so it cannot keep a local running total that
        # disagrees with the budget owner. When the work ran on a local model
        # (accounting_delegated is False exactly in that case, given the two
        # admission states in start_research), the reservation is reconciled
        # down to zero rather than the provider-agnostic token-cost heuristic:
        # local inference has no real external cost, so charging that estimate
        # against DAILY_SPEND_LIMIT_USD would eventually exhaust it on
        # credential-free local runs alone. estimatedModelCostUsd is still
        # returned to the caller in the result for visibility; it is just not
        # charged against this local guard.
        if accounting_delegated:
            spend_reconciled = True
        elif spend_reserved:
            try:
                await storage.reconcile_daily_spend(spend_reserved, 0.0, settings.DAILY_SPEND_LIMIT_USD, op_id)
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
        ingest_task = schedule_outcome_ingest_task(result, op_id, req.mode, inputs=req.inputs)

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
        await _report_progress_best_effort(
            reporter, "cancelled", "Research task cancelled."
        )
        observe_research_result(cancelled_state)
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
        await _report_progress_best_effort(
            reporter,
            "failed",
            "Research task failed. Check server logs for redacted diagnostics.",
        )
        observe_research_result(failed_state)

    finally:
        # This task's own HTTP request (POST /v1/research) already returned
        # 202 before this coroutine started, so Azure Container Apps' request
        # concurrency scaler has no open connection tying this replica to this
        # operation; the replica is free to scale toward zero once the SSE
        # consumer disconnects. A detached Brain Memory ingest task has no
        # such protection and can be killed mid-flight before it completes.
        # Running the await here, in `finally`, covers both the normal-return
        # and cancellation exit paths uniformly: inline right after
        # schedule_outcome_ingest_task() would be skipped if cancellation
        # landed on exactly that await, since it is inside the `try` whose
        # `except asyncio.CancelledError` above never retries it.
        #
        # Awaits `ingest_task` directly - this operation's own task - rather
        # than draining every operation's pending ingest: the per-operation
        # task is what this function actually needs to wait for, and nothing
        # else's unrelated in-flight Brain request should hold up this
        # operation's own lease/concurrency-slot release.
        #
        # asyncio.shield, not a bare await: a cancellation landing exactly
        # while suspended here would otherwise interrupt only the *waiting*,
        # not `ingest_task` itself - a bare await propagates cancellation into
        # what it awaits, so without the shield this function would move on
        # and release the owner lease/concurrency slot while the ingest it
        # already scheduled is still mid-flight, now with nothing tracking
        # it. Shielding keeps that ingest running rather than cancelling it
        # out from under this wait when this task itself gets cancelled
        # again right here - this function's own wait on it is still only up
        # to the bound below, not a guarantee of running it to completion.
        #
        # Shielding alone does not make this wait durable against repeated
        # cancellation, though: the outer await still raises CancelledError
        # immediately when a cancellation lands exactly on it - shield does
        # not change *this* function's own cancellation semantics, only
        # whether `ingest_task` survives being cancelled out from under it.
        # Looped rather than a single retry, so no number of repeated
        # cancellations of this task - including quiesce_tasks() cancelling
        # it again during shutdown, harmlessly, while it waits here (this
        # task stays registered through its own cleanup; see the
        # unregister_task() call at the end of this block) - can ever
        # actually cancel the ingest it is protecting.
        #
        # Bounded to DEFAULT_INGEST_WAIT_TIMEOUT_S overall, not just per
        # cancellation: the Brain Memory HTTP call's own 5s socket timeout
        # resets on every partial read, so a connection that trickles data
        # slowly enough can run far longer than that without ever tripping
        # it. asyncio.wait_for's own timeout only cancels the *outer* shield
        # wrapper it is given, exactly like an external cancellation would -
        # ingest_task itself keeps running, still tracked for the eventual
        # shutdown-time drain, just no longer blocking this operation's own
        # lease/concurrency-slot release.
        if ingest_task is not None:
            loop = asyncio.get_running_loop()
            deadline = loop.time() + DEFAULT_INGEST_WAIT_TIMEOUT_S
            while not ingest_task.done():
                remaining = deadline - loop.time()
                if remaining <= 0:
                    logger.warning(
                        "Brain Memory ingest for operation %s did not finish within %.1fs; releasing cleanup without it.",
                        op_id, DEFAULT_INGEST_WAIT_TIMEOUT_S,
                    )
                    break
                try:
                    await asyncio.wait_for(asyncio.shield(ingest_task), timeout=remaining)
                except asyncio.TimeoutError:
                    logger.warning(
                        "Brain Memory ingest for operation %s did not finish within %.1fs; releasing cleanup without it.",
                        op_id, DEFAULT_INGEST_WAIT_TIMEOUT_S,
                    )
                    break
                except asyncio.CancelledError:
                    continue
                except Exception:
                    logger.warning("Brain Memory ingest failed for operation %s.", op_id, exc_info=True)
                    break
        # Cleanup steps below are independent: a failure in one (for example a
        # transient Redis error releasing the spend hold) must not skip the
        # others, or the owner lease and concurrency slot would stay pinned for
        # the full lease TTL and the task would leak in the local registry.
        # Each one also survives cancellation rather than treating it as
        # failure, through to unregistration: the task stays registered for
        # quiesce_tasks() to find throughout this whole phase (deliberately -
        # see the unregister_task() comment below), so a cancellation landing
        # on any one of these awaits must not propagate out and skip the rest
        # the same way an uncaught one would - nor leave that one release's
        # own outcome ambiguous, which is what _release_cancellation_safe is
        # for.
        if spend_reserved and not spend_reconciled:
            await _release_cancellation_safe(
                storage.release_daily_spend(spend_reserved, op_id), op_id, "release spend hold"
            )
        await _release_cancellation_safe(storage.release_operation_lease(op_id), op_id, "release owner lease")
        await _release_cancellation_safe(storage.release_concurrency_slot(op_id), op_id, "release concurrency slot")
        # Unregistered last, once every cleanup step above has actually run:
        # quiesce_tasks() (shutdown) only cancels/awaits tasks still in
        # active_tasks, so unregistering any earlier would let shutdown race
        # past this task while it still has cleanup left - losing the lease,
        # concurrency slot, or spend-hold release and leaving them pinned
        # until their TTLs expire. cancel_task()'s own terminal-status check
        # is what keeps a merely-registered, already-finished operation from
        # being spuriously cancelled in the meantime, so unregistration
        # timing does not have to do that job too. Synchronous, so it cannot
        # itself be interrupted by cancellation - only Exception applies.
        try:
            cancellation_manager.unregister_task(op_id)
        except Exception:
            logger.warning("Failed to unregister task for operation %s.", op_id, exc_info=True)

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
    admission_token = secrets.token_hex(16)

    claimed_lookup_key = None
    claimed_operation_id = False
    slot_reserved = False
    operation_owned = False
    try:
        # 1. Atomic idempotency check-and-reserve, before any other admission
        # check. A retry of already-accepted work must resolve to its existing
        # operation even when a later admission check (e.g. gateway
        # configuration, changed by a rolling restart or a replica missing a
        # secret) would now reject a *new* request — those checks only matter
        # for new work, and applying them to a retry would turn an ambiguous
        # submission into a spurious rejection instead of letting the client
        # reconcile with its persisted operation.
        existing_op_id = await storage.claim_idempotency_key(
            lookup_key, req.operationId, admission_token
        )
        if existing_op_id:
            logger.info("Idempotency hit for key %s, returning existing operation: %s", lookup_key, existing_op_id)
            op_state = await storage.get_operation(existing_op_id)
            if not op_state:
                raise HTTPException(
                    status_code=409,
                    detail="Idempotency-key admission is pending; retry shortly.",
                    headers={"Retry-After": "1"},
                )
            return {"operationId": existing_op_id, "status": op_state.get("status", "unknown")}
        claimed_lookup_key = lookup_key

        # 2. Secure initial query validation (check secrets, SSRF URLs if query is an explicit URL)
        query_str = req.query.strip()
        if query_str.lower().startswith(("http://", "https://")):
            if not is_safe_url(query_str, req.profile):
                raise HTTPException(status_code=400, detail="SSRF Validation Error: Target query address is blocked.")

        # 2. External model egress is only permitted through AI-OpenRouter.
        if request.headers.get(RAW_LLM_CREDENTIAL_HEADER):
            raise HTTPException(
                status_code=400,
                detail="Raw LLM credentials are not accepted; use the configured AI-OpenRouter gateway."
            )
        # The search key is an independent credential path, not an external-model
        # credential. Accept it only where request-scoped headers are honored, so
        # a remote caller is never told its key was used when it was ignored.
        if request.headers.get(RAW_SEARCH_CREDENTIAL_HEADER) and not raw_header_credentials_allowed():
            raise HTTPException(
                status_code=400,
                detail="Raw search credentials are accepted only in local deployment mode."
            )

        if bool(req.model_provider) != bool(req.model_name):
            raise HTTPException(
                status_code=400,
                detail="model_provider and model_name must be provided together."
            )
        # External-model selection is only permitted through the gateway, and the
        # gateway serves the OpenAI-compatible provider only. Local providers
        # (e.g. ollama) need no external egress and are always allowed, so a
        # broken cloud configuration must not disable local-first operation.
        provider = (req.model_provider or "").lower()
        if provider and provider != "openai" and provider not in LOCAL_MODEL_PROVIDERS:
            raise HTTPException(
                status_code=400,
                detail="AI-OpenRouter supports the OpenAI-compatible model provider only."
            )
        external_gateway_selected = False
        if provider and provider not in LOCAL_MODEL_PROVIDERS:
            # Explicit external selection: the gateway is required and must be valid.
            try:
                gateway = external_openai_gateway_config()
            except ValueError as exc:
                raise HTTPException(
                    status_code=503,
                    detail="AI-OpenRouter gateway configuration is unavailable."
                ) from exc
            if not gateway:
                raise HTTPException(
                    status_code=503,
                    detail="External model selection requires the configured AI-OpenRouter gateway."
                )
            if not is_gateway_destination_allowed(gateway):
                raise HTTPException(
                    status_code=503,
                    detail="AI-OpenRouter gateway configuration is unavailable."
                )
            external_gateway_selected = True
        elif provider:
            # Explicit local provider: the configured endpoint must actually be
            # local. A provider name alone does not establish locality, so an
            # ``OLLAMA_BASE_URL`` pointed at a public host would otherwise be an
            # unmetered external egress path that bypasses the gateway.
            if local_model_endpoint() is None:
                raise HTTPException(
                    status_code=400,
                    detail="Local model endpoint is not configured as a local address."
                )
        else:
            # No explicit selection: an enabled gateway serves the request, so a
            # malformed config must fail closed here rather than mid-task. With no
            # gateway the request defaults to the local model, which must be a
            # validated local endpoint for the same reason as above.
            try:
                gateway = external_openai_gateway_config()
            except ValueError as exc:
                raise HTTPException(
                    status_code=503,
                    detail="AI-OpenRouter gateway configuration is unavailable."
                ) from exc
            if gateway:
                if not is_gateway_destination_allowed(gateway):
                    raise HTTPException(
                        status_code=503,
                        detail="AI-OpenRouter gateway configuration is unavailable."
                    )
                external_gateway_selected = True
            elif local_model_endpoint() is None:
                raise HTTPException(
                    status_code=503,
                    detail="No external gateway is configured and the local model endpoint is not local."
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

        # 3. Atomically reserve budget for this new operation (the idempotency
        # hit above already returned for a retry). The check-and-reserve
        # happens in one step so concurrent replicas cannot all pass a
        # non-atomic read-then-admit and collectively exceed the ceiling. The
        # reservation is reconciled to the observed cost on completion, or
        # released on cancel/failure.
        # The local spend ceiling guards externally-metered model work. When the
        # request is served by AI-OpenRouter it is the budget authority, so this
        # service must not also reject or throttle the operation against its own
        # independent ceiling. Local-model work keeps the local ceiling: its
        # reservation is minor and releasing the hold on completion keeps the
        # running total a harmless (zero-cost) figure.
        enforce_local_budget = not external_gateway_selected
        spend_reserved = 0.0
        if enforce_local_budget:
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
        await _rollback_admission(
            req.operationId, lookup_key, admission_token, claimed_lookup_key,
            claimed_operation_id, slot_reserved, operation_owned, spend_reserved,
        )
        raise e

    op_id = req.operationId
    task_started = False

    try:
        # Register operation shell
        operation_saved = await storage.save_admitted_operation(op_id, {
            "operationId": op_id,
            "idempotency_key": lookup_key,
            "attempt_id": req.attemptId,
            "status": "queued",
            "query": req.query,
            "mode": req.mode,
            "profile": req.profile
        }, lookup_key, admission_token)
        if not operation_saved:
            raise HTTPException(
                status_code=409,
                detail="Idempotency-key admission lease expired; retry the request."
            )

        # Extract loopback credentials headers
        headers = {
            "X-LLM-Key": request.headers.get("X-LLM-Key", ""),
            "X-Search-Key": request.headers.get("X-Search-Key", "")
        }

        reporter = ProgressReporter(op_id)
        await _report_progress_best_effort(
            reporter, "planning", "Request received. Research task queued."
        )

        # Spawn research execution task in background
        task = asyncio.create_task(background_research_task(
            req, reporter, headers, spend_reserved, accounting_delegated=not enforce_local_budget
        ))
        cancellation_manager.register_task(op_id, task)
        task_started = True
    except Exception as e:
        if not task_started:
            await _rollback_admission(
                req.operationId, lookup_key, admission_token, claimed_lookup_key,
                claimed_operation_id, slot_reserved, operation_owned, spend_reserved,
            )
        raise e

    return {"operationId": op_id, "status": "queued"}

@router.get("/v1/research/{operation_id}/events")
async def get_research_events(operation_id: str):
    """Streams research progress updates as Server-Sent Events."""
    op = await storage.get_operation(operation_id)
    if not op:
        raise HTTPException(status_code=404, detail="Research operation not found")

    async def event_generator():
        cursor = None
        # The terminal state is persisted before the terminal progress event is
        # published, so a crash can never announce completion for an unstored
        # result. That means observing a terminal status does not guarantee the
        # final event is readable yet, so the stream waits briefly for it (or for
        # any other residual event) before closing instead of closing early.
        terminal_event_grace_seconds = 0.5
        saw_terminal_event = False
        while True:
            entries = await storage.get_progress_events_after(operation_id, cursor)
            for event_cursor, ev in entries:
                yield {"data": json.dumps(ev)}
                cursor = event_cursor
                if ev.get("stage") in TERMINAL_STATUSES:
                    saw_terminal_event = True

            op = await storage.get_operation(operation_id)
            if op and op.get("status") in TERMINAL_STATUSES:
                waited = 0.0
                while not saw_terminal_event and waited < terminal_event_grace_seconds:
                    await asyncio.sleep(0.1)
                    waited += 0.1
                    for event_cursor, ev in await storage.get_progress_events_after(operation_id, cursor):
                        yield {"data": json.dumps(ev)}
                        cursor = event_cursor
                        if ev.get("stage") in TERMINAL_STATUSES:
                            saw_terminal_event = True
                # Flush anything else that landed after the last poll.
                for event_cursor, ev in await storage.get_progress_events_after(operation_id, cursor):
                    yield {"data": json.dumps(ev)}
                    cursor = event_cursor
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
