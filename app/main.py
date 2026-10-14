# app/main.py
import asyncio
import logging
import secrets
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app.config import auth_is_configured, settings, unauthenticated_docs_allowed
from app.storage import storage, StorageUnavailable
from app.cancellation import cancellation_manager
from app.researcher_adapter import flush_pending_ingest_tasks
from app.api import router
from app.grafana_observability import ObserveASGI, emitter, emit as emit_observability

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("web-intelligence")
PUBLIC_PATHS = {"/health/live", "/health/ready", "/capabilities", "/version", "/metrics"}
DOCS_PATHS = {"/docs", "/redoc", "/openapi.json", "/docs/oauth2-redirect"}

# Azure Container Apps (and Kubernetes-style platforms generally) default to
# a 30s SIGTERM-to-SIGKILL grace period.
ASSUMED_PLATFORM_GRACE_PERIOD_S = 30.0

# Reserved, after the drain sequence below returns (however it returns),
# for cancellation_manager.shutdown() and the storage-backend teardown that
# follows it - both already bounded elsewhere (shutdown() awaits one
# already-cancelled listener task; storage teardown closes a connection
# pool), but still real wall-clock time that must come out of the same
# overall budget rather than being assumed free.
SHUTDOWN_FINAL_STEPS_MARGIN_S = 3.0

# quiesce_tasks()'s own wait and flush_pending_ingest_tasks()'s backstop are
# each sized independently for their own purpose - the former to reliably
# outlast a single task's own worst-case serial cleanup, the latter for a
# slow-but-not-hung Brain Memory response. Earlier versions of this budget
# tried to keep their *sum* under the platform's actual limit by picking
# SHUTDOWN_DRAIN_BUDGET_S and a second, independent "one more bounded
# attempt" timeout by hand - which quietly stopped adding up correctly the
# moment either of the constants that second timeout was borrowed from
# changed for its own, unrelated reason (see this PR's review history).
# Anchoring every subsequent wait to the single ASSUMED_PLATFORM_GRACE_PERIOD_S
# deadline set below, instead, keeps the total bounded regardless of how
# any individual piece is retuned later. In the rare case this still cuts a
# task's cleanup short, that is the same already-accepted TTL-based
# fallback this cleanup path relies on elsewhere (a stuck reservation
# expires on its own), not a new kind of loss.
SHUTDOWN_DRAIN_BUDGET_S = 20.0

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing storage backend lifecycle...")
    try:
        await storage.init()
    except StorageUnavailable as exc:
        # Fail closed: a deployment that requires Redis must not silently serve
        # from process-local memory. Refusing to start surfaces the misconfig
        # instead of losing operations on the next restart.
        emit_observability("startup_failed", reason="storage_unavailable")
        logger.error("Refusing to start: %s", exc)
        raise
    await storage.mark_stale_operations()
    redis_client = getattr(storage, "redis", None)

    async def reconcile_stale_operations():
        # A replacement replica usually runs the startup scan before the
        # crashed owner's lease expires, so a one-shot scan can miss it. Keep
        # reconciling periodically so an ownerless operation eventually becomes
        # terminal instead of staying queued/running forever.
        while True:
            await asyncio.sleep(settings.STALE_RECONCILE_INTERVAL_SECONDS)
            try:
                await storage.mark_stale_operations()
            except Exception:
                logger.warning("Periodic stale-operation reconciliation failed; will retry.", exc_info=True)

    # Recurring reconciliation is only meaningful for lease-aware (Redis)
    # storage. With the single-process in-memory backend the startup scan is
    # the only valid reconciliation point, so do not run a periodic one.
    reconcile_task = None
    if redis_client and not getattr(storage, "degraded", False):
        reconcile_task = asyncio.create_task(reconcile_stale_operations())
    if redis_client and not getattr(storage, "degraded", False):
        await cancellation_manager.init(redis_client)
    emit_observability(
        "storage_initialized",
        storage_degraded=bool(getattr(storage, "degraded", False)),
        redis_enabled=bool(redis_client),
    )
    yield
    if reconcile_task is not None:
        reconcile_task.cancel()
        try:
            await reconcile_task
        except asyncio.CancelledError:
            pass
    # Quiesce in-flight research first: a research task that finished after the
    # ingest snapshot below would schedule an untracked ingest into a closing
    # loop. Cancel/await active tasks, then drain the ingests they produced.
    #
    # Bounded as one sequence by SHUTDOWN_DRAIN_BUDGET_S (see its own comment
    # above): a timeout here only cancels this wrapper, not the individual
    # background tasks quiesce_tasks() already cancelled - asyncio.wait()
    # does not cancel its own members when the wait around it is cancelled,
    # so they keep running to their own completion independently; this just
    # stops them from blocking shutdown past the platform's own patience.
    async def _drain_in_flight_work():
        await cancellation_manager.quiesce_tasks()
        # Best-effort Brain outcome ingests are tracked tasks. Await them
        # before the loop tears down, otherwise a graceful shutdown can
        # close the loop mid-flight and lose the outcome despite scheduling
        # it.
        await flush_pending_ingest_tasks()

    # Anchored once, before the first attempt below even starts, so every
    # subsequent wait in this sequence is bounded against the *same* single
    # deadline rather than each adding its own independently-chosen
    # timeout on top of the last - see the constants' own comments above
    # for why that composition broke silently before.
    shutdown_deadline = (
        asyncio.get_running_loop().time()
        + ASSUMED_PLATFORM_GRACE_PERIOD_S
        - SHUTDOWN_FINAL_STEPS_MARGIN_S
    )

    try:
        await asyncio.wait_for(_drain_in_flight_work(), timeout=SHUTDOWN_DRAIN_BUDGET_S)
    except asyncio.TimeoutError:
        logger.warning(
            "Shutdown drain did not finish within %.1fs; giving any still-running task one bounded extra chance to settle.",
            SHUTDOWN_DRAIN_BUDGET_S,
        )
        # quiesce_tasks() already cancelled every task it found once, but
        # asyncio.wait() does not cancel its own members just because the
        # wait around it was cancelled - a task whose own cleanup is still
        # running when the budget above expires has therefore only ever
        # been cancelled that once, and merely re-cancelling it without
        # also awaiting it would return from this function while that
        # cleanup might still be genuinely, briefly in flight (its own
        # per-step deadline can still have a little real time left even
        # after this outer ceiling gives up). Calling quiesce_tasks() again
        # - it re-cancels and awaits in one step, which is exactly what is
        # needed here too - bounded to whatever time genuinely remains
        # until shutdown_deadline (not a fresh, independently-sized
        # allowance) is what keeps the total bounded against one single
        # number no matter how long this first attempt actually ran: see
        # quiesce_tasks()'s own comment (app/cancellation.py) for why a
        # task's remaining cleanup from any point it is currently in can
        # never exceed that function's own worst-case total, so a shorter
        # remaining budget here is still making real progress, not just
        # repeating a no-op. flush_pending_ingest_tasks() does not get a
        # second bounded pass here: its own ingest is the one thing each
        # task's cleanup above already treats as best-effort and optional,
        # unlike the owner lease and concurrency slot quiesce_tasks()
        # itself protects, so losing it only in this already-degraded
        # double-timeout case is the smaller, already-accepted loss.
        remaining = shutdown_deadline - asyncio.get_running_loop().time()
        if remaining > 0:
            await cancellation_manager.quiesce_tasks(timeout=remaining)
        else:
            logger.warning(
                "No time left in the shutdown budget for a second quiesce attempt; "
                "cancelling any still-running task once more without waiting further."
            )
            for leftover in list(cancellation_manager.active_tasks.values()):
                if not leftover.done():
                    leftover.cancel()
    await cancellation_manager.shutdown()
    logger.info("Shutting down storage backend connections...")

app = FastAPI(
    title="Web Intelligence Sidecar Service",
    description="Python FastAPI sidecar wrapping GPT Researcher for AI Commander",
    version="1.0.0",
    lifespan=lifespan
)

if settings.CORS_ORIGINS:
    origins = [o.strip() for o in settings.CORS_ORIGINS.split(",") if o.strip()]
    logger.info(f"Applying CORS configurations for origins: {origins}")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

@app.middleware("http")
async def verify_auth_token(request: Request, call_next):
    if request.url.path in PUBLIC_PATHS:
        return await call_next(request)
    if request.url.path in DOCS_PATHS and unauthenticated_docs_allowed():
        return await call_next(request)

    token = settings.AUTH_TOKEN
    if not auth_is_configured():
        emit_observability("request_rejected", route="unmatched", reason="auth_not_configured")
        logger.error("Authentication token is not configured; refusing protected request.")
        return JSONResponse(status_code=503, content={"detail": "Service authentication is not configured"})

    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        emit_observability("request_rejected", route="unmatched", reason="missing_bearer")
        return JSONResponse(status_code=401, content={"detail": "Unauthorized: Missing authentication bearer token"})

    provided_token = auth_header.removeprefix("Bearer ").strip()
    if not secrets.compare_digest(provided_token, token):
        emit_observability("request_rejected", route="unmatched", reason="invalid_bearer")
        return JSONResponse(status_code=401, content={"detail": "Unauthorized: Invalid authentication credentials"})

    return await call_next(request)

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

app.include_router(router)
# Install last so it wraps auth middleware, including short-circuited responses.
# Exceptions are recorded and re-raised; existing error handling is unchanged.
app.add_middleware(ObserveASGI, emitter=emitter)
