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
    await cancellation_manager.quiesce_tasks()
    # Best-effort Brain outcome ingests are tracked tasks. Await them before
    # the loop tears down, otherwise a graceful shutdown can close the loop
    # mid-flight and lose the outcome despite scheduling it.
    await flush_pending_ingest_tasks()
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
