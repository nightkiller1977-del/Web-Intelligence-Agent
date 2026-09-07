# app/main.py
import logging
import secrets
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app.config import auth_is_configured, settings, unauthenticated_docs_allowed
from app.storage import storage
from app.cancellation import cancellation_manager
from app.api import router
from app.grafana_observability import emit as emit_observability

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
    await storage.init()
    await storage.mark_stale_operations()
    redis_client = getattr(storage, "redis", None)
    if redis_client and not getattr(storage, "degraded", False):
        await cancellation_manager.init(redis_client)
    emit_observability(
        "service_started",
        storage_degraded=bool(getattr(storage, "degraded", False)),
        redis_enabled=bool(redis_client),
    )
    yield
    await cancellation_manager.shutdown()
    emit_observability("service_stopped")
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
async def observe_request(request: Request, call_next):
    started = time.perf_counter()
    response = await call_next(request)
    emit_observability(
        "http_request",
        method=request.method,
        route=request.url.path,
        status=response.status_code,
        duration_ms=round((time.perf_counter() - started) * 1000, 1),
    )
    return response

@app.middleware("http")
async def verify_auth_token(request: Request, call_next):
    if request.url.path in PUBLIC_PATHS:
        return await call_next(request)
    if request.url.path in DOCS_PATHS and unauthenticated_docs_allowed():
        return await call_next(request)

    token = settings.AUTH_TOKEN
    if not auth_is_configured():
        emit_observability("request_rejected", route=request.url.path, reason="auth_not_configured")
        logger.error("Authentication token is not configured; refusing protected request.")
        return JSONResponse(status_code=503, content={"detail": "Service authentication is not configured"})

    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        emit_observability("request_rejected", route=request.url.path, reason="missing_bearer")
        return JSONResponse(status_code=401, content={"detail": "Unauthorized: Missing authentication bearer token"})

    provided_token = auth_header.removeprefix("Bearer ").strip()
    if not secrets.compare_digest(provided_token, token):
        emit_observability("request_rejected", route=request.url.path, reason="invalid_bearer")
        return JSONResponse(status_code=401, content={"detail": "Unauthorized: Invalid authentication credentials"})

    return await call_next(request)

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

app.include_router(router)
