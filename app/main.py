# app/main.py
import logging
import secrets
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app.config import auth_is_configured, settings
from app.storage import storage
from app.cancellation import cancellation_manager
from app.api import router

# Set up logging format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("web-intelligence")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing storage backend lifecycle...")
    await storage.init()

    # Mark any operations left in queued/running state from a prior crash as failed
    if settings.STORAGE_BACKEND != "redis" or getattr(storage, "degraded", False):
        await storage.mark_stale_operations()
    else:
        logger.info("Skipping global stale-operation scan for shared Redis storage.")

    # Wire up cross-instance cancel pub/sub if Redis is available
    redis_client = getattr(storage, "redis", None)
    if redis_client and not getattr(storage, "degraded", False):
        await cancellation_manager.init(redis_client)

    yield

    await cancellation_manager.shutdown()
    logger.info("Shutting down storage backend connections...")

app = FastAPI(
    title="Web Intelligence Sidecar Service",
    description="Python FastAPI sidecar wrapping GPT Researcher for AI Commander",
    version="1.0.0",
    lifespan=lifespan
)

# CORS Policy configuration (disabled by default in remote mode)
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

# Bearer Token Auth Middleware (Gating all operations routes)
@app.middleware("http")
async def verify_auth_token(request: Request, call_next):
    # Exempt health checks and capabilities discovery from authorization checking
    if request.url.path in ("/health/live", "/health/ready", "/capabilities", "/version", "/metrics"):
        return await call_next(request)

    token = settings.AUTH_TOKEN
    if not auth_is_configured():
        logger.error("Authentication token is not configured; refusing protected request.")
        return JSONResponse(
            status_code=503,
            content={"detail": "Service authentication is not configured"}
        )

    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        return JSONResponse(
            status_code=401,
            content={"detail": "Unauthorized: Missing authentication bearer token"}
        )

    provided_token = auth_header.removeprefix("Bearer ").strip()
    if not secrets.compare_digest(provided_token, token):
        return JSONResponse(
            status_code=401,
            content={"detail": "Unauthorized: Invalid authentication credentials"}
        )

    return await call_next(request)

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

app.include_router(router)
