# Web Intelligence Agent (Python Sidecar)

The Web Intelligence Agent is a lightweight, production-ready Python FastAPI sidecar that wraps the open-source **GPT Researcher** engine. It exposes structured, multi-step web research capabilities to the AI Commander desktop control plane.

This service is designed to run in two modes:
1. **Local Mode**: Spawned and managed locally by the Electron control plane on a dynamic loopback port with ephemeral token authentication.
2. **Remote Mode**: Deployed as an auto-scaling, Docker-based container cluster on Render.com, backed by a Valkey/Redis instance for progress queues and distributed locking.

---

## 1. Features
- **Iterative Research Orchestration**: Plans and executes multi-query web searches, scrapes pages, and synthesizes answers using GPT Researcher.
- **SSRF Hardened Egress**: Implements DNS pre-resolution checks, redirects auditing, and loops validation to block access to private IPv4/IPv6 subnets, carrier-grade NAT, and cloud metadata endpoints.
- **Isolate Request Keys**: Applies OpenAI and Tavily API credentials in request scopes only, preventing race conditions or process-wide leaks.
- **Process Memory Spilling**: Monitored via a background daemon. If memory footprint exceeds 80%, fetches pause, data context spills to disk, and the sidecar synthesizes a partial response.
- **Durable Event Streaming**: Emits live progress logs (`planning`, `searching`, `reading`, `synthesizing`, etc.) via Server-Sent Events (SSE) backed by Redis Streams in cluster mode.
- **Task Cancellations**: Provides REST-based endpoints to abort running async loops.
- **Prometheus Telemetry**: Counts spent token metrics (`research_cost_tokens_total`) and alerts if daily budget spend exceeds $50.

---

## 2. Service Architecture

```mermaid
graph TD
    Client[AI Commander Electron Control Plane]
    API[FastAPI Gateway /app/api.py]
    Auth[Bearer Auth Middleware]
    Manager[Cancellation Manager]
    Storage[Storage Adapter: Memory / Redis]
    Adapter[GPT Researcher Adapter]
    Engine[GPT Researcher Engine]
    SSRF[SSRF Validator /app/security.py]
    LLM[Model/Provider API]

    Client -->|POST /v1/research| API
    Client -->|GET /events| API
    API -->|1. Authenticate| Auth
    API -->|2. Register/Poll| Storage
    API -->|3. Spawn Task| Adapter
    Adapter -->|Request-Scoped Keys| Engine
    Adapter -->|Monitor Task| Manager
    Engine -->|Scrape Pages| SSRF
    SSRF -->|HTTP Get| Web((Public Web))
    Engine -->|LLM Synthesis| LLM
```

---

## 3. REST API Specification

Detailed OpenAPI documentation is available under `api/openapi.yaml`.

### 3.1 Health & Verification
- **`GET /health/live`**: Fast liveness probe checking process response (used by Render).
- **`GET /health/ready`**: Probes storage backends (Redis) and engine dependencies.
- **`GET /capabilities`**: Lists active features supported by the sidecar.
- **`GET /version`**: Returns service and gpt-researcher version strings.

### 3.2 Operations Workflow
- **`POST /v1/research`**: Enqueues a new query. Returns `202 Accepted` with the `operationId`. Supports idempotency key checks via `Idempotency-Key` header.
- **`GET /v1/research/{operationId}/events`**: Server-Sent Events (SSE) stream of task progression logs.
- **`GET /v1/research/{operationId}/result`**: Fetches the completed synthesis report, citations, claims, and evidence passages.
- **`POST /v1/research/{operationId}/cancel`**: Interrupts the active research task.

---

## 4. Configuration & Environment Variables

| Variable | Default | Description |
| :--- | :--- | :--- |
| `DEPLOYMENT_MODE` | `local` | Service deployment mode: `local` (Electron sidecar) or `remote` (Render). |
| `WEB_INTELLIGENCE_AUTH_TOKEN` | `""` | Bearer token verified on incoming API requests (auto-generated in local mode). |
| `STORAGE_BACKEND` | `local` | Persistence mode: `local` (in-memory dicts) or `redis` (Render Valkey cache). |
| `REDIS_URL` | `""` | Connection URL for Redis instances (e.g. `redis://red-xxx:6379`). |
| `MAX_CONCURRENT_OPS` | `3` | Maximum number of concurrent research jobs allowed in the process. |
| `MAX_MEMORY_MB` | `512` | Memory threshold limits (MB) prior to triggering early partial synthesis. |
| `DAILY_SPEND_LIMIT_USD` | `50.0` | Alert threshold for token costs. |
| `CORS_ORIGINS` | `""` | Comma-separated list of approved CORS domain origins. |

---

## 5. Local Setup & Installation

### 5.1 Prerequisites
- Python 3.11+
- virtualenv

### 5.2 Commands
```bash
# 1. Navigate to the agent directory
cd WEB_INTELLIGENCE_AGENT_PATH

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install pinned dependencies
pip install -r requirements.lock

# 4. Spin up the FastAPI server locally
uvicorn app.main:app --host 127.0.0.1 --port 8080
```

---

## 6. Remote Deployment (Render.com)

Render deployments use the Blueprint template (`render.yaml`) and build directly from the `Dockerfile`.

1. Go to your **Render Dashboard** > **New** > **Blueprint**.
2. Select your `Web-Intelligence-Agent` Git repository.
3. Render will provision:
   - The FastAPI web service linked to `Dockerfile` with liveness checks at `/health/live`.
   - The Valkey database instance used for cluster queues.
4. Set your provider API keys (`OPENAI_API_KEY`, `TAVILY_API_KEY`) as **Environment Variables** in the Render service dashboard. These are injected directly into the container's `os.environ` at startup.

> **Note on secrets flow:** In local mode the sidecar inherits API keys from the Electron control plane's `process.env`, which loads them from AI Commander's centralized SOPS-encrypted secrets store (`secrets.enc.env`). In remote mode, Render's native environment variable injection replaces that mechanism. Do not introduce a separate secrets file loader — credentials always flow through environment variables regardless of deployment mode.
