# Web Intelligence Agent

Web Intelligence Agent is a self-hostable FastAPI service for running **bounded, structured web research as an API job**.

Instead of giving an application an unrestricted browser process, callers submit a research request with explicit limits. The service manages the research lifecycle, applies authentication and network-safety controls, streams progress, supports cancellation, and returns a structured result containing sources, evidence, claims, citations, and execution metrics.

The current research engine is built around GPT Researcher, with additional service-level controls for security, idempotency, concurrency, storage, and observability.

## Why use it?

Web research is often long-running, network-heavy, and exposed to untrusted content. Putting that work behind a dedicated API boundary makes it easier to:

- isolate research from your main application process;
- set limits on searches, pages, sources, duration, memory, model usage, and cost;
- prevent accidental duplicate jobs with idempotency keys;
- stream progress without holding one request open for the entire research run;
- cancel work cleanly;
- preserve source-level citations and evidence;
- apply SSRF/network restrictions before research reaches protected destinations;
- deploy the same service locally or as a remote worker.

## Project status

**Active development.** The core API, job lifecycle, local/Redis storage modes, authentication, research limits, progress streaming, cancellation, structured evidence, metrics, and network-safety controls are implemented.

The project continues to harden restart behavior, resource limits, untrusted-content handling, deployment behavior, and source/evidence quality. A successful historical research run does not imply every website or query will always behave the same way.

## Key capabilities

- **Asynchronous research jobs** — submit work and retrieve results separately.
- **Three research modes** — `quick`, `standard`, and `deep`.
- **Research profiles** — `general`, `technical`, `repair`, `code-review`, and `security`.
- **Structured output** — answer, sources, evidence passages, claims, citations, limitations, warnings, and metrics.
- **Claim verification** — optional evidence-backed claim verification records.
- **Source policies** — optionally constrain research to approved domains.
- **Freshness controls** — pass time-window/freshness preferences with a request.
- **Idempotency** — duplicate requests can resolve to the same existing operation instead of starting duplicate work.
- **Server-Sent Events** — stream progress from running operations.
- **Cancellation** — request cancellation of an active operation.
- **Concurrency and resource limits** — bound active work and per-request research budgets.
- **Local or Redis-backed storage** — local mode for development; Redis for durable remote coordination.
- **Bearer-token authentication** — protected API routes fail closed when remote authentication is not configured.
- **SSRF-oriented protections** — explicit URL validation and restrictions around private/protected network destinations.
- **Prometheus metrics** — `/metrics` is available for operational monitoring.
- **Container deployment** — Docker and Render configuration are included.

## Architecture

```text
Client / application
        │
        │ POST /v1/research
        ▼
Web Intelligence Agent
  ├─ authenticate request
  ├─ validate query + source policy
  ├─ reserve idempotency key
  ├─ enforce concurrency/resource limits
  ├─ persist operation state
  ├─ run bounded research
  ├─ stream progress events
  └─ store structured result
        │
        ▼
Authorized public web resources
        │
        ▼
Sources + evidence + claims + citations + metrics
```

The service treats retrieved web pages as **untrusted data**, not as instructions that can override application or security policy.

## Quick start

### 1. Create a Python environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.lock
```

### 2. Create local configuration

```bash
cp .env.example .env
```

At minimum, configure an explicit API token for manual local use and the provider/search credentials required by your selected research engine:

```env
DEPLOYMENT_MODE=local
WEB_INTELLIGENCE_AUTH_TOKEN=replace-with-a-long-random-token
OPENAI_API_KEY=...
TAVILY_API_KEY=...
STORAGE_BACKEND=local
```

Do not commit your `.env` file.

### 3. Start the API

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8080
```

Useful public endpoints:

```text
GET /health/live
GET /health/ready
GET /version
GET /capabilities
GET /metrics
```

Protected research routes require:

```text
Authorization: Bearer <WEB_INTELLIGENCE_AUTH_TOKEN>
```

## Submit a research job

A research request requires an operation ID, attempt ID, an idempotency key, a query, and explicit limits.

```bash
curl -X POST http://127.0.0.1:8080/v1/research \
  -H "Authorization: Bearer $WEB_INTELLIGENCE_AUTH_TOKEN" \
  -H "Content-Type: application/json" \
  -H "Idempotency-Key: example-research-001" \
  -d '{
    "operationId": "research-001",
    "attemptId": "7d0a7564-c0ce-4f24-8827-70d679e84c53",
    "query": "What are the major approaches to local-first AI memory systems?",
    "mode": "standard",
    "profile": "technical",
    "requireCitations": true,
    "requireClaimVerification": true,
    "limits": {
      "maximumDurationSeconds": 300,
      "maximumSearches": 10,
      "maximumPages": 20,
      "maximumSources": 10,
      "maximumMemoryMb": 512
    }
  }'
```

The initial response is asynchronous:

```json
{
  "operationId": "research-001",
  "status": "queued"
}
```

## Follow progress

Progress is available as Server-Sent Events:

```bash
curl -N \
  -H "Authorization: Bearer $WEB_INTELLIGENCE_AUTH_TOKEN" \
  http://127.0.0.1:8080/v1/research/research-001/events
```

Events report stages such as planning, searching, reading, analyzing, verifying, synthesizing, completed, or failed.

## Retrieve the result

```bash
curl \
  -H "Authorization: Bearer $WEB_INTELLIGENCE_AUTH_TOKEN" \
  http://127.0.0.1:8080/v1/research/research-001/result
```

A completed result can include:

- the synthesized answer;
- source metadata;
- evidence passages;
- claims linked to evidence;
- citations linking claims to sources;
- searches performed;
- execution metrics;
- degraded-mode reasons, warnings, or limitations.

## Cancel a job

```bash
curl -X POST \
  -H "Authorization: Bearer $WEB_INTELLIGENCE_AUTH_TOKEN" \
  http://127.0.0.1:8080/v1/research/research-001/cancel
```

## Research controls

### Modes

- `quick` — smaller/faster research pass.
- `standard` — normal research depth.
- `deep` — broader research within the limits you provide.

### Profiles

Supported profiles currently include:

- `general`
- `technical`
- `repair`
- `code-review`
- `security`

Profiles let callers express the type of research being performed while keeping the same API contract.

### Source policy

The current source policy supports `allowedDomains`. This can be used to constrain a request to a known set of domains rather than allowing unrestricted source selection.

### Per-request limits

Requests can bound:

- duration;
- search count;
- pages read;
- sources returned;
- memory;
- estimated model calls;
- estimated model tokens;
- estimated model cost.

These limits are part of the API contract, not just documentation recommendations.

## Local vs. remote deployment

### Local mode

Local mode defaults to local storage and is intended for development or a trusted local controller. Raw provider credential headers are only permitted in local deployment mode.

### Remote mode

Remote deployments should set an explicit `WEB_INTELLIGENCE_AUTH_TOKEN` and normally use Redis-backed storage. If authentication is not configured in remote mode, protected requests fail closed.

The included `render.yaml` configures a Docker-based service plus Redis and illustrates the remote deployment shape.

## Docker

Build and run the service using the included Dockerfile:

```bash
docker build -t web-intelligence-agent .
docker run --rm -p 8080:8080 --env-file .env web-intelligence-agent
```

The container runs:

```text
uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

## Security model

Web research processes untrusted internet content, so several boundaries are deliberately conservative:

- private, loopback, link-local, metadata, and protected internal destinations are restricted by network/URL policy;
- retrieved page content is data, not trusted instructions;
- provider credentials must not be written to logs or committed to Git;
- remote API access requires bearer authentication;
- raw provider credential headers are rejected outside local deployment mode;
- research work is concurrency- and resource-bounded;
- failures should remain explicit instead of being converted into a successful result.

See [`SECURITY.md`](SECURITY.md) for the repository's security guidance.

## Observability

The service exposes:

- `/health/live` — process liveness;
- `/health/ready` — storage/auth/research-engine readiness;
- `/metrics` — Prometheus metrics;
- structured job status and result data.

Remote observability is intended for operational metadata. Prompts, credentials, private content, and unrestricted raw browsing data should not be exported as telemetry.

## Testing

Run the repository's automated tests before changing request validation, SSRF/network controls, job state, idempotency, streaming, cancellation, resource limits, or authentication behavior.

```bash
pytest
```

`REMOTE_MODE_TEST_REPORT.md` contains additional point-in-time remote-mode validation information. Treat historical test reports as evidence for that run, not as a permanent guarantee of current deployment health.

## Relationship to AI Commander

Web Intelligence Agent was developed as the dedicated research service for the broader **AI Commander** ecosystem. AI Commander can use it to offload long-running web research from the desktop process and combine the resulting evidence with other agents and memory workflows.

The HTTP service itself is intentionally understandable and deployable on its own. You do not need the AI Commander desktop application to inspect the API, run the service, or integrate it with another authorized client.

## Contributing

Useful contributions include:

- research-result quality and provenance;
- additional safe source controls;
- SSRF/network hardening;
- restart/durability behavior;
- cancellation and progress handling;
- resource accounting;
- deployment documentation;
- controlled integration tests.

Please preserve the fail-closed authentication and network-security behavior when extending the service.