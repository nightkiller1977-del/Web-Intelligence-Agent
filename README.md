# Web Intelligence Agent

Web-Intelligence-Agent is AI Commander's **structured web-research execution service**. It provides a network-reachable FastAPI boundary for creating, running, streaming, cancelling, and retrieving research jobs while isolating browser/research workloads from the desktop application's main process.

The service wraps research capabilities behind AI Commander-specific security, resource, job-state, and observability controls.

## Current state

**State: Implemented supporting service integrated with AI Commander and under operational/security hardening.**

`main` includes:

- Python/FastAPI API routes for structured research jobs.
- Local and remote execution modes.
- Job creation, status, result, streaming/progress, and cancellation behavior.
- Redis-backed coordination/stream behavior where configured.
- Request-scoped credential handling.
- URL/network safety checks, including SSRF-oriented restrictions.
- Memory/resource guards for expensive research workloads.
- Health and metrics surfaces.
- Docker and Render deployment definitions.
- Automated tests and remote-mode validation artifacts.
- Metadata-only observability with sanitization and bounded export behavior.
- Shared OpenHands and Claude repository skills for consistent security, reuse, simplicity, and validation rules.

This service is **not** an unrestricted autonomous browser. It is a bounded research sidecar: authorized callers submit a research task, the service applies security/resource policy, and callers receive structured progress and results.

## Direction

1. **Keep research isolated from the desktop core.** Browser/network-heavy work should remain behind a clear job boundary rather than expanding the desktop process's attack/resource surface.
2. **Strengthen untrusted-content handling.** Retrieved pages are data, not instructions. Prompt injection or page content must not override tool/network/security policy.
3. **Keep SSRF/network controls fail closed.** Private, loopback, link-local, metadata, and protected internal destinations should remain blocked unless explicitly required by trusted configuration.
4. **Make long-running jobs restart-safe and observable.** Job state, cancellation, progress, and failure evidence should be explicit enough for AI Commander to understand whether work is running, failed, cancelled, or complete.
5. **Bound memory/concurrency.** One research request must not exhaust the service/host or starve unrelated fleet work.
6. **Return structured, source-aware results.** Research output should preserve enough source/provenance metadata for downstream validation and future Brain Memory ingestion.
7. **Integrate with Brain Memory through controlled artifacts.** Useful research results can become durable knowledge under policy; raw browsing state/private content should not be copied indiscriminately.
8. **Feed operational failures into the correct recovery owner.** Network restrictions, provider/search failures, resource exhaustion, service infrastructure, and code defects should remain distinguishable.
9. **Keep credentials scoped.** Search/provider credentials should remain request-scoped or come from the authorized secret store, never from committed plaintext.
10. **Reuse shared AI Commander contracts.** Health, metrics, auth, incident, and workflow integration should extend existing fleet patterns rather than create parallel control planes.

## System role

```text
AI Commander / authorized agent
            │
            ▼
 Web Intelligence Agent
   ├─ validate research request
   ├─ enforce URL/network policy
   ├─ manage durable job state
   ├─ execute bounded research
   ├─ stream progress / cancel
   ├─ enforce resource limits
   └─ return structured result + source evidence
            │
            ▼
 authorized public web resources
```

## Security boundaries

Web research processes untrusted external content. Important controls include:

- reject or constrain private/loopback/link-local/internal network destinations unless trusted configuration explicitly requires them;
- avoid following user-controlled URLs into protected infrastructure;
- treat retrieved web content as untrusted data rather than system/tool instructions;
- keep provider/search credentials request-scoped or in the authorized runtime secret store;
- keep prompts, secrets, private user content, and unrestricted raw URLs out of remote telemetry;
- bound job memory, concurrency, runtime, and cancellation behavior;
- fail clearly rather than reporting success when a job was blocked or incomplete.

See `SECURITY.md` for deeper security guidance.

## Local development

Use the repository's pinned/locked requirements and current application entry point as the source of truth. Run the automated tests before changing request validation, SSRF controls, job state, streaming, cancellation, or resource-limit behavior.

## Deployment

- `Dockerfile` defines the service container.
- `render.yaml` defines the hosted deployment shape.
- `REMOTE_MODE_TEST_REPORT.md` contains point-in-time validation evidence.

Historical validation or a deployment blueprint is not proof of current runtime health. Use live health, fleet status, metrics/logs, and current deployment evidence.

## Relationship to AI Commander

- `Ai-Command-Center-Desktop-App` uses this service for bounded dedicated research jobs.
- `Aicc-Coordinator` can surface service health and wider operational state.
- `AI-Commander-Brain-Memory-` is the direction for policy-approved durable research knowledge with provenance.
- `aicc-secrets` provides managed credentials where centrally configured.

## Documentation rule

This README describes the repository's role, capabilities on `main`, and current direction. Tests, live service health, deployment evidence, and actual integration behavior remain the source of truth. Avoid dated status snapshots and static “production ready” claims.