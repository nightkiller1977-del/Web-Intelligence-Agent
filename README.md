# Web Intelligence Agent

Web-Intelligence-Agent is AI Commander's **structured web-research execution service**. It provides a network-reachable FastAPI boundary for creating, running, streaming, cancelling, and retrieving research jobs while isolating browser/research workloads from the desktop application's main process.

The service wraps research capabilities such as GPT Researcher behind AI Commander-specific security, resource, job-state, and observability controls.

## Current status — September 9, 2026

**State: Implemented supporting service / integrated with AI Commander and under ongoing operational hardening.**

`main` currently includes:

- Python/FastAPI service and API routes for research jobs.
- Local and remote execution modes.
- Structured job creation/status/result workflows.
- Streaming/event-oriented progress support and job cancellation.
- Redis-backed coordination/stream behavior where configured.
- Request-scoped credential handling.
- URL/network safety checks, including SSRF-oriented restrictions.
- Memory/resource guards for expensive research tasks.
- Metrics/health surfaces.
- Docker and Render deployment definitions.
- Automated tests and a remote-mode test report.
- Metadata-only Grafana/Loki observability with sanitization and bounded remote-export behavior.
- Live Render integration work in the AI Commander environment.

The service should **not** be described as an unrestricted autonomous browser. It is a bounded research sidecar: callers submit a research task, the service applies its security/resource policies, and the caller receives structured progress/results.

## System role

```text
AI Commander / authorized agent
            │
            ▼
 Web Intelligence Agent
   ├─ validate research request
   ├─ enforce URL/network policy
   ├─ manage job lifecycle
   ├─ execute research workload
   ├─ stream progress / support cancel
   ├─ enforce memory/resource guards
   └─ return structured result
            │
            ▼
 authorized public web resources
```

Keeping research in a dedicated service reduces the amount of browser/network complexity inside the desktop process and gives AI Commander a consistent job contract for long-running research.

## Security boundaries

Web research processes untrusted external content. Important controls include:

- reject or constrain private/loopback/link-local/internal network destinations unless explicitly required by a trusted configuration;
- avoid following user-controlled URLs into protected infrastructure;
- keep provider/search credentials request-scoped or in the authorized runtime secret store;
- do not expose secrets, prompts, private user content, or unrestricted raw URLs through remote observability;
- bound job memory, concurrency, and cancellation behavior so one research request cannot exhaust the host/service;
- treat retrieved web content as untrusted data, not instructions that override system/tool policy.

See `SECURITY.md` for deeper security guidance.

## Local development

Use the pinned/locked requirements in the repository and the current application entry point as the source of truth. A typical setup starts by creating a Python environment and installing the required packages from the repository's requirements files.

Run the automated tests before changing request validation, SSRF controls, job state, streaming, or cancellation behavior.

## Deployment

- `Dockerfile` defines the service container.
- `render.yaml` defines the Render deployment shape.
- `REMOTE_MODE_TEST_REPORT.md` contains point-in-time remote-mode validation evidence.

A successful historical test report or deployment blueprint does not guarantee current runtime health. Use live health, fleet status, logs, and current deployment evidence for operational decisions.

## Relationship to AI Commander

`Ai-Command-Center-Desktop-App` uses this service when web research is better isolated as a dedicated job rather than executed directly in the local chat loop. `Aicc-Coordinator` can surface service health as part of the wider fleet view.

## Documentation rule

This README describes the repository's current role and capabilities on `main`. Automated tests, live service health, deployment evidence, and current AI Commander integration are the sources of truth for operational status. Avoid static “production ready” claims that are not continuously verified.
