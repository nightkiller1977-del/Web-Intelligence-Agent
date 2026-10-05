---
name: using-aicc-ecosystem
description: Use when work in this repository touches another AI Commander repository, shared secrets, SOPS, age, Render, Tailscale, MongoDB, PostgreSQL, SQLite, Ollama, OpenHands, Grafana Alloy, Jira, GitHub, MCP, systemd, launchd, or cross-service contracts.
---

# Use the AI Commander ecosystem correctly

This repository is `Web-Intelligence-Agent`: bounded browser/research service that collects source-backed web evidence for authorized AI Commander workflows.

Before changing a cross-service integration, read the ecosystem map in the repository root `AGENTS.md`. Then:

1. Identify the owning repository and verify its current contract/source/tests.
2. Reuse its API, scheduler, policy, approval, secret, memory, routing, or observability boundary; do not create a parallel owner here.
3. Resolve secrets through `aicc-secrets`. Never expose plaintext, age private keys, decrypted environment output, or reuse another service's scoped credential.
4. Treat `render-sync-map.json` as a least-privilege distribution allowlist. Update producer, consumer, mapping, and documentation together.
5. Treat Tailscale as transport, not authorization. Current verified uses are Job Agent noVNC re-auth and supported Code Review Agent Ollama tunneling; the broader Coordinator/worker mesh remains planned until accepted.
6. Preserve local-first, fail-closed behavior and reconcile ambiguous side effects before retry.
7. Validate completion with current tests/configuration/runtime evidence; model output, reachability, or a stored secret is not proof.

## AI Commander-first delegation

Use existing Commander tools/workflows and the owning specialized agent before spending general-purpose model context on deterministic work. Coordinator is authoritative for current component routing.

ConnectionSphere and TrustGraph are first-class ecosystem components. TrustGraph owns a PostgreSQL data contract; ConnectionSphere hosting/persistence must be verified from its own repository and must not be inferred as Azure-hosted.

For executable workflow changes, unit/mock and wiring checks alone are not completion proof. Exercise the real production-equivalent behavior whenever safe and record the evidence in Jira.
