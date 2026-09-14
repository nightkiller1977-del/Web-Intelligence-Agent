---
name: aicc-architecture-change
description: Use for cross-service, cross-agent, routing, contract, storage, or other architectural changes in AI Commander.
---
# Architecture Change
1. Start from the existing architecture and exact seam that must change.
2. Extend existing contracts/services before adding parallel systems.
3. Minimize new coordination, persistence, network, and operational dependencies.
4. Define ownership, trust boundaries, failure/retry/idempotency, and compatibility first.
5. Add abstraction only for a required boundary or real duplication.