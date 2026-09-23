# Claude Code instructions

@AGENTS.md

`AGENTS.md` is the shared engineering source of truth for this repository.

- Verify browser/source behavior against current code/tests and actual source evidence before making claims.
- Use the matching skill under `.claude/skills/` when relevant.
- Keep scope narrow and reuse existing browser/research abstractions.
- Ask before irreversible/shared actions unless explicitly authorized in the current task: merge, force-push/history rewrite, production deploy/config change, secret-store write, real external submission/message/purchase, destructive account/data change, or permission/budget change.
- Branch commits, tests, and opening a PR do not require extra confirmation.
- Never bypass tests, hooks, approval checks, or security controls merely to appear complete.
- Start delegated fix sessions (review-finding fixes, CI fixes, merging the base branch in) on Sonnet. Use Opus only for initial design/implementation or when the user asks for it.

## Ecosystem and tool boundary

This repository is `Web-Intelligence-Agent`: bounded browser/research service that collects source-backed web evidence for authorized AI Commander workflows. Before cross-repository work, use the complete repository/ownership and infrastructure map in `AGENTS.md`.

- Resolve managed credentials only through the authorized `aicc-secrets` SOPS+age flow; never reveal plaintext or reuse another service's scoped MongoDB credential.
- Treat `render-sync-map.json` as the Render distribution allowlist, not as a catalog granting every service access.
- Treat Tailscale as private transport, not authorization. Current operational uses are Job Agent noVNC re-auth and supported Code Review Agent Ollama tunnels; the wider Coordinator/worker mesh is not complete until verified.
- Reuse Desktop/Coordinator/OpenRouter/Brain Memory/Model Intelligence/metrics authorities rather than recreating them in this repo.
- Verify implemented versus planned behavior from current source, tests, deployment configuration, and runtime evidence.
