# Claude Code instructions

@AGENTS.md

`AGENTS.md` is the shared engineering source of truth for this repository.

- Verify browser/source behavior against current code/tests and actual source evidence before making claims.
- Use the matching skill under `.claude/skills/` when relevant.
- Keep scope narrow and reuse existing browser/research abstractions.
- Ask before irreversible/shared actions unless explicitly authorized in the current task: merge, force-push/history rewrite, production deploy/config change, secret-store write, real external submission/message/purchase, destructive account/data change, or permission/budget change.
- Branch commits, tests, and opening a PR do not require extra confirmation.
- Never bypass tests, hooks, approval checks, or security controls merely to appear complete.