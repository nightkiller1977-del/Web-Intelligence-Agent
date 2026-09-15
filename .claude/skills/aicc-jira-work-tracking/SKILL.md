---
name: aicc-jira-work-tracking
description: Use whenever starting, updating, reviewing, completing, or discovering implementation work that must be tracked in Jira.
---
# AI Commander Jira Work Tracking

## Instance
- Site: `https://shoalinwu.atlassian.net`
- Active project: `ACES`
- Browse pattern: `https://shoalinwu.atlassian.net/browse/<KEY>`
- `KAN` is legacy/historical unless explicitly requested.
- Component/repository routing is Coordinator-authoritative; do not invent mappings.
- Never put Jira email, API token, Cloud ID, enrollment secret, or credentials in guidance, prompts, logs, tests, or PR text.

Current documented mappings: AI Command Center → `nightkiller1977-del/Ai-Command-Center-Desktop-App`; AICC Coordinator/Routing → `nightkiller1977-del/Aicc-Coordinator`; Job Agent → `nightkiller1977-del/job-agent`; Email Agent → `nightkiller1977-del/email-agent`; ConnectionSphere → `nightkiller1977-del/connectionsphere`; TrustGraph → `nightkiller1977-del/TrustGraph`; Code Review Agent → `nightkiller1977-del/Code-Review-Agent`; AI OpenRouter → `nightkiller1977-del/AI-OpenRouter`. If absent, consult current Coordinator data instead of guessing.

## Before coding
1. Identify the Jira work ticket.
2. Verify it has a parent Epic/roadmap ticket; identify/create the parent first if orphaned.
3. Read parent and child; parent outcome/direction/constraints govern the child.
4. Confirm repository/component scope, dependencies, acceptance criteria, and authorization.
5. Move/update the ticket to the appropriate working state.

## During work
- Keep changes tied to ticket and parent; record meaningful findings/blockers/decisions and validation evidence.
- Create a separate linked child ticket for discovered out-of-scope work.
- Reconcile Jira before changing direction when code reality conflicts with the ticket.

## PR and completion
- PR names work ticket + parent; Jira references the PR.
- Set `PR Open` when opened; record evidence and move through `In Verification`.
- Failed validation → `Verification Failed`; resumed work → `AI Commander Working`.
- Do not mark `Done` until required tests/delivery/reconciliation evidence and ticket/parent criteria are satisfied.