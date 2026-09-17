---
name: aicc-pr-completion
description: Use when preparing a PR, finishing implementation, checking merge readiness, or producing completion evidence.
---
# PR Completion
1. Re-read Jira work ticket and parent; confirm parent direction and scope are followed.
2. Verify parent linkage and that the PR names both ticket and parent.
3. No unrelated work; discovered extras require separate Jira child tickets.
4. Follow existing patterns and avoid duplicate capability.
5. Run targeted tests plus relevant broader validation; verify failure/security/restart paths.
6. Never bypass tests, hooks, approvals, or security controls.
7. Update required docs.
8. Update Jira with PR URL, findings/blockers, validation evidence, and `PR Open` / `In Verification` status as appropriate.
9. Summarize problem, approach, changes, validation, and remaining risk.
10. Do not call merge-ready while validation fails/is missing, Jira is stale, or ticket/parent linkage is missing.
11. Do not mark Jira `Done` until ticket/parent criteria and required delivery/reconciliation evidence are satisfied.