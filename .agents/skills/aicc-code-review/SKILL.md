---
name: aicc-code-review
description: Use to review code or a PR and assess whether AI Commander changes are ready to merge.
---
# Code Review
Review in order: correctness/security, reuse, simplicity, validation evidence.
- Read surrounding code/contracts, not only the diff.
- Check errors, side effects, retries, cleanup, browser state, concurrency, and restart safety where relevant.
- Identify existing patterns/capability that should be reused.
- Challenge unnecessary abstraction/dependencies.
- Verify positive and negative tests.
- Separate blockers from optional cleanup.