---
name: aicc-root-cause-fix
description: Use for bugs, regressions, flaky failures, stabilization, or unexpected AI Commander behavior.
---
# Root-Cause Fix
1. Reproduce with the smallest reliable evidence.
2. Trace current behavior before editing.
3. Fix the component that owns the failure.
4. Add regression coverage when practical.
5. Check retries, cleanup, concurrency, idempotency, restart behavior, and ambiguous side effects where relevant.
6. Run targeted tests, then broader relevant validation.