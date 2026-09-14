---
name: aicc-security-review
description: Use for security review or changes involving authorization, permissions, secrets, trust boundaries, external input, sensitive data, or privileged actions.
---
# Security Review
1. Identify identities, inputs, credentials, sensitive data, side effects, and trust boundaries.
2. Verify authn/authz independently; use least privilege and fail closed.
3. Validate untrusted input before files, shells, URLs, DBs, prompts, browsers, or privileged tools.
4. Keep secrets/private data out of source, logs, metrics, prompts, fixtures, and PR text.
5. Preserve approval, budget, audit, and policy controls.
6. Add negative/adversarial tests and prefer existing security primitives.