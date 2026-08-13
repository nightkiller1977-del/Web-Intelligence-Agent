# Security Policy

## Reporting Vulnerabilities

**Do not open a public issue for security vulnerabilities.**

If you discover a security vulnerability in Web Intelligence Agent, please email security details to the maintainers rather than using the public issue tracker.

### What to Include
- Description of the vulnerability
- Steps to reproduce (if applicable)
- Potential impact
- Suggested remediation (if you have one)

---

## Security Best Practices

### Environment Configuration
- **Never commit `.env` files** to the repository. Use `.env.example` as a template only.
- All sensitive configuration (API keys, tokens, authentication) must be provided via environment variables at runtime.
- In development, use a local `.env` file (ignored by `.gitignore`).
- In production, use your deployment platform's secrets management (Render environment variables, etc.).

### Authentication Security
- All protected routes require bearer token authentication.
- Tokens are compared using `secrets.compare_digest()` to prevent timing attacks.
- In local mode, tokens are auto-generated as ephemeral 64-character hex strings if not provided.
- In remote mode, tokens must be explicitly configured via environment variables.

### API Keys
- Never commit OpenAI API keys, Tavily API keys, or other external API credentials.
- All API keys must be provided via environment variables.
- Use the `OPENAI_API_KEY` and `TAVILY_API_KEY` environment variables (see `.env.example`).
- Rotate API keys regularly.

### Testing
- Test tokens should never be hardcoded in source files.
- Test configuration should use environment variable overrides or fixtures.
- Use `.env.test` (ignored by `.gitignore`) for test-specific credentials.
- Do not document real or example tokens in README or reports.

### Code Security
- All external inputs are validated before processing.
- No sensitive data (tokens, API keys, PII) should be logged.
- Use proper error handling that doesn't expose implementation details.
- Dependencies are regularly audited for vulnerabilities.

### Deployment Security
- Container images use minimal base images (Python slim).
- Health checks and proper startup sequences prevent bad state propagation.
- Logging levels are set to INFO or above in production (never DEBUG).
- Deployment manifests use environment variable substitution for secrets.
- CORS is disabled by default in remote mode.
- Documentation access requires authentication in remote mode.

### Storage Security
- Local storage (SQLite) is used only in development/testing.
- Production deployments use Redis with proper network isolation.
- All storage operations are audited and logged.

---

## Security Audit Checklist

- [ ] No hardcoded API keys in source code
- [ ] No hardcoded tokens in test fixtures
- [ ] `.env.example` uses placeholders only
- [ ] All credentials come from environment variables
- [ ] `.gitignore` properly excludes `.env` files
- [ ] No sensitive data in README or documentation
- [ ] Dependencies are minimal and audited
- [ ] Authentication uses timing-safe comparison
- [ ] Error messages don't leak sensitive information
- [ ] Logging doesn't capture credentials

---

## Dependency Management

Regularly audit Python dependencies for vulnerabilities:

```bash
pip install pip-audit
pip-audit
```

Review `requirements.lock` for suspicious or outdated packages. Keep dependencies current but stable.

---

## Questions?

For security-related questions or clarifications, please reach out to the maintainers privately.
