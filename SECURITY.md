# Security Policy

## Supported Versions

| Version | Supported |
|---|---|
| 0.1.x (current) | ✅ Active |

---

## Reporting a Vulnerability

**Please do NOT open a public GitHub Issue for security vulnerabilities.**

If you believe you have discovered a security issue in llmscope, please
report it privately so we can address it before public disclosure.

### How to Report

1. **Email**: Send details to **chanduchitikam@gmail.com** with subject line:
   `[SECURITY] llmscope — <brief description>`
2. **GitHub Private Advisory**: Open a
   [Security Advisory](https://github.com/Chandu00756/LLmThoughtLens/security/advisories/new)
   via the repository's Security tab.

### What to Include

- A clear description of the vulnerability and potential impact.
- Steps to reproduce (proof of concept if possible).
- Affected version(s).
- Any suggested mitigations you are aware of.

---

## Response Timeline

| Stage | Target |
|---|---|
| Acknowledgement | Within **48 hours** |
| Initial assessment | Within **5 business days** |
| Fix or mitigation plan | Within **30 days** (critical issues sooner) |
| Public disclosure | Coordinated with reporter after fix is released |

---

## Scope

### In Scope

- Arbitrary code execution via crafted model inputs or provider responses.
- Credential or API key leakage through logs or serialised outputs.
- Dependency vulnerabilities introduced by llmscope's dependency tree.
- Path traversal or file read/write via report or graph export features.

### Out of Scope

- Vulnerabilities in the underlying AI model APIs (OpenAI, Anthropic, etc.).
- Issues in user-supplied code or prompts (prompt injection is an LLM
  concern, not a library concern).
- Denial-of-service via intentionally malformed inputs where the attack
  requires local access.

---

## Security Best Practices for Users

- **Never commit API keys** — use environment variables or a secrets manager.
- **Pin dependencies** in production — use `pip-audit` or `safety` to scan.
- **Review outputs** — llmscope surfaces raw model activations; treat them
  as untrusted data from an external service.
- **Restrict file system access** — report export paths should be validated
  at the application layer.

---

## Acknowledgements

We appreciate responsible disclosure and will acknowledge contributors in
our release notes (unless anonymity is requested).
