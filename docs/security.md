# Security Posture

## Threat Model

| Threat | Mitigation |
|---|---|
| API key leakage | .env + detect-secrets in CI + pre-commit hook |
| PII in LLM calls | Regex redaction post-transcription, pre-storage |
| Prompt injection via transcript | Pattern-based detection before LLM nodes |
| Unauthorized access | Gradio auth, no anonymous endpoints |
| Data at rest exposure | SQLCipher encryption, encrypted audio retention |
| Audit trail tampering | Append-only audit log, separate from call data |

## PII Redaction

Runs at two points: post-transcription (before LLM) and pre-storage (before database).
Typed redaction: `[REDACTED_PHONE]`, `[REDACTED_EMAIL]`, `[REDACTED_SSN]`, `[REDACTED_CREDIT_CARD]`.

## Prompt Injection

Regex-based pattern matching for known injection techniques. Detected patterns are logged to LangSmith and processing halts for that call.

## Secret Management

- Local: `.env` file (gitignored)
- CI: GitHub Secrets
- Production: HuggingFace Spaces secrets
- Pre-commit hook scans for key patterns
