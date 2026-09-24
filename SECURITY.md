# Security Policy

## Reporting Security Issues

If you discover a potential security vulnerability in Edward, report it privately rather than creating a public issue. On GitHub, use [Report a vulnerability](https://github.com/nmorton13/edward/security/advisories/new) (private vulnerability reporting); if that option is unavailable, contact the maintainer through the [GitHub profile](https://github.com/nmorton13). We appreciate responsible disclosure and will address confirmed vulnerabilities promptly.

---

## Security Principles & Threat Model

Edward is designed to store personal research material, including notes, bookmarks, email summaries, and agent findings. Untrusted web content must not be permitted to escalate privilege, leak private data, or access local networks.

### 1. SSRF & Network Boundary Defense
- **Pre-Validation:** All external URLs are validated against private, loopback, and cloud metadata IP ranges (`127.0.0.0/8`, `10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16`, `169.254.169.254`, IPv6 `::1`, `fc00::/7`) **before** being passed to internal HTTP clients or external CLI tools (`summarize`, `xurl`, `bird`).
- **Redirect Re-validation:** Edward's HTTP client validates resolved DNS addresses at every hop during redirect chains.
- **External CLI Trust Boundary:** Internal redirects followed by third-party binaries cannot be inspected without an OS-level proxy or sandbox; pre-CLI IP validation ensures initial targets are strictly public.

### 2. Subprocess Execution Security
- Subprocesses are executed exclusively using argument lists (`shell=False`).
- Subprocess timeouts and maximum output buffer limits (20MB) prevent memory exhaustion.
- Sensitive environment variables are stripped from subprocess environments, and stderr logging redacts recognized credential patterns.

### 3. Outbound Data Privacy Policies
- Outbound data transmission to hosted models (LLMs or classifiers) is governed by explicit privacy policies:
  - `EDWARD_HOSTED_GMAIL=deny`
  - `EDWARD_HOSTED_PERSONAL_NOTES=deny`
  - `EDWARD_HOSTED_DOCUMENTS=deny`
  - `EDWARD_HOSTED_PUBLIC_WEB=allow`
- Known hosted providers (`typesafe`, `openrouter`) are forced to `location = hosted` in application code to prevent accidental bypass via configuration errors.
- Any request attempting to send restricted content classes to a hosted provider is halted before network serialization.

### 4. Content-Addressed Blob Confinement
- Managed blob storage is confined to `<data-dir>/blobs/<hash-prefix>/<content-hash>`.
- Path traversal outside the blob directory is explicitly blocked.
- Write operations are atomic and verified against the computed SHA-256 hash.

### 5. Private Diagnostic Protection
- Raw responses from failed model schema validations are saved to `<data-dir>/diagnostics/`, disabled by default for private content, redacted, excluded from regular backups and exports, and ignored by Git.
