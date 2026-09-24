# Developer Guide for Coding Agents Working on Edward

Welcome to **Edward**, the personal research memory and writing workspace.

When developing or extending this codebase, adhere strictly to the following architectural invariants and engineering guidelines.

---

## 1. Core Architectural Invariants

1. **Clean-Slate Design**: There is no legacy migration subsystem or backward-compatibility layer with prior bookmark managers. Do not introduce legacy tables or converters.
2. **Deterministic CLI Contracts**:
   - Machine consumption uses `--json`.
   - On `--json`, standard stdout must contain **only** valid JSON.
   - All human-facing progress messages, warnings, and error diagnostics go strictly to `stderr`.
   - Exit code `0` on success, `1` on domain/input failure, `2` on usage error, `3` on conflict/fatal error.
3. **Database Rules**:
   - SQLite with WAL mode (`PRAGMA journal_mode = WAL;`) and foreign keys enforced (`PRAGMA foreign_keys = ON;`).
   - SQLite schema changes are handled strictly through numbered migration scripts in `src/edward/migrations/*.sql`.
   - Never use model-generated raw SQL queries. Direct queries must use static parameterized SQL.
4. **Content-Addressed Blobs**:
   - Heavy bodies, raw source snapshots, and attachments are stored outside SQLite in `<data-dir>/blobs/<hash-prefix-2>/<sha256-hash>`.
   - Blobs are written atomically (temp file write $\to$ hash verification $\to$ atomic rename).
   - Blobs are immutable and path-confined to `<data-dir>/blobs/`.
5. **FTS5 Lexical Tokenization**:
   - Full-text search uses SQLite FTS5 with `tokenize = 'unicode61 remove_diacritics 2'` (never Porter stemming).
   - `search_documents` is managed by application transactions (synchronously updated on capture, soft-delete, and purge).
   - Exact phrase and quote validation must always check against canonical source text and locators, not FTS token matches alone.
6. **Privacy Invariants**:
   - Known hosted classifier providers (`typesafe`, `openrouter`) are **always forced to `location = hosted`** in application code.
   - Gmail, personal notes, and local documents are never dispatched to hosted providers unless explicitly enabled via environment variables (`EDWARD_HOSTED_GMAIL=allow`, etc.).
   - Privacy checks run and halt **before** serialization or HTTP network dispatch.
   - Raw invalid LLM outputs go only into `<data-dir>/diagnostics/`, never into public logs, SQLite valid findings, or git.
7. **Holistic Backups**:
   - Backups must snapshot SQLite (using `sqlite3_backup_*` API to capture WAL pages) **and** all referenced blobs in `<data-dir>/blobs/`.
   - Every backup generates an integrity `manifest.json` with SHA-256 hashes of all backed-up files.

---

## 2. Directory Layout

```text
edward/
├── pyproject.toml               # Hatchling build, packaging rules, dependencies
├── README.md                    # User quickstart and overview
├── AGENTS.md                    # This guide
├── schemas/                     # Canonical public schemas (research-bundle-v1, evidence-packet-v1)
├── docs/
│   └── AGENT_PROTOCOL.md        # External agent specification
├── .agents/skills/edward/       # Agent skill wrapper
├── src/edward/
│   ├── __init__.py
│   ├── __main__.py
│   ├── cli.py                   # Typer CLI commands
│   ├── mcp_server.py            # Model Context Protocol (MCP) server
│   ├── db.py                    # SQLite connection, WAL, migrations, vector fallback
│   ├── blobs.py                 # Content-addressed blob store
│   ├── models.py                # Pydantic v2 domain schemas
│   ├── migrations/              # Versioned SQL migrations (001_initial_schema.sql, ...)
│   ├── registries/              # Versioned taxonomies (topics, forms, signals, jev, thresholds)
│   ├── services/                # Domain services (capture, search, lifecycle, audit, backup, ...)
│   └── classifiers/             # Classification subsystem (base, system_one, jev, providers)
└── tests/                       # Pytest test suite
```

---

## 3. Development Workflows

Run commands using `uv`:

```bash
# Sync dependencies
uv sync

# Run tests
uv run pytest

# Check formatting and linting
uv run ruff check .
uv run ruff format --check .

# Build package wheel
uv build
```

---

## 4. Preservation of Human Data

Automated operations (adapters, reclassifications, refreshes) must **never** overwrite:
- User-authored notes (`user_note`)
- Human-assigned labels (`source = 'human'`)
- Intent flags (`intents`)
- Human review decisions (`review_state`)

---

## 5. Agent Autonomy & Communication Posture

1. **Autonomous Follow-Through ("Do and Inform")**:
   - When a user asks to capture, sync, or import research, carry out safe, additive follow-through autonomously: unroll available X threads, fetch linked public resources, and run local extraction and embeddings so the new content becomes searchable.
   - Scope processing to the new captures where possible (`edward process --capture-id`). Avoid processing unrelated queued work as a side effect.
   - Download attachments only when the user requested them or they are clearly part of the requested import; this is not the default for Gmail.
   - Before hosted-model dispatch, inspect the affected content classes and privacy settings. Explicit user confirmation is required before private content (Gmail, personal notes, or local documents) is sent to hosted models, even when local configuration would otherwise allow it.
   - Do not interrupt the user for routine, local, additive processing stages.
2. **Tell, Don't Ask**:
   - Inform the user clearly and concisely about what was completed (threads unrolled, media attached, linked pages fetched, embeddings updated).
3. **Guarded Boundaries**:
   - Explicit user confirmation is strictly reserved for destructive mutations (`purge`, `project delete`, `purge-jobs`) or before dispatching private data (Gmail, personal notes) to external hosted models.
