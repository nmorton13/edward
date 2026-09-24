# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed — Edward is "memory, not a mind"

- Clarified the local data location, network boundaries, hosted privacy controls, and which ingestion follow-through is automatic versus explicitly requested.
- Corrected the Agent Protocol's idempotency and MCP coverage/parameter descriptions to match the current CLI and tools.

- **No model runs by default.** `EDWARD_ANSWERER_MODE` unset means the answerer is absent, and a leftover `EDWARD_ANSWER_BASE_URL` or model name no longer re-enables one. Previously the answerer defaulted on, so a model was live in every session without being chosen.
- **Finding extraction is deterministic.** The `finding-extraction` stage no longer constructs a client; it uses only a caller-supplied one. Agent-submitted bundles (`import-research`) are the path that produces findings worth keeping — the heuristic extractor reads navigation chrome as prose (`"Close \n\n Where is this?"` as a question).
- **`ask` resolves deterministically** unless a caller supplies a client. Counts and dates are answered directly; otherwise it returns a bounded evidence packet.
- The model paths were **demoted, not deleted** — a caller may still pass its own client to `extract_findings_for_resource` or outline generation.
- `README.md`, `docs/AGENT_PROTOCOL.md`, and `.agents/skills/edward/SKILL.md` now state this posture.
- Fixed duplicate `2.5` section numbering and cross-references in `docs/AGENT_PROTOCOL.md` and `.agents/skills/edward/SKILL.md`.
- Documented required `--failed` flag on `edward retry` and added missing maintenance and intent commands to `README.md` and `docs/AGENT_PROTOCOL.md`.
- Corrected SQLite database path in `.env.example` to `edward.sqlite3`.

### Fixed

- **Offline outlines route evidence by relationship, not slice position.** `--propose --no-model` built a flat supporting list and sliced it, so items landed in DB order; every membership mapping to `qualification` (`background`, `question`, `gap`) was computed and then **silently dropped** by every section; with two or fewer supporting items the first two sections emitted the same items twice; and every unresolved need was repeated in every section. Now groups by `to_evidence_relationship()` — every membership is placed and needs are split across the sections that host them. Deliberately *not* added: inferring stance from source text, because a wrong assignment is harder to notice than a missing one.
- **Outline proposals validate against their own schema.** The compact evidence payload used `id` where `object_id` is required and passed stored membership relationships verbatim where only `supporting | counterevidence | qualification` are valid, so every *model-generated* outline failed validation 100% of the time. `VALID_EVIDENCE_RELATIONSHIPS` was defined and never referenced; it now does the mapping.
- **`project context` renders the latest outline.** Its help text promised the outline and `--json` carried it, but the text renderer stopped after gaps.
- **A startup read no longer takes a write lock.** Every CLI command calls `run_migrations()`, which ran `BEGIN IMMEDIATE` unconditionally — so while `reindex` held its single long transaction, `status`, `doctor`, `search` and `ask` all failed with `database is locked` after the 5s busy timeout. Now probes read-only and escalates only when a migration, a registry row, or a name mismatch on an applied version needs writing. The probe is fail-open on contention so a pending migration can never be silently skipped.

### Added

- **Native Model Context Protocol (MCP) server (`edward mcp`)**: Exposes 23 strongly typed tools covering discovery (`edward_search`, `edward_ask`, `edward_show`, `edward_export_packet`), capture (`edward_add`, `edward_import_research`, `edward_sync`), human annotations (`edward_annotate`, `edward_list_intents`, `edward_accept_intent`, `edward_remove_intent`), writing project workspaces (`edward_project_list`, `edward_project_create`, `edward_project_context`, `edward_project_add_evidence`, `edward_project_remove_evidence`, `edward_project_add_note`, `edward_project_propose_outline`, `edward_project_accept_outline`, `edward_project_delete`), and maintenance (`edward_status`, `edward_process`, `edward_doctor`).
- **Expanded Taxonomies and Intent Facets**: Added 9 new topic definitions (`infrastructure/energy-grid`, `economics/policy`, `economics/austrian`, `crypto/bitcoin`, `software/systems`, `gaming`, `philosophy`, `politics/commentary`, `creative-tech`), 2 signals (`cool-project`, `data-source`), 19 Jev evaluation questions with tuned decision thresholds across all policies, and 6 new intent questions (`essay-seed`, `deep-dive`, `counterevidence`, `fact-check`, `tool-eval`, `inspiration`).
- `edward purge-jobs --stage X --status pending [--dry-run|--yes]` — delete queued jobs for one stage. Refuses without `--yes`, and aborts if the delete would touch any other stage.
- `edward project delete <PROJECT_ID> --confirm [--json]` — soft-delete an active project workspace with full audit event emission.
- `edward process --reconcile [--json]` — reconcile and mark pending classification jobs as completed when their target objects already possess judgments or labels.
- `tests/test_mcp_server.py`, `tests/test_offline_outline_assignment.py`, `tests/test_startup_write_lock.py`, `tests/test_answerer_posture.py`, `tests/test_findings_deterministic.py`, `tests/test_ask_posture.py`, `tests/test_purge_jobs.py`, `tests/test_docs_claims.py`, `tests/test_project_delete.py`, `tests/test_classification_queue_sync.py` — each pins a documented guarantee so a stale doc, queue de-sync, or regression fails the build.

## [0.1.0] - 2026-09-21

### Added
- Initial project release of Edward.
- GitHub-ready repository foundation with `uv`, Hatchling, and dual-OS CI.
- Complete SQLite domain schema with WAL mode, foreign keys, and versioned migrations.
- Origin vs. collector provenance tracking for delegated collection.
- Dedicated `idempotency_keys` table with conflict detection on hash mismatch.
- Managed content-addressed blob storage under `<data-dir>/blobs/`.
- FTS5 full-text search with `unicode61 remove_diacritics 2` tokenization.
- Core CLI commands: `edward add`, `show`, `search`, `annotate`, `remove-intent`, `backup`, `doctor`, `purge`.
- Agent-neutral JSON contracts: `docs/AGENT_PROTOCOL.md`, `schemas/research-bundle-v1.json`, `schemas/evidence-packet-v1.json`.
- Google Agents / Antigravity skill: `.agents/skills/edward/SKILL.md`.
- Versioned classification registries in `src/edward/registries/`.
- Holistic integrity backup generating `manifest.json`.
- System diagnostics in `edward doctor`.
