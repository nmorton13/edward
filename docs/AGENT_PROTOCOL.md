# Edward Agent Protocol (v1)

This document defines Edward's public interface for external agents (e.g. Claude Code, Codex, Cursor, AutoGPT, LangChain, or custom agents) interacting with an Edward personal research memory instance. Edward is agent-first: all core capabilities must be available through documented non-interactive CLI or library operations, with stable JSON contracts, IDs, provenance, and errors. Agents and people operate on the same records; agents must not write to SQLite directly.

Edward is neutral to agent and model vendor. Any authorized agent can use the public CLI and JSON contracts; no particular agent is part of Edward's identity or source of truth.

---

## 1. Interaction Principles

1. **CLI and MCP**: Agents use the `edward` CLI or the supported MCP tools. The CLI is the broader interface; use the live `--help` output to discover commands available in the installed version.
2. **JSON on Stdout**: When passed the `--json` flag, Edward guarantees that stdout receives **strictly** valid JSON. Any diagnostics, progress bars, or warnings are emitted to stderr.
3. **Exit Codes**:
   - `0`: Success.
   - `1`: Resource not found or domain business rule rejection.
   - `2`: Command line usage error or invalid arguments.
   - `3`: Idempotency conflict or fatal database error.
4. **Idempotency**: The CLI `add` and `import-research` commands accept an optional `--idempotency-key <KEY>`. Replaying a key with identical arguments returns the original result; reusing it with different arguments raises a conflict (`Exit 3`). Other commands do not currently expose this option.

---

## 2. Core Agent Commands

### 2.1 Capturing Evidence (`edward add`)

Capture a URL, raw text, or note:

```bash
edward add \
  --url "https://arxiv.org/abs/2401.00000" \
  --note "Key paper on local speculative decoding" \
  --intent "essay-seed" \
  --origin "arxiv" \
  --collector "research-agent-v1" \
  --idempotency-key "agent-run-42-item-1" \
  --json
```

**JSON Output Format:**
```json
{
  "status": "created",
  "capture_id": "cap_01j7xyz...",
  "resource_id": "res_01j7xyz...",
  "url": "https://arxiv.org/abs/2401.00000",
  "created_at": "2026-09-21T12:00:00Z"
}
```

### 2.2 Ingesting Structured Research Bundles (`edward import-research`)

The research bundle is Edward's own versioned JSON handoff format, not an industry-wide standard. Use it when an agent has completed a multi-source research run and wants to submit the sources, extracted text, findings, and evidence together. For a single URL, note, text, or file, use `edward add` instead. The contract is [`schemas/research-bundle-v1.json`](../schemas/research-bundle-v1.json); the Pydantic model and importer enforce the same v1 shape.

Import a bundle in one transaction:

```bash
edward import-research bundle.json --idempotency-key "run-123" --json
```

The bundle groups one research run under a title and ID. Each source may include a stable identity, origin, URL, title, extracted text or inline snapshot, retrieval time, content hash, and acquisition tool. Findings may include an assertion role, source URL, exact supporting passage, locator, confidence, labels, and entities. Suggested intents are imported as inactive suggestions for review. Use `extracted_text` or the inline `snapshot` field; filesystem paths are not accepted as snapshot content. `brief` is accepted by the schema, but the current importer does not retain it as searchable capture content.

Minimal example:

```json
{
  "type": "research-bundle",
  "schema_version": "1",
  "bundle_id": "costs-research-01",
  "title": "Research on reducing software costs",
  "agent": { "name": "Codex", "run_id": "run-123" },
  "sources": [{
    "origin": "web",
    "url": "https://example.com/report",
    "title": "The report",
    "extracted_text": "The report's extracted text goes here."
  }],
  "findings": [{
    "statement": "Automation reduced operating costs.",
    "assertion_role": "source-claim",
    "source_url": "https://example.com/report",
    "supporting_passage": "Automation reduced operating costs by 18%.",
    "locator": { "page": 4 }
  }]
}
```

By default, `bundle_id` supplies an idempotency key. Reimporting identical content replays the original result; reusing the key with different content is a conflict. `open_questions` is present in the v1 schema but is not yet persisted by the current importer; do not rely on it being retained.

**Required and tolerated fields — an easy trap.** `sources[].origin` is **required**; omitting it fails validation. The asymmetry is what bites: `ResearchBundle` sets `extra="forbid"` at the top level, so an unknown *top-level* key fails loudly, while `SourceItem` sets `extra="allow"`, so an unknown *source* key is silently accepted. A misspelled source field therefore imports without it and reports success. Validate against `schemas/research-bundle-v1.json` before submitting.

Findings submitted here carry provenance: `findings.extractor` records the `agent.name` you supply, so a finding is attributable to the agent that produced it.

### 2.2a Titles on import — what to expect

Titles are not derived automatically on every path. Supply one whenever you can, and know which paths drop it:

| How the item arrived | Title today |
| --- | --- |
| X post via `edward sync x` | Curated title from the capture store, else derived from post text |
| Link found inside an X post | The bare URL until its own fetch and extraction complete |
| `edward add --url` | **None** — `show` prints `Title: (No title)` |
| `edward add --text` | Not applicable: captures have no title field |
| Gmail self-note via `edward sync gmail` | **None** — the email subject is stored as a note, not a title |
| `import-research` source | **Your `title` field is honoured** — the reliable path |
| Markdown report | The `--title` flag |

Nothing later repairs a missing title on a non-X resource, so if a title matters, set it when you import. `import-research` and the `--title` flag are the two paths that let you control it.

### 2.3 Capability map

Agents can use all core Edward capabilities through the CLI:

- **Capture and import:** `add`, `sync x`, `sync gmail`, and `import-research` (bundle or Markdown).
- **Find and inspect:** `search`, `ask`, `show`, and `export` (JSONL corpus or bounded evidence packet).
- **Organize and review:** `annotate`, `remove-intent`, `intents`, `accept-intent`, `classify`, `reclassify`, `judgment list`, and `project` workflows (`create`, `list`, `add`, `note`, `context`, `outline`).
- **Process and maintain:** `status`, `process`, `retry --failed`, `purge-jobs`, `reindex`, `repair-titles`, `repair-threads`, `migrate-media`, `backfill-shortlinks`, `repair-chunks`, `backup`, `doctor`, and `purge --confirm`.

Use `edward --help`, `edward COMMAND --help`, `edward project --help`, and `edward sync --help` for options available in the installed version. Permanent deletion with `purge` requires an explicit user request.

**`purge-jobs` is guarded** because it deletes from a shared queue. It requires `--yes` (or `--dry-run` to preview) and refuses to act if the delete would touch a stage other than the one named:

```bash
edward purge-jobs --stage finding-extraction --dry-run --json
edward purge-jobs --stage finding-extraction --yes --json
```

**No model is used for extraction or answering by default.** Finding extraction is deterministic; `ask` returns deterministic and lookup answers unless `EDWARD_ANSWERER_MODE` is configured. Agent-submitted findings via `import-research` are the intended way to get model-quality extraction, and an agent may supply its own outline proposal with `--input`:

### 2.4 Import follow-through and processing (`edward status`, `edward process`)

Imports commit captures and resources before enrichment. A successful `sync` reports what it imported; it does not mean that classification, embeddings, or linked-page extraction have completed. For X bookmarks, the post text is stored and indexed once on the linked resource. The capture retains the bookmark event, stable X ID, and collection provenance, while the raw Birdclaw snapshot remains available.

Inspect the workspace queue after an import:

```bash
edward status --json
```

For X, pending work can include classification, local embedding, and fetch/extraction of URLs linked from the post. Extracted resources queue local embedding independently of classification; a failed hosted classifier does not block semantic retrieval. For Gmail or files, processing can also include linked pages and attachment extraction. `status` reports aggregate workspace counts by status and stage. Read `by_stage_status` for the completed/failed split and the extracted/embedded resource counts for usable page coverage; do not describe those totals as belonging only to the latest import unless you can attribute them to its IDs.

`edward process` runs pending jobs, up to 10 per invocation by default (use `--limit` to select a larger batch). Processing fetches linked public URLs, extracts clean text, computes local vector embeddings, runs OCR on image attachments, and evaluates taxonomic classification. Agents should follow through on safe, additive ingestion and scope work to new captures with `--capture-id` when possible; an unscoped run can process unrelated pending work. Before hosted-model dispatch, inspect the content classes that will be processed and obtain explicit user confirmation before sending private content (Gmail, personal notes, or local documents), even if local settings permit it. Attachment downloads remain opt-in where the source adapter requires that option. Report what completed and what is searchable.

### 2.5 Searching and Retrieving Evidence (`edward search`)

Agents can perform lexical (FTS5) and semantic retrieval:

```bash
edward search "speculative decoding latency" \
  --topic "ai/local-models" \
  --form "paper" \
  --limit 10 \
  --json
```

**JSON Output Format:**
```json
{
  "query": "speculative decoding latency",
  "count": 1,
  "results": [
    {
      "id": "res_01j7xyz...",
      "object_type": "resource",
      "title": "Fast Inference with Speculative Decoding",
      "snippet": "...achieves 2.5x speedup in <b>speculative decoding latency</b>...",
      "score": -4.21,
      "labels": ["ai", "paper", "benchmark"],
      "created_at": "2026-09-21T12:00:00Z"
    }
  ]
}
```

### 2.6 Exporting Evidence Packets (`edward export --packet`)

For synthesis tasks, agents retrieve bounded evidence packets conforming to [`schemas/evidence-packet-v1.json`](../schemas/evidence-packet-v1.json):

`edward ask QUESTION --json` searches the saved corpus and reviews up to 50 ranked candidates by default. A configured answer model reviews the retrieved evidence in bounded batches before final synthesis; the JSON answer includes `coverage.retrieved_items`, `coverage.reviewed_batches`, and `coverage.cited_items`. With `--no-model --json`, the calling agent receives the complete retrieved evidence packet and should read all `items`, their source metadata, and supporting passages before answering. `--limit` changes the retrieved set size; no bounded answer guarantees that every relevant record was found.

For writing research, an agent should inspect cited records with `show ID --json`, search again for missing angles or counterarguments, and check processing status before relying on linked-page or document contents. A saved X post and its unfetched links are different evidence. Attribute unreviewed source claims instead of presenting them as verified facts.

```bash
edward export --packet --query "local model quantization" --limit 20 --json
```

The packet contains stable IDs, assertion roles, exact quotes, and user notes ready for local LLM consumption.

### 2.7 Annotating and Reviewing (`edward annotate`)

Agents or humans can attach notes, labels, and review judgments:

```bash
edward annotate res_01j7xyz... \
  --label "benchmark" \
  --note "Verified on M3 Max with 64GB" \
  --actor "research-agent" \
  --json
```

### 2.8 Project Workspaces (`edward project`)

Create a workspace, attach canonical corpus objects, and inspect bounded project context:

```bash
edward project create --title "Local AI" --brief "How does local AI change creative work?" --json
edward project add <PROJECT_ID> <OBJECT_ID> --relationship supporting --status accepted --json
edward project context <PROJECT_ID> --refresh-candidates --json
```

Agents can return a structured outline through a JSON file:

```json
{
  "title": "Local AI",
  "premise": "Local inference changes the economics of creative iteration.",
  "sections": [
    {
      "heading": "Private iteration",
      "purpose": "Explain the workflow change.",
      "claim": "Creators can iterate without transmitting drafts.",
      "unresolved_research_needs": [],
      "evidence": [
        {
          "object_id": "find_01j7xyz...",
          "relationship": "supporting",
          "relevance_note": "Directly supports the section claim."
        }
      ]
    }
  ]
}
```

```bash
edward project outline <PROJECT_ID> --propose --input outline.json --json
```

Outline evidence IDs are rejected unless they are accepted or candidate members of the project. Proposals and revisions create immutable numbered versions; acceptance never mutates source evidence.

#### What the offline outline can and cannot do

`--no-model` (or `--propose` with no model configured) builds an outline with no model at all. It reads only the *relationship* each membership was filed under, and routes evidence by that signal:

| Membership relationship (`project add -r`) | Outline relationship it becomes |
| --- | --- |
| `evidence`, `supporting` | `supporting` |
| `counterargument` | `counterevidence` |
| `background`, `question`, `gap` | `qualification` |

Every membership is placed in a section, so nothing disappears silently. Unresolved needs are split across the two sections that host them rather than repeated in each.

**It does not understand what any source says.** With no model, the offline path cannot tell that one article is a counterargument and another is fiscal background. It knows only what you told it via `--relationship`. So:

- If you add evidence without a relationship, it defaults to `evidence`, which becomes `supporting`. A counterargument you did not mark will read as support.
- **Mark the other side.** `edward project add <P> <ID> --relationship counterargument` is what makes the offline outline file it correctly.
- The offline path will not distribute evidence across sections by topic. Section 2 collects the surplus supporting items; choosing which claim a source serves is the calling agent's job.

**Assigning evidence to claims is the agent's job**, via `--input` (or MCP `outline={...}`). That is where semantic judgment belongs, and it is the recommended path for real writing. The offline path is a floor that keeps everything reachable and honestly labelled. Do not read a good offline outline as evidence that the structure is right — only that the links are intact.

---

## 3. Citation Guidelines for Synthesizing Agents

When an agent answers user queries or writes project outlines based on Edward evidence, it must follow strict citation levels:

1. **Level 1 (Valid ID)**: Every claim must reference a valid Edward ID (`[#res_01j...]` or `[#find_01j...]`).
2. **Level 2 (In Packet)**: The referenced ID must be part of the active retrieval packet.
3. **Level 3 (Passage Match)**: Any quoted passage must match the canonical stored content or chunk locator.
4. **Level 4 (Support Verification)**: The synthesis output must indicate whether the claim is directly supported, partially supported, or extrapolated from the evidence.

---

## 4. Model Context Protocol (MCP) Interface

Edward exposes a native **Model Context Protocol (MCP)** server via `edward mcp` (default transport: `stdio`). This allows MCP-compatible agents (Antigravity, Claude Desktop, Cursor, Codex) to invoke Edward tools directly with typed parameters rather than constructing CLI subprocess strings.

### 4.1 Architecture & Posture

- **Memory, Not a Mind**: Edward acts as an evidence packet engine and durable research memory. It does not synthesize prose or draft essays inside the MCP server. Tools such as `edward_ask` return high-recall evidence packets with locators, citations, and source metadata, allowing the calling agent to perform the actual reasoning, sorting, and synthesis.
- **Typed Parameter Contracts**: Every tool defines strict Pydantic input schemas and returns JSON structured outputs.
- **MCP tool coverage**: The server exposes 23 typed tools for common capture, retrieval, project, and status workflows. It is a useful subset of the CLI, not full CLI parity; use `edward --help` for classification, repair, backup, purge, and other CLI-only operations.

### 4.2 Tool Catalog

#### Discovery & Retrieval
- **`edward_search`**: Exact lexical full-text search (SQLite FTS5 `unicode61`) with optional taxonomic topic, form, intent, or project filters.
  - Parameters: `query: str`, `limit: int = 20`, `topic: str | None`, `form: str | None`, `intent: str | None`, `project: str | None`
- **`edward_ask`**: Run hybrid retrieval in deterministic Evidence Packet mode (the MCP tool always disables model synthesis). Returns candidate evidence items, source metadata, and supporting passages.
  - Parameters: `query: str`, `limit: int = 50`, `project: str | None = None`
- **`edward_show`**: Retrieve complete object details (resource, capture, or finding) by ID, including clean text, source URL, labels, intents, and human annotations.
  - Parameters: `object_id: str`
- **`edward_export_packet`**: Export a bounded, standalone Evidence Packet JSON for any topic or query.
  - Parameters: `query: str`, `limit: int = 50`

#### Ingest & Capture
- **`edward_add`**: Capture a URL, direct text, thought note, or local file into research memory.
  - Parameters: `url: str | None`, `text: str | None`, `note: str | None`, `intent: str | None`, `file_path: str | None`, `origin: str = "agent"`
- **`edward_import_research`**: Ingest a complete Markdown research report or structured research bundle JSON.
  - Parameters: `content: str`, `format: str = "json"` (or `"markdown"`), `title: str | None`, `collector: str = "agent"`
- **`edward_sync`**: Import from a local source (`x` from Birdclaw or `gmail` via `gog`). The external source is read-only; imported records are written to Edward. The MCP tool does not expose the CLI's dry-run or attachment-download options.
  - Parameters: `source: str = "x"`, `all_records: bool = False`, `limit: int = 25`

#### Annotations & Intent Facets
- **`edward_annotate`**: Attach private notes, taxonomic labels, or intent flags to any existing object.
  - Parameters: `object_id: str`, `note: str | None`, `intent: str | None`, `label: str | None`, `actor: str = "agent"`
- **`edward_list_intents`**: Retrieve the active intent questions taxonomy (`intent-essay-seed`, `intent-deep-dive`, `intent-counterevidence`, etc.).
  - Parameters: None
- **`edward_accept_intent`**: Convert a suggested intent facet into an accepted, human-owned decision.
  - Parameters: `object_id: str`, `intent: str`
- **`edward_remove_intent`**: Deactivate an active intent facet on an object.
  - Parameters: `object_id: str`, `intent: str`, `actor: str = "agent"`

#### Writing Project Workspaces
- **`edward_project_list`**: List all active writing and research projects.
  - Parameters: None
- **`edward_project_create`**: Create a new writing workspace with title, premise brief, and optional slug.
  - Parameters: `title: str`, `brief: str | None`, `slug: str | None`
- **`edward_project_context`**: Retrieve complete project premise, accepted evidence, notes, research gaps, counterarguments, and the latest outline revision.
  - Parameters: `project_id: str`, `include_rejected: bool = False`
- **`edward_project_add_evidence`**: Pin a resource, capture, or finding to a project workspace with an explicit role (`supporting`, `counterargument`, `qualification`).
  - Parameters: `project_id: str`, `object_id: str`, `relationship: str = "supporting"`, `note: str | None`, `actor: str = "agent"`
- **`edward_project_remove_evidence`**: Unpin evidence from a project workspace.
  - Parameters: `project_id: str`, `object_id: str`
- **`edward_project_add_note`**: Log an open research question, gap, or counterargument note in a project workspace.
  - Parameters: `project_id: str`, `kind: str` (`"note"`, `"question"`, `"gap"`, `"counterargument"`), `text: str`, `actor: str = "agent"`
- **`edward_project_propose_outline`**: Propose a new versioned outline revision. When an outline dict is provided, validates that all cited evidence IDs exist in the project; if omitted, generates a deterministic outline proposal.
  - Parameters: `project_id: str`, `outline: dict | None`, `author_id: str = "agent"`
- **`edward_project_accept_outline`**: Accept and commit a proposed outline revision as the project's official outline.
  - Parameters: `project_id: str`, `version: int | None`, `outline_id: str | None`, `actor: str = "agent"`
- **`edward_project_delete`**: Soft-delete a project workspace (requires `confirm=True`).
  - Parameters: `project_id: str`, `confirm: bool = False`

#### Maintenance & Queue
- **`edward_status`**: Report processing queue depth, job status counts, and total corpus counts.
  - Parameters: None
- **`edward_process`**: Execute pending background extraction, classification, and local embedding jobs.
  - Parameters: `limit: int = 50`
- **`edward_doctor`**: Run system diagnostics verifying SQLite integrity, migrations, and blob store health.
  - Parameters: None

### 4.3 Client Configuration & Automated Setup

When an agent is asked to configure Edward MCP for a user or environment, it should execute the following automated steps:

#### 1. Identify Client Config File
- **Claude Desktop (macOS)**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Claude Desktop (Linux)**: `~/.config/Claude/claude_desktop_config.json`
- **Claude Desktop (Windows)**: `%APPDATA%\Claude\claude_desktop_config.json`
- **Cursor**: `~/.cursor/mcp.json` (global) or `<workspace>/.cursor/mcp.json` (project-local)
- **Windsurf**: `~/.codeium/windsurf/mcp_config.json`
- **Antigravity / Gemini CLI**: User or workspace MCP configuration (`~/.gemini/antigravity/mcp_config.json`)

#### 2. Local stdio Configuration
Read the existing configuration file (or initialize `{ "mcpServers": {} }` if missing), and merge the `edward` server entry into `"mcpServers"`:

```json
{
  "mcpServers": {
    "edward": {
      "command": "uv",
      "args": [
        "run",
        "--directory",
        "<ABSOLUTE_PATH_TO_EDWARD_REPO>",
        "edward",
        "mcp"
      ]
    }
  }
}
```

#### 3. Remote / Cross-Machine Configuration
- **Over SSH (Recommended)**:
  ```json
  {
    "mcpServers": {
      "edward": {
        "command": "ssh",
        "args": [
          "user@remote-host",
          "cd <ABSOLUTE_PATH_TO_EDWARD_REPO> && uv run edward mcp"
        ]
      }
    }
  }
  ```
- **Over Network SSE**:
  Start on the host: `uv run edward mcp --transport sse`
  Client config:
  ```json
  {
    "mcpServers": {
      "edward": {
        "url": "http://<remote-host-ip>:8000/sse"
      }
    }
  }
  ```

#### 4. Automated Verification
The agent should test the configuration:
1. Run `uv run edward mcp --help` to confirm CLI entrypoint operates.
2. Run `uv run edward doctor --json` to confirm database and blob health.
3. Inform the user that restarting their client application will activate the 23 tools.
