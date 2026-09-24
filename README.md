# Edward

**A personal research memory and writing workspace.**

Edward keeps sources, why you saved them, and useful evidence together. Capture a link, email yourself a note, or add a document; Edward queues work to fetch and index it when you run `process` (or schedule that command). Later, retrieve it with exact search, semantic discovery, or evidence-backed questions.

---

## Why Edward?

I started Edward because I am constantly bookmarking things I like on X, sending links and notes to myself in email, reading papers, and collecting articles. And I always had trouble keeping track of it.

I tried all sorts of different apps and systems—read-it-later queues, bookmark managers, personal wikis, and browser tab dumps. But they almost always failed in one of two ways:
1. **The Hoarder's Graveyard**: Loose link dumpers where things go in and are never seen again, because search is poor or the context of *why* you saved it is lost.
2. **The High-Friction Tax**: Overly rigid knowledge bases that demand manual tagging, filing, and folder management up front when you just want to save something interesting and move on.

Like [Mentat](https://github.com/nmorton13/mentat), my opinionated memory system for personal thoughts, Edward is built around a workflow I wanted for research. Captures are quick; queued fetching, extraction, and indexing can be run when you choose or automated with scripts and cron.

The point of saving something is rarely just to own the link. Most of what I save, I want to return to later:
- To seed an essay or commentary piece.
- To study a dense technical topic through a deep dive.
- To hold onto inconvenient counterevidence or dissenting data.
- To check a specific statistic, quote, or claim before citing it.
- To evaluate a cool open-source project, game demo, or new tool.
- Or just to remember a great point for a talk or debate.

### Mentat and Edward: Thoughts vs. Research

If you use both, the division of labor is clean:
- **[Mentat](https://github.com/nmorton13/mentat)** is for your **internal thoughts**: reflections from a walk, quotes that linger, decisions, voice journals, and evolving perspectives across time.
- **Edward** is for your **external research**: articles, X threads, self-sent emails, PDFs, documentation, and data sources. It extracts full clean text, maintains content-addressed snapshots, embeds passages, and assembles verified evidence into structured writing projects.

---

## What Edward Does

- **Zero-Friction Ingestion**: Bookmark on X (via Birdclaw), forward a link or note to yourself in Gmail (via `gog`), paste text or files from your terminal (`edward add`), or import structured research bundles from AI agents.
- **Clean Architecture & Tripartite Separation**:
  | Record | What it preserves |
  | --- | --- |
  | **Capture** | The event and context: why, when, and where a post, email, thought, or report was saved. |
  | **Resource** | The reusable source: an article, post, PDF, or repository with its content-addressed snapshot. Multiple captures link to the same resource without duplicate blobs. |
  | **Finding** | A specific claim, quotation, question, or conclusion extracted from or verified against the source text. |
- **Queued Processing**: Web pages and PDFs can be fetched and extracted into clean readable text by running `edward process`. You can automate it with a scheduler; Edward does not run a background daemon by itself. If `summarize` is installed, it is used for web extraction.
- **Dual-Engine Retrieval**:
  - **Exact Lexical Search**: Powered by SQLite FTS5 (`unicode61` tokenizer) for finding exact phrases, names, and code snippets.
  - **Local Semantic Search**: FastEmbed (`BAAI/bge-small-en-v1.5`) running entirely in local Python for conceptual and thematic retrieval.
- **Intent Facets**: Answers *why* you kept something (`essay-seed`, `deep-dive`, `counterevidence`, `fact-check`, `tool-eval`, `inspiration`, `reference`, `try-later`). Intents are user-level facets for filtering, never opaque weights that distort search rankings.
- **Taxonomic Classification**: Evaluates topics (`ai/local-models`, `software/systems`, `crypto/bitcoin`, `economics/austrian`, `infrastructure/energy-grid`, `creative-tech`, `gaming`, `philosophy`, etc.) and signals (`cool-project`, `field-report`, `benchmark`, `tutorial`, `warning`, `data-source`).
- **Writing Projects & Syntheses**: Group sources into dedicated writing workspaces, assign explicit roles (`supporting`, `counterevidence`, `qualification`), track open research gaps and questions, and generate cited outlines.
- **Agent and Human Pair**: A rock-solid CLI with `--json` output where machine outputs are strictly JSON on `stdout` and logs go to `stderr`, making it trivial for agents (Codex, Claude, Antigravity) to research and draft with you.

---

## Quick Start

### Installation

Requires Python 3.11 or newer and [`uv`](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/nmorton13/edward.git
cd edward
uv sync
```

### Save and Retrieve in Seconds

```bash
# 1. Save a quick thought with an intent
uv run edward add \
  --text "Energy grid constraints will shape data center siting far more than compute efficiency." \
  --intent essay-seed \
  --json

# 2. Save a URL and why you kept it
uv run edward add \
  --url "https://example.com/grid-reliability-study" \
  --note "Key transmission capacity data for PJM and ERCOT" \
  --intent reference \
  --json

# 3. Process pending extraction and local embeddings
uv run edward process --limit 20

# 4. Search exact words via SQLite FTS5
uv run edward search "transmission capacity" --json

# 5. Query evidence without calling an external model
uv run edward ask "What have I saved about grid constraints?" --no-model --json
```

`add` records the capture instantly so you can get right back to reading. `process` runs through pending background jobs—fetching pages, extracting clean text from HTML or PDFs, and generating local embeddings.

---

## Ingesting Content

Edward accepts content from anywhere you encounter it.

| Source | Command | Details |
| --- | --- | --- |
| **URL or Local File** | `edward add --url <URL>` / `edward add --file <PATH>` | Extracts clean text, stores snapshots in content-addressed storage (`blobs/`), and queues embeddings. PDFs are parsed locally. |
| **Quick Note / Thought** | `edward add --text "<NOTE>" --intent <INTENT>` | Saves a thought directly with an intent facet. |
| **Terminal / Clipboard** | `pbpaste \| edward add --stdin --origin personal` | Ingests piped text from standard input. |
| **X Bookmarks** | `edward sync x --all` | Syncs bookmarks from Birdclaw's local archive. Automatically unrolls author threads into full articles with media and external links when `bird` is installed. |
| **Self-Sent Email** | `edward sync gmail` | Ingests notes and links you emailed to yourself via `gog` in read-only mode, verifying headers. |
| **Markdown Report** | `edward import-research report.md` | Ingests structured Markdown synthesis into resources and findings. |
| **Agent Research Bundle** | `edward import-research bundle.json` | Canonical multi-source JSON schema for agent research handoffs. |

### Automating Ingestion (Cron / Scripts)

You can set up a simple script or cron job to run sync and processing automatically in the background:

```bash
#!/usr/bin/env bash
# ~/.local/bin/edward-sync.sh
cd /path/to/edward || exit 1

# Sync new X bookmarks from Birdclaw
uv run edward sync x --all --quiet

# Sync self-sent emails
uv run edward sync gmail --quiet

# Process up to 50 extraction and embedding jobs in the background
uv run edward process --limit 50 --quiet
```

Add to crontab (`crontab -e`) to run every hour:
```cron
0 * * * * /path/to/edward/scripts/edward-sync.sh >> ~/.edward/sync.log 2>&1
```

---

## Search, Discovery & Answering

### Exact Lexical Search (`search`)

```bash
# Search for exact phrases or terms across all resources
uv run edward search "proof-of-work"

# Filter by primary form, topic, or signal
uv run edward search "systems" --topic software/systems --form article
uv run edward search "benchmark" --signal benchmark --topic ai/local-models
```

Lexical search uses SQLite FTS5 with diacritic removal and no destructive stemming, ensuring technical terms, code symbols, and exact quotes match precisely.

### Asking Questions & Assembling Evidence (`ask`)

`ask` retrieves the best matching passages across your corpus using hybrid search:

```bash
# Return a structured Evidence Packet for an agent or personal inspection (no model required)
uv run edward ask "What arguments were made against utility rate decoupling?" --no-model --json

# Answer using an optional configured local model with citation verification
uv run edward ask "Summarize the key findings on transformer latency"
```

When an answer model is configured and active, `ask` verifies every citation against the underlying source text. If citations cannot be verified, it cleanly falls back to returning the retrieved evidence packet rather than hallucinating.

---

## Projects: From Research to Writing

Edward includes dedicated writing project workspaces designed to turn scattered research into coherent essays, articles, or talks.

```bash
# 1. Create a writing project
uv run edward project create \
  --title "The Real Bottlenecks in Modern Compute" \
  --brief "Analyzing electrical grid constraints, cooling limits, and substation lead times." \
  --json

# 2. Add saved evidence to the project with explicit relationships
uv run edward project add <PROJECT_ID> <RESOURCE_ID> \
  --relationship supporting \
  --note "Provides empirical substation transformer lead time data." \
  --json

# Mark the counterarguments explicitly
uv run edward project add <PROJECT_ID> <OTHER_RESOURCE_ID> \
  --relationship counterargument \
  --note "Argues behind-the-meter solar + batteries mitigates grid interconnection queue delays." \
  --json

# 3. Record open research gaps and questions
uv run edward project note <PROJECT_ID> --kind gap --text "Need recent 2025/2026 PJM interconnection queue reports."
uv run edward project note <PROJECT_ID> --kind question --text "What percentage of planned capacity drop out before interconnection?"

# 4. View full project context and candidate sources
uv run edward project context <PROJECT_ID> --refresh-candidates --json

# 5. Propose a structured outline
# Option A: Offline deterministic grouping by relationship (no model)
uv run edward project outline <PROJECT_ID> --propose --no-model --json

# Option B: Calling agent supplies full narrative structure with verified citations
uv run edward project outline <PROJECT_ID> --propose --input outline.json --json

# 6. Accept the outline
uv run edward project outline <PROJECT_ID> --accept
```

### Outline Invariants & Evidence Mapping

Outlines carry explicit relationships for each cited claim: `supporting`, `counterevidence`, or `qualification`.
- **Offline outline (`--no-model`)**: Places memberships into sections based on their assigned relationship. It does not read what the source text says, which is why marking `--relationship counterargument` when adding evidence is crucial.
- **Agent outline (`--input outline.json`)**: An agent (or human) crafts the section narrative and thesis; Edward strictly verifies that every cited object ID corresponds to real project evidence.

---

## Local Models, Classification & Privacy

Edward is **memory, not a mind**. Local capture, indexing, search, and deterministic answers can run without a hosted model. Fetching web sources requires network access, and FastEmbed may download model weights the first time it runs; hosted answer or classification providers are optional.

### What Runs Without Any Model

The default posture requires no external APIs or local LLM runtimes:
- Ingestion, URL fetching, PDF parsing, and text extraction.
- Deterministic finding extraction (sentences, quotes, claims).
- FTS5 full-text indexing and querying.
- FastEmbed vector embeddings (`BAAI/bge-small-en-v1.5` running locally in Python).
- Projects, outline proposals (`--no-model`), and evidence packet exports (`--no-model`).
- Deterministic question answering for counts, dates, and entity lookups.

### Embeddings

Embeddings run locally in Python via FastEmbed. The model weights are automatically cached under `<data-dir>/models/`. To chunk and embed your entire library in one pass:

```bash
uv run edward reindex --embeddings
```

### Opting In to an Answer Model (Off by Default)

Absence is the default. Edward will never contact an LLM unless you explicitly declare an answerer mode:

```bash
# In your ~/.edward/.env or shell:
EDWARD_ANSWERER_MODE=local-model          # local-model | local | hosted | disabled
EDWARD_ANSWER_BASE_URL=http://localhost:11434/v1
EDWARD_ANSWER_MODEL=llama3.2
EDWARD_ANSWER_LOCATION=local
```

Setting `EDWARD_ANSWER_BASE_URL` alone does nothing: model use strictly requires `EDWARD_ANSWERER_MODE`.

### Classification and Privacy Guarantees

Taxonomic classification evaluates topic and signal probabilities. Two distinct rules govern how it interacts with the system:
1. **Not invoked during retrieval**: Queries never trigger a classifier call or wait on an LLM.
2. **Labels are consumed by the query surface**: Stored labels populate `--topic`, `--signal`, and `--form` filters, and results return with their assigned `labels[]`.

**Privacy invariant**: Local files, personal notes, and self-sent emails are blocked from hosted classifiers (`typesafe`, `openrouter`) unless their corresponding `EDWARD_HOSTED_*` environment setting is explicitly `allow`. Privacy checks run before serialization or network dispatch. Classification is disabled by default.

---

## Maintaining Your Research Library

Edward provides self-healing tools to ensure database integrity, repair outdated chunks, and clean stale metadata:

```bash
# Inspect queue depth, processing stats, and corpus coverage
uv run edward status --json

# Verify SQLite database and blob store integrity
uv run edward doctor

# Retry any failed extraction or embedding jobs
uv run edward retry --failed

# Chunk and embed any live resources missing embeddings
uv run edward reindex --embeddings

# Re-chunk resources if extraction was updated or replaced
uv run edward repair-chunks

# Re-derive titles that fell back to author names or URLs
uv run edward repair-titles

# Unroll X bookmark threads into complete multi-tweet articles with media
uv run edward repair-threads --dry-run
uv run edward repair-threads

# Re-fetch resources that were stored as shortlink interstitials (t.co, bit.ly)
uv run edward backfill-shortlinks --dry-run
uv run edward backfill-shortlinks

# Create a snapshot backup of SQLite and content-addressed blobs
uv run edward backup

# Delete queued jobs for a specific stage (guarded, requires confirmation)
uv run edward purge-jobs --stage finding-extraction --dry-run --json
uv run edward purge-jobs --stage finding-extraction --yes --json

# Permanently purge a specific capture, resource, or project
uv run edward purge <OBJECT_ID> --confirm
uv run edward project delete <PROJECT_ID> --confirm
```

---

## Configuration Reference (common settings)

Edward reads environment variables and a `.env` file from the current directory or `~/.edward/.env` (environment variables take precedence). The database, blobs, diagnostics, and model cache live under Edward's local data directory—not in this repository by default. This data can contain private research; do not commit it or share backups unintentionally. The data directory is platform-specific unless `EDWARD_DATA_DIR` is set; set `EDWARD_DB_PATH` to place the database elsewhere.

| Variable | Description | Default |
| --- | --- | --- |
| `EDWARD_DATA_DIR` | Storage directory for SQLite, blobs, diagnostics, and models | `~/.edward` (or platform standard) |
| `EDWARD_DB_PATH` | Path to the SQLite database | `<data-dir>/edward.sqlite3` |
| `EDWARD_EMBEDDING_MODEL` | FastEmbed model name | `BAAI/bge-small-en-v1.5` |
| `EDWARD_ANSWERER_MODE` | Opt-in answer model mode (`local-model`, `local`, `hosted`, `disabled`) | `disabled` |
| `EDWARD_ANSWER_BASE_URL` | OpenAI-compatible API base URL | — |
| `EDWARD_ANSWER_MODEL` | Chat model name for synthesis | — |
| `EDWARD_ANSWER_LOCATION` | Declared location (`local` or `hosted`) | — |
| `EDWARD_ANSWER_API_KEY` | API key for an opted-in answer model | — |
| `EDWARD_ANSWER_PROVIDER` | Answer provider identifier | `llm` |
| `EDWARD_ANSWER_TIMEOUT` | Per-request timeout in seconds | `120` |
| `EDWARD_CLASSIFIER_PROVIDER` | Taxonomic classifier provider (`disabled`, `typesafe`, `openrouter`, `local`, `dry-run`) | `disabled` |
| `EDWARD_CLASSIFIER_MODEL` | Classifier model name | Provider-specific |
| `EDWARD_CLASSIFIER_BASE_URL` | Classifier API base URL | Provider-specific |
| `EDWARD_CLASSIFIER_TRANSPORT` | Hosted transport used by the Jev classifier | `typesafe` |
| `EDWARD_HOSTED_GMAIL` | Allow private Gmail content to hosted classifiers (`allow` or `deny`) | `deny` |
| `EDWARD_HOSTED_PERSONAL_NOTES` | Allow personal notes to hosted classifiers | `deny` |
| `EDWARD_HOSTED_DOCUMENTS` | Allow personal documents to hosted classifiers | `deny` |
| `EDWARD_HOSTED_PUBLIC_WEB` | Allow public-web content to hosted classifiers | `allow` |
| `EDWARD_RECORD_PRIVATE_DIAGNOSTICS` | Opt in to recording private-content diagnostics (`1` enables) | `0` |
| `EDWARD_BIRDCLAW_DB` | Path to Birdclaw's X bookmark archive | `~/.birdclaw/birdclaw.sqlite` |

Known hosted classifier providers are always treated as hosted. Private content is blocked before serialization or network dispatch unless its corresponding `EDWARD_HOSTED_*` setting is `allow`. Merely configuring credentials or an endpoint does not enable answer-model use; set `EDWARD_ANSWERER_MODE` to opt in. `.env.example` is a starter template for common settings; it also shows the privacy defaults.

---

## Agent Protocol & Machine Contracts

Edward is designed from day one to be pair-programmed with AI coding agents:
- Pass `--json` to any command for valid machine-readable JSON on `stdout`.
- Progress logs, warnings, and diagnostic traces always go to `stderr`.
- Deterministic exit codes: `0` = success, `1` = domain failure, `2` = usage error, `3` = fatal/conflict.
- **Native MCP Server**: Run `edward mcp` to connect agents directly via the Model Context Protocol.

See:
- [Agent Protocol](docs/AGENT_PROTOCOL.md)
- [Research Bundle Schema](schemas/research-bundle-v1.json)
- [Evidence Packet Schema](schemas/evidence-packet-v1.json)
- [Edward Agent Skill](.agents/skills/edward/SKILL.md)

---

## Model Context Protocol (MCP) Server

Edward includes a native **Model Context Protocol (MCP)** server for AI coding assistants and desktop agents (Claude Desktop, Antigravity, Cursor, Codex).

Instead of parsing CLI flags or running shell commands, connected agents can invoke Edward's research memory, discovery, project workspaces, and capture capabilities as 23 strongly typed tools:

```bash
uv run edward mcp --transport stdio
```

### Connecting Claude Desktop or Antigravity

Add Edward to your `claude_desktop_config.json` or Antigravity MCP configuration:

```json
{
  "mcpServers": {
    "edward": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/edward", "edward", "mcp"]
    }
  }
}
```

### Available MCP Tools

| Group | Tools | What it does |
| --- | --- | --- |
| **Discovery & Retrieval** | `edward_search`<br>`edward_ask`<br>`edward_show`<br>`edward_export_packet` | Exact FTS5 search, high-recall evidence packets with locators and citations, full object inspection, and JSON evidence export. |
| **Ingest & Capture** | `edward_add`<br>`edward_import_research`<br>`edward_sync` | Quick capture of notes/URLs/files, bulk Markdown or JSON research report import, and read-only source archive synchronization. |
| **Human Annotations & Intent** | `edward_annotate`<br>`edward_list_intents`<br>`edward_accept_intent`<br>`edward_remove_intent` | Attach notes, labels, or intent flags (`essay-seed`, `deep-dive`, `counterevidence`) to objects, review intent taxonomy, and manage human review decisions. |
| **Writing Workspaces** | `edward_project_list`<br>`edward_project_create`<br>`edward_project_context`<br>`edward_project_add_evidence`<br>`edward_project_remove_evidence`<br>`edward_project_add_note`<br>`edward_project_propose_outline`<br>`edward_project_accept_outline`<br>`edward_project_delete` | Full writing project lifecycle: track evidence with roles (`supporting`, `counterargument`, `qualification`), log research questions and gaps, and manage versioned evidence-linked outlines. |
| **Maintenance & Operations** | `edward_status`<br>`edward_process`<br>`edward_doctor` | Queue depth inspection, background job execution, and SQLite/blob store integrity diagnostics. |

---

## Development

```bash
# Sync dependencies
uv sync

# Run tests
uv run pytest

# Format and lint
uv run ruff check .
uv run ruff format .

# Build package
uv build
```

See [AGENTS.md](AGENTS.md) for architectural invariants and engineering guidelines.

---

## License

[MIT](LICENSE) © 2026 Nathan Morton
