---
name: edward
description: Use Edward's complete CLI to capture, import, search, ask, organize, review, and manage personal research and writing projects.
---

# Edward Agent Skill

Edward is a local-first research memory and writing workspace. This skill is the agent's map of Edward's public capabilities. Agents and people use the same CLI and records; agents must not write directly to Edward's SQLite database.

## Operating rules

- Use `edward --help` and `edward COMMAND --help` to inspect the installed version and exact options before unfamiliar operations. Use `edward project --help` and `edward sync --help` for command groups.
- Add `--json` to commands that support it. Parse stdout as JSON and use the exit code; diagnostics belong on stderr.
- Preserve provenance: pass the source/origin, collector, run ID, and stable provider ID when available. Use an idempotency key when retrying a mutation.
- Do not claim a capture was enriched until its processing jobs complete. Use `status`, `process`, and `retry` to advance the queue.
- **Autonomous Follow-Through ("Do and Inform")**: When the user requests ingestion (`add`, `sync x`, `sync gmail`, `import-research`), advance safe, additive follow-through without pausing between routine stages: unroll X threads when available, fetch linked public pages, and run local extraction/embeddings for the new material. Download attachments only when the user requested them or they are clearly part of the requested import; do not treat attachment download as the default for Gmail. Check the processing scope and privacy policy first; never send private content to a hosted model without explicit user confirmation. Tell the user what completed and what is searchable.
- **Where to Still Ask Confirmation**: Reserve confirmation requests strictly for destructive actions (`purge`, `project delete`, `purge-jobs`) or before dispatching private content (Gmail, personal notes) to external hosted models.
- Treat `purge` as permanent deletion; use it only when the user explicitly asks to delete the record.
- Edward's built-in embeddings run locally. Classification and answer generation are separate. Follow Edward's configured privacy policy before sending private material to a hosted provider.
- **No model runs by default.** `ask`, finding extraction, and outline proposals are all deterministic unless a model is explicitly configured. Edward is memory, not a mind: the calling agent does the reasoning. Never imply that Edward thought about something on its own.
- `purge-jobs` deletes from a queue shared by every stage. It requires `--yes`, offers `--dry-run`, and refuses to run if the delete would touch a stage other than the one named. Never use it to tidy up records you created while testing without asking.

## What agents can do

### Capture and import

- `edward add`: capture a URL, file, note, text, or stdin. Include why the source matters as a note; set intent, origin, collector, and idempotency key when relevant.
- `edward sync x`: import new posts from the local Birdclaw archive. `--dry-run` previews; default limit is 25; `--all` imports all available new posts. Edward may use `xurl`, `bird`, archived post text, and `summarize` according to availability.
- `edward sync gmail`: import self-sent messages through `gog`. It searches `from:me to:me` and verifies message headers. Use `--dry-run` to preview and `--download-attachments` only when files should be downloaded.
- `edward import-research`: import a Markdown report or an Edward research bundle. A bundle is Edward's versioned JSON handoff for a completed multi-source research run; it is not a general industry standard. Use it when the agent has sources plus findings and evidence to submit together. For one item, prefer `add`. See the [bundle contract](../../../docs/AGENT_PROTOCOL.md#22-ingesting-structured-research-bundles-edward-import-research) and [v1 schema](../../../schemas/research-bundle-v1.json).

After `sync x`, thread unrolling runs automatically when `bird` is available. Check `status`, then process the new capture with `edward process --capture-id <CAPTURE_ID>` (repeat for imported captures as needed) so linked articles and local embeddings become searchable. Avoid processing unrelated queued work as a side effect. Honor privacy controls before any hosted classification; public-page fetching and local embedding do not require hosted-model dispatch. Summarize what was imported, threads unrolled, attachments downloaded, and pages indexed.

### Find, inspect, and answer

- `edward search QUERY`: exact lexical search, optionally filtered by intent, topic, form, or project.
- `edward ask QUESTION`: search the saved corpus, review up to 50 ranked candidates by default, and return a citation-backed answer. With a configured answer model, Edward reviews the retrieved set in bounded batches before synthesis. Use `--no-model --json` when the calling agent will read the evidence packet and answer itself; inspect the complete `items` array, including source metadata and supporting passages, rather than treating the human lookup text as exhaustive. Use `--project` to limit scope and `--limit` to adjust the number of retrieved items. Report the evidence coverage and do not imply that a bounded retrieval searched every possible phrasing or source.
- `edward show ID`: inspect a capture, resource, or finding and its provenance.
- `edward export`: export the corpus as JSONL or a bounded evidence packet (`--packet`) for a question. Use `--query`, `--limit`, and `--out` to bound or save the result.

For article or essay research, start with `ask`, then inspect the cited records with `show ID --json`. Run follow-up questions or searches for missing angles and counterarguments; one bounded evidence packet is not a claim of complete corpus coverage. Check `status --json` when linked pages or document processing may still be pending, and distinguish saved post text from linked-page contents that have not been fetched. Attribute unreviewed source claims to their sources when answering for the user.

For public PDF URLs, use `add --url URL --origin web`, then `process` to fetch, extract, classify, and embed. Extracted text is retained in full and page-local chunks carry page locators. Use Ask's matching passages for the question at hand, and inspect the full source with `show ID --json` when needed. `summarize --extract` may provide text without a summary. Do not treat a missing summary as missing source text, and do not treat a summary as a substitute for the full PDF. If a small local answer model falls back because it cannot cite sources, use `ask --no-model --json` and synthesize from the packet yourself.

### Organize, classify, and write

- `edward annotate ID`: add a note, label, or intent to an item. `edward remove-intent ID INTENT` deactivates an active intent.
- `edward classify ID` and `edward reclassify`: run or refresh taxonomic classification. `edward judgment list ID` inspects versioned judgments and probabilities. Classification is disabled by default and no query invokes it — but once labels exist, `--topic`/`--form` filters read them and every result carries a `labels` array, so labels change filtering and display. A pending `classify` backlog may be mostly redundant: check whether queued resources already carry labels before treating the count as a coverage gap.
- `edward project create|list|add|note|context|outline`: maintain research and writing workspaces, membership decisions, questions/gaps/counterarguments, and versioned evidence-linked outlines. Treat proposed outlines as drafts; accept or reject only when asked.

**For a writing project, `--input` (or MCP `outline={...}`) is the path that produces a usable outline.** The `--no-model` offline outline cannot read what any source says — it routes evidence only by the relationship each membership was filed under (`evidence`/`supporting` → `supporting`, `counterargument` → `counterevidence`, `background`/`question`/`gap` → `qualification`), and it does not distribute sources across sections by topic. So **mark the other side** with `--relationship counterargument` when adding evidence, or the offline outline will file it as support. Read the sources, write the argument, and submit it as an `--input` JSON file or MCP `outline` parameter; Edward validates that every citation is real project evidence and refuses fabricated ones. Full contract: [AGENT_PROTOCOL §2.8](../../../docs/AGENT_PROTOCOL.md#28-project-workspaces-edward-project).

#### Collaborative Essay Walkthrough Workflow (CLI & MCP)

When the user asks to write or outline an essay on a topic:
1. **Initialize Project Workspace**: Call `edward_project_create` (or `edward project create`) with title and thesis brief.
2. **Collect Evidence**: Use `edward_search` and `edward_ask` to locate relevant claims, passages, and data points. Link them using `edward_project_add_evidence` (marking relationships: `supporting`, `counterargument`, or `qualification`).
3. **Inspect Context**: Pull the full dossier using `edward_project_context` (or `edward project context --json`). Review source passages, user notes, and open research gaps.
4. **Draft and Walk Through with the User**:
   - The agent synthesizes an intellectually ordered narrative with coherent sections, clear claims, and cited `object_id`s.
   - Present the draft outline directly to the user in chat (or as an artifact) for walkthrough and discussion.
   - Iterate on the user's feedback (e.g. moving sections, emphasizing specific evidence, refining claims).
5. **Persist the Agreed Outline in Edward**:
   - Submit the refined outline via `edward_project_propose_outline(project_id, outline={...})` (or `edward project outline --propose --input outline.json`).
   - Edward validates all citations against workspace evidence and creates a versioned revision.
6. **Accept and Lock**:
   - Call `edward_project_accept_outline(project_id, version=...)` (or `edward project outline --accept --version N`) to promote it to the official accepted outline.
7. **Drafting or Export**:
   - The user can write the essay in their own editor guided by the locked outline, or collaborate with the agent to draft individual sections citing the verified evidence packet.

### Process and maintain the workspace

- `edward status`: inspect processing jobs and workspace counts. Read `by_stage_status` for the completed/failed split and the extracted/embedded resource counts for usable page coverage.
- `edward process`: run pending extraction, classification, finding-extraction, PDF attachment-extraction, and embedding jobs. It processes up to 10 jobs by default; use `--limit`, `--capture-id`, or `--stage` to scope a run.
- `edward retry --failed`: make failed jobs eligible to run again.
- `edward reindex`: rebuild the lexical index or embeddings (`--embeddings`). `--embeddings` walks every live resource and chunks **and** embeds it — this is also the repair for resources holding text but never chunked, because chunking is an implicit part of `embed` rather than a stage of its own. Those resources have **no job row at all**, so no pending count ever shows them.
- `edward repair-chunks`, `edward backfill-shortlinks`, `edward repair-threads`, `edward repair-titles`, `edward migrate-media`: targeted repairs, each safe to re-run. Order matters when recovering content: recover content → `repair-chunks` → `repair-titles` (titles last, because `repair-titles` derives from content and only writes when the stored title is author/URL-derived).
- `edward purge-jobs --stage X --status pending`: delete queued jobs for one stage only. Requires `--yes`; supports `--dry-run`.
- `edward backup`: create a database-and-blob backup. Take one before any bulk mutation.
- `edward doctor`: run health checks or verify a backup.
- `edward purge <ID> --confirm`: permanently remove an item and its search projection. Requires explicit `--confirm` and an explicit user request. It does **not** support projects.
- `edward intents` / `edward accept-intent`: list and accept intent questions. An intent is a facet that *filters* — it must never enter ranking.

### Model Context Protocol (MCP) Tools

When connected via Edward's native MCP server (`uv run edward mcp`), 23 strongly typed tools are available directly without shell invocation:

- **Discovery:** `edward_search`, `edward_ask` (Evidence Packet mode), `edward_show`, `edward_export_packet`
- **Capture:** `edward_add`, `edward_import_research` (Markdown report or JSON bundle), `edward_sync`
- **Annotations:** `edward_annotate`, `edward_list_intents`, `edward_accept_intent`, `edward_remove_intent`
- **Workspaces:** `edward_project_list`, `edward_project_create`, `edward_project_context`, `edward_project_add_evidence`, `edward_project_remove_evidence`, `edward_project_add_note`, `edward_project_propose_outline`, `edward_project_accept_outline`, `edward_project_delete`
- **Operations:** `edward_status`, `edward_process`, `edward_doctor`

### Automated MCP Setup For Agents

When the user asks to "set up Edward MCP", "get Edward MCP going", or "connect Edward to my agent", the agent can automate the entire setup:

1. **Locate Client Configuration**:
   - **Claude Desktop (macOS)**: `~/Library/Application Support/Claude/claude_desktop_config.json`
   - **Claude Desktop (Linux)**: `~/.config/Claude/claude_desktop_config.json`
   - **Claude Desktop (Windows)**: `%APPDATA%\Claude\claude_desktop_config.json`
   - **Cursor**: `~/.cursor/mcp.json` or `.cursor/mcp.json` in current workspace
   - **Windsurf**: `~/.codeium/windsurf/mcp_config.json`
   - **Antigravity / Gemini CLI**: User or workspace configuration (`~/.gemini/antigravity/mcp_config.json`)

2. **Add Edward Entry to `mcpServers`**:
   Read the existing JSON configuration file (or create `{ "mcpServers": {} }` if absent), resolve the absolute path to this Edward repository, and merge:
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
   *(For remote access across machines over SSH, use `"command": "ssh"` with `"args": ["user@host", "cd <PATH> && uv run edward mcp"]`)*.

3. **Verify and Report**:
   - Run `uv run edward mcp --help` to confirm entrypoint operation.
   - Run `uv run edward doctor --json` to confirm database and blob health.
   - Inform the user that the configuration is in place and to restart the client application to load the 23 tools.

All current top-level commands are listed by `edward --help`; check the live help because command options evolve. For exact JSON contracts, workflows, and exit codes, read [Edward Agent Protocol](../../../docs/AGENT_PROTOCOL.md). For architectural invariants and engineering guidelines, read [AGENTS.md](../../../AGENTS.md).
