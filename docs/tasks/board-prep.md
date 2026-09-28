# Board prep: Edward core tasks

Five self-contained tasks that improve Edward on its own (search, `ask`, exports, MCP)
and prepare it for the planned infinite-canvas board app. Each can be picked up
independently unless a dependency is noted. Findings below were measured on the real
library at `~/.edward` on 2026-09-27 (432 captures, 916 resources, 501 findings).

All work follows `AGENTS.md`: migrations only via numbered SQL files, `--json` stdout is
JSON only, exit codes 0/1/2/3, privacy checks before any hosted dispatch, and automated
work never overwrites `user_note`, human labels, intents, or `review_state`.

Suggested order: 1 → 2 → 4, with 3 whenever a model is chosen and 5 at any time.

---

## Task 1: Decode HTML entities and strip X page-title wrappers [DONE]

**Problem.** 196 resource titles contain raw HTML entities (`&quot;`, `&#x27;`, `&amp;`),
and 180 carry the X page-title wrapper `Name on X: "…" / X`. These show up in `search`,
`ask` evidence, exports, and project candidate lists.

Example (a `t.co` link that resolved to an X status page and was fetched as an article):

```
OpenAI on X: &quot;Please welcome GPT-6 Sol and GPT-6 Luna to the GPT-6 universe. … / X
```

**Root cause.** `clean_html_simple` in `src/edward/services/extract.py` (around line 65)
takes `<title>` text with a regex and never calls `html.unescape`. Check the other title
paths in `extract.py` (payload `title`/`name` fields, around lines 150–225) for the
same gap.

**Do.**
- Unescape entities wherever a title is taken from HTML.
- Add a pure helper in `src/edward/services/titles.py` that recognises the X wrapper
  `<author> on X: "<text>" / X` and returns the post text. Run it through the existing
  `shorten()` so it matches `derive_post_title` output, and keep the author in the
  `author` column if it is empty. Only strip the wrapper when the whole title matches
  the pattern; leave anything else alone.
- Extend `edward repair-titles` (currently `repair_author_derived_titles` in
  `services/lifecycle.py`) so it also repairs already-stored entity-encoded and wrapped
  titles. Report counts per repair kind in the `--json` payload.
- Keep `search_documents` (FTS5) in sync when a title changes, in the same transaction.

**Done when.**
- Unit tests in `tests/` cover entity decoding, the wrapper pattern (including titles
  cut off with `…` before the closing quote), and non-matching titles that must be left
  unchanged.
- On a copy of the real DB, `repair-titles --json` fixes the ~196 and ~180 cases, and a
  second run repairs 0 (idempotent).
- Do the dry run against a copy (`EDWARD_DATA_DIR=<tmp copy>`). The final run on
  `~/.edward` needs a backup first (`edward backup`).

---

## Task 2: Embed captures that have no text of their own [DONE]

**Problem.** 24 of 432 captures have no embedding, so they are invisible to vector
search and to the board. Almost all are X bookmarks whose `raw_content` and `user_note`
are empty; the substance lives in their linked resources (23 of the 24 have an embedded
linked resource).

**Root cause.** `compute_embedding_input_hash` in `src/edward/services/embed.py`
(around lines 347–360) builds capture input from `user_note + raw_content` only and
returns `None` when that is empty, so the embed stage silently skips the capture.

**Do.**
- When a capture has no text of its own, build its embedding input from its linked
  resources (`capture_resources` → latest `resource_contents`, falling back to the
  resource title), in a deterministic order. Include those resource content hashes in
  the input hash, so the capture re-embeds when a linked resource is re-extracted.
- Make sure the processor enqueues or re-runs `embed` for these captures once their
  resources finish extracting (see the dependency wiring in
  `services/processor.py` around line 879).
- Provide a way to backfill the existing 24: either an `edward reindex` option or
  scoped `edward process` runs. No new top-level command unless needed.

**Done when.**
- Tests: an empty-body capture with an extracted linked resource gets an embedding; a
  capture with neither text nor extracted resources is reported, not silently skipped.
- On a copy of the real DB, after backfill `edward status --json` shows
  `embedded_captures` at 431 of 432 (one manual capture has nothing to embed).

---

## Task 3: Resource summary stage (model-backed, opt-in) [DONE]

**Problem.** Only 17 of 916 resources have `resource_contents.summary`. Those 17 come
from `import-research` bundles (`services/bundle.py`, around line 866), which store the
first 300 characters of text. No processing stage generates summaries. `ask`, search
result snippets, exports, and the board's zoomed-in card view all need one.

**Do.**
- Add a `summarize` processing stage that runs after `extract`, using the existing
  OpenAI-compatible client in `services/llm.py`. It must stay **off unless explicitly
  configured**, like every other model use in Edward.
- Enforce the privacy rules in `services/privacy.py` **before** serialisation or
  dispatch. Gmail, personal notes, and local documents are never sent to a hosted
  provider unless the corresponding `EDWARD_HOSTED_*` allow flag is set. Invalid model
  output goes only to `<data-dir>/diagnostics/`.
- Output: a short summary (2–3 sentences, plain text) stored on the latest
  `resource_contents` row. Do not replace summaries that came from research bundles or
  from a human.
- Make it idempotent on content hash, and support scoping (`edward process --stage
  summarize`, `--capture-id`) so a backfill can be run deliberately.
- Model choice is pending. Design for a small, cheap model; do not hard-code a
  provider.

**Done when.**
- Tests use a fake client: summaries are stored, a private-class resource is refused
  before dispatch, bundle summaries are preserved, and a rerun is a no-op.
- `edward status --json` `summarized_resources` reflects the new stage.

---

## Task 4: `edward similar <id> --json` [DONE]

**Problem.** Edward can find items similar to a *query* (`search_vector` in
`services/embed.py`) but not items similar to an *existing item*. The board needs this
for "related items outside this project", and agents benefit directly.

**Depends on:** Task 2, so every capture has a vector.

**Do.**
- CLI: `edward similar <id> [--limit N] [--type capture|resource|finding]
  [--exclude-project <slug>] [--json]`. Accept capture, resource, and finding ids,
  and reuse the stored embedding (no re-embedding of the source item).
- Reuse the sqlite-vec path and the numpy fallback that `search_vector` already has.
  Refactor so query-vector search and item-vector search share one code path.
- Output per hit: `object_type`, `object_id`, `title`, `similarity`, and the capture id
  for resources, so a caller can map hits to cards. Exclude the item itself and
  soft-deleted rows.
- Expose the same operation as an MCP tool in `src/edward/mcp_server.py`, and document
  it in `docs/AGENT_PROTOCOL.md`.
- Exit codes: `1` for an unknown id or an item with no embedding (with a clear stderr
  message), `2` for bad options.

**Done when.**
- Tests cover each object type, `--exclude-project`, self-exclusion, and the JSON
  contract (stdout is JSON only).

---

## Task 5: Keep agent trial runs out of the real library [DONE]

**Problem.** Test data landed in the real library at `~/.edward`: 6 captures with
`origin_namespace = 'agent-test'` (collector `edward-mcp`), plus several throwaway
projects ("Test Energy Flexibility Deep Dive" and three "Energy Flexibility …"
variants, and "Hermes probe project", which is already soft-deleted). An agent
probing Edward over MCP or the CLI wrote into the user's real data.

**Do.**
- In `docs/AGENT_PROTOCOL.md` and `.agents/skills/edward/`, state that capability
  probes, smoke tests, and experiments must run against a scratch library
  (`EDWARD_DATA_DIR=$(mktemp -d)`), never the default data dir.
- Consider a cheap guard: make the MCP server / CLI print a stderr warning when a
  capture's `origin_namespace` looks like test data (`*-test`) and the data dir is the
  default one. Decide whether a warning or a refusal fits the "Tell, don't ask"
  posture.
- Confirm `tests/` never touches `~/.edward` (the fixtures should always set a temp
  data dir), and add a guard test if not.

**Do not** purge or delete the existing test captures or projects. Purge and
`project delete` are destructive and need the user's explicit confirmation. List the
ids for the user and let them decide.
