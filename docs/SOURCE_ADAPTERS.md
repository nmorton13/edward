# Local source adapters

Edward can import X bookmarks and self-sent Gmail messages through local tools. These adapters are optional and do not add Python dependencies for Birdclaw, X, or Google access.

## X bookmarks

Birdclaw remains responsible for synchronizing X bookmarks into its local archive. Edward reads `~/.birdclaw/birdclaw.sqlite` in SQLite read-only mode; set `EDWARD_BIRDCLAW_DB` if the archive is elsewhere. The Edward adapter does not run a Birdclaw sync or change bookmarks.

```bash
# Inspect how many new bookmarks are available
uv run edward sync x --dry-run --json

# Import 25 new bookmarks, the default
uv run edward sync x --json

# Import all remaining bookmarks
uv run edward sync x --all --json
```

Each capture uses the tweet ID as its stable source ID and `https://x.com/i/status/<id>` as its primary URL. Edward first tries `xurl read`, then `bird read --json`, and uses archived Birdclaw text when a reader is unavailable or fails. If the archive has no post text, Edward can fall back to its normal `summarize` extraction path. Raw archive entities and media metadata are retained in a source snapshot. Post links become referenced resources and are queued for normal extraction.

The capture keeps the bookmark event and its provenance: origin `x`, tweet ID, collection channel, and the link to the primary resource. The post text itself belongs to that resource and is stored and indexed there once. This avoids duplicate capture/resource text and duplicate embeddings while retaining the raw Birdclaw snapshot for source provenance. The capture record remains available for later annotations or personal context.

`sync x` indexes the imported post text for lexical search immediately, but it does not run the processing queue. It leaves classification pending. Resource classification can queue finding extraction, which then schedules the resource embedding job. Linked-page fetches are also pending. Until those jobs run, the post is available to exact search, but semantic retrieval and extracted content from linked pages are not ready.

## Self-sent Gmail

Install and configure `gog` separately. Edward discovers threads with `from:me to:me`, fetches full threads through `gog gmail thread get`, and passes `--readonly --no-input` to both commands. It verifies the actual message headers and imports only messages where a sender address also appears among the recipients. Edward does not modify Gmail.

```bash
# Search and count matching threads without fetching them
uv run edward sync gmail --dry-run --json

# Import all discovered self-sent messages
uv run edward sync gmail --json

# Restrict one run to 20 threads
uv run edward sync gmail --limit 20 --json

# Explicitly download attachments to Edward's content-addressed blob store
uv run edward sync gmail --download-attachments --json
```

Each email message is a separate capture with its stable message ID; its subject, sender, recipients, date, body, thread ID, links, and attachment metadata are retained. Links become reusable referenced resources and enter the same extraction queue as X links. Attachment metadata is retained by default; actual file downloads require the explicit option. Files that `gog` downloads and Edward can match by filename are stored as immutable local blobs and linked to that capture. Downloaded PDFs also enter Edward's local text extraction queue, so selectable text becomes searchable after processing. Scanned PDFs need OCR before Edward can index their text.

After either sync, inspect the queue with `uv run edward status --json`. `uv run edward process` runs pending classification, local embedding, linked-page extraction, and downloaded-PDF text extraction jobs as applicable. Scope it with `--capture-id` where possible so it does not process unrelated queued work. Classification may use the configured classifier; linked public URLs may be fetched over the network. Before hosted-model dispatch, check the affected content classes and get explicit user confirmation before sending private Gmail, notes, or documents, even if local settings permit it. PDF attachment downloads remain opt-in. The normal resource extractor uses `summarize` when installed and has a local fallback. PDF text extraction runs locally. `process` handles at most 10 jobs per invocation by default; use `--limit` to choose a larger batch or run it again while work remains. A failed job does not undo the imported capture; `status` reports the failure count, and `uv run edward retry --failed --json` can make failed jobs eligible to run again.

Gmail uses a seven-day overlap from its last completed scan and skips message IDs already captured. A failed thread fetch leaves the checkpoint unchanged, so the scan can be retried. A limited Gmail run does not advance the complete-scan checkpoint. `--dry-run` performs source discovery only and does not fetch individual posts or email threads.
