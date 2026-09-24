"""Fixture tests for local Birdclaw and gog source adapters."""

import json
import sqlite3
from pathlib import Path

import pytest
from typer.testing import CliRunner

from edward.cli import app
from edward.models import CaptureInput
from edward.services import source_adapters as adapters
from edward.services.capture import capture_item
from edward.services.subprocess_runner import SubprocessResult


def test_birdclaw_bookmark_reader_excludes_other_collection_sources(tmp_path, monkeypatch):
    archive = tmp_path / "birdclaw.sqlite"
    with sqlite3.connect(archive) as conn:
        conn.executescript(
            """
            CREATE TABLE tweet_collections (
                account_id TEXT, tweet_id TEXT, collected_at TEXT, source TEXT,
                raw_json TEXT, kind TEXT
            );
            CREATE TABLE tweets (
                id TEXT, author_profile_id TEXT, text TEXT, created_at TEXT,
                entities_json TEXT, media_json TEXT, deleted_at TEXT
            );
            CREATE TABLE profiles (id TEXT, handle TEXT, display_name TEXT);
            INSERT INTO tweets VALUES
                ('bird-post', NULL, 'Downloaded bookmark', NULL, NULL, NULL, NULL),
                ('demo-post', NULL, 'Demo bookmark', NULL, NULL, NULL, NULL);
            INSERT INTO tweet_collections VALUES
                ('acct', 'bird-post', NULL, 'Bird', NULL, 'bookmarks'),
                ('acct', 'demo-post', NULL, 'demo', NULL, 'bookmarks');
            """
        )
    monkeypatch.setattr(adapters, "_birdclaw_database_path", lambda: archive)

    rows = adapters._read_birdclaw_bookmarks()

    assert [row["tweet_id"] for row in rows] == ["bird-post"]


def test_x_sync_captures_archive_post_links_and_is_idempotent(
    test_db, test_blob_store, monkeypatch
):
    monkeypatch.setattr(
        adapters,
        "_read_birdclaw_bookmarks",
        lambda: [
            {
                "account_id": "acct",
                "tweet_id": "123",
                "collected_at": "2026-01-01T00:00:00Z",
                "source": "sync",
                "author_handle": "writer",
                "author_name": "Writer",
                "published_at": "2025-12-31T00:00:00Z",
                "text": "Reduce spending: https://example.com/story",
                "entities_json": json.dumps(
                    {
                        "urls": [
                            {"expanded_url": "https://example.com/story"},
                            {"expanded_url": "https://x.com/i/status/456"},
                            {"expanded_url": "https://x.com/i/status/123"},
                        ]
                    }
                ),
                "media_json": "[]",
                "deleted_at": None,
            }
        ],
    )
    monkeypatch.setattr(adapters, "_x_reader_payload", lambda _tweet_id: (None, None, "offline"))
    monkeypatch.setattr(adapters, "is_tool_available", lambda _name: False)

    result = adapters.sync_x_bookmarks(test_db, test_blob_store)

    assert result["captured"] == 1
    assert result["linked_urls"] == 2
    assert result["reader_errors"] == [{"tweet_id": "123", "error": "offline"}]
    with test_db.connection() as conn:
        capture = conn.execute("SELECT * FROM captures WHERE origin_namespace = 'x'").fetchone()
        capture_search = conn.execute(
            "SELECT body FROM search_documents WHERE object_type = 'capture' AND object_id = ?",
            (capture["id"],),
        ).fetchone()
        resource = conn.execute(
            "SELECT * FROM resources WHERE id = (SELECT resource_id FROM capture_resources WHERE capture_id = ? AND relationship_type = 'primary')",
            (capture["id"],),
        ).fetchone()
        snapshot = conn.execute(
            "SELECT * FROM source_snapshots WHERE resource_id = ?", (resource["id"],)
        ).fetchone()
        resource_content = conn.execute(
            "SELECT clean_text, extractor FROM resource_contents WHERE resource_id = ?",
            (resource["id"],),
        ).fetchone()
        fetch_job = conn.execute(
            "SELECT status FROM processing_jobs WHERE resource_id = ? AND stage = 'resource-fetch'",
            (resource["id"],),
        ).fetchone()
        classify_job = conn.execute(
            "SELECT status FROM processing_jobs WHERE resource_id = ? AND stage = 'classify'",
            (resource["id"],),
        ).fetchone()
        capture_embed_job = conn.execute(
            "SELECT id FROM processing_jobs WHERE capture_id = ? AND stage = 'embed'",
            (capture["id"],),
        ).fetchone()
        linked = conn.execute(
            "SELECT canonical_url FROM resources WHERE canonical_url = 'https://example.com/story'"
        ).fetchone()
        referenced_post = conn.execute(
            "SELECT canonical_url FROM resources WHERE canonical_url = 'https://x.com/i/status/456'"
        ).fetchone()
        job = conn.execute(
            "SELECT status FROM processing_jobs WHERE resource_id = (SELECT id FROM resources WHERE canonical_url = 'https://example.com/story')"
        ).fetchone()
    assert capture["origin_id"] == "123"
    assert capture["raw_content"] is None
    assert capture_search is not None
    assert "Reduce spending" not in capture_search["body"]
    assert capture_embed_job is None
    assert resource["canonical_url"] == "https://x.com/i/status/123"
    assert snapshot is not None
    assert resource_content["clean_text"].startswith("Reduce spending")
    assert resource_content["extractor"] == "birdclaw"
    assert fetch_job["status"] == "completed"
    assert classify_job["status"] == "pending"
    assert linked is not None
    assert referenced_post is not None
    assert job["status"] == "pending"

    again = adapters.sync_x_bookmarks(test_db, test_blob_store)
    assert again["captured"] == 0
    assert again["already_imported"] == 1


def test_gmail_sync_uses_readonly_gog_and_verifies_headers(test_db, test_blob_store, monkeypatch):
    calls = []

    def fake_run_tool(args, **_kwargs):
        calls.append(args)
        if args[1:3] == ["gmail", "search"]:
            return SubprocessResult(0, json.dumps({"threads": [{"id": "thread-1"}]}), "", 0.01)
        return SubprocessResult(
            0,
            json.dumps(
                {
                    "messages": [
                        {
                            "id": "message-1",
                            "threadId": "thread-1",
                            "from": "Nate <nate@example.test>",
                            "to": "Nate <nate@example.test>",
                            "subject": "Keep this note",
                            "date": "Mon, 01 Jun 2026 10:00:00 +0000",
                            "textBody": "A useful note https://example.test/article",
                        },
                        {
                            "id": "message-2",
                            "from": "Someone <other@example.test>",
                            "to": "Nate <nate@example.test>",
                            "subject": "Not self-sent",
                            "textBody": "Skip me",
                        },
                    ]
                }
            ),
            "",
            0.01,
        )

    monkeypatch.setattr(adapters, "is_tool_available", lambda name: name == "gog")
    monkeypatch.setattr(adapters, "run_tool", fake_run_tool)

    result = adapters.sync_self_sent_gmail(test_db, test_blob_store)

    assert result["captured"] == 1
    assert result["rejected_not_self_sent"] == 1
    assert result["linked_urls"] == 1
    assert all("--readonly" in call and "--no-input" in call for call in calls)
    assert calls[1][1:4] == ["gmail", "thread", "get"]
    with test_db.connection() as conn:
        capture = conn.execute("SELECT * FROM captures WHERE origin_namespace = 'gmail'").fetchone()
        cursor = conn.execute(
            "SELECT cursor_value FROM source_cursors WHERE source = 'gmail-self'"
        ).fetchone()
        linked = conn.execute(
            "SELECT 1 FROM resources WHERE canonical_url = 'https://example.test/article'"
        ).fetchone()
    assert capture["origin_id"] == "message-1"
    assert capture["collection_channel"] == "gog"
    assert cursor["cursor_value"].startswith("2026-06-01")
    assert linked is not None


def test_gmail_partial_failure_does_not_advance_checkpoint(test_db, test_blob_store, monkeypatch):
    calls = 0

    def fake_run_tool(args, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return SubprocessResult(0, '{"threads":[{"id":"thread-1"}]}', "", 0.01)
        return SubprocessResult(1, "", "network unavailable", 0.01)

    monkeypatch.setattr(adapters, "is_tool_available", lambda name: name == "gog")
    monkeypatch.setattr(adapters, "run_tool", fake_run_tool)

    with pytest.raises(adapters.SourceAdapterError, match="network unavailable"):
        adapters.sync_self_sent_gmail(test_db, test_blob_store)
    with test_db.connection() as conn:
        cursor = conn.execute("SELECT 1 FROM source_cursors WHERE source = 'gmail-self'").fetchone()
    assert cursor is None


def test_x_sync_cli_json_dry_run_is_valid_json(test_db, test_blob_store, monkeypatch):
    monkeypatch.setattr(adapters, "_read_birdclaw_bookmarks", lambda: [])
    monkeypatch.setattr("edward.cli.get_services", lambda: (test_db, test_blob_store))

    result = CliRunner().invoke(app, ["sync", "x", "--dry-run", "--json"])

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["bookmarks_found"] == 0


def test_gmail_attachment_download_is_stored_as_blob(test_db, test_blob_store, monkeypatch):
    calls = []

    def fake_run_tool(args, **_kwargs):
        calls.append(args)
        if args[1:3] == ["gmail", "search"]:
            return SubprocessResult(0, '{"threads":[{"id":"thread-1"}]}', "", 0.01)
        output_dir = Path(args[args.index("--out-dir") + 1])
        (output_dir / "note.txt").write_text("mail attachment", encoding="utf-8")
        return SubprocessResult(
            0,
            json.dumps(
                {
                    "messages": [
                        {
                            "id": "message-1",
                            "from": "Nate <nate@example.test>",
                            "to": "Nate <nate@example.test>",
                            "subject": "Attachment",
                            "textBody": "See attached",
                            "attachments": [
                                {
                                    "filename": "note.txt",
                                    "attachmentId": "attachment-1",
                                    "mimeType": "text/plain",
                                    "size": 15,
                                }
                            ],
                        }
                    ]
                }
            ),
            "",
            0.01,
        )

    monkeypatch.setattr(adapters, "is_tool_available", lambda name: name == "gog")
    monkeypatch.setattr(adapters, "run_tool", fake_run_tool)

    result = adapters.sync_self_sent_gmail(test_db, test_blob_store, download_attachments=True)

    assert result["attachments_saved"] == 1
    assert "--download" in calls[1]
    with test_db.connection() as conn:
        attachment = conn.execute("SELECT * FROM attachments").fetchone()
    assert attachment["file_name"] == "note.txt"
    assert test_blob_store.read_bytes(attachment["content_hash"]) == b"mail attachment"


def test_extract_author_thread_filters_strangers_and_walks_chain():
    """Verify bidirectional thread walk keeps only the author's self-replies in order."""
    tweets = [
        {
            "id": "100",
            "author": {"username": "alice"},
            "authorId": "1",
            "text": "1/ Thread start",
            "inReplyToStatusId": None,
        },
        {
            "id": "101",
            "author": {"username": "bob"},
            "authorId": "2",
            "text": "Random stranger reply",
            "inReplyToStatusId": "100",
        },
        {
            "id": "102",
            "author": {"username": "alice"},
            "authorId": "1",
            "text": "2/ Middle point",
            "inReplyToStatusId": "100",
        },
        {
            "id": "103",
            "author": {"username": "alice"},
            "authorId": "1",
            "text": "3/ Thread conclusion",
            "inReplyToStatusId": "102",
        },
    ]

    # Starting from middle tweet (102)
    chain = adapters._extract_author_thread(tweets, "102")
    assert [t["id"] for t in chain] == ["100", "102", "103"]
    assert "bob" not in [t["author"]["username"] for t in chain]

    # Stitched text
    stitched = adapters._format_thread_text(chain)
    assert "1/ Thread start\n\n---\n\n2/ Middle point\n\n---\n\n3/ Thread conclusion" == stitched


def test_x_sync_unrolls_thread_with_media_and_links(test_db, test_blob_store, monkeypatch):
    """Verify sync_x_bookmarks unrolls threads, collects multi-tweet media and links."""
    monkeypatch.setattr(
        adapters,
        "_read_birdclaw_bookmarks",
        lambda: [
            {
                "account_id": "acct",
                "tweet_id": "200",
                "collected_at": "2026-01-01T00:00:00Z",
                "source": "sync",
                "author_handle": "alice",
                "author_name": "Alice",
                "published_at": "2026-01-01T00:00:00Z",
                "text": "Part 1 https://example.com/paper",
                "entities_json": "{}",
                "media_json": "[]",
            }
        ],
    )

    thread_payload = [
        {
            "id": "200",
            "author": {"username": "alice"},
            "text": "Part 1 https://example.com/paper",
            "inReplyToStatusId": None,
            "entities": {"urls": [{"expanded_url": "https://example.com/paper"}]},
        },
        {
            "id": "201",
            "author": {"username": "alice"},
            "text": "Part 2 with image",
            "inReplyToStatusId": "200",
            "media": [{"url": "https://pbs.twimg.com/media/test_chart.jpg"}],
        },
    ]

    monkeypatch.setattr(adapters, "is_tool_available", lambda name: name in ("bird",))
    monkeypatch.setattr(
        adapters,
        "run_tool",
        lambda args, **_kw: SubprocessResult(0, json.dumps(thread_payload), "", 0.01),
    )

    result = adapters.sync_x_bookmarks(test_db, test_blob_store)
    assert result["captured"] == 1
    assert result["linked_urls"] == 1

    with test_db.connection() as conn:
        res = conn.execute(
            "SELECT clean_text, extractor FROM resource_contents ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
        assert res["extractor"] == "bird-thread"
        assert "Part 1" in res["clean_text"]
        assert "Part 2 with image" in res["clean_text"]
        assert "---" in res["clean_text"]


def test_repair_x_threads_command(test_db, test_blob_store, monkeypatch):
    """Verify repair_x_threads and CLI repair-threads expands single-tweet resources."""
    runner = CliRunner()

    # Seed a capture with single-tweet text
    with test_db.transaction() as conn:
        captured = capture_item(
            conn,
            CaptureInput(
                url="https://x.com/i/status/300",
                origin_namespace="x",
                origin_id="300",
                collection_channel="birdclaw",
                collector="test",
                acquisition_method="test",
            ),
        )
        res_id = captured["resource_id"]
        adapters.store_resource_content(
            conn, res_id, "Part 1 only", extractor="bird", title="Part 1 only"
        )

    thread_payload = [
        {
            "id": "300",
            "author": {"username": "alice"},
            "text": "Part 1 of the grand theory",
            "inReplyToStatusId": None,
        },
        {
            "id": "301",
            "author": {"username": "alice"},
            "text": "Part 2 of the grand theory",
            "inReplyToStatusId": "300",
        },
    ]

    monkeypatch.setattr(adapters, "is_tool_available", lambda name: name == "bird")
    monkeypatch.setattr(
        adapters,
        "run_tool",
        lambda args, **_kw: SubprocessResult(0, json.dumps(thread_payload), "", 0.01),
    )
    monkeypatch.setattr("edward.cli.get_services", lambda: (test_db, test_blob_store))

    # Test CLI dry run
    cli_dry = runner.invoke(app, ["repair-threads", "--dry-run", "--json"])
    assert cli_dry.exit_code == 0, cli_dry.output
    dry_data = json.loads(cli_dry.stdout)
    assert dry_data["threads_found"] == 1
    assert dry_data["threads_expanded"] == 0

    # Test CLI execute
    cli_run = runner.invoke(app, ["repair-threads", "--json"])
    assert cli_run.exit_code == 0, cli_run.output
    run_data = json.loads(cli_run.stdout)
    assert run_data["threads_found"] == 1
    assert run_data["threads_expanded"] == 1

    with test_db.connection() as conn:
        res = conn.execute(
            "SELECT clean_text, extractor FROM resource_contents WHERE resource_id = ? ORDER BY created_at DESC LIMIT 1",
            (res_id,),
        ).fetchone()
        assert res["extractor"] == "bird-thread"
        assert (
            "Part 1 of the grand theory\n\n---\n\nPart 2 of the grand theory" == res["clean_text"]
        )
