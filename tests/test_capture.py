"""Tests for capture and ingestion service."""

import pytest

from edward.db import Database
from edward.models import CaptureInput
from edward.services.capture import (
    IdempotencyConflictError,
    canonicalize_url,
    capture_item,
)


def test_url_canonicalization():
    url1 = "https://Example.COM/Path/?utm_source=twitter&utm_medium=social&id=123"
    canon = canonicalize_url(url1)
    assert canon == "https://example.com/Path?id=123"

    url2 = "http://test.org/trailing/slash/"
    assert canonicalize_url(url2) == "http://test.org/trailing/slash"


def test_capture_url_creates_resource_and_capture(test_db: Database):
    with test_db.transaction() as conn:
        inp = CaptureInput(
            url="https://arxiv.org/abs/2401.00000",
            note="Important paper",
            collector="agent-1",
        )
        res = capture_item(conn, inp)

    assert res["status"] == "created"
    assert res["capture_id"].startswith("cap_")
    assert res["resource_id"].startswith("res_")

    with test_db.connection() as conn:
        cap = conn.execute("SELECT * FROM captures WHERE id = ?;", (res["capture_id"],)).fetchone()
        assert cap["user_note"] == "Important paper"
        assert cap["collector"] == "agent-1"

        resource = conn.execute(
            "SELECT * FROM resources WHERE id = ?;", (res["resource_id"],)
        ).fetchone()
        assert resource["canonical_url"] == "https://arxiv.org/abs/2401.00000"


def test_multiple_captures_for_same_resource(test_db: Database):
    with test_db.transaction() as conn:
        inp1 = CaptureInput(
            url="https://example.com/blog/article",
            note="First observation",
            collector="user",
        )
        res1 = capture_item(conn, inp1)

    with test_db.transaction() as conn:
        inp2 = CaptureInput(
            url="https://example.com/blog/article",
            note="Second observation from mobile",
            collector="agent",
        )
        res2 = capture_item(conn, inp2)

    # Different captures, same resource
    assert res1["capture_id"] != res2["capture_id"]
    assert res1["resource_id"] == res2["resource_id"]

    with test_db.connection() as conn:
        caps = conn.execute("SELECT COUNT(*) FROM captures;").fetchone()[0]
        resources = conn.execute("SELECT COUNT(*) FROM resources;").fetchone()[0]
        assert caps == 2
        assert resources == 1


def test_idempotency_key_replay_and_conflict(test_db: Database):
    key = "unique-client-key-1"

    # First attempt: succeeds
    with test_db.transaction() as conn:
        inp1 = CaptureInput(
            url="https://github.com/org/repo",
            note="Initial note",
            idempotency_key=key,
        )
        res1 = capture_item(conn, inp1)
    assert res1["status"] == "created"

    # Second attempt with identical parameters: replay
    with test_db.transaction() as conn:
        inp_same = CaptureInput(
            url="https://github.com/org/repo",
            note="Initial note",
            idempotency_key=key,
        )
        res_replayed = capture_item(conn, inp_same)
    assert res_replayed["status"] == "replayed"
    assert res_replayed["capture_id"] == res1["capture_id"]

    # Third attempt with mismatched parameters: conflict error
    with test_db.transaction() as conn:
        inp_conflict = CaptureInput(
            url="https://github.com/org/repo",
            note="DIFFERENT NOTE",
            idempotency_key=key,
        )
        with pytest.raises(IdempotencyConflictError):
            capture_item(conn, inp_conflict)


def test_url_validation_rejections():
    # Unsupported schemes
    with pytest.raises(ValueError, match="only http and https are supported"):
        canonicalize_url("ftp://example.com/file")
    with pytest.raises(ValueError, match="only http and https are supported"):
        canonicalize_url("javascript:alert(1)")

    # Missing scheme / bare domains
    with pytest.raises(ValueError, match="only http and https are supported"):
        canonicalize_url("example.com/path")

    # Invalid hostname
    with pytest.raises(ValueError, match="Invalid URL hostname"):
        canonicalize_url("https:///example.com")


def test_non_url_resource_and_attachment(test_db: Database):
    with test_db.transaction() as conn:
        att = {
            "file_name": "document.pdf",
            "mime_type": "application/pdf",
            "content_hash": "abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890",
            "size_bytes": 1024,
            "blob_path": "/path/to/blob",
        }
        inp = CaptureInput(note="Uploaded research PDF", collector="agent")
        res = capture_item(conn, inp, attachment_info=att)

    assert res["status"] == "created"
    assert res["resource_id"] is not None

    with test_db.connection() as conn:
        resource = conn.execute(
            "SELECT * FROM resources WHERE id = ?;", (res["resource_id"],)
        ).fetchone()
        assert resource["identity_key"] == f"blob:{att['content_hash']}"
        assert resource["canonical_url"] is None
        assert resource["url_hash"] is None
        assert resource["title"] == "document.pdf"

        attachment = conn.execute(
            "SELECT * FROM attachments WHERE object_id = ?;", (res["resource_id"],)
        ).fetchone()
        assert attachment["file_name"] == "document.pdf"
        assert attachment["mime_type"] == "application/pdf"
        assert attachment["content_hash"] == att["content_hash"]
        assert attachment["blob_path"] == f"{att['content_hash'][:2]}/{att['content_hash']}"


def test_file_idempotency_conflict(test_db: Database):
    key = "file-idemp-test-key-1"
    att1 = {
        "file_name": "data.csv",
        "mime_type": "text/csv",
        "content_hash": "1111111111111111111111111111111111111111111111111111111111111111",
        "size_bytes": 100,
        "blob_path": "11/1111111111111111111111111111111111111111111111111111111111111111",
    }
    att2 = {
        "file_name": "data.csv",
        "mime_type": "text/csv",
        "content_hash": "2222222222222222222222222222222222222222222222222222222222222222",
        "size_bytes": 120,
        "blob_path": "22/2222222222222222222222222222222222222222222222222222222222222222",
    }

    # Initial capture with file 1
    with test_db.transaction() as conn:
        inp1 = CaptureInput(note="Dataset upload", idempotency_key=key)
        res1 = capture_item(conn, inp1, attachment_info=att1)
    assert res1["status"] == "created"

    # Exact replay with same file 1
    with test_db.transaction() as conn:
        inp_same = CaptureInput(note="Dataset upload", idempotency_key=key)
        res_replayed = capture_item(conn, inp_same, attachment_info=att1)
    assert res_replayed["status"] == "replayed"
    assert res_replayed["capture_id"] == res1["capture_id"]

    # Replay with same key but different file bytes (different content_hash): conflict
    with test_db.transaction() as conn:
        inp_diff_file = CaptureInput(note="Dataset upload", idempotency_key=key)
        with pytest.raises(IdempotencyConflictError):
            capture_item(conn, inp_diff_file, attachment_info=att2)


def test_capture_context_preservation_in_fts(test_db: Database):
    from edward.services.search import search_lexical

    # Two captures of the same URL with different notes
    url = "https://example.com/research-topic"
    with test_db.transaction() as conn:
        cap1 = capture_item(conn, CaptureInput(url=url, note="First note on quantum speedup"))
        cap2 = capture_item(conn, CaptureInput(url=url, note="Second note on error correction"))

    with test_db.connection() as conn:
        # Both individual captures must be independently searchable
        res1 = search_lexical(conn, "quantum")
        assert any(r.id == cap1["capture_id"] for r in res1.results)

        res2 = search_lexical(conn, "correction")
        assert any(r.id == cap2["capture_id"] for r in res2.results)
