"""Tests for FTS5 lexical search projection and indexing."""

from typer.testing import CliRunner

from edward.cli import app
from edward.db import Database
from edward.models import CaptureInput
from edward.services.capture import capture_item
from edward.services.search import (
    index_document,
    rebuild_search_index,
    remove_document,
    search_lexical,
)


def test_fts_indexing_and_search(test_db: Database):
    with test_db.transaction() as conn:
        index_document(
            conn,
            object_type="resource",
            object_id="res_test_1",
            title="Speculative Decoding for Local LLMs",
            body="Empirical latency benchmark showing 2.5x speedup using llama.cpp.",
            labels=["ai", "benchmark"],
        )

    with test_db.connection() as conn:
        # Search exact word
        res = search_lexical(conn, "Speculative")
        assert res.count == 1
        assert res.results[0].id == "res_test_1"
        assert "Speculative" in res.results[0].title

        # Search snippet
        res_bench = search_lexical(conn, "benchmark")
        assert res_bench.count == 1
        assert "<b>benchmark</b>" in res_bench.results[0].snippet


def test_cli_fts_reindex_preserves_page_chunks(test_db: Database, monkeypatch):
    monkeypatch.setenv("EDWARD_DB_PATH", str(test_db.db_path))
    with test_db.transaction() as conn:
        conn.execute(
            """INSERT INTO resources (id, identity_key, canonical_url, title, review_state,
                is_deleted, created_at, updated_at)
                VALUES ('res_pdf', 'url:pdf', 'https://example.com/guide.pdf', 'Guide',
                'unreviewed', 0, '2026-01-01', '2026-01-01');"""
        )
        conn.execute(
            """INSERT INTO resource_contents (id, resource_id, content_hash, clean_text,
                char_count, extractor, extractor_version, created_at)
                VALUES ('rc_pdf', 'res_pdf', 'hash', 'Introduction', 12, 'pypdf', '1', '2026-01-01');"""
        )
        conn.execute(
            """INSERT INTO resource_chunks (id, resource_content_id, resource_id,
                chunk_index, text, locator_json, token_count, created_at)
                VALUES ('chk_page_25', 'rc_pdf', 'res_pdf', 25, 'Routstr uses Cashu tokens',
                '{"page": 25}', 5, '2026-01-01');"""
        )
        for family, label in (("form", "documentation"), ("topic", "ai")):
            conn.execute(
                "INSERT OR IGNORE INTO label_families (id, created_at) VALUES (?, '2026-01-01');",
                (family,),
            )
            conn.execute(
                """INSERT OR IGNORE INTO labels (id, family, version, created_at)
                    VALUES (?, ?, '1', '2026-01-01');""",
                (label, family),
            )
            conn.execute(
                """INSERT INTO object_labels (id, object_type, object_id, label_id,
                    source, created_at) VALUES (?, 'resource', 'res_pdf', ?, 'classifier', '2026-01-01');""",
                (f"lbl_{label}", label),
            )

    result = CliRunner().invoke(app, ["reindex", "--fts", "--json"])
    assert result.exit_code == 0
    with test_db.connection() as conn:
        hits = search_lexical(conn, "Routstr Cashu")
        by_form = search_lexical(conn, "Routstr Cashu", form="documentation")
        by_topic = search_lexical(conn, "Routstr Cashu", topic="ai")
    assert any(hit.id == "chk_page_25" for hit in hits.results)
    assert any(hit.id == "chk_page_25" for hit in by_form.results)
    assert any(hit.id == "chk_page_25" for hit in by_topic.results)


def test_research_lexical_search_can_match_any_content_term(test_db: Database):
    """Broad research retrieval can recall related sources without changing exact search."""
    with test_db.transaction() as conn:
        index_document(conn, "resource", "res_local", body="A local inference guide")
        index_document(conn, "resource", "res_models", body="A guide to small models")

    with test_db.connection() as conn:
        exact = search_lexical(conn, "local models")
        broad = search_lexical(conn, "local models", match_any=True)

    assert exact.count == 0
    assert {item.id for item in broad.results} == {"res_local", "res_models"}


def test_fts_filter_by_intent(test_db: Database):
    with test_db.transaction() as conn:
        inp1 = CaptureInput(
            url="https://site1.org", text="Local model quantization", intent="essay-seed"
        )
        r1 = capture_item(conn, inp1)

        inp2 = CaptureInput(url="https://site2.org", text="Another local model note")
        capture_item(conn, inp2)

    with test_db.connection() as conn:
        # Search with intent filter
        res_filtered = search_lexical(conn, "model", intent="essay-seed")
        assert res_filtered.count == 1
        assert res_filtered.results[0].id in (r1["resource_id"], r1["capture_id"])

        # Search without intent filter
        res_all = search_lexical(conn, "model")
        assert res_all.count >= 2


def test_x_capture_and_primary_resource_are_one_lexical_result(test_db: Database):
    with test_db.transaction() as conn:
        captured = capture_item(
            conn,
            CaptureInput(
                url="https://x.com/i/status/12345",
                text="A bookmark about semantic search and local models",
                origin_namespace="x",
                origin_id="12345",
                collection_channel="birdclaw",
                collector="test",
                acquisition_method="test",
            ),
        )

    with test_db.connection() as conn:
        result = search_lexical(conn, "semantic search local models")
        stored_capture = conn.execute(
            "SELECT 1 FROM captures WHERE id = ?;", (captured["capture_id"],)
        ).fetchone()

    assert stored_capture is not None
    assert result.count == 1
    assert result.results[0].id == captured["resource_id"]
    assert result.results[0].object_type == "resource"


def test_x_capture_with_user_note_remains_separately_searchable(test_db: Database):
    with test_db.transaction() as conn:
        captured = capture_item(
            conn,
            CaptureInput(
                url="https://x.com/i/status/67890",
                text="A bookmark about semantic search",
                note="My separate thought about local models",
                origin_namespace="x",
                origin_id="67890",
                collection_channel="birdclaw",
                collector="test",
                acquisition_method="test",
            ),
        )

    with test_db.connection() as conn:
        result = search_lexical(conn, "local models")

    assert result.count == 1
    assert result.results[0].id == captured["capture_id"]


def test_fts_index_lifecycle_and_rebuild(test_db: Database):
    with test_db.transaction() as conn:
        index_document(conn, "resource", "res_temp", title="Temporary Title", body="Body text")

    with test_db.connection() as conn:
        assert search_lexical(conn, "Temporary").count == 1

    with test_db.transaction() as conn:
        remove_document(conn, "resource", "res_temp")

    with test_db.connection() as conn:
        assert search_lexical(conn, "Temporary").count == 0

    # Test rebuild
    with test_db.transaction() as conn:
        rebuild_count = rebuild_search_index(conn)
        assert isinstance(rebuild_count, int)


def test_fts_syntax_error_handled_safely(test_db: Database):
    with test_db.connection() as conn:
        # Unclosed quote or syntax characters in user query should not crash
        res = search_lexical(conn, '"unclosed quote query')
        assert res.count == 0
        assert res.query == '"unclosed quote query'


def test_database_operational_error_reraised(test_db: Database):
    import sqlite3

    import pytest

    with test_db.transaction() as conn:
        # Drop the underlying search projection table
        conn.execute("DROP TABLE search_documents;")

    with test_db.connection() as conn:
        # Real database/schema error must be raised, not swallowed as empty search
        with pytest.raises(sqlite3.OperationalError):
            search_lexical(conn, "anything")


def test_fts_content_version_determinism(test_db: Database):
    from edward.services.lifecycle import reindex_object_document

    res_id = "res_version_test"
    finding_id = "fin_version_test"

    with test_db.transaction() as conn:
        # Insert a resource
        conn.execute(
            """
            INSERT INTO resources (id, identity_key, title, created_at, updated_at)
            VALUES (?, 'ident_version', 'Versioned Resource', '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z');
            """,
            (res_id,),
        )
        # Insert an older version of content
        conn.execute(
            """
            INSERT INTO resource_contents (id, resource_id, content_hash, clean_text, summary, char_count, extractor, extractor_version, created_at)
            VALUES ('rc_old', ?, 'hash_old', 'obsolete ancient information alpha', 'Old summary', 35, 'test', '1.0', '2026-01-01T00:00:00Z');
            """,
            (res_id,),
        )
        # Insert a newer version of content
        conn.execute(
            """
            INSERT INTO resource_contents (id, resource_id, content_hash, clean_text, summary, char_count, extractor, extractor_version, created_at)
            VALUES ('rc_new', ?, 'hash_new', 'fresh latest information beta', 'New summary', 31, 'test', '1.0', '2026-02-01T00:00:00Z');
            """,
            (res_id,),
        )

        # Insert a finding with multiple support passages
        conn.execute(
            """
            INSERT INTO findings (id, statement, assertion_role, created_at, updated_at)
            VALUES (?, 'Finding with multiple passages', 'source-claim', '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z');
            """,
            (finding_id,),
        )
        conn.execute(
            """
            INSERT INTO finding_support (id, finding_id, passage, created_at)
            VALUES ('fs_1', ?, 'supporting snippet gamma', '2026-01-01T10:00:00Z');
            """,
            (finding_id,),
        )
        conn.execute(
            """
            INSERT INTO finding_support (id, finding_id, passage, created_at)
            VALUES ('fs_2', ?, 'supporting snippet delta', '2026-01-01T11:00:00Z');
            """,
            (finding_id,),
        )

    # Rebuild entire search index
    with test_db.transaction() as conn:
        rebuild_search_index(conn)

    with test_db.connection() as conn:
        # Obsolete content from older resource_contents must NOT be in the index
        assert search_lexical(conn, "obsolete").count == 0
        assert search_lexical(conn, "ancient").count == 0

        # Fresh content from latest resource_contents MUST be in the index
        fresh_res = search_lexical(conn, "fresh")
        assert fresh_res.count == 1
        assert fresh_res.results[0].id == res_id

        # Both finding passages must be indexed
        assert search_lexical(conn, "gamma").count == 1
        assert search_lexical(conn, "delta").count == 1

        # Verify ordering in search_documents body
        doc_row = conn.execute(
            "SELECT body FROM search_documents WHERE object_id = ?;", (finding_id,)
        ).fetchone()
        assert doc_row is not None
        assert "supporting snippet gamma\n\nsupporting snippet delta" in doc_row["body"]

    # Test single-object reindex_object_document as well
    with test_db.transaction() as conn:
        reindex_object_document(conn, "resource", res_id)

    with test_db.connection() as conn:
        assert search_lexical(conn, "obsolete").count == 0
        assert search_lexical(conn, "fresh").count == 1
