"""Tests for research bundle v1 and Markdown report ingestion."""

import pytest

from edward.blobs import BlobStore
from edward.db import Database
from edward.services.bundle import import_research_bundle, ingest_markdown_report
from edward.services.capture import IdempotencyConflictError
from edward.services.search import search_lexical


@pytest.fixture
def test_db_and_blobs(tmp_path):
    db_file = tmp_path / "test_edward.db"
    blobs_dir = tmp_path / "blobs"
    db = Database(db_file)
    db.run_migrations()
    blob_store = BlobStore(blobs_dir)
    return db, blob_store


def test_import_research_bundle_full(test_db_and_blobs):
    db, blob_store = test_db_and_blobs

    bundle_data = {
        "type": "research-bundle",
        "schema_version": "1",
        "bundle_id": "bun_phase2_01",
        "idempotency_key": "bundle-key-101",
        "title": "Scaling Laws in Machine Learning",
        "brief": "Analysis of compute-optimal models and Chinchilla scaling.",
        "agent": {
            "name": "deep-researcher",
            "run_id": "run-456",
            "model": "gpt-4o",
        },
        "sources": [
            {
                "origin": "web",
                "url": "https://example.org/chinchilla-paper",
                "title": "Training Compute-Optimal Large Language Models",
                "snapshot": "Raw HTML or snapshot markdown here...",
                "extracted_markdown": "Chinchilla shows that 70B models need 1.4T tokens.",
            }
        ],
        "findings": [
            {
                "statement": "Current LLMs are significantly undertrained relative to compute budget.",
                "assertion_role": "source-claim",
                "source_url": "https://example.org/chinchilla-paper",
                "confidence": 0.95,
                "intents": ["deep-dive", "essay-seed"],
                "support": [
                    {
                        "passage": "Given an 8x increase in compute, tokens and parameters should scale equally.",
                        "locator": {"section": "Abstract", "page": 1},
                    }
                ],
            }
        ],
    }

    with db.transaction() as conn:
        res = import_research_bundle(conn, blob_store, bundle_data)

    assert res["status"] == "imported"
    assert res["bundle_id"] == "bun_phase2_01"
    cap_id = res["capture_id"]
    assert cap_id.startswith("cap_")

    # Verify DB contents
    with db.connection() as conn:
        # Check capture
        c_row = conn.execute("SELECT * FROM captures WHERE id = ?;", (cap_id,)).fetchone()
        assert c_row is not None
        assert c_row["collector"] == "deep-researcher"
        assert c_row["collector_run_id"] == "run-456"

        # Check resource
        r_rows = conn.execute(
            "SELECT * FROM resources WHERE canonical_url = 'https://example.org/chinchilla-paper';"
        ).fetchall()
        assert len(r_rows) == 1
        res_id = r_rows[0]["id"]

        # Check content and snapshot
        rc = conn.execute(
            "SELECT * FROM resource_contents WHERE resource_id = ?;", (res_id,)
        ).fetchone()
        assert rc is not None
        assert "Chinchilla shows" in rc["clean_text"]

        ss = conn.execute(
            "SELECT * FROM source_snapshots WHERE resource_id = ?;", (res_id,)
        ).fetchone()
        assert ss is not None
        assert ss["blob_path"] is not None
        # Verify blob readable from store
        blob_bytes = blob_store.read_bytes(ss["content_hash"])
        assert blob_bytes == b"Raw HTML or snapshot markdown here..."

        # Check findings and support
        f_rows = conn.execute("SELECT * FROM findings;").fetchall()
        assert len(f_rows) == 1
        f_id = f_rows[0]["id"]
        assert f_rows[0]["statement"].startswith("Current LLMs are significantly undertrained")

        sup_rows = conn.execute(
            "SELECT * FROM finding_support WHERE finding_id = ?;", (f_id,)
        ).fetchall()
        assert len(sup_rows) == 1
        assert "Given an 8x increase" in sup_rows[0]["passage"]

        # Check intents
        intent_rows = conn.execute("SELECT * FROM intents WHERE object_id = ?;", (f_id,)).fetchall()
        intents = {row["intent"] for row in intent_rows}
        assert "deep-dive" in intents
        assert "essay-seed" in intents

        # Check FTS index
        s_res = search_lexical(conn, "Chinchilla")
        assert len(s_res.results) > 0


def test_bundle_idempotency_replay_and_conflict(test_db_and_blobs):
    db, blob_store = test_db_and_blobs

    bundle_1 = {
        "type": "research-bundle",
        "schema_version": "1",
        "bundle_id": "bun_001",
        "idempotency_key": "idemp-key-xyz",
        "title": "Quantum Supremacy",
        "sources": [{"origin": "web", "url": "https://example.org/quantum"}],
    }

    # First import
    with db.transaction() as conn:
        res1 = import_research_bundle(conn, blob_store, bundle_1)
    assert res1["status"] == "imported"

    # Exact replay
    with db.transaction() as conn:
        res2 = import_research_bundle(conn, blob_store, bundle_1)
    assert res2["status"] == "replayed"
    assert res2["capture_id"] == res1["capture_id"]

    # Conflict with altered title
    bundle_conflict = dict(bundle_1)
    bundle_conflict["title"] = "Different Quantum Supremacy Title"

    with pytest.raises(IdempotencyConflictError, match="already used with different content"):
        with db.transaction() as conn:
            import_research_bundle(conn, blob_store, bundle_conflict)


def test_ingest_markdown_report(test_db_and_blobs):
    db, _ = test_db_and_blobs
    md_content = """# State of AI Hardware 2026

Modern accelerators utilize optical interconnects and wafer-scale integration.

## Key Observations
- Energy efficiency has improved by 4x per generation.
- Memory bandwidth remains the primary constraint for inference workloads.
"""

    with db.transaction() as conn:
        res = ingest_markdown_report(
            conn,
            md_content,
            idempotency_key="report-key-1",
            collector="evaluator",
        )

    assert res["status"] == "imported"
    assert res["title"] == "State of AI Hardware 2026"
    assert res["resource_id"].startswith("res_")

    with db.connection() as conn:
        r_row = conn.execute(
            "SELECT * FROM resources WHERE id = ?;", (res["resource_id"],)
        ).fetchone()
        assert r_row["primary_form"] == "research-report"

        rc = conn.execute(
            "SELECT * FROM resource_contents WHERE resource_id = ?;", (res["resource_id"],)
        ).fetchone()
        assert rc is not None
        assert "wafer-scale integration" in rc["clean_text"]

        # Searchable in FTS
        search_res = search_lexical(conn, "accelerators")
        assert len(search_res.results) > 0

    # Idempotency replay
    with db.transaction() as conn:
        res_replay = ingest_markdown_report(
            conn,
            md_content,
            idempotency_key="report-key-1",
        )
    assert res_replay["status"] == "replayed"
    assert res_replay["resource_id"] == res["resource_id"]

    # Idempotency conflict
    with pytest.raises(IdempotencyConflictError):
        with db.transaction() as conn:
            ingest_markdown_report(
                conn,
                md_content + "\nExtra paragraph",
                idempotency_key="report-key-1",
            )
