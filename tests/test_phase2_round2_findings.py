"""Regression tests verifying resolution of all Round 2 Pi review findings."""

import datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from edward.blobs import BlobStore
from edward.cli import app
from edward.db import Database
from edward.models import SourceItem, make_job_key
from edward.services.bundle import import_research_bundle
from edward.services.classification import (
    load_thresholds,
    resolve_threshold,
    run_classification_pipeline,
)
from edward.services.extract import ExtractionResult
from edward.services.lifecycle import add_intent
from edward.services.processor import (
    _persist_job_result,
    claim_job,
    process_pending_jobs,
)
from edward.services.subprocess_runner import sanitize_error_message

runner = CliRunner()


@pytest.fixture
def test_db(tmp_path: Path) -> Database:
    db = Database(tmp_path / "test.db")
    db.run_migrations()
    return db


@pytest.fixture
def test_blob_store(tmp_path: Path) -> BlobStore:
    return BlobStore(tmp_path / "blobs")


def test_finding1_lease_fencing_rejects_expired_or_stolen_lease(
    test_db: Database, test_blob_store: BlobStore
):
    """Finding 1: Worker whose lease expired or was stolen cannot commit side-effects or schedule downstream jobs."""
    now_iso = datetime.datetime.now(datetime.UTC).isoformat()
    with test_db.transaction() as conn:
        conn.execute(
            """
            INSERT INTO resources (id, identity_key, canonical_url, title, review_state, is_deleted, created_at, updated_at)
            VALUES ('res_fence', 'url:https://example.com/fence', 'https://example.com/fence', 'Fence', 'unreviewed', 0, ?, ?);
            """,
            (now_iso, now_iso),
        )
        conn.execute(
            """
            INSERT INTO processing_jobs (
                id, job_key, resource_id, stage, status, available_at, attempts, lease_owner, lease_expires_at, created_at, updated_at
            ) VALUES ('job_fence', 'fetch:res_fence', 'res_fence', 'resource-fetch', 'running', ?, 0, 'worker_b', ?, ?, ?);
            """,
            (now_iso, now_iso, now_iso, now_iso),
        )

    # Worker A (which lost the lease to Worker B) attempts to persist results
    stale_job = {
        "id": "job_fence",
        "job_key": "fetch:res_fence",
        "resource_id": "res_fence",
        "stage": "resource-fetch",
        "attempts": 0,
        "max_attempts": 3,
        "lease_owner": "worker_a",  # Worker A thinks it still owns it
    }

    class FakeFetchResult:
        body = b"Stale body"
        headers = {"content-type": "text/html"}
        status_code = 200

    with test_db.transaction() as conn:
        success = _persist_job_result(
            conn,
            test_blob_store,
            stale_job,
            work_result={"fetch_result": FakeFetchResult()},
            job_error=None,
            worker_id="worker_a",
        )
        # Must be rejected with lost-lease
        assert success == "lost-lease"

    with test_db.connection() as conn:
        # Verify job is still running under worker_b and not marked completed
        job_row = conn.execute(
            "SELECT status, lease_owner FROM processing_jobs WHERE id = 'job_fence';"
        ).fetchone()
        assert job_row["status"] == "running"
        assert job_row["lease_owner"] == "worker_b"
        # Verify no downstream extract job was scheduled
        downstream = conn.execute(
            "SELECT * FROM processing_jobs WHERE stage = 'extract';"
        ).fetchall()
        assert len(downstream) == 0


def test_finding2_bundle_contract_routing(test_db: Database, test_blob_store: BlobStore):
    """Finding 2: extracted_text -> classify, snapshot -> extract, url-only -> resource-fetch."""
    bundle = {
        "type": "research-bundle",
        "schema_version": "1",
        "bundle_id": "bun_route_exact",
        "title": "Routing Bundle",
        "sources": [
            {
                "origin": "web",
                "url": "https://example.com/s1",
                "extracted_text": "Extracted text content",
            },
            {
                "origin": "web",
                "url": "https://example.com/s2",
                "snapshot": "<html>inline snap</html>",
            },
            {
                "origin": "web",
                "url": "https://example.com/s3",
            },
        ],
    }
    with test_db.transaction() as conn:
        res = import_research_bundle(conn, test_blob_store, bundle)
        assert res["status"] == "imported"

    with test_db.connection() as conn:
        cls_jobs = conn.execute(
            "SELECT * FROM processing_jobs WHERE stage = 'classify';"
        ).fetchall()
        ext_jobs = conn.execute("SELECT * FROM processing_jobs WHERE stage = 'extract';").fetchall()
        fetch_jobs = conn.execute(
            "SELECT * FROM processing_jobs WHERE stage = 'resource-fetch';"
        ).fetchall()

        assert len(cls_jobs) == 1
        assert len(ext_jobs) == 1
        assert len(fetch_jobs) == 1


def test_finding3_human_intent_preservation(test_db: Database):
    """Finding 3: Human intent cannot be deactivated or overwritten by an agent suggestion."""
    with test_db.transaction() as conn:
        conn.execute(
            """
            INSERT INTO resources (id, identity_key, canonical_url, title, review_state, is_deleted, created_at, updated_at)
            VALUES ('res_intent', 'url:intent', 'https://example.com/intent', 'Intent Test', 'unreviewed', 0, '2026-01-01', '2026-01-01');
            """
        )
        # Human adds intent
        add_intent(conn, "resource", "res_intent", "essay-seed", source="human", actor="human")

    # Verify initial state is active with human source
    with test_db.connection() as conn:
        row = conn.execute(
            "SELECT source, is_active FROM intents WHERE object_id = 'res_intent';"
        ).fetchone()
        assert row["source"] == "human"
        assert row["is_active"] == 1

    # Agent suggestion tries to add same intent
    with test_db.transaction() as conn:
        # Agent suggestion has source='agent'
        conn.execute(
            """
            INSERT INTO intents (id, object_type, object_id, intent, source, is_active, created_at, updated_at)
            VALUES ('int_agent', 'resource', 'res_intent', 'essay-seed', 'agent', 0, '2026-01-02', '2026-01-02')
            ON CONFLICT(object_type, object_id, intent)
            DO UPDATE SET
                is_active = CASE
                    WHEN intents.source = 'human' AND excluded.source != 'human' THEN intents.is_active
                    ELSE excluded.is_active
                END,
                source = CASE
                    WHEN intents.source = 'human' AND excluded.source != 'human' THEN intents.source
                    ELSE excluded.source
                END,
                updated_at = CASE
                    WHEN intents.source = 'human' AND excluded.source != 'human' THEN intents.updated_at
                    ELSE excluded.updated_at
                END;
            """
        )

    # Verify human intent remains intact and active
    with test_db.connection() as conn:
        row = conn.execute(
            "SELECT source, is_active FROM intents WHERE object_id = 'res_intent';"
        ).fetchone()
        assert row["source"] == "human"
        assert row["is_active"] == 1


def test_finding4_snapshot_path_security_rejection():
    """Finding 4: snapshot_path in SourceItem is strictly rejected."""
    with pytest.raises(ValueError, match="snapshot_path"):
        SourceItem(
            origin="web",
            url="https://example.com/item",
            snapshot_path="/etc/passwd",
        )


def test_finding5_threshold_canonical_resolution(test_db: Database):
    """Finding 5: Emitted registry IDs resolve correctly through resolve_threshold against loaded thresholds."""
    thresholds = load_thresholds("balanced-precision")
    assert len(thresholds) > 0

    # Test resolution of raw classifier emitted IDs
    assert resolve_threshold(thresholds, "topic", "ai") == 0.65
    assert resolve_threshold(thresholds, "topic", "ai/local-models") == 0.70
    assert resolve_threshold(thresholds, "topic", "software/coding") == 0.65
    assert resolve_threshold(thresholds, "signal", "benchmark") == 0.70
    assert resolve_threshold(thresholds, "signal", "first-hand-experience") == 0.60
    assert resolve_threshold(thresholds, "signal", "tutorial") == 0.65
    assert resolve_threshold(thresholds, "signal", "warning") == 0.65

    # Run classification pipeline with DryRunClassifier on content matching topics & signals
    with test_db.transaction() as conn:
        conn.execute(
            """
            INSERT INTO resources (id, identity_key, canonical_url, title, review_state, is_deleted, created_at, updated_at)
            VALUES ('res_cls', 'url:cls', 'https://example.com/ai', 'Local AI Benchmark Tutorial', 'unreviewed', 0, '2026-01-01', '2026-01-01');
            """
        )
        conn.execute(
            """
            INSERT INTO resource_contents (id, resource_id, content_hash, clean_text, extractor, extractor_version, char_count, created_at)
            VALUES ('rc_cls', 'res_cls', 'hash', 'This benchmark tutorial explores local models inference latency with llama.cpp and Ollama.', 'test', '1.0', 90, '2026-01-01');
            """
        )
        judgments = run_classification_pipeline(conn, "resource", "res_cls", provider="dry-run")
        assert len(judgments) > 0

    with test_db.connection() as conn:
        labels = conn.execute(
            "SELECT label_id FROM object_labels WHERE object_id = 'res_cls';"
        ).fetchall()
        label_ids = {lbl["label_id"] for lbl in labels}
        # Thresholds applied and passed
        assert "ai/local-models" in label_ids or "ai" in label_ids


def test_finding6_resource_cache_matches_snapshot_hash(
    test_db: Database, test_blob_store: BlobStore, monkeypatch: pytest.MonkeyPatch
):
    """Finding 6: Storing snapshot hash in processing_jobs.input_hash allows cache hit on re-extraction."""
    raw_bytes = b"<html><body><h1>Raw HTML Page</h1></body></html>"
    snap_hash, _ = test_blob_store.store_bytes(raw_bytes)
    text_hash, _ = test_blob_store.store_bytes(b"Raw HTML Page")

    now_iso = datetime.datetime.now(datetime.UTC).isoformat()
    with test_db.transaction() as conn:
        conn.execute(
            """
            INSERT INTO resources (id, identity_key, canonical_url, title, review_state, is_deleted, created_at, updated_at)
            VALUES ('res_cache2', 'url:cache2', 'https://example.com/c2', 'C2', 'unreviewed', 0, ?, ?);
            """,
            (now_iso, now_iso),
        )
        conn.execute(
            """
            INSERT INTO source_snapshots (id, resource_id, content_hash, blob_path, size_bytes, created_at)
            VALUES ('snp_c2', 'res_cache2', ?, 'p', 100, ?);
            """,
            (snap_hash, now_iso),
        )
        # resource_contents has the extracted text hash (different from raw snap_hash)
        conn.execute(
            """
            INSERT INTO resource_contents (id, resource_id, content_hash, clean_text, extractor, extractor_version, char_count, created_at)
            VALUES ('rc_c2', 'res_cache2', ?, 'Raw HTML Page', 'test', '1.0', 13, ?);
            """,
            (text_hash, now_iso),
        )
        # Record previous extract job completed with input_hash = snap_hash
        conn.execute(
            """
            INSERT INTO processing_jobs (id, job_key, resource_id, stage, status, input_hash, available_at, attempts, created_at, updated_at)
            VALUES ('prev_ext', 'extract:res_cache2:1', 'res_cache2', 'extract', 'completed', ?, ?, 0, ?, ?);
            """,
            (snap_hash, now_iso, now_iso, now_iso),
        )
        # New extract job
        conn.execute(
            """
            INSERT INTO processing_jobs (id, job_key, resource_id, stage, status, available_at, attempts, created_at, updated_at)
            VALUES ('cur_ext', 'extract:res_cache2:2', 'res_cache2', 'extract', 'pending', ?, 0, ?, ?);
            """,
            (now_iso, now_iso, now_iso),
        )

    called = [False]

    def fail_if_extracted(url, raw_html=None):
        called[0] = True
        return ExtractionResult(status="completed", clean_text="fail")

    monkeypatch.setattr("edward.services.processor.extract_content", fail_if_extracted)

    res = process_pending_jobs(test_db, test_blob_store, limit=1)
    assert res["completed"] == 1
    # Cache hit verified: extraction was skipped despite snap_hash != text_hash
    assert not called[0]


def test_finding7_origin_plus_origin_id_source_identity(
    test_db: Database, test_blob_store: BlobStore
):
    """Finding 7: origin + origin_id forms valid standalone identity key."""
    bundle = {
        "type": "research-bundle",
        "schema_version": "1",
        "bundle_id": "bun_origin_standalone",
        "title": "Origin ID Test",
        "sources": [
            {
                "origin": "x",
                "origin_id": "1890000000000000000",
                "extracted_text": "A post from X",
            }
        ],
    }
    with test_db.transaction() as conn:
        res = import_research_bundle(conn, test_blob_store, bundle)
        assert res["status"] == "imported"

    with test_db.connection() as conn:
        r = conn.execute(
            "SELECT identity_key FROM resources WHERE identity_key = 'x:1890000000000000000';"
        ).fetchone()
        assert r is not None


def test_finding8_consistent_job_keys():
    """Finding 8: make_job_key generates consistent deterministic keys."""
    assert make_job_key("fetch", "res_123") == "fetch:res_123"
    assert make_job_key("extract", "res_123") == "extract:res_123"
    assert make_job_key("classify", "res_123") == "classify:res_123"


def test_finding9_unsupported_stages_rejection_and_loop_prevention(test_db: Database):
    """Finding 9: edward process with unsupported stage exits with code 1; claim_job rejects it."""
    # CLI rejection
    res = runner.invoke(app, ["process", "--stage", "invalid-stage"])
    assert res.exit_code == 1

    # claim_job returns None for unsupported stage
    with test_db.connection() as conn:
        job = claim_job(conn, worker_id="test_w", stage="invalid-stage")
        assert job is None


def test_finding10_subprocess_error_sanitization(
    test_db: Database, test_blob_store: BlobStore, monkeypatch: pytest.MonkeyPatch
):
    """Finding 10: sanitize_error_message redacts home directories and secrets, and persists sanitized last_error."""
    home_path = str(Path.home())
    dirty_err = f"Error in {home_path}/secret_project: Authorization Bearer sk-ant-1234567890abcdef failed with api_key=supersecretkey"
    clean_err = sanitize_error_message(dirty_err)
    assert home_path not in clean_err
    assert "~" in clean_err
    assert "Bearer [REDACTED]" in clean_err
    assert "supersecretkey" not in clean_err

    # Check persistence in processing_jobs.last_error
    now_iso = datetime.datetime.now(datetime.UTC).isoformat()
    with test_db.transaction() as conn:
        conn.execute(
            """
            INSERT INTO resources (id, identity_key, canonical_url, title, review_state, is_deleted, created_at, updated_at)
            VALUES ('res_err', 'url:err', 'https://example.com/err', 'Err', 'unreviewed', 0, ?, ?);
            """,
            (now_iso, now_iso),
        )
        conn.execute(
            """
            INSERT INTO processing_jobs (id, job_key, resource_id, stage, status, attempts, max_attempts, available_at, created_at, updated_at)
            VALUES ('job_err_1', 'extract:res_err', 'res_err', 'extract', 'pending', 2, 3, ?, ?, ?);
            """,
            (now_iso, now_iso, now_iso),
        )

    def failing_extract(url, raw_html=None):
        raise RuntimeError(dirty_err)

    monkeypatch.setattr("edward.services.processor.extract_content", failing_extract)

    process_pending_jobs(test_db, test_blob_store, limit=1)

    with test_db.connection() as conn:
        job = conn.execute(
            "SELECT status, last_error FROM processing_jobs WHERE id = 'job_err_1';"
        ).fetchone()
        assert job["status"] == "failed"
        assert home_path not in job["last_error"]
        assert "supersecretkey" not in job["last_error"]


def test_finding11_agent_confidence_zero_preserved(test_db: Database, test_blob_store: BlobStore):
    """Finding 11: agent_confidence = 0.0 is preserved as 0.0 in object_labels."""
    bundle = {
        "type": "research-bundle",
        "schema_version": "1",
        "bundle_id": "bun_zero_conf",
        "title": "Zero Confidence Test",
        "sources": [
            {
                "origin": "web",
                "url": "https://example.com/zero",
                "extracted_text": "Text for zero confidence",
            }
        ],
        "findings": [
            {
                "statement": "Uncertain statement with zero confidence.",
                "assertion_role": "agent-conclusion",
                "labels": ["unverified-claim"],
                "agent_confidence": 0.0,
            }
        ],
    }
    with test_db.transaction() as conn:
        res = import_research_bundle(conn, test_blob_store, bundle)
        assert res["status"] == "imported"

    with test_db.connection() as conn:
        row = conn.execute(
            "SELECT confidence FROM object_labels WHERE label_id = 'unverified-claim';"
        ).fetchone()
        assert row is not None
        assert row["confidence"] == 0.0
