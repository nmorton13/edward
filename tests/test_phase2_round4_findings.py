"""Regression tests for Phase 2 Round 4 review findings."""

import datetime
import hashlib
import json

import pytest

from edward.blobs import BlobStore
from edward.db import Database
from edward.models import ResearchBundle, SourceItem, generate_id
from edward.services.bundle import import_research_bundle
from edward.services.extract import ExtractionResult
from edward.services.processor import (
    _load_job_context,
    _persist_job_result,
)


def test_round4_finding1_lease_fencing_prevents_all_side_effects(
    test_db: Database, test_blob_store: BlobStore
):
    """Finding 1: A worker who lost the lease commits zero side effects across fetch, extract, and classify."""
    now = datetime.datetime.now(datetime.UTC)
    now_iso = now.isoformat()
    future_iso = (now + datetime.timedelta(seconds=60)).isoformat()
    res_id = generate_id("res")

    with test_db.transaction() as conn:
        conn.execute(
            """
            INSERT INTO resources (id, identity_key, canonical_url, title, review_state, is_deleted, created_at, updated_at)
            VALUES (?, 'url:https://example.com/fenced', 'https://example.com/fenced', 'Fenced Resource', 'unreviewed', 0, ?, ?);
            """,
            (res_id, now_iso, now_iso),
        )

    # -------------------------------------------------------------
    # 1. Fetch stage: Stale worker attempting to persist fetch result
    # -------------------------------------------------------------
    with test_db.transaction() as conn:
        conn.execute(
            """
            INSERT INTO processing_jobs (
                id, job_key, resource_id, stage, status, available_at, attempts, lease_owner, lease_expires_at, created_at, updated_at
            ) VALUES ('job_f_fetch', 'fetch:' || ?, ?, 'resource-fetch', 'running', ?, 0, 'worker_legit', ?, ?, ?);
            """,
            (res_id, res_id, now_iso, future_iso, now_iso, now_iso),
        )

    class FakeFetchResult:
        body = b"<html><body>Stale Worker Snapshot</body></html>"
        headers = {"content-type": "text/html"}
        status_code = 200

    stale_fetch_job = {
        "id": "job_f_fetch",
        "job_key": f"fetch:{res_id}",
        "resource_id": res_id,
        "stage": "resource-fetch",
        "attempts": 0,
        "max_attempts": 3,
        "lease_owner": "worker_stale",
    }

    with test_db.transaction() as conn:
        outcome = _persist_job_result(
            conn,
            test_blob_store,
            stale_fetch_job,
            work_result={"fetch_result": FakeFetchResult()},
            job_error=None,
            worker_id="worker_stale",
        )
        assert outcome == "lost-lease"

    # Verify ZERO snapshot side effects in SQLite and blob store
    with test_db.connection() as conn:
        snapshots = conn.execute(
            "SELECT * FROM source_snapshots WHERE resource_id = ?;", (res_id,)
        ).fetchall()
        assert len(snapshots) == 0

    stale_snap_hash = hashlib.sha256(FakeFetchResult.body).hexdigest()
    assert not test_blob_store.exists(stale_snap_hash)

    # -------------------------------------------------------------
    # 2. Extract stage: Stale worker attempting to persist extract result
    # -------------------------------------------------------------
    with test_db.transaction() as conn:
        conn.execute(
            """
            INSERT INTO processing_jobs (
                id, job_key, resource_id, stage, status, available_at, attempts, lease_owner, lease_expires_at, created_at, updated_at
            ) VALUES ('job_f_ext', 'extract:' || ?, ?, 'extract', 'running', ?, 0, 'worker_legit', ?, ?, ?);
            """,
            (res_id, res_id, now_iso, future_iso, now_iso, now_iso),
        )

    ext_res = ExtractionResult(
        status="completed",
        clean_text="Stale extracted text that must never be persisted.",
        transcript=json.dumps([{"speaker": "Bob", "text": "Stale transcript segment."}]),
        summary="Stale summary",
        extractor="test-extractor",
        extractor_version="1.0",
        title="Stale Title",
    )

    stale_extract_job = {
        "id": "job_f_ext",
        "job_key": f"extract:{res_id}",
        "resource_id": res_id,
        "stage": "extract",
        "attempts": 0,
        "max_attempts": 3,
        "lease_owner": "worker_stale",
    }

    with test_db.transaction() as conn:
        outcome = _persist_job_result(
            conn,
            test_blob_store,
            stale_extract_job,
            work_result={"ext_result": ext_res, "cached": False},
            job_error=None,
            worker_id="worker_stale",
        )
        assert outcome == "lost-lease"

    # Verify ZERO resource_contents and ZERO resource_chunks were committed
    with test_db.connection() as conn:
        contents = conn.execute(
            "SELECT * FROM resource_contents WHERE resource_id = ?;", (res_id,)
        ).fetchall()
        assert len(contents) == 0
        chunks = conn.execute(
            "SELECT * FROM resource_chunks WHERE resource_id = ?;", (res_id,)
        ).fetchall()
        assert len(chunks) == 0

    # -------------------------------------------------------------
    # 3. Classify stage: Stale worker attempting to persist classify result
    # -------------------------------------------------------------
    with test_db.transaction() as conn:
        conn.execute(
            """
            INSERT INTO processing_jobs (
                id, job_key, resource_id, stage, status, available_at, attempts, lease_owner, lease_expires_at, created_at, updated_at
            ) VALUES ('job_f_cls', 'classify:' || ?, ?, 'classify', 'running', ?, 0, 'worker_legit', ?, ?, ?);
            """,
            (res_id, res_id, now_iso, future_iso, now_iso, now_iso),
        )

    stale_classify_job = {
        "id": "job_f_cls",
        "job_key": f"classify:{res_id}",
        "resource_id": res_id,
        "stage": "classify",
        "attempts": 0,
        "max_attempts": 3,
        "lease_owner": "worker_stale",
    }

    with test_db.transaction() as conn:
        outcome = _persist_job_result(
            conn,
            test_blob_store,
            stale_classify_job,
            work_result={},
            job_error=None,
            worker_id="worker_stale",
        )
        assert outcome == "lost-lease"

    # Verify ZERO object_labels and ZERO object_entities were committed
    with test_db.connection() as conn:
        labels = conn.execute(
            "SELECT * FROM object_labels WHERE object_id = ?;", (res_id,)
        ).fetchall()
        assert len(labels) == 0
        entities = conn.execute(
            "SELECT * FROM object_entities WHERE object_id = ?;", (res_id,)
        ).fetchall()
        assert len(entities) == 0


def test_round4_finding2_extraction_cache_lookup_requires_completed_status(
    test_db: Database, test_blob_store: BlobStore
):
    """Finding 2: Extraction cache hit strictly requires status = 'completed'; pending, running, or failed jobs are rejected."""
    now_iso = datetime.datetime.now(datetime.UTC).isoformat()
    res_id = generate_id("res")
    snap_hash = hashlib.sha256(b"<html>Snapshot for Cache Test</html>").hexdigest()

    with test_db.transaction() as conn:
        conn.execute(
            """
            INSERT INTO resources (id, identity_key, canonical_url, title, review_state, is_deleted, created_at, updated_at)
            VALUES (?, 'url:https://example.com/cache-status', 'https://example.com/cache-status', 'Cache Status', 'unreviewed', 0, ?, ?);
            """,
            (res_id, now_iso, now_iso),
        )
        conn.execute(
            """
            INSERT INTO source_snapshots (id, resource_id, content_hash, blob_path, size_bytes, created_at)
            VALUES ('snp_cs', ?, ?, 'path/to/blob', 100, ?);
            """,
            (res_id, snap_hash, now_iso),
        )
        # Store clean_text in resource_contents with a different content hash
        text_hash = hashlib.sha256(b"Extracted clean text").hexdigest()
        conn.execute(
            """
            INSERT INTO resource_contents (id, resource_id, content_hash, clean_text, extractor, extractor_version, char_count, created_at)
            VALUES ('rc_cs', ?, ?, 'Extracted clean text', 'extractor', '1.0', 20, ?);
            """,
            (res_id, text_hash, now_iso),
        )

    # Case A: An existing extract job has input_hash = snap_hash but status = 'failed'
    with test_db.transaction() as conn:
        conn.execute(
            """
            INSERT INTO processing_jobs (
                id, job_key, resource_id, stage, status, input_hash, available_at, attempts, created_at, updated_at
            ) VALUES ('job_failed_ext', 'extract:' || ?, ?, 'extract', 'failed', ?, ?, 1, ?, ?);
            """,
            (res_id, res_id, snap_hash, now_iso, now_iso, now_iso),
        )

    current_job = {
        "id": "job_cur_ext",
        "job_key": f"extract:{res_id}",
        "resource_id": res_id,
        "stage": "extract",
        "input_hash": snap_hash,
    }

    with test_db.connection() as conn:
        ctx = _load_job_context(conn, current_job)
        # MUST NOT be a cache hit because the existing job was 'failed', not 'completed'
        assert ctx["cached_hit"] is False

    # Case B: Update the existing job to status = 'running' -> still MUST NOT be a cache hit
    with test_db.transaction() as conn:
        conn.execute("UPDATE processing_jobs SET status = 'running' WHERE id = 'job_failed_ext';")

    with test_db.connection() as conn:
        ctx = _load_job_context(conn, current_job)
        assert ctx["cached_hit"] is False

    # Case C: Update the existing job to status = 'completed' -> MUST NOW be a cache hit
    with test_db.transaction() as conn:
        conn.execute("UPDATE processing_jobs SET status = 'completed' WHERE id = 'job_failed_ext';")

    with test_db.connection() as conn:
        ctx = _load_job_context(conn, current_job)
        assert ctx["cached_hit"] is True
        assert ctx["cached_content"]["clean_text"] == "Extracted clean text"


def test_round4_finding3_content_hash_and_snapshot_hash_validation(
    test_db: Database, test_blob_store: BlobStore
):
    """Finding 3: Disambiguated validation when both snapshot and extracted text are supplied."""
    ext_text = "Disambiguated extracted content."
    raw_snap = "<html><body>Disambiguated Snapshot Body</body></html>"
    expected_ext_hash = hashlib.sha256(ext_text.encode("utf-8")).hexdigest()
    expected_snap_hash = hashlib.sha256(raw_snap.encode("utf-8")).hexdigest()

    # Case 1: Both supplied with matching content_hash (for text) and snapshot_hash (for snapshot) -> Success
    bundle_valid = ResearchBundle(
        bundle_id="bnd_both_valid",
        title="Valid Both Hashes",
        sources=[
            SourceItem(
                origin="web",
                url="https://example.com/both-valid",
                title="Both Valid",
                extracted_text=ext_text,
                content_hash=expected_ext_hash,
                snapshot=raw_snap,
                snapshot_hash=expected_snap_hash,
            )
        ],
    )
    with test_db.transaction() as conn:
        res = import_research_bundle(conn, test_blob_store, bundle_valid.model_dump())
        assert res["bundle_id"] == "bnd_both_valid"

    # Case 2: Both supplied with incorrect snapshot_hash -> ValueError
    bundle_bad_snap = ResearchBundle(
        bundle_id="bnd_bad_snap",
        title="Bad Snapshot Hash",
        sources=[
            SourceItem(
                origin="web",
                url="https://example.com/bad-snap",
                title="Bad Snap",
                extracted_text=ext_text,
                content_hash=expected_ext_hash,
                snapshot=raw_snap,
                snapshot_hash="wrong_snapshot_hash_1234567890abcdef1234567890abcdef",
            )
        ],
    )
    with pytest.raises(ValueError, match="Supplied snapshot_hash"):
        with test_db.transaction() as conn:
            import_research_bundle(conn, test_blob_store, bundle_bad_snap.model_dump())

    # Case 3: Both supplied with incorrect content_hash -> ValueError
    bundle_bad_content = ResearchBundle(
        bundle_id="bnd_bad_content",
        title="Bad Content Hash",
        sources=[
            SourceItem(
                origin="web",
                url="https://example.com/bad-content",
                title="Bad Content",
                extracted_text=ext_text,
                content_hash="wrong_content_hash_1234567890abcdef1234567890abcdef",
                snapshot=raw_snap,
                snapshot_hash=expected_snap_hash,
            )
        ],
    )
    with pytest.raises(ValueError, match="Supplied content_hash"):
        with test_db.transaction() as conn:
            import_research_bundle(conn, test_blob_store, bundle_bad_content.model_dump())
