"""Tests verifying Phase 2 Round 6 review findings: in-flight invalidation and input_hash provenance."""

import datetime

from edward.blobs import BlobStore
from edward.db import Database
from edward.models import generate_id
from edward.services.extract import ExtractionResult
from edward.services.processor import (
    _load_job_context,
    _perform_job_work,
    _persist_job_result,
    claim_job,
    process_pending_jobs,
)
from edward.services.resource import (
    save_source_snapshot,
    store_resource_content,
)


def test_round6_in_flight_classification_content_change_invalidates_and_requeues(
    test_db: Database, test_blob_store: BlobStore, monkeypatch
):
    """Round 6: Classification in flight discards stale output on content change and leaves job pending for latest input."""
    monkeypatch.setenv("EDWARD_CLASSIFIER_PROVIDER", "dry-run")
    now_iso = datetime.datetime.now(datetime.UTC).isoformat()
    res_id = generate_id("res")

    # 1. Initial resource with old content and a pending classify job
    with test_db.transaction() as conn:
        conn.execute(
            """
            INSERT INTO resources (id, identity_key, canonical_url, title, primary_form, review_state, is_deleted, created_at, updated_at)
            VALUES (?, 'url:https://example.com/in-flight-cls', 'https://example.com/in-flight-cls', 'Initial Title', 'article', 'unreviewed', 0, ?, ?);
            """,
            (res_id, now_iso, now_iso),
        )
        old_content_id, old_hash = store_resource_content(
            conn, res_id, clean_text="Old initial content about astronomy and stars."
        )
        conn.execute(
            """
            INSERT INTO processing_jobs (
                id, job_key, resource_id, stage, status, attempts, max_attempts, available_at, created_at, updated_at
            ) VALUES ('job_cls_race', 'classify:' || ?, ?, 'classify', 'pending', 0, 3, ?, ?, ?);
            """,
            (res_id, res_id, now_iso, now_iso, now_iso),
        )

    # 2. Worker 1 claims classification for old content
    with test_db.transaction() as conn:
        job = claim_job(conn, worker_id="worker_1", stage="classify")
        assert job is not None
        assert job["id"] == "job_cls_race"
        ctx = _load_job_context(conn, job)

    # 3. Worker 1 performs classification outside the transaction on old content
    work_res = _perform_job_work(test_blob_store, job, ctx)
    assert work_res["input_hash"] == old_hash

    # 4. In the meantime, newer resource content is stored
    new_text = "New updated content about deep ocean exploration and marine biology."
    with test_db.transaction() as conn:
        new_content_id, new_hash = store_resource_content(conn, res_id, clean_text=new_text)
        assert new_hash != old_hash

    # 5. Worker 1 attempts to persist the stale classification result
    with test_db.transaction() as conn:
        outcome = _persist_job_result(
            conn,
            test_blob_store,
            job,
            work_res,
            job_error=None,
            worker_id="worker_1",
            context=ctx,
        )
        assert outcome == "pending"

    # 6. Verify stale judgments were NOT stored and the job was reset to pending for the new content
    with test_db.connection() as conn:
        judgments_old = conn.execute(
            "SELECT * FROM judgments WHERE object_id = ? AND input_content_hash = ?;",
            (res_id, old_hash),
        ).fetchall()
        assert len(judgments_old) == 0

        job_row = conn.execute(
            "SELECT status, input_hash, lease_owner FROM processing_jobs WHERE id = 'job_cls_race';"
        ).fetchone()
        assert job_row["status"] == "pending"
        assert job_row["lease_owner"] is None

    # 7. Next processing loop execution classifies the new content cleanly
    res_loop = process_pending_jobs(test_db, test_blob_store, limit=1)
    assert res_loop["completed"] == 1

    with test_db.connection() as conn:
        job_row2 = conn.execute(
            "SELECT status, input_hash FROM processing_jobs WHERE id = 'job_cls_race';"
        ).fetchone()
        assert job_row2["status"] == "completed"
        assert job_row2["input_hash"] == new_hash

        judgments_new = conn.execute(
            "SELECT * FROM judgments WHERE object_id = ? AND input_content_hash = ?;",
            (res_id, new_hash),
        ).fetchall()
        assert len(judgments_new) > 0


def test_round6_in_flight_extraction_snapshot_change_invalidates_and_requeues(
    test_db: Database, test_blob_store: BlobStore
):
    """Round 6: Extraction in flight discards stale output on new snapshot arrival and leaves job pending."""
    now_iso = datetime.datetime.now(datetime.UTC).isoformat()
    res_id = generate_id("res")

    # 1. Initial resource and initial snapshot S_old
    with test_db.transaction() as conn:
        conn.execute(
            """
            INSERT INTO resources (id, identity_key, canonical_url, title, primary_form, review_state, is_deleted, created_at, updated_at)
            VALUES (?, 'url:https://example.com/in-flight-ext', 'https://example.com/in-flight-ext', 'Article', 'article', 'unreviewed', 0, ?, ?);
            """,
            (res_id, now_iso, now_iso),
        )
        _, old_snap_hash = save_source_snapshot(
            conn,
            test_blob_store,
            res_id,
            b"<html><body><h1>Old Snapshot Content</h1></body></html>",
            {"content-type": "text/html"},
        )
        conn.execute(
            """
            INSERT INTO processing_jobs (
                id, job_key, resource_id, stage, status, attempts, max_attempts, available_at, created_at, updated_at
            ) VALUES ('job_ext_race', 'extract:' || ?, ?, 'extract', 'pending', 0, 3, ?, ?, ?);
            """,
            (res_id, res_id, now_iso, now_iso, now_iso),
        )

    # 2. Worker claims extraction for old snapshot
    with test_db.transaction() as conn:
        job = claim_job(conn, worker_id="worker_ext", stage="extract")
        assert job is not None
        assert job["id"] == "job_ext_race"
        ctx = _load_job_context(conn, job)
        assert ctx["snapshot_content_hash"] == old_snap_hash

    # 3. Worker performs extraction outside transaction on old snapshot
    stale_ext_result = ExtractionResult(
        status="completed",
        title="Old Title",
        clean_text="Old Snapshot Content",
        summary="Old Summary",
        extractor="local",
        extractor_version="1.0",
    )
    work_res = {"status": "completed", "ext_result": stale_ext_result}

    # 4. In the meantime, a new fetch saves a new snapshot S_new
    with test_db.transaction() as conn:
        _, new_snap_hash = save_source_snapshot(
            conn,
            test_blob_store,
            res_id,
            b"<html><body><h1>New Newer Snapshot</h1></body></html>",
            {"content-type": "text/html"},
        )
        assert new_snap_hash != old_snap_hash

    # 5. Worker attempts to persist the stale extraction result
    with test_db.transaction() as conn:
        outcome = _persist_job_result(
            conn,
            test_blob_store,
            job,
            work_res,
            job_error=None,
            worker_id="worker_ext",
            context=ctx,
        )
        assert outcome == "pending"

    # 6. Verify stale text was NOT stored into resource_contents and extract job remains pending
    with test_db.connection() as conn:
        contents = conn.execute(
            "SELECT * FROM resource_contents WHERE resource_id = ?;", (res_id,)
        ).fetchall()
        assert len(contents) == 0

        job_row = conn.execute(
            "SELECT status, input_hash, lease_owner FROM processing_jobs WHERE id = 'job_ext_race';"
        ).fetchone()
        assert job_row["status"] == "pending"
        assert job_row["lease_owner"] is None


def test_round6_completed_classification_records_non_null_input_hash(
    test_db: Database, test_blob_store: BlobStore, monkeypatch
):
    """Round 6: Successfully completed classification job sets input_hash to canonical content hash."""
    monkeypatch.setenv("EDWARD_CLASSIFIER_PROVIDER", "dry-run")
    now_iso = datetime.datetime.now(datetime.UTC).isoformat()
    res_id = generate_id("res")

    with test_db.transaction() as conn:
        conn.execute(
            """
            INSERT INTO resources (id, identity_key, canonical_url, title, primary_form, review_state, is_deleted, created_at, updated_at)
            VALUES (?, 'url:https://example.com/hash-check', 'https://example.com/hash-check', 'Hash Check', 'article', 'unreviewed', 0, ?, ?);
            """,
            (res_id, now_iso, now_iso),
        )
        _, expected_hash = store_resource_content(
            conn, res_id, clean_text="Unique verifiable text for content hash test."
        )
        conn.execute(
            """
            INSERT INTO processing_jobs (
                id, job_key, resource_id, stage, status, attempts, max_attempts, available_at, created_at, updated_at
            ) VALUES ('job_hash_check', 'classify:' || ?, ?, 'classify', 'pending', 0, 3, ?, ?, ?);
            """,
            (res_id, res_id, now_iso, now_iso, now_iso),
        )

    res = process_pending_jobs(test_db, test_blob_store, limit=1)
    assert res["completed"] == 1

    with test_db.connection() as conn:
        row = conn.execute(
            "SELECT status, input_hash FROM processing_jobs WHERE id = 'job_hash_check';"
        ).fetchone()
        assert row["status"] == "completed"
        assert row["input_hash"] is not None
        assert row["input_hash"] == expected_hash
