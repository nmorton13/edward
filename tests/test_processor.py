"""Tests for background processing engine, leases, backoff, and recovery."""

import datetime
from unittest.mock import patch

import pytest

from edward.blobs import BlobStore
from edward.db import Database
from edward.services.extract import ExtractionResult
from edward.services.network import FetchedResource
from edward.services.processor import (
    _perform_job_work,
    claim_job,
    execute_job,
    get_processing_status,
    process_pending_jobs,
    retry_failed_jobs,
)


@pytest.fixture
def test_db_and_blobs(tmp_path):
    db_file = tmp_path / "test_edward.db"
    blobs_dir = tmp_path / "blobs"
    db = Database(db_file)
    db.run_migrations()
    blob_store = BlobStore(blobs_dir)
    return db, blob_store


def test_claim_job_atomic_lease(test_db_and_blobs):
    db, _ = test_db_and_blobs
    now_iso = datetime.datetime.now(datetime.UTC).isoformat()

    with db.transaction() as conn:
        conn.execute(
            """
            INSERT INTO processing_jobs (
                id, job_key, stage, status, attempts, max_attempts, available_at, created_at, updated_at
            ) VALUES ('job_1', 'key_1', 'resource-fetch', 'pending', 0, 3, ?, ?, ?);
            """,
            (now_iso, now_iso, now_iso),
        )

    with db.transaction() as conn:
        job = claim_job(conn, worker_id="worker_a", lease_seconds=30)
        assert job is not None
        assert job["id"] == "job_1"
        # The claim is a single atomic statement, so the returned row reflects
        # the state AFTER leasing. It used to report the pre-update value
        # ("pending") because the row was read by a separate earlier SELECT.
        assert job["status"] == "running"

        # Check DB state has updated to running
        row = conn.execute(
            "SELECT status, lease_owner, lease_expires_at FROM processing_jobs WHERE id = 'job_1';"
        ).fetchone()
        assert row["status"] == "running"
        assert row["lease_owner"] == "worker_a"
        assert row["lease_expires_at"] is not None

        # Another worker tries to claim
        job2 = claim_job(conn, worker_id="worker_b", lease_seconds=30)
        assert job2 is None


def test_claim_job_recovers_expired_lease(test_db_and_blobs):
    db, _ = test_db_and_blobs
    now = datetime.datetime.now(datetime.UTC)
    past_iso = (now - datetime.timedelta(seconds=60)).isoformat()

    with db.transaction() as conn:
        conn.execute(
            """
            INSERT INTO processing_jobs (
                id, job_key, stage, status, lease_owner, lease_expires_at, attempts, max_attempts, available_at, created_at, updated_at
            ) VALUES ('job_dead', 'key_dead', 'resource-fetch', 'running', 'dead_worker', ?, 1, 3, ?, ?, ?);
            """,
            (past_iso, past_iso, past_iso, past_iso),
        )

    with db.transaction() as conn:
        # A new worker should recover the expired lease and claim the job
        job = claim_job(conn, worker_id="worker_live", lease_seconds=60)
        assert job is not None
        assert job["id"] == "job_dead"

        row = conn.execute(
            "SELECT status, lease_owner FROM processing_jobs WHERE id = 'job_dead';"
        ).fetchone()
        assert row["status"] == "running"
        assert row["lease_owner"] == "worker_live"


def test_job_failure_retry_and_max_attempts(test_db_and_blobs):
    db, blob_store = test_db_and_blobs
    now_iso = datetime.datetime.now(datetime.UTC).isoformat()

    with db.transaction() as conn:
        conn.execute(
            """
            INSERT INTO processing_jobs (
                id, job_key, stage, status, attempts, max_attempts, available_at, created_at, updated_at
            ) VALUES ('job_fail', 'key_fail', 'resource-fetch', 'pending', 2, 3, ?, ?, ?);
            """,
            (now_iso, now_iso, now_iso),
        )

    with db.transaction() as conn:
        job = claim_job(conn, worker_id="worker_1")
        assert job is not None

    # Execute without a resource_id -> will fail and hit max_attempts
    with db.transaction() as conn:
        success = execute_job(conn, blob_store, job)
        assert success is False

        row = conn.execute(
            "SELECT status, attempts, last_error FROM processing_jobs WHERE id = 'job_fail';"
        ).fetchone()
        assert row["status"] == "failed"
        assert row["attempts"] == 3
        assert "requires resource_id" in row["last_error"]
        assert get_processing_status(conn)["failure_reasons"] == {"resource-fetch": {"other": 1}}


def test_retry_failed_jobs(test_db_and_blobs):
    db, _ = test_db_and_blobs
    now_iso = datetime.datetime.now(datetime.UTC).isoformat()

    with db.transaction() as conn:
        conn.execute(
            """
            INSERT INTO processing_jobs (
                id, job_key, stage, status, attempts, max_attempts, available_at, created_at, updated_at
            ) VALUES
                ('job_f1', 'k1', 'resource-fetch', 'failed', 3, 3, ?, ?, ?),
                ('job_f2', 'k2', 'extract', 'failed', 3, 3, ?, ?, ?);
            """,
            (now_iso, now_iso, now_iso, now_iso, now_iso, now_iso),
        )

        count = retry_failed_jobs(conn)
        assert count == 2

        rows = conn.execute("SELECT status, attempts FROM processing_jobs;").fetchall()
        for r in rows:
            assert r["status"] == "pending"
            assert r["attempts"] == 0


def test_saved_short_link_html_is_extracted_without_refetch(test_db_and_blobs):
    _, blob_store = test_db_and_blobs
    digest, _ = blob_store.store_bytes(
        b"<html><head><title>Linked article</title></head><body><p>Detailed research findings.</p></body></html>"
    )
    result = _perform_job_work(
        blob_store,
        {"stage": "extract"},
        {
            "canonical_url": "https://t.co/example",
            "snapshot_content_hash": digest,
            "snapshot_content_type": "text/html",
        },
    )
    assert result["ext_result"].clean_text == "Linked article Detailed research findings."
    assert result["ext_result"].extractor == "local-fallback"


def test_saved_image_is_not_misread_as_text(test_db_and_blobs):
    _, blob_store = test_db_and_blobs
    digest, _ = blob_store.store_bytes(b"\x89PNG\r\n\x1a\n")
    result = _perform_job_work(
        blob_store,
        {"stage": "extract"},
        {
            "canonical_url": "https://t.co/image",
            "snapshot_content_hash": digest,
            "snapshot_content_type": "image/png",
        },
    )
    assert result["ext_result"] is None


def test_saved_pdf_uses_local_text_and_url_filename(test_db_and_blobs):
    _, blob_store = test_db_and_blobs
    digest, _ = blob_store.store_bytes(b"%PDF-example")
    with patch("edward.services.processor.extract_pdf_text", return_value="[PDF page 1]\nText"):
        result = _perform_job_work(
            blob_store,
            {"stage": "extract"},
            {
                "canonical_url": "https://example.com/AI-for-Activists-Guide.pdf",
                "snapshot_content_hash": digest,
                "snapshot_content_type": "application/pdf",
            },
        )
    assert result["ext_result"].clean_text == "[PDF page 1]\nText"
    assert result["ext_result"].title == "AI for Activists Guide"
    assert result["ext_result"].extractor == "pypdf"


@patch("edward.services.processor.safe_fetch_url")
@patch("edward.services.processor.extract_content")
def test_full_processing_pipeline(mock_extract, mock_fetch, test_db_and_blobs, monkeypatch):
    monkeypatch.setenv("EDWARD_CLASSIFIER_PROVIDER", "dry-run")
    db, blob_store = test_db_and_blobs
    now_iso = datetime.datetime.now(datetime.UTC).isoformat()

    # Set up mock fetch return
    mock_fetch.return_value = FetchedResource(
        url="https://example.com/test-article",
        final_url="https://example.com/test-article",
        status_code=200,
        headers={"content-type": "text/html"},
        body=b"<html><head><title>Test Article</title></head><body><p>This is a test article about artificial intelligence and machine learning.</p></body></html>",
        content_hash="mock_hash_123",
        elapsed_seconds=0.1,
    )

    # Set up mock extract return
    mock_extract.return_value = ExtractionResult(
        status="completed",
        title="Test Article",
        clean_text="This is a test article about artificial intelligence and machine learning.",
        summary="This is a test article summary.",
        extractor="mock-extractor",
        extractor_version="1.0",
    )

    with db.transaction() as conn:
        # Create resource
        conn.execute(
            """
            INSERT INTO resources (id, identity_key, canonical_url, primary_form, review_state, is_deleted, created_at, updated_at)
            VALUES ('res_pipe', 'url:example', 'https://example.com/test-article', 'article', 'unreviewed', 0, ?, ?);
            """,
            (now_iso, now_iso),
        )
        # Create initial resource-fetch job
        conn.execute(
            """
            INSERT INTO processing_jobs (
                id, job_key, resource_id, stage, status, attempts, max_attempts, available_at, created_at, updated_at
            ) VALUES ('job_pipe_1', 'fetch_pipe', 'res_pipe', 'resource-fetch', 'pending', 0, 3, ?, ?, ?);
            """,
            (now_iso, now_iso, now_iso),
        )

    # Run processing loop: stage 1 (resource-fetch)
    with db.transaction() as conn:
        res1 = process_pending_jobs(conn, blob_store, limit=1)
        assert res1["completed"] == 1
        assert res1["failed"] == 0

        # Verify extract stage was queued
        extract_job = conn.execute(
            "SELECT * FROM processing_jobs WHERE stage = 'extract';"
        ).fetchone()
        assert extract_job is not None
        assert extract_job["status"] == "pending"

    # Run processing loop: stage 2 (extract)
    with db.transaction() as conn:
        res2 = process_pending_jobs(conn, blob_store, limit=1)
        assert res2["completed"] == 1

        # Verify classify stage was queued
        classify_job = conn.execute(
            "SELECT * FROM processing_jobs WHERE stage = 'classify';"
        ).fetchone()
        assert classify_job is not None
        assert classify_job["status"] == "pending"
        embed_job = conn.execute("SELECT * FROM processing_jobs WHERE stage = 'embed';").fetchone()
        assert embed_job is not None
        assert embed_job["status"] == "pending"
        assert (
            conn.execute("SELECT title FROM resources WHERE id = 'res_pipe'").fetchone()[0]
            == "Test Article"
        )

    # Run processing loop: stage 3 (classify)
    with db.transaction() as conn:
        res3 = process_pending_jobs(conn, blob_store, limit=1, stage="classify")
        assert res3["completed"] == 1

        # Check judgments persisted
        judgments = conn.execute("SELECT * FROM judgments WHERE object_id = 'res_pipe';").fetchall()
        assert len(judgments) > 0

    # Run processing loop: stage 4 (finding-extraction)
    with db.transaction() as conn:
        res4 = process_pending_jobs(conn, blob_store, limit=1, stage="finding-extraction")
        assert res4["completed"] == 1

        fe_job = conn.execute(
            "SELECT * FROM processing_jobs WHERE stage = 'finding-extraction';"
        ).fetchone()
        assert fe_job["status"] == "completed"

    # Run processing loop: stage 5 (embed)
    with db.transaction() as conn:
        res5 = process_pending_jobs(conn, blob_store, limit=1)
        assert res5["completed"] == 1

        emb_job = conn.execute("SELECT * FROM processing_jobs WHERE stage = 'embed';").fetchone()
        assert emb_job["status"] == "completed"

    # Verify status report
    with db.connection() as conn:
        status = get_processing_status(conn)
        assert status["total"] == 5
        assert status["by_status"]["completed"] == 5
        assert status["by_status"]["pending"] == 0
        assert status["by_stage_status"]["resource-fetch"] == {"completed": 1}
