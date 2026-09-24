"""Tests verifying Phase 2 Round 5 review findings and classifier failure recovery."""

import datetime
from unittest.mock import patch

from edward.blobs import BlobStore
from edward.db import Database
from edward.models import generate_id
from edward.services.processor import (
    claim_job,
    execute_job,
    process_pending_jobs,
)


def test_round5_classifier_failure_records_retry_and_clears_lease(
    test_db: Database, test_blob_store: BlobStore, monkeypatch
):
    """Round 5: Invalid classifier provider or classification failure records retry and clears lease."""
    monkeypatch.setenv("EDWARD_CLASSIFIER_PROVIDER", "invalid-unknown-provider")
    now_iso = datetime.datetime.now(datetime.UTC).isoformat()
    res_id = generate_id("res")

    with test_db.transaction() as conn:
        conn.execute(
            """
            INSERT INTO resources (id, identity_key, canonical_url, title, primary_form, review_state, is_deleted, created_at, updated_at)
            VALUES (?, 'url:https://example.com/classify-fail', 'https://example.com/classify-fail', 'Fail Test', 'article', 'unreviewed', 0, ?, ?);
            """,
            (res_id, now_iso, now_iso),
        )
        conn.execute(
            """
            INSERT INTO processing_jobs (
                id, job_key, resource_id, stage, status, attempts, max_attempts, available_at, created_at, updated_at
            ) VALUES ('job_cls_fail', 'classify:' || ?, ?, 'classify', 'pending', 0, 3, ?, ?, ?);
            """,
            (res_id, res_id, now_iso, now_iso, now_iso),
        )

    # Process jobs: should NOT raise an unhandled ValueError escaping the loop
    outcome = process_pending_jobs(test_db, test_blob_store, limit=1)
    assert outcome["failed"] == 1
    assert outcome["completed"] == 0

    with test_db.connection() as conn:
        row = conn.execute(
            "SELECT status, attempts, max_attempts, last_error, lease_owner, lease_expires_at, available_at FROM processing_jobs WHERE id = 'job_cls_fail';"
        ).fetchone()

        # Job must NOT remain running or hold an active lease
        assert row["status"] == "pending"
        assert row["attempts"] == 1
        assert row["lease_owner"] is None
        assert row["lease_expires_at"] is None
        assert "Unknown classifier provider" in row["last_error"]
        assert row["available_at"] > now_iso


def test_round5_classifier_failure_reaches_max_attempts_and_fails(
    test_db: Database, test_blob_store: BlobStore, monkeypatch
):
    """Round 5: Repeated classifier failure increments attempts up to max_attempts and marks failed."""
    monkeypatch.setenv("EDWARD_CLASSIFIER_PROVIDER", "invalid-unknown-provider")
    now_iso = datetime.datetime.now(datetime.UTC).isoformat()
    res_id = generate_id("res")

    with test_db.transaction() as conn:
        conn.execute(
            """
            INSERT INTO resources (id, identity_key, canonical_url, title, primary_form, review_state, is_deleted, created_at, updated_at)
            VALUES (?, 'url:https://example.com/max-fail', 'https://example.com/max-fail', 'Max Fail', 'article', 'unreviewed', 0, ?, ?);
            """,
            (res_id, now_iso, now_iso),
        )
        conn.execute(
            """
            INSERT INTO processing_jobs (
                id, job_key, resource_id, stage, status, attempts, max_attempts, available_at, created_at, updated_at
            ) VALUES ('job_max_fail', 'classify:' || ?, ?, 'classify', 'pending', 2, 3, ?, ?, ?);
            """,
            (res_id, res_id, now_iso, now_iso, now_iso),
        )

    outcome = process_pending_jobs(test_db, test_blob_store, limit=1)
    assert outcome["failed"] == 1

    with test_db.connection() as conn:
        row = conn.execute(
            "SELECT status, attempts, completed_at, last_error, lease_owner FROM processing_jobs WHERE id = 'job_max_fail';"
        ).fetchone()

        assert row["status"] == "failed"
        assert row["attempts"] == 3
        assert row["lease_owner"] is None
        assert row["completed_at"] is not None
        assert "Unknown classifier provider" in row["last_error"]


def test_round5_persistence_failure_recorded_in_separate_transaction(
    test_db: Database, test_blob_store: BlobStore
):
    """Round 5: If the persistence transaction fails, error is recorded and lease cleared via separate transaction."""
    now_iso = datetime.datetime.now(datetime.UTC).isoformat()
    res_id = generate_id("res")

    with test_db.transaction() as conn:
        conn.execute(
            """
            INSERT INTO resources (id, identity_key, canonical_url, title, primary_form, review_state, is_deleted, created_at, updated_at)
            VALUES (?, 'url:https://example.com/persist-fail', 'https://example.com/persist-fail', 'Persist Fail', 'article', 'unreviewed', 0, ?, ?);
            """,
            (res_id, now_iso, now_iso),
        )
        conn.execute(
            """
            INSERT INTO processing_jobs (
                id, job_key, resource_id, stage, status, attempts, max_attempts, available_at, created_at, updated_at
            ) VALUES ('job_persist_fail', 'classify:' || ?, ?, 'classify', 'pending', 0, 3, ?, ?, ?);
            """,
            (res_id, res_id, now_iso, now_iso, now_iso),
        )

    # Force a failure during the side-effect portion of the first persistence attempt
    call_count = [0]

    def failing_persist(*args, **kwargs):
        call_count[0] += 1
        raise RuntimeError("Disk full during classification persistence")

    with patch(
        "edward.services.processor.persist_classification_result", side_effect=failing_persist
    ):
        outcome = process_pending_jobs(test_db, test_blob_store, limit=1)
        assert outcome["failed"] == 1

    with test_db.connection() as conn:
        row = conn.execute(
            "SELECT status, attempts, last_error, lease_owner, lease_expires_at FROM processing_jobs WHERE id = 'job_persist_fail';"
        ).fetchone()

        assert row["status"] == "pending"
        assert row["attempts"] == 1
        assert row["lease_owner"] is None
        assert row["lease_expires_at"] is None
        assert "Disk full during classification persistence" in row["last_error"]


def test_round5_execute_job_handles_classifier_failure(
    test_db: Database, test_blob_store: BlobStore, monkeypatch
):
    """Round 5: execute_job directly catches classifier failures, records attempts, and returns False."""
    monkeypatch.setenv("EDWARD_CLASSIFIER_PROVIDER", "unsupported-provider")
    now_iso = datetime.datetime.now(datetime.UTC).isoformat()
    res_id = generate_id("res")

    with test_db.transaction() as conn:
        conn.execute(
            """
            INSERT INTO resources (id, identity_key, canonical_url, title, primary_form, review_state, is_deleted, created_at, updated_at)
            VALUES (?, 'url:https://example.com/exec-fail', 'https://example.com/exec-fail', 'Exec Fail', 'article', 'unreviewed', 0, ?, ?);
            """,
            (res_id, now_iso, now_iso),
        )
        conn.execute(
            """
            INSERT INTO processing_jobs (
                id, job_key, resource_id, stage, status, attempts, max_attempts, available_at, created_at, updated_at
            ) VALUES ('job_exec_fail', 'classify:' || ?, ?, 'classify', 'pending', 0, 3, ?, ?, ?);
            """,
            (res_id, res_id, now_iso, now_iso, now_iso),
        )

        job = claim_job(conn, worker_id="worker_exec")
        assert job is not None

        success = execute_job(conn, test_blob_store, job)
        assert success is False

        row = conn.execute(
            "SELECT status, attempts, last_error, lease_owner FROM processing_jobs WHERE id = 'job_exec_fail';"
        ).fetchone()
        assert row["status"] == "pending"
        assert row["attempts"] == 1
        assert row["lease_owner"] is None
        assert "Unknown classifier provider" in row["last_error"]
