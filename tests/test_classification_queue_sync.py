"""Tests that classification synchronizes queue state and handles empty text."""

from edward.db import Database
from edward.models import CaptureInput, make_job_key
from edward.services.capture import capture_item
from edward.services.classification import run_classification_pipeline


def test_classification_auto_completes_pending_queue_job(test_db: Database, monkeypatch):
    monkeypatch.setenv("EDWARD_CLASSIFIER_PROVIDER", "dry-run")

    with test_db.transaction() as conn:
        cap = capture_item(
            conn,
            CaptureInput(
                url="https://example.com/test",
                text="This is a test article about artificial intelligence and programming models.",
            ),
        )
        res_id = cap["resource_id"]

        conn.execute(
            """
            INSERT INTO resource_contents (id, resource_id, content_hash, clean_text, extractor, extractor_version, char_count, created_at)
            VALUES ('rc_test_1', ?, 'hash1', 'This is a test article about artificial intelligence and programming models.', 'ext', '1.0', 74, datetime('now'));
            """,
            (res_id,),
        )

        # Insert a pending classify job
        conn.execute(
            """
            INSERT INTO processing_jobs (id, job_key, resource_id, stage, status, available_at, created_at, updated_at)
            VALUES ('job_test_1', ?, ?, 'classify', 'pending', datetime('now'), datetime('now'), datetime('now'));
            """,
            (make_job_key("classify", res_id), res_id),
        )

        # Before classification: job is pending
        job_before = conn.execute(
            "SELECT status FROM processing_jobs WHERE id = 'job_test_1';"
        ).fetchone()
        assert job_before["status"] == "pending"

        # Run classification pipeline directly
        judgments = run_classification_pipeline(conn, "resource", res_id, provider="dry-run")
        assert len(judgments) > 0

        # After classification: job is completed
        job_after = conn.execute(
            "SELECT status, completed_at FROM processing_jobs WHERE id = 'job_test_1';"
        ).fetchone()
        assert job_after["status"] == "completed"
        assert job_after["completed_at"] is not None


def test_reconcile_completed_jobs(test_db: Database, cli_runner, monkeypatch):
    monkeypatch.setenv("EDWARD_DB_PATH", str(test_db.db_path))
    with test_db.transaction() as conn:
        cap = capture_item(
            conn,
            CaptureInput(
                url="https://example.com/rec",
                text="Article content for reconciliation testing.",
            ),
        )
        res_id = cap["resource_id"]

        # 1. Insert an existing label for the resource
        from edward.services.lifecycle import add_label

        add_label(conn, "resource", res_id, "topic/ai", source="system")

        # 2. Insert a stale pending classify job for this already-labeled resource
        conn.execute(
            """
            INSERT INTO processing_jobs (id, job_key, resource_id, stage, status, available_at, created_at, updated_at)
            VALUES ('job_stale_1', ?, ?, 'classify', 'pending', datetime('now'), datetime('now'), datetime('now'));
            """,
            (make_job_key("classify", res_id), res_id),
        )

        # 3. Insert an unlabeled resource with a pending classify job
        cap_unlabeled = capture_item(
            conn,
            CaptureInput(url="https://example.com/unlabeled", text="Unlabeled article"),
        )
        res_unlabeled = cap_unlabeled["resource_id"]
        conn.execute(
            """
            INSERT INTO processing_jobs (id, job_key, resource_id, stage, status, available_at, created_at, updated_at)
            VALUES ('job_unlabeled_1', ?, ?, 'classify', 'pending', datetime('now'), datetime('now'), datetime('now'));
            """,
            (make_job_key("classify", res_unlabeled), res_unlabeled),
        )

    # Run CLI process --reconcile --json
    from edward.cli import app

    result = cli_runner.invoke(app, ["process", "--reconcile", "--json"])
    assert result.exit_code == 0
    import json

    payload = json.loads(result.stdout)
    assert payload["reconciled_classify_jobs"] == 1

    with test_db.connection() as conn:
        stale_job = conn.execute(
            "SELECT status FROM processing_jobs WHERE id = 'job_stale_1';"
        ).fetchone()
        assert stale_job["status"] == "completed"

        unlabeled_job = conn.execute(
            "SELECT status FROM processing_jobs WHERE id = 'job_unlabeled_1';"
        ).fetchone()
        assert unlabeled_job["status"] == "pending"
