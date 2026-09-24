"""Job claiming must be atomic across concurrent worker processes.

The original claim implementation did a plain SELECT followed by a guarded
UPDATE. Two worker processes could both read the same pending row and both run
the job, which is how concurrent OCR runs produced duplicate chunks. These
tests pin the property that makes that impossible.
"""

import datetime
import sqlite3

from edward.models import make_job_key
from edward.services.ocr import OCR_STAGE
from edward.services.processor import claim_job


def _seed_job(conn, job_id="job_1", stage=OCR_STAGE, capture_id="cap-1"):
    now = datetime.datetime.now(datetime.UTC).isoformat()
    conn.execute(
        """
        INSERT INTO processing_jobs (
            id, job_key, capture_id, stage, status, available_at,
            attempts, created_at, updated_at
        ) VALUES (?, ?, ?, ?, 'pending', ?, 0, ?, ?);
        """,
        (job_id, make_job_key(stage, job_id), capture_id, stage, now, now, now),
    )


def _seed_capture(conn, capture_id="cap-1"):
    conn.execute(
        """
        INSERT OR IGNORE INTO captures (
            id, origin_namespace, origin_id, collection_channel, collector,
            acquisition_method, retrieved_at, created_at, updated_at
        ) VALUES (?, 'x', 't1', 'birdclaw', 'adapter', 'birdclaw-sqlite',
                  '2026-01-01', '2026-01-01', '2026-01-01');
        """,
        (capture_id,),
    )


def test_claim_returns_the_job_and_marks_it_running(test_db):
    with test_db.transaction() as conn:
        _seed_capture(conn)
        _seed_job(conn)

    with test_db.transaction() as conn:
        claimed = claim_job(conn, "worker-a", stage=OCR_STAGE)

    assert claimed is not None
    assert claimed["id"] == "job_1"
    assert claimed["lease_owner"] == "worker-a"
    with test_db.connection() as conn:
        row = conn.execute("SELECT status, lease_owner FROM processing_jobs;").fetchone()
    assert row["status"] == "running"
    assert row["lease_owner"] == "worker-a"


def test_a_second_worker_cannot_claim_the_same_job(test_db):
    """The core race: one pending job, two workers, exactly one winner."""
    with test_db.transaction() as conn:
        _seed_capture(conn)
        _seed_job(conn)

    with test_db.transaction() as conn:
        first = claim_job(conn, "worker-a", stage=OCR_STAGE)
    with test_db.transaction() as conn:
        second = claim_job(conn, "worker-b", stage=OCR_STAGE)

    assert first is not None
    assert second is None, "a job already leased must not be claimed again"


def test_each_concurrent_worker_gets_a_distinct_job(test_db):
    """With N jobs and N workers, every worker must get a different job."""
    with test_db.transaction() as conn:
        _seed_capture(conn)
        for i in range(5):
            _seed_job(conn, job_id=f"job_{i}")

    claimed_ids = []
    for worker in ("w1", "w2", "w3", "w4", "w5"):
        with test_db.transaction() as conn:
            job = claim_job(conn, worker, stage=OCR_STAGE)
            claimed_ids.append(job["id"] if job else None)

    assert None not in claimed_ids
    assert len(set(claimed_ids)) == 5, f"workers shared jobs: {claimed_ids}"


def test_claiming_nothing_returns_none(test_db):
    with test_db.transaction() as conn:
        assert claim_job(conn, "worker-a", stage=OCR_STAGE) is None


def test_claim_respects_stage_filter(test_db):
    with test_db.transaction() as conn:
        _seed_capture(conn)
        _seed_job(conn, job_id="job_embed", stage="embed")

    with test_db.transaction() as conn:
        assert claim_job(conn, "w", stage=OCR_STAGE) is None
    with test_db.transaction() as conn:
        assert claim_job(conn, "w", stage="embed")["id"] == "job_embed"


def test_claim_rejects_an_unsupported_stage(test_db):
    with test_db.transaction() as conn:
        assert claim_job(conn, "w", stage="not-a-stage") is None


def test_expired_lease_is_recovered_and_reclaimable(test_db):
    """Crash recovery: a dead worker's lease must not strand the job."""
    with test_db.transaction() as conn:
        _seed_capture(conn)
        _seed_job(conn)
    with test_db.transaction() as conn:
        claim_job(conn, "dead-worker", stage=OCR_STAGE, lease_seconds=0)

    with test_db.transaction() as conn:
        reclaimed = claim_job(conn, "live-worker", stage=OCR_STAGE)

    assert reclaimed is not None
    assert reclaimed["lease_owner"] == "live-worker"


def test_claim_is_a_single_statement(test_db):
    """Guards the regression: selection and leasing must not be separable."""
    import inspect

    source = inspect.getsource(claim_job)

    assert "RETURNING" in source, "the claim must select and lease in one statement"
    assert "UPDATE processing_jobs" in source
    # A guarded UPDATE keyed on a previously-SELECTed id is the TOCTOU shape.
    assert "WHERE id = ? AND status = 'pending'" not in source


def test_raw_sqlite_connection_can_claim(test_db):
    """claim_job takes a raw connection as well as a transaction context."""
    with test_db.transaction() as conn:
        _seed_capture(conn)
        _seed_job(conn)

    conn = test_db.connect()
    try:
        conn.row_factory = sqlite3.Row
        claimed = claim_job(conn, "w", stage=OCR_STAGE)
        assert claimed is not None
        conn.commit()
    finally:
        conn.close()
