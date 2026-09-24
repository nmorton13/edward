"""Regression tests for Phase 2 Round 3 review findings."""

import datetime
import hashlib
import json
from unittest.mock import patch

import pytest

from edward.blobs import BlobStore
from edward.db import Database
from edward.models import CaptureInput, ResearchBundle, SourceItem, generate_id
from edward.services.bundle import import_research_bundle
from edward.services.capture import capture_item
from edward.services.extract import ExtractionResult
from edward.services.network import FetchResult
from edward.services.processor import (
    _persist_job_result,
    process_pending_jobs,
)
from edward.services.resource import (
    get_cached_content,
    save_source_snapshot,
    store_resource_content,
)


def test_round3_finding1_url_cache_reuse_skips_fetch_and_extract(
    test_db: Database, test_blob_store: BlobStore
):
    """Finding 1: Repeated ingestion of the same URL reuses cached content and skips fetch/extract."""
    url = "https://example.com/cached-article"

    # 1. First ingestion via capture_item
    with test_db.transaction() as conn:
        cap1 = capture_item(conn, CaptureInput(url=url))
        res_id = cap1["resource_id"]
        assert res_id is not None

        # Simulate fetch and extract completion
        fake_html = b"<html><body><h1>Cached Article</h1><p>Important text.</p></body></html>"
        save_source_snapshot(
            conn, test_blob_store, res_id, fake_html, headers={"content-type": "text/html"}
        )
        store_resource_content(
            conn,
            res_id,
            clean_text="Important text.",
            summary="Cached summary",
            extractor="test",
            extractor_version="1.0",
            title="Cached Article",
        )
        # Mark fetch completed and insert completed extract job with snapshot hash
        snap_hash = hashlib.sha256(fake_html).hexdigest()
        conn.execute(
            """
            UPDATE processing_jobs
            SET status = 'completed', input_hash = ?
            WHERE resource_id = ? AND stage = 'resource-fetch';
            """,
            (snap_hash, res_id),
        )
        conn.execute(
            """
            INSERT INTO processing_jobs (
                id, job_key, capture_id, resource_id, stage, status, input_hash, available_at, attempts, created_at, updated_at
            ) VALUES ('job_ext_cached', 'extract:' || ?, ?, ?, 'extract', 'completed', ?, ?, 0, ?, ?);
            """,
            (
                res_id,
                cap1["capture_id"],
                res_id,
                snap_hash,
                datetime.datetime.now(datetime.UTC).isoformat(),
                datetime.datetime.now(datetime.UTC).isoformat(),
                datetime.datetime.now(datetime.UTC).isoformat(),
            ),
        )

    # 2. Second ingestion: New research bundle containing the same URL without new extracted text
    bundle = ResearchBundle(
        bundle_id="bnd_repeat_test",
        title="Bundle with repeat URL",
        sources=[
            SourceItem(
                origin="web",
                url=url,
                title="Cached Article",
            )
        ],
    )

    with test_db.transaction() as conn:
        import_result = import_research_bundle(conn, test_blob_store, bundle.model_dump())
        assert import_result["bundle_id"] == "bnd_repeat_test"

    # 3. Verify no fetch or extract job was queued for this resource; only classify is queued
    with test_db.connection() as conn:
        jobs = conn.execute(
            """
            SELECT stage, status FROM processing_jobs
            WHERE resource_id = ?;
            """,
            (res_id,),
        ).fetchall()
        stages = {j["stage"]: j["status"] for j in jobs}
        # Fetch and extract remain completed; classify is queued (pending)
        assert stages.get("resource-fetch") == "completed"
        assert stages.get("extract") == "completed"
        assert stages.get("classify") == "pending"


def test_round3_finding2_transcript_preserved_in_content_and_chunks(
    test_db: Database, test_blob_store: BlobStore
):
    """Finding 2: Transcripts from extraction are preserved in resource_contents and resource_chunks with locators."""
    now_iso = datetime.datetime.now(datetime.UTC).isoformat()
    res_id = generate_id("res")
    with test_db.transaction() as conn:
        conn.execute(
            """
            INSERT INTO resources (id, identity_key, canonical_url, title, review_state, is_deleted, created_at, updated_at)
            VALUES (?, 'url:https://example.com/video', 'https://example.com/video', 'Video Lecture', 'unreviewed', 0, ?, ?);
            """,
            (res_id, now_iso, now_iso),
        )
        future_iso = (
            datetime.datetime.now(datetime.UTC) + datetime.timedelta(seconds=60)
        ).isoformat()
        conn.execute(
            """
            INSERT INTO processing_jobs (
                id, job_key, resource_id, stage, status, available_at, attempts, lease_owner, lease_expires_at, created_at, updated_at
            ) VALUES ('job_ext_trans', 'extract:' || ?, ?, 'extract', 'running', ?, 0, 'worker_1', ?, ?, ?);
            """,
            (res_id, res_id, now_iso, future_iso, now_iso, now_iso),
        )

    segments = [
        {"start": 0.0, "end": 12.5, "speaker": "Alice", "text": "Welcome to the lecture."},
        {"start": 12.5, "end": 30.0, "speaker": "Bob", "text": "Today we discuss research memory."},
    ]
    ext_result = ExtractionResult(
        status="completed",
        clean_text="Summary of lecture topic.",
        transcript=json.dumps(segments),
        summary="Lecture summary",
        extractor="audio-transcribe",
        extractor_version="1.0",
        title="Video Lecture",
    )

    job = {
        "id": "job_ext_trans",
        "job_key": f"extract:{res_id}",
        "resource_id": res_id,
        "stage": "extract",
        "attempts": 0,
        "max_attempts": 3,
        "lease_owner": "worker_1",
    }

    with test_db.transaction() as conn:
        outcome = _persist_job_result(
            conn,
            test_blob_store,
            job,
            work_result={"ext_result": ext_result, "cached": False},
            job_error=None,
            worker_id="worker_1",
        )
        assert outcome == "completed"

    with test_db.connection() as conn:
        # Check clean_text includes transcript section
        content = get_cached_content(conn, res_id)
        assert content is not None
        assert "## Transcript" in content["clean_text"]
        assert "Welcome to the lecture." in content["clean_text"]

        # Check resource_chunks rows with locators
        chunks = conn.execute(
            """
            SELECT chunk_index, text, locator_json FROM resource_chunks
            WHERE resource_id = ? ORDER BY chunk_index ASC;
            """,
            (res_id,),
        ).fetchall()
        assert len(chunks) == 2
        assert chunks[0]["text"] == "Welcome to the lecture."
        loc0 = json.loads(chunks[0]["locator_json"])
        assert loc0["speaker"] == "Alice"
        assert loc0["start"] == 0.0
        assert chunks[1]["text"] == "Today we discuss research memory."
        loc1 = json.loads(chunks[1]["locator_json"])
        assert loc1["speaker"] == "Bob"
        assert loc1["end"] == 30.0


def test_round3_finding3_requeued_jobs_update_capture_id(
    test_db: Database, test_blob_store: BlobStore
):
    """Finding 3: Requeued jobs update capture_id so filtering by new capture_id finds the work."""
    url = "https://example.com/shared-provenance"

    # First capture
    with test_db.transaction() as conn:
        cap1 = capture_item(conn, CaptureInput(url=url))
        res_id = cap1["resource_id"]
        cap1_id = cap1["capture_id"]

    with test_db.connection() as conn:
        job = conn.execute(
            "SELECT capture_id FROM processing_jobs WHERE resource_id = ?;", (res_id,)
        ).fetchone()
        assert job["capture_id"] == cap1_id

    # Second capture with different intent/note for same resource
    with test_db.transaction() as conn:
        cap2 = capture_item(conn, CaptureInput(url=url, user_note="Second capture note"))
        cap2_id = cap2["capture_id"]
        assert cap2_id != cap1_id

    # Verify that the requeued job's capture_id was updated to cap2_id
    with test_db.connection() as conn:
        job2 = conn.execute(
            "SELECT capture_id FROM processing_jobs WHERE resource_id = ?;", (res_id,)
        ).fetchone()
        assert job2["capture_id"] == cap2_id

    # Verify processing with capture_id=cap2_id finds the job
    fake_body = b"<html><body>Content</body></html>"
    mock_fetch = FetchResult(
        url=url,
        final_url=url,
        status_code=200,
        headers={"content-type": "text/html"},
        body=fake_body,
        content_hash=hashlib.sha256(fake_body).hexdigest(),
        elapsed_seconds=0.1,
    )
    with patch("edward.services.processor.safe_fetch_url", return_value=mock_fetch):
        res = process_pending_jobs(test_db, test_blob_store, capture_id=cap2_id, limit=1)
        assert res["completed"] == 1


def test_round3_finding4_accurate_outcome_counts(test_db: Database, test_blob_store: BlobStore):
    """Finding 4: process_pending_jobs accurately reports completed, pending, lost_lease, and failed counts."""
    now_iso = datetime.datetime.now(datetime.UTC).isoformat()
    res_id = generate_id("res")

    # 1. Pending extraction: status should be counted as pending, NOT failed
    with test_db.transaction() as conn:
        conn.execute(
            """
            INSERT INTO resources (id, identity_key, canonical_url, title, review_state, is_deleted, created_at, updated_at)
            VALUES (?, 'url:https://example.com/pend-count', 'https://example.com/pend-count', 'Pending Title', 'unreviewed', 0, ?, ?);
            """,
            (res_id, now_iso, now_iso),
        )
        conn.execute(
            """
            INSERT INTO processing_jobs (id, job_key, resource_id, stage, status, available_at, attempts, created_at, updated_at)
            VALUES ('job_pend_cnt', 'extract:' || ?, ?, 'extract', 'pending', ?, 0, ?, ?);
            """,
            (res_id, res_id, now_iso, now_iso, now_iso),
        )

    with patch(
        "edward.services.processor.extract_content", return_value=ExtractionResult(status="pending")
    ):
        result = process_pending_jobs(test_db, test_blob_store, limit=1)
        assert result["completed"] == 0
        assert result["failed"] == 0
        assert result["pending"] == 1
        assert result["lost_lease"] == 0


def test_round3_finding5_agent_content_hash_validation(
    test_db: Database, test_blob_store: BlobStore
):
    """Finding 5: Agent-supplied content_hash is validated against extracted_text or snapshot."""
    valid_text = "Verified extracted text content."
    expected_hash = hashlib.sha256(valid_text.encode("utf-8")).hexdigest()

    # Case 1: Matching content_hash succeeds
    bundle_valid = ResearchBundle(
        bundle_id="bnd_valid_hash",
        title="Valid Hash Bundle",
        sources=[
            SourceItem(
                origin="web",
                url="https://example.com/hash-valid",
                title="Valid Hash",
                extracted_text=valid_text,
                content_hash=expected_hash,
            )
        ],
    )
    with test_db.transaction() as conn:
        res_valid = import_research_bundle(conn, test_blob_store, bundle_valid.model_dump())
        assert res_valid["bundle_id"] == "bnd_valid_hash"

    # Case 2: Mismatched content_hash raises ValueError
    bundle_invalid = ResearchBundle(
        bundle_id="bnd_invalid_hash",
        title="Invalid Hash Bundle",
        sources=[
            SourceItem(
                origin="web",
                url="https://example.com/hash-invalid",
                title="Invalid Hash",
                extracted_text=valid_text,
                content_hash="deadbeef1234567890abcdefdeadbeef1234567890abcdefdeadbeef12345678",
            )
        ],
    )
    with pytest.raises(ValueError, match="does not match computed hash"):
        with test_db.transaction() as conn:
            import_research_bundle(conn, test_blob_store, bundle_invalid.model_dump())
