"""Tests verifying Phase 2 Round 6/7 review findings:
1. Canonical classification input hash computed from full text including summary, detecting in-flight summary changes.
2. Unleased pending jobs update capture_id provenance when re-ingested or captured again.
3. Leased running jobs are protected against status overwrites during content storage or bundle upsert.
"""

import datetime
import hashlib

from edward.blobs import BlobStore
from edward.db import Database
from edward.models import ResearchBundle, SourceItem, generate_id
from edward.services.bundle import import_research_bundle
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


def _insert_test_capture(conn, cap_id: str, res_id: str | None = None) -> None:
    now_iso = datetime.datetime.now(datetime.UTC).isoformat()
    conn.execute(
        """
        INSERT INTO captures (
            id, origin_namespace, origin_id, collection_channel, collector,
            acquisition_method, retrieved_at, raw_content, user_note, review_state, created_at, updated_at
        ) VALUES (?, 'test', ?, 'test', 'test', 'manual', ?, 'content', 'note', 'unreviewed', ?, ?);
        """,
        (cap_id, cap_id, now_iso, now_iso, now_iso),
    )
    if res_id:
        conn.execute(
            """
            INSERT INTO capture_resources (capture_id, resource_id, relationship_type, created_at)
            VALUES (?, ?, 'primary', ?);
            """,
            (cap_id, res_id, now_iso),
        )


def test_in_flight_summary_update_with_identical_clean_text_invalidates_classification(
    test_db: Database, test_blob_store: BlobStore, monkeypatch
):
    """Classification detects in-flight summary updates even when clean_text is identical, and job input_hash matches judgments."""
    monkeypatch.setenv("EDWARD_CLASSIFIER_PROVIDER", "dry-run")
    now_iso = datetime.datetime.now(datetime.UTC).isoformat()
    res_id = generate_id("res")
    clean_text = "The quick brown fox jumps over the lazy dog."
    summary_v1 = "Summary Version 1: Fox jumps."
    summary_v2 = "Summary Version 2: Animal jumping analysis."

    # 1. Store initial content with summary v1 and create pending classify job
    with test_db.transaction() as conn:
        conn.execute(
            """
            INSERT INTO resources (id, identity_key, canonical_url, title, primary_form, review_state, is_deleted, created_at, updated_at)
            VALUES (?, 'url:https://example.com/summary-invalidation', 'https://example.com/summary-invalidation', 'Fox Article', 'article', 'unreviewed', 0, ?, ?);
            """,
            (res_id, now_iso, now_iso),
        )
        store_resource_content(
            conn,
            res_id,
            clean_text=clean_text,
            summary=summary_v1,
        )
        conn.execute(
            """
            INSERT INTO processing_jobs (
                id, job_key, resource_id, stage, status, attempts, max_attempts, available_at, created_at, updated_at
            ) VALUES ('job_cls_summary_race', 'classify:' || ?, ?, 'classify', 'pending', 0, 3, ?, ?, ?);
            """,
            (res_id, res_id, now_iso, now_iso, now_iso),
        )

    # 2. Worker claims job on summary v1
    with test_db.transaction() as conn:
        job = claim_job(conn, worker_id="worker_sum", stage="classify")
        assert job is not None
        assert job["id"] == "job_cls_summary_race"
        ctx = _load_job_context(conn, job)

    expected_old_text = f"{summary_v1}\n\n{clean_text}"
    expected_old_hash = hashlib.sha256(expected_old_text.encode("utf-8")).hexdigest()
    assert ctx["input_hash"] == expected_old_hash

    # 3. Worker performs classification outside transaction
    work_res = _perform_job_work(test_blob_store, job, ctx)
    assert work_res["input_hash"] == expected_old_hash

    # 4. In-flight update: Same clean_text, but updated summary v2
    with test_db.transaction() as conn:
        store_resource_content(
            conn,
            res_id,
            clean_text=clean_text,
            summary=summary_v2,
        )

    # 5. Worker attempts to persist stale classification result
    with test_db.transaction() as conn:
        outcome = _persist_job_result(
            conn,
            test_blob_store,
            job,
            work_res,
            job_error=None,
            worker_id="worker_sum",
            context=ctx,
        )
        assert outcome == "pending"

    # 6. Verify stale judgments were NOT written and job was reset to pending
    with test_db.connection() as conn:
        old_jdgs = conn.execute(
            "SELECT * FROM judgments WHERE object_id = ? AND input_content_hash = ?;",
            (res_id, expected_old_hash),
        ).fetchall()
        assert len(old_jdgs) == 0

        job_row = conn.execute(
            "SELECT status, lease_owner FROM processing_jobs WHERE id = 'job_cls_summary_race';"
        ).fetchone()
        assert job_row["status"] == "pending"
        assert job_row["lease_owner"] is None

    # 7. Next loop processes the updated summary v2
    loop_res = process_pending_jobs(test_db, test_blob_store, limit=1)
    assert loop_res["completed"] == 1

    expected_new_text = f"{summary_v2}\n\n{clean_text}"
    expected_new_hash = hashlib.sha256(expected_new_text.encode("utf-8")).hexdigest()

    with test_db.connection() as conn:
        job_after = conn.execute(
            "SELECT status, input_hash FROM processing_jobs WHERE id = 'job_cls_summary_race';"
        ).fetchone()
        assert job_after["status"] == "completed"
        assert job_after["input_hash"] == expected_new_hash

        new_jdgs = conn.execute(
            "SELECT input_content_hash FROM judgments WHERE object_id = ?;",
            (res_id,),
        ).fetchall()
        assert len(new_jdgs) > 0
        for jdg in new_jdgs:
            assert jdg["input_content_hash"] == expected_new_hash
            assert jdg["input_content_hash"] == job_after["input_hash"]


def test_unleased_pending_jobs_update_capture_provenance_on_subsequent_capture(
    test_db: Database, test_blob_store: BlobStore, monkeypatch
):
    """Unleased pending jobs update capture_id so filtering by new capture_id finds and executes the job."""
    monkeypatch.setenv("EDWARD_CLASSIFIER_PROVIDER", "dry-run")
    now_iso = datetime.datetime.now(datetime.UTC).isoformat()
    url = "https://example.com/unleased-pending-test"
    res_id = generate_id("res")
    cap1_id = generate_id("cap")

    with test_db.transaction() as conn:
        _insert_test_capture(conn, cap1_id)
        conn.execute(
            """
            INSERT INTO resources (id, identity_key, canonical_url, title, primary_form, review_state, is_deleted, created_at, updated_at)
            VALUES (?, 'url:' || ?, ?, 'Article', 'article', 'unreviewed', 0, ?, ?);
            """,
            (res_id, url, url, now_iso, now_iso),
        )
        store_resource_content(
            conn,
            res_id,
            clean_text="Content for provenance test",
            capture_id=cap1_id,
        )
        # Create unleased pending classify job tied to capture 1
        conn.execute(
            """
            INSERT INTO processing_jobs (
                id, job_key, capture_id, resource_id, stage, status, attempts, max_attempts, available_at, created_at, updated_at
            ) VALUES ('job_prov_test', 'classify:' || ?, ?, ?, 'classify', 'pending', 0, 3, ?, ?, ?);
            """,
            (res_id, cap1_id, res_id, now_iso, now_iso, now_iso),
        )

    # Verify initial capture_id is cap1_id
    with test_db.connection() as conn:
        row = conn.execute(
            "SELECT capture_id, status, lease_owner FROM processing_jobs WHERE id = 'job_prov_test';"
        ).fetchone()
        assert row["capture_id"] == cap1_id
        assert row["status"] == "pending"
        assert row["lease_owner"] is None

    # Ingest bundle with cap2 containing the same resource and extracted text
    bundle = ResearchBundle(
        bundle_id="bnd_prov_2",
        title="Second Research Bundle",
        sources=[
            SourceItem(
                origin="web",
                url=url,
                title="Article",
                extracted_content="Content for provenance test",
            )
        ],
    )

    with test_db.transaction() as conn:
        import_research_bundle(conn, test_blob_store, bundle.model_dump())
        cap_created = conn.execute(
            "SELECT id FROM captures WHERE origin_id = 'bnd_prov_2' LIMIT 1;"
        ).fetchone()
        assert cap_created is not None
        actual_cap2_id = cap_created["id"]

    # Verify the unleased pending job received actual_cap2_id
    with test_db.connection() as conn:
        row2 = conn.execute(
            "SELECT capture_id, status FROM processing_jobs WHERE id = 'job_prov_test';"
        ).fetchone()
        assert row2["capture_id"] == actual_cap2_id
        assert row2["status"] == "pending"

    # Processing with capture_id=actual_cap2_id successfully finds and completes the work
    res_proc = process_pending_jobs(test_db, test_blob_store, capture_id=actual_cap2_id, limit=1)
    assert res_proc["completed"] == 1


def test_extraction_stage_passes_capture_id_to_store_resource_content_and_downstream_classify(
    test_db: Database, test_blob_store: BlobStore
):
    """Extract stage persists capture_id into store_resource_content and downschedules classify job with capture_id."""
    now_iso = datetime.datetime.now(datetime.UTC).isoformat()
    res_id = generate_id("res")
    cap_id = generate_id("cap")

    with test_db.transaction() as conn:
        _insert_test_capture(conn, cap_id)
        conn.execute(
            """
            INSERT INTO resources (id, identity_key, canonical_url, title, primary_form, review_state, is_deleted, created_at, updated_at)
            VALUES (?, 'url:https://example.com/ext-cap', 'https://example.com/ext-cap', 'Extract Cap', 'article', 'unreviewed', 0, ?, ?);
            """,
            (res_id, now_iso, now_iso),
        )
        conn.execute(
            """
            INSERT INTO processing_jobs (
                id, job_key, capture_id, resource_id, stage, status, attempts, max_attempts, lease_owner, lease_expires_at, available_at, created_at, updated_at
            ) VALUES ('job_ext_cap', 'extract:' || ?, ?, ?, 'extract', 'running', 0, 3, 'worker_ext', ?, ?, ?, ?);
            """,
            (
                res_id,
                cap_id,
                res_id,
                (datetime.datetime.now(datetime.UTC) + datetime.timedelta(minutes=5)).isoformat(),
                now_iso,
                now_iso,
                now_iso,
            ),
        )

    ext_res = ExtractionResult(
        status="completed",
        title="Extract Cap",
        clean_text="Clean text extracted by worker.",
        summary="Summary of clean text.",
        extractor="local",
        extractor_version="1.0",
    )
    job = {
        "id": "job_ext_cap",
        "job_key": f"extract:{res_id}",
        "capture_id": cap_id,
        "resource_id": res_id,
        "stage": "extract",
        "attempts": 0,
        "max_attempts": 3,
        "lease_owner": "worker_ext",
    }

    with test_db.transaction() as conn:
        outcome = _persist_job_result(
            conn,
            test_blob_store,
            job,
            work_result={"status": "completed", "ext_result": ext_res},
            job_error=None,
            worker_id="worker_ext",
        )
        assert outcome == "completed"

    with test_db.connection() as conn:
        cls_job = conn.execute(
            "SELECT capture_id, status FROM processing_jobs WHERE job_key = 'classify:' || ?;",
            (res_id,),
        ).fetchone()
        assert cls_job is not None
        assert cls_job["status"] == "pending"
        assert cls_job["capture_id"] == cap_id


def test_leased_running_job_is_protected_from_being_reset_by_store_resource_content(
    test_db: Database,
):
    """Active running job with valid lease is never touched or reset by store_resource_content."""
    now_iso = datetime.datetime.now(datetime.UTC).isoformat()
    future_iso = (datetime.datetime.now(datetime.UTC) + datetime.timedelta(minutes=5)).isoformat()
    res_id = generate_id("res")
    cap1_id = generate_id("cap")
    cap2_id = generate_id("cap")

    with test_db.transaction() as conn:
        _insert_test_capture(conn, cap1_id)
        _insert_test_capture(conn, cap2_id)
        conn.execute(
            """
            INSERT INTO resources (id, identity_key, canonical_url, title, primary_form, review_state, is_deleted, created_at, updated_at)
            VALUES (?, 'url:https://example.com/fenced-store', 'https://example.com/fenced-store', 'Fenced Store', 'article', 'unreviewed', 0, ?, ?);
            """,
            (res_id, now_iso, now_iso),
        )
        conn.execute(
            """
            INSERT INTO processing_jobs (
                id, job_key, capture_id, resource_id, stage, status, attempts, max_attempts,
                lease_owner, lease_expires_at, available_at, created_at, updated_at
            ) VALUES ('job_cls_running', 'classify:' || ?, ?, ?, 'classify', 'running', 1, 3, 'active_worker_7', ?, ?, ?, ?);
            """,
            (res_id, cap1_id, res_id, future_iso, now_iso, now_iso, now_iso),
        )

        # Another process stores resource content with cap2_id
        store_resource_content(
            conn,
            res_id,
            clean_text="New content stored while classify job is actively leased and running",
            capture_id=cap2_id,
        )

    # Verify the running job was protected from lease/status reset, while receiving new capture_id
    with test_db.connection() as conn:
        row = conn.execute(
            "SELECT status, lease_owner, capture_id, attempts FROM processing_jobs WHERE id = 'job_cls_running';"
        ).fetchone()
        assert row["status"] == "running"
        assert row["lease_owner"] == "active_worker_7"
        assert row["capture_id"] == cap2_id
        assert row["attempts"] == 1


def test_in_flight_content_change_updates_capture_provenance_on_invalidation(
    test_db: Database, test_blob_store: BlobStore, monkeypatch
):
    """When content changes in flight from cap2, the job reset to pending receives cap2 provenance and can be processed by --capture-id cap2."""
    monkeypatch.setenv("EDWARD_CLASSIFIER_PROVIDER", "dry-run")
    now_iso = datetime.datetime.now(datetime.UTC).isoformat()
    res_id = generate_id("res")
    cap1_id = generate_id("cap")
    cap2_id = generate_id("cap")

    # 1. Classification starts for cap1
    with test_db.transaction() as conn:
        conn.execute(
            """
            INSERT INTO resources (id, identity_key, canonical_url, title, primary_form, review_state, is_deleted, created_at, updated_at)
            VALUES (?, 'url:https://example.com/prov-race', 'https://example.com/prov-race', 'Prov Race', 'article', 'unreviewed', 0, ?, ?);
            """,
            (res_id, now_iso, now_iso),
        )
        _insert_test_capture(conn, cap1_id, res_id=res_id)
        store_resource_content(
            conn,
            res_id,
            clean_text="Initial content for cap1",
            capture_id=cap1_id,
        )
        conn.execute(
            """
            INSERT INTO processing_jobs (
                id, job_key, capture_id, resource_id, stage, status, attempts, max_attempts, available_at, created_at, updated_at
            ) VALUES ('job_prov_race', 'classify:' || ?, ?, ?, 'classify', 'pending', 0, 3, ?, ?, ?);
            """,
            (res_id, cap1_id, res_id, now_iso, now_iso, now_iso),
        )

    # Worker 1 claims classification
    with test_db.transaction() as conn:
        job = claim_job(conn, worker_id="worker_1", stage="classify")
        assert job is not None
        ctx = _load_job_context(conn, job)

    work_res = _perform_job_work(test_blob_store, job, ctx)

    # 2. cap2 stores updated resource content while classification is running
    with test_db.transaction() as conn:
        _insert_test_capture(conn, cap2_id, res_id=res_id)
        store_resource_content(
            conn,
            res_id,
            clean_text="Updated content arriving with cap2",
            capture_id=cap2_id,
        )

    # 3. Old result detects input mismatch and resets to pending
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

    # 4. Verify the pending job now has capture_id = cap2_id
    with test_db.connection() as conn:
        row = conn.execute(
            "SELECT status, capture_id, lease_owner FROM processing_jobs WHERE id = 'job_prov_race';"
        ).fetchone()
        assert row["status"] == "pending"
        assert row["capture_id"] == cap2_id
        assert row["lease_owner"] is None

    # 5. edward process --capture-id cap2 finds and completes the work
    res_proc = process_pending_jobs(test_db, test_blob_store, capture_id=cap2_id, limit=1)
    assert res_proc["completed"] == 1


def test_in_flight_snapshot_change_updates_capture_provenance_on_invalidation(
    test_db: Database, test_blob_store: BlobStore
):
    """When snapshot changes in flight from cap2, extract job reset to pending receives cap2 provenance and completes under cap2."""
    now_iso = datetime.datetime.now(datetime.UTC).isoformat()
    res_id = generate_id("res")
    cap1_id = generate_id("cap")
    cap2_id = generate_id("cap")

    # 1. Extraction starts for cap1
    with test_db.transaction() as conn:
        conn.execute(
            """
            INSERT INTO resources (id, identity_key, canonical_url, title, primary_form, review_state, is_deleted, created_at, updated_at)
            VALUES (?, 'url:https://example.com/ext-prov-race', 'https://example.com/ext-prov-race', 'Ext Prov Race', 'article', 'unreviewed', 0, ?, ?);
            """,
            (res_id, now_iso, now_iso),
        )
        _insert_test_capture(conn, cap1_id, res_id=res_id)
        save_source_snapshot(
            conn,
            test_blob_store,
            res_id,
            b"<html><body><h1>Old Snap</h1></body></html>",
            {"content-type": "text/html"},
            capture_id=cap1_id,
        )
        conn.execute(
            """
            INSERT INTO processing_jobs (
                id, job_key, capture_id, resource_id, stage, status, attempts, max_attempts, available_at, created_at, updated_at
            ) VALUES ('job_ext_prov_race', 'extract:' || ?, ?, ?, 'extract', 'pending', 0, 3, ?, ?, ?);
            """,
            (res_id, cap1_id, res_id, now_iso, now_iso, now_iso),
        )

    # Worker claims extract
    with test_db.transaction() as conn:
        job = claim_job(conn, worker_id="worker_ext_prov", stage="extract")
        assert job is not None
        ctx = _load_job_context(conn, job)

    stale_ext_result = ExtractionResult(
        status="completed",
        title="Old",
        clean_text="Old Snap",
        summary="Old",
        extractor="local",
        extractor_version="1.0",
    )
    work_res = {"status": "completed", "ext_result": stale_ext_result}

    # 2. cap2 saves a new snapshot while extraction is running
    with test_db.transaction() as conn:
        _insert_test_capture(conn, cap2_id, res_id=res_id)
        save_source_snapshot(
            conn,
            test_blob_store,
            res_id,
            b"<html><body><h1>New Snap with Cap2</h1></body></html>",
            {"content-type": "text/html"},
            capture_id=cap2_id,
        )

    # 3. Worker persists stale extract result: detected as mismatch and reset to pending
    with test_db.transaction() as conn:
        outcome = _persist_job_result(
            conn,
            test_blob_store,
            job,
            work_res,
            job_error=None,
            worker_id="worker_ext_prov",
            context=ctx,
        )
        assert outcome == "pending"

    # 4. Verify the pending job now has capture_id = cap2_id
    with test_db.connection() as conn:
        row = conn.execute(
            "SELECT status, capture_id, lease_owner FROM processing_jobs WHERE id = 'job_ext_prov_race';"
        ).fetchone()
        assert row["status"] == "pending"
        assert row["capture_id"] == cap2_id
        assert row["lease_owner"] is None

    # 5. edward process --capture-id cap2 finds and completes the work
    res_proc = process_pending_jobs(test_db, test_blob_store, capture_id=cap2_id, limit=1)
    assert res_proc["completed"] == 1


def test_save_unchanged_snapshot_preserves_completed_extraction_cache(
    test_db: Database, test_blob_store: BlobStore
):
    """Saving an identical snapshot preserves completed extraction cache and updates capture provenance without resetting to pending."""
    now_iso = datetime.datetime.now(datetime.UTC).isoformat()
    res_id = generate_id("res")
    cap1_id = generate_id("cap")
    cap2_id = generate_id("cap")
    cap3_id = generate_id("cap")

    html_v1 = b"<html><body><h1>Stable Article</h1><p>Content</p></body></html>"
    snap_hash_v1 = hashlib.sha256(html_v1).hexdigest()

    with test_db.transaction() as conn:
        conn.execute(
            """
            INSERT INTO resources (id, identity_key, canonical_url, title, primary_form, review_state, is_deleted, created_at, updated_at)
            VALUES (?, 'url:https://example.com/stable', 'https://example.com/stable', 'Stable Article', 'article', 'unreviewed', 0, ?, ?);
            """,
            (res_id, now_iso, now_iso),
        )
        _insert_test_capture(conn, cap1_id, res_id=res_id)
        save_source_snapshot(
            conn,
            test_blob_store,
            res_id,
            html_v1,
            {"content-type": "text/html"},
            capture_id=cap1_id,
        )
        # Mark extraction completed with input_hash = snap_hash_v1
        conn.execute(
            """
            INSERT INTO processing_jobs (
                id, job_key, capture_id, resource_id, stage, status, input_hash, attempts, max_attempts, available_at, created_at, updated_at
            ) VALUES ('job_ext_stable', 'extract:' || ?, ?, ?, 'extract', 'completed', ?, 0, 3, ?, ?, ?);
            """,
            (res_id, cap1_id, res_id, snap_hash_v1, now_iso, now_iso, now_iso),
        )

    # Subsequent capture cap2 saves the EXACT SAME snapshot content
    with test_db.transaction() as conn:
        _insert_test_capture(conn, cap2_id, res_id=res_id)
        save_source_snapshot(
            conn,
            test_blob_store,
            res_id,
            html_v1,
            {"content-type": "text/html"},
            capture_id=cap2_id,
        )

    # Verify the extract job remains COMPLETED, input_hash is preserved, and capture_id is cap2
    with test_db.connection() as conn:
        row = conn.execute(
            "SELECT status, capture_id, input_hash, attempts FROM processing_jobs WHERE id = 'job_ext_stable';"
        ).fetchone()
        assert row["status"] == "completed"
        assert row["input_hash"] == snap_hash_v1
        assert row["capture_id"] == cap2_id
        assert row["attempts"] == 0

    # Subsequent capture cap3 saves a DIFFERENT snapshot content
    html_v2 = b"<html><body><h1>Updated Article</h1><p>New content</p></body></html>"
    with test_db.transaction() as conn:
        _insert_test_capture(conn, cap3_id, res_id=res_id)
        save_source_snapshot(
            conn,
            test_blob_store,
            res_id,
            html_v2,
            {"content-type": "text/html"},
            capture_id=cap3_id,
        )

    # Verify the extract job was reset to PENDING with capture_id = cap3
    with test_db.connection() as conn:
        row = conn.execute(
            "SELECT status, capture_id, input_hash, attempts FROM processing_jobs WHERE id = 'job_ext_stable';"
        ).fetchone()
        assert row["status"] == "pending"
        assert row["capture_id"] == cap3_id
        assert row["attempts"] == 0


def test_running_extraction_worker_propagates_updated_capture_id_downstream(
    test_db: Database, test_blob_store: BlobStore, monkeypatch
):
    """When capture_id updates on a running extraction job, persistence uses authoritative DB capture_id downstream."""
    monkeypatch.setenv("EDWARD_CLASSIFIER_PROVIDER", "dry-run")
    now_iso = datetime.datetime.now(datetime.UTC).isoformat()
    res_id = generate_id("res")
    cap1_id = generate_id("cap")
    cap2_id = generate_id("cap")

    html = b"<html><body><h1>Downstream Prop</h1><p>Test</p></body></html>"

    with test_db.transaction() as conn:
        conn.execute(
            """
            INSERT INTO resources (id, identity_key, canonical_url, title, primary_form, review_state, is_deleted, created_at, updated_at)
            VALUES (?, 'url:https://example.com/downstream-prop', 'https://example.com/downstream-prop', 'Downstream Prop', 'article', 'unreviewed', 0, ?, ?);
            """,
            (res_id, now_iso, now_iso),
        )
        _insert_test_capture(conn, cap1_id, res_id=res_id)
        save_source_snapshot(
            conn,
            test_blob_store,
            res_id,
            html,
            {"content-type": "text/html"},
            capture_id=cap1_id,
        )
        conn.execute(
            """
            INSERT INTO processing_jobs (
                id, job_key, capture_id, resource_id, stage, status, attempts, max_attempts, available_at, created_at, updated_at
            ) VALUES ('job_ext_ds_prop', 'extract:' || ?, ?, ?, 'extract', 'pending', 0, 3, ?, ?, ?);
            """,
            (res_id, cap1_id, res_id, now_iso, now_iso, now_iso),
        )

    # Worker claims extraction with in-memory capture_id = cap1_id
    with test_db.transaction() as conn:
        job = claim_job(conn, worker_id="worker_ds_prop", stage="extract")
        assert job is not None
        assert job["capture_id"] == cap1_id
        ctx = _load_job_context(conn, job)

    ext_res = ExtractionResult(
        status="completed",
        title="Downstream Prop",
        clean_text="Clean text for downstream propagation",
        summary="Summary for downstream propagation",
        extractor="local",
        extractor_version="1.0",
    )
    work_res = {"status": "completed", "ext_result": ext_res}

    # cap2 arrives with the IDENTICAL snapshot while extraction is running
    with test_db.transaction() as conn:
        _insert_test_capture(conn, cap2_id, res_id=res_id)
        save_source_snapshot(
            conn,
            test_blob_store,
            res_id,
            html,
            {"content-type": "text/html"},
            capture_id=cap2_id,
        )

    # Worker finishes and persists result
    with test_db.transaction() as conn:
        outcome = _persist_job_result(
            conn,
            test_blob_store,
            job,
            work_res,
            job_error=None,
            worker_id="worker_ds_prop",
            context=ctx,
        )
        assert outcome == "completed"

    # Verify extract job has capture_id = cap2_id
    with test_db.connection() as conn:
        row_ext = conn.execute(
            "SELECT status, capture_id FROM processing_jobs WHERE id = 'job_ext_ds_prop';"
        ).fetchone()
        assert row_ext["status"] == "completed"
        assert row_ext["capture_id"] == cap2_id

        # Verify downstream classify job was created with capture_id = cap2_id
        cls_key = f"classify:{res_id}"
        row_cls = conn.execute(
            "SELECT status, capture_id FROM processing_jobs WHERE job_key = ?;",
            (cls_key,),
        ).fetchone()
        assert row_cls is not None
        assert row_cls["status"] == "pending"
        assert row_cls["capture_id"] == cap2_id

    # edward process --capture-id cap2 finds and completes the downstream classify job!
    res_proc = process_pending_jobs(test_db, test_blob_store, capture_id=cap2_id, limit=1)
    assert res_proc["completed"] == 1


def test_classification_hash_normalization_consistent_across_services(
    test_db: Database, test_blob_store: BlobStore, monkeypatch
):
    """Storing identical clean_text and summary does not reset completed classification job."""
    monkeypatch.setenv("EDWARD_CLASSIFIER_PROVIDER", "dry-run")
    now_iso = datetime.datetime.now(datetime.UTC).isoformat()
    res_id = generate_id("res")
    cap1_id = generate_id("cap")
    cap2_id = generate_id("cap")

    clean_text = "Detailed article text for consistent hash test."
    summary = "Article summary."

    # 1. Store content with summary
    with test_db.transaction() as conn:
        conn.execute(
            """
            INSERT INTO resources (id, identity_key, canonical_url, title, primary_form, review_state, is_deleted, created_at, updated_at)
            VALUES (?, 'url:https://example.com/norm-hash', 'https://example.com/norm-hash', 'Norm Hash Test', 'article', 'unreviewed', 0, ?, ?);
            """,
            (res_id, now_iso, now_iso),
        )
        _insert_test_capture(conn, cap1_id, res_id=res_id)
        store_resource_content(
            conn,
            res_id,
            clean_text=clean_text,
            summary=summary,
            capture_id=cap1_id,
        )
        conn.execute(
            """
            INSERT INTO processing_jobs (
                id, job_key, capture_id, resource_id, stage, status, attempts, max_attempts, available_at, created_at, updated_at
            ) VALUES ('job_cls_norm_hash', 'classify:' || ?, ?, ?, 'classify', 'pending', 0, 3, ?, ?, ?);
            """,
            (res_id, cap1_id, res_id, now_iso, now_iso, now_iso),
        )

    # 2. Complete classification using process_pending_jobs
    res_proc = process_pending_jobs(test_db, test_blob_store, capture_id=cap1_id, limit=1)
    assert res_proc["completed"] == 1

    # Verify classification completed and recorded input_hash
    with test_db.connection() as conn:
        cls_row = conn.execute(
            "SELECT status, input_hash, capture_id FROM processing_jobs WHERE job_key = ?;",
            (f"classify:{res_id}",),
        ).fetchone()
        assert cls_row["status"] == "completed"
        assert cls_row["input_hash"] is not None
        canonical_hash = cls_row["input_hash"]

    # 3. Store the IDENTICAL content and summary again for capture cap2
    with test_db.transaction() as conn:
        _insert_test_capture(conn, cap2_id, res_id=res_id)
        store_resource_content(
            conn,
            res_id,
            clean_text=clean_text,
            summary=summary,
            capture_id=cap2_id,
        )

    # 4. Verify the job remains COMPLETED, input_hash is preserved, and capture_id is cap2
    with test_db.connection() as conn:
        cls_row2 = conn.execute(
            "SELECT status, input_hash, capture_id FROM processing_jobs WHERE job_key = ?;",
            (f"classify:{res_id}",),
        ).fetchone()
        assert cls_row2["status"] == "completed"
        assert cls_row2["input_hash"] == canonical_hash
        assert cls_row2["capture_id"] == cap2_id
