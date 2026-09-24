"""Comprehensive regression test suite for Phase 2 review findings."""

import socket
from pathlib import Path

import pytest

from edward.blobs import BlobStore
from edward.classifiers.base import ClassificationRequest
from edward.classifiers.dry_run import DryRunClassifier
from edward.classifiers.system_one import ChoiceAnswer, NoulAnswer, ScoreAnswer
from edward.db import Database
from edward.services.bundle import import_research_bundle, ingest_markdown_report
from edward.services.classification import (
    get_classifier,
    load_thresholds,
    run_classification_pipeline,
)
from edward.services.extract import ExtractionResult
from edward.services.network import SSRFError, SSRFSafeSyncBackend
from edward.services.processor import process_pending_jobs
from edward.services.subprocess_runner import OutputLimitExceededError, run_tool


@pytest.fixture
def test_db(tmp_path: Path) -> Database:
    db = Database(tmp_path / "test.db")
    db.run_migrations()
    return db


@pytest.fixture
def test_blob_store(tmp_path: Path) -> BlobStore:
    return BlobStore(tmp_path / "blobs")


def test_bundle_entity_normalized_name(test_db: Database, test_blob_store: BlobStore):
    """Finding 4: Entity-bearing bundles insert normalized_name and link object_entities cleanly."""
    bundle_data = {
        "type": "research-bundle",
        "schema_version": "1",
        "bundle_id": "bun_entity_test",
        "title": "Cloudflare Architecture Research",
        "sources": [
            {
                "origin": "web",
                "url": "https://example.com/cf",
                "title": "Cloudflare Docs",
                "extracted_text": "Cloudflare Workers edge platform",
            }
        ],
        "findings": [
            {
                "statement": "Cloudflare Workers run on V8 isolates.",
                "assertion_role": "source-claim",
                "source_url": "https://example.com/cf",
                "entities": ["Cloudflare", "  Cloudflare Workers  "],
            }
        ],
    }

    with test_db.transaction() as conn:
        res = import_research_bundle(conn, test_blob_store, bundle_data)
        assert res["status"] == "imported"

    with test_db.connection() as conn:
        entities = conn.execute("SELECT * FROM entities;").fetchall()
        assert len(entities) == 2
        norm_names = {e["normalized_name"] for e in entities}
        assert "cloudflare" in norm_names
        assert "cloudflare workers" in norm_names

        obj_entities = conn.execute("SELECT * FROM object_entities;").fetchall()
        assert len(obj_entities) == 2


def test_bundle_inactive_agent_intents(test_db: Database, test_blob_store: BlobStore):
    """Finding 3: Agent suggestions remain inactive (is_active=0) with annotations."""
    bundle_data = {
        "type": "research-bundle",
        "schema_version": "1",
        "bundle_id": "bun_intents_test",
        "title": "Agent Suggested Intents Research",
        "suggested_intents": ["essay-seed", "future-reference"],
        "sources": [
            {
                "origin": "web",
                "url": "https://example.com/test",
                "title": "Test",
                "extracted_text": "Sample text",
            }
        ],
        "findings": [
            {
                "statement": "Sample finding with agent intent.",
                "assertion_role": "agent-conclusion",
                "intents": ["follow-up"],
            }
        ],
    }

    with test_db.transaction() as conn:
        res = import_research_bundle(conn, test_blob_store, bundle_data)
        cap_id = res["capture_id"]

    with test_db.connection() as conn:
        intents = conn.execute("SELECT * FROM intents WHERE object_id = ?;", (cap_id,)).fetchall()
        assert len(intents) == 2
        for it in intents:
            assert it["is_active"] == 0
            assert it["source"] == "agent"

        ann = conn.execute(
            "SELECT * FROM annotations WHERE object_id = ? AND annotation_type = 'suggested-intent';",
            (cap_id,),
        ).fetchall()
        assert len(ann) == 2


def test_bundle_idempotency_default_bundle_id(test_db: Database, test_blob_store: BlobStore):
    """Finding 6: bundle_id serves as default idempotency key when none is passed."""
    bundle_data = {
        "type": "research-bundle",
        "schema_version": "1",
        "bundle_id": "stable_bundle_99",
        "title": "Stable Bundle",
        "sources": [
            {
                "origin": "web",
                "url": "https://example.com/stable",
                "title": "Stable",
                "extracted_text": "Stable text",
            }
        ],
    }

    with test_db.transaction() as conn:
        res1 = import_research_bundle(conn, test_blob_store, bundle_data)
        assert res1["status"] == "imported"

    with test_db.transaction() as conn:
        res2 = import_research_bundle(conn, test_blob_store, bundle_data)
        assert res2["status"] == "replayed"
        assert res2["capture_id"] == res1["capture_id"]

    with test_db.connection() as conn:
        captures_count = conn.execute("SELECT COUNT(*) FROM captures;").fetchone()[0]
        assert captures_count == 1


def test_bundle_snapshot_path_rejected_and_inline_snapshot_routes(
    test_db: Database, test_blob_store: BlobStore, tmp_path: Path
):
    """Finding 2 & 4: snapshot_path is rejected for security; inline snapshot routes to extract, URL to fetch."""
    html_file = tmp_path / "sample.html"
    html_file.write_text("<html><body><h1>From Disk</h1></body></html>", encoding="utf-8")

    # 1. snapshot_path is rejected
    invalid_bundle = {
        "type": "research-bundle",
        "schema_version": "1",
        "bundle_id": "bun_snap_path_rejected",
        "title": "Snapshot Path Test",
        "sources": [
            {
                "origin": "web",
                "url": "https://example.com/has-snapshot-path",
                "snapshot_path": str(html_file),
            }
        ],
    }
    with pytest.raises(Exception, match="snapshot_path"):
        with test_db.transaction() as conn:
            import_research_bundle(conn, test_blob_store, invalid_bundle)

    # 2. Inline snapshot routes to extract, and URL-only routes to resource-fetch
    valid_bundle = {
        "type": "research-bundle",
        "schema_version": "1",
        "bundle_id": "bun_snap_inline_test",
        "title": "Inline Snapshot Test",
        "sources": [
            {
                "origin": "web",
                "url": "https://example.com/has-inline-snapshot",
                "snapshot": "<html><body><h1>Inline HTML</h1></body></html>",
            },
            {
                "origin": "web",
                "url": "https://example.com/needs-fetch",
            },
            {
                "origin": "web",
                "url": "https://example.com/has-text",
                "extracted_text": "Extracted prose text",
            },
        ],
    }

    with test_db.transaction() as conn:
        res = import_research_bundle(conn, test_blob_store, valid_bundle)
        assert res["status"] == "imported"

    with test_db.connection() as conn:
        # Verify inline snapshot was saved
        snapshots = conn.execute("SELECT * FROM source_snapshots;").fetchall()
        assert len(snapshots) == 1
        raw_bytes = test_blob_store.read_bytes(snapshots[0]["content_hash"])
        assert b"Inline HTML" in raw_bytes

        # Verify job routing: extract for snapshot, fetch for url-only, classify for extracted_text
        ext_jobs = conn.execute("SELECT * FROM processing_jobs WHERE stage = 'extract';").fetchall()
        assert len(ext_jobs) == 1

        fetch_jobs = conn.execute(
            "SELECT * FROM processing_jobs WHERE stage = 'resource-fetch';"
        ).fetchall()
        assert len(fetch_jobs) == 1

        cls_jobs = conn.execute(
            "SELECT * FROM processing_jobs WHERE stage = 'classify';"
        ).fetchall()
        assert len(cls_jobs) == 1


def test_markdown_report_queues_finding_extraction(test_db: Database, test_blob_store: BlobStore):
    """Finding 5: Ingesting markdown report queues a durable pending finding-extraction job."""
    with test_db.transaction() as conn:
        res = ingest_markdown_report(
            conn,
            markdown_text="# Autonomous Systems Analysis\nKey insights here.",
            title="Autonomous Systems",
            blob_store=test_blob_store,
        )
        assert res["status"] == "imported"

    with test_db.connection() as conn:
        jobs = conn.execute(
            "SELECT * FROM processing_jobs WHERE stage = 'finding-extraction';"
        ).fetchall()
        assert len(jobs) == 1
        assert jobs[0]["status"] == "pending"
        assert jobs[0]["resource_id"] == res["resource_id"]


def test_dry_run_topic_signal_entries():
    """Finding 8: Dry-run classifier parses topic and signal registries with 'entries'."""
    classifier = DryRunClassifier()
    req = ClassificationRequest(
        record_id="rec_test",
        object_type="resource",
        text="A detailed deep learning tutorial on quantum computing algorithms.",
        metadata={"title": "Quantum Deep Learning", "url": "https://arxiv.org/abs/2601.12345"},
    )
    result = classifier.classify(req)
    judgments = result.judgments
    topics = [j for j in judgments if j.family == "topic"]
    signals = [j for j in judgments if j.family == "signal"]
    assert len(topics) > 0
    assert len(signals) > 0


def test_load_thresholds_policies():
    """Finding 9: Thresholds parse policies -> balanced-precision -> thresholds."""
    thresholds = load_thresholds("balanced-precision")
    assert isinstance(thresholds, dict)
    assert len(thresholds) > 0
    assert "primary-form" in thresholds or any("topic" in k for k in thresholds)


def test_deterministic_classification_runs_when_disabled(test_db: Database):
    """Finding 10: Form classification runs even when classifier provider is disabled."""
    with test_db.transaction() as conn:
        conn.execute(
            """
            INSERT INTO resources (id, identity_key, canonical_url, title, primary_form, review_state, is_deleted, created_at, updated_at)
            VALUES ('res_det', 'url:arxiv', 'https://arxiv.org/abs/2601.00001', 'Arxiv Paper', 'other', 'unreviewed', 0, '2026-01-01', '2026-01-01');
            """
        )
        run_classification_pipeline(conn, "resource", "res_det", provider="disabled")

    with test_db.connection() as conn:
        row = conn.execute("SELECT primary_form FROM resources WHERE id = 'res_det';").fetchone()
        assert row["primary_form"] == "paper"

    # Unknown provider raises ValueError
    with pytest.raises(ValueError, match="Unknown classifier provider"):
        get_classifier("invalid-unknown-provider")


def test_jev_system_one_models():
    """Finding 11: System One models support Jev wire contract and compatibility accessors."""
    c_ans = ChoiceAnswer(choice="article", probabilities={"article": 0.9, "essay": 0.1})
    assert c_ans.selected == "article"
    assert c_ans.choice == "article"

    n_ans = NoulAnswer(noul=0.88)
    assert n_ans.noul == 0.88
    assert n_ans.probability == 0.88

    s_ans = ScoreAnswer(score=0.92, max_score=1.0)
    assert s_ans.score == 0.92


def test_dns_rebinding_defense():
    """Finding 13: SSRFSafeSyncBackend blocks connections to unsafe IPs even if hostname resolution was spoofed."""
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]

    backend = SSRFSafeSyncBackend()
    try:
        with pytest.raises(SSRFError, match="Blocked connection to unsafe IP"):
            backend.connect_tcp("127.0.0.1", port)
    finally:
        srv.close()


def test_streaming_subprocess_output_capping():
    """Finding 14: Subprocess streaming terminates immediately if output limit exceeded."""
    with pytest.raises(OutputLimitExceededError):
        # Generate 100KB with a limit of 1KB
        run_tool(
            ["python3", "-c", "import sys; sys.stdout.write('A' * 100000)"], max_output_bytes=1024
        )


def test_processor_short_transaction_and_cache(
    test_db: Database, test_blob_store: BlobStore, monkeypatch: pytest.MonkeyPatch
):
    """Finding 1 & 15: Processor executes with short transactions, consults cache, and handles extraction status."""
    # Seed resource with snapshot
    snap_hash, blob_path = test_blob_store.store_bytes(
        b"<html><body><h1>Cached Title</h1><p>Cached text body</p></body></html>"
    )
    with test_db.transaction() as conn:
        conn.execute(
            """
            INSERT INTO resources (id, identity_key, canonical_url, title, review_state, is_deleted, created_at, updated_at)
            VALUES ('res_cache', 'url:https://example.com/cache', 'https://example.com/cache', 'Cached Title', 'unreviewed', 0, '2026-01-01', '2026-01-01');
            """
        )
        conn.execute(
            """
            INSERT INTO source_snapshots (id, resource_id, content_hash, blob_path, size_bytes, created_at)
            VALUES ('snp_1', 'res_cache', ?, 'path', 100, '2026-01-01');
            """,
            (snap_hash,),
        )
        # Store initial content in cache
        conn.execute(
            """
            INSERT INTO resource_contents (id, resource_id, content_hash, clean_text, extractor, extractor_version, char_count, created_at)
            VALUES ('rc_1', 'res_cache', ?, 'Cached text body', 'cache-test', '1.0', 16, '2026-01-01');
            """,
            (snap_hash,),
        )
        # Queue extract job
        conn.execute(
            """
            INSERT INTO processing_jobs (id, job_key, resource_id, stage, status, available_at, attempts, created_at, updated_at)
            VALUES ('job_ext_1', 'extract:res_cache', 'res_cache', 'extract', 'pending', '2026-01-01', 0, '2026-01-01', '2026-01-01');
            """
        )

    # Process extract job using short transactions on Database instance
    extract_called = [False]

    def fail_if_called(url, raw_html=None):
        extract_called[0] = True
        return ExtractionResult(status="completed", clean_text="Should not be called")

    monkeypatch.setattr("edward.services.processor.extract_content", fail_if_called)

    res = process_pending_jobs(test_db, test_blob_store, limit=1)
    assert res["completed"] == 1
    # Cached hit prevented redundant extraction call
    assert not extract_called[0]

    with test_db.connection() as conn:
        cls_job = conn.execute("SELECT * FROM processing_jobs WHERE stage = 'classify';").fetchone()
        assert cls_job is not None
        assert cls_job["status"] == "pending"


def test_extraction_pending_status_does_not_complete_or_classify(
    test_db: Database, test_blob_store: BlobStore, monkeypatch: pytest.MonkeyPatch
):
    """Finding 2: Extraction returning pending remains pending and does not schedule classification."""
    with test_db.transaction() as conn:
        conn.execute(
            """
            INSERT INTO resources (id, identity_key, canonical_url, title, review_state, is_deleted, created_at, updated_at)
            VALUES ('res_pend', 'url:https://example.com/pend', 'https://example.com/pend', 'Pending Resource', 'unreviewed', 0, '2026-01-01', '2026-01-01');
            """
        )
        conn.execute(
            """
            INSERT INTO processing_jobs (id, job_key, resource_id, stage, status, available_at, attempts, created_at, updated_at)
            VALUES ('job_pend_1', 'extract:res_pend', 'res_pend', 'extract', 'pending', '2026-01-01', 0, '2026-01-01', '2026-01-01');
            """
        )

    monkeypatch.setattr(
        "edward.services.processor.extract_content",
        lambda url, raw_html=None: ExtractionResult(status="pending"),
    )

    res = process_pending_jobs(test_db, test_blob_store, limit=1)
    assert res["completed"] == 0

    with test_db.connection() as conn:
        job = conn.execute("SELECT * FROM processing_jobs WHERE id = 'job_pend_1';").fetchone()
        assert job["status"] == "pending"
        # No classify job should have been created
        cls_jobs = conn.execute(
            "SELECT * FROM processing_jobs WHERE stage = 'classify';"
        ).fetchall()
        assert len(cls_jobs) == 0
