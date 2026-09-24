"""Tests covering Phase 3 reviewer remediation findings 1 through 10."""

import datetime
import os
from unittest.mock import patch

import pytest

from edward.blobs import BlobStore
from edward.db import Database
from edward.services.answer import (
    determine_packet_data_class,
    synthesize_answer,
)
from edward.services.citations import validate_citations, verify_passage_verbatim
from edward.services.classification import run_classification_pipeline
from edward.services.embed import (
    get_configured_embedding_model,
    store_embedding,
    store_resource_chunks,
)
from edward.services.findings import (
    ExtractedFinding,
    ExtractedPassage,
    ExtractionPayload,
    store_extracted_payload,
)
from edward.services.hybrid import search_hybrid
from edward.services.lifecycle import purge_object
from edward.services.privacy import PrivacyTransmissionError


@pytest.fixture
def test_db_and_blobs(tmp_path):
    db_file = tmp_path / "test_p3.db"
    blobs_dir = tmp_path / "blobs"
    db = Database(db_file)
    db.run_migrations()
    blob_store = BlobStore(blobs_dir)
    return db, blob_store


def test_finding1_privacy_gate_on_research_synthesis(test_db_and_blobs, monkeypatch):
    """Finding 1: Privacy gate halts synthesis before serialization if hosted provider used for private data."""
    db, _ = test_db_and_blobs
    # Force hosted provider for synthesis
    monkeypatch.setenv("EDWARD_SYNTHESIS_BASE_URL", "https://api.together.xyz/v1")
    monkeypatch.setenv("EDWARD_SYNTHESIS_LOCATION", "hosted")
    # Disallow personal notes to hosted
    monkeypatch.delenv("EDWARD_HOSTED_PERSONAL_NOTES", raising=False)

    packet = {
        "items": [
            {
                "id": "res_pub",
                "text": "Public article text",
                "origin_namespace": "web",
            }
        ]
    }

    from edward.services.llm import LLMClient

    dummy_client = LLMClient(
        provider="typesafe",
        base_url="https://api.typesafe.com/v1",
        api_key="test-key",
        location="hosted",
    )

    # Query defaults to personal_notes, which is blocked from hosted provider
    with db.connection() as conn:
        with pytest.raises(PrivacyTransmissionError):
            synthesize_answer(
                conn=conn,
                query="What are my private research notes?",
                evidence_packet=packet,
                llm_client=dummy_client,
            )


def test_finding1_determine_packet_data_class_hierarchy():
    """Finding 1: determine_packet_data_class selects the strictest data class."""
    # gmail > personal_notes > documents > public_web
    p_gmail = {"items": [{"origin_namespace": "gmail"}, {"origin_namespace": "web"}]}
    assert determine_packet_data_class(p_gmail, "public query") == "gmail"

    p_doc = {"items": [{"origin_namespace": "files"}, {"origin_namespace": "web"}]}
    assert determine_packet_data_class(p_doc, "notes query") == "personal_notes"


def test_finding2_jev_provenance_metadata_forwarded(test_db_and_blobs):
    """Finding 2: Classifier request receives rich provenance metadata from resources and captures."""
    db, _ = test_db_and_blobs
    now_iso = datetime.datetime.now(datetime.UTC).isoformat()

    with db.transaction() as conn:
        conn.execute(
            """
            INSERT INTO captures (id, origin_namespace, origin_id, collection_channel, collector, acquisition_method, retrieved_at, created_at, updated_at)
            VALUES ('cap_meta', 'gmail', 'msg_12345', 'cli', 'cli', 'cli', ?, ?, ?);
            """,
            (now_iso, now_iso, now_iso),
        )
        conn.execute(
            """
            INSERT INTO resources (id, identity_key, canonical_url, primary_form, review_state, is_deleted, created_at, updated_at)
            VALUES ('res_meta', 'url:meta', 'https://mail.google.com/mail/u/0/#inbox/12345', 'email', 'unreviewed', 0, ?, ?);
            """,
            (now_iso, now_iso),
        )
        conn.execute(
            """
            INSERT INTO capture_resources (capture_id, resource_id, created_at)
            VALUES ('cap_meta', 'res_meta', ?);
            """,
            (now_iso,),
        )
        conn.execute(
            """
            INSERT INTO resource_contents (id, resource_id, content_hash, clean_text, extractor, extractor_version, char_count, created_at)
            VALUES ('rc_meta', 'res_meta', 'hash_meta', 'Subject: Test\n\nBody of message', 'ext', '1.0', 30, ?);
            """,
            (now_iso,),
        )

    # Run classification pipeline with dry-run provider and inspect judgments
    with db.transaction() as conn:
        with patch.dict(os.environ, {"EDWARD_CLASSIFIER_PROVIDER": "dry-run"}):
            run_classification_pipeline(conn, "resource", "res_meta")

        judgments = conn.execute("SELECT * FROM judgments WHERE object_id = 'res_meta';").fetchall()
        assert len(judgments) > 0


def test_finding3_purge_cleans_chunk_embeddings_and_search_documents(test_db_and_blobs):
    """Finding 3: Purging a resource cascades to its chunk embeddings and chunk search_documents."""
    db, _ = test_db_and_blobs
    now_iso = datetime.datetime.now(datetime.UTC).isoformat()

    with db.transaction() as conn:
        conn.execute(
            """
            INSERT INTO resources (id, identity_key, canonical_url, title, review_state, is_deleted, created_at, updated_at)
            VALUES ('res_purge', 'url:purge', 'https://example.com/p', 'Purge Test', 'unreviewed', 0, ?, ?);
            """,
            (now_iso, now_iso),
        )
        conn.execute(
            """
            INSERT INTO resource_contents (id, resource_id, content_hash, clean_text, extractor, extractor_version, char_count, created_at)
            VALUES ('rc_purge', 'res_purge', 'hash_p', '# Header\n\nParagraph text for purge test.', 'ext', '1.0', 40, ?);
            """,
            (now_iso,),
        )
        # Store chunks and embeddings
        chunk_ids = store_resource_chunks(
            conn, "res_purge", "rc_purge", "# Header\n\nParagraph text for purge test."
        )
        assert len(chunk_ids) > 0
        for cid in chunk_ids:
            store_embedding(conn, "resource_chunk", cid, "chunk text", model="deterministic-v1")

        # Verify chunk FTS and embeddings exist
        fts_chunks = conn.execute(
            "SELECT * FROM search_documents WHERE object_type = 'chunk' AND object_id = ?;",
            (chunk_ids[0],),
        ).fetchall()
        assert len(fts_chunks) > 0
        emb_chunks = conn.execute(
            "SELECT * FROM embeddings WHERE object_type = 'resource_chunk' AND object_id = ?;",
            (chunk_ids[0],),
        ).fetchall()
        assert len(emb_chunks) > 0

    # Purge the resource
    with db.transaction() as conn:
        purge_object(conn, "resource", "res_purge")

        # Verify chunk records, embeddings, and search documents are wiped
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM resource_chunks WHERE resource_id = 'res_purge';"
            ).fetchone()[0]
            == 0
        )
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM search_documents WHERE object_type = 'chunk' AND object_id = ?;",
                (chunk_ids[0],),
            ).fetchone()[0]
            == 0
        )
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM embeddings WHERE object_type = 'resource_chunk' AND object_id = ?;",
                (chunk_ids[0],),
            ).fetchone()[0]
            == 0
        )


def test_finding4_processor_in_flight_content_invalidation(test_db_and_blobs):
    """Finding 4: Processor detects in-flight content changes and leaves work pending for latest input."""
    db, blob_store = test_db_and_blobs
    now_iso = datetime.datetime.now(datetime.UTC).isoformat()

    with db.transaction() as conn:
        conn.execute(
            """
            INSERT INTO resources (id, identity_key, canonical_url, title, review_state, is_deleted, created_at, updated_at)
            VALUES ('res_inflight', 'url:if', 'https://example.com/if', 'Inflight Test', 'unreviewed', 0, ?, ?);
            """,
            (now_iso, now_iso),
        )
        conn.execute(
            """
            INSERT INTO resource_contents (id, resource_id, content_hash, clean_text, extractor, extractor_version, char_count, created_at)
            VALUES ('rc_v1', 'res_inflight', 'hash_v1', 'Old text', 'ext', '1.0', 8, ?);
            """,
            (now_iso,),
        )
        conn.execute(
            """
            INSERT INTO processing_jobs (id, job_key, resource_id, stage, status, attempts, max_attempts, available_at, created_at, updated_at)
            VALUES ('job_fe_if', 'fe:res_inflight', 'res_inflight', 'finding-extraction', 'pending', 0, 3, ?, ?, ?);
            """,
            (now_iso, now_iso, now_iso),
        )

    # Claim job and simulate content change before persistence
    from edward.services.processor import (
        _load_job_context,
        _perform_job_work,
        _persist_job_result,
        claim_job,
    )

    with db.transaction() as conn:
        job = claim_job(conn, "worker_1", lease_seconds=30)
        assert job is not None
        ctx = _load_job_context(conn, job)

        # New content arrives while work is being computed!
        later_iso = (
            datetime.datetime.now(datetime.UTC) + datetime.timedelta(seconds=5)
        ).isoformat()
        conn.execute(
            """
            INSERT INTO resource_contents (id, resource_id, content_hash, clean_text, extractor, extractor_version, char_count, created_at)
            VALUES ('rc_v2', 'res_inflight', 'hash_v2_new', 'New updated text', 'ext', '1.0', 16, ?);
            """,
            (later_iso,),
        )

        work_res = _perform_job_work(blob_store, job, ctx)
        outcome = _persist_job_result(
            conn, blob_store, job, work_res, None, worker_id="worker_1", context=ctx
        )
        # Must detect that content hash changed and requeue as pending
        assert outcome == "pending"

        requeued = conn.execute(
            "SELECT status, lease_owner FROM processing_jobs WHERE id = ?;", (job["id"],)
        ).fetchone()
        assert requeued["status"] == "pending"
        assert requeued["lease_owner"] is None


def test_finding5_finding_support_validation_drops_hallucinations(test_db_and_blobs):
    """Finding 5: store_extracted_payload validates supporting passages against clean_text."""
    db, _ = test_db_and_blobs
    now_iso = datetime.datetime.now(datetime.UTC).isoformat()

    with db.transaction() as conn:
        conn.execute(
            """
            INSERT INTO resources (id, identity_key, canonical_url, title, review_state, is_deleted, created_at, updated_at)
            VALUES ('res_sp', 'url:sp', 'https://example.com/sp', 'Support Test', 'unreviewed', 0, ?, ?);
            """,
            (now_iso, now_iso),
        )
        conn.execute(
            """
            INSERT INTO resource_contents (id, resource_id, content_hash, clean_text, extractor, extractor_version, char_count, created_at)
            VALUES ('rc_sp', 'res_sp', 'hash_sp', 'The quick brown fox jumps over the lazy dog.', 'ext', '1.0', 44, ?);
            """,
            (now_iso,),
        )

        payload = ExtractionPayload(
            findings=[
                ExtractedFinding(
                    statement="Fox jumps over dog.",
                    assertion_role="source-claim",
                    supporting_passages=[
                        ExtractedPassage(
                            passage="quick brown fox jumps over"
                        ),  # Valid verbatim passage
                        ExtractedPassage(
                            passage="completely fabricated sentence not in text"
                        ),  # Hallucinated passage
                    ],
                )
            ],
            entities=[],
        )

        res = store_extracted_payload(conn, "res_sp", payload, source_content_hash="hash_sp")
        f_ids = res["finding_ids"]
        assert len(f_ids) == 1

        # Check finding_support rows
        supports = conn.execute(
            "SELECT passage FROM finding_support WHERE finding_id = ?;", (f_ids[0],)
        ).fetchall()
        passages = [s["passage"] for s in supports]
        assert "quick brown fox jumps over" in passages
        assert "completely fabricated sentence not in text" not in passages


def test_finding5_citation_case_sensitive_and_min_citations(test_db_and_blobs):
    """Finding 5: Level 3 citation verification is case-sensitive and all_levels_passed requires >0 citations."""
    db, _ = test_db_and_blobs
    now_iso = datetime.datetime.now(datetime.UTC).isoformat()

    with db.transaction() as conn:
        conn.execute(
            """
            INSERT INTO resources (id, identity_key, canonical_url, title, review_state, is_deleted, created_at, updated_at)
            VALUES ('res_cs', 'url:cs', 'https://example.com/cs', 'Case Test', 'unreviewed', 0, ?, ?);
            """,
            (now_iso, now_iso),
        )
        conn.execute(
            """
            INSERT INTO resource_contents (id, resource_id, content_hash, clean_text, extractor, extractor_version, char_count, created_at)
            VALUES ('rc_cs', 'res_cs', 'hash_cs', 'Apple M3 Max has 128GB unified memory.', 'ext', '1.0', 38, ?);
            """,
            (now_iso,),
        )

    with db.connection() as conn:
        # Case sensitive check
        assert verify_passage_verbatim(conn, "res_cs", "Apple M3 Max") is True
        assert verify_passage_verbatim(conn, "res_cs", "apple m3 max") is False

        # 0 citations cannot pass factual synthesis
        no_cite_rep = validate_citations(
            conn, "This answer has no citations at all.", evidence_packet={"items": []}
        )
        assert no_cite_rep.all_levels_passed is False
        assert no_cite_rep.total_citations == 0


def test_finding6_chunk_mapping_in_hybrid_retrieval(test_db_and_blobs):
    """Finding 6: Hybrid retrieval maps chunk vector hits to parent resources and collects passages."""
    db, _ = test_db_and_blobs
    now_iso = datetime.datetime.now(datetime.UTC).isoformat()

    with db.transaction() as conn:
        conn.execute(
            """
            INSERT INTO resources (id, identity_key, canonical_url, title, review_state, is_deleted, created_at, updated_at)
            VALUES ('res_chunk_test', 'url:ct', 'https://example.com/ct', 'Hybrid Retrieval Chunk Test', 'unreviewed', 0, ?, ?);
            """,
            (now_iso, now_iso),
        )
        conn.execute(
            """
            INSERT INTO resource_contents (id, resource_id, content_hash, clean_text, extractor, extractor_version, char_count, created_at)
            VALUES ('rc_ct', 'res_chunk_test', 'hash_ct', '# Section 1\n\nImportant finding on inference performance.', 'ext', '1.0', 50, ?);
            """,
            (now_iso,),
        )
        chunk_ids = store_resource_chunks(
            conn,
            "res_chunk_test",
            "rc_ct",
            "# Section 1\n\nImportant finding on inference performance.",
        )
        # Store embedding for chunk
        store_embedding(
            conn,
            "resource_chunk",
            chunk_ids[0],
            "Important finding on inference performance.",
            model="deterministic-v1",
        )

    with db.connection() as conn:
        packet = search_hybrid(conn, "inference performance", limit=5)
        # Verify packet contains evidence item mapped to parent resource
        item_ids = [item["id"] for item in packet["items"]]
        assert "res_chunk_test" in item_ids
        target_item = next(it for it in packet["items"] if it["id"] == "res_chunk_test")
        assert len(target_item["supporting_passages"]) > 0


def test_finding8_model_resolution_and_no_default_literal():
    """The built-in local model is the default, and explicit local overrides are honored."""
    with patch.dict(os.environ, {"EDWARD_EMBEDDING_MODEL": "default"}, clear=True):
        assert get_configured_embedding_model() == "BAAI/bge-small-en-v1.5"

    with patch.dict(
        os.environ,
        {"EDWARD_EMBEDDING_MODEL": "custom-local-model"},
        clear=True,
    ):
        assert get_configured_embedding_model() == "custom-local-model"


def test_finding9_jev_model_identities_persisted(test_db_and_blobs):
    """Finding 9: Requested and resolved model identities are stored in judgments."""
    db, _ = test_db_and_blobs
    now_iso = datetime.datetime.now(datetime.UTC).isoformat()

    with db.transaction() as conn:
        conn.execute(
            """
            INSERT INTO resources (id, identity_key, canonical_url, primary_form, review_state, is_deleted, created_at, updated_at)
            VALUES ('res_model_id', 'url:mid', 'https://example.com/mid', 'article', 'unreviewed', 0, ?, ?);
            """,
            (now_iso, now_iso),
        )
        conn.execute(
            """
            INSERT INTO resource_contents (id, resource_id, content_hash, clean_text, extractor, extractor_version, char_count, created_at)
            VALUES ('rc_mid', 'res_model_id', 'hash_mid', 'Text for classification model check.', 'ext', '1.0', 35, ?);
            """,
            (now_iso,),
        )

        with patch.dict(os.environ, {"EDWARD_CLASSIFIER_PROVIDER": "dry-run"}):
            run_classification_pipeline(conn, "resource", "res_model_id")

        j_row = conn.execute(
            "SELECT requested_model, resolved_model FROM judgments WHERE object_id = 'res_model_id' LIMIT 1;"
        ).fetchone()
        assert j_row is not None
        assert j_row["resolved_model"] is not None
        assert len(j_row["resolved_model"]) > 0


def test_round2_finding1_mixed_independent_privacy(test_db_and_blobs, monkeypatch):
    """Round 2 Finding 1: Mixed data classes are evaluated independently; permitting one does not permit another."""
    db, _ = test_db_and_blobs
    monkeypatch.setenv("EDWARD_SYNTHESIS_LOCATION", "hosted")
    monkeypatch.setenv("EDWARD_HOSTED_GMAIL", "allow")
    # EDWARD_HOSTED_PERSONAL_NOTES is intentionally NOT set (defaults to forbidden)
    monkeypatch.delenv("EDWARD_HOSTED_PERSONAL_NOTES", raising=False)

    from edward.services.answer import synthesize_answer
    from edward.services.llm import LLMClient

    client = LLMClient(
        provider="typesafe",
        base_url="https://api.typesafe.com/v1",
        api_key="k",
        location="hosted",
    )

    mixed_packet = {
        "items": [
            {"id": "res_gmail", "origin_namespace": "gmail", "text": "gmail message"},
            {"id": "res_note", "origin_namespace": "notes", "text": "personal note content"},
        ]
    }

    with db.connection() as conn:
        with pytest.raises(PrivacyTransmissionError):
            synthesize_answer(
                conn=conn, query="test query", evidence_packet=mixed_packet, llm_client=client
            )


def test_round2_finding2_classification_target_url(test_db_and_blobs):
    """Round 2 Finding 2: load_classification_target handles captures without resources and findings."""
    from edward.services.classification import load_classification_target

    db, _ = test_db_and_blobs
    now_iso = datetime.datetime.now(datetime.UTC).isoformat()

    with db.transaction() as conn:
        # Capture without resource
        conn.execute(
            "INSERT INTO captures (id, origin_namespace, raw_content, collection_channel, collector, acquisition_method, retrieved_at, created_at, updated_at) VALUES ('cap_solo', 'manual', 'solo note', 'cli', 'cli', 'cli', ?, ?, ?);",
            (now_iso, now_iso, now_iso),
        )
        # Resource and finding
        conn.execute(
            "INSERT INTO resources (id, identity_key, canonical_url, primary_form, review_state, is_deleted, created_at, updated_at) VALUES ('res_f', 'url:f', 'https://example.com/f', 'article', 'unreviewed', 0, ?, ?);",
            (now_iso, now_iso),
        )
        conn.execute(
            "INSERT INTO findings (id, resource_id, statement, assertion_role, extractor, extractor_version, review_state, is_deleted, created_at, updated_at) VALUES ('fin_1', 'res_f', 'Important claim', 'source-claim', 'ext', '1.0', 'unreviewed', 0, ?, ?);",
            (now_iso, now_iso),
        )

    with db.connection() as conn:
        target_cap = load_classification_target(conn, "capture", "cap_solo")
        assert target_cap["text"] == "solo note"
        assert target_cap["url"] is None

        target_fin = load_classification_target(conn, "finding", "fin_1")
        assert target_fin["text"] == "Important claim"
        assert target_fin["url"] == "https://example.com/f"


def test_round2_finding3_embedding_hash_deduplication(test_db_and_blobs):
    """Round 2 Finding 3: store_embedding avoids re-generating embedding when input_hash is already present."""
    from edward.services.embed import store_embedding

    db, _ = test_db_and_blobs

    with db.transaction() as conn:
        emb_id1 = store_embedding(
            conn, "resource", "res_dedup", "Hello world", model="deterministic-v1"
        )
        assert emb_id1 is not None

        # Call again with same text and model, patching generate_embedding to ensure it is not called
        with patch("edward.services.embed.generate_embedding") as mock_gen:
            emb_id2 = store_embedding(
                conn, "resource", "res_dedup", "Hello world", model="deterministic-v1"
            )
            assert emb_id2 == emb_id1
            mock_gen.assert_not_called()


def test_round2_finding4_embed_findings_and_captures(test_db_and_blobs):
    """Round 2 Finding 4: Findings and captures generate embeddings upon storage."""
    from edward.models import CaptureInput
    from edward.services.capture import capture_item
    from edward.services.findings import (
        ExtractedFinding,
        ExtractionPayload,
        store_extracted_payload,
    )

    db, blob_store = test_db_and_blobs
    now_iso = datetime.datetime.now(datetime.UTC).isoformat()

    with db.transaction() as conn:
        # 1. Capture embedding (enqueued to job queue to avoid blocking capture transaction)
        req = CaptureInput(
            origin_namespace="manual",
            text="Private journal entry about LLMs",
            note="My thought",
            collection_channel="cli",
            collector="cli",
            acquisition_method="manual",
        )
        cap_res = capture_item(conn, req)
        cap_id = cap_res["capture_id"]
        # Capture should have enqueued an embed job
        job_row = conn.execute(
            "SELECT id FROM processing_jobs WHERE capture_id = ? AND stage = 'embed';",
            (cap_id,),
        ).fetchone()
        assert job_row is not None

        # Claim and execute the embed job
        from edward.services.processor import claim_job, execute_job

        job = claim_job(conn, worker_id="test_worker")
        assert job is not None
        assert job["capture_id"] == cap_id
        success = execute_job(conn, blob_store, job)
        assert success is True

        cap_embs = conn.execute(
            "SELECT * FROM embeddings WHERE object_type = 'capture' AND object_id = ?;",
            (cap_id,),
        ).fetchall()
        assert len(cap_embs) > 0

        # 2. Finding embedding via embed stage
        conn.execute(
            "INSERT INTO resources (id, identity_key, canonical_url, title, review_state, is_deleted, created_at, updated_at) VALUES ('res_emb_f', 'url:ef', 'https://example.com/ef', 'Title', 'unreviewed', 0, ?, ?);",
            (now_iso, now_iso),
        )
        conn.execute(
            "INSERT INTO resource_contents (id, resource_id, content_hash, clean_text, extractor, extractor_version, char_count, created_at) VALUES ('rc_ef', 'res_emb_f', 'hash_ef', 'Body text', 'ext', '1.0', 9, ?);",
            (now_iso,),
        )
        payload = ExtractionPayload(
            findings=[
                ExtractedFinding(
                    statement="Vector retrieval improves recall.",
                    assertion_role="source-claim",
                )
            ],
            entities=[],
        )
        res = store_extracted_payload(conn, "res_emb_f", payload, source_content_hash="hash_ef")
        find_id = res["finding_ids"][0]

        from edward.services.embed import embed_resource

        embed_resource(conn, "res_emb_f")

        find_embs = conn.execute(
            "SELECT * FROM embeddings WHERE object_type = 'finding' AND object_id = ?;",
            (find_id,),
        ).fetchall()
        assert len(find_embs) > 0


def test_round2_finding5_unverified_finding_support_not_trusted(test_db_and_blobs):
    """Round 2 Finding 5: Finding support not matching canonical resource text is not trusted by citation validation."""
    from edward.services.citations import validate_citations, verify_passage_verbatim

    db, _ = test_db_and_blobs
    now_iso = datetime.datetime.now(datetime.UTC).isoformat()

    with db.transaction() as conn:
        conn.execute(
            "INSERT INTO resources (id, identity_key, canonical_url, title, review_state, is_deleted, created_at, updated_at) VALUES ('res_g', 'url:g', 'https://example.com/g', 'Title', 'unreviewed', 0, ?, ?);",
            (now_iso, now_iso),
        )
        conn.execute(
            "INSERT INTO resource_contents (id, resource_id, content_hash, clean_text, extractor, extractor_version, char_count, created_at) VALUES ('rc_g', 'res_g', 'hash_g', 'Actual canonical text of resource.', 'ext', '1.0', 33, ?);",
            (now_iso,),
        )
        # Finding with a fabricated finding_support passage
        conn.execute(
            "INSERT INTO findings (id, resource_id, statement, assertion_role, agent_confidence, review_state, is_deleted, created_at, updated_at) VALUES ('fin_fab', 'res_g', 'Fabricated claim', 'source-claim', 0.95, 'unreviewed', 0, ?, ?);",
            (now_iso, now_iso),
        )
        conn.execute(
            "INSERT INTO finding_support (id, finding_id, passage, content_hash, created_at) VALUES ('fs_fab', 'fin_fab', 'fabricated passage not in text', 'hash', ?);",
            (now_iso,),
        )

    with db.connection() as conn:
        # 1. verify_passage_verbatim must reject fabricated passage even if it's in finding_support
        assert verify_passage_verbatim(conn, "fin_fab", "fabricated passage not in text") is False
        assert verify_passage_verbatim(conn, "fin_fab", "Actual canonical text") is True

        # 2. validate_citations must not assign support_likely to ungrounded finding
        packet = {"items": [{"id": "fin_fab", "text": "Fabricated claim"}]}
        rep = validate_citations(
            conn, "According to [#fin_fab] something happened.", evidence_packet=packet
        )
        assert rep.all_levels_passed is False
        assert rep.citations[0].level4_support_status != "support_likely"


def test_round2_finding6_citation_validation_gating_downgrade(test_db_and_blobs):
    """Round 2 Finding 6: synthesize_answer downgrades to Tier 2 when citations fail validation."""
    from edward.services.answer import synthesize_answer

    db, _ = test_db_and_blobs

    packet = {
        "items": [
            {
                "id": "res_real",
                "title": "Real Resource",
                "text": "Real content about quantum computing.",
                "origin_namespace": "web",
            }
        ]
    }

    class FakeLLMClient:
        provider = "ollama"
        location = "local"
        base_url = "http://localhost:11434/v1"
        model = "qwen2.5:7b"

        def chat_completion(self, messages, *args, **kwargs):
            # Cites non-existent ID or quotes fake text
            return (
                "Quantum computing is fast [#res_fake] according to 'fake quote never written' [#res_real].",
                "qwen2.5:7b",
            )

    with db.connection() as conn:
        ans = synthesize_answer(
            conn=conn,
            query="Tell me about quantum computing",
            evidence_packet=packet,
            llm_client=FakeLLMClient(),
        )
        assert ans["status"] == "citation_validation_failed"
        assert ans["tier"] == 2
        assert ans.get("citation_validation") is not None
        assert ans["citation_validation"]["all_levels_passed"] is False


def test_round2_finding7_supersede_older_derived_findings(test_db_and_blobs):
    """Round 2 Finding 7: store_extracted_payload supersedes older unreviewed findings when source hash changes."""
    from edward.services.findings import (
        ExtractedFinding,
        ExtractionPayload,
        store_extracted_payload,
    )

    db, _ = test_db_and_blobs
    now_iso = datetime.datetime.now(datetime.UTC).isoformat()

    with db.transaction() as conn:
        conn.execute(
            "INSERT INTO resources (id, identity_key, canonical_url, title, review_state, is_deleted, created_at, updated_at) VALUES ('res_sup', 'url:sup', 'https://example.com/sup', 'Sup Test', 'unreviewed', 0, ?, ?);",
            (now_iso, now_iso),
        )
        conn.execute(
            "INSERT INTO resource_contents (id, resource_id, content_hash, clean_text, extractor, extractor_version, char_count, created_at) VALUES ('rc_sup_1', 'res_sup', 'hash_v1', 'Version 1 body text', 'ext', '1.0', 19, ?);",
            (now_iso,),
        )
        payload1 = ExtractionPayload(
            findings=[
                ExtractedFinding(
                    statement="V1 finding statement",
                    assertion_role="source-claim",
                )
            ],
            entities=[],
        )
        res1 = store_extracted_payload(conn, "res_sup", payload1, source_content_hash="hash_v1")
        v1_id = res1["finding_ids"][0]

        # Verify v1 exists
        assert (
            conn.execute("SELECT COUNT(*) FROM findings WHERE id = ?;", (v1_id,)).fetchone()[0] == 1
        )

        # Now new content version arrives
        conn.execute(
            "INSERT INTO resource_contents (id, resource_id, content_hash, clean_text, extractor, extractor_version, char_count, created_at) VALUES ('rc_sup_2', 'res_sup', 'hash_v2', 'Version 2 body text', 'ext', '1.0', 19, ?);",
            (now_iso,),
        )
        payload2 = ExtractionPayload(
            findings=[
                ExtractedFinding(
                    statement="V2 finding statement",
                    assertion_role="source-claim",
                )
            ],
            entities=[],
        )
        res2 = store_extracted_payload(conn, "res_sup", payload2, source_content_hash="hash_v2")
        v2_id = res2["finding_ids"][0]

        # V1 should NOT be deleted, but superseded (review_state == 'superseded')
        v1_row = conn.execute(
            "SELECT review_state FROM findings WHERE id = ?;", (v1_id,)
        ).fetchone()
        assert v1_row is not None
        assert v1_row["review_state"] == "superseded"
        # Active FTS and embeddings for v1 should be purged
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM search_documents WHERE object_type = 'finding' AND object_id = ?;",
                (v1_id,),
            ).fetchone()[0]
            == 0
        )
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM embeddings WHERE object_type = 'finding' AND object_id = ?;",
                (v1_id,),
            ).fetchone()[0]
            == 0
        )
        # V2 should be present and unreviewed
        v2_row = conn.execute(
            "SELECT review_state FROM findings WHERE id = ?;", (v2_id,)
        ).fetchone()
        assert v2_row is not None
        assert v2_row["review_state"] == "unreviewed"


def test_round2_finding8_rebuild_search_index_includes_chunks(test_db_and_blobs):
    """Round 2 Finding 8: rebuild_search_index indexes active resource chunks into FTS5."""
    from edward.services.embed import store_resource_chunks
    from edward.services.search import rebuild_search_index

    db, _ = test_db_and_blobs
    now_iso = datetime.datetime.now(datetime.UTC).isoformat()

    with db.transaction() as conn:
        conn.execute(
            "INSERT INTO resources (id, identity_key, canonical_url, title, review_state, is_deleted, created_at, updated_at) VALUES ('res_rebuild', 'url:rb', 'https://example.com/rb', 'Rebuild Test', 'unreviewed', 0, ?, ?);",
            (now_iso, now_iso),
        )
        conn.execute(
            "INSERT INTO resource_contents (id, resource_id, content_hash, clean_text, extractor, extractor_version, char_count, created_at) VALUES ('rc_rb', 'res_rebuild', 'hash_rb', 'Rebuild clean text body', 'ext', '1.0', 22, ?);",
            (now_iso,),
        )
        chunk_ids = store_resource_chunks(conn, "res_rebuild", "rc_rb", "Chunk text to rebuild")

    with db.transaction() as conn:
        rebuild_search_index(conn)
        chunk_fts = conn.execute(
            "SELECT * FROM search_documents WHERE object_type = 'chunk' AND object_id = ?;",
            (chunk_ids[0],),
        ).fetchall()
        assert len(chunk_fts) == 1
        assert "Chunk text to rebuild" in chunk_fts[0]["body"]


def test_explicit_deterministic_embedding_does_not_load_fastembed():
    """The explicitly selected non-semantic fallback does not initialize FastEmbed."""
    from edward.services.embed import generate_embedding

    with patch("edward.services.embed._load_local_embedding_model") as mock_loader:
        emb, resolved_model = generate_embedding(
            "Deterministic test vector", model="deterministic-v1"
        )
        assert len(emb) == 64
        assert resolved_model == "deterministic-v1"
        mock_loader.assert_not_called()


def test_capture_and_extraction_do_not_require_embedding_configuration(test_db_and_blobs):
    """Capture and finding persistence do not require an embedding provider."""
    from edward.models import CaptureInput
    from edward.services.capture import capture_item
    from edward.services.findings import (
        ExtractedFinding,
        ExtractionPayload,
        store_extracted_payload,
    )

    db, _ = test_db_and_blobs
    now_iso = datetime.datetime.now(datetime.UTC).isoformat()

    # 1. Capture of sensitive private data succeeds completely
    with db.transaction() as conn:
        req = CaptureInput(
            origin_namespace="manual",
            text="Private sensitive financial thoughts",
            note="Confidential note",
            collection_channel="cli",
            collector="cli",
            acquisition_method="manual",
        )
        res = capture_item(conn, req)
        cap_id = res["capture_id"]
        row = conn.execute(
            "SELECT id, raw_content FROM captures WHERE id = ?;", (cap_id,)
        ).fetchone()
        assert row is not None
        assert row["raw_content"] == "Private sensitive financial thoughts"

    # 2. Finding extraction persistence succeeds completely
    with db.transaction() as conn:
        conn.execute(
            "INSERT INTO resources (id, identity_key, canonical_url, title, review_state, is_deleted, created_at, updated_at) VALUES ('res_priv', 'url:priv', 'https://example.com/priv', 'Private Doc', 'unreviewed', 0, ?, ?);",
            (now_iso, now_iso),
        )
        conn.execute(
            "INSERT INTO resource_contents (id, resource_id, content_hash, clean_text, extractor, extractor_version, char_count, created_at) VALUES ('rc_priv', 'res_priv', 'hash_p', 'Private sensitive content', 'ext', '1.0', 24, ?);",
            (now_iso,),
        )
        payload = ExtractionPayload(
            findings=[
                ExtractedFinding(
                    statement="Sensitive finding about private accounts.",
                    assertion_role="source-claim",
                )
            ],
            entities=[],
        )
        store_res = store_extracted_payload(conn, "res_priv", payload, source_content_hash="hash_p")
        f_id = store_res["finding_ids"][0]
        f_row = conn.execute("SELECT id, statement FROM findings WHERE id = ?;", (f_id,)).fetchone()
        assert f_row is not None
        assert f_row["statement"] == "Sensitive finding about private accounts."


def test_round3_finding2_level4_citation_validation_requires_explicit_review_or_judgment(
    test_db_and_blobs,
):
    """Round 3 Finding 2: Level 4 citation validation marks unreviewed items as support_unchecked, requiring explicit human review or claim-support judgment."""
    from edward.services.citations import validate_citations

    db, _ = test_db_and_blobs
    now_iso = datetime.datetime.now(datetime.UTC).isoformat()

    with db.transaction() as conn:
        # Create unreviewed resource with clean_text
        conn.execute(
            "INSERT INTO resources (id, identity_key, canonical_url, title, review_state, is_deleted, created_at, updated_at) VALUES ('res_unrev', 'url:unrev', 'https://example.com/unrev', 'Title', 'unreviewed', 0, ?, ?);",
            (now_iso, now_iso),
        )
        conn.execute(
            "INSERT INTO resource_contents (id, resource_id, content_hash, clean_text, extractor, extractor_version, char_count, created_at) VALUES ('rc_unrev', 'res_unrev', 'h1', 'Quantum teleportation requires entanglement.', 'ext', '1.0', 44, ?);",
            (now_iso,),
        )
        # Create unreviewed finding with canonical support
        conn.execute(
            "INSERT INTO findings (id, resource_id, statement, assertion_role, agent_confidence, review_state, is_deleted, created_at, updated_at) VALUES ('fin_unrev', 'res_unrev', 'Quantum teleportation requires entanglement.', 'source-claim', 0.9, 'unreviewed', 0, ?, ?);",
            (now_iso, now_iso),
        )
        conn.execute(
            "INSERT INTO finding_support (id, finding_id, passage, content_hash, created_at) VALUES ('fs_unrev', 'fin_unrev', 'Quantum teleportation requires entanglement.', 'h1', ?);",
            (now_iso,),
        )

    with db.connection() as conn:
        packet = {
            "items": [
                {
                    "id": "fin_unrev",
                    "text": "Quantum teleportation requires entanglement.",
                }
            ]
        }
        # Verbatim quote with unreviewed finding
        answer_text = 'According to [#fin_unrev] "Quantum teleportation requires entanglement."'
        rep = validate_citations(conn, answer_text, evidence_packet=packet)
        # Level 1, 2, 3 should pass, but Level 4 must be support_unchecked because neither human review nor judgment exists
        cit = rep.citations[0]
        assert cit.level1_id_exists is True
        assert cit.level2_in_packet is True
        assert cit.level3_passage_match is True
        assert cit.level4_support_status == "support_unchecked"
        assert rep.all_levels_passed is False

        # Now add human review (approved) to finding
        conn.execute("UPDATE findings SET review_state = 'approved' WHERE id = 'fin_unrev';")
        rep2 = validate_citations(conn, answer_text, evidence_packet=packet)
        assert rep2.citations[0].level4_support_status == "human_verified"
        assert rep2.all_levels_passed is True

        # Reset review state to unreviewed and test mismatched judgment (e.g. form analysis instead of claim-support)
        conn.execute("UPDATE findings SET review_state = 'unreviewed' WHERE id = 'fin_unrev';")
        conn.execute(
            """
            INSERT INTO judgments (
                id, object_type, object_id, family, label_or_question_id,
                primitive, answer_json, requested_model, resolved_model,
                provider, question_registry_version, threshold_policy_version,
                input_content_hash, confidence, created_at
            ) VALUES (
                'jdg_form', 'finding', 'fin_unrev', 'system_one', 'form-analysis',
                'categorical', 'article', 'form-v1', 'form-v1',
                'local', 'v1', 'v1', 'h1', 0.95, ?
            );
            """,
            (now_iso,),
        )
        # Form judgment must NOT make citation support_likely
        rep_form = validate_citations(conn, answer_text, evidence_packet=packet)
        assert rep_form.citations[0].level4_support_status == "support_unchecked"

        # Judgment with negative answer must NOT make citation support_likely
        conn.execute(
            """
            INSERT INTO judgments (
                id, object_type, object_id, family, label_or_question_id,
                primitive, answer_json, requested_model, resolved_model,
                provider, question_registry_version, threshold_policy_version,
                input_content_hash, confidence, created_at
            ) VALUES (
                'jdg_neg', 'finding', 'fin_unrev', 'jev', 'claim-support',
                'boolean', 'false', 'evaluator-v1', 'evaluator-v1',
                'local', 'v1', 'v1', 'h1', 0.90, ?
            );
            """,
            (now_iso,),
        )
        rep_neg = validate_citations(conn, answer_text, evidence_packet=packet)
        assert rep_neg.citations[0].level4_support_status == "support_unchecked"

        # Judgment with mismatched content hash must NOT make citation support_likely
        conn.execute(
            """
            INSERT INTO judgments (
                id, object_type, object_id, family, label_or_question_id,
                primitive, answer_json, requested_model, resolved_model,
                provider, question_registry_version, threshold_policy_version,
                input_content_hash, confidence, created_at
            ) VALUES (
                'jdg_mismatch', 'finding', 'fin_unrev', 'jev', 'claim-support',
                'boolean', 'true', 'evaluator-v1', 'evaluator-v1',
                'local', 'v1', 'v1', 'mismatched_hash_never_matches', 0.90, ?
            );
            """,
            (now_iso,),
        )
        rep_mis = validate_citations(conn, answer_text, evidence_packet=packet)
        assert rep_mis.citations[0].level4_support_status == "support_unchecked"

        # Proper positive claim-support judgment with matching content hash 'h1'
        conn.execute(
            """
            INSERT INTO judgments (
                id, object_type, object_id, family, label_or_question_id,
                primitive, answer_json, requested_model, resolved_model,
                provider, question_registry_version, threshold_policy_version,
                input_content_hash, confidence, created_at
            ) VALUES (
                'jdg_valid', 'finding', 'fin_unrev', 'jev', 'claim-support',
                'boolean', 'true', 'evaluator-v1', 'evaluator-v1',
                'local', 'v1', 'v1', 'h1', 0.85, ?
            );
            """,
            (now_iso,),
        )
        rep3 = validate_citations(conn, answer_text, evidence_packet=packet)
        assert rep3.citations[0].level4_support_status == "support_likely"
        assert rep3.all_levels_passed is True


def test_round3_finding3_rejected_synthesis_omitted_from_payload_and_logged_to_diagnostics(
    test_db_and_blobs, monkeypatch, tmp_path
):
    """Round 3 Finding 3: Rejected synthesis is omitted from user payload and recorded in <data-dir>/diagnostics/."""
    from edward.services.answer import synthesize_answer

    db, _ = test_db_and_blobs
    monkeypatch.setenv("EDWARD_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("EDWARD_RECORD_PRIVATE_DIAGNOSTICS", "1")

    packet = {
        "items": [
            {
                "id": "res_real",
                "title": "Quantum Resource",
                "text": "Quantum computers use qubits.",
                "origin_namespace": "web",
            }
        ]
    }

    class HallucinatingLLMClient:
        provider = "ollama"
        location = "local"
        base_url = "http://localhost:11434/v1"
        model = "qwen2.5:7b"

        def chat_completion(self, messages, *args, **kwargs):
            return "Completely hallucinated text with [#res_fake_id]", "qwen2.5:7b"

    with db.connection() as conn:
        res = synthesize_answer(
            conn=conn,
            query="Tell me about qubits",
            evidence_packet=packet,
            llm_client=HallucinatingLLMClient(),
        )

        assert res["status"] == "citation_validation_failed"
        assert res["tier"] == 2
        # Must NOT expose rejected_synthesis in the return payload
        assert "rejected_synthesis" not in res

        # Diagnostic directory should contain the rejected synthesis log
        diag_dir = tmp_path / "diagnostics"
        assert diag_dir.exists()
        diag_files = list(diag_dir.glob("*.json"))
        assert len(diag_files) > 0
        diag_content = diag_files[0].read_text()
        assert "Completely hallucinated text" in diag_content


def test_round3_finding4_superseded_findings_preserve_rows_and_annotations(
    test_db_and_blobs,
):
    """Round 3 Finding 4: Obsolete extractor findings are marked superseded, purging active FTS and embeddings but preserving rows, annotations, and intents."""
    from edward.services.embed import store_embedding
    from edward.services.findings import (
        ExtractedFinding,
        ExtractionPayload,
        store_extracted_payload,
    )
    from edward.services.lifecycle import add_intent

    db, _ = test_db_and_blobs
    now_iso = datetime.datetime.now(datetime.UTC).isoformat()

    with db.transaction() as conn:
        conn.execute(
            "INSERT INTO resources (id, identity_key, canonical_url, title, review_state, is_deleted, created_at, updated_at) VALUES ('res_obs', 'url:obs', 'https://example.com/obs', 'Obs Title', 'unreviewed', 0, ?, ?);",
            (now_iso, now_iso),
        )
        conn.execute(
            "INSERT INTO resource_contents (id, resource_id, content_hash, clean_text, extractor, extractor_version, char_count, created_at) VALUES ('rc_obs_1', 'res_obs', 'hash_v1', 'Version 1 body text', 'ext', '1.0', 19, ?);",
            (now_iso,),
        )
        payload1 = ExtractionPayload(
            findings=[
                ExtractedFinding(
                    statement="Old finding statement to be superseded",
                    assertion_role="source-claim",
                )
            ],
            entities=[],
        )
        res1 = store_extracted_payload(conn, "res_obs", payload1, source_content_hash="hash_v1")
        f1_id = res1["finding_ids"][0]

        # Attach human intent to the finding
        add_intent(conn, "finding", f1_id, "bookmark", source="human", actor="user")

        # Emulate vector and FTS for f1
        store_embedding(
            conn=conn,
            object_type="finding",
            object_id=f1_id,
            text="Old finding statement to be superseded",
            model="deterministic-v1",
        )

        # Verify f1 has active FTS and embedding
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM search_documents WHERE object_type = 'finding' AND object_id = ?;",
                (f1_id,),
            ).fetchone()[0]
            == 1
        )
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM embeddings WHERE object_type = 'finding' AND object_id = ?;",
                (f1_id,),
            ).fetchone()[0]
            == 1
        )

        # Now new extraction with new hash arrives
        conn.execute(
            "INSERT INTO resource_contents (id, resource_id, content_hash, clean_text, extractor, extractor_version, char_count, created_at) VALUES ('rc_obs_2', 'res_obs', 'hash_v2', 'Version 2 body text', 'ext', '1.0', 19, ?);",
            (now_iso,),
        )
        payload2 = ExtractionPayload(
            findings=[
                ExtractedFinding(
                    statement="New finding statement v2",
                    assertion_role="source-claim",
                )
            ],
            entities=[],
        )
        res2 = store_extracted_payload(conn, "res_obs", payload2, source_content_hash="hash_v2")
        f2_id = res2["finding_ids"][0]
        assert f2_id is not None

        # Check f1: finding row preserved, review_state == 'superseded'
        f1_row = conn.execute(
            "SELECT review_state FROM findings WHERE id = ?;", (f1_id,)
        ).fetchone()
        assert f1_row is not None
        assert f1_row["review_state"] == "superseded"

        # User intent preserved!
        intents = conn.execute(
            "SELECT intent, source FROM intents WHERE object_type = 'finding' AND object_id = ?;",
            (f1_id,),
        ).fetchall()
        assert len(intents) == 1
        assert intents[0]["intent"] == "bookmark"
        assert intents[0]["source"] == "human"

        # Active search documents and embeddings purged for f1
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM search_documents WHERE object_type = 'finding' AND object_id = ?;",
                (f1_id,),
            ).fetchone()[0]
            == 0
        )
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM embeddings WHERE object_type = 'finding' AND object_id = ?;",
                (f1_id,),
            ).fetchone()[0]
            == 0
        )


def test_round3_finding5_embedding_model_coexistence_exact_match(test_db_and_blobs):
    """Round 3 Finding 5: store_embedding uses exact (object_type, object_id, model) identity, allowing model coexistence."""
    from unittest.mock import patch

    from edward.services.embed import store_embedding

    db, _ = test_db_and_blobs
    with db.transaction() as conn:
        # Store deterministic vector
        emb_id1 = store_embedding(
            conn=conn,
            object_type="resource",
            object_id="res_coexist",
            text="Text for model 1",
            model="deterministic-v1",
        )
        # Store another model's vector (mocked so it doesn't fall back to deterministic-v1)
        with patch(
            "edward.services.embed.generate_embedding",
            return_value=([0.2] * 64, "custom-embed-v2"),
        ):
            emb_id2 = store_embedding(
                conn=conn,
                object_type="resource",
                object_id="res_coexist",
                text="Text for model 2",
                model="custom-embed-v2",
            )
        assert emb_id1 is not None
        assert emb_id2 is not None

        rows = conn.execute(
            "SELECT model, dimensions FROM embeddings WHERE object_type = 'resource' AND object_id = 'res_coexist' ORDER BY model;",
        ).fetchall()

        assert len(rows) == 2
        assert rows[0]["model"] == "custom-embed-v2"
        assert rows[1]["model"] == "deterministic-v1"


def test_round3_finding6_finding_embeddings_precomputed_with_resource_model_and_class(
    test_db_and_blobs,
):
    """Round 3 Finding 6: finding embeddings in _perform_job_work use resolved resource model and data_class outside write transaction."""
    from edward.services.processor import (
        _load_job_context,
        _perform_job_work,
        _persist_job_result,
    )

    db, blob_store = test_db_and_blobs
    now_iso = datetime.datetime.now(datetime.UTC).isoformat()

    with db.transaction() as conn:
        conn.execute(
            "INSERT INTO resources (id, identity_key, canonical_url, title, review_state, is_deleted, created_at, updated_at) VALUES ('res_femb', 'url:femb', 'https://example.com/femb', 'Resource for Finding Embed', 'unreviewed', 0, ?, ?);",
            (now_iso, now_iso),
        )
        conn.execute(
            "INSERT INTO resource_contents (id, resource_id, content_hash, clean_text, extractor, extractor_version, char_count, created_at) VALUES ('rc_femb', 'res_femb', 'h_femb', 'Resource content text', 'ext', '1.0', 21, ?);",
            (now_iso,),
        )
        conn.execute(
            "INSERT INTO findings (id, resource_id, statement, assertion_role, agent_confidence, review_state, is_deleted, created_at, updated_at) VALUES ('fin_to_emb', 'res_femb', 'Finding statement to embed', 'source-claim', 0.9, 'unreviewed', 0, ?, ?);",
            (now_iso, now_iso),
        )
        job = {
            "id": "job_embed_f",
            "job_key": "embed:res_femb",
            "resource_id": "res_femb",
            "capture_id": None,
            "stage": "embed",
            "status": "running",
            "lease_owner": "worker_test",
        }
        conn.execute(
            """
            INSERT INTO processing_jobs (
                id, job_key, capture_id, resource_id, stage, status,
                available_at, lease_owner, attempts, created_at, updated_at
            ) VALUES (?, ?, NULL, ?, 'embed', 'running', ?, 'worker_test', 0, ?, ?);
            """,
            (job["id"], job["job_key"], job["resource_id"], now_iso, now_iso, now_iso),
        )

        ctx = _load_job_context(conn, job)
        assert "findings" in ctx
        assert len(ctx["findings"]) == 1
        assert ctx["findings"][0]["id"] == "fin_to_emb"

    # _perform_job_work executed outside transaction
    work = _perform_job_work(blob_store, job, ctx)
    assert "finding_embeddings" in work
    assert len(work["finding_embeddings"]) == 1
    fe = work["finding_embeddings"][0]
    assert fe["finding_id"] == "fin_to_emb"
    assert len(fe["vector"]) > 0

    # Persist job result inside transaction
    with db.transaction() as conn:
        outcome = _persist_job_result(
            conn,
            blob_store,
            job,
            work,
            None,
            worker_id="worker_test",
            context=ctx,
        )
        assert outcome == "completed"

        f_emb = conn.execute(
            "SELECT * FROM embeddings WHERE object_type = 'finding' AND object_id = 'fin_to_emb';",
        ).fetchall()
        assert len(f_emb) == 1
        assert f_emb[0]["dimensions"] == len(fe["vector"])


def test_old_deterministic_vectors_do_not_block_local_model_upgrade(test_db_and_blobs, monkeypatch):
    """A stored hash vector does not prevent creation of a semantic model vector."""
    from edward.services.embed import store_embedding

    db, _ = test_db_and_blobs
    monkeypatch.setenv("EDWARD_EMBEDDING_MODEL", "custom-local-model")
    monkeypatch.setattr(
        "edward.services.embed.generate_embedding",
        lambda text, model=None, **kwargs: ([0.25, 0.5, 0.75], model),
    )

    with db.transaction() as conn:
        store_embedding(
            conn, "resource", "res_fallback_test", "same text", model="deterministic-v1"
        )
        semantic_id = store_embedding(
            conn, "resource", "res_fallback_test", "same text", model="custom-local-model"
        )
        rows = conn.execute(
            "SELECT id, model FROM embeddings WHERE object_type = 'resource' AND object_id = ? ORDER BY model;",
            ("res_fallback_test",),
        ).fetchall()

    assert len(rows) == 2
    assert {row["model"] for row in rows} == {"deterministic-v1", "custom-local-model"}
    assert semantic_id == next(row["id"] for row in rows if row["model"] == "custom-local-model")


def test_round4_finding2_unrelated_judgments_rejected_by_level4(test_db_and_blobs):
    """Round 4 Finding 2: Unrelated judgments (topics, forms, signals, incomplete, or negative) are rejected by Level 4."""
    from edward.services.citations import validate_citations

    db, _ = test_db_and_blobs
    now_iso = datetime.datetime.now(datetime.UTC).isoformat()

    with db.transaction() as conn:
        conn.execute(
            "INSERT INTO resources (id, identity_key, canonical_url, title, review_state, is_deleted, created_at, updated_at) VALUES ('res_jtest', 'url:jtest', 'https://example.com/jtest', 'J Test', 'unreviewed', 0, ?, ?);",
            (now_iso, now_iso),
        )
        conn.execute(
            "INSERT INTO resource_contents (id, resource_id, content_hash, clean_text, extractor, extractor_version, char_count, created_at) VALUES ('rc_jtest', 'res_jtest', 'h_jtest', 'Important scientific claim.', 'ext', '1.0', 26, ?);",
            (now_iso,),
        )
        conn.execute(
            "INSERT INTO findings (id, resource_id, statement, assertion_role, agent_confidence, review_state, is_deleted, created_at, updated_at) VALUES ('fin_jtest', 'res_jtest', 'Important scientific claim.', 'source-claim', 0.9, 'unreviewed', 0, ?, ?);",
            (now_iso, now_iso),
        )
        conn.execute(
            "INSERT INTO finding_support (id, finding_id, passage, content_hash, created_at) VALUES ('fs_jtest', 'fin_jtest', 'Important scientific claim.', 'h_jtest', ?);",
            (now_iso,),
        )

        # 1. Add topic classification judgment (unrelated)
        conn.execute(
            """
            INSERT INTO judgments (
                id, object_type, object_id, family, label_or_question_id,
                primitive, answer_json, requested_model, resolved_model,
                provider, question_registry_version, threshold_policy_version,
                input_content_hash, confidence, status, created_at
            ) VALUES (
                'jdg_topic', 'finding', 'fin_jtest', 'system_one', 'topic_deep_learning',
                'categorical', 'true', 'topic-v1', 'topic-v1',
                'local', 'v1', 'v1', 'h_jtest', 0.99, 'completed', ?
            );
            """,
            (now_iso,),
        )

        # 2. Add uncompleted claim-support judgment
        conn.execute(
            """
            INSERT INTO judgments (
                id, object_type, object_id, family, label_or_question_id,
                primitive, answer_json, requested_model, resolved_model,
                provider, question_registry_version, threshold_policy_version,
                input_content_hash, confidence, status, created_at
            ) VALUES (
                'jdg_running', 'finding', 'fin_jtest', 'jev', 'claim-support',
                'boolean', 'true', 'eval-v1', 'eval-v1',
                'local', 'v1', 'v1', 'h_jtest', 0.99, 'running', ?
            );
            """,
            (now_iso,),
        )

    with db.connection() as conn:
        packet = {"items": [{"id": "fin_jtest", "text": "Important scientific claim."}]}
        answer_text = 'According to [#fin_jtest] "Important scientific claim."'
        rep = validate_citations(conn, answer_text, evidence_packet=packet)
        # Unrelated topic and running judgments must be rejected
        assert rep.citations[0].level4_support_status == "support_unchecked"
        assert rep.all_levels_passed is False


def test_round4_finding3_embed_jobs_scheduled_for_all_captures_and_findings(test_db_and_blobs):
    """Round 4 Finding 3: Embed processing jobs are scheduled for URL captures with notes, findings extraction, and bundle imports."""
    from edward.models import CaptureInput
    from edward.services.bundle import import_research_bundle
    from edward.services.capture import capture_item
    from edward.services.findings import (
        ExtractedFinding,
        ExtractionPayload,
        store_extracted_payload,
    )

    db, blob_store = test_db_and_blobs
    now_iso = datetime.datetime.now(datetime.UTC).isoformat()

    with db.transaction() as conn:
        # 1. URL capture with personal note (has resource_id)
        req = CaptureInput(
            url="https://example.com/doc-with-note",
            note="Personal thought on this documentation",
            collection_channel="cli",
            collector="cli",
            acquisition_method="manual",
        )
        cap_res = capture_item(conn, req)
        cap_id = cap_res["capture_id"]

        # Must have queued an embed job for the capture
        job_cap = conn.execute(
            "SELECT id FROM processing_jobs WHERE capture_id = ? AND stage = 'embed';",
            (cap_id,),
        ).fetchone()
        assert job_cap is not None

        # 2. Finding extraction on a resource
        conn.execute(
            "INSERT INTO resources (id, identity_key, canonical_url, title, review_state, is_deleted, created_at, updated_at) VALUES ('res_f_embed', 'url:fe', 'https://example.com/fe', 'Title', 'unreviewed', 0, ?, ?);",
            (now_iso, now_iso),
        )
        conn.execute(
            "INSERT INTO resource_contents (id, resource_id, content_hash, clean_text, extractor, extractor_version, char_count, created_at) VALUES ('rc_fe', 'res_f_embed', 'h_fe', 'Body text', 'ext', '1.0', 9, ?);",
            (now_iso,),
        )
        payload = ExtractionPayload(
            findings=[
                ExtractedFinding(
                    statement="Vector indices improve recall.",
                    assertion_role="source-claim",
                )
            ],
            entities=[],
        )
        store_extracted_payload(conn, "res_f_embed", payload, source_content_hash="h_fe")

        # Must have queued an embed job for the resource
        job_res = conn.execute(
            "SELECT id FROM processing_jobs WHERE resource_id = 'res_f_embed' AND stage = 'embed';",
        ).fetchone()
        assert job_res is not None

        # 3. Bundle import
        bundle_data = {
            "type": "research-bundle",
            "schema_version": "1",
            "bundle_id": "bnd_embed_test",
            "title": "Bundle Embed Test",
            "brief": "A research brief on neural nets",
            "sources": [
                {
                    "source_id": "src_1",
                    "url": "https://example.com/bundle-source-1",
                    "title": "Bundle Source 1",
                    "origin": "web",
                }
            ],
            "findings": [
                {
                    "statement": "Neural nets learn representations.",
                    "assertion_role": "source-claim",
                    "source_url": "https://example.com/bundle-source-1",
                }
            ],
        }
        b_res = import_research_bundle(conn, blob_store, bundle_data)
        b_cap_id = b_res["capture_id"]

        # Capture from bundle must have an embed job
        b_job_cap = conn.execute(
            "SELECT id FROM processing_jobs WHERE capture_id = ? AND stage = 'embed';",
            (b_cap_id,),
        ).fetchone()
        assert b_job_cap is not None

        # Resource from bundle must have an embed job
        r_row = conn.execute(
            "SELECT id FROM resources WHERE canonical_url = 'https://example.com/bundle-source-1';",
        ).fetchone()
        assert r_row is not None
        b_job_res = conn.execute(
            "SELECT id FROM processing_jobs WHERE resource_id = ? AND stage = 'embed';",
            (r_row["id"],),
        ).fetchone()
        assert b_job_res is not None


def test_round5_finding1_active_job_lease_not_stolen_and_aggregate_hash_invalidation(
    test_db_and_blobs,
):
    """Round 5 Finding 1: Running embed job lease is preserved during finding upserts, and worker detects aggregate hash mismatch to requeue."""
    from edward.models import make_job_key
    from edward.services.findings import (
        ExtractedFinding,
        ExtractionPayload,
        store_extracted_payload,
    )
    from edward.services.processor import (
        _load_job_context,
        _perform_job_work,
        _persist_job_result,
        claim_job,
    )

    db, blob_store = test_db_and_blobs
    now_iso = datetime.datetime.now(datetime.UTC).isoformat()

    with db.transaction() as conn:
        conn.execute(
            "INSERT INTO resources (id, identity_key, canonical_url, title, review_state, is_deleted, created_at, updated_at) VALUES ('res_lease_test', 'url:lease', 'https://example.com/lease', 'Lease Test', 'unreviewed', 0, ?, ?);",
            (now_iso, now_iso),
        )
        conn.execute(
            "INSERT INTO resource_contents (id, resource_id, content_hash, clean_text, extractor, extractor_version, char_count, created_at) VALUES ('rc_lease', 'res_lease_test', 'hash_lease_1', 'Initial text content for embedding.', 'ext', '1.0', 35, ?);",
            (now_iso,),
        )
        job_key = make_job_key("embed", "res_lease_test")
        conn.execute(
            "INSERT INTO processing_jobs (id, job_key, resource_id, stage, status, available_at, attempts, created_at, updated_at) VALUES ('job_emb_lease', ?, 'res_lease_test', 'embed', 'pending', ?, 0, ?, ?);",
            (job_key, now_iso, now_iso, now_iso),
        )

    # 1. Worker claims the job
    with db.transaction() as conn:
        claimed = claim_job(conn, worker_id="worker-a", lease_seconds=300)
        assert claimed is not None
        assert claimed["id"] == "job_emb_lease"
        assert claimed["lease_owner"] == "worker-a"

        db_claim = conn.execute(
            "SELECT status FROM processing_jobs WHERE id = ?;", (claimed["id"],)
        ).fetchone()
        assert db_claim["status"] == "running"

        # Worker loads context and starts computing outside transaction
        context = _load_job_context(conn, claimed)
        old_input_hash = context["input_hash"]

    # 2. Concurrently, a new finding is stored for this resource
    with db.transaction() as conn:
        payload = ExtractionPayload(
            findings=[
                ExtractedFinding(
                    statement="A newly extracted finding arrived while embedding was running.",
                    assertion_role="source-claim",
                    confidence=0.85,
                )
            ]
        )
        store_extracted_payload(conn, "res_lease_test", payload, source_content_hash="h_new")

        # Invariant check: The active job must NOT have had its lease stolen or reset to pending!
        job_row = conn.execute(
            "SELECT status, lease_owner, lease_expires_at FROM processing_jobs WHERE id = 'job_emb_lease';"
        ).fetchone()
        assert job_row["status"] == "running"
        assert job_row["lease_owner"] == "worker-a"
        assert job_row["lease_expires_at"] is not None

    # 3. Worker finishes external work and calls fenced persistence
    work_result = _perform_job_work(blob_store, claimed, context)
    assert work_result["input_hash"] == old_input_hash

    with db.transaction() as conn:
        result_status = _persist_job_result(
            conn=conn,
            blob_store=blob_store,
            job=claimed,
            work_result=work_result,
            job_error=None,
            context=context,
            worker_id="worker-a",
        )
        # Mismatch detected: aggregate hash changed! Job safely requeued as pending.
        assert result_status == "pending"

        job_after = conn.execute(
            "SELECT status, lease_owner, lease_expires_at FROM processing_jobs WHERE id = 'job_emb_lease';"
        ).fetchone()
        assert job_after["status"] == "pending"
        assert job_after["lease_owner"] is None
        assert job_after["lease_expires_at"] is None


def test_round5_finding2_runtime_claim_support_evaluator_and_answer_contract(
    test_db_and_blobs,
):
    """Round 5 Finding 2: evaluate_claim_support creates judgments with provenance and matching hashes, and synthesize_answer surfaces unreviewed support without failing Tier 3."""
    from edward.services.answer import synthesize_answer
    from edward.services.citations import evaluate_claim_support, validate_citations

    db, _ = test_db_and_blobs
    now_iso = datetime.datetime.now(datetime.UTC).isoformat()

    with db.transaction() as conn:
        conn.execute(
            "INSERT INTO resources (id, identity_key, canonical_url, title, review_state, is_deleted, created_at, updated_at) VALUES ('res_cs_test', 'url:cs', 'https://example.com/cs', 'Claim Support Test', 'unreviewed', 0, ?, ?);",
            (now_iso, now_iso),
        )
        conn.execute(
            "INSERT INTO resource_contents (id, resource_id, content_hash, clean_text, extractor, extractor_version, char_count, created_at) VALUES ('rc_cs_test', 'res_cs_test', 'hash_cs_1', 'Superconducting qubits exhibit quantum coherence.', 'ext', '1.0', 48, ?);",
            (now_iso,),
        )

    with db.connection() as conn:
        # 1. evaluate_claim_support creates a completed judgment with model/provider and matching hash
        eval_res = evaluate_claim_support(
            conn=conn,
            object_id="res_cs_test",
            claim_text="Superconducting qubits exhibit quantum coherence.",
        )
        assert eval_res["status"] == "completed"
        assert eval_res["supported"] is True
        assert eval_res["provider"] == "rule-based"

        # Verify judgment in DB
        j_row = conn.execute(
            "SELECT * FROM judgments WHERE object_id = 'res_cs_test' AND label_or_question_id = 'claim-support';"
        ).fetchone()
        assert j_row is not None
        assert j_row["family"] == "claim-support"
        assert j_row["input_content_hash"] == "hash_cs_1"
        assert j_row["status"] == "completed"

        # 2. validate_citations recognizes this as support_likely
        packet = {
            "items": [
                {
                    "id": "res_cs_test",
                    "text": "Superconducting qubits exhibit quantum coherence.",
                }
            ]
        }
        answer_text = "According to [#res_cs_test], coherence is preserved."
        rep = validate_citations(conn, answer_text, evidence_packet=packet)
        assert rep.all_levels_passed is True
        assert rep.citations[0].level4_support_status == "support_likely"

        # 3. synthesize_answer with unreviewed item returns Tier 3 (status="grounded" or "support_unchecked")
        class ValidLLMClient:
            provider = "ollama"
            location = "local"
            base_url = "http://localhost:11434/v1"
            model = "llama3:8b"

            def chat_completion(self, messages, *args, **kwargs):
                return "Qubits maintain coherence [#res_cs_test].", "llama3:8b"

        ans = synthesize_answer(conn, "Explain qubits", packet, ValidLLMClient())
        assert ans["tier"] == 3
        assert ans["status"] in ("grounded", "support_unchecked")
        assert ans["answer"] == "Qubits maintain coherence [#res_cs_test]."


def test_unchanged_content_skips_local_model_inference(test_db_and_blobs, monkeypatch):
    """A matching local model row is reused without rerunning inference."""
    from edward.services.embed import store_embedding

    db, _ = test_db_and_blobs
    monkeypatch.setattr(
        "edward.services.embed.generate_embedding",
        lambda text, model=None, **kwargs: ([0.25, 0.5, 0.75], model),
    )
    with db.transaction() as conn:
        first_id = store_embedding(
            conn,
            "resource",
            "res_fb_test",
            "Unchanged text to embed",
            model="custom-local-model",
        )
        with patch("edward.services.embed.generate_embedding") as generate_again:
            second_id = store_embedding(
                conn,
                "resource",
                "res_fb_test",
                "Unchanged text to embed",
                model="custom-local-model",
            )
            generate_again.assert_not_called()
        assert second_id == first_id
