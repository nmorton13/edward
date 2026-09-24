"""Tests for hybrid search, Reciprocal Rank Fusion, and Evidence Packet assembly."""

import json
from pathlib import Path

from edward.db import Database
from edward.models import CaptureInput
from edward.services.answer import synthesize_answer
from edward.services.capture import capture_item
from edward.services.embed import store_embedding
from edward.services.hybrid import reciprocal_rank_fusion, research_terms, search_hybrid


def test_research_terms_remove_question_framing():
    assert research_terms("What did I bookmark about local models?") == "local models"


def test_matched_document_chunk_is_attached_once_to_its_resource(test_db: Database):
    """A search hit inside a long document must reach Ask with its matched passage."""
    with test_db.transaction() as conn:
        conn.execute(
            """
            INSERT INTO resources (id, identity_key, canonical_url, title, review_state,
                                   is_deleted, created_at, updated_at)
            VALUES ('res_paper', 'url:paper', 'https://example.com/paper', 'Local AI paper',
                    'unreviewed', 0, '2026-01-01', '2026-01-01')
            """
        )
        conn.execute(
            """
            INSERT INTO resource_contents (id, resource_id, content_hash, clean_text,
                                           extractor, extractor_version, char_count, created_at)
            VALUES ('rc_paper', 'res_paper', 'hash_paper', 'Introduction. Distinctive model finding.',
                    'test', '1', 40, '2026-01-01')
            """
        )
        conn.execute(
            """
            INSERT INTO resource_chunks (id, resource_content_id, resource_id, chunk_index,
                                         text, locator_json, token_count, created_at)
            VALUES ('chk_paper', 'rc_paper', 'res_paper', 0,
                    'Distinctive model finding.', '{"page": 3}', 4, '2026-01-01')
            """
        )
        from edward.services.lifecycle import reindex_object_document

        reindex_object_document(conn, "resource", "res_paper")
        reindex_object_document(conn, "chunk", "chk_paper")

    with test_db.connection() as conn:
        packet = search_hybrid(conn, "distinctive model finding", limit=5)

    assert [item["id"] for item in packet["items"]] == ["res_paper"]
    assert packet["items"][0]["supporting_passages"][0] == {
        "passage": "Distinctive model finding.",
        "locator": {"page": 3},
        "content_hash": "",
    }


def test_pdf_capture_with_user_note_keeps_late_source_passage_for_model(test_db: Database):
    """A PDF hit after the opening text must survive both retrieval and model formatting."""
    long_pdf_text = (
        "General introduction. " * 220 + "[PDF page 8]\nDistinctive local inference result."
    )
    with test_db.transaction() as conn:
        captured = capture_item(
            conn,
            CaptureInput(note="For my essay", text=long_pdf_text, origin_namespace="documents"),
        )

    class CapturingClient:
        provider = "ollama"
        location = "local"
        base_url = "http://localhost:11434/v1"
        model = "test-local"

        def __init__(self):
            self.messages = None

        def chat_completion(self, messages, **kwargs):
            self.messages = messages
            return f"The PDF discusses local inference [#{captured['capture_id']}].", None

    client = CapturingClient()
    with test_db.connection() as conn:
        packet = search_hybrid(conn, "distinctive local inference", limit=5)
        result = synthesize_answer(conn, "distinctive local inference", packet, client)

    item = next(item for item in packet["items"] if item["id"] == captured["capture_id"])
    assert item["kind"] == "capture"
    assert item["user_notes"] == ["For my essay"]
    assert len(item["text"]) <= 1000
    assert any(
        "Distinctive local inference result" in p["passage"] for p in item["supporting_passages"]
    )
    assert any(p["locator"] == {"page": 8} for p in item["supporting_passages"])
    assert "Distinctive local inference result" in client.messages[1]["content"]
    assert result["tier"] == 3


def test_reciprocal_rank_fusion_logic():
    """Reciprocal Rank Fusion correctly combines scores from lexical and vector rankings."""
    lexical = [
        {"object_id": "doc_A", "score": 10.0},
        {"object_id": "doc_B", "score": 8.0},
        {"object_id": "doc_C", "score": 6.0},
    ]
    vector = [
        {"object_id": "doc_B", "similarity": 0.95},
        {"object_id": "doc_D", "similarity": 0.85},
        {"object_id": "doc_A", "similarity": 0.70},
    ]

    # With k=60:
    # doc_B: 1/(60+2) + 1/(60+1) = 0.0161 + 0.0164 = 0.0325
    # doc_A: 1/(60+1) + 1/(60+3) = 0.0164 + 0.0159 = 0.0323
    fused = reciprocal_rank_fusion(lexical, vector, k=60, limit=4)
    assert len(fused) == 4
    # doc_B appears high on both and should win or tie top
    ids = [item["id"] for item in fused]
    assert "doc_B" in ids[:2]
    assert "doc_A" in ids[:2]
    assert "doc_D" in ids
    assert "doc_C" in ids


def test_search_hybrid_evidence_packet_schema(test_db: Database):
    """search_hybrid returns an Evidence Packet strictly conforming to evidence-packet-v1 schema."""
    with test_db.transaction() as conn:
        conn.execute(
            """
            INSERT INTO resources (id, identity_key, canonical_url, title, review_state, is_deleted, created_at, updated_at)
            VALUES ('res_hyb_1', 'url:h1', 'https://example.com/hybrid', 'Hybrid Retrieval on Apple Silicon', 'unreviewed', 0, '2026-01-01', '2026-01-01');
            """
        )
        conn.execute(
            """
            INSERT INTO resource_contents (id, resource_id, content_hash, clean_text, summary, extractor, extractor_version, char_count, created_at)
            VALUES ('rc_hyb_1', 'res_hyb_1', 'hash_hyb_1', 'Combining BM25 and vector search with reciprocal rank fusion yields better recall.', 'Hybrid retrieval summary', 'ext', '1.0', 80, '2026-01-01');
            """
        )
        from edward.services.lifecycle import reindex_object_document

        reindex_object_document(conn, "resource", "res_hyb_1")

        # Store embedding
        store_embedding(
            conn=conn,
            object_type="resource",
            object_id="res_hyb_1",
            text="Combining BM25 and vector search with reciprocal rank fusion yields better recall.",
        )

    with test_db.connection() as conn:
        packet = search_hybrid(conn, query="reciprocal rank fusion BM25 vector", limit=5)

    assert packet["type"] == "evidence-packet"
    assert packet["schema_version"] == "1"
    assert packet["query"] == "reciprocal rank fusion BM25 vector"
    assert "created_at" in packet
    assert "parameters" in packet
    assert "items" in packet
    assert len(packet["items"]) >= 1

    item = packet["items"][0]
    assert item["id"] == "res_hyb_1"
    assert item["kind"] in ("resource", "finding", "capture", "note")
    assert item["assertion_role"] in (
        "source-claim",
        "direct-quotation",
        "agent-conclusion",
        "personal-belief",
        "personal-observation",
        "question",
        "hypothesis",
        "connection",
    )
    assert item["review_state"] == "unreviewed"
    assert item["relevance_score"] is not None
    assert "source" in item
    assert "supporting_passages" in item

    # Verify against schemas/evidence-packet-v1.json
    schema_path = Path(__file__).parent.parent / "schemas" / "evidence-packet-v1.json"
    if schema_path.exists():
        try:
            import jsonschema

            schema = json.loads(schema_path.read_text(encoding="utf-8"))
            jsonschema.validate(instance=packet, schema=schema)
        except ImportError:
            pass


def test_search_hybrid_collapses_x_capture_and_resource_vector_hits(test_db: Database, monkeypatch):
    with test_db.transaction() as conn:
        captured = capture_item(
            conn,
            CaptureInput(
                url="https://x.com/i/status/24680",
                text="Bookmark about semantic retrieval and embeddings",
                origin_namespace="x",
                origin_id="24680",
                collection_channel="birdclaw",
                collector="test",
                acquisition_method="test",
            ),
        )

    monkeypatch.setattr(
        "edward.services.hybrid.search_vector",
        lambda **_kwargs: [
            {"object_type": "capture", "object_id": captured["capture_id"], "similarity": 0.99},
            {"object_type": "resource", "object_id": captured["resource_id"], "similarity": 0.98},
        ],
    )

    with test_db.connection() as conn:
        packet = search_hybrid(conn, query="semantic retrieval embeddings", limit=5)

    assert [item["id"] for item in packet["items"]] == [captured["resource_id"]]
    assert packet["items"][0]["source"]["origin_namespace"] == "x"
    assert packet["items"][0]["source"]["origin_id"] == "24680"
