"""Tests for finding and entity extraction, supporting evidence, and search indexing."""

from edward.db import Database
from edward.services.findings import (
    extract_findings_for_resource,
    extract_heuristically,
    normalize_entity_name,
    store_extracted_payload,
)


def test_normalize_entity_name():
    """Entity names are normalized for alias matching and indexing."""
    assert normalize_entity_name("Apple Silicon") == "apple silicon"
    assert normalize_entity_name("  Llama-3.2! ") == "llama-32"
    assert normalize_entity_name("SQLite-Vec") == "sqlite-vec"


def test_extract_heuristically_quotes_and_questions():
    """Heuristic extractor extracts blockquotes, questions, claims, and known entities."""
    text = """# Analysis of On-Device AI
> "On-device models unlock real privacy by eliminating network transmission entirely."

Can small models achieve sufficient reasoning depth for automated agents?

- Quantized models run efficiently on Apple Silicon hardware.
- Memory bandwidth remains the primary bottleneck for large context windows.
"""
    payload = extract_heuristically(text)
    assert len(payload.findings) >= 3

    roles = [f.assertion_role for f in payload.findings]
    assert "direct-quotation" in roles
    assert "question" in roles
    assert "source-claim" in roles

    # Check that quote has supporting passage
    quote_finding = next(f for f in payload.findings if f.assertion_role == "direct-quotation")
    assert len(quote_finding.supporting_passages) == 1
    assert "unlock real privacy" in quote_finding.supporting_passages[0].passage

    # Check entities
    entity_names = [e.name for e in payload.entities]
    assert any("Apple Silicon" in name for name in entity_names)


def test_store_extracted_payload_in_database(test_db: Database):
    """Extracted findings and entities are stored in DB, defaulting to unreviewed state, and indexed in FTS."""
    with test_db.transaction() as conn:
        conn.execute(
            """
            INSERT INTO resources (id, identity_key, canonical_url, title, review_state, is_deleted, created_at, updated_at)
            VALUES ('res_findings_1', 'url:f1', 'https://example.com/findings', 'AI Findings', 'unreviewed', 0, '2026-01-01', '2026-01-01');
            """
        )

    text = """# Report
> "Local models run without latency spikes from external rate limits."

- Apple Silicon provides unified memory ideal for LLMs.
"""
    payload = extract_heuristically(text)

    with test_db.transaction() as conn:
        res = store_extracted_payload(
            conn=conn,
            resource_id="res_findings_1",
            payload=payload,
            source_content_hash="hash_findings_1",
        )
        assert res["findings_count"] >= 2
        assert res["entities_count"] >= 1

    with test_db.connection() as conn:
        # Check findings table
        findings = conn.execute(
            "SELECT * FROM findings WHERE resource_id = 'res_findings_1';"
        ).fetchall()
        assert len(findings) >= 2
        for f in findings:
            assert f["review_state"] == "unreviewed"
            assert f["is_deleted"] == 0

        # Check finding_support table
        support = conn.execute(
            """
            SELECT fs.* FROM finding_support fs
            JOIN findings f ON fs.finding_id = f.id
            WHERE f.resource_id = 'res_findings_1';
            """
        ).fetchall()
        assert len(support) >= 1

        # Check entities and object_entities
        entities = conn.execute(
            """
            SELECT e.name, oe.confidence, oe.review_state
            FROM object_entities oe
            JOIN entities e ON oe.entity_id = e.id
            WHERE oe.object_id = 'res_findings_1';
            """
        ).fetchall()
        assert len(entities) >= 1
        for e in entities:
            assert e["review_state"] == "unreviewed"

        # Check FTS index for finding
        fnd_fts = conn.execute(
            "SELECT * FROM search_documents WHERE object_type = 'finding';"
        ).fetchall()
        assert len(fnd_fts) >= 2


def test_extract_findings_for_resource_end_to_end(test_db: Database):
    """extract_findings_for_resource processes resource content and stores results."""
    content_text = """# Breakthroughs
- Transformer attention mechanisms can be evaluated in linear time with state-space models.
> "Hardware specialization will drive the next decade of computer architecture."
"""
    with test_db.transaction() as conn:
        conn.execute(
            """
            INSERT INTO resources (id, identity_key, canonical_url, title, review_state, is_deleted, created_at, updated_at)
            VALUES ('res_f_e2e', 'url:fe2e', 'https://example.com/fe2e', 'Breakthroughs', 'unreviewed', 0, '2026-01-01', '2026-01-01');
            """
        )
        conn.execute(
            """
            INSERT INTO resource_contents (id, resource_id, content_hash, clean_text, extractor, extractor_version, char_count, created_at)
            VALUES ('rc_fe2e', 'res_f_e2e', 'hash_fe2e', ?, 'ext', '1.0', 120, '2026-01-01');
            """,
            (content_text,),
        )

        res = extract_findings_for_resource(conn, "res_f_e2e")
        assert res["findings_count"] >= 2
