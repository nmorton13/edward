"""Tests for multi-level citation extraction and calibrated validation."""

from edward.db import Database
from edward.services.citations import (
    extract_citation_ids,
    validate_citations,
    verify_passage_verbatim,
)


def test_extract_citation_ids():
    """Extracts citation tokens and identifiers in [#id] and [id] syntax."""
    text = (
        "According to research [#fnd_abc123], local inference is fast. "
        "Another study [res_def456] observed memory bottlenecks. "
        "Also see [cap_ghi789] and markdown link [click here](https://example.com)."
    )
    citations = extract_citation_ids(text)
    ids = [cid for _, cid in citations]
    assert "fnd_abc123" in ids
    assert "res_def456" in ids
    assert "cap_ghi789" in ids
    assert "click here" not in ids
    assert "https" not in ids


def test_citation_validation_levels_1_through_4(test_db: Database):
    """Citation validation tests Levels 1 through 4 comprehensively."""
    with test_db.transaction() as conn:
        conn.execute(
            """
            INSERT INTO resources (id, identity_key, canonical_url, title, review_state, is_deleted, created_at, updated_at)
            VALUES ('res_cite_1', 'url:c1', 'https://example.com/c1', 'Test Citation Resource', 'approved', 0, '2026-01-01', '2026-01-01');
            """
        )
        conn.execute(
            """
            INSERT INTO resource_contents (id, resource_id, content_hash, clean_text, extractor, extractor_version, char_count, created_at)
            VALUES ('rc_cite_1', 'res_cite_1', 'hash_c1', 'Unified memory architecture reduces CPU to GPU copies.', 'ext', '1.0', 50, '2026-01-01');
            """
        )
        conn.execute(
            """
            INSERT INTO findings (id, resource_id, statement, assertion_role, review_state, is_deleted, created_at, updated_at)
            VALUES ('fnd_cite_1', 'res_cite_1', 'Unified memory avoids memory copies.', 'source-claim', 'unreviewed', 0, '2026-01-01', '2026-01-01');
            """
        )
        conn.execute(
            """
            INSERT INTO finding_support (id, finding_id, passage, created_at)
            VALUES ('fsp_cite_1', 'fnd_cite_1', 'Unified memory architecture reduces CPU to GPU copies.', '2026-01-01');
            """
        )

    packet = {
        "items": [
            {"id": "fnd_cite_1", "text": "Unified memory avoids memory copies."},
            {"id": "res_cite_1", "text": "Unified memory architecture reduces CPU to GPU copies."},
        ]
    }

    # Case A: Valid citations present in packet and database
    answer_valid = (
        "Local models benefit from unified memory [#fnd_cite_1] as explained in [#res_cite_1]."
    )
    with test_db.connection() as conn:
        report = validate_citations(conn, answer_valid, evidence_packet=packet)
        assert report.all_levels_passed is True
        assert report.total_citations == 2
        assert report.valid_citations == 2

    # Case B: Hallucinated citation not in DB or packet
    answer_hallucinated = "Quantum models were also tested [#fnd_nonexistent999]."
    with test_db.connection() as conn:
        report = validate_citations(conn, answer_hallucinated, evidence_packet=packet)
        assert report.all_levels_passed is False
        assert report.valid_citations == 0
        assert len(report.unsupported_claims) == 1
        assert report.citations[0].level1_id_exists is False
        assert report.citations[0].level2_in_packet is False
        assert report.citations[0].level4_support_status == "unsupported"

    # Case C: Citation exists in DB but was omitted from the packet
    answer_not_in_packet = "Details can be found in [#res_cite_1]."
    empty_packet = {"items": []}
    with test_db.connection() as conn:
        report = validate_citations(conn, answer_not_in_packet, evidence_packet=empty_packet)
        assert report.citations[0].level1_id_exists is True
        assert report.citations[0].level2_in_packet is False


def test_level3_verbatim_passage_match(test_db: Database):
    """Level 3 verifies verbatim passage matches against source content."""
    with test_db.transaction() as conn:
        conn.execute(
            """
            INSERT INTO resources (id, identity_key, canonical_url, title, review_state, is_deleted, created_at, updated_at)
            VALUES ('res_passage', 'url:p1', 'https://example.com/p1', 'Passage Test', 'unreviewed', 0, '2026-01-01', '2026-01-01');
            """
        )
        conn.execute(
            """
            INSERT INTO resource_contents (id, resource_id, content_hash, clean_text, extractor, extractor_version, char_count, created_at)
            VALUES ('rc_p1', 'res_passage', 'hash_p1', 'Latency dropped from 120ms to 42ms with 4-bit weights.', 'ext', '1.0', 60, '2026-01-01');
            """
        )

    with test_db.connection() as conn:
        # True match
        assert (
            verify_passage_verbatim(conn, "res_passage", "Latency dropped from 120ms to 42ms")
            is True
        )
        # False match (different numbers)
        assert (
            verify_passage_verbatim(conn, "res_passage", "Latency dropped from 500ms to 10ms")
            is False
        )
