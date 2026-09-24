"""Tests for the three-tier answer engine and CLI edward ask command."""

import json

import pytest
from typer.testing import CliRunner

from edward.cli import app
from edward.db import Database
from edward.models import CaptureInput
from edward.services.answer import (
    DEFAULT_RESEARCH_LIMIT,
    _canonicalize_explicit_ids,
    answer_question,
    build_lookup_answer,
    synthesize_answer,
    try_deterministic_answer,
)
from edward.services.capture import capture_item

runner = CliRunner()


def test_explicit_evidence_id_labels_become_citations_only_for_supplied_items():
    notes = "- ID: res_known – Relevant detail.\n- ID: res_other – Unsupported detail."
    normalized = _canonicalize_explicit_ids(notes, {"res_known"})
    assert "[#res_known]" in normalized
    assert "ID: res_other" in normalized


def test_tier1_deterministic_counts(test_db: Database):
    """Tier 1 answers count questions deterministically using static SQL without models."""
    with test_db.transaction() as conn:
        conn.execute(
            """
            INSERT INTO captures (id, origin_namespace, collection_channel, collector, acquisition_method, retrieved_at, raw_content, review_state, is_deleted, created_at, updated_at)
            VALUES ('cap_t1_1', 'manual', 'manual', 'test', 'test', '2026-01-01', 'Note 1', 'unreviewed', 0, '2026-01-01', '2026-01-01'),
                   ('cap_t1_2', 'manual', 'manual', 'test', 'test', '2026-01-01', 'Note 2', 'unreviewed', 0, '2026-01-01', '2026-01-01');
            """
        )

    with test_db.connection() as conn:
        ans = try_deterministic_answer(conn, "How many captures are saved?")
        assert ans is not None
        assert ans["tier"] == 1
        assert "2 captures" in ans["answer"]
        assert ans["data"]["count"] == 2


def test_tier1_deterministic_date_and_title(test_db: Database):
    """Tier 1 answers created_at and title questions deterministically."""
    with test_db.transaction() as conn:
        conn.execute(
            """
            INSERT INTO resources (id, identity_key, canonical_url, title, review_state, is_deleted, created_at, updated_at)
            VALUES ('res_t1_title', 'url:t1', 'https://example.com/t1', 'Apple Silicon LLM Guide', 'unreviewed', 0, '2026-01-15T10:00:00Z', '2026-01-15T10:00:00Z');
            """
        )

    with test_db.connection() as conn:
        # Title query
        ans_title = try_deterministic_answer(conn, "What is the title of res_t1_title?")
        assert ans_title is not None
        assert ans_title["tier"] == 1
        assert "Apple Silicon LLM Guide" in ans_title["answer"]

        # Date query
        ans_date = try_deterministic_answer(conn, "When was res_t1_title captured?")
        assert ans_date is not None
        assert ans_date["tier"] == 1
        assert "2026-01-15T10:00:00Z" in ans_date["answer"]


def test_tier2_lookup_with_no_model(test_db: Database):
    """When --no-model is passed, answers fall back to concise retrieval lookup."""
    with test_db.transaction() as conn:
        conn.execute(
            """
            INSERT INTO resources (id, identity_key, canonical_url, title, review_state, is_deleted, created_at, updated_at)
            VALUES ('res_t2_lookup', 'url:t2', 'https://example.com/t2', 'Ollama Performance', 'unreviewed', 0, '2026-01-01', '2026-01-01');
            """
        )
        conn.execute(
            """
            INSERT INTO resource_contents (id, resource_id, content_hash, clean_text, extractor, extractor_version, char_count, created_at)
            VALUES ('rc_t2', 'res_t2_lookup', 'hash_t2', 'Ollama runs quantized GGUF models efficiently.', 'ext', '1.0', 45, '2026-01-01');
            """
        )
        from edward.services.lifecycle import reindex_object_document

        reindex_object_document(conn, "resource", "res_t2_lookup")

    with test_db.connection() as conn:
        res = answer_question(conn, query="Ollama GGUF models performance", no_model=True)
        assert res["tier"] == 2
        assert "evidence_packet" in res
        assert "[#res_t2_lookup]" in res["answer"]


def test_lookup_displays_every_result_without_cutting_text():
    """A lookup should show all retrieved evidence, including complete short posts."""
    full_post = "Local models help with private research. " * 8
    packet = {
        "query": "local models",
        "items": [
            {"id": f"res_{index}", "text": full_post, "assertion_role": "source-claim"}
            for index in range(6)
        ],
    }

    answer = build_lookup_answer(packet)

    assert "Found 6 relevant items" in answer["answer"]
    assert "6. [#res_5]" in answer["answer"]
    assert full_post.strip() in answer["answer"]
    assert "..." not in answer["answer"]
    assert len(answer["citations"]) == 6


def test_cli_ask_command(test_db: Database, monkeypatch: pytest.MonkeyPatch):
    """Test edward ask command in both JSON and human modes."""
    monkeypatch.setenv("EDWARD_DB_PATH", str(test_db.db_path))

    with test_db.transaction() as conn:
        conn.execute(
            """
            INSERT INTO captures (id, origin_namespace, collection_channel, collector, acquisition_method, retrieved_at, raw_content, review_state, is_deleted, created_at, updated_at)
            VALUES ('cap_cli_ask', 'manual', 'manual', 'test', 'test', '2026-01-01', 'Test Capture', 'unreviewed', 0, '2026-01-01', '2026-01-01');
            """
        )

    # Tier 1 direct count via CLI
    res = runner.invoke(app, ["ask", "how many captures", "--json"])
    assert res.exit_code == 0
    data = json.loads(res.stdout)
    assert data["tier"] == 1
    assert "1 captures" in data["answer"]

    # Tier 2 lookup via CLI
    res2 = runner.invoke(app, ["ask", "anything", "--no-model", "--json"])
    assert res2.exit_code == 0
    data2 = json.loads(res2.stdout)
    assert data2["tier"] == 2


def test_cli_ask_keeps_literal_citation_ids(test_db: Database, monkeypatch: pytest.MonkeyPatch):
    """Rich must not interpret evidence references as formatting tags."""
    monkeypatch.setenv("EDWARD_DB_PATH", str(test_db.db_path))
    monkeypatch.setattr(
        "edward.services.answer.answer_question",
        lambda *args, **kwargs: {
            "tier": 2,
            "answer": "1. [#res_example] (source-claim): Complete post text",
            "coverage": {"retrieved_items": 12, "reviewed_batches": 2, "cited_items": 1},
            "citations": ["[#res_example]"],
        },
    )

    result = runner.invoke(app, ["ask", "local models", "--no-model"])

    assert result.exit_code == 0
    assert "1. [#res_example]" in result.stdout
    assert "Evidence reviewed: 12 items in 2 batches; 1 cited." in result.stdout
    assert "Citations: [#res_example]" in result.stdout


def test_default_ask_returns_broad_packet_for_calling_agent(
    test_db: Database, monkeypatch: pytest.MonkeyPatch
):
    """No-model Ask should expose the wider retrieved set to a calling agent."""
    limits = []

    def fake_search(_conn, *, query, limit, project_filter):
        limits.append(limit)
        return {
            "query": query,
            "items": [{"id": f"res_{index}", "text": f"Source {index}"} for index in range(12)],
        }

    monkeypatch.setattr("edward.services.answer.search_hybrid", fake_search)
    with test_db.connection() as conn:
        result = answer_question(conn, "What did I save about local models?", no_model=True)

    assert limits == [DEFAULT_RESEARCH_LIMIT]
    assert len(result["evidence_packet"]["items"]) == 12
    assert "12. [#res_11]" in result["answer"]


def test_model_reviews_all_retrieved_sources_and_matching_passages(test_db: Database):
    """A cited source after item ten and a matched document passage reach synthesis."""
    with test_db.transaction() as conn:
        conn.executemany(
            """
            INSERT INTO resources (id, identity_key, canonical_url, title, review_state,
                                   is_deleted, created_at, updated_at)
            VALUES (?, ?, ?, ?, 'unreviewed', 0, '2026-01-01', '2026-01-01')
            """,
            [
                (f"res_{index}", f"url:{index}", f"https://example.com/{index}", f"Source {index}")
                for index in range(12)
            ],
        )

    items = [
        {
            "id": f"res_{index}",
            "kind": "resource",
            "text": f"Opening of source {index}",
            "source": {"title": f"Source {index}", "origin_namespace": "documents"},
            "supporting_passages": [{"passage": "Relevant passage about small local models"}]
            if index == 11
            else [],
        }
        for index in range(12)
    ]

    class ResearchClient:
        provider = "ollama"
        location = "local"
        base_url = "http://localhost:11434/v1"
        model = "test-local"

        def __init__(self):
            self.requests = []

        def chat_completion(self, messages, **kwargs):
            self.requests.append(messages)
            if len(self.requests) == 1:
                return "First batch covers local models [#res_0].", None
            if len(self.requests) == 2:
                return "Document passage covers small local models [#res_11].", None
            return "Saved sources and the document discuss local models [#res_0] [#res_11].", None

    client = ResearchClient()
    with test_db.connection() as conn:
        result = synthesize_answer(conn, "local models", {"items": items}, client)

    assert result["tier"] == 3
    assert result["coverage"] == {
        "retrieved_items": 12,
        "reviewed_batches": 2,
        "cited_items": 2,
    }
    assert len(client.requests) == 3
    assert "Relevant passage about small local models" in client.requests[1][1]["content"]
    assert "[#res_11]" in result["answer"]


def test_model_repairs_an_inexact_quotation_once(test_db: Database):
    """A local model can turn an inexact quote into a cited paraphrase."""
    with test_db.transaction() as conn:
        conn.execute(
            """
            INSERT INTO resources (id, identity_key, canonical_url, title, review_state,
                                   is_deleted, created_at, updated_at)
            VALUES ('res_quote', 'url:quote', 'https://example.com/quote', 'Quote source',
                    'unreviewed', 0, '2026-01-01', '2026-01-01')
            """
        )
        conn.execute(
            """
            INSERT INTO resource_contents (id, resource_id, content_hash, clean_text,
                                           extractor, extractor_version, char_count, created_at)
            VALUES ('rc_quote', 'res_quote', 'hash_quote', 'Small models run locally.',
                    'test', '1', 25, '2026-01-01')
            """
        )

    class CorrectingClient:
        provider = "ollama"
        location = "local"
        base_url = "http://localhost:11434/v1"
        model = "test-local"

        def __init__(self):
            self.calls = 0

        def chat_completion(self, messages, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return 'They "run everywhere" [#res_quote].', None
            return "The source says small models can run locally [#res_quote].", None

    client = CorrectingClient()
    with test_db.connection() as conn:
        result = synthesize_answer(
            conn,
            "What did I save about small models?",
            {"items": [{"id": "res_quote", "text": "Small models run locally."}]},
            client,
        )

    assert result["tier"] == 3
    assert client.calls == 2
    assert "run locally [#res_quote]" in result["answer"]


def test_model_answer_uses_canonical_citation_tokens(test_db: Database):
    with test_db.transaction() as conn:
        conn.execute(
            """
            INSERT INTO resources (id, identity_key, canonical_url, review_state,
                                   is_deleted, created_at, updated_at)
            VALUES ('res_cite', 'url:cite', 'https://example.com/cite', 'unreviewed',
                    0, '2026-01-01', '2026-01-01')
            """
        )

    class BareCitationClient:
        provider = "ollama"
        location = "local"
        base_url = "http://localhost:11434/v1"
        model = "test-local"

        def chat_completion(self, messages, **kwargs):
            return "Saved source describes local inference [res_cite].", None

    with test_db.connection() as conn:
        result = synthesize_answer(
            conn,
            "local inference",
            {"items": [{"id": "res_cite", "text": "Local inference is available."}]},
            BareCitationClient(),
        )

    assert result["tier"] == 3
    assert result["citations"] == ["[#res_cite]"]
    assert "[#res_cite]" in result["answer"]


def test_agent_can_follow_citation_to_capture_provenance(
    test_db: Database, monkeypatch: pytest.MonkeyPatch
):
    """show --json exposes the source event behind a cited X resource."""
    monkeypatch.setenv("EDWARD_DB_PATH", str(test_db.db_path))
    with test_db.transaction() as conn:
        captured = capture_item(
            conn,
            CaptureInput(
                url="https://x.com/i/status/123456789",
                text="A saved post about local models",
                origin_namespace="x",
                origin_id="123456789",
                collection_channel="birdclaw",
                collector="test-agent",
                acquisition_method="archive",
            ),
        )

    result = runner.invoke(app, ["show", captured["resource_id"], "--json"])

    assert result.exit_code == 0
    source_event = json.loads(result.stdout)["captures"][0]
    assert source_event["origin_namespace"] == "x"
    assert source_event["origin_id"] == "123456789"
    assert source_event["collection_channel"] == "birdclaw"
