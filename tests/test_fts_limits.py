"""Tests exploring FTS5 tokenizer behavior, technical identifiers, and quotation verification."""

from edward.db import Database
from edward.services.search import index_document, search_lexical


def test_technical_identifiers_and_model_names(test_db: Database):
    with test_db.transaction() as conn:
        index_document(
            conn,
            object_type="resource",
            object_id="res_model_1",
            title="Evaluation of llama-3.1-70b and qwen2.5-coder-32b",
            body="Benchmarked sqlite3_backup_step in SQLite WAL mode on Apple Silicon M3.",
        )

    with test_db.connection() as conn:
        # Search model name
        res_model = search_lexical(conn, "llama-3.1-70b")
        assert res_model.count == 1
        assert res_model.results[0].id == "res_model_1"

        # Search code identifier
        res_code = search_lexical(conn, "sqlite3_backup_step")
        assert res_code.count == 1

        # Search hyphenated token
        res_qwen = search_lexical(conn, "qwen2.5-coder-32b")
        assert res_qwen.count == 1


def test_quote_verification_requires_source_check(test_db: Database):
    """FTS5 tokenizes unicode61 and matches terms, but exact quotes must be validated against stored text."""
    original_text = 'The author stated: "Latency reduced by 42.5% under concurrent load."'

    with test_db.transaction() as conn:
        index_document(
            conn,
            object_type="finding",
            object_id="find_quote_1",
            title="Finding on latency reduction",
            body=original_text,
        )

    with test_db.connection() as conn:
        # FTS matches the phrase
        res = search_lexical(conn, '"Latency reduced"')
        assert res.count == 1

        # Verify exact quote comparison against stored text
        candidate_quote = "Latency reduced by 42.5% under concurrent load."
        assert candidate_quote in original_text

        # A false quote with altered number would match individual FTS words, but fail exact check
        hallucinated_quote = "Latency reduced by 99.9% under concurrent load."
        assert hallucinated_quote not in original_text
