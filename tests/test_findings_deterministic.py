"""Tests that finding extraction is deterministic and never reaches for a model.

Edward's finding-extraction stage can ask a model to read a resource and return claims,
quotations, questions, and entities. That path produced no findings in its history,
while an agent-submitted bundle produced findings immediately. Extraction is therefore
deterministic by design: a caller may supply a client, but the stage never constructs one.
"""

from unittest.mock import MagicMock

import pytest

from edward.services.findings import extract_findings_for_resource


@pytest.fixture(autouse=True)
def _no_ambient_answerer(monkeypatch):
    """Guard against ambient answerer configuration in these tests."""
    for name in (
        "EDWARD_ANSWERER_MODE",
        "EDWARD_ANSWER_BASE_URL",
        "EDWARD_ANSWER_MODEL",
    ):
        monkeypatch.delenv(name, raising=False)


def _seed_resource(db, resource_id: str, text: str) -> None:
    """Insert a resource holding extractable content, matching the existing test pattern."""
    with db.transaction() as conn:
        conn.execute(
            """
            INSERT INTO resources (id, identity_key, canonical_url, title, review_state, is_deleted, created_at, updated_at)
            VALUES (?, ?, ?, 'Seeded Resource', 'unreviewed', 0, '2026-01-01', '2026-01-01');
            """,
            (resource_id, f"url:{resource_id}", f"https://example.com/{resource_id}"),
        )
        conn.execute(
            """
            INSERT INTO resource_contents (id, resource_id, content_hash, clean_text, extractor, extractor_version, char_count, created_at)
            VALUES (?, ?, ?, ?, 'test', '1.0', ?, '2026-01-01');
            """,
            (f"rc_{resource_id}", resource_id, f"hash_{resource_id}", text, len(text)),
        )


def test_extract_findings_does_not_construct_a_model_client(test_db, monkeypatch):
    """The extraction stage must never build a client of its own."""
    constructed = []

    def _forbidden(*args, **kwargs):
        constructed.append(1)
        raise AssertionError("finding extraction must not construct a model client")

    monkeypatch.setattr("edward.services.findings.get_answer_client", _forbidden, raising=False)

    _seed_resource(
        test_db,
        "res_det_1",
        "Data centers consume substantial electricity.\n"
        "- Most facilities use closed-loop cooling to cut water draw.\n"
        "> We estimate rates fell modestly in the United States.\n"
        "Is that trend sustainable?\n",
    )

    with test_db.transaction() as conn:
        result = extract_findings_for_resource(conn, "res_det_1")

    assert constructed == [], "no model client may be constructed during extraction"
    assert result["findings_count"] >= 1


def test_processor_finding_extraction_stage_is_model_free(monkeypatch):
    """The processor's finding-extraction branch must not call out to a model."""
    from edward.services import processor

    def _forbidden(*args, **kwargs):
        raise AssertionError("the finding-extraction stage must not build a model client")

    # The processor imports get_answer_client inside the branch; patch the source module.
    monkeypatch.setattr("edward.services.llm.get_answer_client", _forbidden)

    job = {"id": "job_test", "stage": "finding-extraction", "resource_id": "res_test"}
    context = {
        "text": "Rates fell modestly under an instrumental variables approach.\n"
        "- Regional load grew faster than rates did over the same period.\n"
        "Is the effect durable beyond the study window?\n",
        "data_class": "public_web",
        "content_hash": "abc123",
        "input_hash": "abc123",
    }

    result = processor._perform_job_work(MagicMock(), job, context)

    assert result["status"] == "completed"
    assert result["payload"].findings, "deterministic extraction must still produce findings"


def test_extraction_honours_a_caller_supplied_client(test_db):
    """A caller that supplies a client gets generative extraction; absence uses heuristics."""
    _seed_resource(
        test_db,
        "res_det_2",
        "Local models reduce inference cost dramatically.\n"
        "- Quantization lowers inference cost substantially.\n"
        "Should every team run one?\n",
    )

    client = MagicMock()
    client.model = "caller-supplied-model"
    client.provider = "local"
    client.location = "local"
    client.base_url = "http://localhost:11434/v1"
    client.chat_completion.return_value = (None, None)  # forces the heuristic floor

    with test_db.transaction() as conn:
        result = extract_findings_for_resource(conn, "res_det_2", llm_client=client)

    assert result["findings_count"] >= 1
    assert client.chat_completion.called, "a caller-supplied client must actually be used"


def test_heuristic_extraction_recognises_questions_and_quotations(test_db):
    """The deterministic floor must still recognise the roles it advertises."""
    _seed_resource(
        test_db,
        "res_det_3",
        "Is the grid ready for this load?\n"
        "- Regional rates are rising sharply in several states.\n"
        "> We estimate rates fell modestly in the United States.\n",
    )

    with test_db.transaction() as conn:
        result = extract_findings_for_resource(conn, "res_det_3")

    with test_db.connection() as conn:
        rows = conn.execute(
            "SELECT assertion_role, statement FROM findings WHERE resource_id = ?;",
            ("res_det_3",),
        ).fetchall()

    roles = {row["assertion_role"] for row in rows}
    assert result["findings_count"] == len(rows)
    assert "question" in roles, f"expected a question role, got {roles}"
    assert "direct-quotation" in roles, f"expected a quotation role, got {roles}"
    assert "source-claim" in roles, f"expected a claim role, got {roles}"


def test_extraction_records_its_own_extractor_identity(test_db):
    """Findings must carry the extractor that produced them, for provenance."""
    _seed_resource(
        test_db,
        "res_det_4",
        "- Deterministic extraction must be attributable to its extractor.\n"
        "Is provenance preserved?\n",
    )

    with test_db.transaction() as conn:
        extract_findings_for_resource(conn, "res_det_4")

    with test_db.connection() as conn:
        extractors = {
            row["extractor"]
            for row in conn.execute(
                "SELECT extractor FROM findings WHERE resource_id = ?;", ("res_det_4",)
            )
        }

    assert extractors, "findings must record an extractor"
    assert all(extractors), "extractor identity must not be null"
    assert not any(e.startswith("llm-") for e in extractors), (
        f"no model extractor should appear, got {extractors}"
    )
