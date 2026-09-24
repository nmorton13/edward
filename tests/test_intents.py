"""Intent questions: user-authored 'why did I keep this?', never a ranking input."""

import json

import pytest

from edward.services.intents import (
    REGISTRY_FILE,
    accept_intent,
    assert_ranking_is_unaffected,
    intent_filter_clause,
    load_intent_labels,
    load_intent_questions,
    persist_intent_judgments,
)

# --------------------------------------------------------------------------
# The registry is a question set, not a hardcoded taxonomy
# --------------------------------------------------------------------------


def test_registry_ships_intent_questions():
    questions = load_intent_questions()

    assert questions, "the packaged intent registry must ship editable defaults"
    for q in questions:
        assert q["id"].startswith("intent-")
        assert q["primitive"] == "noul"
        assert q.get("prompt"), f"{q['id']} needs a prompt a user can rewrite"
        assert q.get("label"), f"{q['id']} must name the label it writes"


def test_questions_are_phrased_as_questions_not_fixed_folders():
    """The prompt must be something a different user could reasonably answer."""
    for q in load_intent_questions():
        assert q["prompt"].strip().endswith("?")


def test_no_question_encodes_one_persons_project_name():
    """A personal project name in the registry is the bug this design fixes."""
    raw = json.dumps(load_intent_questions()).lower()

    for personal in ("lfkai", "nates", "nathan", "my newsletter"):
        assert personal not in raw, f"registry leaks a personal intent: {personal}"


def test_dismiss_is_disabled_by_default():
    """The tool should not suggest discarding material until asked to."""
    active = {q["id"] for q in load_intent_questions()}
    everything = {q["id"] for q in load_intent_questions(include_inactive=True)}

    assert "intent-dismiss" not in active
    assert "intent-dismiss" in everything


def test_labels_map_question_ids_to_stored_labels():
    labels = load_intent_labels()

    assert labels
    assert all(isinstance(k, str) and isinstance(v, str) for k, v in labels.items())
    assert "intent-newsletter" in labels


def test_missing_registry_is_empty_not_an_error(monkeypatch):
    import edward.services.intents as mod

    monkeypatch.setattr(mod, "_resolve_registry_path", lambda: None)
    assert mod.load_intent_questions() == []


def test_malformed_registry_is_survivable(monkeypatch, tmp_path):
    import edward.services.intents as mod

    bad = tmp_path / REGISTRY_FILE
    bad.write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(mod, "_resolve_registry_path", lambda: bad)

    assert mod.load_intent_questions() == []


# --------------------------------------------------------------------------
# Provenance: classifier output may never look like the user's decision
# --------------------------------------------------------------------------


def _seed_resource(conn, rid="res_1"):
    conn.execute(
        """
        INSERT INTO resources (id, canonical_url, url_hash, identity_key, title,
                               created_at, updated_at)
        VALUES (?, 'https://example.com/a', 'h', 'k', 'A post', '2026-01-01', '2026-01-01');
        """,
        (rid,),
    )
    return rid


def test_classifier_intents_are_stored_as_inactive_suggestions(test_db):
    with test_db.transaction() as conn:
        _seed_resource(conn)
        written = persist_intent_judgments(
            conn,
            object_type="resource",
            object_id="res_1",
            answers={"intent-newsletter": {"noul": 0.91}},
        )

    assert "newsletter-material" in written
    with test_db.connection() as conn:
        row = conn.execute(
            "SELECT source, is_active FROM intents WHERE intent = 'newsletter-material';"
        ).fetchone()
    assert row["source"] == "classifier"
    assert row["is_active"] == 0, "a machine guess must not claim to be the user's intent"


def test_below_threshold_answers_are_not_recorded(test_db):
    with test_db.transaction() as conn:
        _seed_resource(conn)
        written = persist_intent_judgments(
            conn,
            object_type="resource",
            object_id="res_1",
            answers={"intent-newsletter": {"noul": 0.2}},
        )

    assert written == []
    with test_db.connection() as conn:
        assert conn.execute("SELECT count(*) c FROM intents;").fetchone()["c"] == 0


def test_accepting_a_suggestion_makes_it_human_and_active(test_db):
    with test_db.transaction() as conn:
        _seed_resource(conn)
        persist_intent_judgments(
            conn,
            object_type="resource",
            object_id="res_1",
            answers={"intent-reference": {"noul": 0.8}},
        )
    with test_db.transaction() as conn:
        accept_intent(conn, object_type="resource", object_id="res_1", intent="reference")

    with test_db.connection() as conn:
        row = conn.execute(
            "SELECT source, is_active FROM intents WHERE intent = 'reference';"
        ).fetchone()
    assert row["source"] == "human"
    assert row["is_active"] == 1


def test_a_human_intent_is_not_downgraded_by_a_later_classifier_run(test_db):
    """The user's own decision outranks anything the classifier later says."""
    with test_db.transaction() as conn:
        _seed_resource(conn)
        accept_intent(conn, object_type="resource", object_id="res_1", intent="reference")
    with test_db.transaction() as conn:
        persist_intent_judgments(
            conn,
            object_type="resource",
            object_id="res_1",
            answers={"intent-reference": {"noul": 0.5}},
        )

    with test_db.connection() as conn:
        row = conn.execute(
            "SELECT source, is_active FROM intents WHERE intent = 'reference';"
        ).fetchone()
    assert row["source"] == "human"
    assert row["is_active"] == 1


def test_intent_labels_are_created_and_attached(test_db):
    with test_db.transaction() as conn:
        _seed_resource(conn)
        persist_intent_judgments(
            conn,
            object_type="resource",
            object_id="res_1",
            answers={"intent-try-later": {"noul": 0.75}},
        )

    with test_db.connection() as conn:
        label = conn.execute("SELECT family FROM labels WHERE id = 'try-later';").fetchone()
        attached = conn.execute(
            """
            SELECT confidence FROM object_labels
            WHERE object_id = 'res_1' AND label_id = 'try-later';
            """
        ).fetchone()
    assert label is not None
    assert attached is not None
    assert attached["confidence"] == pytest.approx(0.75)


def test_answers_for_unknown_questions_are_ignored(test_db):
    with test_db.transaction() as conn:
        _seed_resource(conn)
        written = persist_intent_judgments(
            conn,
            object_type="resource",
            object_id="res_1",
            answers={"topic-ai": {"noul": 0.99}},
        )

    assert written == [], "intent persistence must not swallow topic questions"


# --------------------------------------------------------------------------
# The guardrail: facets, never scores
# --------------------------------------------------------------------------


def test_filter_clause_is_an_in_list_not_a_score():
    clause, params = intent_filter_clause(["reference", "try-later"])

    assert "IN (?, ?)" in clause
    assert params == ["reference", "try-later"]
    for forbidden in ("rank", "score", "weight", "rrf", "order by"):
        assert forbidden not in clause.lower()


def test_empty_filter_is_a_no_op():
    clause, params = intent_filter_clause([])

    assert clause == ""
    assert params == []


def test_intents_are_not_wired_into_ranking_tables(test_db):
    """Intents must stay out of embeddings and search documents entirely."""
    with test_db.transaction() as conn:
        _seed_resource(conn)
        persist_intent_judgments(
            conn,
            object_type="resource",
            object_id="res_1",
            answers={"intent-newsletter": {"noul": 0.99}},
        )
        assert_ranking_is_unaffected(conn)

    with test_db.connection() as conn:
        # Storing an intent must not create an embedding or reorder search.
        assert conn.execute("SELECT count(*) c FROM embeddings;").fetchone()["c"] == 0
        docs = conn.execute(
            "SELECT count(*) c FROM search_documents WHERE object_id = 'res_1';"
        ).fetchone()["c"]
        assert docs == 0


def test_active_intent_is_what_search_filters_on(test_db):
    """search.py filters on is_active = 1 — suggestions must not leak into results."""
    with test_db.transaction() as conn:
        _seed_resource(conn)
        persist_intent_judgments(
            conn,
            object_type="resource",
            object_id="res_1",
            answers={"intent-newsletter": {"noul": 0.95}},
        )

    with test_db.connection() as conn:
        active = conn.execute("SELECT count(*) c FROM intents WHERE is_active = 1;").fetchone()["c"]
    assert active == 0, "a suggestion must not appear in an intent-filtered search"
