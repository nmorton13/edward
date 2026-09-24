"""Tests for classifier wire models, stubs, heuristics, and judgment persistence."""

import pytest

from edward.classifiers.base import ClassificationRequest
from edward.classifiers.dry_run import DryRunClassifier
from edward.classifiers.local import LocalClassifier
from edward.db import Database
from edward.services.classification import detect_primary_form, run_classification_pipeline
from edward.services.lifecycle import add_label


@pytest.fixture
def test_db(tmp_path):
    db_file = tmp_path / "test_edward.db"
    db = Database(db_file)
    db.run_migrations()
    return db


def test_dry_run_classifier_probabilities():
    classifier = DryRunClassifier()
    req = ClassificationRequest(
        object_type="resource",
        record_id="test_rec_1",
        text="Check out this new repository for optimizing deep learning neural networks: https://github.com/org/repo",
        metadata={"url": "https://github.com/org/repo"},
    )
    res = classifier.classify(req)
    assert res.provider in ("dry_run", "dry-run")
    assert res.model == "dry-run-v1"
    assert len(res.judgments) > 0

    # Verify that Choice question distributions sum to 1.0
    for j in res.judgments:
        if j.primitive == "choice" and isinstance(j.answer, dict):
            probs = j.answer.get("probabilities", {})
            total_prob = sum(probs.values())
            assert pytest.approx(total_prob, rel=1e-3) == 1.0


def test_local_classifier_stub():
    classifier = LocalClassifier()
    req = ClassificationRequest(
        object_type="capture",
        record_id="test_rec_2",
        text="A brief thought about writing software and designing tools.",
    )
    res = classifier.classify(req)
    assert res.provider == "local"
    assert res.model == "local-rules-v1"
    assert res.cost == 0.0
    assert len(res.judgments) > 0


def test_detect_primary_form_heuristics():
    assert (
        detect_primary_form("https://github.com/anthropics/anthropic-sdk-python", "SDK repository")
        == "repository"
    )
    assert (
        detect_primary_form("https://arxiv.org/abs/2301.00000", "Deep learning research paper")
        == "paper"
    )
    assert (
        detect_primary_form("https://x.com/user/status/12345", "Just launched our new feature!")
        == "x-post"
    )
    assert detect_primary_form("https://twitter.com/user/status/12345", "Quick thought") == "x-post"
    assert detect_primary_form(None, "Quick personal note for later") == "personal-note"
    assert (
        detect_primary_form(
            "https://example.com/blog/scaling", "A comprehensive guide to scaling systems" * 20
        )
        == "article"
    )


def test_run_classification_pipeline_preserves_human_labels(test_db, monkeypatch):
    monkeypatch.setenv("EDWARD_CLASSIFIER_PROVIDER", "dry-run")
    with test_db.transaction() as conn:
        # Create resource
        conn.execute(
            """
            INSERT INTO resources (id, identity_key, canonical_url, primary_form, review_state, is_deleted, created_at, updated_at)
            VALUES ('res_cls_1', 'url:github', 'https://github.com/org/repo', 'other', 'unreviewed', 0, datetime('now'), datetime('now'));
            """
        )
        # Add a human-assigned label
        add_label(
            conn,
            "resource",
            "res_cls_1",
            label_id="topic/software-engineering",
            source="human",
            actor="user",
        )

        # Run automated classification pipeline
        run_classification_pipeline(
            conn,
            object_type="resource",
            object_id="res_cls_1",
        )

        # 1. Judgments were recorded
        judgments = conn.execute(
            "SELECT * FROM judgments WHERE object_id = 'res_cls_1';"
        ).fetchall()
        assert len(judgments) > 0

        # 2. Human label remains untouched with source = 'human'
        hl = conn.execute(
            "SELECT source FROM object_labels WHERE object_id = 'res_cls_1' AND label_id = 'topic/software-engineering';"
        ).fetchone()
        assert hl is not None
        assert hl["source"] == "human"

        # 3. Form heuristic updated primary_form
        r = conn.execute("SELECT primary_form FROM resources WHERE id = 'res_cls_1';").fetchone()
        assert r["primary_form"] == "repository"
