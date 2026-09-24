"""Unit tests for JevClassifier orchestrator and threshold evaluations."""

import httpx
import pytest

from edward.classifiers.base import ClassificationRequest
from edward.classifiers.jev import JevClassifier
from edward.classifiers.providers.typesafe import TypeSafeProvider
from edward.db import Database
from edward.services.classification import (
    persist_classification_result,
)


def test_jev_classifier_maps_questions_to_families():
    """JevClassifier correctly evaluates packaged questions and produces typed judgments."""

    def handler(request: httpx.Request) -> httpx.Response:
        payload = {
            "id": "req_jev_test",
            "model": "typesafe/jev-1.13",
            "usage": {"input_tokens": 100, "output_tokens": 40, "cost": 0.001},
            "answers": [
                {
                    "question_id": "primary-form",
                    "primitive": "choice",
                    "choice": "repository",
                    "probabilities": {"repository": 0.88, "article": 0.08, "other": 0.04},
                    "confidence": 0.95,
                },
                {
                    "question_id": "topic-ai",
                    "primitive": "noul",
                    "noul": 0.92,
                    "confidence": 0.90,
                },
                {
                    "question_id": "topic-local-ai",
                    "primitive": "noul",
                    "noul": 0.85,
                    "confidence": 0.88,
                },
                {
                    "question_id": "signal-benchmark",
                    "primitive": "noul",
                    "noul": 0.78,
                    "confidence": 0.80,
                },
                {
                    "question_id": "evidence-depth",
                    "primitive": "score",
                    "score": 3.0,
                    "max_score": 3.0,
                },
            ],
        }
        return httpx.Response(200, json=payload)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    ts_provider = TypeSafeProvider(api_key="test-key")
    classifier = JevClassifier(transport=ts_provider, provider_name="typesafe")

    req = ClassificationRequest(
        record_id="res_jev_1",
        object_type="resource",
        text="A GitHub repository for local LLM inference benchmarks on Apple Silicon.",
        metadata={"canonical_url": "https://github.com/example/local-bench"},
    )

    # Monkeypatch transport to use mock client
    orig_eval = ts_provider.evaluate_questions
    ts_provider.evaluate_questions = lambda *args, **kwargs: orig_eval(
        *args, **{**kwargs, "client": client}
    )

    res = classifier.classify(req)

    assert res.record_id == "res_jev_1"
    assert res.provider == "typesafe"
    assert res.model == "typesafe/jev-1.13"
    assert len(res.judgments) == 5

    by_id = {j.label_or_question_id: j for j in res.judgments}
    assert by_id["primary-form"].family == "form"
    assert by_id["primary-form"].metadata["selected_choice"] == "repository"

    assert by_id["topic-ai"].family == "topic"
    assert by_id["topic-ai"].metadata["target_label"] == "ai"
    assert by_id["topic-ai"].probability == 0.92

    assert by_id["topic-local-ai"].family == "topic"
    assert by_id["topic-local-ai"].metadata["target_label"] == "ai/local-models"

    assert by_id["signal-benchmark"].family == "signal"
    assert by_id["signal-benchmark"].metadata["target_label"] == "benchmark"

    assert by_id["evidence-depth"].family == "custom"
    assert by_id["evidence-depth"].probability == 3.0


def test_jev_persist_judgments_and_threshold_labels(
    test_db: Database, monkeypatch: pytest.MonkeyPatch
):
    """Persisting Jev judgments writes to judgments table and creates derived object_labels above threshold."""

    def handler(request: httpx.Request) -> httpx.Response:
        payload = {
            "id": "req_persist_1",
            "model": "typesafe/jev-1.13",
            "answers": [
                {
                    "question_id": "primary-form",
                    "primitive": "choice",
                    "choice": "article",
                    "probabilities": {"article": 0.90, "other": 0.10},
                },
                {
                    "question_id": "topic-local-ai",
                    "primitive": "noul",
                    "noul": 0.85,  # Exceeds balanced-precision threshold (0.65)
                },
                {
                    "question_id": "signal-warning",
                    "primitive": "noul",
                    "noul": 0.20,  # Below warning threshold (0.65)
                },
            ],
        }
        return httpx.Response(200, json=payload)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    ts_provider = TypeSafeProvider(api_key="test-key")
    orig_eval = ts_provider.evaluate_questions
    ts_provider.evaluate_questions = lambda *args, **kwargs: orig_eval(
        *args, **{**kwargs, "client": client}
    )

    classifier = JevClassifier(transport=ts_provider, provider_name="typesafe")

    with test_db.transaction() as conn:
        conn.execute(
            """
            INSERT INTO resources (id, identity_key, canonical_url, title, review_state, is_deleted, created_at, updated_at)
            VALUES ('res_jev_db', 'url:jev_db', 'https://example.com/ai-note', 'AI Note', 'unreviewed', 0, '2026-01-01', '2026-01-01');
            """
        )
        conn.execute(
            """
            INSERT INTO resource_contents (id, resource_id, content_hash, clean_text, extractor, extractor_version, char_count, created_at)
            VALUES ('rc_jev_db', 'res_jev_db', 'hash_jev', 'Local AI tutorial and guide.', 'ext', '1.0', 30, '2026-01-01');
            """
        )

    req = ClassificationRequest(
        record_id="res_jev_db",
        object_type="resource",
        text="Local AI tutorial and guide.",
        metadata={"canonical_url": "https://example.com/ai-note"},
    )
    result = classifier.classify(req)

    with test_db.transaction() as conn:
        persist_classification_result(
            conn=conn,
            object_type="resource",
            object_id="res_jev_db",
            detected_form="article",
            result=result,
            text="Local AI tutorial and guide.",
        )

    # Verify judgments stored
    with test_db.connection() as conn:
        judgments = conn.execute(
            "SELECT * FROM judgments WHERE object_id = 'res_jev_db';"
        ).fetchall()
        assert len(judgments) == 3

        # Verify labels passed threshold: 'ai/local-models' should be applied, but 'warning' should not
        labels = conn.execute(
            "SELECT label_id FROM object_labels WHERE object_id = 'res_jev_db';"
        ).fetchall()
        label_set = {row["label_id"] for row in labels}
        assert "ai/local-models" in label_set
        assert "warning" not in label_set
