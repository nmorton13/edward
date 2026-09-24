"""Tests for packaged classification registries."""

import json
from pathlib import Path


def get_registries_dir() -> Path:
    return Path(__file__).parent.parent / "src" / "edward" / "registries"


def test_topics_registry():
    path = get_registries_dir() / "topics-v1.json"
    assert path.exists()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["version"] == "1.0"
    assert data["family"] == "topic"
    assert len(data["entries"]) > 0

    ids = set()
    for entry in data["entries"]:
        assert "id" in entry
        assert "description" in entry
        assert "inclusion_criteria" in entry
        assert "exclusion_criteria" in entry
        assert entry["active"] is True
        assert entry["id"] not in ids, f"Duplicate topic ID: {entry['id']}"
        ids.add(entry["id"])


def test_forms_registry():
    path = get_registries_dir() / "forms-v1.json"
    assert path.exists()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["version"] == "1.0"
    assert data["family"] == "form"
    assert len(data["entries"]) > 0

    ids = set()
    for entry in data["entries"]:
        assert "id" in entry
        assert "description" in entry
        assert entry["id"] not in ids, f"Duplicate form ID: {entry['id']}"
        ids.add(entry["id"])


def test_signals_registry():
    path = get_registries_dir() / "signals-v1.json"
    assert path.exists()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["version"] == "1.0"
    assert data["family"] == "signal"
    assert len(data["entries"]) > 0

    ids = set()
    for entry in data["entries"]:
        assert "id" in entry
        assert "description" in entry
        assert entry["id"] not in ids, f"Duplicate signal ID: {entry['id']}"
        ids.add(entry["id"])


def test_jev_questions_registry():
    path = get_registries_dir() / "jev-questions-v1.json"
    assert path.exists()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["version"] == "1.0"
    assert len(data["questions"]) > 0

    ids = set()
    primitives = set()
    for q in data["questions"]:
        assert "id" in q
        assert "primitive" in q
        assert q["primitive"] in ["choice", "noul", "score"]
        primitives.add(q["primitive"])
        assert "prompt" in q
        assert "instructions" in q
        assert q["id"] not in ids, f"Duplicate question ID: {q['id']}"
        ids.add(q["id"])

    assert "choice" in primitives
    assert "noul" in primitives


def test_thresholds_registry():
    path = get_registries_dir() / "thresholds-v1.json"
    assert path.exists()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["version"] == "1.0"
    assert "default_policy" in data
    assert data["default_policy"] in data["policies"]
    for _policy_name, policy in data["policies"].items():
        assert "thresholds" in policy
        for _q_id, val in policy["thresholds"].items():
            assert 0.0 <= val <= 1.0


def test_jev_question_label_mapping_integrity():
    from edward.classifiers.jev import QUESTION_LABEL_MAPPING

    registries_dir = get_registries_dir()
    topics = {
        e["id"]
        for e in json.loads((registries_dir / "topics-v1.json").read_text(encoding="utf-8"))[
            "entries"
        ]
    }
    signals = {
        e["id"]
        for e in json.loads((registries_dir / "signals-v1.json").read_text(encoding="utf-8"))[
            "entries"
        ]
    }
    jev_questions = {
        q["id"]
        for q in json.loads((registries_dir / "jev-questions-v1.json").read_text(encoding="utf-8"))[
            "questions"
        ]
    }

    for q_id, (family, target_label) in QUESTION_LABEL_MAPPING.items():
        assert q_id in jev_questions, f"Mapped question {q_id} not in jev-questions-v1.json"
        if family == "topic":
            assert target_label in topics, f"Target topic {target_label} not in topics-v1.json"
        elif family == "signal":
            assert target_label in signals, f"Target signal {target_label} not in signals-v1.json"


def test_thresholds_cover_all_mapped_questions():
    from edward.classifiers.jev import QUESTION_LABEL_MAPPING

    thresholds_data = json.loads(
        (get_registries_dir() / "thresholds-v1.json").read_text(encoding="utf-8")
    )
    for policy_name, policy in thresholds_data["policies"].items():
        t = policy["thresholds"]
        for q_id, (_family, target_label) in QUESTION_LABEL_MAPPING.items():
            assert q_id in t or target_label in t, (
                f"Missing threshold in policy '{policy_name}' for {q_id} / {target_label}"
            )
