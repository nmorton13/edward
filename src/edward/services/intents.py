"""Intent questions: why you kept something, as an editable question set.

A topic says what a document is about. An intent says why it is in your
archive — newsletter material, reference, something to try, something to throw
away. Edward's original design hardcoded the author's personal intents, which
is exactly what makes a tool unusable by anyone else. This module turns intent
into a *user-authored question registry* instead: the same typed-question
mechanism Jev already uses for topics and forms, so a new user rewrites the
prompts for their own purposes or writes their own.

The hard rule: **an intent is a facet, never a score.** Intents are stored as
labels and can be filtered by, but they must never enter retrieval ranking.
Nothing in this module reads or writes embeddings, RRF weights, or search
ordering, and ``assert_ranking_is_unaffected`` exists to keep it that way.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any

from edward.models import generate_id
from edward.services.lifecycle import add_intent

logger = logging.getLogger(__name__)

REGISTRY_FILE = "intent-questions-v1.json"


def load_intent_questions(include_inactive: bool = False) -> list[dict[str, Any]]:
    """Load intent questions from the packaged registry.

    Unlike Jev's topics, this registry is meant to be edited by the person
    using the tool. It lives in the data directory when one exists so
    upgrades do not overwrite a user's own questions.
    """
    path = _resolve_registry_path()
    if not path or not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        logger.warning("Failed to load %s: %s", REGISTRY_FILE, exc)
        return []

    questions = data.get("intents", [])
    if not isinstance(questions, list):
        return []
    if include_inactive:
        return [q for q in questions if isinstance(q, dict)]
    return [q for q in questions if isinstance(q, dict) and q.get("active", True)]


def _resolve_registry_path() -> Path | None:
    """Prefer a user-owned copy in the data dir, then the packaged default."""
    import os

    data_dir = os.environ.get("EDWARD_DATA_DIR")
    if data_dir:
        candidate = Path(data_dir) / "registries" / REGISTRY_FILE
        if candidate.exists():
            return candidate
    try:
        from edward.db import get_default_data_dir

        user_copy = get_default_data_dir() / "registries" / REGISTRY_FILE
        if user_copy.exists():
            return user_copy
    except Exception:
        pass
    packaged = Path(__file__).parent.parent / "registries" / REGISTRY_FILE
    return packaged if packaged.exists() else None


def load_intent_labels() -> dict[str, str]:
    """Map intent question ID to the label it writes, for active questions."""
    return {
        str(q["id"]): str(q.get("label") or q["id"]) for q in load_intent_questions() if q.get("id")
    }


def persist_intent_judgments(
    conn: sqlite3.Connection,
    *,
    object_type: str,
    object_id: str,
    answers: dict[str, Any],
    source: str = "classifier",
    threshold: float = 0.5,
) -> list[str]:
    """Store intent answers as labels and suggested intents.

    Anything the classifier asserts is written with a non-human ``source`` and
    as an *inactive* suggestion, so it can never masquerade as the user's own
    decision. A user accepting a suggestion promotes it separately.
    """
    labels = load_intent_labels()
    written: list[str] = []

    for question_id, label in labels.items():
        answer = answers.get(question_id)
        if not isinstance(answer, dict):
            continue
        probability = answer.get("noul", answer.get("probability"))
        if probability is None:
            continue
        try:
            probability = float(probability)
        except (TypeError, ValueError):
            continue
        if probability < threshold:
            continue

        _attach_intent_label(conn, object_type, object_id, label, source, probability)
        add_intent(
            conn,
            object_type,
            object_id,
            label,
            source=source,
            actor=f"intent-registry:{REGISTRY_FILE}",
            is_active=False,  # a suggestion until a human accepts it
        )
        written.append(label)

    return written


def _attach_intent_label(
    conn: sqlite3.Connection,
    object_type: str,
    object_id: str,
    label: str,
    source: str,
    confidence: float,
) -> None:
    """Ensure the label exists in the registry, then attach it to the object."""
    row = conn.execute("SELECT id FROM labels WHERE id = ?;", (label,)).fetchone()
    if not row:
        # labels.family is a foreign key to label_families, which a fresh
        # database has not necessarily populated yet.
        conn.execute(
            """
            INSERT OR IGNORE INTO label_families (id, description, created_at)
            VALUES ('custom', 'Taxonomic custom labels', datetime('now'));
            """
        )
        conn.execute(
            """
            INSERT OR IGNORE INTO labels (id, family, description, active, version, created_at)
            VALUES (?, 'custom', ?, 1, '1.0', datetime('now'));
            """,
            (label, f"Intent from {REGISTRY_FILE}"),
        )
    label_id = row["id"] if row else label
    conn.execute(
        """
        INSERT OR IGNORE INTO object_labels (
            id, object_type, object_id, label_id, source, confidence, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, datetime('now'));
        """,
        (generate_id("lbl"), object_type, object_id, label_id, source, confidence),
    )


def accept_intent(
    conn: sqlite3.Connection,
    *,
    object_type: str,
    object_id: str,
    intent: str,
) -> None:
    """Promote a suggested intent to an active, human-owned one."""
    add_intent(
        conn,
        object_type,
        object_id,
        intent,
        source="human",
        actor="human",
        is_active=True,
    )


RANKING_TABLES = ("embeddings", "search_documents")


def assert_ranking_is_unaffected(conn: sqlite3.Connection) -> None:
    """Fail loudly if intent storage has touched anything ranking reads.

    Intents are facets. If they ever start influencing retrieval order, that is
    a design violation, and it should break the build rather than ship quietly.
    """
    offending = [
        row["sql"]
        for row in conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND sql IS NOT NULL;"
        )
        if any(table in (row["sql"] or "") for table in RANKING_TABLES)
        and "intent" in (row["sql"] or "").lower()
    ]
    if offending:
        raise AssertionError(
            "Intent storage must not be wired into ranking tables; found: " + "; ".join(offending)
        )


def intent_filter_clause(aliases: list[str]) -> tuple[str, list[str]]:
    """Build a WHERE fragment for filtering by intent.

    Filtering is the whole point of an intent: it narrows a result set the
    ranker already produced. It must never be composed into a scoring
    expression, so only an IN-clause is offered here.
    """
    if not aliases:
        return "", []
    placeholders = ", ".join("?" for _ in aliases)
    return f" AND i.intent IN ({placeholders})", list(aliases)
