"""Suggest project evidence by meaning, judged against the project's question.

Two stages, the shape tested on the real library before this was built:

1. Find: rank every capture by embedding similarity to the project's profile, which
   is the brief (as a query) blended with the centroid of evidence already accepted.
   Keyword search on the brief found almost none of the accepted evidence; this
   finds most of it in the top 25-50.
2. Judge: ask Jev one yes/no question per top-ranked candidate, "is this relevant
   evidence for <brief>?". Embeddings return everything on the subject; the judge
   keeps what bears on the project's actual question.

Each capture is checked against a project once per brief and the result recorded
in ``project_matches``, so after the first pass only new captures are looked at.
Suggestions are added as ``candidate`` evidence with ``added_by='system'``; human
decisions (accepted, rejected) are never changed and rejected items never return.

Only public content is sent to the hosted judge. Private captures are ranked
locally and suggested only when they are among the very closest matches.
"""

from __future__ import annotations

import datetime
import hashlib
import logging
import sqlite3
from collections.abc import Callable
from typing import Any

import numpy as np

from edward.db import is_test_namespace
from edward.services.embed import generate_embedding, get_configured_embedding_model
from edward.services.privacy import classify_content_data_class
from edward.services.projects import add_project_object

logger = logging.getLogger(__name__)

SUGGEST_RANK = 40
UNJUDGED_RANK = 15
# Tuned on the real library: real evidence judged >= 0.6; the 0.50-0.55 band was an
# "about us" page and a generic take.
RELEVANCE_THRESHOLD = 0.6
MIN_ACCEPTED_FOR_CENTROID = 3
JUDGE_TEXT_CHARS = 8000
QUESTION_ID = "project-relevance"

Embedder = Callable[[str], list[float]]


def _now_iso() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat()


def _unit(v: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    return v / norm if norm else v


def project_brief(project: sqlite3.Row | dict[str, Any]) -> str:
    return f"{project['title']}. {project['description'] or ''}".strip()


def brief_hash(brief: str) -> str:
    return hashlib.sha256(brief.encode("utf-8")).hexdigest()[:16]


def _default_embedder(text: str) -> list[float]:
    return generate_embedding(text, is_query=True)[0]


# --- Finding ---------------------------------------------------------------------


def _resource_vectors(conn: sqlite3.Connection) -> dict[str, np.ndarray]:
    return {
        r["object_id"]: _unit(np.frombuffer(r["embedding_blob"], dtype=np.float32))
        for r in conn.execute(
            "SELECT object_id, embedding_blob FROM embeddings WHERE object_type = 'resource' AND model = ?;",
            (get_configured_embedding_model(),),
        )
    }


def _capture_resources(conn: sqlite3.Connection) -> dict[str, list[str]]:
    links: dict[str, list[str]] = {}
    for r in conn.execute(
        """
        SELECT c.id AS capture_id, c.origin_namespace, cr.resource_id
        FROM captures c
        JOIN capture_resources cr ON cr.capture_id = c.id
        JOIN resources r ON r.id = cr.resource_id AND r.is_deleted = 0
        WHERE c.is_deleted = 0;
        """
    ):
        if not is_test_namespace(r["origin_namespace"]):
            links.setdefault(r["capture_id"], []).append(r["resource_id"])
    return links


def member_captures(conn: sqlite3.Connection, project_id: str) -> set[str]:
    """Captures already in the project in any state, including rejected."""
    return {
        r["capture_id"]
        for r in conn.execute(
            """
            SELECT po.object_id AS capture_id FROM project_objects po
            WHERE po.project_id = ? AND po.object_type = 'capture'
            UNION
            SELECT cr.capture_id FROM project_objects po
            JOIN capture_resources cr ON cr.resource_id = po.object_id
            WHERE po.project_id = ? AND po.object_type = 'resource'
            UNION
            SELECT cr.capture_id FROM project_objects po
            JOIN findings f ON f.id = po.object_id
            JOIN capture_resources cr ON cr.resource_id = f.resource_id
            WHERE po.project_id = ? AND po.object_type = 'finding';
            """,
            (project_id, project_id, project_id),
        )
    }


def project_profile(
    conn: sqlite3.Connection,
    project_id: str,
    brief: str,
    vectors: dict[str, np.ndarray],
    embedder: Embedder,
) -> np.ndarray:
    """The brief as a query, blended half-and-half with accepted evidence once there is enough."""
    profile = _unit(np.asarray(embedder(brief), dtype=np.float32))
    accepted = [
        r["resource_id"]
        for r in conn.execute(
            """
            SELECT po.object_id AS resource_id FROM project_objects po
            WHERE po.project_id = ? AND po.membership_status = 'accepted' AND po.object_type = 'resource'
            UNION
            SELECT cr.resource_id FROM project_objects po
            JOIN capture_resources cr ON cr.capture_id = po.object_id
            WHERE po.project_id = ? AND po.membership_status = 'accepted' AND po.object_type = 'capture';
            """,
            (project_id, project_id),
        )
        if r["resource_id"] in vectors
    ]
    if len(accepted) >= MIN_ACCEPTED_FOR_CENTROID:
        centroid = _unit(np.mean([vectors[r] for r in accepted], axis=0))
        profile = _unit(0.5 * profile + 0.5 * centroid)
    return profile


def rank_captures(
    links: dict[str, list[str]], vectors: dict[str, np.ndarray], profile: np.ndarray
) -> list[tuple[str, str, float]]:
    """(capture_id, best resource_id, similarity), best first. A capture scores as its best page."""
    scored = []
    for capture_id, resource_ids in links.items():
        best = max(
            ((float(vectors[r] @ profile), r) for r in resource_ids if r in vectors),
            default=None,
        )
        if best is not None:
            scored.append((capture_id, best[1], best[0]))
    scored.sort(key=lambda item: (-item[2], item[0]))
    return scored


# --- Judging ---------------------------------------------------------------------


def relevance_question(brief: str) -> dict[str, Any]:
    return {
        "id": QUESTION_ID,
        "primitive": "noul",
        "prompt": f"Is this content relevant evidence for this research project: {brief}",
        "instructions": (
            "Answer Yes if it contains facts, data, arguments, examples or claims that someone "
            "writing this project would cite, verify, or need to answer. Answer No if it only "
            "shares the general subject without bearing on the project's question."
        ),
    }


def get_judge() -> Any | None:
    """The configured Jev transport, or None when no hosted judge is configured."""
    from edward.services.classification import get_classifier

    classifier = get_classifier()
    transport = getattr(classifier, "transport", None)
    return transport if hasattr(transport, "evaluate_questions") else None


def _judge_input(
    conn: sqlite3.Connection, capture_id: str, resource_id: str
) -> tuple[str, str, str]:
    row = conn.execute(
        """
        SELECT r.title, r.canonical_url, c.origin_namespace,
               (SELECT clean_text FROM resource_contents rc WHERE rc.resource_id = r.id
                ORDER BY rc.created_at DESC LIMIT 1) AS clean_text
        FROM resources r, captures c WHERE r.id = ? AND c.id = ?;
        """,
        (resource_id, capture_id),
    ).fetchone()
    data_class = classify_content_data_class(
        origin_namespace=row["origin_namespace"] or "manual", canonical_url=row["canonical_url"]
    )
    text = f"{row['title'] or ''}\n\n{(row['clean_text'] or '')[:JUDGE_TEXT_CHARS]}".strip()
    return row["title"] or "", text, data_class


# --- Matching --------------------------------------------------------------------


def _plan(
    conn: sqlite3.Connection,
    project: sqlite3.Row,
    links: dict[str, list[str]],
    vectors: dict[str, np.ndarray],
    embedder: Embedder,
    limit: int,
    full: bool,
) -> dict[str, Any]:
    """Decide, read-only, which captures to judge and how to record the rest."""
    brief = project_brief(project)
    bhash = brief_hash(brief)
    evaluated = {
        r["capture_id"]
        for r in conn.execute(
            "SELECT capture_id FROM project_matches WHERE project_id = ? AND brief_hash = ?;",
            (project["id"], bhash),
        )
    }
    first_pass = full or not evaluated
    ranked = rank_captures(
        links, vectors, project_profile(conn, project["id"], brief, vectors, embedder)
    )
    members = member_captures(conn, project["id"])
    member_objects = {
        r["object_id"]
        for r in conn.execute(
            "SELECT object_id FROM project_objects WHERE project_id = ?;", (project["id"],)
        )
    }
    seen_objects: set[str] = set()

    records: list[dict[str, Any]] = []
    to_judge: list[dict[str, Any]] = []
    non_member_rank = 0
    for position, (capture_id, resource_id, similarity) in enumerate(ranked, 1):
        # Several captures of the same page share one resource: suggest it once.
        is_member = (
            capture_id in members or resource_id in member_objects or resource_id in seen_objects
        )
        seen_objects.add(resource_id)
        if not is_member:
            non_member_rank += 1
        if not first_pass and capture_id in evaluated:
            continue
        # Rank among captures not yet in the project: the list a suggestion competes in.
        record = {
            "capture_id": capture_id,
            "object_id": resource_id,
            "similarity": similarity,
            "rank": position if is_member else non_member_rank,
        }
        cutoff = limit if first_pass else SUGGEST_RANK
        if is_member:
            records.append({**record, "outcome": "member"})
        elif non_member_rank <= cutoff:
            title, text, data_class = _judge_input(conn, capture_id, resource_id)
            to_judge.append({**record, "title": title, "text": text, "data_class": data_class})
        else:
            records.append({**record, "outcome": "below-rank"})
    return {
        "project": project,
        "brief": brief,
        "brief_hash": bhash,
        "first_pass": first_pass,
        "records": records,
        "to_judge": to_judge,
    }


def _judge_all(plan: dict[str, Any], judge: Any | None) -> dict[str, Any]:
    """Network phase: no database access. Stops judging at the first provider error."""
    question = relevance_question(plan["brief"])
    stats = {"judged": 0, "cost": 0.0, "error": None}
    for item in plan["to_judge"]:
        item["relevance"] = None
        if judge is None or item["data_class"] != "public_web":
            continue
        if stats["error"]:
            item["deferred"] = True
            continue
        try:
            result = judge.evaluate_questions(
                text=item["text"], questions=[question], data_class=item["data_class"]
            )
        except Exception as e:  # rate limits, outages: leave unjudged items for the next pass
            stats["error"] = str(e)
            item["deferred"] = True
            continue
        answer = result["answers"].get(QUESTION_ID) or {}
        value = answer.get("noul", answer.get("probability"))
        item["relevance"] = float(value) if value is not None else 0.0
        stats["judged"] += 1
        stats["cost"] += float(result.get("cost") or 0.0)
    return stats


def _decide(item: dict[str, Any]) -> str:
    if item["relevance"] is not None:
        return "suggested" if item["relevance"] >= RELEVANCE_THRESHOLD else "not-relevant"
    # Not judged (private content, or no judge configured): only the closest matches.
    return "suggested" if item["rank"] <= UNJUDGED_RANK else "below-rank"


def _note(item: dict[str, Any]) -> str:
    if item["relevance"] is not None:
        return (
            f"Suggested by meaning (rank {item['rank']}, similarity {item['similarity']:.2f}); "
            f"judged relevant to the project question ({item['relevance']:.2f})."
        )
    reason = (
        "private content stays local"
        if item["data_class"] != "public_web"
        else "no judge configured"
    )
    return f"Suggested by meaning (rank {item['rank']}, similarity {item['similarity']:.2f}); not judged ({reason})."


def _store(conn: sqlite3.Connection, plan: dict[str, Any]) -> list[dict[str, Any]]:
    project_id = plan["project"]["id"]
    now = _now_iso()
    if plan["first_pass"]:
        conn.execute("DELETE FROM project_matches WHERE project_id = ?;", (project_id,))
    suggested = []
    rows = list(plan["records"])
    for item in plan["to_judge"]:
        if item.get("deferred"):
            continue
        outcome = _decide(item)
        rows.append({**item, "outcome": outcome})
        if outcome == "suggested":
            add_project_object(
                conn,
                project_id,
                item["object_id"],
                relationship="evidence",
                membership_status="candidate",
                added_by="system",
                relevance_note=_note(item),
            )
            suggested.append(
                {
                    "capture_id": item["capture_id"],
                    "object_id": item["object_id"],
                    "title": item["title"],
                    "similarity": round(item["similarity"], 4),
                    "relevance": item["relevance"],
                }
            )
    for row in rows:
        conn.execute(
            """
            INSERT INTO project_matches (project_id, capture_id, brief_hash, object_id, similarity,
                                         rank, relevance, outcome, evaluated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(project_id, capture_id) DO UPDATE SET
                brief_hash = excluded.brief_hash, object_id = excluded.object_id,
                similarity = excluded.similarity, rank = excluded.rank,
                relevance = excluded.relevance, outcome = excluded.outcome,
                evaluated_at = excluded.evaluated_at;
            """,
            (
                project_id,
                row["capture_id"],
                plan["brief_hash"],
                row["object_id"],
                row["similarity"],
                row["rank"],
                row.get("relevance"),
                row["outcome"],
                now,
            ),
        )
    return suggested


def match_projects(
    db: Any,
    project_ids: list[str] | None = None,
    *,
    full: bool = False,
    limit: int = SUGGEST_RANK,
    judge: Any | None = None,
    use_judge: bool = True,
    embedder: Embedder | None = None,
) -> dict[str, Any]:
    """Check captures against active projects and add relevant ones as candidates.

    The first pass for a project (or ``full=True``, or a changed brief) judges its top
    ``limit`` non-member captures. Later passes only look at captures not yet checked,
    and judge those that rank in the project's top ``SUGGEST_RANK``.
    """
    embedder = embedder or _default_embedder
    if use_judge and judge is None:
        judge = get_judge()
    if not use_judge:
        judge = None

    with db.connection() as conn:
        if project_ids:
            projects = [
                conn.execute(
                    "SELECT id, title, description FROM projects WHERE id = ? AND is_deleted = 0 AND status = 'active';",
                    (pid,),
                ).fetchone()
                for pid in project_ids
            ]
            missing = [pid for pid, row in zip(project_ids, projects, strict=True) if row is None]
            if missing:
                raise ValueError(f"Active project not found: {', '.join(missing)}")
        else:
            projects = conn.execute(
                "SELECT id, title, description FROM projects WHERE is_deleted = 0 AND status = 'active' ORDER BY created_at;"
            ).fetchall()
        vectors = _resource_vectors(conn)
        if not vectors or not projects:
            return {"projects": [], "judge": judge is not None}
        links = _capture_resources(conn)
        # Skip projects with nothing new before paying for a query embedding.
        plans = []
        for project in projects:
            if not full and not _has_unchecked(conn, project, links, vectors):
                continue
            plans.append(_plan(conn, project, links, vectors, embedder, limit, full))

    summary = []
    for plan in plans:
        stats = _judge_all(plan, judge)
        with db.transaction() as conn:
            suggested = _store(conn, plan)
        summary.append(
            {
                "project_id": plan["project"]["id"],
                "title": plan["project"]["title"],
                "mode": "full" if plan["first_pass"] else "new",
                "checked": len(plan["records"]) + len(plan["to_judge"]),
                "judged": stats["judged"],
                "suggested": suggested,
                "cost": round(stats["cost"], 6),
                "deferred": sum(1 for i in plan["to_judge"] if i.get("deferred")),
                "error": stats["error"],
            }
        )
    return {"projects": summary, "judge": judge is not None}


def _has_unchecked(
    conn: sqlite3.Connection,
    project: sqlite3.Row,
    links: dict[str, list[str]],
    vectors: dict[str, np.ndarray],
) -> bool:
    """True when some embedded capture has not been checked against the current brief."""
    bhash = brief_hash(project_brief(project))
    checked = {
        r["capture_id"]
        for r in conn.execute(
            "SELECT capture_id FROM project_matches WHERE project_id = ? AND brief_hash = ?;",
            (project["id"], bhash),
        )
    }
    return any(
        capture_id not in checked and any(r in vectors for r in resource_ids)
        for capture_id, resource_ids in links.items()
    )
