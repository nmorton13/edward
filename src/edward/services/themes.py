"""Themes: named groups of captures, and a time-ordered view of recent saves.

A theme is a group of captures that sit close together in embedding space. Groups
are found with spherical k-means over one vector per capture (the mean of the
capture's own embedding and its linked resources' embeddings), so they need no
model and no network.

Stability matters more than optimality: people remember where things are. New
captures are assigned to the nearest existing theme without moving anything else,
and a full rebuild matches new groups to old ones by shared members so that
theme ids and names survive.

Names come from a small, opt-in model reading the titles and summaries of a
theme's most central public members. Private captures (Gmail, personal notes,
local documents) are never sent. Without a model, a theme gets a fallback name
from its members' topic labels. Human-assigned names are never overwritten.
"""

from __future__ import annotations

import datetime
import hashlib
import math
import sqlite3
from collections import Counter
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from edward.db import is_test_namespace
from edward.models import generate_id
from edward.services.embed import get_configured_embedding_model, serialize_vector
from edward.services.privacy import classify_content_data_class

MIN_THEMES = 4
MAX_THEMES = 40
# A new group keeps an old theme's id and name when at least this share of its
# members came from that theme.
MATCH_OVERLAP = 0.5
NAME_SAMPLE_SIZE = 10
MIN_PUBLIC_FOR_NAMING = 3
# Captures from an import run larger than this were bulk backfills: their save
# date is the import date, so the timeline uses the post's own date instead.
BACKFILL_RUN_SIZE = 50

_KMEANS_SEED = 7
_KMEANS_RESTARTS = 8
_KMEANS_ITERATIONS = 100


class ThemeError(Exception):
    """Raised for invalid theme operations (unknown id, empty name)."""


class ThemeNamePayload(BaseModel):
    """Structured model response for naming a theme."""

    name: str = Field(..., min_length=2, max_length=60)
    description: str = Field(default="", max_length=300)


def _now_iso() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat()


def _vector(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


# --- Capture vectors -------------------------------------------------------------


def load_capture_vectors(
    conn: sqlite3.Connection, model: str | None = None
) -> tuple[list[str], np.ndarray]:
    """One unit vector per live, non-test capture that has any embedding.

    The vector is the mean of the capture's own embedding and those of its linked
    resources, so an empty-bodied bookmark is placed by the page it points at.
    """
    model = model or get_configured_embedding_model()
    captures = [
        r["id"]
        for r in conn.execute(
            "SELECT id, origin_namespace FROM captures WHERE is_deleted = 0 ORDER BY id;"
        )
        if not is_test_namespace(r["origin_namespace"])
    ]
    vectors: dict[tuple[str, str], np.ndarray] = {
        (r["object_type"], r["object_id"]): _vector(r["embedding_blob"])
        for r in conn.execute(
            "SELECT object_type, object_id, embedding_blob FROM embeddings "
            "WHERE model = ? AND object_type IN ('capture', 'resource');",
            (model,),
        )
    }
    links: dict[str, list[str]] = {}
    for r in conn.execute(
        """
        SELECT cr.capture_id, cr.resource_id FROM capture_resources cr
        JOIN resources r ON r.id = cr.resource_id AND r.is_deleted = 0;
        """
    ):
        links.setdefault(r["capture_id"], []).append(r["resource_id"])

    ids: list[str] = []
    rows: list[np.ndarray] = []
    for cid in captures:
        parts = [vectors[("capture", cid)]] if ("capture", cid) in vectors else []
        parts += [
            vectors[("resource", rid)] for rid in links.get(cid, []) if ("resource", rid) in vectors
        ]
        if not parts:
            continue
        mean = np.mean(parts, axis=0)
        norm = float(np.linalg.norm(mean))
        if norm == 0.0:
            continue
        ids.append(cid)
        rows.append(mean / norm)
    matrix = np.array(rows, dtype=np.float32) if rows else np.zeros((0, 0), dtype=np.float32)
    return ids, matrix


# --- Grouping --------------------------------------------------------------------


def default_theme_count(n_captures: int) -> int:
    """Roughly sqrt(n/2) themes: ~14 for 400 captures, ~32 for 2,000."""
    if n_captures <= 0:
        return 0
    k = round(math.sqrt(n_captures / 2))
    return max(1, min(n_captures, max(MIN_THEMES, min(MAX_THEMES, k))))


def _kmeans(matrix: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic spherical k-means with k-means++ seeding; best of several restarts."""
    rng = np.random.default_rng(_KMEANS_SEED)
    n = len(matrix)
    best: tuple[float, np.ndarray, np.ndarray] | None = None
    for _ in range(_KMEANS_RESTARTS):
        centroids = [matrix[rng.integers(n)]]
        for _ in range(1, k):
            dist = 1.0 - np.max(matrix @ np.array(centroids).T, axis=1)
            dist = np.clip(dist, 0.0, None) ** 2
            total = dist.sum()
            idx = rng.choice(n, p=dist / total) if total > 0 else rng.integers(n)
            centroids.append(matrix[idx])
        c = np.array(centroids)
        labels = np.zeros(n, dtype=int)
        for _ in range(_KMEANS_ITERATIONS):
            labels = (matrix @ c.T).argmax(axis=1)
            new_c = np.array(
                [
                    matrix[labels == j].mean(axis=0) if (labels == j).any() else c[j]
                    for j in range(k)
                ]
            )
            new_c /= np.linalg.norm(new_c, axis=1, keepdims=True)
            if np.allclose(new_c, c):
                break
            c = new_c
        score = float((matrix * c[labels]).sum())
        if best is None or score > best[0]:
            best = (score, labels, c)
    assert best is not None
    return best[1], best[2]


def _theme_rows(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute("SELECT * FROM themes ORDER BY created_at, id;").fetchall()


def _refresh_member_counts(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        UPDATE themes SET member_count = (
            SELECT COUNT(*) FROM theme_members tm WHERE tm.theme_id = themes.id
        );
        """
    )


def rebuild_themes(conn: sqlite3.Connection, k: int | None = None) -> dict[str, Any]:
    """Regroup every capture, keeping the ids and names of themes that survive.

    A new group inherits an old theme when most of its members came from it.
    Old themes with no successor are removed, except human-named ones, which are
    kept (empty) so the name is not lost.
    """
    model = get_configured_embedding_model()
    ids, matrix = load_capture_vectors(conn, model)
    now = _now_iso()
    if not ids:
        return {"themes": 0, "created": 0, "kept": 0, "removed": 0, "captures": 0}

    k = min(k or default_theme_count(len(ids)), len(ids))
    labels, centroids = _kmeans(matrix, k)

    old_members: dict[str, set[str]] = {}
    for r in conn.execute("SELECT theme_id, capture_id FROM theme_members;"):
        old_members.setdefault(r["theme_id"], set()).add(r["capture_id"])
    old_themes = {r["id"]: r for r in _theme_rows(conn)}

    groups = [{ids[i] for i in np.where(labels == j)[0]} for j in range(k)]
    pairs = sorted(
        (
            (len(group & members), j, theme_id)
            for j, group in enumerate(groups)
            for theme_id, members in old_members.items()
            if group & members
        ),
        reverse=True,
    )
    group_to_theme: dict[int, str] = {}
    used: set[str] = set()
    for overlap, j, theme_id in pairs:
        if j in group_to_theme or theme_id in used:
            continue
        if overlap >= MATCH_OVERLAP * len(groups[j]):
            group_to_theme[j] = theme_id
            used.add(theme_id)

    conn.execute("DELETE FROM theme_members;")
    created = 0
    for j in range(k):
        if not groups[j]:
            continue
        blob = serialize_vector(centroids[j].tolist())
        theme_id = group_to_theme.get(j)
        if theme_id:
            conn.execute(
                "UPDATE themes SET centroid_blob = ?, embedding_model = ?, updated_at = ? WHERE id = ?;",
                (blob, model, now, theme_id),
            )
        else:
            theme_id = generate_id("thm")
            created += 1
            conn.execute(
                """
                INSERT INTO themes (id, name, description, name_source, embedding_model,
                                    centroid_blob, member_count, created_at, updated_at)
                VALUES (?, NULL, NULL, NULL, ?, ?, 0, ?, ?);
                """,
                (theme_id, model, blob, now, now),
            )
        for i in np.where(labels == j)[0]:
            conn.execute(
                "INSERT INTO theme_members (capture_id, theme_id, similarity, assigned_at) VALUES (?, ?, ?, ?);",
                (ids[i], theme_id, float(matrix[i] @ centroids[j]), now),
            )

    removed = 0
    for theme_id, row in old_themes.items():
        if theme_id in used:
            continue
        if row["name_source"] == "human":
            continue
        conn.execute("DELETE FROM themes WHERE id = ?;", (theme_id,))
        removed += 1

    _refresh_member_counts(conn)
    return {
        "themes": k,
        "created": created,
        "kept": len(used),
        "removed": removed,
        "captures": len(ids),
    }


def assign_new_captures(conn: sqlite3.Connection) -> int:
    """Put captures that have no theme yet into their nearest theme. Nothing else moves."""
    model = get_configured_embedding_model()
    themes = conn.execute(
        "SELECT id, centroid_blob FROM themes WHERE embedding_model = ? AND member_count > 0;",
        (model,),
    ).fetchall()
    if not themes:
        return 0
    assigned = {r["capture_id"] for r in conn.execute("SELECT capture_id FROM theme_members;")}
    ids, matrix = load_capture_vectors(conn, model)
    todo = [i for i, cid in enumerate(ids) if cid not in assigned]
    if not todo:
        return 0
    centroids = np.array([_vector(t["centroid_blob"]) for t in themes])
    now = _now_iso()
    sims = matrix[todo] @ centroids.T
    for row, i in enumerate(todo):
        best = int(sims[row].argmax())
        conn.execute(
            "INSERT INTO theme_members (capture_id, theme_id, similarity, assigned_at) VALUES (?, ?, ?, ?);",
            (ids[i], themes[best]["id"], float(sims[row, best]), now),
        )
    _refresh_member_counts(conn)
    return len(todo)


def refresh_themes(
    conn: sqlite3.Connection, rebuild: bool = False, k: int | None = None
) -> dict[str, Any]:
    """Build themes the first time (or on request); otherwise just place new captures."""
    has_themes = conn.execute("SELECT 1 FROM themes LIMIT 1;").fetchone() is not None
    if rebuild or not has_themes:
        result = rebuild_themes(conn, k=k)
        result["mode"] = "rebuild"
        return result
    return {"mode": "assign", "assigned": assign_new_captures(conn)}


# --- Capture details -------------------------------------------------------------


def capture_title(conn: sqlite3.Connection, capture_id: str) -> str:
    """The label a person would recognise: the saved item's own title, else the note or text."""
    row = conn.execute(
        """
        SELECT c.user_note, c.raw_content,
               (SELECT r.title FROM capture_resources cr
                JOIN resources r ON r.id = cr.resource_id AND r.is_deleted = 0
                WHERE cr.capture_id = c.id AND coalesce(r.title, '') != ''
                ORDER BY cr.created_at, r.id LIMIT 1) AS resource_title
        FROM captures c WHERE c.id = ?;
        """,
        (capture_id,),
    ).fetchone()
    if not row:
        return "Untitled capture"
    if row["resource_title"]:
        return str(row["resource_title"])
    if row["user_note"]:
        return str(row["user_note"]).strip().splitlines()[0][:120]
    if row["raw_content"]:
        return str(row["raw_content"]).strip().splitlines()[0][:120]
    return "Untitled capture"


def _capture_data_class(conn: sqlite3.Connection, capture_id: str) -> str:
    row = conn.execute(
        """
        SELECT c.origin_namespace,
               (SELECT r.canonical_url FROM capture_resources cr
                JOIN resources r ON r.id = cr.resource_id
                WHERE cr.capture_id = c.id ORDER BY cr.created_at LIMIT 1) AS url
        FROM captures c WHERE c.id = ?;
        """,
        (capture_id,),
    ).fetchone()
    if not row:
        return "personal_notes"
    return classify_content_data_class(
        origin_namespace=row["origin_namespace"] or "manual", canonical_url=row["url"]
    )


def _capture_summary(conn: sqlite3.Connection, capture_id: str) -> str:
    row = conn.execute(
        """
        SELECT rc.summary FROM capture_resources cr
        JOIN resource_contents rc ON rc.resource_id = cr.resource_id
        WHERE cr.capture_id = ? AND length(trim(rc.summary)) > 0
        ORDER BY length(rc.clean_text) DESC LIMIT 1;
        """,
        (capture_id,),
    ).fetchone()
    return (row["summary"] or "").strip() if row else ""


# --- Naming ----------------------------------------------------------------------


def _members_hash(capture_ids: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(capture_ids)).encode("utf-8")).hexdigest()


def _label_name(label_id: str) -> str:
    leaf = label_id.split("/")[-1].replace("-", " ")
    return leaf.upper() if len(leaf) <= 3 else leaf.capitalize()


def fallback_theme_name(conn: sqlite3.Connection, capture_ids: list[str]) -> str:
    """A model-free name from the topic labels most common among a theme's members."""
    if not capture_ids:
        return "Unnamed theme"
    placeholders = ",".join("?" * len(capture_ids))
    counts = Counter(
        r["label_id"]
        for r in conn.execute(
            f"""
            SELECT DISTINCT cr.capture_id, ol.label_id
            FROM capture_resources cr
            JOIN object_labels ol ON ol.object_type = 'resource' AND ol.object_id = cr.resource_id
            JOIN labels l ON l.id = ol.label_id AND l.family = 'topic'
            WHERE cr.capture_id IN ({placeholders});
            """,
            capture_ids,
        )
    )
    names: list[str] = []
    for label_id, _ in counts.most_common():
        name = _label_name(label_id)
        if name not in names:
            names.append(name)
        if len(names) == 2:
            break
    return " · ".join(names) if names else "Unnamed theme"


def collect_naming_requests(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """Themes that still need a name, with their most central members' titles and summaries.

    Only public members are included in ``samples``; private ones are counted but never
    read into the request.
    """
    requests: list[dict[str, Any]] = []
    taken = [
        r["name"]
        for r in conn.execute(
            "SELECT name FROM themes WHERE name IS NOT NULL AND name_source != 'fallback';"
        )
    ]
    for theme in conn.execute(
        "SELECT id FROM themes WHERE member_count > 0 AND (name_source IS NULL OR name_source = 'fallback') ORDER BY member_count DESC;"
    ).fetchall():
        members = [
            r["capture_id"]
            for r in conn.execute(
                "SELECT capture_id FROM theme_members WHERE theme_id = ? ORDER BY similarity DESC;",
                (theme["id"],),
            )
        ]
        samples: list[dict[str, str]] = []
        private = 0
        for cid in members:
            if len(samples) >= NAME_SAMPLE_SIZE:
                break
            if _capture_data_class(conn, cid) != "public_web":
                private += 1
                continue
            samples.append(
                {"title": capture_title(conn, cid), "summary": _capture_summary(conn, cid)}
            )
        requests.append(
            {
                "theme_id": theme["id"],
                "member_ids": members,
                "samples": samples,
                "skipped_private": private,
                "fallback_name": fallback_theme_name(conn, members),
                "taken_names": taken,
            }
        )
    return requests


def generate_theme_name(request: dict[str, Any], client: Any) -> ThemeNamePayload:
    """Ask the model for a short, specific name. Public samples only; privacy checked before dispatch."""
    lines = []
    for n, sample in enumerate(request["samples"], 1):
        summary = f" — {sample['summary'][:300]}" if sample["summary"] else ""
        lines.append(f"{n}. {sample['title'][:150]}{summary}")
    taken = ", ".join(request["taken_names"]) or "none"
    system_prompt = (
        "You name groups of saved research items for a personal library. "
        "Reply with a short, specific name (2-5 words, Title Case, no quotes, no emoji) that says "
        "what the items are about, and a one-sentence description. Prefer concrete subjects "
        "over generic words like 'Technology', 'Various' or 'Insights'. "
        'Output strictly valid JSON: {"name": "...", "description": "..."}'
    )
    user_prompt = (
        f"Items in this group:\n{chr(10).join(lines)}\n\n"
        f"Names already used for other groups (pick a different one): {taken}"
    )
    _, parsed = client.chat_completion(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_model=ThemeNamePayload,
        data_class="public_web",
        temperature=0.2,
    )
    if parsed is None or not parsed.name.strip():
        raise ValueError("Theme naming model returned no name")
    return ThemeNamePayload(
        name=parsed.name.strip().strip("\"'"), description=parsed.description.strip()
    )


def store_theme_name(
    conn: sqlite3.Connection,
    theme_id: str,
    name: str,
    description: str | None,
    source: str,
    member_ids: list[str],
) -> bool:
    """Write an automated name unless a human has named the theme in the meantime."""
    cursor = conn.execute(
        """
        UPDATE themes
        SET name = ?, description = ?, name_source = ?, named_members_hash = ?, updated_at = ?
        WHERE id = ? AND coalesce(name_source, '') != 'human';
        """,
        (name, description, source, _members_hash(member_ids), _now_iso(), theme_id),
    )
    return cursor.rowcount > 0


def rename_theme(conn: sqlite3.Connection, theme_id: str, name: str) -> dict[str, Any]:
    """Human rename: final, never overwritten by rebuilds or model naming."""
    name = name.strip()
    if not name:
        raise ThemeError("Theme name cannot be empty")
    cursor = conn.execute(
        "UPDATE themes SET name = ?, name_source = 'human', updated_at = ? WHERE id = ?;",
        (name, _now_iso(), theme_id),
    )
    if cursor.rowcount == 0:
        raise ThemeError(f"Theme '{theme_id}' not found")
    return {"theme_id": theme_id, "name": name, "name_source": "human"}


# --- Listing ---------------------------------------------------------------------


def list_themes(conn: sqlite3.Connection, examples: int = 3) -> list[dict[str, Any]]:
    themes = []
    for t in conn.execute(
        "SELECT id, name, description, name_source, member_count FROM themes "
        "WHERE member_count > 0 ORDER BY member_count DESC, id;"
    ):
        sample = [
            {"capture_id": r["capture_id"], "title": capture_title(conn, r["capture_id"])}
            for r in conn.execute(
                "SELECT capture_id FROM theme_members WHERE theme_id = ? ORDER BY similarity DESC LIMIT ?;",
                (t["id"], examples),
            )
        ]
        themes.append(
            {
                "theme_id": t["id"],
                "name": t["name"] or "Unnamed theme",
                "description": t["description"],
                "name_source": t["name_source"],
                "count": t["member_count"],
                "examples": sample,
            }
        )
    return themes


def theme_members(conn: sqlite3.Connection, theme_id: str, limit: int = 50) -> dict[str, Any]:
    theme = conn.execute("SELECT * FROM themes WHERE id = ?;", (theme_id,)).fetchone()
    if not theme:
        raise ThemeError(f"Theme '{theme_id}' not found")
    members = [
        {
            "capture_id": r["capture_id"],
            "title": capture_title(conn, r["capture_id"]),
            "similarity": round(r["similarity"], 4),
        }
        for r in conn.execute(
            "SELECT capture_id, similarity FROM theme_members WHERE theme_id = ? ORDER BY similarity DESC LIMIT ?;",
            (theme_id, limit),
        )
    ]
    return {
        "theme_id": theme_id,
        "name": theme["name"] or "Unnamed theme",
        "description": theme["description"],
        "name_source": theme["name_source"],
        "count": theme["member_count"],
        "members": members,
    }


# --- Recent: what you saved, by when ----------------------------------------------


def parse_since(
    text: str, now: datetime.datetime | None = None
) -> tuple[datetime.datetime, datetime.datetime | None]:
    """Turn 'today', 'yesterday', 'week', 'last-week', '7d' or 'YYYY-MM-DD' into a local window.

    Returns (start, end); end is None for "until now".
    """
    now = now or datetime.datetime.now().astimezone()
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)
    key = text.strip().lower().replace(" ", "-")
    if key == "today":
        return today, None
    if key == "yesterday":
        return today - datetime.timedelta(days=1), today
    if key in ("week", "this-week"):
        return today - datetime.timedelta(days=today.weekday()), None
    if key == "last-week":
        this_monday = today - datetime.timedelta(days=today.weekday())
        return this_monday - datetime.timedelta(days=7), this_monday
    if key.endswith("d") and key[:-1].isdigit():
        return today - datetime.timedelta(days=int(key[:-1]) - 1), None
    try:
        day = datetime.date.fromisoformat(key)
    except ValueError as exc:
        raise ValueError(
            f"Unrecognised --since '{text}'. Use today, yesterday, week, last-week, Nd (e.g. 7d) or YYYY-MM-DD."
        ) from exc
    return datetime.datetime.combine(day, datetime.time.min, tzinfo=now.tzinfo), None


def _parse_ts(value: str | None) -> datetime.datetime | None:
    if not value:
        return None
    try:
        ts = datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return ts if ts.tzinfo else ts.replace(tzinfo=datetime.UTC)


def capture_dates(conn: sqlite3.Connection) -> dict[str, tuple[datetime.datetime, str]]:
    """When each capture entered your life: the save date, or for bulk backfills the post date."""
    rows = conn.execute(
        """
        SELECT c.id, c.origin_namespace, c.created_at,
               (SELECT COUNT(*) FROM captures c2 WHERE c2.collector_run_id = c.collector_run_id) AS run_size,
               (SELECT MIN(r.published_at) FROM capture_resources cr
                JOIN resources r ON r.id = cr.resource_id
                WHERE cr.capture_id = c.id AND r.published_at IS NOT NULL) AS posted_at
        FROM captures c
        WHERE c.is_deleted = 0;
        """
    ).fetchall()
    dates: dict[str, tuple[datetime.datetime, str]] = {}
    for r in rows:
        if is_test_namespace(r["origin_namespace"]):
            continue
        saved = _parse_ts(r["created_at"])
        posted = _parse_ts(r["posted_at"])
        if r["run_size"] and r["run_size"] > BACKFILL_RUN_SIZE and posted:
            dates[r["id"]] = (posted, "posted")
        elif saved:
            dates[r["id"]] = (saved, "saved")
    return dates


def _day_label(day: datetime.date, today: datetime.date) -> str:
    delta = (today - day).days
    if delta == 0:
        return "Today"
    if delta == 1:
        return "Yesterday"
    if 1 < delta < 7:
        return day.strftime("%A")
    return day.strftime("%a %b %-d")


def recent_digest(
    conn: sqlite3.Connection,
    since: datetime.datetime,
    until: datetime.datetime | None = None,
    now: datetime.datetime | None = None,
) -> dict[str, Any]:
    """Captures in a time window, grouped by day and summarised by theme.

    Captures that have no stored theme yet are placed in their nearest theme for
    display only; nothing is written.
    """
    now = now or datetime.datetime.now().astimezone()
    local = now.tzinfo
    in_window = [
        (cid, ts.astimezone(local), basis)
        for cid, (ts, basis) in capture_dates(conn).items()
        if ts >= since and (until is None or ts < until)
    ]
    in_window.sort(key=lambda item: item[1], reverse=True)

    names = {
        r["id"]: r["name"] or "Unnamed theme" for r in conn.execute("SELECT id, name FROM themes;")
    }
    membership = {
        r["capture_id"]: r["theme_id"]
        for r in conn.execute("SELECT capture_id, theme_id FROM theme_members;")
    }
    missing = [cid for cid, _, _ in in_window if cid not in membership]
    if missing:
        themes = conn.execute(
            "SELECT id, centroid_blob FROM themes WHERE member_count > 0 AND embedding_model = ?;",
            (get_configured_embedding_model(),),
        ).fetchall()
        if themes:
            ids, matrix = load_capture_vectors(conn)
            index = {cid: i for i, cid in enumerate(ids)}
            centroids = np.array([_vector(t["centroid_blob"]) for t in themes])
            for cid in missing:
                if cid in index:
                    membership[cid] = themes[int((matrix[index[cid]] @ centroids.T).argmax())]["id"]

    days: dict[datetime.date, list[dict[str, Any]]] = {}
    theme_counts: Counter[str | None] = Counter()
    for cid, ts, basis in in_window:
        theme_id = membership.get(cid)
        theme_counts[theme_id] += 1
        days.setdefault(ts.date(), []).append(
            {
                "capture_id": cid,
                "title": capture_title(conn, cid),
                "theme_id": theme_id,
                "theme_name": names.get(theme_id, "Unsorted") if theme_id else "Unsorted",
                "date": ts.isoformat(),
                "date_basis": basis,
            }
        )

    return {
        "since": since.isoformat(),
        "until": until.isoformat() if until else None,
        "total": len(in_window),
        "themes": [
            {
                "theme_id": theme_id,
                "name": names.get(theme_id, "Unsorted") if theme_id else "Unsorted",
                "count": count,
            }
            for theme_id, count in theme_counts.most_common()
        ],
        "days": [
            {
                "date": day.isoformat(),
                "label": _day_label(day, now.date()),
                "count": len(items),
                "items": items,
            }
            for day, items in sorted(days.items(), reverse=True)
        ],
    }
