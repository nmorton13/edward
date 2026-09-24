"""Hybrid retrieval combining FTS5 lexical search and vector similarity with Reciprocal Rank Fusion (RRF).

Assembles bounded Evidence Packets strictly conforming to schemas/evidence-packet-v1.json.
"""

import datetime
import json
import logging
import re
import sqlite3
from typing import Any

from edward.services.embed import search_vector
from edward.services.search import search_lexical, x_capture_resource_id

logger = logging.getLogger(__name__)

QUESTION_WORDS = {
    "a",
    "about",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "bookmarked",
    "bookmark",
    "bookmarks",
    "can",
    "collected",
    "did",
    "do",
    "does",
    "for",
    "from",
    "have",
    "how",
    "i",
    "in",
    "is",
    "it",
    "me",
    "my",
    "of",
    "on",
    "or",
    "saved",
    "say",
    "the",
    "these",
    "those",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}


def research_terms(query: str) -> str:
    """Keep content words for lexical recall while vectors use the whole question."""
    terms = [
        term
        for term in re.findall(r"[\w-]+", query.lower())
        if term not in QUESTION_WORDS and len(term) > 1
    ]
    return " ".join(dict.fromkeys(terms[:8])) or query


def matching_capture_passages(text: str, query: str) -> list[dict[str, Any]]:
    """Extract exact windows around distinct query terms in a long note or PDF."""
    passages: list[dict[str, Any]] = []
    for term in research_terms(query).split():
        for match in re.finditer(rf"\b{re.escape(term)}\b", text, flags=re.IGNORECASE):
            start = max(0, match.start() - 300)
            end = min(len(text), match.end() + 500)
            if any(start < existing["end"] and end > existing["start"] for existing in passages):
                continue
            locator: dict[str, Any] = {}
            marker_start = text.rfind("[PDF page ", 0, match.start())
            if marker_start >= 0:
                marker = re.match(r"\[PDF page (\d+)\]", text[marker_start:])
                if marker:
                    locator["page"] = int(marker.group(1))
            passages.append(
                {"start": start, "end": end, "passage": text[start:end], "locator": locator}
            )
            break
        if len(passages) == 3:
            break
    return [
        {"passage": passage["passage"], "locator": passage["locator"], "content_hash": ""}
        for passage in passages
    ]


def reciprocal_rank_fusion(
    lexical_results: Any,
    vector_results: list[dict[str, Any]],
    k: int = 60,
    limit: int = 15,
) -> list[dict[str, Any]]:
    """Combine ranked results using Reciprocal Rank Fusion: RRF = sum(1 / (k + rank))."""
    scores: dict[str, float] = {}
    item_map: dict[str, dict[str, Any]] = {}

    # 1. Lexical ranking
    raw_lexical = (
        lexical_results.results if hasattr(lexical_results, "results") else lexical_results
    )
    for rank, item in enumerate(raw_lexical, start=1):
        if hasattr(item, "id"):
            obj_id = item.id
            item_dict = (
                item.model_dump()
                if hasattr(item, "model_dump")
                else {"id": item.id, "object_type": getattr(item, "object_type", "resource")}
            )
        elif isinstance(item, dict):
            obj_id = item.get("object_id") or item.get("id")
            item_dict = dict(item)
        else:
            continue

        if not obj_id:
            continue
        scores[obj_id] = scores.get(obj_id, 0.0) + (1.0 / (k + rank))
        if obj_id not in item_map:
            item_map[obj_id] = item_dict
            item_map[obj_id]["id"] = obj_id

    # 2. Vector ranking
    for rank, item in enumerate(vector_results, start=1):
        obj_id = item.get("object_id") or item.get("id")
        if not obj_id:
            continue
        scores[obj_id] = scores.get(obj_id, 0.0) + (1.0 / (k + rank))
        if obj_id not in item_map:
            item_map[obj_id] = dict(item)
            item_map[obj_id]["id"] = obj_id

    # 3. Sort by fused score
    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

    fused_results: list[dict[str, Any]] = []
    for obj_id in sorted_ids[:limit]:
        entry = item_map[obj_id]
        entry["relevance_score"] = round(scores[obj_id], 4)
        fused_results.append(entry)

    return fused_results


def build_evidence_item(
    conn: sqlite3.Connection,
    object_id: str,
    relevance_score: float | None = None,
    extra_passages: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """Load object metadata, source, passages, notes, and labels into an evidence item."""
    # 0. Check if chunk ID: map to parent resource while preserving chunk text
    if object_id.startswith("chk_"):
        c_row = conn.execute(
            "SELECT id, resource_id, text, locator_json FROM resource_chunks WHERE id = ?;",
            (object_id,),
        ).fetchone()
        if c_row:
            p_res_id = c_row["resource_id"]
            loc = {}
            if c_row["locator_json"]:
                try:
                    loc = json.loads(c_row["locator_json"])
                except Exception:
                    pass
            passage = {"passage": c_row["text"], "locator": loc, "content_hash": ""}
            combined = (extra_passages or []) + [passage]
            return build_evidence_item(
                conn, p_res_id, relevance_score=relevance_score, extra_passages=combined
            )

    # 1. Check if finding
    f_row = conn.execute(
        """
        SELECT f.id, f.resource_id, f.statement, f.assertion_role,
               f.agent_confidence, f.review_state, r.canonical_url, r.title,
               c.origin_namespace, c.origin_id,
               COALESCE(c.retrieved_at, r.created_at) as res_created
        FROM findings f
        LEFT JOIN resources r ON f.resource_id = r.id
        LEFT JOIN capture_resources cr ON cr.resource_id = r.id
        LEFT JOIN captures c ON cr.capture_id = c.id
        WHERE f.id = ? AND f.is_deleted = 0;
        """,
        (object_id,),
    ).fetchone()

    if f_row:
        # Load supporting passages
        passages_rows = conn.execute(
            "SELECT passage, locator_json, content_hash FROM finding_support WHERE finding_id = ?;",
            (object_id,),
        ).fetchall()
        passages = []
        for pr in passages_rows:
            loc = {}
            if pr["locator_json"]:
                try:
                    loc = json.loads(pr["locator_json"])
                except Exception:
                    pass
            passages.append(
                {
                    "passage": pr["passage"],
                    "locator": loc,
                    "content_hash": pr["content_hash"] or "",
                }
            )

        # Load labels
        lbl_rows = conn.execute(
            "SELECT label_id FROM object_labels WHERE object_id = ?;", (object_id,)
        ).fetchall()
        labels = [lr["label_id"] for lr in lbl_rows]

        # Load entities
        ent_rows = conn.execute(
            """
            SELECT e.name FROM object_entities oe
            JOIN entities e ON oe.entity_id = e.id
            WHERE oe.object_id = ?;
            """,
            (object_id,),
        ).fetchall()
        entities = [er["name"] for er in ent_rows]

        # Load user notes from parent resource or capture
        user_notes: list[str] = []
        if f_row["resource_id"]:
            ann_rows = conn.execute(
                "SELECT content FROM annotations WHERE object_id = ? AND author = 'human';",
                (f_row["resource_id"],),
            ).fetchall()
            user_notes = [ar["content"] for ar in ann_rows]

        source_dict = {}
        if f_row["resource_id"]:
            source_dict = {
                "id": f_row["resource_id"],
                "url": f_row["canonical_url"] or "",
                "title": f_row["title"] or "",
                "origin_namespace": f_row["origin_namespace"] or "web",
                "origin_id": f_row["origin_id"] or "",
                "retrieved_at": f_row["res_created"] or "",
            }

        return {
            "id": f_row["id"],
            "kind": "finding",
            "text": f_row["statement"],
            "assertion_role": f_row["assertion_role"] or "source-claim",
            "review_state": f_row["review_state"] or "unreviewed",
            "confidence": f_row["agent_confidence"],
            "relevance_score": relevance_score,
            "source": source_dict,
            "supporting_passages": passages,
            "user_notes": user_notes,
            "labels": labels,
            "entities": entities,
        }

    # 2. Check if resource
    r_row = conn.execute(
        """
        SELECT r.id, r.canonical_url, r.title, r.primary_form,
               r.review_state, r.created_at,
               c.origin_namespace, c.origin_id,
               COALESCE(c.retrieved_at, r.created_at) as res_created,
               rc.clean_text, rc.summary
        FROM resources r
        LEFT JOIN capture_resources cr ON cr.resource_id = r.id
        LEFT JOIN captures c ON cr.capture_id = c.id
        LEFT JOIN resource_contents rc ON rc.resource_id = r.id
        WHERE r.id = ? AND r.is_deleted = 0
        ORDER BY rc.created_at DESC LIMIT 1;
        """,
        (object_id,),
    ).fetchone()

    if r_row:
        text_body = (r_row["summary"] or r_row["clean_text"] or r_row["title"] or "").strip()

        # Load labels
        lbl_rows = conn.execute(
            "SELECT label_id FROM object_labels WHERE object_id = ?;", (object_id,)
        ).fetchall()
        labels = [lr["label_id"] for lr in lbl_rows]

        # Load entities
        ent_rows = conn.execute(
            """
            SELECT e.name FROM object_entities oe
            JOIN entities e ON oe.entity_id = e.id
            WHERE oe.object_id = ?;
            """,
            (object_id,),
        ).fetchall()
        entities = [er["name"] for er in ent_rows]

        # Load user notes
        ann_rows = conn.execute(
            "SELECT content FROM annotations WHERE object_id = ? AND author = 'human';",
            (object_id,),
        ).fetchall()
        user_notes = [ar["content"] for ar in ann_rows]

        source_dict = {
            "id": r_row["id"],
            "url": r_row["canonical_url"] or "",
            "title": r_row["title"] or "",
            "origin_namespace": r_row["origin_namespace"] or "web",
            "origin_id": r_row["origin_id"] or "",
            "retrieved_at": r_row["res_created"] or r_row["created_at"] or "",
        }

        passages = list(extra_passages or [])
        if text_body and not passages:
            passages.append({"passage": text_body[:500], "locator": {}, "content_hash": ""})

        return {
            "id": r_row["id"],
            "kind": "resource",
            "text": text_body[:1000],
            "assertion_role": "source-claim",
            "review_state": r_row["review_state"] or "unreviewed",
            "confidence": 1.0,
            "relevance_score": relevance_score,
            "source": source_dict,
            "supporting_passages": passages,
            "user_notes": user_notes,
            "labels": labels,
            "entities": entities,
        }

    # 3. Check if capture
    c_row = conn.execute(
        """
        SELECT c.id, c.user_note, c.raw_content, c.review_state, c.origin_namespace,
               c.origin_id, c.created_at
        FROM captures c
        WHERE c.id = ? AND c.is_deleted = 0;
        """,
        (object_id,),
    ).fetchone()

    if c_row:
        raw_text = (c_row["raw_content"] or "").strip()
        user_note = (c_row["user_note"] or "").strip()
        return {
            "id": c_row["id"],
            "kind": "capture" if raw_text else "note",
            "text": raw_text or user_note,
            "assertion_role": "source-claim" if raw_text else "personal-observation",
            "review_state": c_row["review_state"] or "unreviewed",
            "confidence": 1.0,
            "relevance_score": relevance_score,
            "source": {
                "id": c_row["id"],
                "url": "",
                "title": "Capture Note",
                "origin_namespace": c_row["origin_namespace"] or "manual",
                "origin_id": c_row["origin_id"] or "",
                "retrieved_at": c_row["created_at"] or "",
            },
            "supporting_passages": [],
            "user_notes": [user_note] if user_note else [],
            "labels": [],
            "entities": [],
        }

    return None


def search_hybrid(
    conn: sqlite3.Connection,
    query: str,
    limit: int = 10,
    intent_filter: str | None = None,
    topic_filter: str | None = None,
    form_filter: str | None = None,
    min_relevance: float | None = None,
    project_filter: str | None = None,
) -> dict[str, Any]:
    """Execute hybrid search combining FTS5 and vector similarity, and build an Evidence Packet.

    Conforms strictly to schemas/evidence-packet-v1.json.
    """
    # 1. Lexical retrieval
    lexical_query = research_terms(query)
    exact_matches = search_lexical(
        conn=conn,
        query=lexical_query,
        limit=limit * 2,
        intent=intent_filter,
        topic=topic_filter,
        form=form_filter,
        project=project_filter,
    )
    broad_matches = search_lexical(
        conn=conn,
        query=lexical_query,
        limit=limit * 2,
        intent=intent_filter,
        topic=topic_filter,
        form=form_filter,
        project=project_filter,
        match_any=True,
    )
    exact_ids = {item.id for item in exact_matches.results}
    lexical_matches = list(exact_matches.results) + [
        item for item in broad_matches.results if item.id not in exact_ids
    ]

    # Chunk hits cite their parent resource while keeping the matched text.
    chunk_passages_by_resource: dict[str, list[dict[str, Any]]] = {}
    mapped_lexical: list[dict[str, Any]] = []
    seen_lexical_ids: set[str] = set()
    for match in lexical_matches:
        object_id = match.id
        if match.object_type == "chunk":
            chunk = conn.execute(
                "SELECT resource_id, text, locator_json FROM resource_chunks WHERE id = ?;",
                (object_id,),
            ).fetchone()
            if not chunk:
                continue
            object_id = chunk["resource_id"]
            try:
                locator = json.loads(chunk["locator_json"]) if chunk["locator_json"] else {}
            except json.JSONDecodeError:
                locator = {}
            chunk_passages_by_resource.setdefault(object_id, []).append(
                {"passage": chunk["text"], "locator": locator, "content_hash": ""}
            )
        if object_id in seen_lexical_ids:
            continue
        seen_lexical_ids.add(object_id)
        mapped_lexical.append({"id": object_id, "object_type": match.object_type})

    # 2. Vector retrieval
    vector_matches = search_vector(
        conn=conn,
        query_text=query,
        limit=limit * 2,
    )

    # Map chunk hits to their parent resource while preserving chunk text and locator
    mapped_vector: list[dict[str, Any]] = []
    x_resource_ids: set[str] = set()

    for vm in vector_matches:
        v_type = vm.get("object_type")
        v_id = vm.get("object_id") or vm.get("id")
        if v_type == "resource_chunk" or (v_id and v_id.startswith("chk_")):
            c_row = conn.execute(
                "SELECT id, resource_id, text, locator_json FROM resource_chunks WHERE id = ?;",
                (v_id,),
            ).fetchone()
            if c_row:
                p_res_id = c_row["resource_id"]
                loc = {}
                if c_row["locator_json"]:
                    try:
                        loc = json.loads(c_row["locator_json"])
                    except Exception:
                        pass
                p_info = {"passage": c_row["text"], "locator": loc, "content_hash": ""}
                chunk_passages_by_resource.setdefault(p_res_id, []).append(p_info)
                mapped_vector.append(
                    {
                        "object_type": "resource",
                        "object_id": p_res_id,
                        "similarity": vm.get("similarity", 0.0),
                        "embedding_id": vm.get("embedding_id"),
                    }
                )
                continue
        if v_type == "capture" or (v_id and v_id.startswith("cap_")):
            resource_id = x_capture_resource_id(conn, v_id)
            if resource_id:
                x_resource_ids.add(resource_id)
                mapped_vector.append(
                    {
                        **vm,
                        "object_type": "resource",
                        "object_id": resource_id,
                    }
                )
                continue
        mapped_vector.append(vm)

    # If both the X capture and its resource (or a chunk from it) were returned,
    # let that post contribute only its best vector rank to RRF. Chunk passages
    # remain attached to the resource below.
    seen_x_resource_ids: set[str] = set()
    deduplicated_vector: list[dict[str, Any]] = []
    for item in mapped_vector:
        object_id = item.get("object_id") or item.get("id")
        if object_id in x_resource_ids:
            if object_id in seen_x_resource_ids:
                continue
            seen_x_resource_ids.add(object_id)
        deduplicated_vector.append(item)
    mapped_vector = deduplicated_vector

    if project_filter:
        allowed_rows = conn.execute(
            """
            SELECT object_id FROM project_objects
            WHERE project_id = ? AND membership_status IN ('accepted', 'candidate');
            """,
            (project_filter,),
        ).fetchall()
        allowed_ids = {row["object_id"] for row in allowed_rows}
        mapped_vector = [
            item
            for item in mapped_vector
            if (item.get("object_id") or item.get("id")) in allowed_ids
        ]

    # 3. Reciprocal Rank Fusion
    fused = reciprocal_rank_fusion(
        lexical_results=mapped_lexical,
        vector_results=mapped_vector,
        k=60,
        limit=limit,
    )

    # 4. Assemble Evidence Packet Items
    items: list[dict[str, Any]] = []
    seen_ids = set()

    for entry in fused:
        obj_id = entry["id"]
        if obj_id in seen_ids:
            continue
        seen_ids.add(obj_id)

        item = build_evidence_item(
            conn,
            obj_id,
            relevance_score=entry.get("relevance_score"),
            extra_passages=list(
                {
                    passage["passage"]: passage
                    for passage in chunk_passages_by_resource.get(obj_id, [])
                }.values()
            )[:3],
        )
        if item:
            if min_relevance is not None and item.get("relevance_score", 0.0) < min_relevance:
                continue
            if item["kind"] in ("capture", "note") and len(item["text"]) > 2000:
                full_text = item["text"]
                item["supporting_passages"] = matching_capture_passages(full_text, query)
                item["text"] = full_text[:1000]
            items.append(item)

    # 5. Detect Gaps and Disagreements
    gaps: list[dict[str, Any]] = []
    # If unreviewed items dominate, flag uncertainty
    unreviewed_ids = [it["id"] for it in items if it.get("review_state") == "unreviewed"]
    if unreviewed_ids and len(unreviewed_ids) >= len(items) // 2:
        gaps.append(
            {
                "kind": "uncertainty",
                "description": "Majority of retrieved evidence items are in unreviewed state.",
                "involved_ids": unreviewed_ids[:5],
            }
        )

    # Build conforming Evidence Packet
    now_iso = datetime.datetime.now(datetime.UTC).isoformat()
    packet = {
        "type": "evidence-packet",
        "schema_version": "1",
        "query": query,
        "created_at": now_iso,
        "parameters": {
            k: v
            for k, v in {
                "limit": limit,
                "intent_filter": intent_filter,
                "topic_filter": topic_filter,
                "form_filter": form_filter,
                "min_relevance": min_relevance,
                "project_filter": project_filter,
            }.items()
            if v is not None
        },
        "items": items,
        "gaps_and_disagreements": gaps,
    }

    return packet
