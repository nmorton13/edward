"""Full-Text Search (FTS5) service for Edward."""

import logging
import re
import sqlite3

from edward.models import SearchItem, SearchResponse

logger = logging.getLogger(__name__)


def sanitize_fts_query(query: str, match_any: bool = False) -> str:
    """Sanitize a raw user query string into a safe FTS5 query."""
    clean = query.strip()
    if not clean:
        return ""

    tokens = re.findall(r"\"[^\"]+\"|\S+", clean)
    safe_terms: list[str] = []

    for token in tokens:
        if token.startswith('"') and token.endswith('"') and len(token) > 2:
            inner = token[1:-1].replace('"', '""')
            safe_terms.append(f'"{inner}"')
        else:
            sanitized = re.sub(r"[^\w\-\.:/]", "", token)
            if sanitized:
                safe_terms.append(f'"{sanitized}"')

    return (" OR " if match_any else " ").join(safe_terms)


def index_document(
    conn: sqlite3.Connection,
    object_type: str,
    object_id: str,
    title: str | None = None,
    body: str | None = None,
    labels: list[str] | None = None,
    entities: list[str] | None = None,
) -> None:
    """Synchronously insert or update a document in the FTS5 search index."""
    title_str = title or ""
    body_str = body or ""
    labels_str = " ".join(labels) if labels else ""
    entities_str = " ".join(entities) if entities else ""

    # Remove existing projection
    conn.execute(
        "DELETE FROM search_documents WHERE object_type = ? AND object_id = ?;",
        (object_type, object_id),
    )

    # Insert updated projection
    conn.execute(
        """
        INSERT INTO search_documents (object_type, object_id, title, body, labels, entities)
        VALUES (?, ?, ?, ?, ?, ?);
        """,
        (object_type, object_id, title_str, body_str, labels_str, entities_str),
    )


def remove_document(conn: sqlite3.Connection, object_type: str, object_id: str) -> None:
    """Remove a document from the FTS5 search index."""
    conn.execute(
        "DELETE FROM search_documents WHERE object_type = ? AND object_id = ?;",
        (object_type, object_id),
    )


def rebuild_search_index(conn: sqlite3.Connection) -> int:
    """Rebuild the entire search_documents virtual table from active database records."""
    conn.execute("DELETE FROM search_documents;")
    indexed_count = 0

    # 1. Index non-deleted resources (using latest extracted content)
    cursor = conn.execute(
        """
        SELECT r.id, r.title, r.canonical_url,
               (SELECT rc.clean_text FROM resource_contents rc
                WHERE rc.resource_id = r.id
                ORDER BY rc.created_at DESC LIMIT 1) AS clean_text
        FROM resources r
        WHERE r.is_deleted = 0;
        """
    )
    for row in cursor.fetchall():
        r_id = row["id"]
        title = row["title"] or row["canonical_url"] or ""
        body = row["clean_text"] or ""

        # Include annotations
        ann_cursor = conn.execute(
            "SELECT content FROM annotations WHERE object_type = 'resource' AND object_id = ? ORDER BY created_at ASC;",
            (r_id,),
        )
        ann_texts = [r["content"] for r in ann_cursor.fetchall() if r["content"]]
        if ann_texts:
            body = (body + "\n" + "\n".join(ann_texts)).strip()

        lbl_cursor = conn.execute(
            "SELECT label_id FROM object_labels WHERE object_type = 'resource' AND object_id = ?;",
            (r_id,),
        )
        labels = [lbl["label_id"] for lbl in lbl_cursor.fetchall()]

        ent_cursor = conn.execute(
            """
            SELECT e.name FROM object_entities oe
            JOIN entities e ON oe.entity_id = e.id
            WHERE oe.object_type = 'resource' AND oe.object_id = ?;
            """,
            (r_id,),
        )
        entities = [e["name"] for e in ent_cursor.fetchall()]

        index_document(
            conn, "resource", r_id, title=title, body=body, labels=labels, entities=entities
        )
        indexed_count += 1

    # 2. Index all non-deleted captures (preserving context for every capture)
    c_cursor = conn.execute(
        """
        SELECT c.id, c.raw_content, c.user_note
        FROM captures c
        WHERE c.is_deleted = 0;
        """
    )
    for row in c_cursor.fetchall():
        c_id = row["id"]
        title = row["user_note"] or "Capture Note"
        body = row["raw_content"] or ""

        ann_cursor = conn.execute(
            "SELECT content FROM annotations WHERE object_type = 'capture' AND object_id = ? ORDER BY created_at ASC;",
            (c_id,),
        )
        ann_texts = [r["content"] for r in ann_cursor.fetchall() if r["content"]]
        if ann_texts:
            body = (body + "\n" + "\n".join(ann_texts)).strip()

        index_document(conn, "capture", c_id, title=title, body=body)
        indexed_count += 1

    # 3. Index non-deleted findings (aggregating all supporting passages deterministically)
    f_cursor = conn.execute(
        """
        SELECT f.id, f.statement
        FROM findings f
        WHERE f.is_deleted = 0;
        """
    )
    for row in f_cursor.fetchall():
        f_id = row["id"]
        title = row["statement"]
        fs_cursor = conn.execute(
            "SELECT passage FROM finding_support WHERE finding_id = ? ORDER BY created_at ASC;",
            (f_id,),
        )
        passages = [p["passage"] for p in fs_cursor.fetchall() if p["passage"]]
        body = "\n\n".join(passages)

        ann_cursor = conn.execute(
            "SELECT content FROM annotations WHERE object_type = 'finding' AND object_id = ? ORDER BY created_at ASC;",
            (f_id,),
        )
        ann_texts = [r["content"] for r in ann_cursor.fetchall() if r["content"]]
        if ann_texts:
            body = (body + "\n" + "\n".join(ann_texts)).strip()

        lbl_cursor = conn.execute(
            "SELECT label_id FROM object_labels WHERE object_type = 'finding' AND object_id = ?;",
            (f_id,),
        )
        labels = [lbl["label_id"] for lbl in lbl_cursor.fetchall()]

        index_document(conn, "finding", f_id, title=title, body=body, labels=labels)
        indexed_count += 1

    # 4. Index active resource chunks (where parent resource is not deleted)
    chk_cursor = conn.execute(
        """
        SELECT rc.id, rc.text, r.title, r.canonical_url
        FROM resource_chunks rc
        JOIN resources r ON rc.resource_id = r.id
        WHERE r.is_deleted = 0;
        """
    )
    for row in chk_cursor.fetchall():
        chk_id = row["id"]
        title = row["title"] or row["canonical_url"] or "Chunk"
        body = row["text"] or ""
        index_document(conn, "chunk", chk_id, title=title, body=body)
        indexed_count += 1

    return indexed_count


def x_capture_resource_id(conn: sqlite3.Connection, capture_id: str) -> str | None:
    """Return the primary resource for a context-free imported X capture."""
    row = conn.execute(
        """
        SELECT r.id
        FROM captures c
        JOIN capture_resources cr ON cr.capture_id = c.id
        JOIN resources r ON r.id = cr.resource_id
        WHERE c.id = ?
          AND c.origin_namespace = 'x'
          AND c.user_note IS NULL
          AND c.is_deleted = 0
          AND cr.relationship_type = 'primary'
          AND r.is_deleted = 0
          AND NOT EXISTS (
              SELECT 1 FROM annotations a
              WHERE a.object_type = 'capture'
                AND a.object_id = c.id
                AND a.author = 'human'
          )
        LIMIT 1;
        """,
        (capture_id,),
    ).fetchone()
    return row["id"] if row else None


def search_lexical(
    conn: sqlite3.Connection,
    query: str,
    limit: int = 20,
    offset: int = 0,
    intent: str | None = None,
    topic: str | None = None,
    form: str | None = None,
    project: str | None = None,
    match_any: bool = False,
) -> SearchResponse:
    """Execute FTS5 search query with snippet extraction and filters, re-raising operational errors."""
    safe_query = sanitize_fts_query(query, match_any=match_any)
    if not safe_query:
        return SearchResponse(query=query, count=0, results=[])

    base_sql = """
        SELECT
            sd.object_type,
            sd.object_id,
            sd.title,
            sd.body,
            sd.labels,
            sd.entities,
            bm25(search_documents) AS score,
            snippet(search_documents, -1, '<b>', '</b>', '...', 15) AS snippet
        FROM search_documents sd
        WHERE search_documents MATCH ?
          AND NOT (
            sd.object_type = 'capture'
            AND EXISTS (
                SELECT 1
                FROM captures c
                JOIN capture_resources cr ON cr.capture_id = c.id
                JOIN resources r ON r.id = cr.resource_id
                WHERE c.id = sd.object_id
                  AND c.origin_namespace = 'x'
                  AND c.user_note IS NULL
                  AND c.is_deleted = 0
                  AND cr.relationship_type = 'primary'
                  AND r.is_deleted = 0
                  AND NOT EXISTS (
                      SELECT 1 FROM annotations a
                      WHERE a.object_type = 'capture'
                        AND a.object_id = c.id
                        AND a.author = 'human'
                  )
            )
          )
    """
    params: list = [safe_query]

    conditions: list[str] = []

    if intent:
        conditions.append(
            """
            EXISTS (
                SELECT 1 FROM intents i
                WHERE i.object_id = sd.object_id
                  AND i.intent = ?
                  AND i.is_active = 1
            )
            """
        )
        params.append(intent)

    if topic:
        conditions.append(
            """
            EXISTS (
                SELECT 1 FROM object_labels ol
                WHERE ol.object_id = CASE
                    WHEN sd.object_type = 'chunk' THEN (
                        SELECT rc.resource_id FROM resource_chunks rc WHERE rc.id = sd.object_id
                    )
                    ELSE sd.object_id
                END
                  AND (ol.label_id = ? OR ol.label_id LIKE ? || '/%')
            )
            """
        )
        params.extend([topic, topic])

    if form:
        conditions.append(
            """
            EXISTS (
                SELECT 1 FROM resources r
                WHERE r.id = CASE
                    WHEN sd.object_type = 'chunk' THEN (
                        SELECT rc.resource_id FROM resource_chunks rc WHERE rc.id = sd.object_id
                    )
                    ELSE sd.object_id
                END
                  AND (
                    r.primary_form = ?
                    OR EXISTS (
                        SELECT 1 FROM object_labels ol
                        JOIN labels l ON l.id = ol.label_id
                        WHERE ol.object_type = 'resource' AND ol.object_id = r.id
                          AND ol.label_id = ? AND l.family = 'form'
                    )
                  )
            )
            """
        )
        params.extend([form, form])

    if project:
        conditions.append(
            """
            EXISTS (
                SELECT 1 FROM project_objects po
                WHERE po.project_id = ?
                  AND po.membership_status IN ('accepted', 'candidate')
                  AND (
                    (po.object_type = sd.object_type AND po.object_id = sd.object_id)
                    OR (
                      sd.object_type = 'chunk'
                      AND po.object_type = 'resource'
                      AND po.object_id = (
                        SELECT rc.resource_id FROM resource_chunks rc WHERE rc.id = sd.object_id
                      )
                    )
                  )
            )
            """
        )
        params.append(project)

    if conditions:
        base_sql += " AND " + " AND ".join(conditions)

    base_sql += " ORDER BY score ASC LIMIT ? OFFSET ?;"
    params.extend([limit, offset])

    try:
        cursor = conn.execute(base_sql, params)
        rows = cursor.fetchall()
    except sqlite3.OperationalError as e:
        err_msg = str(e).lower()
        if (
            "fts5: syntax error" in err_msg
            or "fts5: parse error" in err_msg
            or "unterminated string" in err_msg
            or "syntax error near" in err_msg
        ):
            logger.debug("FTS5 query syntax error for %r: %s", query, e)
            return SearchResponse(query=query, count=0, results=[])
        # Re-raise genuine database/table corruption/missing column/locking errors
        raise

    results: list[SearchItem] = []
    for row in rows:
        snippet_val = row["snippet"] or ""
        labels_list = [lbl for lbl in (row["labels"] or "").split() if lbl]
        entities_list = [e for e in (row["entities"] or "").split() if e]

        results.append(
            SearchItem(
                id=row["object_id"],
                object_type=row["object_type"],
                title=row["title"],
                snippet=snippet_val,
                score=round(row["score"], 4) if row["score"] is not None else None,
                labels=labels_list,
                entities=entities_list,
            )
        )

    return SearchResponse(query=query, count=len(results), results=results)
