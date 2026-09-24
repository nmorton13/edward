"""Lifecycle management service for review, soft-deletion, intents, annotations, and purge."""

import datetime
import hashlib
import json
import mimetypes
import sqlite3
from typing import TYPE_CHECKING, Any

from edward.models import Annotation, ReviewState, generate_id
from edward.services.audit import parse_timestamp, record_audit_event
from edward.services.search import index_document, remove_document
from edward.services.shortlinks import (
    is_shortlink,
    looks_like_shortlink_interstitial,
    resolve_shortlink,
)
from edward.services.titles import derive_post_title, is_author_derived_title

if TYPE_CHECKING:
    from edward.blobs import BlobStore
    from edward.db import Database


class LifecycleError(Exception):
    """Base exception for lifecycle operations."""

    pass


class ObjectNotFoundError(LifecycleError):
    """Raised when the target object does not exist."""

    pass


def reindex_object_document(
    conn: sqlite3.Connection,
    object_type: str,
    object_id: str,
) -> None:
    """Re-compute and update the FTS search document for an object including all attached annotations and labels."""
    title = ""
    body = ""
    if object_type == "resource":
        row = conn.execute(
            "SELECT title, canonical_url, is_deleted FROM resources WHERE id = ?;",
            (object_id,),
        ).fetchone()
        if not row or row["is_deleted"] == 1:
            remove_document(conn, object_type, object_id)
            return
        title = row["title"] or row["canonical_url"] or "Resource"
        rc = conn.execute(
            "SELECT clean_text FROM resource_contents WHERE resource_id = ? ORDER BY created_at DESC LIMIT 1;",
            (object_id,),
        ).fetchone()
        body = rc["clean_text"] if rc else ""
    elif object_type == "capture":
        row = conn.execute(
            "SELECT raw_content, user_note, is_deleted FROM captures WHERE id = ?;",
            (object_id,),
        ).fetchone()
        if not row or row["is_deleted"] == 1:
            remove_document(conn, object_type, object_id)
            return
        title = row["user_note"] or "Capture Note"
        body = row["raw_content"] or ""
    elif object_type == "finding":
        row = conn.execute(
            "SELECT statement, is_deleted FROM findings WHERE id = ?;", (object_id,)
        ).fetchone()
        if not row or row["is_deleted"] == 1:
            remove_document(conn, object_type, object_id)
            return
        title = row["statement"]
        fs_rows = conn.execute(
            "SELECT passage FROM finding_support WHERE finding_id = ? ORDER BY created_at ASC;",
            (object_id,),
        ).fetchall()
        body = "\n\n".join([r["passage"] for r in fs_rows if r["passage"]])
    elif object_type == "chunk":
        row = conn.execute(
            """
            SELECT rc.text, r.title, r.is_deleted
            FROM resource_chunks rc
            JOIN resources r ON rc.resource_id = r.id
            WHERE rc.id = ?;
            """,
            (object_id,),
        ).fetchone()
        if not row or row["is_deleted"] == 1:
            remove_document(conn, object_type, object_id)
            return
        title = row["title"] or "Chunk"
        body = row["text"] or ""

    # Append all annotations content into the searchable body
    ann_cursor = conn.execute(
        "SELECT content FROM annotations WHERE object_type = ? AND object_id = ? ORDER BY created_at ASC;",
        (object_type, object_id),
    )
    ann_texts = [r["content"] for r in ann_cursor.fetchall() if r["content"]]
    if ann_texts:
        body = (body + "\n" + "\n".join(ann_texts)).strip()

    # Collect labels
    lbl_cursor = conn.execute(
        "SELECT label_id FROM object_labels WHERE object_type = ? AND object_id = ?;",
        (object_type, object_id),
    )
    labels = [r["label_id"] for r in lbl_cursor.fetchall()]

    # Collect entities
    ent_cursor = conn.execute(
        """
        SELECT e.name FROM object_entities oe
        JOIN entities e ON oe.entity_id = e.id
        WHERE oe.object_type = ? AND oe.object_id = ?;
        """,
        (object_type, object_id),
    )
    entities = [r["name"] for r in ent_cursor.fetchall()]

    index_document(
        conn, object_type, object_id, title=title, body=body, labels=labels, entities=entities
    )


def add_annotation(
    conn: sqlite3.Connection,
    object_type: str,
    object_id: str,
    content: str,
    annotation_type: str = "note",
    author: str = "human",
) -> Annotation:
    """Append a new annotation to an object, update FTS projection, and record audit event."""
    table = (
        "captures"
        if object_type == "capture"
        else ("resources" if object_type == "resource" else "findings")
    )
    row = conn.execute(f"SELECT id FROM {table} WHERE id = ?;", (object_id,)).fetchone()
    if not row:
        raise ObjectNotFoundError(f"{object_type} with ID {object_id} not found")

    ann_id = generate_id("ann")
    now = datetime.datetime.now(datetime.UTC)
    now_iso = now.isoformat()

    conn.execute(
        """
        INSERT INTO annotations (id, object_type, object_id, annotation_type, content, author, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?);
        """,
        (ann_id, object_type, object_id, annotation_type, content.strip(), author, now_iso),
    )

    reindex_object_document(conn, object_type, object_id)

    content_clean = content.strip()
    content_hash = hashlib.sha256(content_clean.encode("utf-8")).hexdigest()
    record_audit_event(
        conn,
        event_type="annotation.added",
        object_type=object_type,
        object_id=object_id,
        actor=author,
        payload={
            "annotation_id": ann_id,
            "annotation_type": annotation_type,
            "content_hash": content_hash,
            "char_count": len(content_clean),
        },
    )

    return Annotation(
        id=ann_id,
        object_type=object_type,
        object_id=object_id,
        annotation_type=annotation_type,
        content=content.strip(),
        author=author,
        created_at=now,
    )


def get_annotations(
    conn: sqlite3.Connection,
    object_type: str,
    object_id: str,
) -> list[Annotation]:
    """Retrieve all annotations for an object in chronological order."""
    cursor = conn.execute(
        """
        SELECT id, object_type, object_id, annotation_type, content, author, created_at
        FROM annotations
        WHERE object_type = ? AND object_id = ?
        ORDER BY created_at ASC;
        """,
        (object_type, object_id),
    )
    return [
        Annotation(
            id=row["id"],
            object_type=row["object_type"],
            object_id=row["object_id"],
            annotation_type=row["annotation_type"],
            content=row["content"],
            author=row["author"],
            created_at=parse_timestamp(row["created_at"]),
        )
        for row in cursor.fetchall()
    ]


def add_label(
    conn: sqlite3.Connection,
    object_type: str,
    object_id: str,
    label_id: str,
    source: str = "human",
    confidence: float = 1.0,
    actor: str = "human",
) -> None:
    """Attach a taxonomic label to an object, update FTS projection, and record audit event."""
    table = (
        "captures"
        if object_type == "capture"
        else ("resources" if object_type == "resource" else "findings")
    )
    row = conn.execute(f"SELECT id FROM {table} WHERE id = ?;", (object_id,)).fetchone()
    if not row:
        raise ObjectNotFoundError(f"{object_type} with ID {object_id} not found")

    now_iso = datetime.datetime.now(datetime.UTC).isoformat()
    conn.execute(
        """
        INSERT OR IGNORE INTO label_families (id, description, created_at)
        VALUES ('custom', 'Custom user labels', ?);
        """,
        (now_iso,),
    )
    conn.execute(
        """
        INSERT OR IGNORE INTO labels (id, family, description, active, version, created_at)
        VALUES (?, 'custom', 'Custom label', 1, '1.0', ?);
        """,
        (label_id, now_iso),
    )
    lbl_id = f"lbl_{object_id}_{label_id}_{source}"
    conn.execute(
        """
        INSERT OR IGNORE INTO object_labels (id, object_type, object_id, label_id, source, confidence, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?);
        """,
        (lbl_id, object_type, object_id, label_id, source, confidence, now_iso),
    )

    reindex_object_document(conn, object_type, object_id)

    record_audit_event(
        conn,
        event_type="label.added",
        object_type=object_type,
        object_id=object_id,
        actor=actor,
        payload={"label_id": label_id, "source": source},
    )


def set_review_state(
    conn: sqlite3.Connection,
    object_type: str,
    object_id: str,
    review_state: ReviewState,
    actor: str = "human",
) -> None:
    """Transition the review state of a capture, resource, or finding."""
    now = datetime.datetime.now(datetime.UTC).isoformat()
    table = (
        "captures"
        if object_type == "capture"
        else ("resources" if object_type == "resource" else "findings")
    )

    cursor = conn.execute(
        f"UPDATE {table} SET review_state = ?, updated_at = ? WHERE id = ?;",
        (review_state, now, object_id),
    )
    if cursor.rowcount == 0:
        raise ObjectNotFoundError(f"{object_type} with ID {object_id} not found")

    record_audit_event(
        conn,
        event_type="lifecycle.review_state_changed",
        object_type=object_type,
        object_id=object_id,
        actor=actor,
        payload={"review_state": review_state},
    )


def soft_delete(
    conn: sqlite3.Connection,
    object_type: str,
    object_id: str,
    actor: str = "human",
) -> None:
    """Soft delete an object and remove it from the search index."""
    now = datetime.datetime.now(datetime.UTC).isoformat()
    table = (
        "captures"
        if object_type == "capture"
        else ("resources" if object_type == "resource" else "findings")
    )

    cursor = conn.execute(
        f"UPDATE {table} SET is_deleted = 1, deleted_at = ?, updated_at = ? WHERE id = ? AND is_deleted = 0;",
        (now, now, object_id),
    )
    if cursor.rowcount == 0:
        raise ObjectNotFoundError(f"Active {object_type} with ID {object_id} not found")

    remove_document(conn, object_type, object_id)

    record_audit_event(
        conn,
        event_type="lifecycle.soft_deleted",
        object_type=object_type,
        object_id=object_id,
        actor=actor,
    )


def restore(
    conn: sqlite3.Connection,
    object_type: str,
    object_id: str,
    actor: str = "human",
) -> None:
    """Restore a previously soft-deleted object."""
    now = datetime.datetime.now(datetime.UTC).isoformat()
    table = (
        "captures"
        if object_type == "capture"
        else ("resources" if object_type == "resource" else "findings")
    )

    cursor = conn.execute(
        f"UPDATE {table} SET is_deleted = 0, deleted_at = NULL, updated_at = ? WHERE id = ? AND is_deleted = 1;",
        (now, object_id),
    )
    if cursor.rowcount == 0:
        raise ObjectNotFoundError(f"Soft-deleted {object_type} with ID {object_id} not found")

    reindex_object_document(conn, object_type, object_id)

    record_audit_event(
        conn,
        event_type="lifecycle.restored",
        object_type=object_type,
        object_id=object_id,
        actor=actor,
    )


def add_intent(
    conn: sqlite3.Connection,
    object_type: str,
    object_id: str,
    intent: str,
    source: str = "human",
    actor: str = "human",
    is_active: bool = True,
) -> None:
    """Attach an active or suggested/inactive intent to an object."""
    now = datetime.datetime.now(datetime.UTC).isoformat()
    intent_id = generate_id("int")
    active_int = 1 if is_active else 0

    conn.execute(
        """
        INSERT INTO intents (id, object_type, object_id, intent, source, is_active, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(object_type, object_id, intent)
        DO UPDATE SET
            is_active = CASE
                WHEN intents.source = 'human' AND excluded.source != 'human' THEN intents.is_active
                ELSE excluded.is_active
            END,
            source = CASE
                WHEN intents.source = 'human' AND excluded.source != 'human' THEN intents.source
                ELSE excluded.source
            END,
            updated_at = CASE
                WHEN intents.source = 'human' AND excluded.source != 'human' THEN intents.updated_at
                ELSE excluded.updated_at
            END;
        """,
        (intent_id, object_type, object_id, intent.strip(), source, active_int, now, now),
    )

    record_audit_event(
        conn,
        event_type="intent.added",
        object_type=object_type,
        object_id=object_id,
        actor=actor,
        payload={"intent": intent, "source": source, "is_active": is_active},
    )


def remove_intent(
    conn: sqlite3.Connection,
    object_type: str,
    object_id: str,
    intent: str,
    actor: str = "human",
) -> None:
    """Deactivate an intent while preserving history."""
    now = datetime.datetime.now(datetime.UTC).isoformat()
    cursor = conn.execute(
        """
        UPDATE intents
        SET is_active = 0, updated_at = ?
        WHERE object_type = ? AND object_id = ? AND intent = ? AND is_active = 1;
        """,
        (now, object_type, object_id, intent.strip()),
    )
    if cursor.rowcount == 0:
        raise ObjectNotFoundError(
            f"Active intent '{intent}' not found on {object_type} {object_id}"
        )

    record_audit_event(
        conn,
        event_type="intent.removed",
        object_type=object_type,
        object_id=object_id,
        actor=actor,
        payload={"intent": intent},
    )


def purge_object(
    conn: sqlite3.Connection,
    object_type: str,
    object_id: str,
    actor: str = "human",
) -> None:
    """Permanently delete an object from database and search index."""
    table = (
        "captures"
        if object_type == "capture"
        else ("resources" if object_type == "resource" else "findings")
    )

    # Record audit event before deleting
    record_audit_event(
        conn,
        event_type="lifecycle.purged",
        object_type=object_type,
        object_id=object_id,
        actor=actor,
        payload={"object_type": object_type, "purged": True},
    )

    # Collect dependent IDs before cascade deletion
    chunk_ids: list[str] = []
    if object_type == "resource":
        chunk_rows = conn.execute(
            "SELECT id FROM resource_chunks WHERE resource_id = ?;", (object_id,)
        ).fetchall()
        chunk_ids = [r["id"] for r in chunk_rows]

    # Remove polymorphic project references that SQLite foreign keys cannot enforce.
    conn.execute(
        "DELETE FROM project_objects WHERE object_type = ? AND object_id = ?;",
        (object_type, object_id),
    )
    conn.execute(
        "DELETE FROM outline_section_evidence WHERE object_type = ? AND object_id = ?;",
        (object_type, object_id),
    )

    # Delete row and let ON DELETE CASCADE handle children
    cursor = conn.execute(f"DELETE FROM {table} WHERE id = ?;", (object_id,))
    if cursor.rowcount == 0:
        raise ObjectNotFoundError(f"{object_type} with ID {object_id} not found")

    # Also delete annotations, intents, labels, and entities for this object
    conn.execute(
        "DELETE FROM annotations WHERE object_type = ? AND object_id = ?;", (object_type, object_id)
    )
    conn.execute(
        "DELETE FROM intents WHERE object_type = ? AND object_id = ?;", (object_type, object_id)
    )
    conn.execute(
        "DELETE FROM object_labels WHERE object_type = ? AND object_id = ?;",
        (object_type, object_id),
    )
    conn.execute(
        "DELETE FROM object_entities WHERE object_type = ? AND object_id = ?;",
        (object_type, object_id),
    )
    conn.execute("DELETE FROM attachments WHERE object_id = ?;", (object_id,))

    # Delete embeddings for this object
    conn.execute(
        "DELETE FROM embeddings WHERE object_type = ? AND object_id = ?;",
        (object_type, object_id),
    )

    # Delete dependent chunk and finding embeddings and search documents
    if chunk_ids:
        c_ph = ",".join("?" * len(chunk_ids))
        conn.execute(
            f"DELETE FROM embeddings WHERE object_type = 'resource_chunk' AND object_id IN ({c_ph});",
            chunk_ids,
        )
        conn.execute(
            f"DELETE FROM search_documents WHERE object_type = 'chunk' AND object_id IN ({c_ph});",
            chunk_ids,
        )

    # Scrub historical audit payloads containing user content for this purged object
    conn.execute(
        """
        UPDATE audit_events
        SET payload_json = '{"redacted": true, "reason": "purged"}'
        WHERE object_id = ? AND event_type != 'lifecycle.purged';
        """,
        (object_id,),
    )

    remove_document(conn, object_type, object_id)


def prune_unreferenced_blobs(db: "Database", blob_store: "BlobStore") -> list[str]:
    """Identify and delete blobs on disk that are no longer referenced by active SQLite records."""
    referenced_hashes: set[str] = set()
    with db.connection() as conn:
        for q in [
            "SELECT DISTINCT content_hash FROM source_snapshots;",
            "SELECT DISTINCT content_hash FROM attachments;",
        ]:
            try:
                for row in conn.execute(q).fetchall():
                    if row[0]:
                        referenced_hashes.add(row[0])
            except Exception:
                pass

    disk_hashes = blob_store.list_all_hashes()
    orphaned = disk_hashes - referenced_hashes
    pruned: list[str] = []
    for b_hash in orphaned:
        path = blob_store.get_path(b_hash)
        if path and path.exists():
            path.unlink(missing_ok=True)
            pruned.append(b_hash)
            try:
                path.parent.rmdir()
            except OSError:
                pass
    return pruned


def _snapshot_author_names(snapshot: dict[str, Any] | None) -> list[str]:
    """Author display name and handle recorded in a retained source snapshot."""
    if not isinstance(snapshot, dict):
        return []
    birdclaw = snapshot.get("birdclaw")
    if not isinstance(birdclaw, dict):
        return []
    return [
        value
        for value in (birdclaw.get("author_name"), birdclaw.get("author_handle"))
        if isinstance(value, str) and value.strip()
    ]


def repair_author_derived_titles(
    db: "Database",
    blob_store: "BlobStore",
) -> dict[str, Any]:
    """Re-derive titles that were set from an author instead of the source text.

    An X resource's title was previously written as the post's author name, so
    every retrieved item read as a bare person rather than the thing saved. The
    post text was always stored, so the correct title is recoverable in place.

    Only a title that exactly matches one of the resource's author identities is
    rewritten; a title already derived from the source is left untouched. The
    display name comes from the retained source snapshot, because a handle alone
    does not always match the stored title.
    """
    repaired: list[str] = []
    skipped = 0
    with db.connection() as conn:
        rows = conn.execute(
            """
            SELECT r.id AS id, r.title AS title, r.author AS author,
                   r.canonical_url AS url, s.content_hash AS content_hash
            FROM resources r
            LEFT JOIN source_snapshots s ON s.resource_id = r.id
            WHERE r.is_deleted = 0
              AND r.canonical_url LIKE 'http%://%/status/%';
            """
        ).fetchall()

    snapshot_authors: dict[str, list[str]] = {}
    for row in rows:
        content_hash = row["content_hash"]
        if not content_hash or content_hash in snapshot_authors:
            continue
        snapshot_authors[content_hash] = _snapshot_author_names(
            _read_snapshot_blob(blob_store, content_hash)
        )

    with db.transaction() as conn:
        for row in rows:
            identities: list[str | None] = [row["author"], row["url"]]
            identities.extend(snapshot_authors.get(row["content_hash"] or "", []))
            if not is_author_derived_title(row["title"], *identities):
                skipped += 1
                continue
            content = conn.execute(
                """
                SELECT clean_text FROM resource_contents
                WHERE resource_id = ? ORDER BY created_at DESC LIMIT 1;
                """,
                (row["id"],),
            ).fetchone()
            if not content or not (content["clean_text"] or "").strip():
                skipped += 1
                continue
            new_title = derive_post_title(
                content["clean_text"],
                fallback=row["author"] or row["url"] or row["id"],
            )
            if not new_title or new_title == row["title"]:
                skipped += 1
                continue
            conn.execute(
                "UPDATE resources SET title = ? WHERE id = ?;",
                (new_title, row["id"]),
            )
            reindex_object_document(conn, "resource", row["id"])
            repaired.append(row["id"])

    return {"repaired": len(repaired), "skipped": skipped, "resource_ids": repaired}


def backfill_shortlink_extractions(
    db: "Database",
    *,
    limit: int | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Re-extract resources whose content came from a shortlink's interstitial.

    Those resources hold the shortener's page shell — sign-in prompts, trending
    lists, cookie banners — because the snapshot was read directly instead of the
    link being resolved first. The stored text is therefore chrome, and the real
    article was never fetched.

    Each resource is re-extracted from its resolved destination and its content
    is replaced in place. Identity is untouched: the resource keeps its URL,
    annotations, intents, and project membership. Extraction jobs are also not
    re-queued, because the cache is keyed on the snapshot hash and would return
    the interstitial again.

    ``dry_run`` resolves and extracts without writing, so the caller can see the
    character counts before committing to the pass.
    """
    from edward.services.embed import store_resource_chunks
    from edward.services.extract import extract_content
    from edward.services.resource import store_resource_content

    recovered: list[dict[str, Any]] = []
    unresolved: list[str] = []
    skipped: list[str] = []
    failed: list[dict[str, str]] = []

    with db.connection() as conn:
        rows = conn.execute(
            """
            SELECT r.id AS id, r.canonical_url AS url, rc.clean_text AS clean_text
            FROM resources r
            JOIN resource_contents rc ON rc.resource_id = r.id
            WHERE r.is_deleted = 0
              AND rc.extractor = 'local-fallback'
            ORDER BY r.id;
            """
        ).fetchall()

    candidates = []
    for row in rows:
        if not is_shortlink(row["url"]):
            skipped.append(row["id"])
            continue
        # Only chrome is a defect. A shortlink whose stored text is already the
        # article must not be re-fetched and risked.
        if not looks_like_shortlink_interstitial(row["clean_text"]):
            skipped.append(row["id"])
            continue
        candidates.append(row)
    if limit is not None:
        candidates = candidates[:limit]

    for row in candidates:
        resolved_url = resolve_shortlink(row["url"])
        if resolved_url is None:
            unresolved.append(row["id"])
            continue
        try:
            result = extract_content(resolved_url)
        except Exception as exc:
            failed.append({"resource_id": row["id"], "error": str(exc)[:200]})
            continue
        new_text = result.clean_text or ""
        if not new_text:
            failed.append(
                {
                    "resource_id": row["id"],
                    "error": result.error or "resolved page produced no text",
                }
            )
            continue

        old_chars = len(row["clean_text"] or "")
        entry = {
            "resource_id": row["id"],
            "resolved_url": resolved_url,
            "old_chars": old_chars,
            "new_chars": len(new_text),
            "title": result.title,
            "extractor": result.extractor,
        }

        if not dry_run:
            with db.transaction() as conn:
                content_id, _ = store_resource_content(
                    conn,
                    row["id"],
                    new_text,
                    summary=result.summary,
                    extractor=result.extractor,
                    extractor_version=result.extractor_version,
                    extraction_note=f"backfilled from {resolved_url}",
                    title=result.title,
                )
                # store_resource_content only reprojects the resource document.
                # The superseded interstitial's chunks stay indexed until this
                # runs, which would leave the chrome searchable alongside the
                # article it replaced.
                store_resource_chunks(conn, row["id"], content_id, new_text)
                reindex_object_document(conn, "resource", row["id"])
        recovered.append(entry)

    return {
        "candidates": len(candidates),
        "recovered": len(recovered),
        "unresolved": len(unresolved),
        "skipped": len(skipped),
        "failed": failed,
        "dry_run": dry_run,
        "details": recovered,
    }


def repair_superseded_chunks(db: "Database") -> dict[str, Any]:
    """Re-chunk resources whose indexed chunks belong to a replaced extraction.

    ``store_resource_content`` writes a new extraction and reprojects the
    resource document, but it does not touch ``resource_chunks``. A caller that
    stores content without also calling ``store_resource_chunks`` therefore
    leaves the previous extraction's chunks — and their FTS rows and embeddings —
    indexed and searchable. Search then returns the superseded text alongside the
    text that replaced it.

    Repair is safe and re-runnable: ``store_resource_chunks`` deletes the
    resource's existing chunks, their embeddings, and their search documents
    before inserting the current content's chunks.

    Only resources whose chunks all belong to a non-latest extraction are
    touched. A resource with no chunks is left alone here; that is a separate
    question about whether it should have been chunked at all.
    """
    from edward.services.embed import store_resource_chunks

    repaired: list[str] = []
    considered = 0

    with db.connection() as conn:
        rows = conn.execute(
            """
            SELECT r.id AS id,
                   (SELECT rc.id FROM resource_contents rc
                     WHERE rc.resource_id = r.id
                     ORDER BY rc.created_at DESC LIMIT 1) AS latest_content_id,
                   (SELECT rc.clean_text FROM resource_contents rc
                     WHERE rc.resource_id = r.id
                     ORDER BY rc.created_at DESC LIMIT 1) AS latest_text
            FROM resources r
            WHERE r.is_deleted = 0
              AND EXISTS (SELECT 1 FROM resource_chunks k WHERE k.resource_id = r.id)
            ORDER BY r.id;
            """
        ).fetchall()

    for row in rows:
        considered += 1
        resource_id = row["id"]
        latest = row["latest_content_id"]
        if not latest:
            continue
        with db.connection() as conn:
            stale = conn.execute(
                """
                SELECT 1 FROM resource_chunks
                WHERE resource_id = ? AND resource_content_id IS NOT ?
                LIMIT 1;
                """,
                (resource_id, latest),
            ).fetchone()
        if not stale:
            continue

        with db.transaction() as conn:
            store_resource_chunks(conn, resource_id, latest, row["latest_text"] or "")
            reindex_object_document(conn, "resource", resource_id)
        repaired.append(resource_id)

    return {"considered": considered, "repaired": len(repaired), "resource_ids": repaired}


def _read_snapshot_blob(blob_store: "BlobStore", content_hash: str) -> dict[str, Any] | None:
    """Parse a retained source snapshot blob, or None when it is unreadable."""
    try:
        raw = blob_store.read_bytes(content_hash)
    except (FileNotFoundError, OSError):
        return None
    try:
        parsed = json.loads(raw.decode("utf-8", errors="replace"))
    except (ValueError, UnicodeDecodeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def migrate_media_resources_to_attachments(
    db: "Database",
    blob_store: "BlobStore",
) -> dict[str, Any]:
    """Retire media-asset resources, re-homing their bytes as post attachments.

    Older ingests turned every image URL in a post's entities into its own
    resource, so 243 standalone "sources" existed whose entire content was a
    JPEG that nothing could read. Those rows are not sources and never were.

    For each one: attach the already-downloaded image bytes to the post that
    carried it, queue OCR on that attachment, then purge the bogus resource.
    The bytes are copied into the attachment row first, so no image is lost if
    the purge fails partway.
    """
    from edward.services.ocr import enqueue_ocr_extraction, is_image_attachment
    from edward.services.source_adapters import is_twitter_media_url

    converted: list[str] = []
    skipped: list[str] = []
    failed: list[dict[str, str]] = []

    with db.connection() as conn:
        candidates = conn.execute(
            """
            SELECT r.id, r.canonical_url, s.content_hash
            FROM resources r
            LEFT JOIN source_snapshots s ON s.resource_id = r.id
            ORDER BY r.id;
            """
        ).fetchall()

    for row in candidates:
        resource_id = row["id"]
        url = row["canonical_url"] or ""
        if not is_twitter_media_url(url):
            continue

        try:
            with db.transaction() as conn:
                parent = conn.execute(
                    """
                    SELECT cr.capture_id, cr2.resource_id AS parent_resource_id
                    FROM capture_resources cr
                    LEFT JOIN capture_resources cr2
                      ON cr2.capture_id = cr.capture_id
                     AND cr2.relationship_type = 'primary'
                    WHERE cr.resource_id = ?
                    ORDER BY cr.created_at ASC LIMIT 1;
                    """,
                    (resource_id,),
                ).fetchone()

                if not parent or not parent["parent_resource_id"]:
                    skipped.append(resource_id)
                    continue

                # Re-home the bytes as an attachment of the post.
                content_hash = row["content_hash"]
                attachment_saved = False
                if content_hash and blob_store.exists(content_hash):
                    file_name = url.rsplit("/", 1)[-1] or "image"
                    mime_type = mimetypes.guess_type(file_name)[0] or "application/octet-stream"
                    if is_image_attachment(file_name, mime_type):
                        existing = conn.execute(
                            """
                            SELECT 1 FROM attachments
                            WHERE object_type = 'resource' AND object_id = ?
                              AND content_hash = ? LIMIT 1;
                            """,
                            (parent["parent_resource_id"], content_hash),
                        ).fetchone()
                        if not existing:
                            attachment_id = generate_id("att")
                            size_bytes = conn.execute(
                                "SELECT size_bytes FROM source_snapshots WHERE content_hash = ? LIMIT 1;",
                                (content_hash,),
                            ).fetchone()
                            conn.execute(
                                """
                                INSERT INTO attachments (
                                    id, object_type, object_id, file_name, mime_type,
                                    content_hash, size_bytes, blob_path, created_at
                                ) VALUES (?, 'resource', ?, ?, ?, ?, ?, ?, ?);
                                """,
                                (
                                    attachment_id,
                                    parent["parent_resource_id"],
                                    file_name,
                                    mime_type,
                                    content_hash,
                                    size_bytes["size_bytes"] if size_bytes else None,
                                    f"{content_hash[:2]}/{content_hash}",
                                    datetime.datetime.now(datetime.UTC).isoformat(),
                                ),
                            )
                            enqueue_ocr_extraction(
                                conn,
                                parent["capture_id"],
                                attachment_id,
                            )
                            attachment_saved = True

                # Only retire the bogus resource once its bytes are preserved.
                if attachment_saved or not content_hash or not blob_store.exists(content_hash):
                    purge_object(conn, "resource", resource_id, actor="migration:media-attachments")
                    converted.append(resource_id)
                else:
                    skipped.append(resource_id)
        except Exception as exc:
            failed.append({"resource_id": resource_id, "error": str(exc)})

    return {
        "converted": len(converted),
        "skipped": len(skipped),
        "failed": failed,
        "converted_ids": converted,
    }
