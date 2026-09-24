"""Resource management, source snapshots, and content caching."""

import datetime
import hashlib
import json
import sqlite3

from edward.blobs import BlobStore
from edward.models import generate_id
from edward.services.capture import canonicalize_url, hash_url
from edward.services.classification import compute_classification_input_hash
from edward.services.lifecycle import reindex_object_document
from edward.services.network import filter_snapshot_headers


def get_or_create_url_resource(
    conn: sqlite3.Connection,
    raw_url: str,
    title: str | None = None,
) -> tuple[str, bool]:
    """Retrieve existing resource by canonical URL or create a new one."""
    canonical_url = canonicalize_url(raw_url)
    u_hash = hash_url(canonical_url)
    identity_key = f"url:{canonical_url}"

    row = conn.execute(
        "SELECT id FROM resources WHERE identity_key = ? OR canonical_url = ?;",
        (identity_key, canonical_url),
    ).fetchone()
    if row:
        return row["id"], False

    now_iso = datetime.datetime.now(datetime.UTC).isoformat()
    resource_id = generate_id("res")
    conn.execute(
        """
        INSERT INTO resources (
            id, identity_key, canonical_url, url_hash, title,
            review_state, is_deleted, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, 'unreviewed', 0, ?, ?);
        """,
        (resource_id, identity_key, canonical_url, u_hash, title, now_iso, now_iso),
    )
    return resource_id, True


def get_cached_content(conn: sqlite3.Connection, resource_id: str) -> dict | None:
    """Return the latest cached extracted content for a resource if available."""
    row = conn.execute(
        """
        SELECT id, content_hash, clean_text, summary, char_count, extractor, extractor_version, created_at
        FROM resource_contents
        WHERE resource_id = ?
        ORDER BY created_at DESC
        LIMIT 1;
        """,
        (resource_id,),
    ).fetchone()
    return dict(row) if row else None


def save_source_snapshot(
    conn: sqlite3.Connection,
    blob_store: BlobStore,
    resource_id: str,
    content: bytes,
    headers: dict[str, str],
    capture_id: str | None = None,
) -> tuple[str, str]:
    """Store raw snapshot in content-addressed blob store and record in source_snapshots with filtered headers."""
    content_hash, blob_path = blob_store.store_bytes(content)
    rel_blob_path = f"{content_hash[:2]}/{content_hash}"
    filtered_hdrs = filter_snapshot_headers(headers)
    headers_json = json.dumps(filtered_hdrs) if filtered_hdrs else None

    snapshot_id = generate_id("snp")
    now_iso = datetime.datetime.now(datetime.UTC).isoformat()

    conn.execute(
        """
        INSERT INTO source_snapshots (
            id, resource_id, content_hash, headers_json, blob_path, size_bytes, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?);
        """,
        (
            snapshot_id,
            resource_id,
            content_hash,
            headers_json,
            rel_blob_path,
            len(content),
            now_iso,
        ),
    )

    if not capture_id:
        cap_row = conn.execute(
            "SELECT capture_id FROM capture_resources WHERE resource_id = ? ORDER BY created_at DESC LIMIT 1;",
            (resource_id,),
        ).fetchone()
        if cap_row:
            capture_id = cap_row["capture_id"]

    ext_job_key = f"extract:{resource_id}"
    conn.execute(
        """
        UPDATE processing_jobs
        SET capture_id = COALESCE(?, capture_id),
            status = CASE
                WHEN (status = 'completed' AND (input_hash IS NULL OR input_hash != ?))
                     OR status = 'failed'
                     OR (status = 'pending' AND lease_owner IS NULL)
                THEN 'pending'
                ELSE status
            END,
            available_at = CASE
                WHEN (status = 'completed' AND (input_hash IS NULL OR input_hash != ?))
                     OR status = 'failed'
                     OR (status = 'pending' AND lease_owner IS NULL)
                THEN ?
                ELSE available_at
            END,
            attempts = CASE
                WHEN (status = 'completed' AND (input_hash IS NULL OR input_hash != ?))
                     OR status = 'failed'
                     OR (status = 'pending' AND lease_owner IS NULL)
                THEN 0
                ELSE attempts
            END,
            last_error = CASE
                WHEN (status = 'completed' AND (input_hash IS NULL OR input_hash != ?))
                     OR status = 'failed'
                     OR (status = 'pending' AND lease_owner IS NULL)
                THEN NULL
                ELSE last_error
            END,
            completed_at = CASE
                WHEN (status = 'completed' AND (input_hash IS NULL OR input_hash != ?))
                     OR status = 'failed'
                     OR (status = 'pending' AND lease_owner IS NULL)
                THEN NULL
                ELSE completed_at
            END,
            started_at = CASE
                WHEN (status = 'completed' AND (input_hash IS NULL OR input_hash != ?))
                     OR status = 'failed'
                     OR (status = 'pending' AND lease_owner IS NULL)
                THEN NULL
                ELSE started_at
            END,
            lease_owner = CASE
                WHEN (status = 'completed' AND (input_hash IS NULL OR input_hash != ?))
                     OR status = 'failed'
                     OR (status = 'pending' AND lease_owner IS NULL)
                THEN NULL
                ELSE lease_owner
            END,
            lease_expires_at = CASE
                WHEN (status = 'completed' AND (input_hash IS NULL OR input_hash != ?))
                     OR status = 'failed'
                     OR (status = 'pending' AND lease_owner IS NULL)
                THEN NULL
                ELSE lease_expires_at
            END,
            updated_at = ?
        WHERE job_key = ?;
        """,
        (
            capture_id,
            content_hash,
            content_hash,
            now_iso,
            content_hash,
            content_hash,
            content_hash,
            content_hash,
            content_hash,
            content_hash,
            now_iso,
            ext_job_key,
        ),
    )

    return snapshot_id, content_hash


def store_resource_content(
    conn: sqlite3.Connection,
    resource_id: str,
    clean_text: str,
    summary: str | None = None,
    extractor: str = "local",
    extractor_version: str = "1.0",
    title: str | None = None,
    capture_id: str | None = None,
    extraction_note: str | None = None,
) -> tuple[str, str]:
    """Store extracted clean text in resource_contents, update resource metadata, and reproject to FTS.

    ``extraction_note`` records *why* an extraction produced what it did — for
    example which local fallback fired and on what grounds. Without it, a row's
    extractor name is the only clue and it cannot distinguish a tool failure
    from a placeholder from an unreadable output shape.
    """
    text_clean = clean_text.strip()
    c_hash = hashlib.sha256(text_clean.encode("utf-8")).hexdigest()
    content_id = generate_id("rc")
    now_iso = datetime.datetime.now(datetime.UTC).isoformat()

    conn.execute(
        """
        INSERT INTO resource_contents (
            id, resource_id, content_hash, clean_text, summary, char_count,
            extractor, extractor_version, extraction_note, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """,
        (
            content_id,
            resource_id,
            c_hash,
            text_clean,
            summary,
            len(text_clean),
            extractor,
            extractor_version,
            extraction_note,
            now_iso,
        ),
    )

    # Update resource latest hash, title, and timestamp
    if title:
        conn.execute(
            """
            UPDATE resources
            SET latest_content_hash = ?,
                title = CASE WHEN title IS NULL OR title = canonical_url THEN ? ELSE title END,
                updated_at = ?
            WHERE id = ?;
            """,
            (c_hash, title, now_iso, resource_id),
        )
    else:
        conn.execute(
            """
            UPDATE resources
            SET latest_content_hash = ?,
                updated_at = ?
            WHERE id = ?;
            """,
            (c_hash, now_iso, resource_id),
        )

    # Synchronously update search projection
    reindex_object_document(conn, "resource", resource_id)

    if not capture_id:
        cap_row = conn.execute(
            "SELECT capture_id FROM capture_resources WHERE resource_id = ? ORDER BY created_at DESC LIMIT 1;",
            (resource_id,),
        ).fetchone()
        if cap_row:
            capture_id = cap_row["capture_id"]

    # Re-queue downstream classify job if previously completed with different input, failed, or unleased pending, and update capture_id
    cls_job_key = f"classify:{resource_id}"
    cls_input_hash = compute_classification_input_hash(
        clean_text=text_clean,
        summary=summary,
        title=title,
    )

    conn.execute(
        """
        UPDATE processing_jobs
        SET capture_id = COALESCE(?, capture_id),
            status = CASE
                WHEN (status = 'completed' AND (input_hash IS NULL OR input_hash != ?))
                     OR status = 'failed'
                     OR (status = 'pending' AND lease_owner IS NULL)
                THEN 'pending'
                ELSE status
            END,
            available_at = CASE
                WHEN (status = 'completed' AND (input_hash IS NULL OR input_hash != ?))
                     OR status = 'failed'
                     OR (status = 'pending' AND lease_owner IS NULL)
                THEN ?
                ELSE available_at
            END,
            attempts = CASE
                WHEN (status = 'completed' AND (input_hash IS NULL OR input_hash != ?))
                     OR status = 'failed'
                     OR (status = 'pending' AND lease_owner IS NULL)
                THEN 0
                ELSE attempts
            END,
            last_error = CASE
                WHEN (status = 'completed' AND (input_hash IS NULL OR input_hash != ?))
                     OR status = 'failed'
                     OR (status = 'pending' AND lease_owner IS NULL)
                THEN NULL
                ELSE last_error
            END,
            completed_at = CASE
                WHEN (status = 'completed' AND (input_hash IS NULL OR input_hash != ?))
                     OR status = 'failed'
                     OR (status = 'pending' AND lease_owner IS NULL)
                THEN NULL
                ELSE completed_at
            END,
            started_at = CASE
                WHEN (status = 'completed' AND (input_hash IS NULL OR input_hash != ?))
                     OR status = 'failed'
                     OR (status = 'pending' AND lease_owner IS NULL)
                THEN NULL
                ELSE started_at
            END,
            lease_owner = CASE
                WHEN (status = 'completed' AND (input_hash IS NULL OR input_hash != ?))
                     OR status = 'failed'
                     OR (status = 'pending' AND lease_owner IS NULL)
                THEN NULL
                ELSE lease_owner
            END,
            lease_expires_at = CASE
                WHEN (status = 'completed' AND (input_hash IS NULL OR input_hash != ?))
                     OR status = 'failed'
                     OR (status = 'pending' AND lease_owner IS NULL)
                THEN NULL
                ELSE lease_expires_at
            END,
            updated_at = ?
        WHERE job_key = ?;
        """,
        (
            capture_id,
            cls_input_hash,
            cls_input_hash,
            now_iso,
            cls_input_hash,
            cls_input_hash,
            cls_input_hash,
            cls_input_hash,
            cls_input_hash,
            cls_input_hash,
            now_iso,
            cls_job_key,
        ),
    )

    return content_id, c_hash
