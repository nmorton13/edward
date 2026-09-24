"""Capture and ingestion service implementing CAP-1 with idempotency conflict detection."""

import datetime
import hashlib
import json
import sqlite3
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from edward.models import CaptureInput, generate_id, make_job_key
from edward.services.audit import record_audit_event
from edward.services.lifecycle import add_intent
from edward.services.pdf import enqueue_pdf_extraction, is_pdf_attachment
from edward.services.search import index_document


class CaptureError(Exception):
    """Base error for capture operations."""

    pass


class IdempotencyConflictError(CaptureError):
    """Raised when an idempotency key is reused with a different request hash."""

    pass


def canonicalize_url(raw_url: str) -> str:
    """Canonicalize a URL: validate scheme and hostname, normalize, strip tracking params and trailing slash."""
    clean = raw_url.strip()
    if not clean:
        raise ValueError("URL cannot be empty")

    parsed = urlparse(clean)
    scheme = parsed.scheme.lower()
    if scheme not in ("http", "https"):
        raise ValueError(
            f"Invalid or unsupported URL scheme in '{raw_url}': only http and https are supported"
        )

    netloc = parsed.netloc.lower()
    if not netloc or netloc.startswith(".") or netloc.endswith("."):
        raise ValueError(f"Invalid URL hostname in '{raw_url}'")

    # Filter tracking query parameters
    tracking_params = {
        "utm_source",
        "utm_medium",
        "utm_campaign",
        "utm_term",
        "utm_content",
        "fbclid",
        "gclid",
        "msclkid",
        "ref",
        "mc_eid",
    }
    filtered_query = [
        (k, v)
        for k, v in parse_qsl(parsed.query, keep_blank_values=True)
        if k.lower() not in tracking_params
    ]
    query_str = urlencode(filtered_query)

    path = parsed.path
    if len(path) > 1 and path.endswith("/"):
        path = path.rstrip("/")

    return urlunparse((scheme, netloc, path, parsed.params, query_str, parsed.fragment))


def hash_url(canonical_url: str) -> str:
    """Compute SHA-256 hash of canonical URL."""
    return hashlib.sha256(canonical_url.encode("utf-8")).hexdigest()


def compute_request_hash(data: dict[str, Any]) -> str:
    """Compute deterministic SHA-256 hash of a request payload."""
    serialized = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def capture_item(
    conn: sqlite3.Connection,
    input_data: CaptureInput,
    attachment_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Capture a URL, text, or note into Edward with strict idempotency conflict detection."""
    now = datetime.datetime.now(datetime.UTC)
    now_iso = now.isoformat()

    # 1. Idempotency Check
    if input_data.idempotency_key:
        namespace = "capture"
        operation = "add"
        key = input_data.idempotency_key
        req_payload = input_data.model_dump(exclude={"idempotency_key"})
        if attachment_info:
            req_payload["attachment"] = {
                "content_hash": attachment_info.get("content_hash"),
                "file_name": attachment_info.get("file_name"),
                "size_bytes": attachment_info.get("size_bytes"),
                "mime_type": attachment_info.get("mime_type"),
            }
        req_hash = compute_request_hash(req_payload)

        cursor = conn.execute(
            """
            SELECT result_object_id, request_hash
            FROM idempotency_keys
            WHERE namespace = ? AND operation = ? AND key = ?;
            """,
            (namespace, operation, key),
        )
        row = cursor.fetchone()
        if row:
            if row["request_hash"] != req_hash:
                raise IdempotencyConflictError(
                    f"Idempotency key '{key}' was already used with different request parameters"
                )
            # Match: Return previously created capture
            capture_row = conn.execute(
                "SELECT id, origin_namespace, collection_channel, collector, created_at FROM captures WHERE id = ?;",
                (row["result_object_id"],),
            ).fetchone()
            if capture_row:
                res_link = conn.execute(
                    "SELECT resource_id FROM capture_resources WHERE capture_id = ?;",
                    (capture_row["id"],),
                ).fetchone()
                return {
                    "status": "replayed",
                    "capture_id": capture_row["id"],
                    "resource_id": res_link["resource_id"] if res_link else None,
                    "created_at": capture_row["created_at"],
                }

    # 2. Resource Resolution or Creation
    resource_id: str | None = None
    canonical_url: str | None = None

    if input_data.url:
        canonical_url = canonicalize_url(input_data.url)
        url_h = hash_url(canonical_url)
        identity_key = f"url:{url_h}"

        res_row = conn.execute(
            "SELECT id FROM resources WHERE identity_key = ? OR url_hash = ?;",
            (identity_key, url_h),
        ).fetchone()

        if res_row:
            resource_id = res_row["id"]
            conn.execute(
                "UPDATE resources SET updated_at = ? WHERE id = ?;", (now_iso, resource_id)
            )
        else:
            resource_id = generate_id("res")
            conn.execute(
                """
                INSERT INTO resources (
                    id, identity_key, canonical_url, url_hash, primary_form, review_state, created_at, updated_at
                ) VALUES (?, ?, ?, ?, 'other', 'unreviewed', ?, ?);
                """,
                (resource_id, identity_key, canonical_url, url_h, now_iso, now_iso),
            )
    elif attachment_info and not input_data.url:
        # File attachment without URL becomes a document resource
        content_hash = attachment_info["content_hash"]
        identity_key = f"blob:{content_hash}"
        res_row = conn.execute(
            "SELECT id FROM resources WHERE identity_key = ?;",
            (identity_key,),
        ).fetchone()

        if res_row:
            resource_id = res_row["id"]
            conn.execute(
                "UPDATE resources SET updated_at = ? WHERE id = ?;", (now_iso, resource_id)
            )
        else:
            resource_id = generate_id("res")
            conn.execute(
                """
                INSERT INTO resources (
                    id, identity_key, title, primary_form, latest_content_hash, review_state, created_at, updated_at
                ) VALUES (?, ?, ?, 'other', ?, 'unreviewed', ?, ?);
                """,
                (
                    resource_id,
                    identity_key,
                    attachment_info.get("file_name"),
                    content_hash,
                    now_iso,
                    now_iso,
                ),
            )

    # 3. Create Capture Record (Always distinct for capture provenance)
    capture_id = generate_id("cap")
    origin_ns = input_data.origin_namespace
    origin_id = input_data.origin_id or canonical_url or capture_id

    conn.execute(
        """
        INSERT INTO captures (
            id, origin_namespace, origin_id, collection_channel, collector,
            collector_run_id, acquisition_method, retrieved_at, raw_content,
            user_note, review_state, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'unreviewed', ?, ?);
        """,
        (
            capture_id,
            origin_ns,
            origin_id,
            input_data.collection_channel,
            input_data.collector,
            input_data.collector_run_id,
            input_data.acquisition_method,
            now_iso,
            input_data.text,
            input_data.note,
            now_iso,
            now_iso,
        ),
    )

    # 4. Link Capture and Resource
    if resource_id:
        conn.execute(
            """
            INSERT OR IGNORE INTO capture_resources (capture_id, resource_id, relationship_type, created_at)
            VALUES (?, ?, 'primary', ?);
            """,
            (capture_id, resource_id, now_iso),
        )

    # 5. Attach File if provided
    if attachment_info:
        att_id = generate_id("att")
        target_obj = resource_id or capture_id
        target_type = "resource" if resource_id else "capture"
        c_hash = attachment_info["content_hash"]
        rel_blob_path = f"{c_hash[:2]}/{c_hash}"
        conn.execute(
            """
            INSERT INTO attachments (
                id, object_type, object_id, file_name, mime_type, content_hash, size_bytes, blob_path, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                att_id,
                target_type,
                target_obj,
                attachment_info["file_name"],
                attachment_info["mime_type"],
                c_hash,
                attachment_info["size_bytes"],
                rel_blob_path,
                now_iso,
            ),
        )
        if is_pdf_attachment(attachment_info["file_name"], attachment_info["mime_type"]):
            enqueue_pdf_extraction(conn, capture_id, att_id, now_iso=now_iso)

    # 6. Handle Intent
    if input_data.intent:
        target_obj = resource_id or capture_id
        target_type = "resource" if resource_id else "capture"
        add_intent(
            conn,
            target_type,
            target_obj,
            input_data.intent,
            source="human",
            actor=input_data.collector,
        )

    # 7. Index into Search Projection (Preserving Capture Context)
    # Always index the contextual capture so user notes and thoughts are independently searchable
    cap_search_title = input_data.note or (canonical_url or "Captured Item")
    cap_search_body = (input_data.text or "") + (f"\n{canonical_url}" if canonical_url else "")
    index_document(conn, "capture", capture_id, title=cap_search_title, body=cap_search_body)

    # If linked to a resource, also maintain the resource projection
    if resource_id:
        res_search_title = canonical_url or input_data.note or "Resource"
        index_document(
            conn, "resource", resource_id, title=res_search_title, body=input_data.text or ""
        )

    # 7.5 Queue Embed Processing Job for capture note/text (regardless of resource linkage)
    embed_text = (input_data.note or "") + ("\n\n" + input_data.text if input_data.text else "")
    embed_text = embed_text.strip()
    if embed_text:
        emb_job_id = generate_id("job")
        emb_job_key = make_job_key("embed", capture_id)
        conn.execute(
            """
            INSERT INTO processing_jobs (
                id, job_key, capture_id, resource_id, stage, status,
                available_at, attempts, created_at, updated_at
            ) VALUES (?, ?, ?, NULL, 'embed', 'pending', ?, 0, ?, ?)
            ON CONFLICT(job_key) DO UPDATE SET
                status = CASE
                    WHEN processing_jobs.status = 'running' AND (processing_jobs.lease_expires_at IS NULL OR processing_jobs.lease_expires_at > ?)
                    THEN processing_jobs.status
                    ELSE 'pending'
                END,
                lease_owner = CASE
                    WHEN processing_jobs.status = 'running' AND (processing_jobs.lease_expires_at IS NULL OR processing_jobs.lease_expires_at > ?)
                    THEN processing_jobs.lease_owner
                    ELSE NULL
                END,
                lease_expires_at = CASE
                    WHEN processing_jobs.status = 'running' AND (processing_jobs.lease_expires_at IS NULL OR processing_jobs.lease_expires_at > ?)
                    THEN processing_jobs.lease_expires_at
                    ELSE NULL
                END,
                available_at = CASE
                    WHEN processing_jobs.status = 'running' AND (processing_jobs.lease_expires_at IS NULL OR processing_jobs.lease_expires_at > ?)
                    THEN processing_jobs.available_at
                    ELSE excluded.available_at
                END,
                updated_at = excluded.updated_at;
            """,
            (
                emb_job_id,
                emb_job_key,
                capture_id,
                now_iso,
                now_iso,
                now_iso,
                now_iso,
                now_iso,
                now_iso,
                now_iso,
            ),
        )

    # 8. Queue Background Processing Job (if URL)
    if resource_id and canonical_url:
        from edward.services.resource import get_cached_content

        cached = get_cached_content(conn, resource_id)
        if cached and cached.get("clean_text"):
            # Content already cached: queue or update classify job
            cls_job_id = generate_id("job")
            cls_job_key = make_job_key("classify", resource_id)
            conn.execute(
                """
                INSERT INTO processing_jobs (
                    id, job_key, capture_id, resource_id, stage, status,
                    available_at, attempts, created_at, updated_at
                ) VALUES (?, ?, ?, ?, 'classify', 'pending', ?, 0, ?, ?)
                ON CONFLICT(job_key) DO UPDATE SET
                    capture_id = COALESCE(excluded.capture_id, processing_jobs.capture_id),
                    updated_at = excluded.updated_at;
                """,
                (cls_job_id, cls_job_key, capture_id, resource_id, now_iso, now_iso, now_iso),
            )
        else:
            job_id = generate_id("job")
            job_key = make_job_key("fetch", resource_id)
            conn.execute(
                """
                INSERT INTO processing_jobs (
                    id, job_key, capture_id, resource_id, stage, status,
                    available_at, attempts, created_at, updated_at
                ) VALUES (?, ?, ?, ?, 'resource-fetch', 'pending', ?, 0, ?, ?)
                ON CONFLICT(job_key) DO UPDATE SET
                    capture_id = COALESCE(excluded.capture_id, processing_jobs.capture_id),
                    updated_at = excluded.updated_at;
                """,
                (job_id, job_key, capture_id, resource_id, now_iso, now_iso, now_iso),
            )

    # 9. Record Idempotency Key (if provided)
    if input_data.idempotency_key:
        idemp_id = generate_id("idk")
        conn.execute(
            """
            INSERT INTO idempotency_keys (id, namespace, operation, key, result_object_id, request_hash, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?);
            """,
            (idemp_id, "capture", "add", input_data.idempotency_key, capture_id, req_hash, now_iso),
        )

    # 10. Audit Event
    record_audit_event(
        conn,
        event_type="capture.created",
        object_type="capture",
        object_id=capture_id,
        actor=input_data.collector,
        payload={
            "url": canonical_url,
            "resource_id": resource_id,
            "has_note": bool(input_data.note),
            "has_attachment": bool(attachment_info),
        },
    )

    return {
        "status": "created",
        "capture_id": capture_id,
        "resource_id": resource_id,
        "url": canonical_url,
        "created_at": now_iso,
    }
