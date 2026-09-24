"""Background and offline job processing engine with atomic leases, backoff, and crash recovery."""

import datetime
import hashlib
import json
import logging
import re
import sqlite3
import uuid
from contextlib import contextmanager
from pathlib import PurePosixPath
from typing import Any
from urllib.parse import unquote, urlparse

from edward.blobs import BlobStore
from edward.models import generate_id, make_job_key
from edward.services.classification import (
    classify_target,
    load_classification_target,
    persist_classification_result,
    run_classification_pipeline,
)
from edward.services.extract import ExtractionResult, clean_html_simple, extract_content
from edward.services.network import NetworkError, safe_fetch_url
from edward.services.ocr import OCR_STAGE, format_ocr_passage, ocr_image_bytes
from edward.services.pdf import PDF_STAGE, extract_pdf_text
from edward.services.resource import (
    get_cached_content,
    save_source_snapshot,
    store_resource_content,
)
from edward.services.search import index_document
from edward.services.shortlinks import (
    is_shortlink,
    looks_like_shortlink_interstitial,
    resolve_shortlink,
)
from edward.services.subprocess_runner import sanitize_error_message

logger = logging.getLogger(__name__)


def _extract_via_resolved_shortlink(canonical_url: str) -> ExtractionResult | None:
    """Extract the real destination behind a shortlink.

    The stored snapshot for a t.co URL is X's interstitial, so extracting from it
    yields chrome. Resolving the shortlink and extracting the destination is what
    recovers the article. Returns None when the destination cannot be resolved or
    cannot be extracted, so the caller falls back to the snapshot.

    The destination URL is deliberately not persisted as the resource's identity:
    the bookmark is the source, and swapping identity mid-lifecycle would orphan
    annotations, intents, and projects attached to the row.
    """
    resolved_url = resolve_shortlink(canonical_url)
    if resolved_url is None:
        return None
    try:
        result = extract_content(resolved_url)
    except Exception as exc:
        logger.debug("extraction failed for resolved link %s: %s", resolved_url, exc)
        return None
    if not result.clean_text:
        return None
    note = f"resolved {canonical_url} -> {resolved_url}"
    if result.error:
        note = f"{note}; {result.error}"
    result.error = note
    return result


def _append_ocr_passage(
    conn: sqlite3.Connection,
    resource_id: str,
    passage: str,
    now_iso: str,
) -> None:
    """Attach machine-read image text to the post that carried the image.

    The text is appended to the resource's stored content so it becomes
    searchable and chunked like any other passage, but it always keeps the
    OCR provenance marker so a reader can tell it apart from the author's words.

    Existing content is left intact: if the post already had text, the image
    text is appended after it. If it had none, this is what makes the post
    searchable at all.
    """
    existing = conn.execute(
        """
        SELECT id, clean_text FROM resource_contents
        WHERE resource_id = ? ORDER BY created_at DESC LIMIT 1;
        """,
        (resource_id,),
    ).fetchone()

    # Idempotency: the passage already present means this job already ran. The
    # check is by exact passage text, not by marker count, because one post can
    # legitimately carry several images and therefore several passages.
    if existing and passage in (existing["clean_text"] or ""):
        return

    # An identical passage from a different attachment (same image attached
    # twice, or a re-run that lost its lease) must not be appended again.
    already_appended = conn.execute(
        """
        SELECT 1 FROM resource_contents
        WHERE resource_id = ? AND instr(clean_text, ?) > 0
        LIMIT 1;
        """,
        (resource_id, passage),
    ).fetchone()
    if already_appended:
        return

    if existing and (existing["clean_text"] or "").strip():
        merged = f"{existing['clean_text'].rstrip()}\n\n{passage}"
    else:
        merged = passage

    from edward.services.embed import store_resource_chunks

    content_id, _ = store_resource_content(
        conn,
        resource_id,
        clean_text=merged,
        extractor="tesseract-ocr",
        extractor_version="ocr-v1",
    )
    store_resource_chunks(conn, resource_id, content_id, merged)


def _get_transaction(db_or_conn: Any):
    """Obtain a transaction context manager from either a Database instance or a raw connection."""
    if hasattr(db_or_conn, "transaction") and callable(db_or_conn.transaction):
        return db_or_conn.transaction()

    @contextmanager
    def _conn_tx():
        try:
            yield db_or_conn
            if hasattr(db_or_conn, "commit"):
                db_or_conn.commit()
        except Exception:
            if hasattr(db_or_conn, "rollback"):
                db_or_conn.rollback()
            raise

    return _conn_tx()


def claim_job(
    conn: sqlite3.Connection,
    worker_id: str,
    lease_seconds: int = 60,
    stage: str | None = None,
    capture_id: str | None = None,
) -> dict[str, Any] | None:
    """Atomically recover expired leases and claim the next available pending job."""
    now_iso = datetime.datetime.now(datetime.UTC).isoformat()
    lease_expires = (
        datetime.datetime.now(datetime.UTC) + datetime.timedelta(seconds=lease_seconds)
    ).isoformat()

    # 1. Crash recovery: reclaim running jobs whose lease has expired
    conn.execute(
        """
        UPDATE processing_jobs
        SET status = 'pending',
            lease_owner = NULL,
            lease_expires_at = NULL,
            updated_at = ?
        WHERE status = 'running' AND lease_expires_at < ?;
        """,
        (now_iso, now_iso),
    )

    # 2. Claim the next available job.
    supported_stages = (
        "resource-fetch",
        "extract",
        "classify",
        "finding-extraction",
        "embed",
        PDF_STAGE,
        OCR_STAGE,
    )
    if stage and stage not in supported_stages:
        return None

    # The claim must happen in the SAME statement that selects the row.
    #
    # A separate SELECT followed by a guarded UPDATE is a TOCTOU race: two
    # worker processes can both read the same pending row, both pass the
    # guard, and both run the job — which is how concurrent OCR runs produced
    # duplicate chunks. Selection and lease acquisition must be one atomic
    # statement so exactly one worker can win.
    claim_sql = f"""
        UPDATE processing_jobs
        SET status = 'running',
            started_at = ?,
            lease_owner = ?,
            lease_expires_at = ?,
            updated_at = ?
        WHERE id = (
            SELECT id FROM processing_jobs
            WHERE status = 'pending' AND available_at <= ?
            {"AND stage = ?" if stage else f"AND stage IN ({', '.join('?' for _ in supported_stages)})"}
            {"AND capture_id = ?" if capture_id else ""}
            ORDER BY available_at ASC, created_at ASC
            LIMIT 1
        )
        AND status = 'pending'
        RETURNING id, job_key, capture_id, resource_id, stage, depends_on,
                  status, attempts, max_attempts, available_at;
    """

    claim_params: list[Any] = [now_iso, worker_id, lease_expires, now_iso, now_iso]
    if stage:
        claim_params.append(stage)
    else:
        claim_params.extend(supported_stages)
    if capture_id:
        claim_params.append(capture_id)

    cursor = conn.execute(claim_sql, claim_params)
    claimed_row = cursor.fetchone()
    if not claimed_row:
        return None

    claimed = dict(claimed_row)
    claimed["lease_owner"] = worker_id
    return claimed


def _load_job_context(conn: sqlite3.Connection, job: dict[str, Any]) -> dict[str, Any]:
    """Read metadata necessary to perform external work outside transaction boundaries."""
    stage = job["stage"]
    resource_id = job.get("resource_id")
    capture_id = job.get("capture_id")
    context: dict[str, Any] = {"stage": stage}

    if stage == "resource-fetch":
        if not resource_id:
            raise ValueError("Stage 'resource-fetch' requires resource_id")
        res_row = conn.execute(
            "SELECT canonical_url FROM resources WHERE id = ?;", (resource_id,)
        ).fetchone()
        if not res_row or not res_row["canonical_url"]:
            raise ValueError(f"Resource {resource_id} has no canonical_url")
        context["canonical_url"] = res_row["canonical_url"]

    elif stage == "extract":
        if not resource_id:
            raise ValueError("Stage 'extract' requires resource_id")
        res_row = conn.execute(
            "SELECT canonical_url FROM resources WHERE id = ?;", (resource_id,)
        ).fetchone()
        context["canonical_url"] = res_row["canonical_url"] if res_row else ""

        # Check existing snapshot
        snp_row = conn.execute(
            """
            SELECT content_hash, headers_json FROM source_snapshots
            WHERE resource_id = ?
            ORDER BY created_at DESC LIMIT 1;
            """,
            (resource_id,),
        ).fetchone()
        context["snapshot_content_hash"] = snp_row["content_hash"] if snp_row else None
        snapshot_headers = json.loads(snp_row["headers_json"] or "{}") if snp_row else {}
        context["snapshot_content_type"] = snapshot_headers.get("content-type", "")

        # Check resource cache
        snap_hash = context["snapshot_content_hash"]
        cached = get_cached_content(conn, resource_id)
        context["cached_hit"] = False
        if cached and cached.get("clean_text"):
            if snap_hash:
                ext_match = conn.execute(
                    """
                    SELECT 1 FROM processing_jobs
                    WHERE resource_id = ? AND stage = 'extract' AND status = 'completed' AND input_hash = ?
                    LIMIT 1;
                    """,
                    (resource_id, snap_hash),
                ).fetchone()
                if ext_match or cached.get("content_hash") == snap_hash:
                    context["cached_hit"] = True
                    context["cached_content"] = cached
            else:
                context["cached_hit"] = True
                context["cached_content"] = cached

    elif stage == PDF_STAGE:
        if not capture_id:
            raise ValueError("Stage 'attachment-extract' requires capture_id")
        attachment_id = job["job_key"].removeprefix(f"{PDF_STAGE}:")
        attachment = conn.execute(
            """
            SELECT a.id, a.file_name, a.mime_type, a.content_hash, a.size_bytes
            FROM attachments a
            WHERE a.id = ? AND (
                (a.object_type = 'capture' AND a.object_id = ?)
                OR (a.object_type = 'resource' AND EXISTS (
                    SELECT 1 FROM capture_resources cr
                    WHERE cr.capture_id = ? AND cr.resource_id = a.object_id
                ))
            );
            """,
            (attachment_id, capture_id, capture_id),
        ).fetchone()
        if not attachment:
            raise ValueError(f"PDF attachment {attachment_id} not found for capture {capture_id}")
        context["attachment"] = dict(attachment)

    elif stage == OCR_STAGE:
        if not capture_id:
            raise ValueError("Stage 'attachment-ocr' requires capture_id")
        attachment_id = job["job_key"].removeprefix(f"{OCR_STAGE}:")
        attachment = conn.execute(
            """
            SELECT a.id, a.file_name, a.mime_type, a.content_hash, a.size_bytes,
                   a.object_type, a.object_id
            FROM attachments a
            WHERE a.id = ? AND (
                (a.object_type = 'capture' AND a.object_id = ?)
                OR (a.object_type = 'resource' AND EXISTS (
                    SELECT 1 FROM capture_resources cr
                    WHERE cr.capture_id = ? AND cr.resource_id = a.object_id
                ))
            );
            """,
            (attachment_id, capture_id, capture_id),
        ).fetchone()
        if not attachment:
            raise ValueError(f"Image attachment {attachment_id} not found for capture {capture_id}")
        context["attachment"] = dict(attachment)
        # OCR text belongs to the post that carried the image.
        if attachment["object_type"] == "resource":
            context["parent_resource_id"] = attachment["object_id"]
        else:
            parent = conn.execute(
                """
                SELECT resource_id FROM capture_resources
                WHERE capture_id = ? AND relationship_type = 'primary'
                ORDER BY created_at ASC LIMIT 1;
                """,
                (capture_id,),
            ).fetchone()
            context["parent_resource_id"] = parent["resource_id"] if parent else None

    elif stage == "classify":
        target_id = resource_id or capture_id
        target_type = "resource" if resource_id else "capture"
        if not target_id:
            raise ValueError("Stage 'classify' requires resource_id or capture_id")
        context["target_id"] = target_id
        context["target_type"] = target_type
        target_data = load_classification_target(conn, target_type, target_id)
        if not target_data:
            raise ValueError(f"Classification target {target_type} '{target_id}' not found")
        context["target_data"] = target_data
        context["input_hash"] = target_data["content_hash"]

    elif stage in ("finding-extraction", "embed"):
        if not resource_id:
            if stage == "embed" and capture_id:
                cap_row = conn.execute(
                    "SELECT raw_content, user_note, origin_namespace FROM captures WHERE id = ?;",
                    (capture_id,),
                ).fetchone()
                if not cap_row:
                    raise ValueError(f"Capture {capture_id} not found")
                cap_text = (
                    (cap_row["user_note"] or "") + ("\n\n" + (cap_row["raw_content"] or ""))
                ).strip()
                context["text"] = cap_text
                context["content_hash"] = hashlib.sha256(cap_text.encode("utf-8")).hexdigest()
                context["input_hash"] = context["content_hash"]
                from edward.services.privacy import classify_content_data_class

                context["data_class"] = classify_content_data_class(
                    origin_namespace=cap_row["origin_namespace"] or "manual", canonical_url=None
                )
                return context
            else:
                raise ValueError(f"Stage '{stage}' requires resource_id")

        content_row = conn.execute(
            "SELECT clean_text, summary, content_hash FROM resource_contents WHERE resource_id = ? ORDER BY created_at DESC LIMIT 1;",
            (resource_id,),
        ).fetchone()
        if not content_row and stage == "finding-extraction":
            raise ValueError(
                f"Resource {resource_id} has no extracted content for finding extraction"
            )
        context["text"] = (
            (content_row["clean_text"] or content_row["summary"] or "").strip()
            if content_row
            else ""
        )
        context["content_hash"] = content_row["content_hash"] if content_row else None
        context["input_hash"] = content_row["content_hash"] if content_row else None

        # Authoritative origin provenance for privacy gating
        cap_link = conn.execute(
            """
            SELECT c.origin_namespace
            FROM capture_resources cr
            JOIN captures c ON cr.capture_id = c.id
            WHERE cr.resource_id = ?
            ORDER BY cr.created_at DESC LIMIT 1;
            """,
            (resource_id,),
        ).fetchone()
        res_row = conn.execute(
            "SELECT canonical_url FROM resources WHERE id = ?;", (resource_id,)
        ).fetchone()
        from edward.services.privacy import classify_content_data_class

        origin_ns = cap_link["origin_namespace"] if cap_link else "web"
        can_url = res_row["canonical_url"] if res_row else None
        context["data_class"] = classify_content_data_class(
            origin_namespace=origin_ns, canonical_url=can_url
        )

        if stage == "embed":
            f_rows = conn.execute(
                "SELECT id, statement FROM findings WHERE resource_id = ? AND is_deleted = 0 AND review_state != 'superseded' ORDER BY id ASC;",
                (resource_id,),
            ).fetchall()
            context["findings"] = [
                {"id": r["id"], "statement": r["statement"]} for r in f_rows if r["statement"]
            ]
            from edward.services.embed import (
                compute_embedding_input_hash,
                deserialize_vector,
            )

            context["input_hash"] = compute_embedding_input_hash(
                conn, resource_id=resource_id, capture_id=capture_id
            )

            # Pre-load existing embeddings for unchanged fallback / exact deduplication
            existing_emb_map: dict[tuple[str, str], dict[str, Any]] = {}
            from edward.services.embed import get_configured_embedding_model

            target_model = get_configured_embedding_model()
            target_ids = [resource_id] if resource_id else ([capture_id] if capture_id else [])
            target_ids.extend([f["id"] for f in context["findings"]])
            if resource_id:
                chk_rows = conn.execute(
                    "SELECT id FROM resource_chunks WHERE resource_id = ?;", (resource_id,)
                ).fetchall()
                target_ids.extend([cr["id"] for cr in chk_rows])

            if target_ids:
                placeholders = ",".join("?" for _ in target_ids)
                existing_rows = conn.execute(
                    f"SELECT object_type, object_id, model, input_hash, embedding_blob FROM embeddings WHERE object_id IN ({placeholders});",
                    target_ids,
                ).fetchall()
                for er in existing_rows:
                    if er["model"] != target_model:
                        continue
                    cached_value = {
                        "vector": deserialize_vector(er["embedding_blob"]),
                        "model": er["model"],
                    }
                    keys = [(er["object_id"], er["input_hash"])]
                    if er["object_type"] == "resource_chunk":
                        keys.append(("resource_chunk", er["input_hash"]))
                    for k in keys:
                        existing_emb_map[k] = cached_value
            context["existing_embeddings"] = existing_emb_map

    return context


def _perform_job_work(
    blob_store: BlobStore,
    job: dict[str, Any],
    context: dict[str, Any],
) -> dict[str, Any]:
    """Execute long-running network, subprocess, or external computation outside database transactions."""
    stage = job["stage"]

    if stage == "resource-fetch":
        canonical_url = context["canonical_url"]
        fetch_result = safe_fetch_url(canonical_url)
        if fetch_result.status_code >= 400:
            raise NetworkError(
                f"HTTP fetch returned error status {fetch_result.status_code} for '{canonical_url}'"
            )
        return {"fetch_result": fetch_result}

    elif stage == "extract":
        if context.get("cached_hit"):
            return {"cached": True, "cached_content": context.get("cached_content")}

        canonical_url = context["canonical_url"]
        raw_bytes = None
        snap_hash = context.get("snapshot_content_hash")
        if snap_hash:
            try:
                raw_bytes = blob_store.read_bytes(snap_hash)
            except Exception:
                pass

        content_type = context.get("snapshot_content_type", "").split(";", 1)[0].lower()
        if content_type == "application/pdf" and raw_bytes is not None:
            file_stem = PurePosixPath(urlparse(canonical_url).path).stem
            ext_result = ExtractionResult(
                status="completed",
                clean_text=extract_pdf_text(raw_bytes),
                title=unquote(file_stem).replace("-", " ").replace("_", " ").strip() or None,
                extractor="pypdf",
            )
            return {"status": "completed", "ext_result": ext_result}
        if content_type and not (
            content_type.startswith("text/")
            or content_type in ("application/xhtml+xml", "application/xml")
        ):
            return {"status": "completed", "ext_result": None}

        raw_html = raw_bytes.decode("utf-8", errors="replace") if raw_bytes else None
        # A shortlink snapshot can be the shortener's interstitial rather than
        # the destination page — reading it directly is what produced
        # page-sized blobs of nav links and login prompts. Only in that case is
        # the link resolved and the real page fetched; a snapshot that is
        # already the article is left alone.
        if raw_html and is_shortlink(canonical_url) and looks_like_shortlink_interstitial(raw_html):
            resolved_extraction = _extract_via_resolved_shortlink(canonical_url)
            if resolved_extraction is not None:
                return {"status": "completed", "ext_result": resolved_extraction}
            title, clean_text = clean_html_simple(raw_html)
            return {
                "status": "completed",
                "ext_result": ExtractionResult(
                    status="completed",
                    clean_text=clean_text,
                    title=title,
                    extractor="local-fallback",
                    error=(
                        "t.co snapshot is an X interstitial and the destination "
                        "could not be resolved or fetched"
                    ),
                ),
            }

        ext_result = extract_content(canonical_url, raw_html=raw_html)
        if ext_result.status == "failed":
            raise RuntimeError(ext_result.error or f"Extraction failed for '{canonical_url}'")
        elif ext_result.status == "pending":
            return {"status": "pending", "ext_result": ext_result}
        else:
            return {"status": "completed", "ext_result": ext_result}

    elif stage == PDF_STAGE:
        attachment = context["attachment"]
        pdf_bytes = blob_store.read_bytes(attachment["content_hash"])
        text = extract_pdf_text(pdf_bytes)
        return {
            "attachment_id": attachment["id"],
            "file_name": attachment["file_name"],
            "text": text,
            "input_hash": attachment["content_hash"],
        }

    elif stage == OCR_STAGE:
        attachment = context["attachment"]
        image_bytes = blob_store.read_bytes(attachment["content_hash"])
        text = ocr_image_bytes(image_bytes)
        return {
            "attachment_id": attachment["id"],
            "file_name": attachment["file_name"],
            "text": text,
            "parent_resource_id": context.get("parent_resource_id"),
            "input_hash": attachment["content_hash"],
        }
    elif stage == "classify":
        target_data = context["target_data"]
        form, result = classify_target(
            text=target_data["text"],
            url=target_data.get("url"),
            record_id=context["target_id"],
            object_type=context["target_type"],
            provider=context.get("provider"),
            metadata=target_data.get("metadata"),
        )
        return {
            "detected_form": form,
            "result": result,
            "text": target_data["text"],
            "input_hash": context.get("input_hash"),
        }

    elif stage == "finding-extraction":
        from edward.services.findings import extract_heuristically

        # Deterministic by design. This stage does not construct a model client:
        # a calling agent that wants model-based extraction submits findings through
        # import-research, or supplies its own client to extract_findings_for_resource.
        text = context.get("text", "")
        payload = extract_heuristically(text)

        return {
            "status": "completed",
            "payload": payload,
            "content_hash": context.get("content_hash"),
            "input_hash": context.get("input_hash"),
        }

    elif stage == "embed":
        from edward.services.embed import (
            chunk_markdown_text,
            generate_embedding,
            get_configured_embedding_model,
        )

        text = context.get("text", "")
        target_model = get_configured_embedding_model()

        # 1. Structural text chunking outside transaction
        chunks = chunk_markdown_text(text) if text else []
        existing_emb_map = context.get("existing_embeddings") or {}

        # 2. Compute embeddings for chunks outside transaction
        chunk_embeddings: list[dict[str, Any]] = []
        for c in chunks:
            c_text = c.get("text", "")
            if c_text:
                c_hash = hashlib.sha256(c_text.encode("utf-8")).hexdigest()
                cached = existing_emb_map.get(("resource_chunk", c_hash))
                if cached and cached["model"] == target_model:
                    vec, actual_m = cached["vector"], cached["model"]
                else:
                    vec, actual_m = generate_embedding(c_text, model=target_model)
                chunk_embeddings.append(
                    {
                        "chunk": c,
                        "vector": vec,
                        "model": actual_m,
                    }
                )

        # 3. Compute top-level resource embedding outside transaction
        resource_embedding: dict[str, Any] | None = None
        if text and job.get("resource_id"):
            doc_text = text[:1500]
            doc_hash = hashlib.sha256(doc_text.encode("utf-8")).hexdigest()
            cached = existing_emb_map.get((job.get("resource_id"), doc_hash))
            if cached and cached["model"] == target_model:
                res_vec, res_m = cached["vector"], cached["model"]
            else:
                res_vec, res_m = generate_embedding(doc_text, model=target_model)
            resource_embedding = {
                "vector": res_vec,
                "model": res_m,
                "text": doc_text,
            }

        # 4. Compute finding embeddings outside the write transaction using the same model
        finding_embeddings: list[dict[str, Any]] = []
        for f in context.get("findings", []):
            f_text = f["statement"]
            if f_text:
                f_hash = hashlib.sha256(f_text.encode("utf-8")).hexdigest()
                cached = existing_emb_map.get((f["id"], f_hash))
                if cached and cached["model"] == target_model:
                    f_vec, f_m = cached["vector"], cached["model"]
                else:
                    f_vec, f_m = generate_embedding(f_text, model=target_model)
                finding_embeddings.append(
                    {
                        "finding_id": f["id"],
                        "vector": f_vec,
                        "model": f_m,
                        "text": f_text,
                    }
                )

        # 5. If standalone capture, compute capture embedding outside transaction
        capture_embedding: dict[str, Any] | None = None
        if not job.get("resource_id") and job.get("capture_id") and text:
            doc_text = text[:1500]
            doc_hash = hashlib.sha256(doc_text.encode("utf-8")).hexdigest()
            cached = existing_emb_map.get((job.get("capture_id"), doc_hash))
            if cached and cached["model"] == target_model:
                cap_vec, cap_m = cached["vector"], cached["model"]
            else:
                cap_vec, cap_m = generate_embedding(doc_text, model=target_model)
            capture_embedding = {
                "capture_id": job.get("capture_id"),
                "vector": cap_vec,
                "model": cap_m,
                "text": doc_text,
            }

        return {
            "status": "completed",
            "resource_id": job.get("resource_id"),
            "capture_id": job.get("capture_id"),
            "chunks": chunks,
            "chunk_embeddings": chunk_embeddings,
            "resource_embedding": resource_embedding,
            "finding_embeddings": finding_embeddings,
            "capture_embedding": capture_embedding,
            "input_hash": context.get("input_hash"),
        }

    else:
        raise ValueError(f"Unknown processing stage '{stage}'")


def _persist_job_result(
    conn: sqlite3.Connection,
    blob_store: BlobStore,
    job: dict[str, Any],
    work_result: dict[str, Any] | None,
    job_error: Exception | None,
    worker_id: str | None = None,
    context: dict[str, Any] | None = None,
) -> str:
    """Commit results of work and schedule downstream jobs in a short transaction, or record retry/failure.

    Returns one of: 'completed', 'pending', 'failed', 'lost-lease'.
    """
    job_id = job["id"]
    resource_id = job.get("resource_id")
    capture_id = job.get("capture_id")
    stage = job["stage"]
    now_iso = datetime.datetime.now(datetime.UTC).isoformat()
    owner = worker_id or job.get("lease_owner")

    # Fetch authoritative capture_id from the database row to capture any updates that occurred while running
    db_job_row = conn.execute(
        "SELECT capture_id FROM processing_jobs WHERE id = ?;",
        (job_id,),
    ).fetchone()
    if db_job_row and db_job_row["capture_id"]:
        capture_id = db_job_row["capture_id"]
        job["capture_id"] = capture_id
    elif not capture_id and resource_id:
        cap_row = conn.execute(
            "SELECT capture_id FROM capture_resources WHERE resource_id = ? ORDER BY created_at DESC LIMIT 1;",
            (resource_id,),
        ).fetchone()
        if cap_row:
            capture_id = cap_row["capture_id"]
            job["capture_id"] = capture_id

    if job_error is not None:
        attempts = job["attempts"] + 1
        max_attempts = job.get("max_attempts", 3)
        err_str = sanitize_error_message(str(job_error))

        if attempts >= max_attempts:
            if owner:
                cursor = conn.execute(
                    """
                    UPDATE processing_jobs
                    SET status = 'failed',
                        attempts = ?,
                        last_error = ?,
                        completed_at = ?,
                        lease_owner = NULL,
                        lease_expires_at = NULL,
                        updated_at = ?
                    WHERE id = ? AND status = 'running' AND lease_owner = ?
                      AND (lease_expires_at IS NULL OR lease_expires_at > ?);
                    """,
                    (attempts, err_str, now_iso, now_iso, job_id, owner, now_iso),
                )
            else:
                cursor = conn.execute(
                    """
                    UPDATE processing_jobs
                    SET status = 'failed',
                        attempts = ?,
                        last_error = ?,
                        completed_at = ?,
                        lease_owner = NULL,
                        lease_expires_at = NULL,
                        updated_at = ?
                    WHERE id = ? AND status = 'running';
                    """,
                    (attempts, err_str, now_iso, now_iso, job_id),
                )
        else:
            backoff_sec = 5 * (2**attempts)
            next_avail = (
                datetime.datetime.now(datetime.UTC) + datetime.timedelta(seconds=backoff_sec)
            ).isoformat()
            if owner:
                cursor = conn.execute(
                    """
                    UPDATE processing_jobs
                    SET status = 'pending',
                        attempts = ?,
                        last_error = ?,
                        available_at = ?,
                        lease_owner = NULL,
                        lease_expires_at = NULL,
                        updated_at = ?
                    WHERE id = ? AND status = 'running' AND lease_owner = ?
                      AND (lease_expires_at IS NULL OR lease_expires_at > ?);
                    """,
                    (attempts, err_str, next_avail, now_iso, job_id, owner, now_iso),
                )
            else:
                cursor = conn.execute(
                    """
                    UPDATE processing_jobs
                    SET status = 'pending',
                        attempts = ?,
                        last_error = ?,
                        available_at = ?,
                        lease_owner = NULL,
                        lease_expires_at = NULL,
                        updated_at = ?
                    WHERE id = ? AND status = 'running';
                    """,
                    (attempts, err_str, next_avail, now_iso, job_id),
                )
        if cursor.rowcount == 0:
            conn.rollback()
            return "lost-lease"
        return "failed"

    assert work_result is not None

    if stage == PDF_STAGE:
        if owner:
            cursor = conn.execute(
                """
                UPDATE processing_jobs
                SET status = 'completed', input_hash = ?, completed_at = ?, updated_at = ?,
                    lease_owner = NULL, lease_expires_at = NULL
                WHERE id = ? AND status = 'running' AND lease_owner = ?
                  AND (lease_expires_at IS NULL OR lease_expires_at > ?);
                """,
                (work_result["input_hash"], now_iso, now_iso, job_id, owner, now_iso),
            )
        else:
            cursor = conn.execute(
                """
                UPDATE processing_jobs
                SET status = 'completed', input_hash = ?, completed_at = ?, updated_at = ?,
                    lease_owner = NULL, lease_expires_at = NULL
                WHERE id = ? AND status = 'running';
                """,
                (work_result["input_hash"], now_iso, now_iso, job_id),
            )
        if cursor.rowcount == 0:
            conn.rollback()
            return "lost-lease"

        capture = conn.execute(
            "SELECT user_note, raw_content FROM captures WHERE id = ?;", (capture_id,)
        ).fetchone()
        if not capture:
            raise ValueError(f"Capture {capture_id} not found while saving extracted PDF text")
        raw_content = (capture["raw_content"] or "").rstrip()
        pdf_content = f"PDF attachment: {work_result['file_name']}\n{work_result['text']}"
        raw_content = f"{raw_content}\n\n{pdf_content}" if raw_content else pdf_content
        conn.execute(
            "UPDATE captures SET raw_content = ?, updated_at = ? WHERE id = ?;",
            (raw_content, now_iso, capture_id),
        )
        index_document(
            conn,
            "capture",
            capture_id,
            title=capture["user_note"] or "Captured Item",
            body=(f"{capture['user_note'] or ''}\n\n{raw_content}").strip(),
        )
        embed_job_id = generate_id("job")
        embed_job_key = make_job_key("embed", capture_id)
        conn.execute(
            """
            INSERT INTO processing_jobs (
                id, job_key, capture_id, stage, status, available_at, attempts, created_at, updated_at
            ) VALUES (?, ?, ?, 'embed', 'pending', ?, 0, ?, ?)
            ON CONFLICT(job_key) DO UPDATE SET
                status = CASE WHEN processing_jobs.status IN ('completed', 'failed')
                              THEN 'pending' ELSE processing_jobs.status END,
                available_at = CASE WHEN processing_jobs.status IN ('completed', 'failed')
                                    THEN excluded.available_at ELSE processing_jobs.available_at END,
                attempts = CASE WHEN processing_jobs.status IN ('completed', 'failed')
                                THEN 0 ELSE processing_jobs.attempts END,
                input_hash = CASE WHEN processing_jobs.status IN ('completed', 'failed')
                                  THEN NULL ELSE processing_jobs.input_hash END,
                last_error = CASE WHEN processing_jobs.status IN ('completed', 'failed')
                                  THEN NULL ELSE processing_jobs.last_error END,
                completed_at = CASE WHEN processing_jobs.status IN ('completed', 'failed')
                                    THEN NULL ELSE processing_jobs.completed_at END,
                updated_at = excluded.updated_at;
            """,
            (embed_job_id, embed_job_key, capture_id, now_iso, now_iso, now_iso),
        )
        return "completed"

    elif stage == OCR_STAGE:
        if owner:
            cursor = conn.execute(
                """
                UPDATE processing_jobs
                SET status = 'completed', input_hash = ?, completed_at = ?, updated_at = ?,
                    lease_owner = NULL, lease_expires_at = NULL
                WHERE id = ? AND status = 'running' AND lease_owner = ?
                  AND (lease_expires_at IS NULL OR lease_expires_at > ?);
                """,
                (work_result["input_hash"], now_iso, now_iso, job_id, owner, now_iso),
            )
        else:
            cursor = conn.execute(
                """
                UPDATE processing_jobs
                SET status = 'completed', input_hash = ?, completed_at = ?, updated_at = ?,
                    lease_owner = NULL, lease_expires_at = NULL
                WHERE id = ? AND status = 'running';
                """,
                (work_result["input_hash"], now_iso, now_iso, job_id),
            )
        if cursor.rowcount == 0:
            conn.rollback()
            return "lost-lease"

        # OCR text joins the post that carried the image, under a provenance
        # marker, so image-read words are searchable but never mistaken for
        # something the author typed.
        parent_resource_id = work_result.get("parent_resource_id")
        if parent_resource_id:
            passage = format_ocr_passage(work_result["file_name"], work_result["text"])
            _append_ocr_passage(conn, parent_resource_id, passage, now_iso)
        return "completed"

    if stage == "resource-fetch":
        # 1. Atomic lease fencing update FIRST: ensure worker still owns active lease before committing any side effects
        if owner:
            cursor = conn.execute(
                """
                UPDATE processing_jobs
                SET status = 'completed', completed_at = ?, updated_at = ?, lease_owner = NULL, lease_expires_at = NULL
                WHERE id = ? AND status = 'running' AND lease_owner = ?
                  AND (lease_expires_at IS NULL OR lease_expires_at > ?);
                """,
                (now_iso, now_iso, job_id, owner, now_iso),
            )
        else:
            cursor = conn.execute(
                """
                UPDATE processing_jobs
                SET status = 'completed', completed_at = ?, updated_at = ?, lease_owner = NULL, lease_expires_at = NULL
                WHERE id = ? AND status = 'running';
                """,
                (now_iso, now_iso, job_id),
            )

        if cursor.rowcount == 0:
            conn.rollback()
            return "lost-lease"

        # 2. Side effects execute ONLY after successful fenced ownership update
        fetch_result = work_result["fetch_result"]
        _, snap_hash = save_source_snapshot(
            conn,
            blob_store,
            resource_id,
            fetch_result.body,
            fetch_result.headers,
            capture_id=capture_id,
        )

        # Check if snapshot hash unchanged and content already extracted
        cached = get_cached_content(conn, resource_id)
        already_extracted = conn.execute(
            "SELECT 1 FROM processing_jobs WHERE resource_id = ? AND stage = 'extract' AND status = 'completed' AND input_hash = ? LIMIT 1;",
            (resource_id, snap_hash),
        ).fetchone()

        if cached and cached.get("clean_text") and already_extracted:
            # Snapshot unchanged and content already extracted: skip extract and schedule classify directly
            cls_job_id = generate_id("job")
            cls_job_key = make_job_key("classify", resource_id)
            conn.execute(
                """
                INSERT INTO processing_jobs (
                    id, job_key, capture_id, resource_id, stage, depends_on, status,
                    available_at, attempts, created_at, updated_at
                ) VALUES (?, ?, ?, ?, 'classify', ?, 'pending', ?, 0, ?, ?)
                ON CONFLICT(job_key) DO UPDATE SET
                    capture_id = COALESCE(excluded.capture_id, processing_jobs.capture_id),
                    status = CASE
                        WHEN processing_jobs.status IN ('completed', 'failed')
                             OR (processing_jobs.status = 'pending' AND processing_jobs.lease_owner IS NULL)
                        THEN 'pending'
                        ELSE processing_jobs.status
                    END,
                    depends_on = CASE
                        WHEN processing_jobs.status IN ('completed', 'failed')
                             OR (processing_jobs.status = 'pending' AND processing_jobs.lease_owner IS NULL)
                        THEN excluded.depends_on
                        ELSE processing_jobs.depends_on
                    END,
                    available_at = CASE
                        WHEN processing_jobs.status IN ('completed', 'failed')
                             OR (processing_jobs.status = 'pending' AND processing_jobs.lease_owner IS NULL)
                        THEN excluded.available_at
                        ELSE processing_jobs.available_at
                    END,
                    attempts = CASE
                        WHEN processing_jobs.status IN ('completed', 'failed')
                             OR (processing_jobs.status = 'pending' AND processing_jobs.lease_owner IS NULL)
                        THEN 0
                        ELSE processing_jobs.attempts
                    END,
                    last_error = CASE
                        WHEN processing_jobs.status IN ('completed', 'failed')
                             OR (processing_jobs.status = 'pending' AND processing_jobs.lease_owner IS NULL)
                        THEN NULL
                        ELSE processing_jobs.last_error
                    END,
                    completed_at = CASE
                        WHEN processing_jobs.status IN ('completed', 'failed')
                             OR (processing_jobs.status = 'pending' AND processing_jobs.lease_owner IS NULL)
                        THEN NULL
                        ELSE processing_jobs.completed_at
                    END,
                    started_at = CASE
                        WHEN processing_jobs.status IN ('completed', 'failed')
                             OR (processing_jobs.status = 'pending' AND processing_jobs.lease_owner IS NULL)
                        THEN NULL
                        ELSE processing_jobs.started_at
                    END,
                    lease_owner = CASE
                        WHEN processing_jobs.status IN ('completed', 'failed')
                             OR (processing_jobs.status = 'pending' AND processing_jobs.lease_owner IS NULL)
                        THEN NULL
                        ELSE processing_jobs.lease_owner
                    END,
                    lease_expires_at = CASE
                        WHEN processing_jobs.status IN ('completed', 'failed')
                             OR (processing_jobs.status = 'pending' AND processing_jobs.lease_owner IS NULL)
                        THEN NULL
                        ELSE processing_jobs.lease_expires_at
                    END,
                    updated_at = excluded.updated_at;
                """,
                (
                    cls_job_id,
                    cls_job_key,
                    capture_id,
                    resource_id,
                    job_id,
                    now_iso,
                    now_iso,
                    now_iso,
                ),
            )
        else:
            # Schedule downstream extract job with supersedable key
            ext_job_id = generate_id("job")
            ext_job_key = make_job_key("extract", resource_id)
            conn.execute(
                """
                INSERT INTO processing_jobs (
                    id, job_key, capture_id, resource_id, stage, depends_on, status,
                    available_at, attempts, created_at, updated_at
                ) VALUES (?, ?, ?, ?, 'extract', ?, 'pending', ?, 0, ?, ?)
                ON CONFLICT(job_key) DO UPDATE SET
                    capture_id = COALESCE(excluded.capture_id, processing_jobs.capture_id),
                    id = CASE
                        WHEN processing_jobs.status IN ('completed', 'failed')
                             OR (processing_jobs.status = 'pending' AND processing_jobs.lease_owner IS NULL)
                        THEN excluded.id
                        ELSE processing_jobs.id
                    END,
                    status = CASE
                        WHEN processing_jobs.status IN ('completed', 'failed')
                             OR (processing_jobs.status = 'pending' AND processing_jobs.lease_owner IS NULL)
                        THEN 'pending'
                        ELSE processing_jobs.status
                    END,
                    depends_on = CASE
                        WHEN processing_jobs.status IN ('completed', 'failed')
                             OR (processing_jobs.status = 'pending' AND processing_jobs.lease_owner IS NULL)
                        THEN excluded.depends_on
                        ELSE processing_jobs.depends_on
                    END,
                    available_at = CASE
                        WHEN processing_jobs.status IN ('completed', 'failed')
                             OR (processing_jobs.status = 'pending' AND processing_jobs.lease_owner IS NULL)
                        THEN excluded.available_at
                        ELSE processing_jobs.available_at
                    END,
                    attempts = CASE
                        WHEN processing_jobs.status IN ('completed', 'failed')
                             OR (processing_jobs.status = 'pending' AND processing_jobs.lease_owner IS NULL)
                        THEN 0
                        ELSE processing_jobs.attempts
                    END,
                    last_error = CASE
                        WHEN processing_jobs.status IN ('completed', 'failed')
                             OR (processing_jobs.status = 'pending' AND processing_jobs.lease_owner IS NULL)
                        THEN NULL
                        ELSE processing_jobs.last_error
                    END,
                    completed_at = CASE
                        WHEN processing_jobs.status IN ('completed', 'failed')
                             OR (processing_jobs.status = 'pending' AND processing_jobs.lease_owner IS NULL)
                        THEN NULL
                        ELSE processing_jobs.completed_at
                    END,
                    started_at = CASE
                        WHEN processing_jobs.status IN ('completed', 'failed')
                             OR (processing_jobs.status = 'pending' AND processing_jobs.lease_owner IS NULL)
                        THEN NULL
                        ELSE processing_jobs.started_at
                    END,
                    lease_owner = CASE
                        WHEN processing_jobs.status IN ('completed', 'failed')
                             OR (processing_jobs.status = 'pending' AND processing_jobs.lease_owner IS NULL)
                        THEN NULL
                        ELSE processing_jobs.lease_owner
                    END,
                    lease_expires_at = CASE
                        WHEN processing_jobs.status IN ('completed', 'failed')
                             OR (processing_jobs.status = 'pending' AND processing_jobs.lease_owner IS NULL)
                        THEN NULL
                        ELSE processing_jobs.lease_expires_at
                    END,
                    updated_at = excluded.updated_at;
                """,
                (
                    ext_job_id,
                    ext_job_key,
                    capture_id,
                    resource_id,
                    job_id,
                    now_iso,
                    now_iso,
                    now_iso,
                ),
            )
        return "completed"

    elif stage == "extract":
        if work_result.get("status") == "pending":
            # Pending extraction remains pending with retry delay and does NOT schedule classify
            delay_avail = (
                datetime.datetime.now(datetime.UTC) + datetime.timedelta(seconds=10)
            ).isoformat()
            if owner:
                cursor = conn.execute(
                    """
                    UPDATE processing_jobs
                    SET status = 'pending',
                        lease_owner = NULL,
                        lease_expires_at = NULL,
                        available_at = ?,
                        updated_at = ?
                    WHERE id = ? AND status = 'running' AND lease_owner = ?
                      AND (lease_expires_at IS NULL OR lease_expires_at > ?);
                    """,
                    (delay_avail, now_iso, job_id, owner, now_iso),
                )
            else:
                cursor = conn.execute(
                    """
                    UPDATE processing_jobs
                    SET status = 'pending',
                        lease_owner = NULL,
                        lease_expires_at = NULL,
                        available_at = ?,
                        updated_at = ?
                    WHERE id = ? AND status = 'running';
                    """,
                    (delay_avail, now_iso, job_id),
                )
            if cursor.rowcount == 0:
                conn.rollback()
                return "lost-lease"
            return "pending"

        snap_hash = context.get("snapshot_content_hash") if context else None

        # Check if canonical snapshot changed while extraction was in flight
        current_snp_row = conn.execute(
            """
            SELECT content_hash FROM source_snapshots
            WHERE resource_id = ?
            ORDER BY created_at DESC LIMIT 1;
            """,
            (resource_id,),
        ).fetchone()
        current_snap_hash = current_snp_row["content_hash"] if current_snp_row else None

        if snap_hash and current_snap_hash and snap_hash != current_snap_hash:
            # Canonical snapshot changed while extraction was running!
            # Discard stale derived outputs and leave work pending for the latest snapshot.
            if owner:
                cursor = conn.execute(
                    """
                    UPDATE processing_jobs
                    SET status = 'pending',
                        lease_owner = NULL,
                        lease_expires_at = NULL,
                        capture_id = COALESCE((
                            SELECT cr.capture_id FROM capture_resources cr
                            WHERE cr.resource_id = processing_jobs.resource_id
                            ORDER BY cr.created_at DESC LIMIT 1
                        ), processing_jobs.capture_id),
                        available_at = ?,
                        updated_at = ?
                    WHERE id = ? AND status = 'running' AND lease_owner = ?
                      AND (lease_expires_at IS NULL OR lease_expires_at > ?);
                    """,
                    (now_iso, now_iso, job_id, owner, now_iso),
                )
            else:
                cursor = conn.execute(
                    """
                    UPDATE processing_jobs
                    SET status = 'pending',
                        lease_owner = NULL,
                        lease_expires_at = NULL,
                        capture_id = COALESCE((
                            SELECT cr.capture_id FROM capture_resources cr
                            WHERE cr.resource_id = processing_jobs.resource_id
                            ORDER BY cr.created_at DESC LIMIT 1
                        ), processing_jobs.capture_id),
                        available_at = ?,
                        updated_at = ?
                    WHERE id = ? AND status = 'running';
                    """,
                    (now_iso, now_iso, job_id),
                )
            if cursor.rowcount == 0:
                conn.rollback()
                return "lost-lease"
            return "pending"

        # 1. Atomic lease fencing update FIRST
        if owner:
            cursor = conn.execute(
                """
                UPDATE processing_jobs
                SET status = 'completed', input_hash = ?, completed_at = ?, updated_at = ?, lease_owner = NULL, lease_expires_at = NULL
                WHERE id = ? AND status = 'running' AND lease_owner = ?
                  AND (lease_expires_at IS NULL OR lease_expires_at > ?);
                """,
                (snap_hash, now_iso, now_iso, job_id, owner, now_iso),
            )
        else:
            cursor = conn.execute(
                """
                UPDATE processing_jobs
                SET status = 'completed', input_hash = ?, completed_at = ?, updated_at = ?, lease_owner = NULL, lease_expires_at = NULL
                WHERE id = ? AND status = 'running';
                """,
                (snap_hash, now_iso, now_iso, job_id),
            )

        if cursor.rowcount == 0:
            conn.rollback()
            return "lost-lease"

        # 2. Side effects execute ONLY after successful fenced ownership update
        if not work_result.get("cached"):
            ext_res = work_result.get("ext_result")
            if ext_res:
                # Ensure transcript content is preserved (Finding 2)
                text_to_store = ext_res.clean_text or ""
                if ext_res.transcript:
                    trans_text = ext_res.transcript
                    if isinstance(trans_text, (dict, list)):
                        import json

                        trans_text = json.dumps(trans_text, indent=2)
                    if not text_to_store:
                        text_to_store = trans_text
                    elif trans_text not in text_to_store:
                        text_to_store = f"{text_to_store}\n\n## Transcript\n{trans_text}"

                if text_to_store:
                    content_id, _ = store_resource_content(
                        conn,
                        resource_id,
                        clean_text=text_to_store,
                        summary=ext_res.summary,
                        extractor=ext_res.extractor,
                        extractor_version=ext_res.extractor_version,
                        extraction_note=ext_res.error,
                        title=ext_res.title,
                        capture_id=capture_id,
                    )
                    # If structured transcript segments exist, persist into resource_chunks with locators
                    if ext_res.transcript:
                        try:
                            import json

                            raw_t = (
                                json.loads(ext_res.transcript)
                                if isinstance(ext_res.transcript, str)
                                else ext_res.transcript
                            )
                            if isinstance(raw_t, list):
                                for idx, seg in enumerate(raw_t):
                                    seg_text = seg.get("text") or seg.get("content") or str(seg)
                                    chunk_id = generate_id("chk")
                                    loc_json = json.dumps(
                                        {
                                            "start": seg.get("start"),
                                            "end": seg.get("end"),
                                            "speaker": seg.get("speaker"),
                                        }
                                    )
                                    conn.execute(
                                        """
                                        INSERT INTO resource_chunks (
                                            id, resource_content_id, resource_id, chunk_index, text, locator_json, token_count, created_at
                                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                                        """,
                                        (
                                            chunk_id,
                                            content_id,
                                            resource_id,
                                            idx,
                                            seg_text,
                                            loc_json,
                                            len(seg_text.split()),
                                            now_iso,
                                        ),
                                    )
                        except Exception:
                            pass

        if not work_result.get("cached") and not work_result.get("ext_result"):
            return "completed"
        if not get_cached_content(conn, resource_id):
            return "completed"

        # Retrieval must not wait for an optional hosted classifier to succeed.
        conn.execute(
            """
            INSERT INTO processing_jobs (
                id, job_key, capture_id, resource_id, stage, depends_on, status,
                available_at, attempts, created_at, updated_at
            ) VALUES (?, ?, ?, ?, 'embed', ?, 'pending', ?, 0, ?, ?)
            ON CONFLICT(job_key) DO UPDATE SET
                status = CASE WHEN processing_jobs.status IN ('completed', 'failed')
                    THEN 'pending' ELSE processing_jobs.status END,
                depends_on = CASE WHEN processing_jobs.status IN ('completed', 'failed')
                    THEN excluded.depends_on ELSE processing_jobs.depends_on END,
                available_at = CASE WHEN processing_jobs.status IN ('completed', 'failed')
                    THEN excluded.available_at ELSE processing_jobs.available_at END,
                attempts = CASE WHEN processing_jobs.status IN ('completed', 'failed')
                    THEN 0 ELSE processing_jobs.attempts END,
                last_error = CASE WHEN processing_jobs.status IN ('completed', 'failed')
                    THEN NULL ELSE processing_jobs.last_error END,
                completed_at = CASE WHEN processing_jobs.status IN ('completed', 'failed')
                    THEN NULL ELSE processing_jobs.completed_at END,
                updated_at = excluded.updated_at;
            """,
            (
                generate_id("job"),
                make_job_key("embed", resource_id),
                capture_id,
                resource_id,
                job_id,
                now_iso,
                now_iso,
                now_iso,
            ),
        )

        # Schedule downstream classify job with supersedable key
        cls_job_id = generate_id("job")
        cls_job_key = make_job_key("classify", resource_id)
        conn.execute(
            """
            INSERT INTO processing_jobs (
                id, job_key, capture_id, resource_id, stage, depends_on, status,
                available_at, attempts, created_at, updated_at
            ) VALUES (?, ?, ?, ?, 'classify', ?, 'pending', ?, 0, ?, ?)
            ON CONFLICT(job_key) DO UPDATE SET
                capture_id = COALESCE(excluded.capture_id, processing_jobs.capture_id),
                id = CASE
                    WHEN processing_jobs.status IN ('completed', 'failed')
                         OR (processing_jobs.status = 'pending' AND processing_jobs.lease_owner IS NULL)
                    THEN excluded.id
                    ELSE processing_jobs.id
                END,
                status = CASE
                    WHEN processing_jobs.status IN ('completed', 'failed')
                         OR (processing_jobs.status = 'pending' AND processing_jobs.lease_owner IS NULL)
                    THEN 'pending'
                    ELSE processing_jobs.status
                END,
                depends_on = CASE
                    WHEN processing_jobs.status IN ('completed', 'failed')
                         OR (processing_jobs.status = 'pending' AND processing_jobs.lease_owner IS NULL)
                    THEN excluded.depends_on
                    ELSE processing_jobs.depends_on
                END,
                available_at = CASE
                    WHEN processing_jobs.status IN ('completed', 'failed')
                         OR (processing_jobs.status = 'pending' AND processing_jobs.lease_owner IS NULL)
                    THEN excluded.available_at
                    ELSE processing_jobs.available_at
                END,
                attempts = CASE
                    WHEN processing_jobs.status IN ('completed', 'failed')
                         OR (processing_jobs.status = 'pending' AND processing_jobs.lease_owner IS NULL)
                    THEN 0
                    ELSE processing_jobs.attempts
                END,
                last_error = CASE
                    WHEN processing_jobs.status IN ('completed', 'failed')
                         OR (processing_jobs.status = 'pending' AND processing_jobs.lease_owner IS NULL)
                    THEN NULL
                    ELSE processing_jobs.last_error
                END,
                completed_at = CASE
                    WHEN processing_jobs.status IN ('completed', 'failed')
                         OR (processing_jobs.status = 'pending' AND processing_jobs.lease_owner IS NULL)
                    THEN NULL
                    ELSE processing_jobs.completed_at
                END,
                started_at = CASE
                    WHEN processing_jobs.status IN ('completed', 'failed')
                         OR (processing_jobs.status = 'pending' AND processing_jobs.lease_owner IS NULL)
                    THEN NULL
                    ELSE processing_jobs.started_at
                END,
                lease_owner = CASE
                    WHEN processing_jobs.status IN ('completed', 'failed')
                         OR (processing_jobs.status = 'pending' AND processing_jobs.lease_owner IS NULL)
                    THEN NULL
                    ELSE processing_jobs.lease_owner
                END,
                lease_expires_at = CASE
                    WHEN processing_jobs.status IN ('completed', 'failed')
                         OR (processing_jobs.status = 'pending' AND processing_jobs.lease_owner IS NULL)
                    THEN NULL
                    ELSE processing_jobs.lease_expires_at
                END,
                updated_at = excluded.updated_at;
            """,
            (cls_job_id, cls_job_key, capture_id, resource_id, job_id, now_iso, now_iso, now_iso),
        )
        return "completed"

    elif stage == "classify":
        target_id = resource_id or capture_id
        target_type = "resource" if resource_id else "capture"

        # Check if canonical input changed while classification was in flight
        current_target = load_classification_target(conn, target_type, target_id)
        current_hash = current_target["content_hash"] if current_target else None
        work_input_hash = (
            work_result.get("input_hash")
            if work_result
            else (context.get("input_hash") if context else None)
        )

        if current_hash and work_input_hash and current_hash != work_input_hash:
            # Canonical content changed while classification was running!
            # Discard stale derived outputs and leave work pending for the latest input.
            if owner:
                cursor = conn.execute(
                    """
                    UPDATE processing_jobs
                    SET status = 'pending',
                        lease_owner = NULL,
                        lease_expires_at = NULL,
                        capture_id = COALESCE((
                            SELECT cr.capture_id FROM capture_resources cr
                            WHERE cr.resource_id = processing_jobs.resource_id
                            ORDER BY cr.created_at DESC LIMIT 1
                        ), processing_jobs.capture_id),
                        available_at = ?,
                        updated_at = ?
                    WHERE id = ? AND status = 'running' AND lease_owner = ?
                      AND (lease_expires_at IS NULL OR lease_expires_at > ?);
                    """,
                    (now_iso, now_iso, job_id, owner, now_iso),
                )
            else:
                cursor = conn.execute(
                    """
                    UPDATE processing_jobs
                    SET status = 'pending',
                        lease_owner = NULL,
                        lease_expires_at = NULL,
                        capture_id = COALESCE((
                            SELECT cr.capture_id FROM capture_resources cr
                            WHERE cr.resource_id = processing_jobs.resource_id
                            ORDER BY cr.created_at DESC LIMIT 1
                        ), processing_jobs.capture_id),
                        available_at = ?,
                        updated_at = ?
                    WHERE id = ? AND status = 'running';
                    """,
                    (now_iso, now_iso, job_id),
                )
            if cursor.rowcount == 0:
                conn.rollback()
                return "lost-lease"
            return "pending"

        # 1. Atomic lease fencing update FIRST
        if owner:
            cursor = conn.execute(
                """
                UPDATE processing_jobs
                SET status = 'completed', input_hash = ?, completed_at = ?, updated_at = ?, lease_owner = NULL, lease_expires_at = NULL
                WHERE id = ? AND status = 'running' AND lease_owner = ?
                  AND (lease_expires_at IS NULL OR lease_expires_at > ?);
                """,
                (work_input_hash, now_iso, now_iso, job_id, owner, now_iso),
            )
        else:
            cursor = conn.execute(
                """
                UPDATE processing_jobs
                SET status = 'completed', input_hash = ?, completed_at = ?, updated_at = ?, lease_owner = NULL, lease_expires_at = NULL
                WHERE id = ? AND status = 'running';
                """,
                (work_input_hash, now_iso, now_iso, job_id),
            )

        if cursor.rowcount == 0:
            conn.rollback()
            return "lost-lease"

        # 2. Side effects execute ONLY after successful fenced ownership update
        if work_result:
            detected_form = work_result.get("detected_form", "other")
            result = work_result.get("result")
            text = work_result.get("text", "")
            persist_classification_result(conn, target_type, target_id, detected_form, result, text)
        else:
            run_classification_pipeline(conn, target_type, target_id)

        # Schedule downstream finding-extraction if target is a resource
        if target_type == "resource" and resource_id:
            fe_job_id = generate_id("job")
            fe_job_key = make_job_key("finding-extraction", resource_id)
            conn.execute(
                """
                INSERT INTO processing_jobs (
                    id, job_key, capture_id, resource_id, stage, depends_on, status,
                    available_at, attempts, created_at, updated_at
                ) VALUES (?, ?, ?, ?, 'finding-extraction', ?, 'pending', ?, 0, ?, ?)
                ON CONFLICT(job_key) DO UPDATE SET
                    capture_id = COALESCE(excluded.capture_id, processing_jobs.capture_id),
                    status = CASE
                        WHEN processing_jobs.status IN ('completed', 'failed')
                             OR (processing_jobs.status = 'pending' AND processing_jobs.lease_owner IS NULL)
                        THEN 'pending'
                        ELSE processing_jobs.status
                    END,
                    depends_on = CASE
                        WHEN processing_jobs.status IN ('completed', 'failed')
                             OR (processing_jobs.status = 'pending' AND processing_jobs.lease_owner IS NULL)
                        THEN excluded.depends_on
                        ELSE processing_jobs.depends_on
                    END,
                    available_at = CASE
                        WHEN processing_jobs.status IN ('completed', 'failed')
                             OR (processing_jobs.status = 'pending' AND processing_jobs.lease_owner IS NULL)
                        THEN excluded.available_at
                        ELSE processing_jobs.available_at
                    END,
                    attempts = CASE
                        WHEN processing_jobs.status IN ('completed', 'failed')
                             OR (processing_jobs.status = 'pending' AND processing_jobs.lease_owner IS NULL)
                        THEN 0
                        ELSE processing_jobs.attempts
                    END,
                    last_error = CASE
                        WHEN processing_jobs.status IN ('completed', 'failed')
                             OR (processing_jobs.status = 'pending' AND processing_jobs.lease_owner IS NULL)
                        THEN NULL
                        ELSE processing_jobs.last_error
                    END,
                    completed_at = CASE
                        WHEN processing_jobs.status IN ('completed', 'failed')
                             OR (processing_jobs.status = 'pending' AND processing_jobs.lease_owner IS NULL)
                        THEN NULL
                        ELSE processing_jobs.completed_at
                    END,
                    started_at = CASE
                        WHEN processing_jobs.status IN ('completed', 'failed')
                             OR (processing_jobs.status = 'pending' AND processing_jobs.lease_owner IS NULL)
                        THEN NULL
                        ELSE processing_jobs.started_at
                    END,
                    lease_owner = CASE
                        WHEN processing_jobs.status IN ('completed', 'failed')
                             OR (processing_jobs.status = 'pending' AND processing_jobs.lease_owner IS NULL)
                        THEN NULL
                        ELSE processing_jobs.lease_owner
                    END,
                    lease_expires_at = CASE
                        WHEN processing_jobs.status IN ('completed', 'failed')
                             OR (processing_jobs.status = 'pending' AND processing_jobs.lease_owner IS NULL)
                        THEN NULL
                        ELSE processing_jobs.lease_expires_at
                    END,
                    updated_at = excluded.updated_at;
                """,
                (fe_job_id, fe_job_key, capture_id, resource_id, job_id, now_iso, now_iso, now_iso),
            )
        return "completed"

    elif stage == "finding-extraction":
        # Check if canonical input changed while finding extraction was in flight
        content_row = conn.execute(
            "SELECT content_hash FROM resource_contents WHERE resource_id = ? ORDER BY created_at DESC LIMIT 1;",
            (resource_id,),
        ).fetchone()
        current_hash = content_row["content_hash"] if content_row else None
        work_input_hash = (
            work_result.get("input_hash")
            if work_result
            else (context.get("input_hash") if context else None)
        )

        if current_hash and work_input_hash and current_hash != work_input_hash:
            # Canonical content changed while finding extraction was running!
            # Discard stale derived outputs and leave work pending for the latest input.
            if owner:
                cursor = conn.execute(
                    """
                    UPDATE processing_jobs
                    SET status = 'pending',
                        lease_owner = NULL,
                        lease_expires_at = NULL,
                        capture_id = COALESCE((
                            SELECT cr.capture_id FROM capture_resources cr
                            WHERE cr.resource_id = processing_jobs.resource_id
                            ORDER BY cr.created_at DESC LIMIT 1
                        ), processing_jobs.capture_id),
                        available_at = ?,
                        updated_at = ?
                    WHERE id = ? AND status = 'running' AND lease_owner = ?
                      AND (lease_expires_at IS NULL OR lease_expires_at > ?);
                    """,
                    (now_iso, now_iso, job_id, owner, now_iso),
                )
            else:
                cursor = conn.execute(
                    """
                    UPDATE processing_jobs
                    SET status = 'pending',
                        lease_owner = NULL,
                        lease_expires_at = NULL,
                        capture_id = COALESCE((
                            SELECT cr.capture_id FROM capture_resources cr
                            WHERE cr.resource_id = processing_jobs.resource_id
                            ORDER BY cr.created_at DESC LIMIT 1
                        ), processing_jobs.capture_id),
                        available_at = ?,
                        updated_at = ?
                    WHERE id = ? AND status = 'running';
                    """,
                    (now_iso, now_iso, job_id),
                )
            if cursor.rowcount == 0:
                conn.rollback()
                return "lost-lease"
            return "pending"

        # 1. Atomic lease fencing update FIRST
        if owner:
            cursor = conn.execute(
                """
                UPDATE processing_jobs
                SET status = 'completed', input_hash = ?, completed_at = ?, updated_at = ?, lease_owner = NULL, lease_expires_at = NULL
                WHERE id = ? AND status = 'running' AND lease_owner = ?
                  AND (lease_expires_at IS NULL OR lease_expires_at > ?);
                """,
                (work_input_hash, now_iso, now_iso, job_id, owner, now_iso),
            )
        else:
            cursor = conn.execute(
                """
                UPDATE processing_jobs
                SET status = 'completed', input_hash = ?, completed_at = ?, updated_at = ?, lease_owner = NULL, lease_expires_at = NULL
                WHERE id = ? AND status = 'running';
                """,
                (work_input_hash, now_iso, now_iso, job_id),
            )
        if cursor.rowcount == 0:
            conn.rollback()
            return "lost-lease"

        # 2. Side effects: store findings and entities
        from edward.services.findings import store_extracted_payload

        payload = work_result.get("payload") if work_result else None
        content_hash = work_result.get("content_hash", "") if work_result else ""
        if payload and resource_id:
            store_extracted_payload(
                conn=conn,
                resource_id=resource_id,
                payload=payload,
                source_content_hash=content_hash,
            )

        # 3. Schedule downstream embed job
        if resource_id:
            emb_job_id = generate_id("job")
            emb_job_key = make_job_key("embed", resource_id)
            conn.execute(
                """
                INSERT INTO processing_jobs (
                    id, job_key, capture_id, resource_id, stage, depends_on, status,
                    available_at, attempts, created_at, updated_at
                ) VALUES (?, ?, ?, ?, 'embed', ?, 'pending', ?, 0, ?, ?)
                ON CONFLICT(job_key) DO UPDATE SET
                    capture_id = COALESCE(excluded.capture_id, processing_jobs.capture_id),
                    status = CASE
                        WHEN processing_jobs.status IN ('completed', 'failed')
                             OR (processing_jobs.status = 'pending' AND processing_jobs.lease_owner IS NULL)
                        THEN 'pending'
                        ELSE processing_jobs.status
                    END,
                    depends_on = CASE
                        WHEN processing_jobs.status IN ('completed', 'failed')
                             OR (processing_jobs.status = 'pending' AND processing_jobs.lease_owner IS NULL)
                        THEN excluded.depends_on
                        ELSE processing_jobs.depends_on
                    END,
                    available_at = CASE
                        WHEN processing_jobs.status IN ('completed', 'failed')
                             OR (processing_jobs.status = 'pending' AND processing_jobs.lease_owner IS NULL)
                        THEN excluded.available_at
                        ELSE processing_jobs.available_at
                    END,
                    attempts = CASE
                        WHEN processing_jobs.status IN ('completed', 'failed')
                             OR (processing_jobs.status = 'pending' AND processing_jobs.lease_owner IS NULL)
                        THEN 0
                        ELSE processing_jobs.attempts
                    END,
                    last_error = CASE
                        WHEN processing_jobs.status IN ('completed', 'failed')
                             OR (processing_jobs.status = 'pending' AND processing_jobs.lease_owner IS NULL)
                        THEN NULL
                        ELSE processing_jobs.last_error
                    END,
                    completed_at = CASE
                        WHEN processing_jobs.status IN ('completed', 'failed')
                             OR (processing_jobs.status = 'pending' AND processing_jobs.lease_owner IS NULL)
                        THEN NULL
                        ELSE processing_jobs.completed_at
                    END,
                    started_at = CASE
                        WHEN processing_jobs.status IN ('completed', 'failed')
                             OR (processing_jobs.status = 'pending' AND processing_jobs.lease_owner IS NULL)
                        THEN NULL
                        ELSE processing_jobs.started_at
                    END,
                    lease_owner = CASE
                        WHEN processing_jobs.status IN ('completed', 'failed')
                             OR (processing_jobs.status = 'pending' AND processing_jobs.lease_owner IS NULL)
                        THEN NULL
                        ELSE processing_jobs.lease_owner
                    END,
                    lease_expires_at = CASE
                        WHEN processing_jobs.status IN ('completed', 'failed')
                             OR (processing_jobs.status = 'pending' AND processing_jobs.lease_owner IS NULL)
                        THEN NULL
                        ELSE processing_jobs.lease_expires_at
                    END,
                    updated_at = excluded.updated_at;
                """,
                (
                    emb_job_id,
                    emb_job_key,
                    capture_id,
                    resource_id,
                    job_id,
                    now_iso,
                    now_iso,
                    now_iso,
                ),
            )
        return "completed"

    elif stage == "embed":
        # Check if canonical input changed while embedding was in flight
        from edward.services.embed import compute_embedding_input_hash

        content_row = None
        if resource_id:
            content_row = conn.execute(
                "SELECT id, content_hash FROM resource_contents WHERE resource_id = ? ORDER BY created_at DESC LIMIT 1;",
                (resource_id,),
            ).fetchone()

        current_hash = compute_embedding_input_hash(
            conn, resource_id=resource_id, capture_id=capture_id
        )
        work_input_hash = (
            work_result.get("input_hash")
            if work_result
            else (context.get("input_hash") if context else None)
        )

        if current_hash and work_input_hash and current_hash != work_input_hash:
            # Canonical content changed while embedding was running!
            # Discard stale derived outputs and leave work pending for the latest input.
            if owner:
                cursor = conn.execute(
                    """
                    UPDATE processing_jobs
                    SET status = 'pending',
                        lease_owner = NULL,
                        lease_expires_at = NULL,
                        capture_id = COALESCE((
                            SELECT cr.capture_id FROM capture_resources cr
                            WHERE cr.resource_id = processing_jobs.resource_id
                            ORDER BY cr.created_at DESC LIMIT 1
                        ), processing_jobs.capture_id),
                        available_at = ?,
                        updated_at = ?
                    WHERE id = ? AND status = 'running' AND lease_owner = ?
                      AND (lease_expires_at IS NULL OR lease_expires_at > ?);
                    """,
                    (now_iso, now_iso, job_id, owner, now_iso),
                )
            else:
                cursor = conn.execute(
                    """
                    UPDATE processing_jobs
                    SET status = 'pending',
                        lease_owner = NULL,
                        lease_expires_at = NULL,
                        capture_id = COALESCE((
                            SELECT cr.capture_id FROM capture_resources cr
                            WHERE cr.resource_id = processing_jobs.resource_id
                            ORDER BY cr.created_at DESC LIMIT 1
                        ), processing_jobs.capture_id),
                        available_at = ?,
                        updated_at = ?
                    WHERE id = ? AND status = 'running';
                    """,
                    (now_iso, now_iso, job_id),
                )
            if cursor.rowcount == 0:
                conn.rollback()
                return "lost-lease"
            return "pending"

        # 1. Atomic lease fencing update FIRST
        if owner:
            cursor = conn.execute(
                """
                UPDATE processing_jobs
                SET status = 'completed', input_hash = ?, completed_at = ?, updated_at = ?, lease_owner = NULL, lease_expires_at = NULL
                WHERE id = ? AND status = 'running' AND lease_owner = ?
                  AND (lease_expires_at IS NULL OR lease_expires_at > ?);
                """,
                (work_input_hash, now_iso, now_iso, job_id, owner, now_iso),
            )
        else:
            cursor = conn.execute(
                """
                UPDATE processing_jobs
                SET status = 'completed', input_hash = ?, completed_at = ?, updated_at = ?, lease_owner = NULL, lease_expires_at = NULL
                WHERE id = ? AND status = 'running';
                """,
                (work_input_hash, now_iso, now_iso, job_id),
            )
        if cursor.rowcount == 0:
            conn.rollback()
            return "lost-lease"

        # 2. Side effects: persist precomputed chunks and embeddings (pure SQL, no network inside transaction)
        from edward.services.embed import persist_precomputed_embeddings, serialize_vector

        if resource_id:
            content_id = content_row["id"] if content_row else None
            chunk_embeddings = work_result.get("chunk_embeddings") if work_result else None
            resource_embedding = work_result.get("resource_embedding") if work_result else None
            finding_embeddings = work_result.get("finding_embeddings") if work_result else None

            if chunk_embeddings is not None or finding_embeddings is not None:
                persist_precomputed_embeddings(
                    conn=conn,
                    resource_id=resource_id,
                    content_id=content_id,
                    chunk_embeddings=chunk_embeddings or [],
                    resource_embedding=resource_embedding,
                    finding_embeddings=finding_embeddings,
                )
            else:
                raise ValueError("Embed persistence requires precomputed embedding results")
        elif job.get("capture_id"):
            cap_emb = work_result.get("capture_embedding") if work_result else None
            if cap_emb:
                cap_id = cap_emb["capture_id"]
                cap_vec = cap_emb["vector"]
                cap_m = cap_emb.get("model", "deterministic-v1")
                cap_text = cap_emb.get("text", "")
                cap_hash = hashlib.sha256(cap_text.encode("utf-8")).hexdigest()
                cap_blob = serialize_vector(cap_vec)
                existing_cap = conn.execute(
                    "SELECT id FROM embeddings WHERE object_type = 'capture' AND object_id = ? AND model = ?;",
                    (cap_id, cap_m),
                ).fetchone()
                if existing_cap:
                    conn.execute(
                        """
                        UPDATE embeddings
                        SET dimensions = ?, embedding_blob = ?, input_hash = ?, created_at = ?
                        WHERE id = ?;
                        """,
                        (len(cap_vec), cap_blob, cap_hash, now_iso, existing_cap["id"]),
                    )
                else:
                    conn.execute(
                        """
                        INSERT INTO embeddings (
                            id, object_type, object_id, model, dimensions,
                            embedding_blob, input_hash, created_at
                        ) VALUES (?, 'capture', ?, ?, ?, ?, ?, ?);
                        """,
                        (
                            generate_id("emb"),
                            cap_id,
                            cap_m,
                            len(cap_vec),
                            cap_blob,
                            cap_hash,
                            now_iso,
                        ),
                    )
        return "completed"

    return "failed"


def execute_job(
    conn: sqlite3.Connection,
    blob_store: BlobStore,
    job: dict[str, Any],
) -> bool:
    """Execute a single claimed job directly (for testing or atomic step execution)."""
    ctx = None
    try:
        ctx = _load_job_context(conn, job)
        work_res = _perform_job_work(blob_store, job, ctx)
        job_err = None
    except Exception as e:
        work_res = None
        job_err = e

    try:
        outcome = _persist_job_result(
            conn, blob_store, job, work_res, job_err, worker_id=job.get("lease_owner"), context=ctx
        )
        return outcome == "completed"
    except Exception as persist_err:
        try:
            conn.rollback()
        except Exception:
            pass
        try:
            outcome = _persist_job_result(
                conn,
                blob_store,
                job,
                work_result=None,
                job_error=persist_err,
                worker_id=job.get("lease_owner"),
                context=ctx,
            )
            return outcome == "completed"
        except Exception:
            return False


def process_pending_jobs(
    db_or_conn: Any,
    blob_store: BlobStore,
    worker_id: str | None = None,
    limit: int = 10,
    capture_id: str | None = None,
    stage: str | None = None,
) -> dict[str, int]:
    """Process pending background jobs in a loop up to limit using short transaction boundaries."""
    w_id = worker_id or f"worker_{uuid.uuid4().hex[:8]}"
    completed = 0
    failed = 0
    pending = 0
    lost_lease = 0

    for _ in range(limit):
        # 1. Short transaction: claim next pending job and load context
        with _get_transaction(db_or_conn) as conn:
            job = claim_job(conn, w_id, stage=stage, capture_id=capture_id)
            if not job:
                break
            try:
                ctx = _load_job_context(conn, job)
                load_err = None
            except Exception as e:
                ctx = {}
                load_err = e

        # 2. External work performed outside the transaction
        if load_err is not None:
            work_res = None
            job_err = load_err
        else:
            try:
                work_res = _perform_job_work(blob_store, job, ctx)
                job_err = None
            except Exception as e:
                work_res = None
                job_err = e

        # 3. Short transaction: persist results and commit
        try:
            with _get_transaction(db_or_conn) as conn:
                outcome = _persist_job_result(
                    conn, blob_store, job, work_res, job_err, worker_id=w_id, context=ctx
                )
        except Exception as persist_err:
            # Persistence itself failed: transaction was rolled back.
            # Record the sanitized failure and retry state in a separate transaction.
            try:
                with _get_transaction(db_or_conn) as retry_conn:
                    outcome = _persist_job_result(
                        retry_conn,
                        blob_store,
                        job,
                        work_result=None,
                        job_error=persist_err,
                        worker_id=w_id,
                        context=ctx,
                    )
            except Exception:
                outcome = "failed"

        if outcome == "completed":
            completed += 1
        elif outcome == "failed":
            failed += 1
        elif outcome == "pending":
            pending += 1
        elif outcome == "lost-lease":
            lost_lease += 1

    with _get_transaction(db_or_conn) as conn:
        remaining = conn.execute(
            "SELECT COUNT(*) FROM processing_jobs WHERE status = 'pending';"
        ).fetchone()[0]

    return {
        "completed": completed,
        "failed": failed,
        "pending": pending,
        "lost_lease": lost_lease,
        "remaining_pending": remaining,
    }


def retry_failed_jobs(conn: sqlite3.Connection) -> int:
    """Reset all failed jobs to pending status with attempts reset to zero."""
    now_iso = datetime.datetime.now(datetime.UTC).isoformat()
    cursor = conn.execute(
        """
        UPDATE processing_jobs
        SET status = 'pending',
            attempts = 0,
            available_at = ?,
            lease_owner = NULL,
            lease_expires_at = NULL,
            updated_at = ?
        WHERE status = 'failed';
        """,
        (now_iso, now_iso),
    )
    return cursor.rowcount


def count_jobs(
    conn: sqlite3.Connection,
    stage: str | None = None,
    status: str | None = None,
) -> int:
    """Count processing jobs matching a stage and/or status filter."""
    clauses: list[str] = []
    params: list[str] = []
    if stage is not None:
        clauses.append("stage = ?")
        params.append(stage)
    if status is not None:
        clauses.append("status = ?")
        params.append(status)
    where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
    return conn.execute(
        f"SELECT COUNT(*) FROM processing_jobs{where};",
        params,  # noqa: S608 -- fixed fragments
    ).fetchone()[0]


def purge_jobs(
    conn: sqlite3.Connection,
    stage: str,
    status: str = "pending",
    exclude_running: bool = True,
) -> dict[str, Any]:
    """Delete queued jobs for a stage, leaving every other stage and table untouched.

    Deleting a backlog is a destructive operation on a shared queue, so this reports
    counts taken on both sides of the delete rather than trusting the delete's own
    rowcount. A job currently leased by a worker is not removed: ``exclude_running``
    keeps the delete away from anything a live process might still be executing.
    """
    before_stage = count_jobs(conn, stage=stage, status=status)
    before_total = count_jobs(conn)

    lease_clause = (
        " AND (lease_owner IS NULL OR lease_expires_at IS NULL)"
        if exclude_running and status == "running"
        else ""
    )
    cursor = conn.execute(
        f"DELETE FROM processing_jobs WHERE stage = ? AND status = ?{lease_clause};",  # noqa: S608
        (stage, status),
    )

    after_stage = count_jobs(conn, stage=stage, status=status)
    after_total = count_jobs(conn)

    return {
        "stage": stage,
        "status": status,
        "before_stage": before_stage,
        "after_stage": after_stage,
        "deleted": cursor.rowcount,
        "before_total": before_total,
        "after_total": after_total,
        "other_stages_deleted": max(0, (before_total - before_stage) - (after_total - after_stage)),
    }


def get_processing_status(conn: sqlite3.Connection) -> dict[str, Any]:
    """Retrieve summary counts of processing jobs across stages and statuses."""
    rows = conn.execute(
        """
        SELECT status, stage, COUNT(*) as cnt
        FROM processing_jobs
        GROUP BY status, stage;
        """
    ).fetchall()

    by_status: dict[str, int] = {"pending": 0, "running": 0, "completed": 0, "failed": 0}
    by_stage: dict[str, int] = {}
    by_stage_status: dict[str, dict[str, int]] = {}

    for r in rows:
        st = r["status"]
        sg = r["stage"]
        c = r["cnt"]
        by_status[st] = by_status.get(st, 0) + c
        by_stage[sg] = by_stage.get(sg, 0) + c
        by_stage_status.setdefault(sg, {})[st] = c

    total = sum(by_status.values())
    failure_reasons: dict[str, dict[str, int]] = {}
    for row in conn.execute(
        "SELECT stage, last_error FROM processing_jobs WHERE status = 'failed';"
    ):
        error = row["last_error"] or ""
        http_status = re.search(r"(?:error status|returned (?:unexpected )?status) (\d{3})", error)
        if http_status:
            reason = f"HTTP {http_status.group(1)}"
        elif "exceeds limit" in error:
            reason = "response too large"
        elif "timed out" in error.lower():
            reason = "timeout"
        else:
            reason = "other"
        stage_reasons = failure_reasons.setdefault(row["stage"], {})
        stage_reasons[reason] = stage_reasons.get(reason, 0) + 1

    return {
        "total": total,
        "by_status": by_status,
        "by_stage": by_stage,
        "by_stage_status": by_stage_status,
        "failure_reasons": failure_reasons,
    }


def reconcile_completed_jobs(conn: sqlite3.Connection) -> dict[str, int]:
    """Mark pending classify jobs whose targets already have judgments or labels as completed."""
    now_iso = datetime.datetime.now(datetime.UTC).isoformat()
    cursor = conn.execute(
        """
        UPDATE processing_jobs
        SET status = 'completed', completed_at = ?, updated_at = ?
        WHERE stage = 'classify' AND status = 'pending'
          AND (
            EXISTS (
                SELECT 1 FROM object_labels ol
                WHERE (ol.object_type = 'resource' AND ol.object_id = processing_jobs.resource_id)
                   OR (ol.object_type = 'capture' AND ol.object_id = processing_jobs.capture_id)
            )
            OR EXISTS (
                SELECT 1 FROM judgments jdg
                WHERE (jdg.object_type = 'resource' AND jdg.object_id = processing_jobs.resource_id)
                   OR (jdg.object_type = 'capture' AND jdg.object_id = processing_jobs.capture_id)
            )
          );
        """,
        (now_iso, now_iso),
    )
    return {"reconciled_classify_jobs": cursor.rowcount}
