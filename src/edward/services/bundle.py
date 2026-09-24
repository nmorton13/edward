"""Research bundle and Markdown report ingestion service with idempotency conflict checks."""

import datetime
import hashlib
import json
import re
import sqlite3
from typing import Any

from edward.blobs import BlobStore
from edward.models import ResearchBundle, generate_id, make_job_key
from edward.services.audit import record_audit_event
from edward.services.capture import IdempotencyConflictError, canonicalize_url, hash_url
from edward.services.classification import compute_classification_input_hash
from edward.services.lifecycle import add_intent, reindex_object_document
from edward.services.resource import (
    get_cached_content,
    save_source_snapshot,
    store_resource_content,
)


def import_research_bundle(
    conn: sqlite3.Connection,
    blob_store: BlobStore,
    bundle_data: dict[str, Any],
    idempotency_key: str | None = None,
) -> dict[str, Any]:
    """Validate and import a structured research bundle into Edward in an atomic transaction."""
    # 1. Pydantic validation against contract
    bundle = ResearchBundle.model_validate(bundle_data)

    effective_key = idempotency_key or bundle.idempotency_key or f"bundle:{bundle.bundle_id}"
    req_hash = hashlib.sha256(
        json.dumps(bundle_data, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    now_iso = datetime.datetime.now(datetime.UTC).isoformat()

    # 2. Idempotency and conflict detection
    if effective_key:
        row = conn.execute(
            """
            SELECT result_object_id, request_hash
            FROM idempotency_keys
            WHERE namespace = 'bundle' AND operation = 'import' AND key = ?;
            """,
            (effective_key,),
        ).fetchone()

        if row:
            if row["request_hash"] == req_hash:
                # Replay
                return {
                    "status": "replayed",
                    "bundle_id": bundle.bundle_id,
                    "capture_id": row["result_object_id"],
                }
            else:
                raise IdempotencyConflictError(
                    f"Idempotency key '{effective_key}' already used with different content"
                )

    # 3. Create Capture record for bundle import
    cap_id = generate_id("cap")
    agent_info = bundle.agent or {}
    collector_name = agent_info.get("name") or "agent"
    run_id = agent_info.get("run_id")

    conn.execute(
        """
        INSERT INTO captures (
            id, origin_namespace, origin_id, collection_channel, collector,
            collector_run_id, acquisition_method, retrieved_at, raw_content,
            user_note, review_state, created_at, updated_at
        ) VALUES (?, 'agent', ?, 'agent', ?, ?, 'bundle-import', ?, ?, ?, 'unreviewed', ?, ?);
        """,
        (
            cap_id,
            bundle.bundle_id,
            collector_name,
            run_id,
            now_iso,
            bundle.summary,
            bundle.title,
            now_iso,
            now_iso,
        ),
    )

    url_to_resource_id: dict[str, str] = {}
    created_resource_ids: list[str] = []
    source_resource_ids: list[str] = []

    # 4. Ingest Sources
    for src in bundle.sources:
        canon_url = None
        u_hash = None
        if src.url:
            canon_url = canonicalize_url(src.url)
            u_hash = hash_url(canon_url)

        ident_key = src.identity_key
        if not ident_key:
            if canon_url:
                ident_key = f"url:{canon_url}"
            elif src.source_id:
                ident_key = src.source_id
            elif src.origin and src.origin_id:
                ident_key = f"{src.origin}:{src.origin_id}"
            else:
                ident_key = f"{src.origin}:{generate_id('src')}"

        # Check existing resource
        res_row = None
        if src.source_id:
            res_row = conn.execute(
                "SELECT id FROM resources WHERE id = ?;", (src.source_id,)
            ).fetchone()
        if not res_row:
            res_row = conn.execute(
                "SELECT id FROM resources WHERE identity_key = ? OR (canonical_url IS NOT NULL AND canonical_url = ?);",
                (ident_key, canon_url),
            ).fetchone()

        if res_row:
            r_id = res_row["id"]
        else:
            r_id = generate_id("res")
            conn.execute(
                """
                INSERT INTO resources (
                    id, identity_key, canonical_url, url_hash, title,
                    review_state, is_deleted, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, 'unreviewed', 0, ?, ?);
                """,
                (r_id, ident_key, canon_url, u_hash, src.title, now_iso, now_iso),
            )
            created_resource_ids.append(r_id)

        source_resource_ids.append(r_id)

        if src.url:
            url_to_resource_id[src.url] = r_id
            if canon_url:
                url_to_resource_id[canon_url] = r_id

        # Link capture to resource
        conn.execute(
            """
            INSERT OR IGNORE INTO capture_resources (capture_id, resource_id, relationship_type, created_at)
            VALUES (?, ?, 'primary', ?);
            """,
            (cap_id, r_id, now_iso),
        )

        # Store extracted text if provided
        extracted_content = src.extracted_text or getattr(src, "extracted_markdown", None)

        # Store source snapshot if provided via inline snapshot
        snap_bytes = None
        raw_snap = src.snapshot or getattr(src, "snapshot", None)
        if raw_snap:
            snap_bytes = raw_snap.encode("utf-8") if isinstance(raw_snap, str) else raw_snap

        # Validate agent-supplied content_hash and snapshot_hash if content was provided (Finding 3)
        ext_hash = (
            hashlib.sha256(extracted_content.encode("utf-8")).hexdigest()
            if extracted_content
            else None
        )
        snap_computed_hash = hashlib.sha256(snap_bytes).hexdigest() if snap_bytes else None

        supplied_snap_hash = getattr(src, "snapshot_hash", None)
        if supplied_snap_hash:
            if snap_bytes and supplied_snap_hash != snap_computed_hash:
                raise ValueError(
                    f"Supplied snapshot_hash '{supplied_snap_hash}' does not match computed hash '{snap_computed_hash}' of snapshot"
                )

        if src.content_hash:
            if extracted_content and snap_bytes:
                # Disambiguate when both forms are present
                if src.content_hash == ext_hash:
                    pass
                elif src.content_hash == snap_computed_hash and not supplied_snap_hash:
                    pass
                else:
                    raise ValueError(
                        f"Supplied content_hash '{src.content_hash}' does not match computed hash '{ext_hash}' of extracted_text "
                        f"or '{snap_computed_hash}' of snapshot"
                    )
            elif extracted_content:
                if src.content_hash != ext_hash:
                    raise ValueError(
                        f"Supplied content_hash '{src.content_hash}' does not match computed hash '{ext_hash}' of extracted_text"
                    )
            elif snap_bytes:
                if src.content_hash != snap_computed_hash:
                    raise ValueError(
                        f"Supplied content_hash '{src.content_hash}' does not match computed hash '{snap_computed_hash}' of snapshot"
                    )

        if extracted_content:
            store_resource_content(
                conn,
                r_id,
                clean_text=extracted_content,
                extractor=collector_name,
                extractor_version=str(agent_info.get("version") or "1.0"),
                title=src.title,
                capture_id=cap_id,
            )

        if snap_bytes:
            save_source_snapshot(
                conn,
                blob_store,
                r_id,
                snap_bytes,
                headers={"content-type": "text/html; charset=utf-8"},
                capture_id=cap_id,
            )

        # Check if existing resource already has extracted content in cache
        cached_content = get_cached_content(conn, r_id) if not extracted_content else None
        snap_already_extracted = False
        if snap_bytes and cached_content and cached_content.get("clean_text"):
            ext_row = conn.execute(
                "SELECT 1 FROM processing_jobs WHERE resource_id = ? AND stage = 'extract' AND status = 'completed' AND input_hash = ? LIMIT 1;",
                (r_id, snap_computed_hash),
            ).fetchone()
            if ext_row:
                snap_already_extracted = True

        # Enter processing pipeline with exact contract routing
        if extracted_content or (
            cached_content
            and cached_content.get("clean_text")
            and (not snap_bytes or snap_already_extracted)
        ):
            # Extracted text or reused cached content -> classify
            cls_job_id = generate_id("job")
            cls_job_key = make_job_key("classify", r_id)
            c_text = extracted_content or (
                cached_content.get("clean_text") if cached_content else ""
            )
            c_summary = cached_content.get("summary") if cached_content else None
            bnd_cls_input_hash = (
                compute_classification_input_hash(
                    clean_text=c_text,
                    summary=c_summary,
                    title=src.title,
                    url=src.url,
                )
                if (c_text or c_summary)
                else None
            )
            conn.execute(
                """
                INSERT INTO processing_jobs (
                    id, job_key, capture_id, resource_id, stage, status,
                    available_at, attempts, created_at, updated_at
                ) VALUES (?, ?, ?, ?, 'classify', 'pending', ?, 0, ?, ?)
                ON CONFLICT(job_key) DO UPDATE SET
                    capture_id = COALESCE(excluded.capture_id, processing_jobs.capture_id),
                    status = CASE
                        WHEN (processing_jobs.status = 'completed' AND (? IS NULL OR processing_jobs.input_hash IS NULL OR processing_jobs.input_hash != ?))
                             OR processing_jobs.status = 'failed'
                             OR (processing_jobs.status = 'pending' AND processing_jobs.lease_owner IS NULL)
                        THEN 'pending'
                        ELSE processing_jobs.status
                    END,
                    available_at = CASE
                        WHEN (processing_jobs.status = 'completed' AND (? IS NULL OR processing_jobs.input_hash IS NULL OR processing_jobs.input_hash != ?))
                             OR processing_jobs.status = 'failed'
                             OR (processing_jobs.status = 'pending' AND processing_jobs.lease_owner IS NULL)
                        THEN excluded.available_at
                        ELSE processing_jobs.available_at
                    END,
                    attempts = CASE
                        WHEN (processing_jobs.status = 'completed' AND (? IS NULL OR processing_jobs.input_hash IS NULL OR processing_jobs.input_hash != ?))
                             OR processing_jobs.status = 'failed'
                             OR (processing_jobs.status = 'pending' AND processing_jobs.lease_owner IS NULL)
                        THEN 0
                        ELSE processing_jobs.attempts
                    END,
                    last_error = CASE
                        WHEN (processing_jobs.status = 'completed' AND (? IS NULL OR processing_jobs.input_hash IS NULL OR processing_jobs.input_hash != ?))
                             OR processing_jobs.status = 'failed'
                             OR (processing_jobs.status = 'pending' AND processing_jobs.lease_owner IS NULL)
                        THEN NULL
                        ELSE processing_jobs.last_error
                    END,
                    completed_at = CASE
                        WHEN (processing_jobs.status = 'completed' AND (? IS NULL OR processing_jobs.input_hash IS NULL OR processing_jobs.input_hash != ?))
                             OR processing_jobs.status = 'failed'
                             OR (processing_jobs.status = 'pending' AND processing_jobs.lease_owner IS NULL)
                        THEN NULL
                        ELSE processing_jobs.completed_at
                    END,
                    started_at = CASE
                        WHEN (processing_jobs.status = 'completed' AND (? IS NULL OR processing_jobs.input_hash IS NULL OR processing_jobs.input_hash != ?))
                             OR processing_jobs.status = 'failed'
                             OR (processing_jobs.status = 'pending' AND processing_jobs.lease_owner IS NULL)
                        THEN NULL
                        ELSE processing_jobs.started_at
                    END,
                    lease_owner = CASE
                        WHEN (processing_jobs.status = 'completed' AND (? IS NULL OR processing_jobs.input_hash IS NULL OR processing_jobs.input_hash != ?))
                             OR processing_jobs.status = 'failed'
                             OR (processing_jobs.status = 'pending' AND processing_jobs.lease_owner IS NULL)
                        THEN NULL
                        ELSE processing_jobs.lease_owner
                    END,
                    lease_expires_at = CASE
                        WHEN (processing_jobs.status = 'completed' AND (? IS NULL OR processing_jobs.input_hash IS NULL OR processing_jobs.input_hash != ?))
                             OR processing_jobs.status = 'failed'
                             OR (processing_jobs.status = 'pending' AND processing_jobs.lease_owner IS NULL)
                        THEN NULL
                        ELSE processing_jobs.lease_expires_at
                    END,
                    updated_at = excluded.updated_at;
                """,
                (
                    cls_job_id,
                    cls_job_key,
                    cap_id,
                    r_id,
                    now_iso,
                    now_iso,
                    now_iso,
                    bnd_cls_input_hash,
                    bnd_cls_input_hash,
                    bnd_cls_input_hash,
                    bnd_cls_input_hash,
                    bnd_cls_input_hash,
                    bnd_cls_input_hash,
                    bnd_cls_input_hash,
                    bnd_cls_input_hash,
                    bnd_cls_input_hash,
                    bnd_cls_input_hash,
                    bnd_cls_input_hash,
                    bnd_cls_input_hash,
                    bnd_cls_input_hash,
                    bnd_cls_input_hash,
                    bnd_cls_input_hash,
                    bnd_cls_input_hash,
                ),
            )
        elif snap_bytes:
            # Snapshot -> extract -> classify
            ext_job_id = generate_id("job")
            ext_job_key = make_job_key("extract", r_id)
            conn.execute(
                """
                INSERT INTO processing_jobs (
                    id, job_key, capture_id, resource_id, stage, status,
                    available_at, attempts, created_at, updated_at
                ) VALUES (?, ?, ?, ?, 'extract', 'pending', ?, 0, ?, ?)
                ON CONFLICT(job_key) DO UPDATE SET
                    capture_id = COALESCE(excluded.capture_id, processing_jobs.capture_id),
                    status = CASE
                        WHEN (processing_jobs.status = 'completed' AND (processing_jobs.input_hash IS NULL OR processing_jobs.input_hash != ?))
                             OR processing_jobs.status = 'failed'
                             OR (processing_jobs.status = 'pending' AND processing_jobs.lease_owner IS NULL)
                        THEN 'pending'
                        ELSE processing_jobs.status
                    END,
                    available_at = CASE
                        WHEN (processing_jobs.status = 'completed' AND (processing_jobs.input_hash IS NULL OR processing_jobs.input_hash != ?))
                             OR processing_jobs.status = 'failed'
                             OR (processing_jobs.status = 'pending' AND processing_jobs.lease_owner IS NULL)
                        THEN excluded.available_at
                        ELSE processing_jobs.available_at
                    END,
                    attempts = CASE
                        WHEN (processing_jobs.status = 'completed' AND (processing_jobs.input_hash IS NULL OR processing_jobs.input_hash != ?))
                             OR processing_jobs.status = 'failed'
                             OR (processing_jobs.status = 'pending' AND processing_jobs.lease_owner IS NULL)
                        THEN 0
                        ELSE processing_jobs.attempts
                    END,
                    last_error = CASE
                        WHEN (processing_jobs.status = 'completed' AND (processing_jobs.input_hash IS NULL OR processing_jobs.input_hash != ?))
                             OR processing_jobs.status = 'failed'
                             OR (processing_jobs.status = 'pending' AND processing_jobs.lease_owner IS NULL)
                        THEN NULL
                        ELSE processing_jobs.last_error
                    END,
                    completed_at = CASE
                        WHEN (processing_jobs.status = 'completed' AND (processing_jobs.input_hash IS NULL OR processing_jobs.input_hash != ?))
                             OR processing_jobs.status = 'failed'
                             OR (processing_jobs.status = 'pending' AND processing_jobs.lease_owner IS NULL)
                        THEN NULL
                        ELSE processing_jobs.completed_at
                    END,
                    started_at = CASE
                        WHEN (processing_jobs.status = 'completed' AND (processing_jobs.input_hash IS NULL OR processing_jobs.input_hash != ?))
                             OR processing_jobs.status = 'failed'
                             OR (processing_jobs.status = 'pending' AND processing_jobs.lease_owner IS NULL)
                        THEN NULL
                        ELSE processing_jobs.started_at
                    END,
                    lease_owner = CASE
                        WHEN (processing_jobs.status = 'completed' AND (processing_jobs.input_hash IS NULL OR processing_jobs.input_hash != ?))
                             OR processing_jobs.status = 'failed'
                             OR (processing_jobs.status = 'pending' AND processing_jobs.lease_owner IS NULL)
                        THEN NULL
                        ELSE processing_jobs.lease_owner
                    END,
                    lease_expires_at = CASE
                        WHEN (processing_jobs.status = 'completed' AND (processing_jobs.input_hash IS NULL OR processing_jobs.input_hash != ?))
                             OR processing_jobs.status = 'failed'
                             OR (processing_jobs.status = 'pending' AND processing_jobs.lease_owner IS NULL)
                        THEN NULL
                        ELSE processing_jobs.lease_expires_at
                    END,
                    updated_at = excluded.updated_at;
                """,
                (
                    ext_job_id,
                    ext_job_key,
                    cap_id,
                    r_id,
                    now_iso,
                    now_iso,
                    now_iso,
                    snap_computed_hash,
                    snap_computed_hash,
                    snap_computed_hash,
                    snap_computed_hash,
                    snap_computed_hash,
                    snap_computed_hash,
                    snap_computed_hash,
                    snap_computed_hash,
                ),
            )
        elif src.url:
            # URL only -> resource-fetch -> extract -> classify
            fetch_job_id = generate_id("job")
            fetch_job_key = make_job_key("fetch", r_id)
            conn.execute(
                """
                INSERT INTO processing_jobs (
                    id, job_key, capture_id, resource_id, stage, status,
                    available_at, attempts, created_at, updated_at
                ) VALUES (?, ?, ?, ?, 'resource-fetch', 'pending', ?, 0, ?, ?)
                ON CONFLICT(job_key) DO UPDATE SET
                    capture_id = COALESCE(excluded.capture_id, processing_jobs.capture_id),
                    status = CASE
                        WHEN processing_jobs.status IN ('completed', 'failed')
                             OR (processing_jobs.status = 'pending' AND processing_jobs.lease_owner IS NULL)
                        THEN 'pending'
                        ELSE processing_jobs.status
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
                (fetch_job_id, fetch_job_key, cap_id, r_id, now_iso, now_iso, now_iso),
            )

    # 5. Ingest Findings
    created_finding_ids: list[str] = []
    for f in bundle.findings:
        find_id = generate_id("fin")
        res_id = None
        if f.source_url:
            c_url = canonicalize_url(f.source_url)
            res_id = url_to_resource_id.get(f.source_url) or url_to_resource_id.get(c_url)

        conn.execute(
            """
            INSERT INTO findings (
                id, resource_id, statement, assertion_role, agent_confidence,
                extractor, extractor_version, review_state, is_deleted, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, 'unreviewed', 0, ?, ?);
            """,
            (
                find_id,
                res_id,
                f.statement,
                f.assertion_role,
                f.agent_confidence,
                collector_name,
                str(agent_info.get("version") or "1.0"),
                now_iso,
                now_iso,
            ),
        )
        created_finding_ids.append(find_id)

        # Supporting passage(s)
        passages_to_insert: list[tuple[str, Any]] = []
        if f.supporting_passage and f.supporting_passage.strip():
            passages_to_insert.append((f.supporting_passage.strip(), f.locator))
        for supp in getattr(f, "support", None) or []:
            if isinstance(supp, dict) and supp.get("passage") and supp["passage"].strip():
                passages_to_insert.append((supp["passage"].strip(), supp.get("locator")))

        for p_text, p_loc in passages_to_insert:
            loc_json = json.dumps(p_loc) if p_loc else None
            supp_id = generate_id("fs")
            p_hash = hashlib.sha256(p_text.encode("utf-8")).hexdigest()
            conn.execute(
                """
                INSERT INTO finding_support (id, finding_id, passage, locator_json, content_hash, created_at)
                VALUES (?, ?, ?, ?, ?, ?);
                """,
                (supp_id, find_id, p_text, loc_json, p_hash, now_iso),
            )

        # Agent-suggested intents remain inactive suggestions (is_active=0)
        for it in getattr(f, "intents", None) or []:
            if isinstance(it, str) and it.strip():
                add_intent(
                    conn,
                    "finding",
                    find_id,
                    it.strip(),
                    source="agent",
                    actor=collector_name,
                    is_active=False,
                )
                ann_id = generate_id("ann")
                conn.execute(
                    """
                    INSERT INTO annotations (id, object_type, object_id, annotation_type, content, author, created_at)
                    VALUES (?, 'finding', ?, 'suggested-intent', ?, ?, ?);
                    """,
                    (ann_id, find_id, it.strip(), collector_name, now_iso),
                )

        # Labels
        for lbl in f.labels:
            lbl_clean = lbl.strip().lower()
            if lbl_clean:
                conn.execute(
                    "INSERT OR IGNORE INTO label_families (id, description, created_at) VALUES ('custom', 'Labels', ?);",
                    (now_iso,),
                )
                conn.execute(
                    "INSERT OR IGNORE INTO labels (id, family, description, active, version, created_at) VALUES (?, 'custom', ?, 1, '1.0', ?);",
                    (lbl_clean, lbl_clean, now_iso),
                )
                lbl_obj_id = f"lbl_{find_id}_{lbl_clean}_agent"
                conf_val = f.agent_confidence if f.agent_confidence is not None else 1.0
                conn.execute(
                    """
                    INSERT OR IGNORE INTO object_labels (id, object_type, object_id, label_id, source, confidence, created_at)
                    VALUES (?, 'finding', ?, ?, 'agent', ?, ?);
                    """,
                    (lbl_obj_id, find_id, lbl_clean, conf_val, now_iso),
                )

        # Entities with normalized_name
        for ent in f.entities:
            ent_clean = ent.strip()
            if ent_clean:
                norm_name = re.sub(r"\s+", " ", ent_clean).lower()
                row = conn.execute(
                    "SELECT id FROM entities WHERE normalized_name = ? OR name = ?;",
                    (norm_name, ent_clean),
                ).fetchone()
                if row:
                    ent_id = row["id"]
                else:
                    ent_id = f"ent_{hashlib.sha256(norm_name.encode('utf-8')).hexdigest()[:12]}"
                    conn.execute(
                        """
                        INSERT INTO entities (id, name, normalized_name, entity_type, created_at)
                        VALUES (?, ?, ?, 'general', ?)
                        ON CONFLICT(name) DO UPDATE SET normalized_name = excluded.normalized_name;
                        """,
                        (ent_id, ent_clean, norm_name, now_iso),
                    )
                obj_ent_id = f"oe_{find_id}_{ent_id}"
                conn.execute(
                    """
                    INSERT OR IGNORE INTO object_entities (id, object_type, object_id, entity_id, extractor, created_at)
                    VALUES (?, 'finding', ?, ?, ?, ?);
                    """,
                    (obj_ent_id, find_id, ent_id, collector_name, now_iso),
                )

        reindex_object_document(conn, "finding", find_id)

    # 6. Apply Suggested Intents as inactive suggestions (is_active=0)
    for it in bundle.suggested_intents:
        add_intent(
            conn,
            "capture",
            cap_id,
            it.strip(),
            source="agent",
            actor=collector_name,
            is_active=False,
        )
        ann_id = generate_id("ann")
        conn.execute(
            """
            INSERT INTO annotations (id, object_type, object_id, annotation_type, content, author, created_at)
            VALUES (?, 'capture', ?, 'suggested-intent', ?, ?, ?);
            """,
            (ann_id, cap_id, it.strip(), collector_name, now_iso),
        )

    # 7. Reproject capture in FTS
    reindex_object_document(conn, "capture", cap_id)

    # 7.5 Queue Embed Processing Jobs for capture and all imported resources
    cap_embed_text = bundle.title or ""
    if bundle.brief:
        cap_embed_text += "\n\n" + bundle.brief
    if bundle.summary:
        cap_embed_text += "\n\n" + bundle.summary
    if cap_embed_text.strip():
        emb_job_id = generate_id("job")
        emb_job_key = make_job_key("embed", cap_id)
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
                cap_id,
                now_iso,
                now_iso,
                now_iso,
                now_iso,
                now_iso,
                now_iso,
                now_iso,
            ),
        )

    for r_id in set(source_resource_ids):
        r_emb_job_id = generate_id("job")
        r_emb_job_key = make_job_key("embed", r_id)
        conn.execute(
            """
            INSERT INTO processing_jobs (
                id, job_key, capture_id, resource_id, stage, status,
                available_at, attempts, created_at, updated_at
            ) VALUES (?, ?, NULL, ?, 'embed', 'pending', ?, 0, ?, ?)
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
                r_emb_job_id,
                r_emb_job_key,
                r_id,
                now_iso,
                now_iso,
                now_iso,
                now_iso,
                now_iso,
                now_iso,
                now_iso,
            ),
        )

    # 8. Record Idempotency Key
    if effective_key:
        idk_id = generate_id("idk")
        conn.execute(
            """
            INSERT INTO idempotency_keys (id, namespace, operation, key, result_object_id, request_hash, created_at)
            VALUES (?, 'bundle', 'import', ?, ?, ?, ?);
            """,
            (idk_id, effective_key, cap_id, req_hash, now_iso),
        )

    # 9. Audit Event
    record_audit_event(
        conn,
        event_type="bundle.imported",
        object_type="capture",
        object_id=cap_id,
        actor=collector_name,
        payload={
            "bundle_id": bundle.bundle_id,
            "title": bundle.title,
            "sources_count": len(bundle.sources),
            "findings_count": len(bundle.findings),
        },
    )

    return {
        "status": "imported",
        "bundle_id": bundle.bundle_id,
        "capture_id": cap_id,
        "resources_count": len(bundle.sources),
        "findings_count": len(bundle.findings),
    }


def ingest_markdown_report(
    conn: sqlite3.Connection,
    markdown_text: str,
    title: str | None = None,
    idempotency_key: str | None = None,
    collector: str = "agent",
    blob_store: BlobStore | None = None,
) -> dict[str, Any]:
    """Ingest a raw Markdown research report as a first-class resource, leaving finding extraction pending."""
    clean_md = markdown_text.strip()
    if not clean_md:
        raise ValueError("Markdown report cannot be empty")

    now_iso = datetime.datetime.now(datetime.UTC).isoformat()
    req_hash = hashlib.sha256(clean_md.encode("utf-8")).hexdigest()

    if idempotency_key:
        row = conn.execute(
            """
            SELECT result_object_id, request_hash
            FROM idempotency_keys
            WHERE namespace = 'markdown_report' AND operation = 'ingest' AND key = ?;
            """,
            (idempotency_key,),
        ).fetchone()
        if row:
            if row["request_hash"] == req_hash:
                return {
                    "status": "replayed",
                    "resource_id": row["result_object_id"],
                }
            else:
                raise IdempotencyConflictError(
                    f"Idempotency key '{idempotency_key}' already used with different content"
                )

    # Extract title from first markdown header if not supplied
    report_title = title
    if not report_title:
        h1_match = re.search(r"^#\s+(.+)$", clean_md, re.MULTILINE)
        if h1_match:
            report_title = h1_match.group(1).strip()
        else:
            report_title = "Research Report"

    # Create Resource
    res_id = generate_id("res")
    ident_key = f"report:{req_hash[:16]}"
    conn.execute(
        """
        INSERT INTO resources (
            id, identity_key, title, primary_form, review_state, is_deleted, created_at, updated_at
        ) VALUES (?, ?, ?, 'research-report', 'unreviewed', 0, ?, ?);
        """,
        (res_id, ident_key, report_title, now_iso, now_iso),
    )

    # Create Capture
    cap_id = generate_id("cap")
    conn.execute(
        """
        INSERT INTO captures (
            id, origin_namespace, origin_id, collection_channel, collector,
            acquisition_method, retrieved_at, raw_content, user_note, review_state, created_at, updated_at
        ) VALUES (?, 'agent', ?, 'agent', ?, 'markdown-report', ?, ?, ?, 'unreviewed', ?, ?);
        """,
        (cap_id, ident_key, collector, now_iso, clean_md, report_title, now_iso, now_iso),
    )
    conn.execute(
        """
        INSERT INTO capture_resources (capture_id, resource_id, relationship_type, created_at)
        VALUES (?, ?, 'primary', ?);
        """,
        (cap_id, res_id, now_iso),
    )

    # Store Content
    store_resource_content(
        conn,
        res_id,
        clean_text=clean_md,
        summary=clean_md[:300] + "..." if len(clean_md) > 300 else clean_md,
        extractor="markdown-report",
        extractor_version="1.0",
        title=report_title,
        capture_id=cap_id,
    )

    if blob_store:
        save_source_snapshot(
            conn,
            blob_store,
            res_id,
            clean_md.encode("utf-8"),
            headers={"content-type": "text/markdown; charset=utf-8"},
            capture_id=cap_id,
        )

    # Enqueue durable pending finding-extraction job representing markdown report in pipeline
    fe_job_id = generate_id("job")
    fe_job_key = make_job_key("finding-extraction", res_id)
    conn.execute(
        """
        INSERT INTO processing_jobs (
            id, job_key, capture_id, resource_id, stage, status,
            available_at, attempts, input_hash, created_at, updated_at
        ) VALUES (?, ?, ?, ?, 'finding-extraction', 'pending', ?, 0, ?, ?, ?)
        ON CONFLICT(job_key) DO NOTHING;
        """,
        (fe_job_id, fe_job_key, cap_id, res_id, now_iso, req_hash, now_iso, now_iso),
    )

    reindex_object_document(conn, "capture", cap_id)

    if idempotency_key:
        idk_id = generate_id("idk")
        conn.execute(
            """
            INSERT INTO idempotency_keys (id, namespace, operation, key, result_object_id, request_hash, created_at)
            VALUES (?, 'markdown_report', 'ingest', ?, ?, ?, ?);
            """,
            (idk_id, idempotency_key, res_id, req_hash, now_iso),
        )

    record_audit_event(
        conn,
        event_type="report.ingested",
        object_type="resource",
        object_id=res_id,
        actor=collector,
        payload={"title": report_title, "char_count": len(clean_md)},
    )

    return {
        "status": "imported",
        "format": "markdown",
        "resource_id": res_id,
        "capture_id": cap_id,
        "title": report_title,
    }
