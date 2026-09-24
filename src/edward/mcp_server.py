"""Edward Model Context Protocol (MCP) Server.

Exposes Edward's research memory, discovery, writing workspaces, and capture capabilities
as typed tools for AI agents (Antigravity, Claude Desktop, Cursor, Codex, etc.).
"""

from __future__ import annotations

import json
import logging
import mimetypes
from pathlib import Path
from typing import Any

from mcp.server.mcpserver import MCPServer

from edward.blobs import BlobStore
from edward.db import Database, get_default_data_dir, get_default_db_path
from edward.models import CaptureInput
from edward.services.answer import answer_question
from edward.services.bundle import import_research_bundle, ingest_markdown_report
from edward.services.capture import capture_item
from edward.services.diagnostics import run_doctor
from edward.services.intents import accept_intent, load_intent_questions
from edward.services.lifecycle import (
    add_annotation,
    add_intent,
    add_label,
    get_annotations,
    remove_intent,
)
from edward.services.processor import get_processing_status, process_pending_jobs
from edward.services.projects import (
    add_project_note,
    add_project_object,
    create_project,
    delete_project,
    generate_outline_proposal,
    get_project_context,
    save_outline,
    set_outline_status,
)
from edward.services.search import search_lexical
from edward.services.source_adapters import sync_self_sent_gmail, sync_x_bookmarks

logger = logging.getLogger(__name__)


def _get_services(
    db: Database | None = None, blob_store: BlobStore | None = None
) -> tuple[Database, BlobStore]:
    """Resolve Database and BlobStore instances, running migrations if needed."""
    if db is None:
        db = Database(get_default_db_path())
        db.run_migrations()
    if blob_store is None:
        data_dir = get_default_data_dir()
        blob_store = BlobStore(data_dir / "blobs")
    return db, blob_store


def create_mcp_server(
    db: Database | None = None,
    blob_store: BlobStore | None = None,
    server_name: str = "edward",
) -> MCPServer:
    """Create and configure the Edward MCP server with full CLI parity tools."""
    server = MCPServer(
        server_name,
        instructions=(
            "Edward is a personal research memory and writing workspace. "
            "Use these tools to search saved research (exact lexical and semantic hybrid), "
            "retrieve structured Evidence Packets with citations, capture links and thoughts, "
            "manage intent facets, and organize evidence into writing projects with validated outlines."
        ),
    )

    # -------------------------------------------------------------------------
    # 1. Search, Discovery & Answering
    # -------------------------------------------------------------------------

    @server.tool(name="edward_ask")
    def edward_ask(query: str, limit: int = 50, project: str | None = None) -> dict[str, Any]:
        """Ask a question or research topic against Edward memory.

        Returns either direct factual lookups (Tier 1 counts/dates) or a high-recall,
        citation-backed Evidence Packet (Tier 2) containing verified passages, locators,
        source metadata, and research gaps for synthesis.
        """
        database, _ = _get_services(db, blob_store)
        with database.connection() as conn:
            ans = answer_question(
                conn,
                query=query,
                no_model=True,
                limit=limit,
                project_id=project,
            )
            return ans

    @server.tool(name="edward_search")
    def edward_search(
        query: str,
        topic: str | None = None,
        intent: str | None = None,
        form: str | None = None,
        project: str | None = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        """Perform exact lexical full-text search across Edward captures, resources, and findings.

        Supports filtering by topic (e.g. 'software/systems', 'crypto/bitcoin', 'infrastructure/energy-grid'),
        intent ('essay-seed', 'deep-dive', 'counterevidence', 'fact-check', 'tool-eval', 'inspiration'),
        form ('article', 'paper', 'x-post', 'repository'), or project ID.
        """
        database, _ = _get_services(db, blob_store)
        with database.connection() as conn:
            response = search_lexical(
                conn,
                query=query,
                limit=limit,
                intent=intent,
                topic=topic,
                form=form,
                project=project,
            )
            return response.model_dump()

    @server.tool(name="edward_show")
    def edward_show(object_id: str) -> dict[str, Any]:
        """Inspect the full details of any object (resource, capture, or finding) by ID.

        Returns metadata, clean extracted text, source URL, author, labels, active intents,
        linked captures, and annotations.
        """
        database, _ = _get_services(db, blob_store)
        with database.connection() as conn:
            # Check resource
            r_row = conn.execute("SELECT * FROM resources WHERE id = ?;", (object_id,)).fetchone()
            if r_row:
                data = dict(r_row)
                rc = conn.execute(
                    "SELECT clean_text, summary FROM resource_contents WHERE resource_id = ? ORDER BY created_at DESC LIMIT 1;",
                    (object_id,),
                ).fetchone()
                if rc:
                    data["content"] = dict(rc)
                labels = conn.execute(
                    "SELECT label_id, source FROM object_labels WHERE object_type = 'resource' AND object_id = ?;",
                    (object_id,),
                ).fetchall()
                data["labels"] = [dict(lbl) for lbl in labels]
                intents = conn.execute(
                    "SELECT intent, source FROM intents WHERE object_type = 'resource' AND object_id = ? AND is_active = 1;",
                    (object_id,),
                ).fetchall()
                data["intents"] = [dict(it) for it in intents]
                caps = conn.execute(
                    """
                    SELECT c.id, c.user_note, c.created_at, c.collector,
                           c.origin_namespace, c.origin_id, c.collection_channel,
                           c.acquisition_method, c.retrieved_at
                    FROM captures c
                    JOIN capture_resources cr ON cr.capture_id = c.id
                    WHERE cr.resource_id = ?;
                    """,
                    (object_id,),
                ).fetchall()
                data["captures"] = [dict(c) for c in caps]
                data["annotations"] = [
                    a.model_dump() for a in get_annotations(conn, "resource", object_id)
                ]
                return {"type": "resource", "data": data}

            # Check capture
            c_row = conn.execute("SELECT * FROM captures WHERE id = ?;", (object_id,)).fetchone()
            if c_row:
                data = dict(c_row)
                res_link = conn.execute(
                    "SELECT resource_id FROM capture_resources WHERE capture_id = ?;",
                    (object_id,),
                ).fetchone()
                data["resource_id"] = res_link["resource_id"] if res_link else None
                intents = conn.execute(
                    "SELECT intent, source FROM intents WHERE object_type = 'capture' AND object_id = ? AND is_active = 1;",
                    (object_id,),
                ).fetchall()
                data["intents"] = [dict(it) for it in intents]
                data["annotations"] = [
                    a.model_dump() for a in get_annotations(conn, "capture", object_id)
                ]
                return {"type": "capture", "data": data}

            # Check finding
            f_row = conn.execute("SELECT * FROM findings WHERE id = ?;", (object_id,)).fetchone()
            if f_row:
                data = dict(f_row)
                fs = conn.execute(
                    "SELECT passage, locator_json FROM finding_support WHERE finding_id = ? ORDER BY created_at ASC;",
                    (object_id,),
                ).fetchall()
                data["support"] = [dict(s) for s in fs]
                data["annotations"] = [
                    a.model_dump() for a in get_annotations(conn, "finding", object_id)
                ]
                return {"type": "finding", "data": data}

        return {"error": f"Object with ID '{object_id}' not found"}

    @server.tool(name="edward_export_packet")
    def edward_export_packet(query: str, limit: int = 50) -> dict[str, Any]:
        """Export a bounded, standalone Evidence Packet JSON for any topic or query."""
        database, _ = _get_services(db, blob_store)
        with database.connection() as conn:
            search_res = search_lexical(conn, query=query, limit=limit)
            items = []
            for s in search_res.results:
                items.append(
                    {
                        "id": s.id,
                        "kind": s.object_type,
                        "text": s.snippet or s.title or "",
                        "review_state": "unreviewed",
                        "labels": s.labels,
                        "entities": s.entities,
                    }
                )

            now_str = conn.execute("SELECT datetime('now');").fetchone()[0]
            evidence_packet = {
                "type": "evidence-packet",
                "schema_version": "1",
                "query": query,
                "created_at": now_str,
                "parameters": {"limit": limit},
                "items": items,
                "gaps_and_disagreements": [],
            }
            return evidence_packet

    # -------------------------------------------------------------------------
    # 2. Ingest & Capture
    # -------------------------------------------------------------------------

    @server.tool(name="edward_add")
    def edward_add(
        url: str | None = None,
        text: str | None = None,
        note: str | None = None,
        intent: str | None = None,
        file_path: str | None = None,
        origin: str = "agent",
    ) -> dict[str, Any]:
        """Capture a URL, thought note, local text, or file into Edward research memory.

        Saves the capture and context immediately and queues background jobs for text
        extraction and local vector embedding.
        """
        captured_text = text
        attachment_info: dict[str, Any] | None = None
        database, blobs = _get_services(db, blob_store)

        if file_path:
            p = Path(file_path).expanduser().resolve()
            if not p.exists():
                return {"error": f"File not found: {file_path}"}
            file_bytes = p.read_bytes()
            content_hash, blob_path = blobs.store_bytes(file_bytes)
            mime_type, _ = mimetypes.guess_type(p.name)
            mime_type = mime_type or "application/octet-stream"
            attachment_info = {
                "file_name": p.name,
                "mime_type": mime_type,
                "content_hash": content_hash,
                "size_bytes": len(file_bytes),
                "blob_path": f"{content_hash[:2]}/{content_hash}",
            }
            if captured_text is None:
                try:
                    captured_text = file_bytes.decode("utf-8")
                except UnicodeDecodeError:
                    captured_text = None

        if not url and not captured_text and not note and not attachment_info:
            return {"error": "At least one of url, text, note, or file_path must be provided."}

        input_data = CaptureInput(
            url=url,
            text=captured_text,
            note=note,
            intent=intent,
            origin_namespace=origin,
            collector="edward-mcp",
        )

        with database.transaction() as conn:
            try:
                res = capture_item(conn, input_data, attachment_info=attachment_info)
                return res
            except Exception as e:
                return {"error": str(e)}

    @server.tool(name="edward_import_research")
    def edward_import_research(
        content: str,
        format: str = "json",
        title: str | None = None,
        collector: str = "agent",
    ) -> dict[str, Any]:
        """Import a Markdown research report or a structured research bundle JSON."""
        database, blobs = _get_services(db, blob_store)
        fmt = format.lower().strip()
        try:
            with database.transaction() as conn:
                if fmt == "json":
                    bundle_dict = json.loads(content)
                    return import_research_bundle(conn, blobs, bundle_dict)
                elif fmt == "markdown":
                    return ingest_markdown_report(
                        conn, content, title=title, collector=collector, blob_store=blobs
                    )
                else:
                    return {"error": f"Unsupported format '{format}'. Use 'json' or 'markdown'."}
        except Exception as e:
            return {"error": str(e)}

    @server.tool(name="edward_sync")
    def edward_sync(
        source: str = "x",
        all_records: bool = False,
        limit: int = 25,
    ) -> dict[str, Any]:
        """Trigger an on-demand sync from local source archives ('x' for Birdclaw X bookmarks, 'gmail' for self-sent emails)."""
        database, blobs = _get_services(db, blob_store)
        src = source.lower().strip()
        try:
            if src == "x":
                imported = sync_x_bookmarks(database, blobs, all_records=all_records, limit=limit)
                return {"status": "synced", "source": "x", "imported_count": len(imported)}
            elif src == "gmail":
                imported = sync_self_sent_gmail(database, blobs)
                return {"status": "synced", "source": "gmail", "imported_count": len(imported)}
            else:
                return {"error": f"Unknown source '{source}'. Supported: 'x', 'gmail'."}
        except Exception as e:
            return {"error": str(e)}

    # -------------------------------------------------------------------------
    # 3. Intents & Annotations
    # -------------------------------------------------------------------------

    @server.tool(name="edward_annotate")
    def edward_annotate(
        object_id: str,
        note: str | None = None,
        intent: str | None = None,
        label: str | None = None,
        actor: str = "agent",
    ) -> dict[str, Any]:
        """Attach a user-authored note, intent facet, or taxonomic label to an existing object."""
        database, _ = _get_services(db, blob_store)
        with database.transaction() as conn:
            # Detect object type
            if conn.execute("SELECT 1 FROM resources WHERE id = ?;", (object_id,)).fetchone():
                obj_type = "resource"
            elif conn.execute("SELECT 1 FROM captures WHERE id = ?;", (object_id,)).fetchone():
                obj_type = "capture"
            elif conn.execute("SELECT 1 FROM findings WHERE id = ?;", (object_id,)).fetchone():
                obj_type = "finding"
            else:
                return {"error": f"Object with ID '{object_id}' not found"}

            if intent:
                add_intent(conn, obj_type, object_id, intent, source=actor, actor=actor)
            if label:
                add_label(conn, obj_type, object_id, label, source=actor, actor=actor)
            if note:
                add_annotation(
                    conn, obj_type, object_id, note, annotation_type="note", author=actor
                )

        return {
            "status": "annotated",
            "id": object_id,
            "note": note,
            "intent": intent,
            "label": label,
        }

    @server.tool(name="edward_remove_intent")
    def edward_remove_intent(object_id: str, intent: str, actor: str = "agent") -> dict[str, Any]:
        """Deactivate an active intent facet on an object."""
        database, _ = _get_services(db, blob_store)
        with database.transaction() as conn:
            obj_type = None
            if conn.execute("SELECT 1 FROM resources WHERE id = ?;", (object_id,)).fetchone():
                obj_type = "resource"
            elif conn.execute("SELECT 1 FROM captures WHERE id = ?;", (object_id,)).fetchone():
                obj_type = "capture"
            elif conn.execute("SELECT 1 FROM findings WHERE id = ?;", (object_id,)).fetchone():
                obj_type = "finding"

            if not obj_type:
                return {"error": f"Object with ID '{object_id}' not found"}

            try:
                remove_intent(conn, obj_type, object_id, intent, actor=actor)
                return {"status": "deactivated", "id": object_id, "intent": intent}
            except Exception as e:
                return {"error": f"Active intent '{intent}' not found on object '{object_id}': {e}"}

    @server.tool(name="edward_accept_intent")
    def edward_accept_intent(object_id: str, intent: str) -> dict[str, Any]:
        """Accept a machine-suggested intent, converting it into an active human-owned decision."""
        database, _ = _get_services(db, blob_store)
        with database.transaction() as conn:
            # Detect object type
            obj_type = None
            if conn.execute("SELECT 1 FROM resources WHERE id = ?;", (object_id,)).fetchone():
                obj_type = "resource"
            elif conn.execute("SELECT 1 FROM captures WHERE id = ?;", (object_id,)).fetchone():
                obj_type = "capture"
            elif conn.execute("SELECT 1 FROM findings WHERE id = ?;", (object_id,)).fetchone():
                obj_type = "finding"

            if not obj_type:
                return {"error": f"Object with ID '{object_id}' not found"}

            try:
                accept_intent(conn, object_type=obj_type, object_id=object_id, intent=intent)
                return {"status": "accepted", "id": object_id, "intent": intent}
            except Exception as e:
                return {"error": f"Could not accept intent '{intent}' on object '{object_id}': {e}"}

    @server.tool(name="edward_list_intents")
    def edward_list_intents() -> dict[str, Any]:
        """List all packaged intent questions and their meanings."""
        intents = load_intent_questions(include_inactive=True)
        return {"intents": intents, "count": len(intents)}

    # -------------------------------------------------------------------------
    # 4. Writing Project Workspaces
    # -------------------------------------------------------------------------

    @server.tool(name="edward_project_list")
    def edward_project_list() -> dict[str, Any]:
        """List all active writing and research projects."""
        database, _ = _get_services(db, blob_store)
        with database.connection() as conn:
            rows = conn.execute(
                """
                SELECT id, title, slug, description AS brief, status, created_at, updated_at
                FROM projects WHERE is_deleted = 0 ORDER BY updated_at DESC;
                """
            ).fetchall()
            projects = [dict(r) for r in rows]
            return {"projects": projects, "count": len(projects)}

    @server.tool(name="edward_project_create")
    def edward_project_create(
        title: str, brief: str | None = None, slug: str | None = None
    ) -> dict[str, Any]:
        """Create a new research and writing project workspace."""
        database, _ = _get_services(db, blob_store)
        try:
            with database.transaction() as conn:
                project = create_project(conn, title=title, brief=brief, slug=slug)
                return project.model_dump(mode="json")
        except Exception as e:
            return {"error": str(e)}

    @server.tool(name="edward_project_context")
    def edward_project_context(project_id: str, include_rejected: bool = False) -> dict[str, Any]:
        """Retrieve the complete working context of a project.

        Returns the project premise, accepted evidence, candidate sources, open research
        questions and gaps, counterarguments, and the latest outline revision.
        """
        database, _ = _get_services(db, blob_store)
        with database.connection() as conn:
            try:
                return get_project_context(conn, project_id, include_rejected=include_rejected)
            except Exception as e:
                return {"error": str(e)}

    @server.tool(name="edward_project_add_evidence")
    def edward_project_add_evidence(
        project_id: str,
        object_id: str,
        relationship: str = "supporting",
        note: str | None = None,
        actor: str = "agent",
    ) -> dict[str, Any]:
        """Add a resource, capture, or finding to a project workspace.

        Relationship should be 'supporting', 'counterargument', or 'qualification'.
        """
        database, _ = _get_services(db, blob_store)
        with database.transaction() as conn:
            try:
                return add_project_object(
                    conn,
                    project_id=project_id,
                    object_id=object_id,
                    relationship=relationship,
                    membership_status="accepted",
                    added_by=actor,
                    relevance_note=note,
                )
            except Exception as e:
                return {"error": str(e)}

    @server.tool(name="edward_project_remove_evidence")
    def edward_project_remove_evidence(project_id: str, object_id: str) -> dict[str, Any]:
        """Remove a piece of evidence from a project workspace."""
        database, _ = _get_services(db, blob_store)
        with database.transaction() as conn:
            conn.execute(
                "DELETE FROM project_objects WHERE project_id = ? AND object_id = ?;",
                (project_id, object_id),
            )
            return {"status": "removed", "project_id": project_id, "object_id": object_id}

    @server.tool(name="edward_project_add_note")
    def edward_project_add_note(
        project_id: str,
        kind: str,
        text: str,
        author: str = "agent",
    ) -> dict[str, Any]:
        """Record an open research question, gap, or counterargument note to a project workspace.

        Kind should be 'gap', 'question', 'counterargument', or 'note'.
        """
        database, _ = _get_services(db, blob_store)
        with database.transaction() as conn:
            try:
                return add_project_note(
                    conn, project_id=project_id, kind=kind, content=text, author=author
                )
            except Exception as e:
                return {"error": str(e)}

    @server.tool(name="edward_project_propose_outline")
    def edward_project_propose_outline(
        project_id: str,
        outline: dict[str, Any] | None = None,
        author_id: str = "agent",
    ) -> dict[str, Any]:
        """Propose a new versioned outline for a writing project.

        If outline dict is provided, it validates that all cited object IDs exist in the project
        evidence. If outline is omitted, Edward deterministically groups existing accepted evidence
        and research gaps into draft sections.
        """
        database, _ = _get_services(db, blob_store)
        with database.transaction() as conn:
            try:
                if outline:
                    return save_outline(
                        conn,
                        project_id=project_id,
                        proposal=outline,
                        author_type="calling-agent",
                        author_id=author_id,
                    )
                else:
                    context = get_project_context(conn, project_id)
                    proposal = generate_outline_proposal(context, None)
                    return save_outline(
                        conn,
                        project_id=project_id,
                        proposal=proposal,
                        author_type="system",
                        author_id=author_id,
                    )
            except Exception as e:
                return {"error": str(e)}

    @server.tool(name="edward_project_accept_outline")
    def edward_project_accept_outline(
        project_id: str,
        version: int | None = None,
        outline_id: str | None = None,
        actor: str = "agent",
    ) -> dict[str, Any]:
        """Accept and commit a proposed outline revision as the project's official outline."""
        database, _ = _get_services(db, blob_store)
        with database.transaction() as conn:
            try:
                target_version = version
                if target_version is None and outline_id:
                    row = conn.execute(
                        "SELECT version FROM outlines WHERE id = ? AND project_id = ?;",
                        (outline_id, project_id),
                    ).fetchone()
                    if row:
                        target_version = row["version"]
                return set_outline_status(
                    conn,
                    project_id=project_id,
                    status="accepted",
                    version=target_version,
                    actor=actor,
                )
            except Exception as e:
                return {"error": str(e)}

    @server.tool(name="edward_project_delete")
    def edward_project_delete(project_id: str, confirm: bool = False) -> dict[str, Any]:
        """Soft-delete a project workspace."""
        if not confirm:
            return {"error": "Project deletion requires explicit confirm=True"}
        database, _ = _get_services(db, blob_store)
        with database.transaction() as conn:
            try:
                res = delete_project(conn, project_id)
                res["status"] = "deleted"
                return res
            except Exception as e:
                return {"error": str(e)}

    # -------------------------------------------------------------------------
    # 5. Queue, Diagnostics & Maintenance
    # -------------------------------------------------------------------------

    @server.tool(name="edward_status")
    def edward_status() -> dict[str, Any]:
        """Report background processing queue status, pending/failed jobs, and corpus counts."""
        database, _ = _get_services(db, blob_store)
        with database.connection() as conn:
            proc_status = get_processing_status(conn)
            captures_c = conn.execute(
                "SELECT COUNT(*) FROM captures WHERE is_deleted = 0;"
            ).fetchone()[0]
            resources_c = conn.execute(
                "SELECT COUNT(*) FROM resources WHERE is_deleted = 0;"
            ).fetchone()[0]
            findings_c = conn.execute(
                "SELECT COUNT(*) FROM findings WHERE is_deleted = 0;"
            ).fetchone()[0]
            attachments_c = conn.execute("SELECT COUNT(*) FROM attachments;").fetchone()[0]

            return {
                "processing_queue": proc_status,
                "corpus": {
                    "captures": captures_c,
                    "resources": resources_c,
                    "findings": findings_c,
                    "attachments": attachments_c,
                },
            }

    @server.tool(name="edward_process")
    def edward_process(limit: int = 50) -> dict[str, Any]:
        """Execute pending background extraction, classification, and local embedding jobs."""
        database, blobs = _get_services(db, blob_store)
        try:
            return process_pending_jobs(database, blobs, limit=limit)
        except Exception as e:
            return {"error": str(e)}

    @server.tool(name="edward_doctor")
    def edward_doctor() -> dict[str, Any]:
        """Run system diagnostics verifying database integrity, migrations, and blob store health."""
        database, blobs = _get_services(db, blob_store)
        try:
            return run_doctor(database, blobs)
        except Exception as e:
            return {"error": str(e), "healthy": False}

    return server
