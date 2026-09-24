"""Edward Command-Line Interface (Typer)."""

import json
import mimetypes
import sys
from pathlib import Path
from typing import Any, NoReturn

import typer
from rich.console import Console

from edward.blobs import BlobStore
from edward.db import Database, get_default_data_dir, get_default_db_path
from edward.models import CaptureInput
from edward.services.backup import create_backup
from edward.services.bundle import import_research_bundle, ingest_markdown_report
from edward.services.capture import IdempotencyConflictError, capture_item
from edward.services.diagnostics import run_doctor
from edward.services.lifecycle import (
    ObjectNotFoundError,
    add_annotation,
    add_intent,
    add_label,
    get_annotations,
    prune_unreferenced_blobs,
    purge_object,
    remove_intent,
)
from edward.services.privacy import PrivacyTransmissionError
from edward.services.processor import (
    count_jobs,
    get_processing_status,
    process_pending_jobs,
    purge_jobs,
    reconcile_completed_jobs,
    retry_failed_jobs,
)
from edward.services.projects import (
    ProjectConflictError,
    ProjectError,
    add_project_note,
    add_project_object,
    create_project,
    delete_project,
    generate_outline_proposal,
    get_outline,
    get_project_context,
    save_outline,
    set_outline_status,
    suggest_project_evidence,
)
from edward.services.search import search_lexical
from edward.services.source_adapters import (
    sync_self_sent_gmail,
    sync_x_bookmarks,
)

app = typer.Typer(
    name="edward",
    help="Edward - Personal Research Memory and Writing Workspace",
    no_args_is_help=True,
    add_completion=False,
)
project_app = typer.Typer(help="Manage active research and writing workspaces.")
app.add_typer(project_app, name="project")
sync_app = typer.Typer(help="Read-only imports from local source archives and accounts.")
app.add_typer(sync_app, name="sync")

err_console = Console(stderr=True)
out_console = Console()


def get_services(db_path: Path | None = None) -> tuple[Database, BlobStore]:
    """Initialize and migrate database, and return Database and BlobStore instances."""
    data_dir = get_default_data_dir()
    database = Database(db_path or get_default_db_path())
    database.run_migrations()
    blob_store = BlobStore(data_dir / "blobs")
    return database, blob_store


def output_json_payload(data: Any, exit_code: int = 0) -> None:
    """Output clean JSON on stdout and exit."""
    if hasattr(data, "model_dump"):
        dumped = data.model_dump()
    elif isinstance(data, dict):
        dumped = data
    else:
        dumped = data
    sys.stdout.write(json.dumps(dumped, indent=2, default=str) + "\n")
    sys.stdout.flush()
    raise typer.Exit(code=exit_code)


def handle_error(msg: str, exit_code: int = 1, as_json: bool = False) -> NoReturn:
    """Report error and exit.

    Annotated NoReturn because this always raises: it never returns to its caller, so
    the name it was guarding is unbound afterwards.
    """
    if as_json:
        sys.stderr.write(json.dumps({"error": msg}) + "\n")
    else:
        err_console.print(f"[bold red]Error:[/bold red] {msg}")
    raise typer.Exit(code=exit_code)


@sync_app.command("x")
def sync_x_command(
    limit: int = typer.Option(25, "--limit", min=1, help="Maximum new bookmarks to import"),
    all_bookmarks: bool = typer.Option(
        False, "--all", help="Import all new bookmarks from the archive"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Report counts without fetching or importing posts"
    ),
    json_mode: bool = typer.Option(False, "--json", help="Emit machine-readable JSON"),
) -> None:
    """Import new X bookmarks from the local Birdclaw archive."""
    try:
        database, blob_store = get_services()
        result = sync_x_bookmarks(
            database, blob_store, limit=None if all_bookmarks else limit, dry_run=dry_run
        )
    except Exception as exc:
        handle_error(str(exc), as_json=json_mode)
    if json_mode:
        output_json_payload(result)
    else:
        out_console.print(
            f"X bookmarks: {result['captured']} imported; "
            f"{result['already_imported']} already present; "
            f"{result['linked_urls']} linked pages attached."
        )
        if result["reader_errors"]:
            err_console.print(f"{len(result['reader_errors'])} post(s) had reader issues.")


@sync_app.command("gmail")
def sync_gmail_command(
    limit: int | None = typer.Option(
        None, "--limit", min=1, help="Maximum threads to inspect; default scans all results"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Report discovery counts without fetching threads"
    ),
    download_attachments: bool = typer.Option(
        False,
        "--download-attachments",
        help="Download matching attachments into Edward's local blob store",
    ),
    json_mode: bool = typer.Option(False, "--json", help="Emit machine-readable JSON"),
) -> None:
    """Import self-sent Gmail messages through gog's read-only mode."""
    try:
        database, blob_store = get_services()
        result = sync_self_sent_gmail(
            database,
            blob_store,
            limit=limit,
            dry_run=dry_run,
            download_attachments=download_attachments,
        )
    except Exception as exc:
        handle_error(str(exc), as_json=json_mode)
    if json_mode:
        output_json_payload(result)
    else:
        out_console.print(
            f"Gmail: {result['captured']} messages imported; "
            f"{result['already_imported']} already present; "
            f"{result['linked_urls']} linked pages attached; "
            f"{result['rejected_not_self_sent']} non-self-sent messages skipped."
        )


@app.command("add")
def add_command(
    url: str | None = typer.Option(None, "--url", "-u", help="URL to capture"),
    text: str | None = typer.Option(None, "--text", "-t", help="Raw text or note content"),
    file: Path | None = typer.Option(None, "--file", "-f", help="Path to file to capture"),
    stdin: bool = typer.Option(False, "--stdin", help="Read content from stdin"),
    note: str | None = typer.Option(None, "--note", "-n", help="Personal contextual note"),
    intent: str | None = typer.Option(
        None, "--intent", "-i", help="Initial intent (e.g. essay-seed)"
    ),
    origin: str = typer.Option("manual", "--origin", help="Origin namespace (web, arxiv, x, etc.)"),
    collector: str = typer.Option(
        "edward-cli", "--collector", help="Collector or agent identifier"
    ),
    idempotency_key: str | None = typer.Option(
        None, "--idempotency-key", help="Unique idempotency key"
    ),
    interactive: bool = typer.Option(
        False, "--interactive", help="Prompt interactively for capture fields"
    ),
    json_mode: bool = typer.Option(False, "--json", help="Output machine-readable JSON on stdout"),
) -> None:
    """Capture a URL, file, or raw text into memory."""
    if interactive:
        if not url:
            val = typer.prompt(
                "URL (press Enter to skip)", default="", show_default=False, err=True
            ).strip()
            if val:
                url = val
        if not text and not file and not stdin:
            val = typer.prompt(
                "Text / content (press Enter to skip)",
                default="",
                show_default=False,
                err=True,
            ).strip()
            if val:
                text = val
        if not note:
            val = typer.prompt(
                "Note / context (press Enter to skip)",
                default="",
                show_default=False,
                err=True,
            ).strip()
            if val:
                note = val
        if not intent:
            val = typer.prompt(
                "Intent (press Enter to skip)", default="", show_default=False, err=True
            ).strip()
            if val:
                intent = val

    captured_text = text
    attachment_info: dict[str, Any] | None = None
    db, blob_store = get_services()

    if file:
        if not file.exists():
            handle_error(f"File not found: {file}", exit_code=1, as_json=json_mode)
        file_bytes = file.read_bytes()
        content_hash, blob_path = blob_store.store_bytes(file_bytes)
        mime_type, _ = mimetypes.guess_type(file.name)
        mime_type = mime_type or "application/octet-stream"
        attachment_info = {
            "file_name": file.name,
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
    elif stdin:
        captured_text = sys.stdin.read()

    if not url and not captured_text and not note and not attachment_info:
        handle_error(
            "At least one of --url, --text, --file, --stdin, or --note must be provided",
            exit_code=2,
            as_json=json_mode,
        )

    input_data = CaptureInput(
        url=url,
        text=captured_text,
        note=note,
        intent=intent,
        origin_namespace=origin,
        collector=collector,
        idempotency_key=idempotency_key,
    )

    try:
        with db.transaction() as conn:
            result = capture_item(conn, input_data, attachment_info=attachment_info)
    except IdempotencyConflictError as e:
        handle_error(str(e), exit_code=3, as_json=json_mode)
    except ValueError as e:
        handle_error(str(e), exit_code=1, as_json=json_mode)
    except Exception as e:
        handle_error(f"Failed to capture item: {e}", exit_code=1, as_json=json_mode)

    if json_mode:
        output_json_payload(result)
    else:
        out_console.print(
            f"[bold green]Captured successfully![/bold green] (ID: {result['capture_id']})"
        )
        if result.get("resource_id"):
            out_console.print(f"Resource ID: [cyan]{result['resource_id']}[/cyan]")
        if result.get("url"):
            out_console.print(f"URL: [link]{result['url']}[/link]")


@app.command("show")
def show_command(
    target_id: str = typer.Argument(..., help="Capture, Resource, or Finding ID"),
    json_mode: bool = typer.Option(False, "--json", help="Output machine-readable JSON on stdout"),
) -> None:
    """Inspect full details of an item by ID."""
    db, _ = get_services()

    with db.connection() as conn:
        # Check resources
        r_row = conn.execute("SELECT * FROM resources WHERE id = ?;", (target_id,)).fetchone()
        if r_row:
            data = dict(r_row)
            rc = conn.execute(
                "SELECT clean_text, summary FROM resource_contents WHERE resource_id = ? ORDER BY created_at DESC LIMIT 1;",
                (target_id,),
            ).fetchone()
            if rc:
                data["content"] = dict(rc)
            labels = conn.execute(
                "SELECT label_id, source FROM object_labels WHERE object_type = 'resource' AND object_id = ?;",
                (target_id,),
            ).fetchall()
            data["labels"] = [dict(lbl) for lbl in labels]
            intents = conn.execute(
                "SELECT intent, source FROM intents WHERE object_type = 'resource' AND object_id = ? AND is_active = 1;",
                (target_id,),
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
                (target_id,),
            ).fetchall()
            data["captures"] = [dict(c) for c in caps]
            data["annotations"] = [
                a.model_dump() for a in get_annotations(conn, "resource", target_id)
            ]

            if json_mode:
                output_json_payload(data)
            else:
                out_console.print(f"[bold cyan]Resource:[/bold cyan] {target_id}")
                out_console.print(f"Title: {data.get('title') or '(No title)'}")
                out_console.print(f"URL: {data.get('canonical_url') or '(None)'}")
                out_console.print(f"Review State: {data.get('review_state')}")
                out_console.print(
                    f"Intents: {', '.join([i['intent'] for i in data['intents']]) or 'none'}"
                )
                if data["annotations"]:
                    out_console.print("Annotations:")
                    for a in data["annotations"]:
                        out_console.print(
                            f"  - [{a['annotation_type']}] {a['content']} (by {a['author']})"
                        )
            return

        # Check captures
        c_row = conn.execute("SELECT * FROM captures WHERE id = ?;", (target_id,)).fetchone()
        if c_row:
            data = dict(c_row)
            res_link = conn.execute(
                "SELECT resource_id FROM capture_resources WHERE capture_id = ?;", (target_id,)
            ).fetchone()
            data["resource_id"] = res_link["resource_id"] if res_link else None
            data["annotations"] = [
                a.model_dump() for a in get_annotations(conn, "capture", target_id)
            ]

            if json_mode:
                output_json_payload(data)
            else:
                out_console.print(f"[bold cyan]Capture:[/bold cyan] {target_id}")
                out_console.print(f"Collector: {data.get('collector')}")
                out_console.print(f"Note: {data.get('user_note') or '(No note)'}")
                out_console.print(f"Linked Resource: {data.get('resource_id') or '(None)'}")
                if data["annotations"]:
                    out_console.print("Annotations:")
                    for a in data["annotations"]:
                        out_console.print(
                            f"  - [{a['annotation_type']}] {a['content']} (by {a['author']})"
                        )
            return

        # Check findings
        f_row = conn.execute("SELECT * FROM findings WHERE id = ?;", (target_id,)).fetchone()
        if f_row:
            data = dict(f_row)
            fs = conn.execute(
                "SELECT passage, locator_json FROM finding_support WHERE finding_id = ? ORDER BY created_at ASC;",
                (target_id,),
            ).fetchall()
            data["support"] = [dict(s) for s in fs]
            data["annotations"] = [
                a.model_dump() for a in get_annotations(conn, "finding", target_id)
            ]

            if json_mode:
                output_json_payload(data)
            else:
                out_console.print(f"[bold cyan]Finding:[/bold cyan] {target_id}")
                out_console.print(f"Statement: {data.get('statement')}")
                out_console.print(f"Assertion Role: {data.get('assertion_role')}")
                if data["annotations"]:
                    out_console.print("Annotations:")
                    for a in data["annotations"]:
                        out_console.print(
                            f"  - [{a['annotation_type']}] {a['content']} (by {a['author']})"
                        )
            return

    handle_error(f"Object with ID '{target_id}' not found", exit_code=1, as_json=json_mode)


@app.command("search")
def search_command(
    query: str = typer.Argument(..., help="Search query string"),
    intent: str | None = typer.Option(None, "--intent", "-i", help="Filter by active intent"),
    topic: str | None = typer.Option(None, "--topic", "-t", help="Filter by topic label"),
    form: str | None = typer.Option(None, "--form", help="Filter by form (article, paper, etc.)"),
    project: str | None = typer.Option(None, "--project", help="Filter by project ID"),
    limit: int = typer.Option(20, "--limit", "-l", help="Maximum results to return"),
    json_mode: bool = typer.Option(False, "--json", help="Output machine-readable JSON on stdout"),
) -> None:
    """Search captures, resources, and findings using FTS5."""
    db, _ = get_services()
    with db.connection() as conn:
        response = search_lexical(
            conn,
            query=query,
            limit=limit,
            intent=intent,
            topic=topic,
            form=form,
            project=project,
        )

    if json_mode:
        output_json_payload(response)
    else:
        out_console.print(f"[bold]Search Results for:[/bold] '{query}' ({response.count} matches)")
        for idx, item in enumerate(response.results, 1):
            out_console.print(
                f"\n{idx}. [{item.object_type.upper()}] [cyan]{item.id}[/cyan] - {item.title or '(Untitled)'}"
            )
            if item.snippet:
                out_console.print(f"   {item.snippet}")


@app.command("annotate")
def annotate_command(
    target_id: str = typer.Argument(..., help="Object ID to annotate"),
    note: str | None = typer.Option(None, "--note", "-n", help="Note to append"),
    intent: str | None = typer.Option(None, "--intent", "-i", help="Intent to add"),
    label: str | None = typer.Option(None, "--label", help="Taxonomic label to attach"),
    actor: str = typer.Option("human", "--actor", help="Actor recording annotation"),
    json_mode: bool = typer.Option(False, "--json", help="Output machine-readable JSON on stdout"),
) -> None:
    """Annotate an existing item with notes, intents, or labels."""
    if not note and not intent and not label:
        handle_error(
            "At least one of --note, --intent, or --label must be provided",
            exit_code=2,
            as_json=json_mode,
        )

    db, _ = get_services()
    with db.transaction() as conn:
        # Determine object type
        r = conn.execute("SELECT id FROM resources WHERE id = ?;", (target_id,)).fetchone()
        obj_type = "resource" if r else None
        if not obj_type:
            c = conn.execute("SELECT id FROM captures WHERE id = ?;", (target_id,)).fetchone()
            obj_type = "capture" if c else None
        if not obj_type:
            f = conn.execute("SELECT id FROM findings WHERE id = ?;", (target_id,)).fetchone()
            obj_type = "finding" if f else None

        if not obj_type:
            handle_error(f"Object with ID '{target_id}' not found", exit_code=1, as_json=json_mode)

        if intent:
            add_intent(conn, obj_type, target_id, intent, source=actor, actor=actor)

        if label:
            add_label(conn, obj_type, target_id, label, source=actor, actor=actor)

        if note:
            add_annotation(conn, obj_type, target_id, note, annotation_type="note", author=actor)

    res = {"status": "annotated", "id": target_id, "note": note, "intent": intent, "label": label}
    if json_mode:
        output_json_payload(res)
    else:
        out_console.print(f"[bold green]Annotated successfully![/bold green] (ID: {target_id})")


@app.command("remove-intent")
def remove_intent_command(
    target_id: str = typer.Argument(..., help="Object ID"),
    intent: str = typer.Argument(..., help="Intent name to remove"),
    json_mode: bool = typer.Option(False, "--json", help="Output machine-readable JSON on stdout"),
) -> None:
    """Deactivate an active intent on an object."""
    db, _ = get_services()
    try:
        with db.transaction() as conn:
            for obj_type in ["resource", "capture", "finding"]:
                try:
                    remove_intent(conn, obj_type, target_id, intent, actor="human")
                    break
                except ObjectNotFoundError:
                    continue
            else:
                handle_error(
                    f"Active intent '{intent}' not found on object '{target_id}'",
                    exit_code=1,
                    as_json=json_mode,
                )
    except Exception as e:
        handle_error(str(e), exit_code=1, as_json=json_mode)

    res = {"status": "removed", "id": target_id, "intent": intent}
    if json_mode:
        output_json_payload(res)
    else:
        out_console.print(f"[bold green]Removed intent '{intent}' from {target_id}[/bold green]")


@app.command("backup")
def backup_command(
    dest: Path | None = typer.Option(None, "--dest", "-d", help="Destination backup directory"),
    json_mode: bool = typer.Option(False, "--json", help="Output machine-readable JSON on stdout"),
) -> None:
    """Create a holistic backup of the SQLite database and all referenced blobs."""
    db, blob_store = get_services()
    try:
        summary = create_backup(db, blob_store, dest_path=dest)
    except Exception as e:
        handle_error(f"Backup failed: {e}", exit_code=1, as_json=json_mode)

    if json_mode:
        output_json_payload(summary)
    else:
        out_console.print("[bold green]Backup completed successfully![/bold green]")
        out_console.print(f"Location: {summary['backup_path']}")
        out_console.print(f"Database SHA-256: {summary['database_sha256']}")
        out_console.print(f"Blobs Backed Up: {summary['blobs_backed_up']}")


@app.command("doctor")
def doctor_command(
    backup: Path | None = typer.Option(None, "--backup", help="Path to backup directory to verify"),
    prune_blobs: bool = typer.Option(
        False, "--prune-blobs", help="Prune unreferenced blobs from blob store"
    ),
    json_mode: bool = typer.Option(False, "--json", help="Output machine-readable JSON on stdout"),
) -> None:
    """Run diagnostics on database, blobs, and migrations, or verify a backup."""
    db, blob_store = get_services()
    report = run_doctor(db, blob_store, backup_to_verify=backup)

    if prune_blobs:
        pruned = prune_unreferenced_blobs(db, blob_store)
        report["pruned_blobs"] = pruned

    is_healthy = report.get("healthy", True)
    if "result" in report:
        is_healthy = report["result"].get("valid", False)

    if json_mode:
        output_json_payload(report, exit_code=0 if is_healthy else 1)
    else:
        if is_healthy:
            out_console.print("[bold green]System health check passed cleanly![/bold green]")
        else:
            err_console.print("[bold red]Diagnostics detected issues:[/bold red]")
        if prune_blobs:
            out_console.print(f"Pruned {len(report.get('pruned_blobs', []))} unreferenced blobs.")
        sys.stdout.write(json.dumps(report, indent=2) + "\n")
        if not is_healthy:
            raise typer.Exit(code=1)


@app.command("purge")
def purge_command(
    target_id: str = typer.Argument(..., help="Object ID to permanently delete"),
    confirm: bool = typer.Option(
        False, "--confirm", help="Explicit confirmation required to permanently purge"
    ),
    json_mode: bool = typer.Option(False, "--json", help="Output machine-readable JSON on stdout"),
) -> None:
    """Permanently delete an object from Edward database and search projection."""
    if not confirm:
        handle_error(
            "Permanent deletion requires explicit --confirm flag", exit_code=2, as_json=json_mode
        )

    db, blob_store = get_services()
    with db.transaction() as conn:
        for obj_type in ["resource", "capture", "finding"]:
            try:
                purge_object(conn, obj_type, target_id)
                break
            except ObjectNotFoundError:
                continue
        else:
            handle_error(f"Object with ID '{target_id}' not found", exit_code=1, as_json=json_mode)

    pruned = prune_unreferenced_blobs(db, blob_store)
    res = {"status": "purged", "id": target_id, "pruned_blobs": pruned}
    if json_mode:
        output_json_payload(res)
    else:
        out_console.print(f"[bold red]Permanently purged object {target_id}[/bold red]")
        if pruned:
            out_console.print(f"Pruned {len(pruned)} unreferenced blobs.")


@app.command("export")
def export_command(
    format: str = typer.Option("jsonl", "--format", help="Export format: 'jsonl' or 'packet'"),
    packet: bool = typer.Option(
        False, "--packet", help="Export bounded evidence packet (alias for --format packet)"
    ),
    query: str = typer.Option("", "--query", "-q", help="Query filter for export"),
    out: Path | None = typer.Option(None, "--out", "-o", help="Output file path"),
    limit: int | None = typer.Option(
        None,
        "--limit",
        "-l",
        help="Maximum items to export (defaults to 100 for packet, unlimited for jsonl)",
    ),
    offset: int = typer.Option(0, "--offset", help="Number of items to skip for JSONL export"),
    json_mode: bool = typer.Option(False, "--json", help="Output machine-readable JSON summary"),
) -> None:
    """Export memory corpus as JSONL or bounded evidence packet."""
    db, _ = get_services()
    export_format = "packet" if packet else format.lower()

    if export_format == "packet":
        packet_limit = limit if limit is not None else 100
        with db.connection() as conn:
            search_res = search_lexical(conn, query=query, limit=packet_limit)
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

            evidence_packet = {
                "type": "evidence-packet",
                "schema_version": "1",
                "query": query,
                "created_at": conn.execute("SELECT datetime('now');").fetchone()[0],
                "parameters": {"limit": packet_limit},
                "items": items,
                "gaps_and_disagreements": [],
            }

        if out:
            out.write_text(json.dumps(evidence_packet, indent=2), encoding="utf-8")
            if not json_mode:
                out_console.print(f"[bold green]Exported evidence packet to {out}[/bold green]")
            else:
                output_json_payload({"status": "exported", "format": "packet", "path": str(out)})
        else:
            output_json_payload(evidence_packet)
    elif export_format == "jsonl":
        lines: list[dict[str, Any]] = []
        with db.connection() as conn:
            filter_ids: set[str] | None = None
            if query.strip():
                s_res = search_lexical(conn, query=query, limit=limit or 10000)
                filter_ids = {r.id for r in s_res.results}

            # 1. Resources
            r_rows = conn.execute(
                "SELECT * FROM resources WHERE is_deleted = 0 ORDER BY created_at ASC;"
            ).fetchall()
            active_resource_ids = {r["id"] for r in r_rows}
            matched_resource_ids = (
                {r_id for r_id in active_resource_ids if r_id in filter_ids}
                if filter_ids is not None
                else active_resource_ids
            )
            for r in r_rows:
                r_id = r["id"]
                if r_id not in matched_resource_ids:
                    continue
                r_dict = dict(r)
                rc = conn.execute(
                    "SELECT clean_text, summary FROM resource_contents WHERE resource_id = ? ORDER BY created_at DESC LIMIT 1;",
                    (r_id,),
                ).fetchone()
                if rc:
                    r_dict["content"] = dict(rc)
                lines.append({"object_type": "resource", "data": r_dict})

            # 2. Captures
            c_rows = conn.execute(
                "SELECT * FROM captures WHERE is_deleted = 0 ORDER BY created_at ASC;"
            ).fetchall()
            active_capture_ids = {c["id"] for c in c_rows}
            matched_capture_ids = (
                {c_id for c_id in active_capture_ids if c_id in filter_ids}
                if filter_ids is not None
                else active_capture_ids
            )
            for c in c_rows:
                c_id = c["id"]
                if c_id not in matched_capture_ids:
                    continue
                lines.append({"object_type": "capture", "data": dict(c)})

            # 3. Capture-Resource relationships
            cr_rows = conn.execute(
                "SELECT * FROM capture_resources ORDER BY capture_id ASC, resource_id ASC;"
            ).fetchall()
            for cr in cr_rows:
                if (
                    cr["capture_id"] in matched_capture_ids
                    or cr["resource_id"] in matched_resource_ids
                ):
                    lines.append({"object_type": "capture_resource", "data": dict(cr)})

            # 4. Findings
            f_rows = conn.execute(
                "SELECT * FROM findings WHERE is_deleted = 0 ORDER BY created_at ASC;"
            ).fetchall()
            active_finding_ids = {f["id"] for f in f_rows}
            matched_finding_ids = (
                {f_id for f_id in active_finding_ids if f_id in filter_ids}
                if filter_ids is not None
                else active_finding_ids
            )
            for f in f_rows:
                f_id = f["id"]
                if f_id not in matched_finding_ids:
                    continue
                f_dict = dict(f)
                fs = conn.execute(
                    "SELECT passage, locator_json FROM finding_support WHERE finding_id = ? ORDER BY created_at ASC;",
                    (f_id,),
                ).fetchall()
                f_dict["support"] = [dict(s) for s in fs]
                lines.append({"object_type": "finding", "data": f_dict})

            # Combined matched parent objects for metadata filtering
            matched_object_ids = matched_resource_ids | matched_capture_ids | matched_finding_ids

            # 5. Annotations
            ann_rows = conn.execute("SELECT * FROM annotations ORDER BY created_at ASC;").fetchall()
            for a in ann_rows:
                if a["object_id"] in matched_object_ids:
                    lines.append({"object_type": "annotation", "data": dict(a)})

            # 6. Intents
            intent_rows = conn.execute("SELECT * FROM intents ORDER BY created_at ASC;").fetchall()
            for it in intent_rows:
                if it["object_id"] in matched_object_ids:
                    lines.append({"object_type": "intent", "data": dict(it)})

            # 7. Labels
            label_rows = conn.execute(
                "SELECT * FROM object_labels ORDER BY created_at ASC;"
            ).fetchall()
            for lbl in label_rows:
                if lbl["object_id"] in matched_object_ids:
                    lines.append({"object_type": "object_label", "data": dict(lbl)})

            # 8. Entities
            entity_rows = conn.execute(
                "SELECT * FROM object_entities ORDER BY created_at ASC;"
            ).fetchall()
            for ent in entity_rows:
                if ent["object_id"] in matched_object_ids:
                    lines.append({"object_type": "object_entity", "data": dict(ent)})

            # 9. Attachments
            att_rows = conn.execute("SELECT * FROM attachments ORDER BY created_at ASC;").fetchall()
            for att in att_rows:
                if att["object_id"] in matched_object_ids:
                    lines.append({"object_type": "attachment", "data": dict(att)})

            # 10. Audit events
            audit_rows = conn.execute(
                "SELECT * FROM audit_events ORDER BY created_at ASC;"
            ).fetchall()
            for ae in audit_rows:
                if ae["object_id"] in matched_object_ids:
                    lines.append({"object_type": "audit_event", "data": dict(ae)})

        # Apply optional pagination
        if offset > 0:
            lines = lines[offset:]
        if limit is not None and len(lines) > limit:
            lines = lines[:limit]

        jsonl_content = "\n".join(json.dumps(item, default=str) for item in lines) + (
            "\n" if lines else ""
        )

        if out:
            out.write_text(jsonl_content, encoding="utf-8")
            if not json_mode:
                out_console.print(f"[bold green]Exported {len(lines)} items to {out}[/bold green]")
            else:
                output_json_payload(
                    {"status": "exported", "format": "jsonl", "path": str(out), "count": len(lines)}
                )
        else:
            if json_mode:
                output_json_payload(
                    {"status": "exported", "format": "jsonl", "count": len(lines), "items": lines}
                )
            else:
                sys.stdout.write(jsonl_content)
                sys.stdout.flush()
    else:
        handle_error(
            f"Unknown export format: '{export_format}'. Choose 'jsonl' or 'packet'.",
            exit_code=2,
            as_json=json_mode,
        )


@app.command("process")
def process_command(
    capture_id: str | None = typer.Option(
        None, "--capture-id", help="Filter processing jobs by capture ID"
    ),
    stage: str | None = typer.Option(
        None, "--stage", help="Filter processing jobs by pipeline stage"
    ),
    limit: int = typer.Option(10, "--limit", "-l", help="Maximum number of jobs to execute"),
    worker_id: str | None = typer.Option(
        None, "--worker-id", help="Worker identifier for lease acquisition"
    ),
    reconcile: bool = typer.Option(
        False,
        "--reconcile",
        help="Mark pending jobs whose targets are already completed as completed",
    ),
    json_mode: bool = typer.Option(False, "--json", help="Output machine-readable JSON on stdout"),
) -> None:
    db, blob_store = get_services()
    if reconcile:
        with db.transaction() as conn:
            rec_result = reconcile_completed_jobs(conn)
        if json_mode:
            output_json_payload(rec_result)
        else:
            out_console.print(
                f"[bold green]Reconciled jobs:[/bold green] {rec_result['reconciled_classify_jobs']} classify job(s) marked completed."
            )
        return

    valid_stages = (
        "resource-fetch",
        "extract",
        "classify",
        "finding-extraction",
        "embed",
        "attachment-extract",
        "attachment-ocr",
    )
    if stage and stage not in valid_stages:
        handle_error(
            f"Invalid stage '{stage}'. Supported stages: {', '.join(valid_stages)}",
            exit_code=1,
            as_json=json_mode,
        )

    result = process_pending_jobs(
        db,
        blob_store,
        worker_id=worker_id,
        limit=limit,
        capture_id=capture_id,
        stage=stage,
    )

    if json_mode:
        output_json_payload(result)
    else:
        out_console.print(
            f"[bold green]Processed jobs:[/bold green] completed={result['completed']}, "
            f"failed={result['failed']}, pending={result.get('pending', 0)}, "
            f"lost_lease={result.get('lost_lease', 0)}, remaining_pending={result['remaining_pending']}"
        )


@app.command("retry")
def retry_command(
    failed: bool = typer.Option(False, "--failed", help="Retry all failed processing jobs"),
    json_mode: bool = typer.Option(False, "--json", help="Output machine-readable JSON on stdout"),
) -> None:
    """Retry processing jobs that have failed."""
    if not failed:
        handle_error(
            "Specify --failed to retry failed processing jobs", exit_code=2, as_json=json_mode
        )

    db, _ = get_services()
    with db.transaction() as conn:
        count = retry_failed_jobs(conn)

    if json_mode:
        output_json_payload({"status": "retried", "count": count})
    else:
        out_console.print(f"[bold green]Reset {count} failed jobs to pending.[/bold green]")


@app.command("purge-jobs")
def purge_jobs_command(
    stage: str = typer.Option(
        ..., "--stage", help="Processing stage whose queued jobs should be removed"
    ),
    status: str = typer.Option("pending", "--status", help="Job status to remove"),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Report what would be deleted without deleting"
    ),
    yes: bool = typer.Option(False, "--yes", help="Confirm the deletion"),
    json_mode: bool = typer.Option(False, "--json", help="Output machine-readable JSON on stdout"),
) -> None:
    """Delete queued jobs for a stage, leaving other stages untouched."""
    if not dry_run and not yes:
        handle_error(
            "Refusing to delete jobs without --yes (use --dry-run to preview)",
            exit_code=2,
            as_json=json_mode,
        )

    db, _ = get_services()
    if dry_run:
        with db.connection() as conn:
            preview = {
                "stage": stage,
                "status": status,
                "would_delete": count_jobs(conn, stage=stage, status=status),
                "total_jobs_before": count_jobs(conn),
                "dry_run": True,
            }
        if json_mode:
            output_json_payload(preview)
        else:
            out_console.print(
                f"[bold]{preview['would_delete']}[/bold] {status} job(s) in stage "
                f"'{stage}' would be deleted (of {preview['total_jobs_before']} total)."
            )
        return

    with db.transaction() as conn:
        result = purge_jobs(conn, stage=stage, status=status)
        result["dry_run"] = False

    if result["other_stages_deleted"] != 0:
        handle_error(
            f"Aborted: the delete touched {result['other_stages_deleted']} job(s) "
            f"outside stage '{stage}'. Inspect the queue before retrying.",
            exit_code=3,
            as_json=json_mode,
        )
    if json_mode:
        output_json_payload(result)
    else:
        out_console.print(
            f"[bold green]Deleted {result['deleted']} {status} job(s) from '{stage}'.[/bold green] "
            f"({result['before_stage']} -> {result['after_stage']}; "
            f"queue {result['before_total']} -> {result['after_total']})"
        )


@app.command("status")
def status_command(
    json_mode: bool = typer.Option(False, "--json", help="Output machine-readable JSON on stdout"),
) -> None:
    """Show processing queue and workspace status."""
    db, _ = get_services()
    with db.connection() as conn:
        proc_status = get_processing_status(conn)
        cap_count = conn.execute("SELECT COUNT(*) FROM captures WHERE is_deleted = 0;").fetchone()[
            0
        ]
        res_count = conn.execute("SELECT COUNT(*) FROM resources WHERE is_deleted = 0;").fetchone()[
            0
        ]
        find_count = conn.execute("SELECT COUNT(*) FROM findings WHERE is_deleted = 0;").fetchone()[
            0
        ]
        att_count = conn.execute("SELECT COUNT(*) FROM attachments;").fetchone()[0]
        extracted_count = conn.execute(
            """
            SELECT COUNT(DISTINCT rc.resource_id)
            FROM resource_contents rc
            WHERE length(trim(rc.clean_text)) > 0
              AND NOT EXISTS (
                SELECT 1 FROM source_snapshots ss
                WHERE ss.resource_id = rc.resource_id
                  AND COALESCE(json_extract(ss.headers_json, '$."content-type"'), '') != ''
                  AND COALESCE(json_extract(ss.headers_json, '$."content-type"'), '')
                      NOT LIKE 'text/%'
                  AND COALESCE(json_extract(ss.headers_json, '$."content-type"'), '')
                      NOT LIKE 'application/pdf%'
                  AND COALESCE(json_extract(ss.headers_json, '$."content-type"'), '')
                      NOT LIKE 'application/xhtml+xml%'
                  AND COALESCE(json_extract(ss.headers_json, '$."content-type"'), '')
                      NOT LIKE 'application/xml%'
              );
            """
        ).fetchone()[0]
        summary_count = conn.execute(
            "SELECT COUNT(DISTINCT resource_id) FROM resource_contents WHERE length(trim(summary)) > 0;"
        ).fetchone()[0]
        embedded_resource_count = conn.execute(
            "SELECT COUNT(DISTINCT object_id) FROM embeddings WHERE object_type = 'resource';"
        ).fetchone()[0]
        embedded_capture_count = conn.execute(
            "SELECT COUNT(DISTINCT object_id) FROM embeddings WHERE object_type = 'capture';"
        ).fetchone()[0]

    status_data = {
        "processing_jobs": proc_status,
        "counts": {
            "captures": cap_count,
            "resources": res_count,
            "findings": find_count,
            "attachments": att_count,
            "extracted_resources": extracted_count,
            "summarized_resources": summary_count,
            "embedded_resources": embedded_resource_count,
            "embedded_captures": embedded_capture_count,
        },
    }

    if json_mode:
        output_json_payload(status_data)
    else:
        out_console.print("[bold cyan]Edward Status[/bold cyan]")
        out_console.print(f"Total jobs: {proc_status['total']}")
        out_console.print(f"  Pending:   {proc_status['by_status'].get('pending', 0)}")
        out_console.print(f"  Running:   {proc_status['by_status'].get('running', 0)}")
        out_console.print(f"  Completed: {proc_status['by_status'].get('completed', 0)}")
        out_console.print(f"  Failed:    {proc_status['by_status'].get('failed', 0)}")
        if proc_status["by_stage"]:
            out_console.print("By stage:")
            for stg, counts in proc_status["by_stage_status"].items():
                detail = ", ".join(f"{status} {count}" for status, count in counts.items())
                out_console.print(f"  {stg}: {detail}")
        if proc_status["failure_reasons"]:
            out_console.print("Failed jobs by reason:")
            for stg, reasons in proc_status["failure_reasons"].items():
                detail = ", ".join(f"{reason} {count}" for reason, count in reasons.items())
                out_console.print(f"  {stg}: {detail}")
        out_console.print("\nCorpus:")
        out_console.print(f"  Captures:    {cap_count}")
        out_console.print(f"  Resources:   {res_count}")
        out_console.print(f"  Findings:    {find_count}")
        out_console.print(f"  Attachments: {att_count}")
        out_console.print(f"  Extracted resources: {extracted_count}")
        out_console.print(f"  Summarized resources: {summary_count}")
        out_console.print(f"  Embedded resources:  {embedded_resource_count}")
        out_console.print(f"  Embedded captures:   {embedded_capture_count}")


@app.command("import-research")
def import_research_command(
    file: Path | None = typer.Argument(
        None, help="Path to research bundle (JSON) or Markdown report"
    ),
    file_opt: Path | None = typer.Option(
        None, "--file", "-f", help="Path to research bundle or Markdown report"
    ),
    stdin: bool = typer.Option(
        False, "--stdin", help="Read bundle or markdown from standard input"
    ),
    format: str = typer.Option(
        "json", "--format", help="Input format: 'json' (bundle) or 'markdown'"
    ),
    idempotency_key: str | None = typer.Option(
        None, "--idempotency-key", help="Unique idempotency key"
    ),
    title: str | None = typer.Option(None, "--title", help="Optional title for markdown report"),
    collector: str = typer.Option("agent", "--collector", help="Collector or agent identifier"),
    json_mode: bool = typer.Option(False, "--json", help="Output machine-readable JSON on stdout"),
) -> None:
    """Import a structured research bundle or Markdown report."""
    target_path = file or file_opt
    if stdin:
        raw_content = sys.stdin.read()
    elif target_path:
        if not target_path.exists():
            handle_error(f"File not found: {target_path}", exit_code=1, as_json=json_mode)
        raw_content = target_path.read_text(encoding="utf-8")
    else:
        handle_error(
            "Either a file or --stdin must be provided",
            exit_code=2,
            as_json=json_mode,
        )

    if not raw_content.strip():
        handle_error("Empty research payload", exit_code=1, as_json=json_mode)

    fmt = format.lower().strip()
    if target_path and target_path.suffix.lower() in [".md", ".markdown"]:
        fmt = "markdown"

    db, blob_store = get_services()

    try:
        if fmt == "json":
            try:
                bundle_dict = json.loads(raw_content)
            except json.JSONDecodeError as e:
                handle_error(f"Invalid JSON payload: {e}", exit_code=1, as_json=json_mode)

            with db.transaction() as conn:
                res = import_research_bundle(
                    conn,
                    blob_store,
                    bundle_dict,
                    idempotency_key=idempotency_key,
                )
        elif fmt == "markdown":
            with db.transaction() as conn:
                res = ingest_markdown_report(
                    conn,
                    raw_content,
                    title=title,
                    collector=collector,
                    idempotency_key=idempotency_key,
                    blob_store=blob_store,
                )
        else:
            handle_error(
                f"Unknown format '{format}'. Supported formats: 'json', 'markdown'.",
                exit_code=2,
                as_json=json_mode,
            )
    except IdempotencyConflictError as e:
        handle_error(str(e), exit_code=3, as_json=json_mode)
    except Exception as e:
        handle_error(str(e), exit_code=1, as_json=json_mode)

    if json_mode:
        output_json_payload(res)
    else:
        if fmt == "markdown":
            out_console.print(
                f"[bold green]Imported Markdown report successfully![/bold green] (Resource ID: {res['resource_id']})"
            )
        else:
            out_console.print(
                f"[bold green]Imported research bundle successfully![/bold green] (Capture ID: {res['capture_id']})"
            )


@app.command("classify")
def classify_command(
    record_id: str | None = typer.Argument(
        None, help="Record ID (resource or capture) to classify"
    ),
    pending: bool = typer.Option(
        False, "--pending", help="Classify all pending unclassified resources"
    ),
    provider: str | None = typer.Option(
        None, "--provider", "-p", help="Classifier provider (typesafe, openrouter, local, dry-run)"
    ),
    limit: int = typer.Option(10, "--limit", "-l", help="Max pending records to classify"),
    json_mode: bool = typer.Option(False, "--json", help="Output machine-readable JSON on stdout"),
) -> None:
    """Run taxonomic classification on an item or all pending items."""
    if not record_id and not pending:
        handle_error("Specify either a record ID or --pending.", exit_code=2, as_json=json_mode)

    from edward.services.classification import run_classification_pipeline

    db, _ = get_services()
    results = []
    try:
        with db.transaction() as conn:
            if record_id:
                r = conn.execute(
                    "SELECT id FROM resources WHERE id = ? AND is_deleted = 0;", (record_id,)
                ).fetchone()
                target_type = "resource" if r else "capture"
                judgments = run_classification_pipeline(
                    conn, target_type, record_id, provider=provider
                )
                results.append(
                    {"id": record_id, "target_type": target_type, "judgments": judgments}
                )
            elif pending:
                rows = conn.execute(
                    """
                    SELECT id FROM resources
                    WHERE is_deleted = 0 AND id NOT IN (SELECT DISTINCT object_id FROM judgments)
                    LIMIT ?;
                    """,
                    (limit,),
                ).fetchall()
                for row in rows:
                    rid = row["id"]
                    judgments = run_classification_pipeline(
                        conn, "resource", rid, provider=provider
                    )
                    results.append({"id": rid, "target_type": "resource", "judgments": judgments})
    except Exception as e:
        handle_error(str(e), exit_code=1, as_json=json_mode)

    if json_mode:
        output_json_payload({"classified_count": len(results), "results": results})
    else:
        out_console.print(f"[bold green]Successfully classified {len(results)} items.[/bold green]")


judgment_app = typer.Typer(
    name="judgment", help="Manage and inspect taxonomic judgments", no_args_is_help=True
)
app.add_typer(judgment_app, name="judgment")


@judgment_app.command("list")
def judgment_list_command(
    record_id: str = typer.Argument(..., help="Object ID to list judgments for"),
    json_mode: bool = typer.Option(False, "--json", help="Output machine-readable JSON on stdout"),
) -> None:
    """List versioned judgments and probabilities for an object."""
    db, _ = get_services()
    with db.connection() as conn:
        rows = conn.execute(
            """
            SELECT id, object_type, object_id, family, label_or_question_id,
                   primitive, answer_json, probability, confidence, provider,
                   requested_model, resolved_model, status, created_at
            FROM judgments WHERE object_id = ? ORDER BY created_at ASC;
            """,
            (record_id,),
        ).fetchall()

    if not rows:
        handle_error(
            f"No judgments found for object '{record_id}'.", exit_code=1, as_json=json_mode
        )

    judgments = []
    for r in rows:
        item = dict(r)
        if item.get("answer_json"):
            try:
                item["answer"] = json.loads(item["answer_json"])
            except Exception:
                pass
        judgments.append(item)

    if json_mode:
        output_json_payload(judgments)
    else:
        from rich.table import Table

        table = Table(title=f"Judgments for {record_id}")
        table.add_column("Question / Label", style="cyan")
        table.add_column("Family", style="magenta")
        table.add_column("Primitive")
        table.add_column("Probability / Choice", style="green")
        table.add_column("Provider")
        for j in judgments:
            val = str(j.get("probability")) if j.get("probability") is not None else ""
            if j.get("primitive") == "choice" and isinstance(j.get("answer"), dict):
                val = j["answer"].get("choice", val)
            table.add_row(
                j["label_or_question_id"],
                j["family"],
                j["primitive"],
                val,
                j["provider"],
            )
        out_console.print(table)


@app.command("reclassify")
def reclassify_command(
    registry_version: str | None = typer.Option(
        None, "--registry-version", help="Target taxonomy registry version"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Run reclassification in dry-run mode without persisting"
    ),
    provider: str | None = typer.Option(None, "--provider", "-p", help="Classifier provider"),
    limit: int = typer.Option(20, "--limit", "-l", help="Maximum records to reclassify"),
    json_mode: bool = typer.Option(False, "--json", help="Output machine-readable JSON on stdout"),
) -> None:
    """Rerun classification against new taxonomy or threshold versions."""
    eff_provider = "dry-run" if dry_run else (provider or "dry-run")
    db, _ = get_services()
    results = []
    try:
        with db.transaction() as conn:
            rows = conn.execute(
                "SELECT id FROM resources WHERE is_deleted = 0 ORDER BY updated_at DESC LIMIT ?;",
                (limit,),
            ).fetchall()
            for r in rows:
                rid = r["id"]
                if dry_run:
                    from edward.services.classification import (
                        classify_target,
                        load_classification_target,
                    )

                    tgt = load_classification_target(conn, "resource", rid)
                    if tgt:
                        form, res = classify_target(
                            tgt["text"], tgt.get("url"), rid, "resource", provider=eff_provider
                        )
                        results.append(
                            {
                                "id": rid,
                                "form": form,
                                "judgments_count": len(res.judgments) if res else 0,
                            }
                        )
                else:
                    from edward.services.classification import run_classification_pipeline

                    judgments = run_classification_pipeline(
                        conn, "resource", rid, provider=eff_provider
                    )
                    results.append({"id": rid, "judgments_count": len(judgments)})
            if dry_run:
                conn.rollback()
    except Exception as e:
        handle_error(str(e), exit_code=1, as_json=json_mode)

    if json_mode:
        output_json_payload(
            {
                "mode": "dry-run" if dry_run else "executed",
                "reclassified_count": len(results),
                "results": results,
            }
        )
    else:
        out_console.print(
            f"[bold green]Reclassified {len(results)} items (mode: {'dry-run' if dry_run else 'executed'}).[/bold green]"
        )


@app.command("ask")
def ask_command(
    question: str = typer.Argument(..., help="Question to ask research memory"),
    no_model: bool = typer.Option(
        False, "--no-model", help="Bypass generative model synthesis and return lookup"
    ),
    limit: int = typer.Option(50, "--limit", "-l", min=1, help="Maximum evidence items to review"),
    project: str | None = typer.Option(None, "--project", help="Restrict evidence to a project"),
    json_mode: bool = typer.Option(False, "--json", help="Output machine-readable JSON on stdout"),
) -> None:
    """Ask a question and receive a deterministic, lookup, or citation-backed answer."""
    db, _ = get_services()
    try:
        with db.connection() as conn:
            from edward.services.answer import answer_question

            ans = answer_question(
                conn,
                query=question,
                no_model=no_model,
                limit=limit,
                project_id=project,
            )
    except Exception as e:
        handle_error(str(e), exit_code=1, as_json=json_mode)

    if json_mode:
        output_json_payload(ans)
    else:
        tier_labels = {
            1: "Tier 1: Direct Factual",
            2: "Tier 2: Lookup Reference",
            3: "Tier 3: Model Synthesis",
        }
        tier_title = tier_labels.get(ans.get("tier", 2), "Answer")
        out_console.print(f"[bold cyan]{tier_title}[/bold cyan]\n")
        out_console.print(ans.get("answer", ""), markup=False)
        coverage = ans.get("coverage")
        if coverage:
            out_console.print(
                f"\nEvidence reviewed: {coverage['retrieved_items']} items in "
                f"{coverage['reviewed_batches']} batches; {coverage['cited_items']} cited."
            )
        if ans.get("status") == "support_unchecked":
            out_console.print("Claim support has not been independently verified.")
        if ans.get("citations"):
            out_console.print("\nCitations: " + ", ".join(ans["citations"]), markup=False)


@app.command("repair-titles")
def repair_titles_command(
    json_mode: bool = typer.Option(False, "--json", help="Output machine-readable JSON on stdout"),
) -> None:
    """Re-derive resource titles that were set from an author instead of the source."""
    from edward.services.lifecycle import repair_author_derived_titles

    try:
        db, blob_store = get_services()
        result = repair_author_derived_titles(db, blob_store)
    except Exception as e:
        handle_error(str(e), exit_code=1, as_json=json_mode)
        return

    if json_mode:
        output_json_payload(result)
    else:
        out_console.print(
            f"[bold green]Title repair complete![/bold green] "
            f"(repaired: {result['repaired']}, left alone: {result['skipped']})"
        )


@app.command("migrate-media")
def migrate_media_command(
    json_mode: bool = typer.Option(False, "--json", help="Output machine-readable JSON on stdout"),
) -> None:
    """Retire media-asset resources, re-homing their images as post attachments.

    Older ingests treated every image URL in a post as its own source, so the
    archive carried hundreds of unreadable resources. This moves each image
    under the post that carried it and queues OCR on it.
    """
    from edward.services.lifecycle import migrate_media_resources_to_attachments

    try:
        db, blob_store = get_services()
        result = migrate_media_resources_to_attachments(db, blob_store)
    except Exception as e:
        handle_error(str(e), exit_code=1, as_json=json_mode)
        return

    if json_mode:
        output_json_payload(result)
        return

    print(
        f"Media migration complete! (converted: {result['converted']}, "
        f"skipped: {result['skipped']}, failed: {len(result['failed'])})"
    )
    if result["failed"]:
        for item in result["failed"][:10]:
            print(f"  {item['resource_id']}: {item['error']}")


@app.command("backfill-shortlinks")
def backfill_shortlinks_command(
    limit: int | None = typer.Option(
        None, "--limit", help="Process at most this many resources (for a trial run)"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Resolve and extract without writing anything"
    ),
    json_mode: bool = typer.Option(False, "--json", help="Output machine-readable JSON on stdout"),
) -> None:
    """Re-extract resources whose content came from a shortlink's interstitial.

    A link shortened by X (t.co) was previously read from its own snapshot, which
    is the shortener's page shell rather than the destination. Those resources
    hold sign-in prompts and cookie banners instead of the article. This resolves
    each link and replaces the stored text with the real page, leaving the
    resource's identity, annotations, intents, and project membership untouched.

    Use --dry-run first: it shows the character counts without modifying anything.
    """
    from edward.services.lifecycle import backfill_shortlink_extractions

    try:
        db, _blob_store = get_services()
        result = backfill_shortlink_extractions(db, limit=limit, dry_run=dry_run)
    except Exception as e:
        handle_error(str(e), exit_code=1, as_json=json_mode)
        return

    if json_mode:
        output_json_payload(result)
        return

    mode = "DRY RUN — nothing written" if dry_run else "complete"
    print(
        f"Shortlink backfill {mode} (candidates: {result['candidates']}, "
        f"recovered: {result['recovered']}, unresolved: {result['unresolved']}, "
        f"skipped: {result['skipped']}, failed: {len(result['failed'])})"
    )
    for item in result["details"][:10]:
        print(
            f"  {item['old_chars']:>7,} -> {item['new_chars']:>7,} chars  "
            f"{(item['title'] or '')[:52]}"
        )
    if len(result["details"]) > 10:
        print(f"  ... and {len(result['details']) - 10} more")
    for item in result["failed"][:10]:
        print(f"  FAILED {item['resource_id']}: {item['error']}")


@app.command("repair-chunks")
def repair_chunks_command(
    json_mode: bool = typer.Option(False, "--json", help="Output machine-readable JSON on stdout"),
) -> None:
    """Re-chunk resources whose searchable chunks belong to a replaced extraction.

    When content is stored without also re-chunking it, the previous extraction's
    chunks stay indexed, so search returns superseded text alongside the text that
    replaced it. This re-derives each affected resource's chunks from its current
    content and drops the old ones, along with their FTS rows and embeddings.

    Safe to re-run: the operation replaces a resource's chunks rather than adding
    to them.
    """
    from edward.services.lifecycle import repair_superseded_chunks

    try:
        db, _blob_store = get_services()
        result = repair_superseded_chunks(db)
    except Exception as e:
        handle_error(str(e), exit_code=1, as_json=json_mode)
        return

    if json_mode:
        output_json_payload(result)
        return

    print(
        f"Chunk repair complete! (considered: {result['considered']}, "
        f"repaired: {result['repaired']})"
    )


@app.command("repair-threads")
def repair_threads_command(
    limit: int | None = typer.Option(None, "--limit", help="Maximum X captures to inspect"),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Report thread expansion candidates without updating"
    ),
    force: bool = typer.Option(
        False, "--force", help="Re-check posts that were already unrolled as threads"
    ),
    json_mode: bool = typer.Option(False, "--json", help="Output machine-readable JSON on stdout"),
) -> None:
    """Scan existing X bookmarks and unroll multi-tweet author threads into full articles.

    Uses `bird thread` to discover author self-reply chains, stitches their text,
    updates clean text and search projections, downloads thread images as attachments,
    and queues any linked URLs across the thread.
    """
    from edward.services.source_adapters import repair_x_threads

    def progress(current: int, total: int, message: str) -> None:
        if not json_mode:
            err_console.print(f"[{current}/{total}] {message}")

    try:
        db, blob_store = get_services()
        result = repair_x_threads(
            db,
            blob_store,
            limit=limit,
            dry_run=dry_run,
            force=force,
            on_progress=progress if not json_mode else None,
        )
    except Exception as e:
        handle_error(str(e), exit_code=1, as_json=json_mode)
        return

    if json_mode:
        output_json_payload(result)
        return

    dry_prefix = "[DRY RUN] " if dry_run else ""
    print(
        f"{dry_prefix}Thread repair complete! (checked: {result['captures_checked']}, "
        f"threads found: {result['threads_found']}, expanded: {result['threads_expanded']}, "
        f"media attached: {result['media_attached']}, linked URLs queued: {result['linked_urls_queued']})"
    )
    for detail in result["details"][:10]:
        print(f"  • {detail['tweet_id']} ({detail['thread_length']} tweets): {detail['new_title']}")


@app.command("intents")
def intents_command(
    json_mode: bool = typer.Option(False, "--json", help="Output machine-readable JSON on stdout"),
) -> None:
    """List the intent questions this archive asks.

    Intents answer "why did I keep this?" as an editable question set rather
    than a fixed taxonomy. Edit the registry to make them yours.
    """
    from edward.services.intents import _resolve_registry_path, load_intent_questions

    questions = load_intent_questions(include_inactive=True)
    path = _resolve_registry_path()
    payload = {
        "registry": str(path) if path else None,
        "questions": [
            {
                "id": q.get("id"),
                "label": q.get("label"),
                "active": bool(q.get("active", True)),
                "prompt": q.get("prompt"),
            }
            for q in questions
        ],
    }
    if json_mode:
        output_json_payload(payload)
        return

    if path:
        print(f"Registry: {path}\n")
    if not questions:
        print("No intent questions are defined. Create the registry to add some.")
        return
    for q in payload["questions"]:
        mark = " " if q["active"] else "-"
        print(f" [{mark}] {q['id']}  ->  {q['label']}")
        print(f"       {q['prompt']}")


@app.command("accept-intent")
def accept_intent_command(
    object_type: str = typer.Argument(..., help="Object type: resource or capture"),
    object_id: str = typer.Argument(..., help="Object ID"),
    intent: str = typer.Argument(..., help="Intent label to accept"),
    json_mode: bool = typer.Option(False, "--json", help="Output machine-readable JSON on stdout"),
) -> None:
    """Accept a suggested intent, making it an active human-owned decision."""
    from edward.services.intents import accept_intent

    try:
        db, _ = get_services()
        with db.transaction() as conn:
            accept_intent(conn, object_type=object_type, object_id=object_id, intent=intent)
    except Exception as e:
        handle_error(str(e), exit_code=1, as_json=json_mode)
        return

    if json_mode:
        output_json_payload({"status": "accepted", "intent": intent, "object_id": object_id})
        return
    print(f"Accepted intent '{intent}' for {object_type} {object_id}.")


@app.command("reindex")
def reindex_command(
    embeddings: bool = typer.Option(False, "--embeddings", help="Rebuild vector embeddings"),
    fts: bool = typer.Option(False, "--fts", help="Rebuild FTS5 lexical index"),
    json_mode: bool = typer.Option(False, "--json", help="Output machine-readable JSON on stdout"),
) -> None:
    """Rebuild FTS5 search projection and/or vector embeddings."""
    db, _ = get_services()
    rebuild_all = not embeddings and not fts
    rebuilt_fts = 0
    rebuilt_embeddings = 0

    try:
        with db.transaction() as conn:
            if fts or rebuild_all:
                from edward.services.lifecycle import reindex_object_document

                conn.execute("DELETE FROM search_documents;")
                for row in conn.execute("SELECT id FROM captures WHERE is_deleted = 0;").fetchall():
                    reindex_object_document(conn, "capture", row["id"])
                    rebuilt_fts += 1
                for row in conn.execute(
                    "SELECT id FROM resources WHERE is_deleted = 0;"
                ).fetchall():
                    reindex_object_document(conn, "resource", row["id"])
                    rebuilt_fts += 1
                for row in conn.execute("SELECT id FROM findings WHERE is_deleted = 0;").fetchall():
                    reindex_object_document(conn, "finding", row["id"])
                    rebuilt_fts += 1
                for row in conn.execute(
                    """
                    SELECT rc.id FROM resource_chunks rc
                    JOIN resources r ON r.id = rc.resource_id
                    WHERE r.is_deleted = 0;
                    """
                ).fetchall():
                    reindex_object_document(conn, "chunk", row["id"])
                    rebuilt_fts += 1

            if embeddings or rebuild_all:
                from edward.services.embed import embed_resource

                for row in conn.execute(
                    "SELECT id FROM resources WHERE is_deleted = 0;"
                ).fetchall():
                    embed_resource(conn, row["id"])
                    rebuilt_embeddings += 1
    except Exception as e:
        handle_error(str(e), exit_code=1, as_json=json_mode)

    result = {
        "status": "completed",
        "rebuilt_fts_documents": rebuilt_fts,
        "rebuilt_embedded_resources": rebuilt_embeddings,
    }
    if json_mode:
        output_json_payload(result)
    else:
        out_console.print(
            f"[bold green]Reindex complete![/bold green] (FTS docs: {rebuilt_fts}, Resources embedded: {rebuilt_embeddings})"
        )


@project_app.command("create")
def project_create_command(
    title: str = typer.Option(..., "--title", help="Working project title"),
    brief: str | None = typer.Option(None, "--brief", help="Project brief or central question"),
    slug: str | None = typer.Option(None, "--slug", help="Stable URL-style project name"),
    json_mode: bool = typer.Option(False, "--json", help="Output machine-readable JSON"),
) -> None:
    """Create an active research and writing workspace."""
    try:
        db, _ = get_services()
        with db.transaction() as conn:
            project = create_project(conn, title=title, brief=brief, slug=slug)
        payload = project.model_dump()
    except (typer.Exit, SystemExit):
        raise
    except ProjectConflictError as e:
        handle_error(str(e), exit_code=3, as_json=json_mode)
    except (ProjectError, ValueError) as e:
        handle_error(str(e), exit_code=1, as_json=json_mode)
    except Exception as e:
        handle_error(f"Project create failed: {e}", exit_code=3, as_json=json_mode)
    if json_mode:
        output_json_payload(payload)
    out_console.print(
        f"[bold green]Created project[/bold green] {project.title} ([cyan]{project.id}[/cyan])"
    )


@project_app.command("list")
def project_list_command(
    include_archived: bool = typer.Option(False, "--all", help="Include non-active projects"),
    json_mode: bool = typer.Option(False, "--json", help="Output machine-readable JSON"),
) -> None:
    """List project workspaces."""
    try:
        db, _ = get_services()
        with db.connection() as conn:
            rows = conn.execute(
                """
                SELECT id, title, slug, description AS brief, status, created_at, updated_at
                FROM projects
                WHERE is_deleted = 0 AND (? = 1 OR status = 'active')
                ORDER BY updated_at DESC;
                """,
                (1 if include_archived else 0,),
            ).fetchall()
        payload = {"count": len(rows), "projects": [dict(row) for row in rows]}
    except (typer.Exit, SystemExit):
        raise
    except Exception as e:
        handle_error(f"Project list failed: {e}", exit_code=3, as_json=json_mode)
    if json_mode:
        output_json_payload(payload)
    for project in payload["projects"]:
        out_console.print(f"[cyan]{project['id']}[/cyan]  {project['title']} ({project['status']})")


@project_app.command("add")
def project_add_command(
    project_id: str = typer.Argument(..., help="Project ID"),
    object_id: str = typer.Argument(..., help="Capture, resource, or finding ID"),
    relationship: str = typer.Option("evidence", "--relationship", "-r"),
    status: str = typer.Option("accepted", "--status", help="candidate, accepted, or rejected"),
    relevance_note: str | None = typer.Option(None, "--note", help="Why this object matters"),
    json_mode: bool = typer.Option(False, "--json", help="Output machine-readable JSON"),
) -> None:
    """Add evidence to a project or change its membership decision."""
    try:
        db, _ = get_services()
        with db.transaction() as conn:
            result = add_project_object(
                conn,
                project_id,
                object_id,
                relationship=relationship,
                membership_status=status,
                added_by="human",
                relevance_note=relevance_note,
            )
    except (typer.Exit, SystemExit):
        raise
    except ProjectConflictError as e:
        handle_error(str(e), exit_code=3, as_json=json_mode)
    except (ProjectError, ValueError) as e:
        handle_error(str(e), exit_code=1, as_json=json_mode)
    except Exception as e:
        handle_error(f"Project add failed: {e}", exit_code=3, as_json=json_mode)
    if json_mode:
        output_json_payload(result)
    out_console.print(
        f"[bold green]Project membership updated[/bold green] ({object_id}: {status}, {relationship})"
    )


@project_app.command("note")
def project_note_command(
    project_id: str = typer.Argument(..., help="Project ID"),
    text: str = typer.Option(..., "--text", help="Workspace note text"),
    kind: str = typer.Option("note", "--kind", help="note, question, gap, or counterargument"),
    json_mode: bool = typer.Option(False, "--json", help="Output machine-readable JSON"),
) -> None:
    """Append a human-authored note, question, gap, or counterargument."""
    try:
        db, _ = get_services()
        with db.transaction() as conn:
            result = add_project_note(conn, project_id, text, kind=kind)
    except (typer.Exit, SystemExit):
        raise
    except ProjectConflictError as e:
        handle_error(str(e), exit_code=3, as_json=json_mode)
    except (ProjectError, ValueError) as e:
        handle_error(str(e), exit_code=1, as_json=json_mode)
    except Exception as e:
        handle_error(f"Project note failed: {e}", exit_code=3, as_json=json_mode)
    if json_mode:
        output_json_payload(result)
    out_console.print(f"[bold green]Added project {kind}[/bold green] ({result['id']})")


@project_app.command("context")
def project_context_command(
    project_id: str = typer.Argument(..., help="Project ID"),
    refresh_candidates: bool = typer.Option(
        False, "--refresh-candidates", help="Retrieve and persist new candidate evidence"
    ),
    limit: int = typer.Option(10, "--limit", min=1, max=100),
    include_rejected: bool = typer.Option(False, "--include-rejected"),
    json_mode: bool = typer.Option(False, "--json", help="Output machine-readable JSON"),
) -> None:
    """Show the brief, evidence, gaps, counterarguments, notes, and latest outline."""
    try:
        db, _ = get_services()
        if refresh_candidates:
            with db.transaction() as conn:
                suggest_project_evidence(conn, project_id, limit=limit, persist=True)
        with db.connection() as conn:
            result = get_project_context(conn, project_id, include_rejected=include_rejected)
    except (typer.Exit, SystemExit):
        raise
    except ProjectConflictError as e:
        handle_error(str(e), exit_code=3, as_json=json_mode)
    except (ProjectError, ValueError) as e:
        handle_error(str(e), exit_code=1, as_json=json_mode)
    except Exception as e:
        handle_error(f"Project context failed: {e}", exit_code=3, as_json=json_mode)
    if json_mode:
        output_json_payload(result)
    project = result["project"]
    out_console.print(f"[bold cyan]{project['title']}[/bold cyan] ({project_id})")
    out_console.print(project.get("brief") or "(No brief)")
    out_console.print(
        f"Accepted evidence: {len(result['accepted_evidence'])}; candidates: {len(result['candidate_evidence'])}"
    )
    for gap in result["gaps_and_counterarguments"]:
        out_console.print(f"- [{gap['kind']}] {gap['description']}")

    outline = result.get("latest_outline")
    if outline:
        out_console.print(
            f"\n[bold]Latest outline[/bold] version {outline['version']} ({outline['status']})"
        )
        if outline.get("premise"):
            out_console.print(outline["premise"])
        for section in outline.get("sections", []):
            out_console.print(f"{section['order']}. {section['heading']}")
            if section.get("claim"):
                out_console.print(f"     claim: {section['claim']}")
            for need in section.get("unresolved_research_needs") or []:
                out_console.print(f"     [yellow]needs:[/yellow] {need}")
            for link in section.get("evidence") or []:
                note = link.get("relevance_note")
                suffix = f" — {note}" if note else ""
                out_console.print(
                    f"     evidence ({link['relationship']}): {link['object_id']}{suffix}"
                )


@project_app.command("outline")
def project_outline_command(
    project_id: str = typer.Argument(..., help="Project ID"),
    propose: bool = typer.Option(False, "--propose", help="Create a new outline proposal"),
    show: bool = typer.Option(False, "--show", help="Show an outline"),
    revise: bool = typer.Option(False, "--revise", help="Create a revised outline version"),
    accept: bool = typer.Option(False, "--accept", help="Accept an outline version"),
    reject: bool = typer.Option(False, "--reject", help="Reject an outline version"),
    instructions: str | None = typer.Option(None, "--instructions", help="Revision instructions"),
    input_file: Path | None = typer.Option(
        None, "--input", help="Structured outline JSON supplied by a calling agent"
    ),
    version: int | None = typer.Option(None, "--version", min=1),
    no_model: bool = typer.Option(False, "--no-model", help="Use deterministic offline outlining"),
    json_mode: bool = typer.Option(False, "--json", help="Output machine-readable JSON"),
) -> None:
    """Propose, inspect, revise, accept, or reject a versioned evidence-linked outline."""
    operations = [propose, show, revise, accept, reject]
    if sum(bool(op) for op in operations) != 1:
        handle_error(
            "Choose exactly one of --propose, --show, --revise, --accept, or --reject",
            exit_code=2,
            as_json=json_mode,
        )
    if revise and not instructions and input_file is None:
        handle_error(
            "--revise requires --instructions or --input",
            exit_code=2,
            as_json=json_mode,
        )
    try:
        db, _ = get_services()
        if show:
            with db.connection() as conn:
                result = get_outline(conn, project_id, version=version)
        elif accept or reject:
            with db.transaction() as conn:
                result = set_outline_status(
                    conn, project_id, "accepted" if accept else "rejected", version=version
                )
        else:
            # Read context and perform any model inference before opening a write transaction.
            with db.connection() as conn:
                context = get_project_context(conn, project_id)
            parent = context.get("latest_outline") if revise else None
            if revise and parent is None:
                raise ProjectError("Cannot revise a project that has no outline")
            if input_file is not None:
                try:
                    proposal_data = json.loads(input_file.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as e:
                    raise ValueError(f"Could not read outline input: {e}") from e
                author_type = "calling-agent"
                author_id = "external"
            else:
                from edward.services.llm import get_answer_client

                client = None if no_model else get_answer_client()
                proposal_data = generate_outline_proposal(
                    context, client, revision_instructions=instructions
                )
                author_type = "system" if client is None else "model"
                author_id = None if client is None else client.model
            with db.transaction() as conn:
                result = save_outline(
                    conn,
                    project_id,
                    proposal_data,
                    author_type=author_type,
                    author_id=author_id,
                    parent_outline_id=parent["id"] if parent else None,
                    revision_instructions=instructions,
                )
    except (typer.Exit, SystemExit):
        raise
    except PrivacyTransmissionError as e:
        handle_error(str(e), exit_code=1, as_json=json_mode)
    except ProjectConflictError as e:
        handle_error(str(e), exit_code=3, as_json=json_mode)
    except (ProjectError, ValueError) as e:
        handle_error(str(e), exit_code=1, as_json=json_mode)
    except Exception as e:
        handle_error(f"Outline operation failed: {e}", exit_code=3, as_json=json_mode)

    if json_mode:
        output_json_payload(result)
    out_console.print(
        f"[bold green]Outline version {result['version']}[/bold green] ({result['status']})"
    )
    for section in result["sections"]:
        out_console.print(f"{section['order']}. {section['heading']}")


@project_app.command("delete")
def project_delete_command(
    project_id: str = typer.Argument(..., help="Project ID to delete"),
    confirm: bool = typer.Option(
        False, "--confirm", help="Explicit confirmation required to delete"
    ),
    json_mode: bool = typer.Option(False, "--json", help="Output machine-readable JSON"),
) -> None:
    """Soft-delete a project workspace."""
    if not confirm:
        handle_error(
            "Project deletion requires explicit --confirm flag",
            exit_code=2,
            as_json=json_mode,
        )
    try:
        db, _ = get_services()
        with db.transaction() as conn:
            result = delete_project(conn, project_id)
    except (typer.Exit, SystemExit):
        raise
    except ProjectError as e:
        handle_error(str(e), exit_code=1, as_json=json_mode)
    except Exception as e:
        handle_error(f"Project delete failed: {e}", exit_code=3, as_json=json_mode)

    if json_mode:
        output_json_payload(result)
    out_console.print(
        f"[bold red]Deleted project[/bold red] {result['title']} ([cyan]{result['id']}[/cyan])"
    )


@app.command("mcp")
def mcp_command(
    transport: str = typer.Option(
        "stdio", "--transport", "-t", help="MCP transport protocol (stdio, sse, streamable-http)"
    ),
) -> None:
    """Run the native Edward Model Context Protocol (MCP) server for AI agents."""
    from edward.mcp_server import create_mcp_server

    if transport not in ("stdio", "sse", "streamable-http"):
        handle_error(
            f"Unsupported transport '{transport}'. Must be stdio, sse, or streamable-http.",
            exit_code=2,
        )

    server = create_mcp_server()
    server.run(transport=transport)  # type: ignore[arg-type]


def main() -> int:
    """Entry point for CLI runner."""
    try:
        app()
        return 0
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else 0


if __name__ == "__main__":
    sys.exit(main())
