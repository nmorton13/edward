"""Diagnostics and doctor service for Edward."""

import sqlite3
from pathlib import Path
from typing import Any

from edward.blobs import BlobStore
from edward.db import Database
from edward.services.backup import verify_backup_integrity
from edward.services.embed import get_configured_embedding_model


def run_doctor(
    db: Database,
    blob_store: BlobStore,
    backup_to_verify: Path | None = None,
) -> dict[str, Any]:
    """Run comprehensive system health checks or verify a specific backup."""
    if backup_to_verify:
        return {
            "mode": "backup_verification",
            "result": verify_backup_integrity(backup_to_verify),
        }

    report: dict[str, Any] = {
        "mode": "system_diagnostics",
        "healthy": True,
        "database": {},
        "migrations": {},
        "search_index": {},
        "blob_store": {},
        "vector_extension": {},
    }

    # 1. SQLite PRAGMA integrity_check
    with db.connection() as conn:
        cursor = conn.execute("PRAGMA integrity_check;")
        rows = [r[0] for r in cursor.fetchall()]
        db_ok = len(rows) == 1 and rows[0] == "ok"
        report["database"] = {
            "integrity_ok": db_ok,
            "details": rows,
        }
        if not db_ok:
            report["healthy"] = False

        # 2. Migrations Check
        applied = db.get_applied_migrations(conn)
        migrations_dir = Path(__file__).parent.parent / "migrations"
        total_migrations = len(list(migrations_dir.glob("*.sql"))) if migrations_dir.exists() else 0
        report["migrations"] = {
            "applied_count": len(applied),
            "expected_count": total_migrations,
            "up_to_date": len(applied) >= total_migrations,
        }
        if len(applied) < total_migrations:
            report["healthy"] = False

        # 3. FTS5 Index Check
        try:
            fts_count = conn.execute("SELECT COUNT(*) FROM search_documents;").fetchone()[0]
            report["search_index"] = {
                "accessible": True,
                "document_count": fts_count,
            }
        except sqlite3.OperationalError as e:
            report["search_index"] = {
                "accessible": False,
                "error": str(e),
            }
            report["healthy"] = False

        # 4. Blob Store Check
        referenced_hashes = set()
        for q in [
            "SELECT DISTINCT content_hash FROM source_snapshots WHERE content_hash IS NOT NULL;",
            "SELECT DISTINCT content_hash FROM attachments WHERE content_hash IS NOT NULL;",
        ]:
            try:
                for r in conn.execute(q).fetchall():
                    if r[0]:
                        referenced_hashes.add(r[0])
            except Exception:
                pass

    missing_blobs = []
    corrupted_blobs = []
    for b_hash in referenced_hashes:
        if not blob_store.exists(b_hash):
            missing_blobs.append(b_hash)
        elif not blob_store.verify_blob(b_hash):
            corrupted_blobs.append(b_hash)

    disk_hashes = blob_store.list_all_hashes()
    unreferenced_blobs = list(disk_hashes - referenced_hashes)

    report["blob_store"] = {
        "referenced_count": len(referenced_hashes),
        "disk_count": len(disk_hashes),
        "missing_count": len(missing_blobs),
        "missing_hashes": missing_blobs,
        "corrupted_count": len(corrupted_blobs),
        "corrupted_hashes": corrupted_blobs,
        "unreferenced_count": len(unreferenced_blobs),
    }

    if missing_blobs or corrupted_blobs:
        report["healthy"] = False

    # 5. Vector Extension Status
    report["vector_extension"] = {
        "available": db.has_vec,
        "mode": "sqlite-vec" if db.has_vec else "python-fallback",
        "detail": (
            "sqlite-vec native acceleration active"
            if db.has_vec
            else "Pure-Python cosine similarity fallback active (sqlite-vec not installed)"
        ),
    }
    embedding_model = get_configured_embedding_model()
    report["embedding_model"] = {
        "name": embedding_model,
        "backend": (
            "deterministic-fallback" if embedding_model == "deterministic-v1" else "fastembed-local"
        ),
        "semantic": embedding_model != "deterministic-v1",
    }

    return report


def record_model_diagnostic(
    provider: str,
    raw_output: str,
    error_message: str,
    data_class: str = "public_web",
    context: dict[str, Any] | None = None,
    diagnostics_dir: Path | None = None,
) -> Path | None:
    """Record invalid raw model output to <data-dir>/diagnostics/ with privacy safeguards.

    Disabled by default for private content classes ('gmail', 'personal_notes', 'documents')
    unless EDWARD_RECORD_PRIVATE_DIAGNOSTICS=1.
    """
    import datetime
    import hashlib
    import json
    import os

    from edward.db import get_default_data_dir
    from edward.services.subprocess_runner import sanitize_error_message

    if data_class in ("gmail", "personal_notes", "documents"):
        allow_private = os.environ.get("EDWARD_RECORD_PRIVATE_DIAGNOSTICS", "0").strip() in (
            "1",
            "true",
            "yes",
        )
        if not allow_private:
            return None

    target_dir = diagnostics_dir or (get_default_data_dir() / "diagnostics")
    target_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%d_%H%M%S")
    output_hash = hashlib.sha256(raw_output.encode("utf-8")).hexdigest()[:12]
    filename = f"model_err_{timestamp}_{output_hash}.json"
    file_path = target_dir / filename

    sanitized_error = sanitize_error_message(error_message)

    payload = {
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
        "provider": provider,
        "data_class": data_class,
        "error": sanitized_error,
        "raw_output": raw_output,
        "context": context or {},
    }

    file_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return file_path


def list_model_diagnostics(diagnostics_dir: Path | None = None) -> list[Path]:
    """List diagnostic files sorted newest first."""
    from edward.db import get_default_data_dir

    target_dir = diagnostics_dir or (get_default_data_dir() / "diagnostics")
    if not target_dir.exists():
        return []
    return sorted(target_dir.glob("model_err_*.json"), reverse=True)


def prune_model_diagnostics(
    max_count: int = 50,
    diagnostics_dir: Path | None = None,
) -> int:
    """Prune oldest model diagnostic logs keeping at most max_count files."""
    files = list_model_diagnostics(diagnostics_dir)
    pruned = 0
    if len(files) > max_count:
        for f in files[max_count:]:
            try:
                f.unlink(missing_ok=True)
                pruned += 1
            except Exception:
                pass
    return pruned
