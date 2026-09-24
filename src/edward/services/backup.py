"""Holistic backup service covering SQLite (with WAL) and content-addressed blobs."""

import datetime
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from edward.blobs import BlobStore
from edward.db import Database


class BackupError(Exception):
    """Base error for backup operations."""

    pass


def file_sha256(path: Path) -> str:
    """Calculate the SHA-256 hash of a file on disk."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def create_backup(
    db: Database,
    blob_store: BlobStore,
    dest_path: Path | None = None,
) -> dict[str, Any]:
    """Create a holistic backup archive containing SQLite snapshot and referenced blobs."""
    now = datetime.datetime.now(datetime.UTC)
    ts_str = now.strftime("%Y%m%d_%H%M%S")

    if dest_path is None:
        dest_path = db.db_path.parent / "backups" / f"backup_{ts_str}"
    dest_path = Path(dest_path).resolve()
    dest_path.mkdir(parents=True, exist_ok=True)

    # 1. Online SQLite Backup
    backup_db_path = dest_path / "edward.sqlite3"
    db.backup_database(backup_db_path)
    db_hash = file_sha256(backup_db_path)
    db_size = backup_db_path.stat().st_size

    # 2. Collect referenced blob hashes from SQLite
    referenced_hashes: set[str] = set()
    with db.connection() as conn:
        for query in [
            "SELECT DISTINCT content_hash FROM source_snapshots WHERE content_hash IS NOT NULL;",
            "SELECT DISTINCT content_hash FROM attachments WHERE content_hash IS NOT NULL;",
        ]:
            try:
                for row in conn.execute(query).fetchall():
                    if row[0]:
                        referenced_hashes.add(row[0])
            except Exception:
                pass

    # 3. Copy referenced blobs
    manifest_files: list[dict[str, Any]] = [
        {
            "path": "edward.sqlite3",
            "sha256": db_hash,
            "size_bytes": db_size,
        }
    ]

    missing_blobs: list[str] = []
    blobs_dest_dir = dest_path / "blobs"
    blobs_dest_dir.mkdir(parents=True, exist_ok=True)

    for b_hash in referenced_hashes:
        source_blob_path = blob_store.get_path(b_hash)
        if not source_blob_path or not source_blob_path.exists():
            missing_blobs.append(b_hash)
            continue

        prefix = b_hash[:2]
        dest_blob_subdir = blobs_dest_dir / prefix
        dest_blob_subdir.mkdir(parents=True, exist_ok=True)
        dest_blob_file = dest_blob_subdir / b_hash

        shutil.copy2(source_blob_path, dest_blob_file)
        b_hash_check = file_sha256(dest_blob_file)
        b_size = dest_blob_file.stat().st_size

        manifest_files.append(
            {
                "path": f"blobs/{prefix}/{b_hash}",
                "sha256": b_hash_check,
                "size_bytes": b_size,
            }
        )

    # 4. Write integrity manifest
    manifest_data = {
        "manifest_version": "1.0",
        "created_at": now.isoformat(),
        "database": {
            "filename": "edward.sqlite3",
            "sha256": db_hash,
            "size_bytes": db_size,
        },
        "blobs_total_referenced": len(referenced_hashes),
        "blobs_backed_up": len(referenced_hashes) - len(missing_blobs),
        "missing_blobs": missing_blobs,
        "files": manifest_files,
    }

    manifest_path = dest_path / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    return {
        "backup_path": str(dest_path),
        "database_sha256": db_hash,
        "blobs_backed_up": len(referenced_hashes) - len(missing_blobs),
        "missing_blobs": missing_blobs,
        "total_files": len(manifest_files),
    }


def verify_backup_integrity(backup_path: Path) -> dict[str, Any]:
    """Verify backup files against integrity manifest.json."""
    backup_path = Path(backup_path).resolve()
    manifest_file = backup_path / "manifest.json"
    if not manifest_file.exists():
        return {"valid": False, "errors": ["manifest.json not found in backup directory"]}

    try:
        with open(manifest_file, encoding="utf-8") as f:
            manifest = json.load(f)
    except Exception as e:
        return {"valid": False, "errors": [f"Malformed manifest.json: {e}"]}

    errors: list[str] = []
    verified_files = 0

    for file_info in manifest.get("files", []):
        rel_path = file_info["path"]
        expected_hash = file_info["sha256"]
        target = backup_path / rel_path

        if not target.exists():
            errors.append(f"Missing file: {rel_path}")
            continue

        actual_hash = file_sha256(target)
        if actual_hash != expected_hash:
            errors.append(f"Corrupted file {rel_path}: expected {expected_hash}, got {actual_hash}")
        else:
            verified_files += 1

    return {
        "valid": len(errors) == 0,
        "verified_files": verified_files,
        "errors": errors,
        "missing_blobs_at_backup": manifest.get("missing_blobs", []),
    }
