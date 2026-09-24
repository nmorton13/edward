"""Tests for holistic backup service and integrity verification."""

import json
from pathlib import Path

from edward.blobs import BlobStore
from edward.db import Database
from edward.models import CaptureInput
from edward.services.backup import create_backup, verify_backup_integrity
from edward.services.capture import capture_item


def test_holistic_backup_and_manifest(
    test_db: Database, test_blob_store: BlobStore, tmp_path: Path
):
    # 1. Insert database records
    with test_db.transaction() as conn:
        res = capture_item(conn, CaptureInput(url="https://example.com", note="Test note"))
        res_id = res["resource_id"]

        # Store a blob and attach to resource
        blob_bytes = b"Attachment or snapshot blob content"
        b_hash, b_path = test_blob_store.store_bytes(blob_bytes)

        conn.execute(
            """
            INSERT INTO attachments (id, object_type, object_id, file_name, mime_type, content_hash, size_bytes, blob_path, created_at)
            VALUES ('att_1', 'resource', ?, 'file.txt', 'text/plain', ?, ?, ?, datetime('now'));
            """,
            (res_id, b_hash, len(blob_bytes), str(b_path)),
        )

    # 2. Perform backup
    backup_dest = tmp_path / "backup_archive"
    summary = create_backup(test_db, test_blob_store, dest_path=backup_dest)

    assert Path(summary["backup_path"]).exists()
    assert summary["blobs_backed_up"] == 1
    assert len(summary["missing_blobs"]) == 0

    # 3. Verify backup against manifest
    verification = verify_backup_integrity(backup_dest)
    assert verification["valid"] is True
    assert verification["verified_files"] == 2  # sqlite db + 1 blob


def test_tampered_backup_fails_verification(
    test_db: Database, test_blob_store: BlobStore, tmp_path: Path
):
    backup_dest = tmp_path / "tampered_archive"
    create_backup(test_db, test_blob_store, dest_path=backup_dest)

    # Tamper with sqlite file
    db_file = backup_dest / "edward.sqlite3"
    db_file.write_bytes(db_file.read_bytes() + b"\xff\xff")

    verification = verify_backup_integrity(backup_dest)
    assert verification["valid"] is False
    assert len(verification["errors"]) > 0


def test_resource_contents_without_blob_does_not_fail_backup(
    test_db: Database, test_blob_store: BlobStore, tmp_path: Path
):
    from edward.services.diagnostics import run_doctor

    with test_db.transaction() as conn:
        res = capture_item(conn, CaptureInput(url="https://example.com/clean-article"))
        res_id = res["resource_id"]

        # Insert clean text directly into SQLite (resource_contents has a content_hash, but NO blob file)
        conn.execute(
            """
            INSERT INTO resource_contents (
                id, resource_id, content_hash, clean_text, char_count, extractor, extractor_version, created_at
            ) VALUES ('rc_1', ?, 'nonexistent_blob_hash_in_disk', 'Extracted clean text', 20, 'test', '1.0', datetime('now'));
            """,
            (res_id,),
        )

    # Doctor and backup should NOT report nonexistent_blob_hash_in_disk as a missing blob
    report = run_doctor(test_db, test_blob_store)
    assert report["healthy"] is True
    assert "nonexistent_blob_hash_in_disk" not in report.get("diagnostics", {}).get(
        "missing_blobs", []
    )

    backup_dest = tmp_path / "clean_backup"
    summary = create_backup(test_db, test_blob_store, dest_path=backup_dest)
    assert "nonexistent_blob_hash_in_disk" not in summary["missing_blobs"]


def test_verify_backup_missing_manifest(tmp_path: Path):
    """verify_backup_integrity returns invalid when manifest.json is missing."""
    backup_dir = tmp_path / "empty_backup"
    backup_dir.mkdir()
    result = verify_backup_integrity(backup_dir)
    assert result["valid"] is False
    assert any("manifest.json not found" in e for e in result["errors"])


def test_verify_backup_malformed_manifest(tmp_path: Path):
    """verify_backup_integrity returns invalid when manifest.json is corrupt."""
    backup_dir = tmp_path / "bad_manifest"
    backup_dir.mkdir()
    (backup_dir / "manifest.json").write_text("not valid json{{{", encoding="utf-8")
    result = verify_backup_integrity(backup_dir)
    assert result["valid"] is False
    assert any("Malformed manifest.json" in e for e in result["errors"])


def test_verify_backup_missing_file_in_manifest(tmp_path: Path):
    """verify_backup_integrity detects files listed in manifest but missing on disk."""
    backup_dir = tmp_path / "incomplete_backup"
    backup_dir.mkdir()
    manifest = {"files": [{"path": "missing_file.sqlite3", "sha256": "a" * 64, "size_bytes": 100}]}
    (backup_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    result = verify_backup_integrity(backup_dir)
    assert result["valid"] is False
    assert any("Missing file" in e for e in result["errors"])
