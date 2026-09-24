"""Tests for protected model diagnostics logging and privacy safeguards."""

from pathlib import Path

import pytest

from edward.blobs import BlobStore
from edward.db import Database
from edward.services.diagnostics import (
    list_model_diagnostics,
    prune_model_diagnostics,
    record_model_diagnostic,
    run_doctor,
)


def test_record_model_diagnostic_public_web(tmp_path: Path):
    """Diagnostics are recorded for public_web content with sanitized errors."""
    raw_output = "Invalid JSON output from model: {unquoted_key: 123"
    home_dir = str(Path.home())
    dirty_err = f"Failed to parse at {home_dir}/work/edward with Bearer sk-12345"

    file_path = record_model_diagnostic(
        provider="test-llm",
        raw_output=raw_output,
        error_message=dirty_err,
        data_class="public_web",
        diagnostics_dir=tmp_path,
    )
    assert file_path is not None
    assert file_path.exists()

    content = file_path.read_text(encoding="utf-8")
    assert "Invalid JSON output" in content
    # Ensure home dir and bearer secrets were sanitized
    assert home_dir not in content
    assert "~" in content
    assert "sk-12345" not in content


def test_record_model_diagnostic_private_content_disabled_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Private data classes do NOT record diagnostics by default."""
    monkeypatch.delenv("EDWARD_RECORD_PRIVATE_DIAGNOSTICS", raising=False)

    p1 = record_model_diagnostic(
        provider="test-llm",
        raw_output="Private email content",
        error_message="Schema error",
        data_class="gmail",
        diagnostics_dir=tmp_path,
    )
    assert p1 is None

    p2 = record_model_diagnostic(
        provider="test-llm",
        raw_output="Private note content",
        error_message="Schema error",
        data_class="personal_notes",
        diagnostics_dir=tmp_path,
    )
    assert p2 is None

    # Enable explicitly
    monkeypatch.setenv("EDWARD_RECORD_PRIVATE_DIAGNOSTICS", "1")
    p3 = record_model_diagnostic(
        provider="test-llm",
        raw_output="Private note content",
        error_message="Schema error",
        data_class="personal_notes",
        diagnostics_dir=tmp_path,
    )
    assert p3 is not None
    assert p3.exists()


def test_list_and_prune_model_diagnostics(tmp_path: Path):
    """Pruning diagnostics retains the most recent N files."""
    for i in range(10):
        record_model_diagnostic(
            provider="test-llm",
            raw_output=f"Output {i}",
            error_message=f"Error {i}",
            data_class="public_web",
            diagnostics_dir=tmp_path,
        )

    all_files = list_model_diagnostics(tmp_path)
    assert len(all_files) == 10

    # Prune to keep 4
    pruned_count = prune_model_diagnostics(max_count=4, diagnostics_dir=tmp_path)
    assert pruned_count == 6
    remaining = list_model_diagnostics(tmp_path)
    assert len(remaining) == 4


def test_list_model_diagnostics_nonexistent_dir(tmp_path: Path):
    """list_model_diagnostics returns empty list when directory does not exist."""
    assert list_model_diagnostics(tmp_path / "nonexistent_dir") == []


def test_prune_model_diagnostics_under_max(tmp_path: Path):
    """Pruning does nothing when file count is under max_count."""
    record_model_diagnostic(
        provider="test-llm",
        raw_output="test output",
        error_message="test error",
        data_class="public_web",
        diagnostics_dir=tmp_path,
    )
    pruned = prune_model_diagnostics(max_count=50, diagnostics_dir=tmp_path)
    assert pruned == 0
    assert len(list_model_diagnostics(tmp_path)) == 1


def test_record_model_diagnostic_documents_class_denied_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Documents data class does not record diagnostics by default."""
    monkeypatch.delenv("EDWARD_RECORD_PRIVATE_DIAGNOSTICS", raising=False)
    p = record_model_diagnostic(
        provider="test-llm",
        raw_output="Private document content",
        error_message="Parse error",
        data_class="documents",
        diagnostics_dir=tmp_path,
    )
    assert p is None


def test_doctor_reports_missing_blobs(tmp_path: Path):
    """run_doctor detects missing blobs referenced in database snapshots."""
    db = Database(tmp_path / "missing_blob.sqlite3")
    db.run_migrations()
    blob_store = BlobStore(tmp_path / "blobs")

    fake_hash = "a" * 64
    with db.transaction() as conn:
        conn.execute(
            "INSERT INTO resources (id, identity_key, title, canonical_url, review_state, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, 'unreviewed', datetime('now'), datetime('now'));",
            ("res_test1", "ik_test1", "Test Resource", "https://example.com/test"),
        )
        conn.execute(
            "INSERT INTO source_snapshots (id, resource_id, content_hash, blob_path, size_bytes, created_at) "
            "VALUES (?, ?, ?, ?, ?, datetime('now'));",
            ("snap_test1", "res_test1", fake_hash, f"{fake_hash[:2]}/{fake_hash}", 100),
        )

    report = run_doctor(db, blob_store)
    assert report["healthy"] is False
    assert report["blob_store"]["missing_count"] == 1
    assert fake_hash in report["blob_store"]["missing_hashes"]


def test_doctor_reports_corrupted_blobs(tmp_path: Path):
    """run_doctor detects corrupted blobs with on-disk hash mismatches."""
    db = Database(tmp_path / "corrupt_blob.sqlite3")
    db.run_migrations()
    blob_store = BlobStore(tmp_path / "blobs")

    content = b"original content for corruption test"
    content_hash, blob_path = blob_store.store_bytes(content)

    with db.transaction() as conn:
        conn.execute(
            "INSERT INTO resources (id, identity_key, title, canonical_url, review_state, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, 'unreviewed', datetime('now'), datetime('now'));",
            ("res_test2", "ik_test2", "Corrupt Resource", "https://example.com/corrupt"),
        )
        conn.execute(
            "INSERT INTO source_snapshots (id, resource_id, content_hash, blob_path, size_bytes, created_at) "
            "VALUES (?, ?, ?, ?, ?, datetime('now'));",
            (
                "snap_test2",
                "res_test2",
                content_hash,
                f"{content_hash[:2]}/{content_hash}",
                len(content),
            ),
        )

    # Corrupt on disk
    blob_path.write_bytes(b"tampered content that will not match hash")

    report = run_doctor(db, blob_store)
    assert report["healthy"] is False
    assert report["blob_store"]["corrupted_count"] == 1
    assert content_hash in report["blob_store"]["corrupted_hashes"]


def test_doctor_reports_unreferenced_blobs(tmp_path: Path):
    """run_doctor reports unreferenced blobs without marking system unhealthy."""
    db = Database(tmp_path / "unref_blob.sqlite3")
    db.run_migrations()
    blob_store = BlobStore(tmp_path / "blobs")

    blob_store.store_bytes(b"unreferenced blob data")

    report = run_doctor(db, blob_store)
    assert report["healthy"] is True
    assert report["blob_store"]["unreferenced_count"] == 1
    assert report["blob_store"]["disk_count"] == 1
    assert report["blob_store"]["referenced_count"] == 0
