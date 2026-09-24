"""Tests for run_doctor vector extension diagnostic accuracy."""

from pathlib import Path

from edward.blobs import BlobStore
from edward.db import Database
from edward.services.diagnostics import run_doctor


def test_doctor_vector_extension_reports_python_fallback(tmp_path: Path):
    """When sqlite-vec is unavailable, mode must report 'python-fallback', not 'FTS5 only'."""
    db_file = tmp_path / "diag_vec.sqlite3"
    db = Database(db_file)
    db.run_migrations()
    blob_store = BlobStore(tmp_path / "blobs")

    report = run_doctor(db, blob_store)
    vec = report["vector_extension"]
    embedding = report["embedding_model"]

    assert embedding == {
        "name": "deterministic-v1",
        "backend": "deterministic-fallback",
        "semantic": False,
    }

    # Regardless of whether sqlite-vec is installed in the test environment,
    # the mode value must be one of these two canonical strings.
    assert vec["mode"] in ("sqlite-vec", "python-fallback")
    assert "detail" in vec

    if db.has_vec:
        assert vec["mode"] == "sqlite-vec"
        assert vec["available"] is True
        assert "native acceleration" in vec["detail"]
    else:
        assert vec["mode"] == "python-fallback"
        assert vec["available"] is False
        assert "Pure-Python cosine similarity" in vec["detail"]
        # The old inaccurate string must not appear
        assert "FTS5 only" not in vec["mode"]
        assert "vector-free" not in vec["mode"]


def test_doctor_vector_extension_with_vec_forced_off(tmp_path: Path, monkeypatch):
    """Simulate sqlite-vec being unavailable by forcing has_vec = False."""
    db_file = tmp_path / "diag_forced.sqlite3"
    db = Database(db_file)
    db.run_migrations()

    # Force vector extension off
    db.has_vec = False
    blob_store = BlobStore(tmp_path / "blobs")

    report = run_doctor(db, blob_store)
    vec = report["vector_extension"]

    assert vec["available"] is False
    assert vec["mode"] == "python-fallback"
    assert "Pure-Python cosine similarity" in vec["detail"]


def test_doctor_vector_extension_with_vec_forced_on(tmp_path: Path, monkeypatch):
    """Simulate sqlite-vec being available by forcing has_vec = True."""
    db_file = tmp_path / "diag_forced_on.sqlite3"
    db = Database(db_file)
    db.run_migrations()

    # Prevent connect() from resetting has_vec via _try_load_vec
    monkeypatch.setattr(db, "_try_load_vec", lambda conn: None)
    db.has_vec = True
    blob_store = BlobStore(tmp_path / "blobs")

    report = run_doctor(db, blob_store)
    vec = report["vector_extension"]

    assert vec["available"] is True
    assert vec["mode"] == "sqlite-vec"
    assert "native acceleration" in vec["detail"]


def test_doctor_identifies_builtin_semantic_embedding_model(tmp_path: Path, monkeypatch):
    db = Database(tmp_path / "diag_model.sqlite3")
    db.run_migrations()
    monkeypatch.delenv("EDWARD_EMBEDDING_MODEL", raising=False)

    report = run_doctor(db, BlobStore(tmp_path / "blobs"))

    assert report["embedding_model"] == {
        "name": "BAAI/bge-small-en-v1.5",
        "backend": "fastembed-local",
        "semantic": True,
    }
