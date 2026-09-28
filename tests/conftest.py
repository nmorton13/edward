"""Shared pytest fixtures for Edward test suite."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from edward.blobs import BlobStore
from edward.db import Database


@pytest.fixture
def temp_dir(tmp_path: Path) -> Path:
    """Fixture providing a clean temporary data directory."""
    d = tmp_path / "edward_data"
    d.mkdir(parents=True, exist_ok=True)
    (d / "blobs").mkdir(parents=True, exist_ok=True)
    (d / "diagnostics").mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture(autouse=True)
def deterministic_embeddings_for_tests(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the suite offline unless a test explicitly exercises FastEmbed."""
    monkeypatch.setenv("EDWARD_EMBEDDING_MODEL", "deterministic-v1")


@pytest.fixture(autouse=True)
def isolate_data_dir_for_tests(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure tests never touch the user's real ~/.edward directory."""
    d = tmp_path / "edward_test_env"
    d.mkdir(parents=True, exist_ok=True)
    (d / "blobs").mkdir(parents=True, exist_ok=True)
    (d / "diagnostics").mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("EDWARD_DATA_DIR", str(d))
    monkeypatch.delenv("EDWARD_DB_PATH", raising=False)
    monkeypatch.setenv("EDWARD_SUMMARIZER_MODE", "disabled")
    monkeypatch.setenv("EDWARD_ANSWERER_MODE", "disabled")


@pytest.fixture
def test_db(temp_dir: Path) -> Database:
    """Fixture providing a Database instance with migrations applied."""
    db_file = temp_dir / "test_edward.sqlite3"
    db = Database(db_file)
    db.run_migrations()
    return db


@pytest.fixture
def test_blob_store(temp_dir: Path) -> BlobStore:
    """Fixture providing an initialized BlobStore."""
    return BlobStore(temp_dir / "blobs")


@pytest.fixture
def cli_runner() -> CliRunner:
    """Fixture providing a Typer CLI runner."""
    return CliRunner()
