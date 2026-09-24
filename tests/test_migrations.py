"""Tests for database migrations and schema setup."""

from pathlib import Path

from edward.db import Database


def test_migrations_applied_cleanly(tmp_path: Path):
    db_file = tmp_path / "test.sqlite3"
    db = Database(db_file)

    # First run: should apply migrations
    applied = db.run_migrations()
    assert len(applied) >= 4
    assert "001_initial_schema.sql" in applied
    assert "003_resource_identity_key.sql" in applied
    assert "004_restore_annotations.sql" in applied

    with db.connection() as conn:
        resource_columns = {row["name"] for row in conn.execute("PRAGMA table_info(resources)")}
    assert "identity_key" in resource_columns

    # Second run: should be a no-op
    applied_again = db.run_migrations()
    assert len(applied_again) == 0


def test_migration_restores_missing_resource_identity_key(tmp_path: Path):
    db = Database(tmp_path / "legacy.sqlite3")
    with db.transaction() as conn:
        conn.execute(
            "CREATE TABLE schema_migrations (version INTEGER PRIMARY KEY, name TEXT NOT NULL, applied_at TEXT DEFAULT CURRENT_TIMESTAMP)"
        )
        conn.executemany(
            "INSERT INTO schema_migrations (version, name) VALUES (?, ?)",
            [(1, "001_initial_schema.sql"), (2, "002_project_workspaces.sql")],
        )
        conn.execute(
            "CREATE TABLE resources (id TEXT PRIMARY KEY, canonical_url TEXT, latest_content_hash TEXT)"
        )
        conn.executemany(
            "INSERT INTO resources (id, canonical_url, latest_content_hash) VALUES (?, ?, ?)",
            [
                ("res_url", "https://example.test/article", None),
                ("res_blob", None, "a" * 64),
            ],
        )
    db.sync_registries = lambda _conn: None

    applied = db.run_migrations()

    assert applied == [
        "003_resource_identity_key.sql",
        "004_restore_annotations.sql",
        "005_extraction_note.sql",
    ]
    with db.connection() as conn:
        keys = {
            row["id"]: row["identity_key"]
            for row in conn.execute("SELECT id, identity_key FROM resources")
        }
        annotations = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='annotations'"
        ).fetchone()
    assert keys == {
        "res_url": "url:https://example.test/article",
        "res_blob": f"blob:{'a' * 64}",
    }
    assert annotations is not None


def test_schema_tables_exist(test_db: Database):
    expected_tables = {
        "schema_migrations",
        "captures",
        "resources",
        "capture_resources",
        "source_snapshots",
        "resource_contents",
        "resource_chunks",
        "attachments",
        "annotations",
        "findings",
        "finding_support",
        "label_families",
        "labels",
        "object_labels",
        "entities",
        "object_entities",
        "intents",
        "projects",
        "project_objects",
        "outlines",
        "outline_sections",
        "outline_section_evidence",
        "judgments",
        "embeddings",
        "research_runs",
        "research_tasks",
        "idempotency_keys",
        "processing_jobs",
        "audit_events",
        "source_cursors",
        "search_documents",
    }

    with test_db.connection() as conn:
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = {row["name"] for row in cursor.fetchall()}
        missing = expected_tables - tables
        assert not missing, f"Missing tables in schema: {missing}"


def test_wal_and_foreign_keys(test_db: Database):
    with test_db.connection() as conn:
        fk = conn.execute("PRAGMA foreign_keys;").fetchone()[0]
        assert fk == 1, "Foreign keys must be enabled"

        mode = conn.execute("PRAGMA journal_mode;").fetchone()[0]
        assert mode.lower() == "wal", "Journal mode must be WAL"


def test_atomic_migration_rollback_on_failure(tmp_path: Path):
    import sqlite3

    import pytest

    db_file = tmp_path / "atomic_test.sqlite3"
    db = Database(db_file)

    # Initial valid migration to set up schema_migrations
    db.run_migrations()

    # Create a failing migration
    migrations_dir = tmp_path / "test_migrations"
    migrations_dir.mkdir(parents=True, exist_ok=True)
    bad_migration = migrations_dir / "002_broken.sql"
    bad_migration.write_text(
        "CREATE TABLE test_rollback_table (id INT);\nTHIS IS AN INVALID SQL STATEMENT THAT WILL FAIL;"
    )

    with pytest.raises(sqlite3.OperationalError):
        db.run_migrations(migrations_dir=migrations_dir)

    # Verify rollback: test_rollback_table must NOT exist, and 002_broken.sql not in schema_migrations
    with db.connection() as conn:
        check_table = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='test_rollback_table';"
        ).fetchone()
        assert check_table is None, "Table should have been rolled back"

        check_mig = conn.execute(
            "SELECT 1 FROM schema_migrations WHERE version='002_broken.sql';"
        ).fetchone()
        assert check_mig is None, "Migration record should not exist"
