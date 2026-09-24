"""Read-only commands must not be locked out by a long-running writer.

Every CLI command calls `Database.run_migrations()` on startup. It used to take
`BEGIN IMMEDIATE` unconditionally, so an up-to-date database was still write-locked on
every invocation: while `reindex` held its single long transaction, `status`, `doctor`
and `search` all failed with "database is locked" after a 5s busy timeout.

Migrations and registry sync are idempotent by construction, so an up-to-date database
has nothing to write. These tests pin both halves: no write lock when clean, and a
correct write path whenever real work exists.
"""

import shutil
import sqlite3
from pathlib import Path

import pytest

from edward.db import Database

# Use the real packaged migrations so the schema (and therefore the registry probe)
# behaves exactly as it does in production.
REAL_MIGRATIONS = Path(__file__).resolve().parents[1] / "src" / "edward" / "migrations"


def _real_migrations_copy(tmp_path, extra: dict[str, str] | None = None) -> Path:
    """A temp migrations dir holding the real schema, optionally plus extra files."""
    target = tmp_path / "migrations"
    target.mkdir(exist_ok=True)
    for source in sorted(REAL_MIGRATIONS.glob("*.sql")):
        shutil.copy(source, target / source.name)
    for name, sql in (extra or {}).items():
        (target / name).write_text(sql)
    return target


def _hold_write_lock(db_path) -> sqlite3.Connection:
    """Simulate a long-running writer (what reindex does for its whole run)."""
    holder = sqlite3.connect(str(db_path), timeout=0.5)
    holder.execute("PRAGMA journal_mode = WAL;")
    holder.execute("BEGIN IMMEDIATE;")
    return holder


def test_up_to_date_database_takes_no_write_lock(tmp_path):
    """The core regression: run_migrations must be callable while a writer is active."""
    db_path = tmp_path / "edward.sqlite3"
    db = Database(db_path)
    migrations = _real_migrations_copy(tmp_path)
    db.run_migrations(migrations)

    holder = _hold_write_lock(db_path)
    try:
        # Raises OperationalError("database is locked") before the fix.
        assert db.run_migrations(migrations) == []
    finally:
        holder.rollback()
        holder.close()


def test_pending_migration_still_applies_under_contention(tmp_path):
    """The probe is fail-open: a concurrent writer must not cause a skipped migration."""
    db_path = tmp_path / "edward.sqlite3"
    db = Database(db_path)
    migrations = _real_migrations_copy(tmp_path)
    db.run_migrations(migrations)

    # A sixth migration exists but is not applied yet.
    (migrations / "006_add_probe_table.sql").write_text(
        "CREATE TABLE IF NOT EXISTS probe_table (id TEXT PRIMARY KEY);"
    )

    holder = _hold_write_lock(db_path)
    try:
        # Contention means the probe cannot confirm state, so it must try the write path
        # and surface the lock rather than silently skipping a pending migration.
        with pytest.raises(sqlite3.OperationalError):
            db.run_migrations(migrations)
    finally:
        holder.rollback()
        holder.close()

    # Once contention clears, the pending migration must apply.
    assert db.run_migrations(migrations) == ["006_add_probe_table.sql"]

    with db.connection() as conn:
        versions = {r["version"] for r in conn.execute("SELECT version FROM schema_migrations;")}
        assert 6 in versions
        assert conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='probe_table';"
        ).fetchone()


def test_fresh_database_applies_every_migration(tmp_path):
    """A fresh database has real work, so the write path must run."""
    db_path = tmp_path / "edward.sqlite3"
    db = Database(db_path)
    migrations = _real_migrations_copy(tmp_path)

    applied = db.run_migrations(migrations)
    assert applied, "fresh database should apply the packaged migrations"
    assert len(applied) == len(list(REAL_MIGRATIONS.glob("*.sql")))


def test_wiped_registries_are_re_synced(tmp_path):
    """A wiped labels table means the registries need re-syncing."""
    db_path = tmp_path / "edward.sqlite3"
    db = Database(db_path)
    migrations = _real_migrations_copy(tmp_path)
    db.run_migrations(migrations)

    with db.transaction() as conn:
        conn.execute("DELETE FROM labels;")
        conn.execute("DELETE FROM label_families;")

    db.run_migrations(migrations)

    with db.connection() as conn:
        assert conn.execute("SELECT count(*) FROM labels;").fetchone()[0] > 0
