"""Database engine and migrations manager for Edward."""

import json
import logging
import os
import re
import sqlite3
import sys
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger(__name__)

# Feature flag tracking vector extension availability
HAS_SQLITE_VEC = False


def parse_env_line(line: str) -> tuple[str, str] | None:
    """Parse a single line from a .env file, respecting quotes and discarding inline comments."""
    line = line.strip()
    if not line or line.startswith("#") or "=" not in line:
        return None

    key, val = line.split("=", 1)
    key = key.strip()
    val = val.strip()

    if key.startswith("export ") or key.startswith("export\t"):
        key = key[6:].strip()

    if not key:
        return None

    if val.startswith(('"', "'")):
        quote = val[0]
        end_idx = val.find(quote, 1)
        if end_idx != -1:
            val = val[1:end_idx]
        else:
            val = val[1:]
    else:
        # Unquoted: strip trailing inline comments starting with '#'
        if "#" in val:
            val = re.split(r"\s+#", val, maxsplit=1)[0].strip()

    return key, val


def load_env_file(env_path: Path | None = None) -> None:
    """Load key-value pairs from a .env file into os.environ if not already set."""
    candidates = (
        [env_path]
        if env_path
        else [
            Path.cwd() / ".env",
            Path.home() / ".edward" / ".env",
        ]
    )
    for candidate in candidates:
        if candidate and candidate.exists() and candidate.is_file():
            try:
                for line in candidate.read_text(encoding="utf-8").splitlines():
                    parsed = parse_env_line(line)
                    if parsed:
                        key, val = parsed
                        os.environ.setdefault(key, val)
            except Exception as e:
                logger.debug("Failed to read .env from %s: %s", candidate, e)


# Load environment on module import
load_env_file()


def get_default_data_dir() -> Path:
    """Return default platform app data directory from environment or user home."""
    env_dir = os.environ.get("EDWARD_DATA_DIR")
    if env_dir:
        path = Path(env_dir).expanduser().resolve()
    elif (Path.home() / ".edward").exists():
        path = Path.home() / ".edward"
    elif sys.platform == "darwin":
        path = Path.home() / "Library" / "Application Support" / "edward"
    elif sys.platform == "win32":
        path = Path(os.environ.get("APPDATA", Path.home())) / "edward"
    else:
        path = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / "edward"

    path.mkdir(parents=True, exist_ok=True)
    (path / "blobs").mkdir(parents=True, exist_ok=True)
    (path / "diagnostics").mkdir(parents=True, exist_ok=True)
    return path


def get_default_db_path() -> Path:
    """Return default sqlite database path from environment or default data dir."""
    env_db = os.environ.get("EDWARD_DB_PATH")
    if env_db:
        path = Path(env_db).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
    return get_default_data_dir() / "edward.sqlite3"


def split_sql_statements(sql: str) -> list[str]:
    """Split a multi-statement SQL script into individual statements safely without committing transactions."""
    statements: list[str] = []
    current: list[str] = []
    in_single = False
    in_double = False
    for line in sql.splitlines():
        trimmed = line.strip()
        if trimmed.startswith("--"):
            continue
        for char in line:
            if char == "'" and not in_double:
                in_single = not in_single
            elif char == '"' and not in_single:
                in_double = not in_double
            if char == ";" and not in_single and not in_double:
                stmt = "".join(current).strip()
                if stmt:
                    statements.append(stmt)
                current = []
            else:
                current.append(char)
        current.append("\n")
    stmt = "".join(current).strip()
    if stmt:
        statements.append(stmt)
    return statements


def execute_migration_statement(conn: sqlite3.Connection, statement: str) -> None:
    """Run migration SQL, tolerating an ALTER against a schema that already has it.

    Two cases are skipped rather than failed, because both mean "the schema is
    already where this migration wants it":

    * the column already exists (a database created after the migration landed)
    * the table does not exist at all (a partial or synthetic schema, such as a
      legacy fixture or an export that never carried the table)
    """
    add_column = re.fullmatch(
        r"ALTER\s+TABLE\s+([A-Za-z_]\w*)\s+ADD\s+COLUMN\s+([A-Za-z_]\w*)\s+.+",
        statement.strip(),
        flags=re.IGNORECASE | re.DOTALL,
    )
    if add_column:
        table_name, column_name = add_column.groups()
        columns = {row["name"] for row in conn.execute(f'PRAGMA table_info("{table_name}");')}
        if not columns:
            # No such table: nothing to alter, and creating it is not this
            # migration's job.
            return
        if column_name in columns:
            return
    conn.execute(statement)


class Database:
    """Encapsulates SQLite connection management, WAL mode, migrations, and extension loading."""

    def __init__(self, db_path: Path | str | None = None):
        self.db_path = Path(db_path) if db_path is not None else get_default_db_path()
        self.has_vec = False
        self._ensure_parent_dir()

    def _ensure_parent_dir(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def connect(self) -> sqlite3.Connection:
        """Create and configure a new SQLite connection."""
        conn = sqlite3.connect(
            str(self.db_path),
            timeout=10.0,
        )
        conn.row_factory = sqlite3.Row

        # Invariants: WAL mode, foreign keys, busy timeout
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA busy_timeout = 5000;")

        # Dynamic vector extension attempt with graceful fallback
        self._try_load_vec(conn)

        return conn

    def _try_load_vec(self, conn: sqlite3.Connection) -> None:
        """Attempt to load sqlite-vec extension gracefully."""
        global HAS_SQLITE_VEC
        try:
            import sqlite_vec

            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.enable_load_extension(False)
            self.has_vec = True
            HAS_SQLITE_VEC = True
        except ImportError:
            self.has_vec = False
        except Exception as e:
            logger.debug("sqlite-vec load failed (falling back to vector-free mode): %s", e)
            self.has_vec = False

    @contextmanager
    def connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager yielding a managed connection."""
        conn = self.connect()
        try:
            yield conn
        finally:
            conn.close()

    @contextmanager
    def transaction(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager for an atomic database transaction."""
        conn = self.connect()
        try:
            conn.execute("BEGIN IMMEDIATE;")
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def get_applied_migrations(self, conn: sqlite3.Connection) -> set[int]:
        """Return the set of migration versions that have already been applied."""
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                applied_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        cursor = conn.execute("SELECT version FROM schema_migrations ORDER BY version ASC;")
        return {row["version"] for row in cursor.fetchall()}

    def _has_pending_work(self, migration_files: list[Path]) -> bool:
        """Report whether any migration or registry row still needs writing.

        Read-only, and deliberately fail-open: if the probe cannot read the schema
        (for example a concurrent writer holds the lock) it returns True so the caller
        takes the write path. Skipping a pending migration would be far worse than
        briefly holding a lock that was going to be needed anyway.

        A *name mismatch* on an already-applied version also counts as pending work.
        That is how run_migrations detects a migration file being renamed or renumbered
        under an applied version, which is a conflict that must surface as an error --
        the fast path must not swallow it.
        """
        try:
            with self.connection() as conn:
                applied_rows = conn.execute(
                    "SELECT version, name FROM schema_migrations;"
                ).fetchall()
                applied_names = {row["version"]: row["name"] for row in applied_rows}
                applied_versions = set(applied_names)

                for file_path in migration_files:
                    match = re.match(r"^(\d+)_(.*)\.sql$", file_path.name)
                    if not match:
                        continue
                    version = int(match.group(1))
                    if version not in applied_versions:
                        return True
                    if applied_names.get(version) != file_path.name:
                        return True
                return self._has_pending_registry_rows(conn)
        except sqlite3.OperationalError:
            logger.debug("Could not probe schema state (locked?); taking the write path")
            return True

    def _has_pending_registry_rows(self, conn: sqlite3.Connection) -> bool:
        """Report whether the packaged registries are already fully synced."""
        registries_dir = Path(__file__).parent / "registries"
        if not registries_dir.exists():
            return False

        for filename in ["forms-v1.json", "signals-v1.json", "topics-v1.json"]:
            file_path = registries_dir / filename
            if not file_path.exists():
                continue
            try:
                data = json.loads(file_path.read_text(encoding="utf-8"))
            except Exception:
                # A malformed registry is sync_registries' problem to log, not a reason
                # to take the write path on every single command.
                continue
            family = data.get("family")
            if not family:
                continue
            if not conn.execute("SELECT 1 FROM label_families WHERE id = ?;", (family,)).fetchone():
                return True
            for entry in data.get("entries", []):
                entry_id = entry.get("id")
                if not entry_id:
                    continue
                if not conn.execute("SELECT 1 FROM labels WHERE id = ?;", (entry_id,)).fetchone():
                    return True
        return False

    def sync_registries(self, conn: sqlite3.Connection) -> None:
        """Sync packaged classification registries into label_families and labels tables."""
        registries_dir = Path(__file__).parent / "registries"
        if not registries_dir.exists():
            return

        for filename in ["forms-v1.json", "signals-v1.json", "topics-v1.json"]:
            file_path = registries_dir / filename
            if not file_path.exists():
                continue
            try:
                data = json.loads(file_path.read_text(encoding="utf-8"))
                family = data.get("family")
                if not family:
                    continue

                conn.execute(
                    """
                    INSERT OR IGNORE INTO label_families (id, description, created_at)
                    VALUES (?, ?, datetime('now'));
                    """,
                    (family, f"{family.capitalize()} taxonomy family"),
                )

                for entry in data.get("entries", []):
                    entry_id = entry.get("id")
                    if not entry_id:
                        continue
                    conn.execute(
                        """
                        INSERT OR IGNORE INTO labels (id, family, description, parent, active, version, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, datetime('now'));
                        """,
                        (
                            entry_id,
                            family,
                            entry.get("description"),
                            entry.get("parent"),
                            1 if entry.get("active", True) else 0,
                            entry.get("version", "1.0"),
                        ),
                    )
            except Exception as e:
                logger.warning("Failed to sync registry %s: %s", filename, e)

    def run_migrations(self, migrations_dir: Path | None = None) -> list[str]:
        """Run all pending versioned migrations in order and sync registries in atomic transactions.

        Migrations and registry sync are *idempotent by construction* (versions are
        recorded, registry rows are inserted with INSERT OR IGNORE), so an up-to-date
        database has nothing to do. Taking a write lock to discover that is not free:
        ``BEGIN IMMEDIATE`` blocks readers, and because every CLI command calls this on
        startup, a long-running writer (``reindex``) would lock out ``status``, ``doctor``
        and ``search`` for its entire run.

        So probe read-only first and only escalate to a write transaction when there is
        real work. The probe tolerates "database is locked" and falls back to the write
        path, so a concurrent writer can never make a pending migration be skipped.
        """
        if migrations_dir is None:
            migrations_dir = Path(__file__).parent / "migrations"

        if not migrations_dir.exists():
            return []

        migration_files = sorted(
            migrations_dir.glob("*.sql"),
            key=lambda p: (
                int(re.match(r"^(\d+)", p.name).group(1)) if re.match(r"^(\d+)", p.name) else 999999
            ),
        )

        if not self._has_pending_work(migration_files):
            logger.debug("Schema and registries already up to date; skipping write lock")
            return []

        applied: list[str] = []
        with self.transaction() as conn:
            applied_versions = self.get_applied_migrations(conn)
            applied_names = {
                row["version"]: row["name"]
                for row in conn.execute("SELECT version, name FROM schema_migrations;").fetchall()
            }

            for file_path in migration_files:
                match = re.match(r"^(\d+)_(.*)\.sql$", file_path.name)
                if not match:
                    continue

                version = int(match.group(1))

                if version in applied_versions:
                    if applied_names.get(version) != file_path.name:
                        raise sqlite3.OperationalError(
                            f"Migration version {version} is already applied as "
                            f"'{applied_names.get(version)}', not '{file_path.name}'"
                        )
                    continue

                sql = file_path.read_text(encoding="utf-8")
                # Execute statements one by one to avoid executescript's implicit COMMIT
                for stmt in split_sql_statements(sql):
                    execute_migration_statement(conn, stmt)

                conn.execute(
                    "INSERT INTO schema_migrations (version, name) VALUES (?, ?);",
                    (version, file_path.name),
                )
                applied.append(file_path.name)
                logger.info("Applied migration atomically: %s", file_path.name)

            # Sync packaged registries into labels tables
            self.sync_registries(conn)

        return applied

    def backup_database(self, dest_path: Path) -> None:
        """Perform an online SQLite backup safely capturing WAL pages."""
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_conn = sqlite3.connect(str(dest_path))
        try:
            with self.connection() as src_conn:
                src_conn.backup(dest_conn)
        finally:
            dest_conn.close()
