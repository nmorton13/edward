"""End-to-end subprocess CLI contract tests for Edward.

These tests execute the installed Edward CLI through a real subprocess
(sys.executable -m edward) rather than Typer's in-process CliRunner,
verifying the most important deterministic CLI guarantees:

- `--json` success output contains only valid JSON on stdout.
- Human-facing diagnostics and errors do not contaminate JSON stdout.
- Exit codes follow the documented contract (0, 1, 2, 3).
- Evidence packet output validates against the public schema.
- Tests use an isolated temporary data directory.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

# Schema validation is available via the dev dependency
import jsonschema
import pytest

SCHEMAS_DIR = Path(__file__).parent.parent / "schemas"


def _run_edward(
    args: list[str],
    *,
    env_overrides: dict[str, str] | None = None,
    stdin_data: str | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run edward via subprocess with isolated environment."""
    env = {
        **os.environ,
        # Prevent any .env file loading from polluting tests
        "EDWARD_CLASSIFIER_PROVIDER": "disabled",
        "EDWARD_ANSWER_BASE_URL": "",
    }
    if env_overrides:
        env.update(env_overrides)
    return subprocess.run(
        [sys.executable, "-m", "edward", *args],
        capture_output=True,
        text=True,
        env=env,
        input=stdin_data,
        timeout=30,
    )


@pytest.fixture
def isolated_data_dir(tmp_path: Path) -> dict[str, str]:
    """Create an isolated data directory and return env overrides."""
    data_dir = tmp_path / "edward_e2e"
    data_dir.mkdir(parents=True)
    (data_dir / "blobs").mkdir()
    (data_dir / "diagnostics").mkdir()
    db_path = data_dir / "test.sqlite3"
    return {
        "EDWARD_DATA_DIR": str(data_dir),
        "EDWARD_DB_PATH": str(db_path),
    }


# -----------------------------------------------------------------------
# Exit code 0 — success
# -----------------------------------------------------------------------


class TestExitCode0:
    """Verify exit code 0 on successful operations with valid JSON stdout."""

    def test_add_text_json_stdout_is_valid_json(self, isolated_data_dir):
        """--json add produces only valid JSON on stdout, exit 0."""
        result = _run_edward(
            ["add", "--text", "Test content for subprocess", "--json"],
            env_overrides=isolated_data_dir,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        # stdout must be parseable JSON
        data = json.loads(result.stdout)
        assert data["status"] == "created"
        assert "capture_id" in data
        assert "resource_id" in data

    def test_search_json_stdout_is_valid_json(self, isolated_data_dir):
        """--json search produces only valid JSON on stdout."""
        # Seed data first and assert it succeeded
        seed_res = _run_edward(
            ["add", "--text", "Searchable research note", "--json"],
            env_overrides=isolated_data_dir,
        )
        assert seed_res.returncode == 0, f"Seed failed: {seed_res.stderr}"
        result = _run_edward(
            ["search", "research", "--json"],
            env_overrides=isolated_data_dir,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        data = json.loads(result.stdout)
        assert "count" in data
        assert "results" in data
        assert data["count"] >= 1

    def test_doctor_json_stdout_is_valid_json(self, isolated_data_dir):
        """--json doctor produces only valid JSON with healthy status."""
        result = _run_edward(
            ["doctor", "--json"],
            env_overrides=isolated_data_dir,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        data = json.loads(result.stdout)
        assert data["healthy"] is True
        assert data["mode"] == "system_diagnostics"

    def test_status_json_stdout_is_valid_json(self, isolated_data_dir):
        """--json status produces valid JSON."""
        result = _run_edward(
            ["status", "--json"],
            env_overrides=isolated_data_dir,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        data = json.loads(result.stdout)
        assert "queue" in data or "captures" in data or isinstance(data, dict)

    def test_backup_json(self, isolated_data_dir, tmp_path):
        """--json backup produces valid JSON with backup_path."""
        backup_dir = tmp_path / "backups"
        result = _run_edward(
            ["backup", "--dest", str(backup_dir), "--json"],
            env_overrides=isolated_data_dir,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        data = json.loads(result.stdout)
        assert "backup_path" in data


# -----------------------------------------------------------------------
# Exit code 1 — domain/input failure
# -----------------------------------------------------------------------


class TestExitCode1:
    """Verify exit code 1 on domain or input failures."""

    def test_show_nonexistent_id(self, isolated_data_dir):
        """show with nonexistent ID returns exit code 1."""
        result = _run_edward(
            ["show", "nonexistent_id_xyz", "--json"],
            env_overrides=isolated_data_dir,
        )
        assert result.returncode == 1

    def test_add_invalid_url(self, isolated_data_dir):
        """add with a clearly invalid URL returns exit code 1 (domain error)."""
        result = _run_edward(
            ["add", "--url", "not-a-valid-url", "--json"],
            env_overrides=isolated_data_dir,
        )
        assert result.returncode == 1


# -----------------------------------------------------------------------
# Exit code 2 — usage error
# -----------------------------------------------------------------------


class TestExitCode2:
    """Verify exit code 2 on CLI usage errors."""

    def test_add_no_input(self, isolated_data_dir):
        """add with no URL, text, file, or stdin returns exit code 2 (usage error)."""
        result = _run_edward(
            ["add", "--json"],
            env_overrides=isolated_data_dir,
        )
        assert result.returncode == 2

    def test_purge_without_confirm(self, isolated_data_dir):
        """purge without --confirm returns exit code 2."""
        # First add an item to get a real ID
        add_result = _run_edward(
            ["add", "--text", "Item to purge", "--json"],
            env_overrides=isolated_data_dir,
        )
        cap_id = json.loads(add_result.stdout)["capture_id"]

        result = _run_edward(
            ["purge", cap_id],
            env_overrides=isolated_data_dir,
        )
        assert result.returncode == 2

    def test_unknown_subcommand(self, isolated_data_dir):
        """Unknown subcommand returns exit code 2."""
        result = _run_edward(
            ["nonexistent-command"],
            env_overrides=isolated_data_dir,
        )
        assert result.returncode == 2


# -----------------------------------------------------------------------
# Exit code 3 — conflict/fatal error
# -----------------------------------------------------------------------


class TestExitCode3:
    """Verify exit code 3 on idempotency conflicts."""

    def test_idempotency_conflict(self, isolated_data_dir):
        """Replaying an idempotency key with different args returns exit code 3."""
        idem_key = "test-idem-conflict-key"

        # First call succeeds
        result1 = _run_edward(
            [
                "add",
                "--text",
                "Original content",
                "--idempotency-key",
                idem_key,
                "--json",
            ],
            env_overrides=isolated_data_dir,
        )
        assert result1.returncode == 0, f"stderr: {result1.stderr}"

        # Second call with same key but different content triggers conflict
        result2 = _run_edward(
            [
                "add",
                "--text",
                "Different content",
                "--idempotency-key",
                idem_key,
                "--json",
            ],
            env_overrides=isolated_data_dir,
        )
        assert result2.returncode == 3


# -----------------------------------------------------------------------
# Stdout/stderr isolation
# -----------------------------------------------------------------------


class TestStreamIsolation:
    """Verify that JSON stdout is not contaminated by human-facing output."""

    def test_json_stdout_contains_no_rich_markup(self, isolated_data_dir):
        """--json stdout must not contain Rich markup tags."""
        result = _run_edward(
            ["add", "--text", "Clean stdout test", "--json"],
            env_overrides=isolated_data_dir,
        )
        assert result.returncode == 0
        # Rich markup tags like [bold green] must not appear in stdout
        assert "[bold" not in result.stdout
        assert "[/bold" not in result.stdout
        assert "[red" not in result.stdout
        # Must be valid JSON
        json.loads(result.stdout)

    def test_error_json_goes_to_stderr(self, isolated_data_dir):
        """On error with --json, error message goes to stderr, not stdout."""
        result = _run_edward(
            ["show", "nonexistent_id", "--json"],
            env_overrides=isolated_data_dir,
        )
        assert result.returncode == 1
        # stderr should contain the error
        assert "error" in result.stderr.lower() or "not found" in result.stderr.lower()
        # stdout should be empty or not contain the error diagnostic
        if result.stdout.strip():
            # If there's any stdout, it must be valid JSON (not error text)
            json.loads(result.stdout)

    def test_doctor_json_stdout_only_json(self, isolated_data_dir):
        """doctor --json produces only JSON on stdout with no human text mixed in."""
        result = _run_edward(
            ["doctor", "--json"],
            env_overrides=isolated_data_dir,
        )
        assert result.returncode == 0
        # Entire stdout must parse as a single JSON document
        data = json.loads(result.stdout)
        assert isinstance(data, dict)


# -----------------------------------------------------------------------
# Schema validation
# -----------------------------------------------------------------------


class TestSchemaValidation:
    """Verify output conformance to published JSON schemas."""

    def test_evidence_packet_validates_against_schema(self, isolated_data_dir):
        """Exported evidence packet conforms to schemas/evidence-packet-v1.json."""
        # Seed data and assert seed success
        seed_res = _run_edward(
            ["add", "--text", "Quantum computing research note for schema test", "--json"],
            env_overrides=isolated_data_dir,
        )
        assert seed_res.returncode == 0, f"Seed failed: {seed_res.stderr}"

        # Export evidence packet
        result = _run_edward(
            ["export", "--packet", "--query", "quantum"],
            env_overrides=isolated_data_dir,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        packet = json.loads(result.stdout)
        assert len(packet["items"]) >= 1, "Evidence packet items must not be empty after seeding"

        # Validate against the canonical schema
        schema_path = SCHEMAS_DIR / "evidence-packet-v1.json"
        assert schema_path.exists(), f"Schema file not found: {schema_path}"
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        jsonschema.validate(instance=packet, schema=schema)

    def test_evidence_packet_required_fields(self, isolated_data_dir):
        """Evidence packet contains all required top-level fields."""
        seed_res = _run_edward(
            ["add", "--text", "Test data for packet fields", "--json"],
            env_overrides=isolated_data_dir,
        )
        assert seed_res.returncode == 0, f"Seed failed: {seed_res.stderr}"
        result = _run_edward(
            ["export", "--packet", "--query", "test"],
            env_overrides=isolated_data_dir,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        packet = json.loads(result.stdout)

        assert packet["type"] == "evidence-packet"
        assert packet["schema_version"] == "1"
        assert "query" in packet
        assert "created_at" in packet
        assert "items" in packet
        assert isinstance(packet["items"], list)
        assert len(packet["items"]) >= 1
