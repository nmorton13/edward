"""Tests for the job-backlog purge.

Deleting queued jobs is destructive on a shared queue, so these tests pin the
safety properties rather than the happy path alone: only the named stage and status
may be touched, counts are reported from both sides of the delete, and the CLI
refuses to delete without explicit confirmation.
"""

import json

import pytest

from edward.services.processor import count_jobs, purge_jobs


def _seed_jobs(db, stage: str, status: str, count: int, prefix: str) -> None:
    with db.transaction() as conn:
        for i in range(count):
            conn.execute(
                """
                INSERT INTO processing_jobs (id, job_key, stage, status, available_at, attempts, created_at, updated_at)
                VALUES (?, ?, ?, ?, '2026-01-01', 0, '2026-01-01', '2026-01-01');
                """,
                (f"{prefix}_{i}", f"{stage}:{prefix}_{i}", stage, status),
            )


def test_purge_removes_only_the_named_stage(test_db):
    """A purge of one stage must leave every other stage exactly as it was."""
    _seed_jobs(test_db, "finding-extraction", "pending", 7, "fe")
    _seed_jobs(test_db, "classify", "pending", 5, "cl")
    _seed_jobs(test_db, "attachment-ocr", "pending", 3, "ocr")

    with test_db.transaction() as conn:
        result = purge_jobs(conn, stage="finding-extraction", status="pending")

    assert result["before_stage"] == 7
    assert result["after_stage"] == 0
    assert result["deleted"] == 7
    assert result["other_stages_deleted"] == 0

    with test_db.connection() as conn:
        assert count_jobs(conn, stage="finding-extraction", status="pending") == 0
        assert count_jobs(conn, stage="classify", status="pending") == 5
        assert count_jobs(conn, stage="attachment-ocr", status="pending") == 3


def test_purge_removes_only_the_named_status(test_db):
    """Completed and failed jobs of the same stage must survive a pending purge."""
    _seed_jobs(test_db, "finding-extraction", "pending", 4, "fe_p")
    _seed_jobs(test_db, "finding-extraction", "completed", 2, "fe_c")
    _seed_jobs(test_db, "finding-extraction", "failed", 1, "fe_f")

    with test_db.transaction() as conn:
        result = purge_jobs(conn, stage="finding-extraction", status="pending")

    assert result["deleted"] == 4

    with test_db.connection() as conn:
        assert count_jobs(conn, stage="finding-extraction", status="pending") == 0
        assert count_jobs(conn, stage="finding-extraction", status="completed") == 2
        assert count_jobs(conn, stage="finding-extraction", status="failed") == 1


def test_purge_reports_counts_from_both_sides(test_db):
    """The returned counts must be measured before and after, not self-reported."""
    _seed_jobs(test_db, "finding-extraction", "pending", 6, "fe")
    _seed_jobs(test_db, "classify", "pending", 4, "cl")

    with test_db.transaction() as conn:
        result = purge_jobs(conn, stage="finding-extraction", status="pending")

    assert result["before_total"] == 10
    assert result["after_total"] == 4
    assert result["before_stage"] - result["after_stage"] == result["deleted"] == 6


def test_purge_of_an_empty_stage_is_a_no_op(test_db):
    """Purging nothing must report zero rather than error."""
    _seed_jobs(test_db, "classify", "pending", 2, "cl")

    with test_db.transaction() as conn:
        result = purge_jobs(conn, stage="finding-extraction", status="pending")

    assert result["deleted"] == 0
    assert result["before_total"] == result["after_total"] == 2


def test_purge_leaves_derived_data_tables_untouched(test_db):
    """Job deletion must not cascade into findings, resources, or contents."""
    _seed_jobs(test_db, "finding-extraction", "pending", 3, "fe")
    with test_db.transaction() as conn:
        conn.execute(
            """
            INSERT INTO resources (id, identity_key, canonical_url, title, review_state, is_deleted, created_at, updated_at)
            VALUES ('res_keep', 'url:keep', 'https://example.com/keep', 'Keep', 'unreviewed', 0, '2026-01-01', '2026-01-01');
            """
        )
        conn.execute(
            """
            INSERT INTO findings (id, resource_id, statement, assertion_role, review_state, is_deleted, created_at, updated_at)
            VALUES ('fin_keep', 'res_keep', 'A durable finding', 'source-claim', 'unreviewed', 0, '2026-01-01', '2026-01-01');
            """
        )

    with test_db.transaction() as conn:
        purge_jobs(conn, stage="finding-extraction", status="pending")

    with test_db.connection() as conn:
        assert conn.execute("SELECT COUNT(*) FROM findings;").fetchone()[0] == 1
        assert (
            conn.execute("SELECT COUNT(*) FROM resources WHERE id='res_keep';").fetchone()[0] == 1
        )


# ---------------------------------------------------------------------------
# CLI contract
# ---------------------------------------------------------------------------


def test_cli_refuses_to_delete_without_confirmation(test_db, cli_runner, monkeypatch):
    """A destructive delete requires --yes; refusal is a usage error."""
    from edward import cli

    monkeypatch.setattr(cli, "get_services", lambda: (test_db, None))
    _seed_jobs(test_db, "finding-extraction", "pending", 3, "fe")

    result = cli_runner.invoke(cli.app, ["purge-jobs", "--stage", "finding-extraction", "--json"])

    assert result.exit_code == 2
    assert "Refusing to delete" in json.loads(result.stderr)["error"]

    with test_db.connection() as conn:
        assert count_jobs(conn, stage="finding-extraction", status="pending") == 3


def test_cli_dry_run_deletes_nothing(test_db, cli_runner, monkeypatch):
    """--dry-run must report the count and change nothing."""
    from edward import cli

    monkeypatch.setattr(cli, "get_services", lambda: (test_db, None))
    _seed_jobs(test_db, "finding-extraction", "pending", 5, "fe")

    result = cli_runner.invoke(
        cli.app, ["purge-jobs", "--stage", "finding-extraction", "--dry-run", "--json"]
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["would_delete"] == 5
    assert payload["dry_run"] is True

    with test_db.connection() as conn:
        assert count_jobs(conn, stage="finding-extraction", status="pending") == 5


def test_cli_deletes_with_confirmation(test_db, cli_runner, monkeypatch):
    """--yes performs the delete and reports the counts."""
    from edward import cli

    monkeypatch.setattr(cli, "get_services", lambda: (test_db, None))
    _seed_jobs(test_db, "finding-extraction", "pending", 5, "fe")
    _seed_jobs(test_db, "classify", "pending", 2, "cl")

    result = cli_runner.invoke(
        cli.app, ["purge-jobs", "--stage", "finding-extraction", "--yes", "--json"]
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["deleted"] == 5
    assert payload["other_stages_deleted"] == 0

    with test_db.connection() as conn:
        assert count_jobs(conn, stage="finding-extraction", status="pending") == 0
        assert count_jobs(conn, stage="classify", status="pending") == 2


@pytest.mark.parametrize("bad_stage", ["nonexistent-stage", ""])
def test_cli_purge_of_an_unknown_stage_deletes_nothing(test_db, cli_runner, monkeypatch, bad_stage):
    """An unmatched stage or status is a no-op, never a wider delete."""
    from edward import cli

    monkeypatch.setattr(cli, "get_services", lambda: (test_db, None))
    _seed_jobs(test_db, "classify", "pending", 3, "cl")

    result = cli_runner.invoke(cli.app, ["purge-jobs", "--stage", bad_stage, "--yes", "--json"])

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["deleted"] == 0

    with test_db.connection() as conn:
        assert count_jobs(conn, stage="classify", status="pending") == 3
