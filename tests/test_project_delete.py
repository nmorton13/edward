"""Tests for edward project delete command and delete_project service."""

import pytest
from typer.testing import CliRunner

from edward.cli import app
from edward.db import Database
from edward.services.projects import (
    ProjectError,
    create_project,
    delete_project,
    get_project_context,
)

runner = CliRunner()


def test_delete_project_service(test_db: Database):
    with test_db.transaction() as conn:
        p = create_project(conn, title="Test Deletion", brief="A brief")
        pid = p.id

        res = delete_project(conn, pid, actor="test-user")
        assert res["status"] == "deleted"
        assert res["id"] == pid
        assert res["title"] == "Test Deletion"

        # Verify audit event
        audit = conn.execute(
            "SELECT * FROM audit_events WHERE object_type = 'project' AND object_id = ? AND event_type = 'project.deleted';",
            (pid,),
        ).fetchone()
        assert audit is not None
        assert audit["actor"] == "test-user"

        # Verify project is marked is_deleted = 1
        row = conn.execute("SELECT is_deleted FROM projects WHERE id = ?;", (pid,)).fetchone()
        assert row["is_deleted"] == 1

        # Trying to delete again raises ProjectError
        with pytest.raises(ProjectError, match="not found"):
            delete_project(conn, pid)

        # get_project_context raises ProjectError on deleted project
        with pytest.raises(ProjectError, match="not found"):
            get_project_context(conn, pid)


def test_cli_project_delete_requires_confirm(test_db: Database, monkeypatch):
    monkeypatch.setenv("EDWARD_DB_PATH", str(test_db.db_path))

    with test_db.transaction() as conn:
        p = create_project(conn, title="CLI Project", brief="A brief")
        pid = p.id

    # Refuses without --confirm
    res = runner.invoke(app, ["project", "delete", pid, "--json"])
    assert res.exit_code == 2
    assert "Project deletion requires explicit --confirm flag" in res.stderr

    # Succeeds with --confirm
    res_ok = runner.invoke(app, ["project", "delete", pid, "--confirm", "--json"])
    assert res_ok.exit_code == 0
    assert '"status": "deleted"' in res_ok.stdout

    # Subsequent deletion fails with exit code 1
    res_again = runner.invoke(app, ["project", "delete", pid, "--confirm", "--json"])
    assert res_again.exit_code == 1
