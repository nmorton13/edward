"""Regression tests for `edward project context` human-readable rendering.

Bug B: the command's help promises it shows "the brief, evidence, gaps,
counterarguments, notes, and latest outline", and `--json` includes
``latest_outline``. The text renderer printed the brief, evidence counts, and gaps
and then stopped, so the outline an author depends on was invisible in the default
human-readable view.
"""

import json

from edward.models import CaptureInput
from edward.services.capture import capture_item
from edward.services.projects import add_project_object, create_project


def _project_with_outline(db) -> tuple[str, dict]:
    """Create a project with one evidence item and a saved outline. Returns (project_id, outline)."""
    with db.transaction() as conn:
        captured = capture_item(
            conn,
            CaptureInput(
                text="Evidence body for the context rendering test",
                origin_namespace="manual",
                collection_channel="cli",
                collector="test",
                acquisition_method="manual",
            ),
        )
        project = create_project(conn, "Context Render Project", "Show me everything")
        add_project_object(
            conn,
            project.id,
            captured["capture_id"],
            membership_status="accepted",
            relevance_note="Primary evidence",
        )

    from edward.services.projects import save_outline

    outline_payload = {
        "title": "Context Render Project",
        "premise": "Render everything the brief promises.",
        "sections": [
            {
                "heading": "Opening stakes",
                "purpose": "Set up the question.",
                "claim": "The premise is contested.",
                "evidence": [{"object_id": captured["capture_id"], "relationship": "supporting"}],
            },
            {
                "heading": "The counter case",
                "purpose": "Present the opposing view.",
                "claim": "The premise may be overstated.",
                "evidence": [],
            },
        ],
    }
    with db.transaction() as conn:
        saved = save_outline(conn, project.id, outline_payload)

    return project.id, saved


def test_context_text_output_includes_the_latest_outline(test_db, cli_runner, monkeypatch):
    """The default human-readable view must show the outline, not just the JSON view."""
    from edward import cli

    monkeypatch.setattr(cli, "get_services", lambda: (test_db, None))
    project_id, outline = _project_with_outline(test_db)

    result = cli_runner.invoke(cli.app, ["project", "context", project_id])

    assert result.exit_code == 0, result.output
    assert "Opening stakes" in result.output, "outline section headings must be rendered"
    assert "The counter case" in result.output, "every section heading must be rendered"


def test_context_text_output_labels_the_outline_version_and_status(
    test_db, cli_runner, monkeypatch
):
    """An author needs to know which outline version they are looking at and whether it is accepted."""
    from edward import cli

    monkeypatch.setattr(cli, "get_services", lambda: (test_db, None))
    project_id, outline = _project_with_outline(test_db)

    result = cli_runner.invoke(cli.app, ["project", "context", project_id])

    assert f"version {outline['version']}" in result.output.lower()
    assert outline["status"] in result.output.lower()


def test_context_text_output_still_shows_gaps_and_evidence(test_db, cli_runner, monkeypatch):
    """Adding outline rendering must not drop the existing brief, gaps, or evidence output."""
    from edward import cli

    monkeypatch.setattr(cli, "get_services", lambda: (test_db, None))
    project_id, _ = _project_with_outline(test_db)

    result = cli_runner.invoke(cli.app, ["project", "context", project_id])

    assert "Accepted evidence: 1" in result.output
    assert "Show me everything" in result.output


def test_context_without_an_outline_renders_cleanly(test_db, cli_runner, monkeypatch):
    """A project with no outline must not crash or print a misleading outline header."""
    from edward import cli

    monkeypatch.setattr(cli, "get_services", lambda: (test_db, None))
    with test_db.transaction() as conn:
        project = create_project(conn, "No Outline Project", "Nothing to outline yet")

    result = cli_runner.invoke(cli.app, ["project", "context", project.id])

    assert result.exit_code == 0, result.output
    assert "No Outline Project" in result.output


def test_context_json_still_carries_the_outline(test_db, cli_runner, monkeypatch):
    """The JSON contract must not regress while the text renderer is fixed."""
    from edward import cli

    monkeypatch.setattr(cli, "get_services", lambda: (test_db, None))
    project_id, outline = _project_with_outline(test_db)

    result = cli_runner.invoke(cli.app, ["project", "context", project_id, "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["latest_outline"] is not None
    assert payload["latest_outline"]["version"] == outline["version"]
    assert len(payload["latest_outline"]["sections"]) == 2
