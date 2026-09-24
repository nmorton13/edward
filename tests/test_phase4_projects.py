"""Phase 4 project workspace and outline tests."""

import json

import pytest

from edward.models import CaptureInput
from edward.services.capture import capture_item
from edward.services.lifecycle import purge_object
from edward.services.projects import (
    ProjectConflictError,
    add_project_note,
    add_project_object,
    create_project,
    deterministic_outline,
    get_outline,
    get_project_context,
    save_outline,
    set_outline_status,
    suggest_project_evidence,
)


def _capture(db, text: str) -> str:
    with db.transaction() as conn:
        result = capture_item(conn, CaptureInput(text=text))
    return result["capture_id"]


def test_project_create_membership_and_context(test_db):
    capture_id = _capture(test_db, "Local AI tools change creative work and editing workflows.")
    with test_db.transaction() as conn:
        project = create_project(
            conn,
            "Local AI and creative work",
            "How are local AI tools changing creative work?",
        )
        membership = add_project_object(
            conn,
            project.id,
            capture_id,
            relationship="supporting",
            membership_status="accepted",
            relevance_note="First-hand framing",
        )
        add_project_note(conn, project.id, "Find evidence about adoption rates", kind="gap")

    with test_db.connection() as conn:
        context = get_project_context(conn, project.id)

    assert project.slug == "local-ai-and-creative-work"
    assert membership["membership_status"] == "accepted"
    assert context["accepted_evidence"][0]["item"]["id"] == capture_id
    assert any(
        g["description"] == "Find evidence about adoption rates"
        for g in context["gaps_and_counterarguments"]
    )
    assert any("counterargument" in g["description"] for g in context["gaps_and_counterarguments"])


def test_project_slug_conflict_is_explicit(test_db):
    with test_db.transaction() as conn:
        create_project(conn, "One", slug="same")
    with pytest.raises(ProjectConflictError):
        with test_db.transaction() as conn:
            create_project(conn, "Two", slug="same")


def test_automated_candidate_cannot_overwrite_human_decision(test_db):
    capture_id = _capture(test_db, "Evidence")
    with test_db.transaction() as conn:
        project = create_project(conn, "Durable decisions")
        add_project_object(
            conn, project.id, capture_id, membership_status="rejected", added_by="human"
        )
        row = add_project_object(
            conn, project.id, capture_id, membership_status="candidate", added_by="system"
        )
    assert row["membership_status"] == "rejected"
    assert row["added_by"] == "human"


def test_search_can_filter_by_project_membership(test_db):
    included = _capture(test_db, "Shared keyword alpha")
    _capture(test_db, "Shared keyword beta")
    with test_db.transaction() as conn:
        project = create_project(conn, "Filtered search")
        add_project_object(conn, project.id, included)
    from edward.services.search import search_lexical

    with test_db.connection() as conn:
        response = search_lexical(conn, "Shared keyword", project=project.id)
    assert [item.id for item in response.results] == [included]


def test_project_scoped_deterministic_answer(test_db):
    from edward.services.answer import answer_question

    with test_db.transaction() as conn:
        project = create_project(conn, "Scoped answer")
    with test_db.connection() as conn:
        answer = answer_question(
            conn,
            "How many unreviewed findings are in this project?",
            no_model=True,
            project_id=project.id,
        )
    assert answer["tier"] == 1
    assert answer["data"] == {"count": 0, "project_id": project.id}


def test_candidate_retrieval_persists_suggestions(test_db):
    capture_id = _capture(test_db, "Ceramic robots make studio pottery faster.")
    with test_db.transaction() as conn:
        project = create_project(conn, "Ceramic robots", "ceramic robots studio pottery")
        suggestions = suggest_project_evidence(conn, project.id, persist=True)
    assert any(item["id"] == capture_id for item in suggestions)
    with test_db.connection() as conn:
        context = get_project_context(conn, project.id)
    assert context["candidate_evidence"][0]["added_by"] == "system"


def test_versioned_outline_links_survive_revision_and_acceptance(test_db):
    capture_id = _capture(test_db, "Local inference gives creators private iterative tools.")
    with test_db.transaction() as conn:
        project = create_project(conn, "Local creative tools", "Local inference changes creation")
        add_project_object(conn, project.id, capture_id)
    with test_db.connection() as conn:
        context = get_project_context(conn, project.id)
    proposal = deterministic_outline(context)

    with test_db.transaction() as conn:
        first = save_outline(conn, project.id, proposal, author_type="system")
    proposal.sections[0].heading = "A revised opening"
    with test_db.transaction() as conn:
        second = save_outline(
            conn,
            project.id,
            proposal,
            author_type="human",
            parent_outline_id=first["id"],
            revision_instructions="Strengthen the opening",
        )
        accepted = set_outline_status(conn, project.id, "accepted", version=2)

    assert first["version"] == 1
    assert second["version"] == 2
    assert second["parent_outline_id"] == first["id"]
    assert accepted["status"] == "accepted"
    assert second["sections"][0]["evidence"][0]["object_id"] == capture_id
    with test_db.connection() as conn:
        unchanged = get_outline(conn, project.id, version=1)
    assert unchanged["sections"][0]["heading"] == "Premise and stakes"


def test_outline_rejects_evidence_outside_project(test_db):
    outsider = _capture(test_db, "Not project evidence")
    with test_db.transaction() as conn:
        project = create_project(conn, "Bounded outline")
    proposal = {
        "title": "Bad proposal",
        "sections": [{"heading": "Section", "evidence": [{"object_id": outsider}]}],
    }
    with pytest.raises(ValueError, match="not accepted or candidate"):
        with test_db.transaction() as conn:
            save_outline(conn, project.id, proposal)


def test_purge_removes_polymorphic_project_references(test_db):
    capture_id = _capture(test_db, "Temporary project evidence")
    with test_db.transaction() as conn:
        project = create_project(conn, "Purge references")
        add_project_object(conn, project.id, capture_id)
    with test_db.connection() as conn:
        proposal = deterministic_outline(get_project_context(conn, project.id))
    with test_db.transaction() as conn:
        saved = save_outline(conn, project.id, proposal)
        purge_object(conn, "capture", capture_id)
    with test_db.connection() as conn:
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM project_objects WHERE object_id = ?", (capture_id,)
            ).fetchone()[0]
            == 0
        )
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM outline_section_evidence WHERE object_id = ?", (capture_id,)
            ).fetchone()[0]
            == 0
        )
        outline = get_outline(conn, project.id, version=saved["version"])
    assert all(not section["evidence"] for section in outline["sections"])


def test_project_cli_json_workflow(test_db, cli_runner, monkeypatch):
    from edward import cli

    monkeypatch.setattr(cli, "get_services", lambda: (test_db, None))
    created = cli_runner.invoke(
        cli.app,
        ["project", "create", "--title", "CLI project", "--brief", "A brief", "--json"],
    )
    assert created.exit_code == 0
    project_id = json.loads(created.stdout)["id"]

    outlined = cli_runner.invoke(
        cli.app,
        ["project", "outline", project_id, "--propose", "--no-model", "--json"],
    )
    assert outlined.exit_code == 0
    payload = json.loads(outlined.stdout)
    assert payload["version"] == 1
    assert payload["status"] == "proposal"
