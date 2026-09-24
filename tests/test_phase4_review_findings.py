"""Regression tests for Phase 4 reviewer findings.

Covers:
1. High: Outline generation privacy gate across mixed data classes (Gmail, documents, personal_notes).
2. High: Resource purge preserves findings and their project memberships and outline links.
3. Medium: Project-scoped deterministic answer routing for counts, dates, and titles.
4. Medium: Deterministic CLI exit codes (0 success, 1 domain, 2 usage, 3 conflict/fatal) across all project commands.
"""

import json
from unittest.mock import MagicMock

import pytest

from edward.models import CaptureInput, OutlineProposalInput
from edward.services.answer import try_deterministic_answer
from edward.services.capture import capture_item
from edward.services.lifecycle import purge_object
from edward.services.privacy import PrivacyTransmissionError
from edward.services.projects import (
    add_project_object,
    create_project,
    deterministic_outline,
    generate_outline_proposal,
    get_outline,
    get_project_context,
    save_outline,
)


def _capture_with_ns(db, text: str, origin_namespace: str = "manual") -> str:
    with db.transaction() as conn:
        res = capture_item(
            conn,
            CaptureInput(
                text=text,
                origin_namespace=origin_namespace,
                collection_channel="cli",
                collector="test",
                acquisition_method="manual",
            ),
        )
    return res["capture_id"]


# ---------------------------------------------------------------------------
# Finding 1: Outline Privacy Gate on Mixed Data Classes
# ---------------------------------------------------------------------------


def test_finding1_outline_generation_privacy_gate_mixed_classes(test_db, monkeypatch):
    """Outline generation checks all represented data classes before serialization or dispatch."""
    monkeypatch.setenv("EDWARD_SYNTHESIS_LOCATION", "hosted")
    # Allow personal notes, but keep gmail forbidden
    monkeypatch.setenv("EDWARD_HOSTED_PERSONAL_NOTES", "allow")
    monkeypatch.delenv("EDWARD_HOSTED_GMAIL", raising=False)

    gmail_cap = _capture_with_ns(test_db, "Private email text", origin_namespace="gmail")

    with test_db.transaction() as conn:
        project = create_project(conn, "Mixed Privacy Project", "Investigate email notes")
        add_project_object(conn, project.id, gmail_cap, membership_status="accepted")

    with test_db.connection() as conn:
        context = get_project_context(conn, project.id)

    fake_client = MagicMock()
    fake_client.provider = "typesafe"
    fake_client.location = "hosted"
    fake_client.base_url = "https://api.typesafe.com/v1"
    fake_client.model = "test-model"

    # Must raise PrivacyTransmissionError because gmail is not allowed
    with pytest.raises(PrivacyTransmissionError, match="gmail"):
        generate_outline_proposal(context, llm_client=fake_client)

    fake_client.chat_completion.assert_not_called()

    # Now allow gmail as well
    monkeypatch.setenv("EDWARD_HOSTED_GMAIL", "allow")
    proposal_obj = deterministic_outline(context)
    fake_client.chat_completion.return_value = ("{}", proposal_obj)

    result = generate_outline_proposal(context, llm_client=fake_client)
    assert result is not None
    assert fake_client.chat_completion.called
    called_data_class = fake_client.chat_completion.call_args[1]["data_class"]
    assert "gmail" in called_data_class
    assert "personal_notes" in called_data_class


@pytest.mark.parametrize("origin_ns", ["documents", "file", "attachment"])
def test_finding1_annotated_document_file_attachment_privacy_gate(test_db, monkeypatch, origin_ns):
    """Document, file, and attachment evidence with human notes cannot bypass the documents privacy gate."""
    monkeypatch.setenv("EDWARD_SYNTHESIS_LOCATION", "hosted")
    # Allow personal notes, but keep documents forbidden
    monkeypatch.setenv("EDWARD_HOSTED_PERSONAL_NOTES", "allow")
    monkeypatch.delenv("EDWARD_HOSTED_DOCUMENTS", raising=False)

    doc_cap = _capture_with_ns(
        test_db, f"Sensitive content from {origin_ns}", origin_namespace=origin_ns
    )

    # Attach human notes to the document
    with test_db.transaction() as conn:
        conn.execute(
            """
            INSERT INTO annotations (id, object_type, object_id, annotation_type, content, author, created_at)
            VALUES (?, 'capture', ?, 'note', 'My private user observation on doc', 'human', datetime('now'));
            """,
            (f"ann_{origin_ns}", doc_cap),
        )
        project = create_project(conn, f"Annotated {origin_ns} Project", "Research question")
        add_project_object(
            conn,
            project.id,
            doc_cap,
            membership_status="accepted",
            relevance_note="Important item",
        )

    with test_db.connection() as conn:
        context = get_project_context(conn, project.id)

    fake_client = MagicMock()
    fake_client.provider = "typesafe"
    fake_client.location = "hosted"
    fake_client.base_url = "https://api.typesafe.com/v1"
    fake_client.model = "test-model"

    # Even though personal notes are allowed, documents are forbidden; must raise PrivacyTransmissionError
    with pytest.raises(PrivacyTransmissionError, match="documents"):
        generate_outline_proposal(context, llm_client=fake_client)

    fake_client.chat_completion.assert_not_called()

    # When documents are also allowed, both classes are transmitted
    monkeypatch.setenv("EDWARD_HOSTED_DOCUMENTS", "allow")
    proposal_obj = deterministic_outline(context)
    fake_client.chat_completion.return_value = ("{}", proposal_obj)

    result = generate_outline_proposal(context, llm_client=fake_client)
    assert result is not None
    assert fake_client.chat_completion.called
    called_data_class = fake_client.chat_completion.call_args[1]["data_class"]
    assert "documents" in called_data_class
    assert "personal_notes" in called_data_class


# ---------------------------------------------------------------------------
# Finding 2: Resource Purge Preserves Findings and Project/Outline Links
# ---------------------------------------------------------------------------


def test_finding2_resource_purge_preserves_findings_and_project_links(test_db):
    """Purging a resource sets finding.resource_id = NULL but preserves the finding and its project links."""
    with test_db.transaction() as conn:
        conn.execute(
            """
            INSERT INTO resources (id, identity_key, canonical_url, title, review_state, is_deleted, created_at, updated_at)
            VALUES ('res_survive', 'url:survive', 'https://example.com/s', 'Survive Test', 'unreviewed', 0, datetime('now'), datetime('now'));
            """
        )
        conn.execute(
            """
            INSERT INTO findings (id, resource_id, statement, assertion_role, review_state, is_deleted, created_at, updated_at)
            VALUES ('fin_survive', 'res_survive', 'Durable finding statement', 'source-claim', 'unreviewed', 0, datetime('now'), datetime('now'));
            """
        )
        project = create_project(conn, "Preserve Links Project")
        add_project_object(conn, project.id, "fin_survive", membership_status="accepted")

    proposal = OutlineProposalInput.model_validate(
        {
            "title": "Preserve outline",
            "sections": [
                {
                    "heading": "Section 1",
                    "evidence": [{"object_id": "fin_survive", "relationship": "supporting"}],
                }
            ],
        }
    )
    with test_db.transaction() as conn:
        saved_outline = save_outline(conn, project.id, proposal)

    # Purge the parent resource
    with test_db.transaction() as conn:
        purge_object(conn, "resource", "res_survive")

    # Verify resource is gone, but finding survives with resource_id = NULL
    with test_db.connection() as conn:
        assert (
            conn.execute("SELECT COUNT(*) FROM resources WHERE id = 'res_survive';").fetchone()[0]
            == 0
        )
        finding_row = conn.execute(
            "SELECT id, resource_id, statement FROM findings WHERE id = 'fin_survive';"
        ).fetchone()
        assert finding_row is not None
        assert finding_row["resource_id"] is None

        # Verify project_objects membership survives
        po_count = conn.execute(
            "SELECT COUNT(*) FROM project_objects WHERE project_id = ? AND object_id = 'fin_survive';",
            (project.id,),
        ).fetchone()[0]
        assert po_count == 1

        # Verify outline section evidence survives
        outline = get_outline(conn, project.id, version=saved_outline["version"])
        assert outline is not None
        assert len(outline["sections"][0]["evidence"]) == 1
        assert outline["sections"][0]["evidence"][0]["object_id"] == "fin_survive"

    # Now purge the finding directly
    with test_db.transaction() as conn:
        purge_object(conn, "finding", "fin_survive")

    # Now finding and its polymorphic project references must be cleaned up
    with test_db.connection() as conn:
        assert (
            conn.execute("SELECT COUNT(*) FROM findings WHERE id = 'fin_survive';").fetchone()[0]
            == 0
        )
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM project_objects WHERE object_id = 'fin_survive';"
            ).fetchone()[0]
            == 0
        )
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM outline_section_evidence WHERE object_id = 'fin_survive';"
            ).fetchone()[0]
            == 0
        )


# ---------------------------------------------------------------------------
# Finding 3: Deterministic Answers Honor Project Scope
# ---------------------------------------------------------------------------


def test_finding3_deterministic_answers_scoped_to_project(test_db):
    """Deterministic counts, dates, and titles respect project scope and return None for non-members."""
    cap_a = _capture_with_ns(test_db, "Alpha text")
    cap_b = _capture_with_ns(test_db, "Beta text")
    cap_unrelated = _capture_with_ns(test_db, "Unrelated text")

    with test_db.transaction() as conn:
        conn.execute(
            """
            INSERT INTO resources (id, identity_key, canonical_url, title, review_state, is_deleted, created_at, updated_at)
            VALUES ('res_a', 'url:a', 'https://example.com/a', 'Title A', 'unreviewed', 0, datetime('now'), datetime('now')),
                   ('res_b', 'url:b', 'https://example.com/b', 'Title B', 'unreviewed', 0, datetime('now'), datetime('now'));
            """
        )
        conn.execute(
            """
            INSERT INTO findings (id, resource_id, statement, assertion_role, review_state, is_deleted, created_at, updated_at)
            VALUES ('fin_a1', 'res_a', 'Finding A1', 'source-claim', 'unreviewed', 0, datetime('now'), datetime('now')),
                   ('fin_a2', 'res_a', 'Finding A2', 'source-claim', 'unreviewed', 0, datetime('now'), datetime('now')),
                   ('fin_b1', 'res_b', 'Finding B1', 'source-claim', 'unreviewed', 0, datetime('now'), datetime('now'));
            """
        )
        prj_a = create_project(conn, "Project A")
        prj_b = create_project(conn, "Project B")

        add_project_object(conn, prj_a.id, cap_a)
        add_project_object(conn, prj_a.id, "res_a")
        add_project_object(conn, prj_a.id, "fin_a1")
        add_project_object(conn, prj_a.id, "fin_a2")

        add_project_object(conn, prj_b.id, cap_b)
        add_project_object(conn, prj_b.id, "res_b")
        add_project_object(conn, prj_b.id, "fin_b1")

    with test_db.connection() as conn:
        # Scoped count of findings
        ans_a = try_deterministic_answer(conn, "how many findings", project_id=prj_a.id)
        assert ans_a is not None
        assert ans_a["data"]["count"] == 2
        assert ans_a["data"]["project_id"] == prj_a.id
        assert "in this project" in ans_a["answer"]

        ans_b = try_deterministic_answer(conn, "how many findings", project_id=prj_b.id)
        assert ans_b is not None
        assert ans_b["data"]["count"] == 1

        # Global count of findings
        ans_global = try_deterministic_answer(conn, "how many findings")
        assert ans_global is not None
        assert ans_global["data"]["count"] == 3
        assert "in Edward" in ans_global["answer"]

        # Scoped count of captures
        ans_cap_a = try_deterministic_answer(conn, "how many captures", project_id=prj_a.id)
        assert ans_cap_a is not None
        assert ans_cap_a["data"]["count"] == 1

        # Scoped date lookup: allowed member vs denied non-member
        assert (
            try_deterministic_answer(conn, f"when was {cap_a} created", project_id=prj_a.id)
            is not None
        )
        assert (
            try_deterministic_answer(conn, f"when was {cap_b} created", project_id=prj_a.id) is None
        )
        assert (
            try_deterministic_answer(conn, f"when was {cap_unrelated} created", project_id=prj_a.id)
            is None
        )

        # Scoped title lookup: allowed member vs denied non-member
        assert (
            try_deterministic_answer(conn, "what is the title of res_a", project_id=prj_a.id)
            is not None
        )
        assert (
            try_deterministic_answer(conn, "what is the title of res_b", project_id=prj_a.id)
            is None
        )


# ---------------------------------------------------------------------------
# Finding 4: CLI Exit Code Contract (0, 1, 2, 3) across Phase 4 commands
# ---------------------------------------------------------------------------


def test_finding4_cli_exit_code_contract(test_db, cli_runner, monkeypatch):
    """Every Phase 4 CLI command conforms to exit codes: 0 success, 1 domain, 2 usage, 3 conflict/fatal."""
    from edward import cli

    monkeypatch.setattr(cli, "get_services", lambda: (test_db, None))

    # 1. Project create
    # Success (0)
    c0 = cli_runner.invoke(cli.app, ["project", "create", "--title", "CLI Code 0", "--json"])
    assert c0.exit_code == 0
    p_data = json.loads(c0.stdout)
    p_id = p_data["id"]

    # Conflict (3) - duplicate slug
    c3 = cli_runner.invoke(cli.app, ["project", "create", "--title", "CLI Code 0", "--json"])
    assert c3.exit_code == 3
    assert c3.stdout.strip() == ""
    err_json = json.loads(c3.stderr)
    assert "already exists" in err_json["error"]

    # Domain error (1) - empty title
    c1 = cli_runner.invoke(cli.app, ["project", "create", "--title", "   ", "--json"])
    assert c1.exit_code == 1
    assert json.loads(c1.stderr)["error"]

    # Fatal error (3) - injected fatal error
    def _fail_create(*args, **kwargs):
        raise RuntimeError("Database disk full")

    monkeypatch.setattr("edward.cli.create_project", _fail_create)
    cfatal = cli_runner.invoke(cli.app, ["project", "create", "--title", "Fatal Prj", "--json"])
    assert cfatal.exit_code == 3
    assert "Database disk full" in json.loads(cfatal.stderr)["error"]
    monkeypatch.undo()
    monkeypatch.setattr(cli, "get_services", lambda: (test_db, None))

    # 2. Project outline
    # Usage error (2) - multiple flags or no flags
    u2 = cli_runner.invoke(cli.app, ["project", "outline", p_id, "--propose", "--show", "--json"])
    assert u2.exit_code == 2

    u2_none = cli_runner.invoke(cli.app, ["project", "outline", p_id, "--json"])
    assert u2_none.exit_code == 2

    # Domain error (1) - non-existent project
    d1 = cli_runner.invoke(cli.app, ["project", "outline", "prj_nonexistent", "--show", "--json"])
    assert d1.exit_code == 1

    # Injected fatal error (3)
    def _fail_outline(*args, **kwargs):
        raise RuntimeError("Database connection corrupted")

    monkeypatch.setattr("edward.cli.get_outline", _fail_outline)
    o3 = cli_runner.invoke(cli.app, ["project", "outline", p_id, "--show", "--json"])
    assert o3.exit_code == 3
    assert "Database connection corrupted" in json.loads(o3.stderr)["error"]


def test_finding2_injected_get_services_failure_exit_contract(cli_runner, monkeypatch):
    """A fatal failure during get_services() emits structured JSON on stderr and exits with code 3."""
    from edward import cli

    def _broken_get_services(*args, **kwargs):
        raise RuntimeError("Migration runner failed: disk corrupted")

    monkeypatch.setattr(cli, "get_services", _broken_get_services)

    subcommands = [
        ["project", "list", "--json"],
        ["project", "create", "--title", "Fatal Project", "--json"],
        ["project", "add", "prj_test", "cap_test", "--json"],
        ["project", "note", "prj_test", "--text", "Test note", "--json"],
        ["project", "context", "prj_test", "--json"],
        ["project", "outline", "prj_test", "--show", "--json"],
    ]

    for cmd in subcommands:
        res = cli_runner.invoke(cli.app, cmd)
        assert res.exit_code == 3, f"Expected exit code 3 for {cmd}, got {res.exit_code}"
        assert res.stdout.strip() == "", f"Expected empty stdout on fatal error for {cmd}"
        err = json.loads(res.stderr)
        assert "Migration runner failed: disk corrupted" in err["error"]


def test_finding_privacy_policy_rejection_cli_exit_code_1(test_db, cli_runner, monkeypatch):
    """Privacy policy rejection in project outline emits structured JSON on stderr, exit code 1, and no model dispatch."""
    from edward import cli

    monkeypatch.setattr(cli, "get_services", lambda: (test_db, None))
    monkeypatch.setenv("EDWARD_SYNTHESIS_LOCATION", "hosted")
    # Allow personal notes so workspace brief/notes pass, but keep gmail forbidden
    monkeypatch.setenv("EDWARD_HOSTED_PERSONAL_NOTES", "allow")
    monkeypatch.delenv("EDWARD_HOSTED_GMAIL", raising=False)
    monkeypatch.delenv("EDWARD_HOSTED_DOCUMENTS", raising=False)

    # Create project and add Gmail evidence to it
    gmail_cap = _capture_with_ns(test_db, "Sensitive email content", origin_namespace="gmail")
    with test_db.transaction() as conn:
        project = create_project(conn, "Privacy Rejection CLI Project")
        add_project_object(conn, project.id, gmail_cap, membership_status="accepted")

    fake_client = MagicMock()
    fake_client.provider = "typesafe"
    fake_client.location = "hosted"
    fake_client.base_url = "https://api.typesafe.com/v1"
    fake_client.model = "test-model"

    monkeypatch.setattr("edward.services.llm.get_answer_client", lambda: fake_client)

    res = cli_runner.invoke(cli.app, ["project", "outline", project.id, "--propose", "--json"])
    assert res.exit_code == 1, (
        f"Expected exit code 1 for domain policy failure, got {res.exit_code}"
    )
    assert res.stdout.strip() == "", "Expected empty stdout on domain error under --json"
    err = json.loads(res.stderr)
    assert "gmail" in err["error"].lower()
    fake_client.chat_completion.assert_not_called()


def test_finding_privacy_policy_rejection_personal_notes_denied(test_db, cli_runner, monkeypatch):
    """When personal notes are forbidden on hosted provider, outline proposal exits code 1 with clean error."""
    from edward import cli

    monkeypatch.setattr(cli, "get_services", lambda: (test_db, None))
    monkeypatch.setenv("EDWARD_SYNTHESIS_LOCATION", "hosted")
    monkeypatch.delenv("EDWARD_HOSTED_PERSONAL_NOTES", raising=False)

    with test_db.transaction() as conn:
        project = create_project(conn, "Personal Notes Denied Project", "Private brief")

    fake_client = MagicMock()
    fake_client.provider = "typesafe"
    fake_client.location = "hosted"
    fake_client.base_url = "https://api.typesafe.com/v1"
    fake_client.model = "test-model"

    monkeypatch.setattr("edward.services.llm.get_answer_client", lambda: fake_client)

    res = cli_runner.invoke(cli.app, ["project", "outline", project.id, "--propose", "--json"])
    assert res.exit_code == 1
    assert res.stdout.strip() == ""
    err = json.loads(res.stderr)
    assert "personal_notes" in err["error"].lower()
    fake_client.chat_completion.assert_not_called()
