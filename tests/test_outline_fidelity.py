"""Regression tests for outline proposal schema fidelity.

Bug A: the compact evidence payload sent to an outline model used the key ``id``
and passed the stored membership ``relationship`` through verbatim. A model that
echoes the supplied evidence shape therefore produced ``object_id``-less evidence
with an invalid ``relationship`` literal, so every model-generated outline failed
validation against the very schema it was asked to fill.

These tests do not use a canned response. They emulate what a real model does --
read the evidence in the prompt and echo that shape back -- which is why the
defect escaped the earlier suite.
"""

import json
from unittest.mock import MagicMock

from edward.models import CaptureInput, OutlineProposalInput
from edward.services.capture import capture_item
from edward.services.projects import (
    VALID_EVIDENCE_RELATIONSHIPS,
    add_project_object,
    create_project,
    generate_outline_proposal,
    get_project_context,
)


def _echoing_client() -> MagicMock:
    """A client that builds its proposal by echoing the evidence it was given.

    This is the realistic failure mode: a model mirrors the payload's key names and
    relationship values instead of inventing the schema's own vocabulary.
    """
    client = MagicMock()
    client.provider = "local"
    client.location = "local"
    client.base_url = "http://localhost:11434/v1"
    client.model = "echo-model"

    def _chat_completion(messages, response_model, **kwargs):
        payload = json.loads(messages[-1]["content"])
        evidence = payload.get("evidence", [])
        # Echo each evidence item exactly as supplied, as a model would.
        sections = [
            {
                "heading": "Section from evidence",
                "purpose": "Exercise the schema boundary.",
                "evidence": [
                    {
                        "object_id": item["object_id"],
                        "relationship": item["relationship"],
                        "relevance_note": item.get("relevance_note"),
                    }
                    for item in evidence
                ],
            }
        ]
        return "{}", response_model.model_validate(
            {"title": payload["project"]["title"], "sections": sections}
        )

    client.chat_completion.side_effect = _chat_completion
    return client


def _project_with_evidence(db, text: str = "Evidence body text") -> tuple[str, str]:
    """Create a project holding one accepted evidence capture. Returns (project_id, capture_id)."""
    with db.transaction() as conn:
        captured = capture_item(
            conn,
            CaptureInput(
                text=text,
                origin_namespace="manual",
                collection_channel="cli",
                collector="test",
                acquisition_method="manual",
            ),
        )
        project = create_project(conn, "Outline Fidelity Project", "Test the outline boundary")
        add_project_object(
            conn,
            project.id,
            captured["capture_id"],
            membership_status="accepted",
            relevance_note="Chosen as supporting evidence",
        )
    return project.id, captured["capture_id"]


def test_outline_payload_uses_object_id_not_id(test_db):
    """The evidence sent to a model must use the schema's ``object_id`` field name."""
    project_id, capture_id = _project_with_evidence(test_db)
    with test_db.connection() as conn:
        context = get_project_context(conn, project_id)

    client = _echoing_client()
    generate_outline_proposal(context, llm_client=client)

    payload = json.loads(client.chat_completion.call_args[1]["messages"][-1]["content"])
    assert payload["evidence"], "expected evidence to be supplied to the model"
    for item in payload["evidence"]:
        assert "object_id" in item, "model payload must name the field object_id"
        assert "id" not in item, "bare `id` is echoed back as an invalid evidence link"
        assert item["object_id"] == capture_id


def test_outline_payload_relationship_is_a_valid_evidence_literal(test_db):
    """Stored membership relationships must be mapped to outline evidence literals."""
    project_id, _ = _project_with_evidence(test_db)
    with test_db.connection() as conn:
        context = get_project_context(conn, project_id)

    client = _echoing_client()
    generate_outline_proposal(context, llm_client=client)

    payload = json.loads(client.chat_completion.call_args[1]["messages"][-1]["content"])
    for item in payload["evidence"]:
        assert item["relationship"] in VALID_EVIDENCE_RELATIONSHIPS, (
            f"relationship {item['relationship']!r} is not a valid outline evidence literal"
        )


def test_default_membership_relationship_round_trips_through_a_model(test_db):
    """The default membership relationship ('evidence') must not reach the model verbatim.

    This is the exact production failure: project add defaults to 'evidence', which is
    not in the outline evidence vocabulary, so the echoed proposal failed validation.
    """
    project_id, _ = _project_with_evidence(test_db)
    with test_db.connection() as conn:
        context = get_project_context(conn, project_id)
        assert context["memberships"][0]["relationship"] == "evidence"

    client = _echoing_client()
    proposal = generate_outline_proposal(context, llm_client=client)

    assert isinstance(proposal, OutlineProposalInput)
    assert proposal.sections, "a proposal must survive the round trip"
    for section in proposal.sections:
        for link in section.evidence:
            assert link.relationship in VALID_EVIDENCE_RELATIONSHIPS


def test_counterargument_membership_maps_to_counterevidence(test_db):
    """A counterargument membership should surface as counterevidence, not 'counterargument'."""
    with test_db.transaction() as conn:
        captured = capture_item(
            conn,
            CaptureInput(
                text="Opposing evidence body",
                origin_namespace="manual",
                collection_channel="cli",
                collector="test",
                acquisition_method="manual",
            ),
        )
        project = create_project(conn, "Counter View Project", "Weigh the counter view")
        add_project_object(
            conn,
            project.id,
            captured["capture_id"],
            relationship="counterargument",
            membership_status="accepted",
            relevance_note="Directly opposes the premise",
        )

    with test_db.connection() as conn:
        context = get_project_context(conn, project.id)

    client = _echoing_client()
    generate_outline_proposal(context, llm_client=client)

    payload = json.loads(client.chat_completion.call_args[1]["messages"][-1]["content"])
    assert payload["evidence"][0]["relationship"] == "counterevidence"


def test_outline_proposal_succeeds_for_every_valid_membership_relationship(test_db):
    """No stored membership relationship may produce a proposal that fails validation."""
    from edward.services.projects import VALID_RELATIONSHIPS

    for membership_relationship in sorted(VALID_RELATIONSHIPS):
        with test_db.transaction() as conn:
            captured = capture_item(
                conn,
                CaptureInput(
                    text=f"Body for {membership_relationship}",
                    origin_namespace="manual",
                    collection_channel="cli",
                    collector="test",
                    acquisition_method="manual",
                ),
            )
            project = create_project(
                conn, f"Relationship {membership_relationship}", "Boundary check"
            )
            add_project_object(
                conn,
                project.id,
                captured["capture_id"],
                relationship=membership_relationship,
                membership_status="accepted",
            )

        with test_db.connection() as conn:
            context = get_project_context(conn, project.id)

        client = _echoing_client()
        proposal = generate_outline_proposal(context, llm_client=client)

        assert isinstance(proposal, OutlineProposalInput), (
            f"membership relationship {membership_relationship!r} broke outline validation"
        )
        for section in proposal.sections:
            for link in section.evidence:
                assert link.relationship in VALID_EVIDENCE_RELATIONSHIPS
