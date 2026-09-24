"""Tests verifying exact domain enum alignment with the Edward PRD specification."""

import json
from pathlib import Path
from typing import get_args

from edward.models import AssertionRole, ReviewState


def test_assertion_roles_match_prd():
    expected = (
        "source-claim",
        "direct-quotation",
        "agent-conclusion",
        "personal-belief",
        "personal-observation",
        "question",
        "hypothesis",
        "connection",
    )
    assert get_args(AssertionRole) == expected


def test_review_states_match_prd():
    expected = (
        "unreviewed",
        "reviewed",
        "approved",
        "corrected",
        "disputed",
        "dismissed",
        "superseded",
    )
    assert get_args(ReviewState) == expected


def test_schemas_enum_alignment():
    root = Path(__file__).parent.parent
    bundle_schema_path = root / "schemas" / "research-bundle-v1.json"
    evidence_schema_path = root / "schemas" / "evidence-packet-v1.json"

    expected_roles = list(get_args(AssertionRole))
    expected_reviews = list(get_args(ReviewState))

    if bundle_schema_path.exists():
        bundle_schema = json.loads(bundle_schema_path.read_text())
        finding_props = bundle_schema.get("$defs", {}).get("Finding", {}).get("properties", {})
        if "assertion_role" in finding_props and "enum" in finding_props["assertion_role"]:
            assert finding_props["assertion_role"]["enum"] == expected_roles
        if "review_state" in finding_props and "enum" in finding_props["review_state"]:
            assert finding_props["review_state"]["enum"] == expected_reviews

    if evidence_schema_path.exists():
        evidence_schema = json.loads(evidence_schema_path.read_text())
        item_props = evidence_schema.get("$defs", {}).get("EvidenceItem", {}).get("properties", {})
        if "review_state" in item_props and "enum" in item_props["review_state"]:
            assert item_props["review_state"]["enum"] == expected_reviews


def test_research_bundle_v1_properties_and_source_identity():
    import pytest
    from pydantic import ValidationError

    from edward.models import ResearchBundle, SourceItem

    root = Path(__file__).parent.parent
    bundle_schema_path = root / "schemas" / "research-bundle-v1.json"
    assert bundle_schema_path.exists()
    bundle_schema = json.loads(bundle_schema_path.read_text())

    # Verify JSON Schema requirements
    assert "bundle_id" in bundle_schema["required"]
    assert "bundle_id" in bundle_schema["properties"]
    assert "idempotency_key" in bundle_schema["properties"]
    source_items = bundle_schema["properties"]["sources"]["items"]
    assert "anyOf" in source_items
    any_of_required = [cond["required"] for cond in source_items["anyOf"]]
    assert ["identity_key"] in any_of_required
    assert ["url"] in any_of_required
    assert ["source_id"] in any_of_required

    # Verify Pydantic ResearchBundle model
    with pytest.raises(ValidationError):
        # Missing bundle_id
        ResearchBundle(title="Test", sources=[], findings=[])  # type: ignore

    bundle = ResearchBundle(
        bundle_id="bundle-123",
        idempotency_key="idemp-bundle-123",
        title="Valid Bundle",
        sources=[SourceItem(origin="web", url="https://example.com")],
        findings=[],
    )
    assert bundle.bundle_id == "bundle-123"
    assert bundle.idempotency_key == "idemp-bundle-123"

    # Verify SourceItem with URL only
    s_url = SourceItem(origin="web", url="https://example.com")
    assert s_url.url == "https://example.com"
    assert s_url.source_id is None
    assert s_url.identity_key == "url:https://example.com"

    # Verify SourceItem with source_id only (URL-less)
    s_src = SourceItem(origin="local", source_id="res_local_123")
    assert s_src.source_id == "res_local_123"
    assert s_src.url is None
    assert s_src.identity_key == "res_local_123"

    # Verify SourceItem with identity_key only (URL-less)
    s_ident = SourceItem(origin="blob", identity_key="blob:abcdef123456")
    assert s_ident.identity_key == "blob:abcdef123456"
    assert s_ident.url is None

    # Verify SourceItem with none of the three raises ValidationError
    with pytest.raises(ValidationError, match="SourceItem must specify at least one of"):
        SourceItem(origin="web")
