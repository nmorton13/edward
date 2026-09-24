"""Tests verifying exact symmetry between schemas/research-bundle-v1.json and Pydantic ResearchBundle."""

import json
from pathlib import Path

import jsonschema
import pytest
from pydantic import ValidationError

from edward.models import ResearchBundle


@pytest.fixture
def bundle_schema():
    schema_path = Path(__file__).parent.parent / "schemas" / "research-bundle-v1.json"
    assert schema_path.exists()
    return json.loads(schema_path.read_text(encoding="utf-8"))


def test_valid_bundle_with_url_source(bundle_schema):
    payload = {
        "type": "research-bundle",
        "schema_version": "1",
        "bundle_id": "bun_01j7xyz",
        "idempotency_key": "idemp-001",
        "title": "Quantum Error Mitigation",
        "brief": "Summary of 2026 quantum mitigation techniques",
        "agent": {"name": "research-agent", "model": "claude-3-7-sonnet"},
        "sources": [
            {
                "origin": "web",
                "url": "https://example.com/quantum",
                "title": "Quantum Mitigation Review",
            }
        ],
        "findings": [
            {
                "statement": "Zero-noise extrapolation achieves 90% error reduction.",
                "assertion_role": "source-claim",
                "source_url": "https://example.com/quantum",
            }
        ],
    }
    # Must pass both JSON Schema and Pydantic
    jsonschema.validate(payload, bundle_schema)
    bundle = ResearchBundle.model_validate(payload)
    assert bundle.bundle_id == "bun_01j7xyz"
    assert bundle.sources[0].url == "https://example.com/quantum"
    assert bundle.sources[0].identity_key == "url:https://example.com/quantum"


def test_valid_bundle_with_identity_key_source(bundle_schema):
    payload = {
        "type": "research-bundle",
        "schema_version": "1",
        "bundle_id": "bun_01j7abc",
        "title": "Local Blob Analysis",
        "sources": [
            {
                "origin": "blob",
                "identity_key": "blob:abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890",
            }
        ],
    }
    jsonschema.validate(payload, bundle_schema)
    bundle = ResearchBundle.model_validate(payload)
    assert bundle.sources[0].identity_key.startswith("blob:")


def test_valid_bundle_with_source_id(bundle_schema):
    payload = {
        "type": "research-bundle",
        "schema_version": "1",
        "bundle_id": "bun_01j7def",
        "title": "Existing Resource Reference",
        "sources": [
            {
                "origin": "local",
                "source_id": "source-id-1",
            }
        ],
    }
    jsonschema.validate(payload, bundle_schema)
    bundle = ResearchBundle.model_validate(payload)
    assert bundle.sources[0].source_id == "source-id-1"
    assert bundle.sources[0].identity_key == "source-id-1"


def test_origin_and_origin_id_accepted_by_both(bundle_schema):
    """SourceItem with origin and origin_id is accepted by BOTH contracts (Finding 7)."""
    payload = {
        "type": "research-bundle",
        "schema_version": "1",
        "bundle_id": "bun_origin_id",
        "title": "Origin ID Source Identity",
        "sources": [
            {
                "origin": "arxiv",
                "origin_id": "2401.00000",
            }
        ],
    }
    # Passes JSON Schema
    jsonschema.validate(payload, bundle_schema)

    # Passes Pydantic and derives identity_key
    bundle = ResearchBundle.model_validate(payload)
    assert bundle.sources[0].identity_key == "arxiv:2401.00000"


def test_source_without_identity_rejected_by_both(bundle_schema):
    """SourceItem with no identity fields must fail BOTH contracts."""
    payload = {
        "type": "research-bundle",
        "schema_version": "1",
        "bundle_id": "bun_invalid_no_id",
        "title": "Invalid Source Without ID",
        "sources": [
            {
                "origin": "arxiv",
            }
        ],
    }
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(payload, bundle_schema)
    with pytest.raises(ValidationError):
        ResearchBundle.model_validate(payload)


def test_empty_sources_rejected_by_both(bundle_schema):
    """Empty sources list must fail BOTH contracts."""
    payload = {
        "type": "research-bundle",
        "schema_version": "1",
        "bundle_id": "bun_no_sources",
        "title": "Bundle without sources",
        "sources": [],
    }
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(payload, bundle_schema)

    with pytest.raises(ValidationError):
        ResearchBundle.model_validate(payload)


def test_empty_bundle_id_rejected_by_both(bundle_schema):
    """Empty string bundle_id must fail BOTH contracts."""
    payload = {
        "type": "research-bundle",
        "schema_version": "1",
        "bundle_id": "",
        "title": "Bundle with empty ID",
        "sources": [{"origin": "web", "url": "https://example.com"}],
    }
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(payload, bundle_schema)

    with pytest.raises(ValidationError):
        ResearchBundle.model_validate(payload)


def test_empty_title_rejected_by_both(bundle_schema):
    """Empty string title must fail BOTH contracts."""
    payload = {
        "type": "research-bundle",
        "schema_version": "1",
        "bundle_id": "bun_valid",
        "title": "",
        "sources": [{"origin": "web", "url": "https://example.com"}],
    }
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(payload, bundle_schema)

    with pytest.raises(ValidationError):
        ResearchBundle.model_validate(payload)


def test_empty_finding_statement_rejected_by_both(bundle_schema):
    """Empty string finding statement must fail BOTH contracts."""
    payload = {
        "type": "research-bundle",
        "schema_version": "1",
        "bundle_id": "bun_valid",
        "title": "Valid Title",
        "sources": [{"origin": "web", "url": "https://example.com"}],
        "findings": [{"statement": ""}],
    }
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(payload, bundle_schema)

    with pytest.raises(ValidationError):
        ResearchBundle.model_validate(payload)


def test_invalid_assertion_role_rejected_by_both(bundle_schema):
    """Unknown assertion_role must fail BOTH contracts."""
    payload = {
        "type": "research-bundle",
        "schema_version": "1",
        "bundle_id": "bun_valid",
        "title": "Valid Title",
        "sources": [{"origin": "web", "url": "https://example.com"}],
        "findings": [{"statement": "Some statement", "assertion_role": "invalid-role"}],
    }
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(payload, bundle_schema)

    with pytest.raises(ValidationError):
        ResearchBundle.model_validate(payload)


def test_extra_bundle_property_rejected_by_both(bundle_schema):
    """Undeclared root property must fail BOTH contracts (additionalProperties: false)."""
    payload = {
        "type": "research-bundle",
        "schema_version": "1",
        "bundle_id": "bun_valid",
        "title": "Valid Title",
        "sources": [{"origin": "web", "url": "https://example.com"}],
        "unsupported_extra_field": 123,
    }
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(payload, bundle_schema)

    with pytest.raises(ValidationError):
        ResearchBundle.model_validate(payload)
