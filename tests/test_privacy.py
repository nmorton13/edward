"""Tests for privacy enforcement and transmission policies."""

import pytest

from edward.classifiers.providers.openrouter import OpenRouterProvider
from edward.classifiers.providers.typesafe import TypeSafeProvider
from edward.services.privacy import (
    PrivacyTransmissionError,
    assert_transmission_permitted,
    classify_content_data_class,
    is_transmission_allowed,
    resolve_provider_location,
)


def test_forced_hosted_providers():
    """Known hosted providers ('typesafe', 'openrouter') are ALWAYS forced to hosted."""
    assert resolve_provider_location("typesafe") == "hosted"
    assert resolve_provider_location("typesafe", declared_location="local") == "hosted"
    assert resolve_provider_location("openrouter") == "hosted"
    assert resolve_provider_location("openrouter", declared_location="local") == "hosted"

    assert TypeSafeProvider.location == "hosted"
    assert OpenRouterProvider.location == "hosted"


def test_local_providers_resolved_to_local():
    """Local and dry-run providers resolve to local."""
    assert resolve_provider_location("dry-run") == "local"
    assert resolve_provider_location("local") == "local"
    assert resolve_provider_location("disabled") == "local"


def test_classify_content_data_class():
    """Classify content data class based on origin, url, and metadata."""
    assert classify_content_data_class(origin_namespace="gmail") == "gmail"
    assert classify_content_data_class(origin_namespace="file") == "documents"
    assert classify_content_data_class(origin_namespace="attachment") == "documents"
    assert classify_content_data_class(origin_namespace="manual") == "personal_notes"
    assert classify_content_data_class(form="personal-note") == "personal_notes"
    assert classify_content_data_class(canonical_url="https://example.com/blog") == "public_web"
    assert classify_content_data_class(origin_namespace="arxiv") == "public_web"
    assert classify_content_data_class(origin_namespace="x") == "public_web"


def test_default_privacy_policies(monkeypatch: pytest.MonkeyPatch):
    """By default, gmail, personal_notes, and documents are denied to hosted providers."""
    monkeypatch.delenv("EDWARD_HOSTED_GMAIL", raising=False)
    monkeypatch.delenv("EDWARD_HOSTED_PERSONAL_NOTES", raising=False)
    monkeypatch.delenv("EDWARD_HOSTED_DOCUMENTS", raising=False)
    monkeypatch.delenv("EDWARD_HOSTED_PUBLIC_WEB", raising=False)

    # Local is always allowed
    assert is_transmission_allowed("gmail", "local") is True
    assert is_transmission_allowed("personal_notes", "local") is True
    assert is_transmission_allowed("documents", "local") is True
    assert is_transmission_allowed("public_web", "local") is True

    # Hosted: private classes denied by default
    assert is_transmission_allowed("gmail", "hosted") is False
    assert is_transmission_allowed("personal_notes", "hosted") is False
    assert is_transmission_allowed("documents", "hosted") is False
    # Public web is allowed by default
    assert is_transmission_allowed("public_web", "hosted") is True


def test_explicit_privacy_policies_allow(monkeypatch: pytest.MonkeyPatch):
    """Enabling EDWARD_HOSTED_* permits transmission."""
    monkeypatch.setenv("EDWARD_HOSTED_GMAIL", "allow")
    monkeypatch.setenv("EDWARD_HOSTED_PERSONAL_NOTES", "allow")
    monkeypatch.setenv("EDWARD_HOSTED_DOCUMENTS", "allow")

    assert is_transmission_allowed("gmail", "hosted") is True
    assert is_transmission_allowed("personal_notes", "hosted") is True
    assert is_transmission_allowed("documents", "hosted") is True


def test_assert_transmission_permitted_halts_and_raises(monkeypatch: pytest.MonkeyPatch):
    """assert_transmission_permitted raises PrivacyTransmissionError before any dispatch."""
    monkeypatch.delenv("EDWARD_HOSTED_GMAIL", raising=False)

    with pytest.raises(PrivacyTransmissionError) as exc_info:
        assert_transmission_permitted("typesafe", data_class="gmail")

    assert "Privacy policy violation" in str(exc_info.value)
    assert exc_info.value.data_class == "gmail"
    assert exc_info.value.provider == "typesafe"


def test_resolve_provider_location_declared_locations():
    """Declared locations map to local or hosted correctly."""
    assert resolve_provider_location("custom", declared_location="hosted") == "hosted"
    assert resolve_provider_location("custom", declared_location="remote") == "hosted"
    assert resolve_provider_location("custom", declared_location="local") == "local"
    assert resolve_provider_location("custom", declared_location="on-premise") == "local"
    assert resolve_provider_location("custom", declared_location="on_premise") == "local"
    # Unknown provider with no declared location or URL defaults to hosted
    assert resolve_provider_location("custom-unknown") == "hosted"


def test_resolve_provider_location_url_loopback():
    """Localhost, subdomains of localhost, and valid loopback IPs resolve to local."""
    assert resolve_provider_location("custom", base_url="http://localhost:11434/v1") == "local"
    assert resolve_provider_location("custom", base_url="http://sub.localhost:8080") == "local"
    assert resolve_provider_location("custom", base_url="http://127.0.0.1:8080") == "local"
    assert resolve_provider_location("custom", base_url="http://127.0.0.2:8080") == "local"
    # Valid RFC 3986 bracketed IPv6 loopback
    assert resolve_provider_location("custom", base_url="http://[::1]:8080") == "local"
    assert resolve_provider_location("custom", base_url="http://[::1]") == "local"
    assert resolve_provider_location("custom", base_url="localhost:11434") == "local"


def test_resolve_provider_location_adversarial_hosted_urls():
    """Deceptive, external, or malformed URLs must resolve to hosted, not local."""
    # Deceptive domains containing 'localhost' or '127.0.0.1' as subdomains/paths
    assert (
        resolve_provider_location("custom", base_url="https://localhost.evil.example") == "hosted"
    )
    assert (
        resolve_provider_location("custom", base_url="https://evil.example/?redirect=localhost")
        == "hosted"
    )
    assert (
        resolve_provider_location("custom", base_url="https://127.0.0.1.evil.example") == "hosted"
    )
    assert (
        resolve_provider_location("custom", base_url="https://evil.example/#localhost") == "hosted"
    )
    assert resolve_provider_location("custom", base_url="https://notlocalhost.com") == "hosted"

    # Private but non-loopback IP
    assert resolve_provider_location("custom", base_url="http://192.168.1.1:8000") == "hosted"
    assert resolve_provider_location("custom", base_url="http://10.0.0.1:8000") == "hosted"

    # Public hosted endpoints
    assert resolve_provider_location("custom", base_url="https://api.openai.com/v1") == "hosted"

    # Malformed or unbracketed IPv6 (http://::1:8080 is invalid IPv6 syntax)
    assert resolve_provider_location("custom", base_url="http://::1:8080") == "hosted"
    assert resolve_provider_location("custom", base_url="") == "hosted"


def test_classify_content_data_class_file_and_metadata():
    """File URLs and metadata source_types are classified as documents."""
    assert classify_content_data_class(canonical_url="file:///home/user/doc.pdf") == "documents"
    assert classify_content_data_class(canonical_url="/home/user/doc.pdf") == "documents"
    assert classify_content_data_class(metadata={"source_type": "gmail"}) == "gmail"
    assert classify_content_data_class(metadata={"source_type": "document"}) == "documents"
    for origin in ("web", "arxiv", "x", "github", "gitlab", "public"):
        assert classify_content_data_class(origin_namespace=origin) == "public_web"


def test_assert_transmission_denied_for_documents(monkeypatch: pytest.MonkeyPatch):
    """Documents cannot be transmitted to hosted providers without explicit permission."""
    monkeypatch.delenv("EDWARD_HOSTED_DOCUMENTS", raising=False)
    with pytest.raises(PrivacyTransmissionError, match="documents"):
        assert_transmission_permitted(
            provider_name="openrouter",
            data_class="documents",
        )


def test_privacy_transmission_error_reason():
    """PrivacyTransmissionError formats reason and attributes cleanly."""
    err = PrivacyTransmissionError("gmail", "openrouter", reason="Test policy reason")
    assert "Test policy reason" in str(err)
    assert err.data_class == "gmail"
    assert err.provider == "openrouter"
