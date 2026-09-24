"""Privacy enforcement service for external model and classifier transmissions.

Enforces strict boundaries before dispatching content to hosted providers.
Known hosted providers ('typesafe', 'openrouter') are always forced to location = 'hosted'.
"""

import ipaddress
import os
import urllib.parse
from typing import Any, Literal

DataClass = Literal["gmail", "personal_notes", "documents", "public_web"]
ProviderLocation = Literal["local", "hosted"]

FORCED_HOSTED_PROVIDERS: frozenset[str] = frozenset({"typesafe", "openrouter"})
LOCAL_PROVIDERS: frozenset[str] = frozenset({"local", "dry-run", "dry_run", "disabled", "none"})


class PrivacyTransmissionError(Exception):
    """Raised when privacy policy forbids transmitting content to a hosted provider."""

    def __init__(self, data_class: str, provider: str, reason: str | None = None):
        msg = (
            f"Privacy policy violation: transmission of data class '{data_class}' "
            f"to hosted provider '{provider}' is denied."
        )
        if reason:
            msg += f" Reason: {reason}"
        super().__init__(msg)
        self.data_class = data_class
        self.provider = provider


def resolve_provider_location(
    provider_name: str,
    declared_location: str | None = None,
    base_url: str | None = None,
) -> ProviderLocation:
    """Determine the effective location ('local' or 'hosted') for a provider.

    Invariant: 'typesafe' and 'openrouter' are always forced to 'hosted' in code,
    regardless of declared_location or environment variables.
    """
    prov_lower = provider_name.strip().lower()

    if prov_lower in FORCED_HOSTED_PROVIDERS:
        return "hosted"

    if prov_lower in LOCAL_PROVIDERS:
        return "local"

    if declared_location:
        loc = declared_location.strip().lower()
        if loc in ("hosted", "remote"):
            return "hosted"
        if loc in ("local", "on-premise", "on_premise"):
            return "local"

    # Infer from base_url if available
    if base_url:
        try:
            url_to_parse = base_url if "://" in base_url else f"//{base_url}"
            parsed = urllib.parse.urlsplit(url_to_parse)
            hostname = parsed.hostname
            if hostname:
                hostname = hostname.lower()
                if hostname == "localhost" or hostname.endswith(".localhost"):
                    return "local"
                try:
                    ip_obj = ipaddress.ip_address(hostname)
                    if ip_obj.is_loopback:
                        return "local"
                except ValueError:
                    pass
        except Exception:
            return "hosted"
        return "hosted"

    return "hosted"


def classify_content_data_class(
    origin_namespace: str | None = None,
    canonical_url: str | None = None,
    form: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> DataClass:
    """Classify the privacy data class of a piece of content."""
    meta = metadata or {}
    origin = (origin_namespace or meta.get("origin_namespace") or meta.get("origin") or "").lower()

    if origin == "gmail" or meta.get("source_type") == "gmail":
        return "gmail"

    if (
        origin in ("file", "attachment", "document", "documents")
        or origin.startswith("file:")
        or meta.get("source_type") in ("file", "attachment", "document")
    ):
        return "documents"

    if origin in ("manual", "note", "personal") or form == "personal-note":
        return "personal_notes"

    # Check canonical URL
    if canonical_url:
        u_lower = canonical_url.lower()
        if u_lower.startswith(("http://", "https://")):
            return "public_web"
        if u_lower.startswith(("file://", "/")):
            return "documents"

    # Public web origins
    if origin in ("web", "arxiv", "x", "github", "gitlab", "public"):
        return "public_web"

    # Default to personal_notes for safety if uncertain
    return "personal_notes"


def is_transmission_allowed(
    data_class: DataClass,
    provider_location: ProviderLocation,
) -> bool:
    """Check if transmission of the given data class is permitted to the provider location."""
    if provider_location == "local":
        return True

    # Provider is hosted: check explicit environment policies
    if data_class == "gmail":
        return os.environ.get("EDWARD_HOSTED_GMAIL", "deny").strip().lower() == "allow"
    elif data_class == "personal_notes":
        return os.environ.get("EDWARD_HOSTED_PERSONAL_NOTES", "deny").strip().lower() == "allow"
    elif data_class == "documents":
        return os.environ.get("EDWARD_HOSTED_DOCUMENTS", "deny").strip().lower() == "allow"
    elif data_class == "public_web":
        return os.environ.get("EDWARD_HOSTED_PUBLIC_WEB", "allow").strip().lower() == "allow"

    return False


def assert_transmission_permitted(
    provider_name: str,
    data_class: DataClass,
    declared_location: str | None = None,
    base_url: str | None = None,
) -> None:
    """Assert that transmission is permitted; raise PrivacyTransmissionError if denied.

    Must be invoked BEFORE payload serialization or network dispatch.
    """
    location = resolve_provider_location(
        provider_name=provider_name,
        declared_location=declared_location,
        base_url=base_url,
    )

    if not is_transmission_allowed(data_class, location):
        env_var_map = {
            "gmail": "EDWARD_HOSTED_GMAIL=allow",
            "personal_notes": "EDWARD_HOSTED_PERSONAL_NOTES=allow",
            "documents": "EDWARD_HOSTED_DOCUMENTS=allow",
            "public_web": "EDWARD_HOSTED_PUBLIC_WEB=allow",
        }
        env_hint = env_var_map.get(data_class, "")
        raise PrivacyTransmissionError(
            data_class=data_class,
            provider=provider_name,
            reason=f"Requires {env_hint} in environment to permit transmission.",
        )
