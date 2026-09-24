"""Shortlink resolution.

X stores every outgoing link as a ``t.co`` shortlink, and the snapshot Edward
retains for such a URL is X's own interstitial — the sign-in shell, trending
list, and cookie banner. Extracting from that snapshot stored page chrome as the
resource's body, which is why the archive's largest entries were its least
useful ones.

Resolving the shortlink before extraction is what recovers the actual article.
Resolution lives here rather than in the extraction pipeline so that the
backfill path and the ingest path share one implementation.

The fetch is SSRF-guarded per hop by ``safe_fetch_url``, so a redirect chain
cannot walk a request somewhere unsafe.
"""

from __future__ import annotations

import logging
from urllib.parse import urlparse

from edward.services.network import safe_fetch_url

logger = logging.getLogger(__name__)

# Hosts that serve redirects rather than content. Kept narrow on purpose: a
# URL that merely looks short is not evidence that its snapshot is useless.
SHORTLINK_HOSTS = frozenset({"t.co", "bit.ly", "buff.ly", "ow.ly", "lnkd.in", "dlvr.it"})

# How long to wait for a redirect chain before giving up on a link.
RESOLVE_TIMEOUT_SECONDS = 15.0

# Markers of a shortener's interstitial page rather than the destination. These
# appear in the snapshot X serves for a t.co link: the sign-in shell, trending
# list, and cookie banner, but not the destination's text.
_INTERSTITIAL_MARKERS = (
    "log in or sign up for x",
    "sign in to x",
    "trending now",
    "continue with apple",
    "continue with google",
    "continue with phone",
    "email or username",
)


def looks_like_shortlink_interstitial(text: str | None) -> bool:
    """True when stored text is a shortener's page shell, not the destination.

    This is the gate for re-extracting a shortlink. Without it, resolution is
    attempted for every shortlink, which both wastes a fetch on links whose
    snapshot is already the real article and risks replacing good content with
    worse. Snapshot chrome is the actual defect, so it is the thing to test for.
    """
    if not text:
        return False
    lowered = text.lower()
    return sum(marker in lowered for marker in _INTERSTITIAL_MARKERS) >= 2


def is_shortlink(url: str | None) -> bool:
    """True when a URL is served by a known shortener host."""
    if not url:
        return False
    try:
        hostname = urlparse(url).hostname
    except ValueError:
        return False
    return bool(hostname) and hostname.lower() in SHORTLINK_HOSTS


def resolve_shortlink(url: str, timeout: float = RESOLVE_TIMEOUT_SECONDS) -> str | None:
    """Follow a shortlink and return its final destination URL.

    Returns None when the link cannot be resolved, which includes a fetch
    failure and a chain that ends where it started. Callers treat None as "no
    better source available" rather than as an error, because an unresolvable
    link is a normal outcome for a deleted or private destination.
    """
    if not is_shortlink(url):
        return None
    try:
        fetched = safe_fetch_url(url, timeout=timeout)
    except Exception as exc:  # network, SSRF, size, or timeout — all non-fatal
        logger.debug("shortlink resolution failed for %s: %s", url, exc)
        return None
    final_url = getattr(fetched, "final_url", None)
    if not isinstance(final_url, str):
        return None
    final_url = final_url.strip()
    if not final_url or final_url == url:
        return None
    return final_url
