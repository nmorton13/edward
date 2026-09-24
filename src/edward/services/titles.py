"""Human-readable titles for source resources.

A resource's title is the label every downstream surface uses: `search` results,
`ask` evidence items, `export` packets, and project candidate lists. For an X
bookmark the post text IS the source, so the title comes from that text — not
from the author. The author already has its own column on the resource, and
copying it into the title makes every retrieved item read as a bare person's name
("Andrew Ng") instead of the thing that was saved ("AI Risk Hype and Reality").

This module is pure: no database, no network. Callers own persistence.
"""

from __future__ import annotations

import re
import unicodedata

# A line that is not usable as a title even though it is prose-length.
_JUNK_LINE = re.compile(
    r"^("
    r"skip to|select |what to read|most popular|most p|menu|sign in|subscribe|"
    r"advertisement|cookie|share|related|recommended|trending"
    r")",
    re.IGNORECASE,
)
_BARE_HANDLE = re.compile(r"^@[\w.]+$")
_NUMERIC = re.compile(r"^\d{6,}$")
_MARKDOWN_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_URL = re.compile(r"https?://\S+")
_LIST_MARKER = re.compile(r"^[-*\u2022\u00b7]\s+")

MIN_TITLE_CHARS = 18
MAX_TITLE_CHARS = 78


def _strip_list_marker(line: str) -> str:
    return _LIST_MARKER.sub("", (line or "").strip()).strip()


def _tidy(line: str) -> str:
    """Collapse a line into a single-space string with links and URLs removed."""
    line = _strip_list_marker(line)
    line = _MARKDOWN_LINK.sub(r"\1", line)
    line = _URL.sub("", line)
    line = re.sub(r"\s+", " ", line)
    return line.strip(" -\u2014\u00b7:,")


def shorten(text: str, limit: int = MAX_TITLE_CHARS) -> str:
    """Trim to a word boundary and mark that text was cut."""
    text = re.sub(r"\s+", " ", (text or "")).strip()
    if len(text) <= limit:
        return text
    cut = text[:limit]
    if " " in cut:
        cut = cut[: cut.rfind(" ")]
    return cut.rstrip(" .,;:-\u2014") + "\u2026"


def normalize_identity(value: str | None) -> str:
    """Compare author labels: fold diacritics, drop punctuation and case.

    'Cr\u00e9mieux' and 'cremieux' normalize alike; a handle's punctuation and a
    display name's spacing both stop mattering. Diacritics are folded rather than
    dropped so a name does not silently lose a letter ('Cr\u00e9mieux' must not
    become 'crmieux').
    """
    decomposed = unicodedata.normalize("NFKD", value or "")
    without_marks = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9]", "", without_marks.lower())


def is_author_derived_title(title: str | None, *identities: str | None) -> bool:
    """True when a resource title is really one of its author identities.

    Exact after normalization, deliberately not fuzzy: a title that only nearly
    matches an author may still carry information the source produced, and
    rewriting it would destroy that. A title that *is* the author holds nothing
    the author column does not already hold, so re-deriving it is safe.
    """
    if not title:
        return False
    normalized_title = normalize_identity(title)
    if not normalized_title:
        return False
    return any(
        normalized_title == normalize_identity(identity) for identity in identities if identity
    )


def is_usable_line(line: str) -> bool:
    """True when a line reads as meaningful prose rather than chrome or a link."""
    line = _strip_list_marker(line)
    if not line or len(line) < MIN_TITLE_CHARS:
        return False
    if line.lower().startswith("http") or line.startswith("["):
        return False
    if _JUNK_LINE.match(line):
        return False
    if _BARE_HANDLE.match(line) or _NUMERIC.match(line):
        return False
    return True


def first_meaningful_line(text: str) -> str | None:
    """The first prose line of a document body, or None when there is none.

    Lines shorter than the minimum are skipped rather than accepted, because X
    posts routinely open with a one-word hook ("BREAKING:", "BONUS!", "/goal")
    on its own line before the substance.
    """
    for raw in (text or "").splitlines():
        line = _tidy(raw)
        if is_usable_line(line):
            return line
    return None


def derive_post_title(text: str, *, fallback: str | None = None) -> str:
    """A title for a saved post, taken from its own text.

    Returns the fallback when the post carries no usable prose (a bare link, or a
    media-only post). The caller decides what that fallback is; the author is a
    reasonable last resort there and nowhere else.
    """
    line = first_meaningful_line(text)
    if line:
        return shorten(line)
    return fallback or ""
