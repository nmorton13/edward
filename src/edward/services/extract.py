"""Content extraction service leveraging summarize CLI with defensive parsing and safe fallback.

Read this before changing the fallback logic.

``summarize --extract --json`` nests its values under an ``extracted`` object::

    {"input": {...}, "env": {...}, "extracted": {"title": ..., "content": ...}}

Edward originally read those fields from the *top level*, so every lookup
returned ``None``, the "did we get content?" check failed, and every extraction
silently degraded to :func:`clean_html_simple` — which strips tags but keeps
navigation, footers, and page chrome. That is why the archive carried
page-sized blobs of nav links, why its biggest resources were its least useful
ones, and why ``extractor='summarize'`` appeared zero times across the whole
corpus despite summarize working.

The fallback also reported "summarize returned no text" whenever it fired,
describing a summarize failure that had not happened and pointing every later
investigation at the wrong suspect. Fallbacks now carry a specific reason that
can be persisted alongside the content.
"""

import json
import re
from dataclasses import dataclass

from edward.services.subprocess_runner import (
    ToolNotFoundError,
    is_tool_available,
    run_tool,
    sanitize_error_message,
)

# A summarize result shorter than this that is also just a URL or a couple of
# words is a placeholder, not a page: summarize resolved the request but could
# not read the destination, so the raw HTML is the better source.
MIN_MEANINGFUL_CONTENT_CHARS = 120

FALLBACK_NO_SUMMARIZE_TEXT = "summarize returned no extractable text"
FALLBACK_PLACEHOLDER = "summarize returned only a URL placeholder, not page content"
FALLBACK_SCHEMA_MISMATCH = "summarize output had no 'extracted' object; its format may have changed"


@dataclass
class ExtractionResult:
    status: str  # "completed", "pending", "failed"
    clean_text: str | None = None
    summary: str | None = None
    title: str | None = None
    transcript: str | None = None
    extractor: str = "none"
    extractor_version: str = "1.0"
    error: str | None = None


def clean_html_simple(html_text: str) -> tuple[str | None, str]:
    """Extract a title and plain body text locally, without executing code.

    This is a last resort, not a reader. It removes scripts, styles, and tags
    but has no notion of page structure, so navigation, sidebars, footers, and
    cookie banners all survive as body text. Prefer :func:`extract_content`.
    """
    # Extract title
    title_match = re.search(r"<title[^>]*>(.*?)</title>", html_text, re.IGNORECASE | re.DOTALL)
    title = title_match.group(1).strip() if title_match else None
    if title:
        title = re.sub(r"\s+", " ", title)

    # Strip script and style blocks
    cleaned = re.sub(
        r"<(script|style|svg|noscript)[^>]*>.*?</\1>",
        " ",
        html_text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    # Strip HTML tags
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    # Normalize whitespace
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n\s*\n+", "\n\n", cleaned).strip()

    return title, cleaned


def _payload_field(payload: dict, *keys: str) -> str | None:
    """Read a field from summarize's output, nested first then top-level.

    summarize nests its values under ``extracted``; older or future releases
    may differ, so check the nested object first and fall back to the top level
    rather than assuming either shape.
    """
    nested = payload.get("extracted")
    if isinstance(nested, dict):
        for key in keys:
            value = nested.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
            if value and not isinstance(value, str):
                return value
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if value and not isinstance(value, str):
            return value
    return None


def _is_placeholder_content(content: str | None, url: str) -> bool:
    """True when 'content' is really the URL or a stub, not the page body."""
    text = (content or "").strip()
    if not text:
        return True
    if len(text) >= MIN_MEANINGFUL_CONTENT_CHARS:
        return False
    # A short value that is a bare URL, or echoes the requested host, is the
    # redirect destination being mistaken for content.
    if text.startswith(("http://", "https://")):
        return True
    try:
        from urllib.parse import urlparse

        host = (urlparse(url).hostname or "").lower()
        if host and host in text.lower():
            return True
    except ValueError:
        pass
    return len(text.split()) <= 2


def _local_fallback(
    raw_html: str | None,
    *,
    reason: str,
    title_hint: str | None = None,
) -> ExtractionResult:
    """Build an explicitly-reasoned local fallback result.

    ``reason`` describes why summarize was not used, so a later investigation
    can tell a real failure from a placeholder from an unreadable output shape
    instead of inferring it from a misleading extractor name.
    """
    if not raw_html:
        return ExtractionResult(
            status="failed",
            extractor="summarize",
            error=reason,
        )
    fallback_title, clean = clean_html_simple(raw_html)
    return ExtractionResult(
        status="completed",
        clean_text=clean,
        title=title_hint or fallback_title,
        extractor="local-fallback",
        extractor_version="1.0",
        error=reason,
    )


def extract_content(url: str, raw_html: str | None = None) -> ExtractionResult:
    """Extract readable content and metadata from a URL using summarize or local fallback."""
    if is_tool_available("summarize"):
        try:
            res = run_tool(["summarize", url, "--extract", "--plain", "--json"], timeout=45.0)
            if res.exit_code == 0 and res.stdout.strip():
                try:
                    payload = json.loads(res.stdout)
                except json.JSONDecodeError:
                    # If plain text returned rather than JSON
                    payload = {"content": res.stdout.strip()}
                if not isinstance(payload, dict):
                    payload = {"content": str(payload)}

                # summarize nests its results under "extracted"; older or
                # alternate shapes keep them at the top level.
                title = _payload_field(payload, "title", "name")
                summary = _payload_field(payload, "description", "summary")
                content = _payload_field(
                    payload, "content", "extracted_content", "text", "clean_text"
                )
                transcript = _payload_field(payload, "transcript", "transcriptTimedText")
                if isinstance(transcript, (dict, list)):
                    transcript = json.dumps(transcript)

                version = payload.get("version") or "0.21.x"

                has_content = bool(content) and not _is_placeholder_content(content, url)
                if has_content or summary or transcript:
                    return ExtractionResult(
                        status="completed",
                        clean_text=content or summary or "",
                        summary=summary,
                        title=title,
                        transcript=transcript,
                        extractor="summarize",
                        extractor_version=version,
                    )

                # summarize answered, but the answer is not page text.
                if not isinstance(payload.get("extracted"), dict):
                    reason = FALLBACK_SCHEMA_MISMATCH
                elif content:
                    reason = FALLBACK_PLACEHOLDER
                else:
                    reason = FALLBACK_NO_SUMMARIZE_TEXT
                return _local_fallback(raw_html, reason=reason, title_hint=title)
            else:
                # Execution failed
                raw_err = res.stderr.strip() or f"summarize exited with code {res.exit_code}"
                error_msg = sanitize_error_message(raw_err)
                return _local_fallback(raw_html, reason=f"summarize failed ({error_msg})")
        except ToolNotFoundError:
            pass
        except Exception as e:
            sanitized_e = sanitize_error_message(str(e))
            return _local_fallback(raw_html, reason=f"summarize error ({sanitized_e})")

    # When summarize is not installed
    if raw_html:
        t, clean = clean_html_simple(raw_html)
        return ExtractionResult(
            status="completed",
            clean_text=clean,
            title=t,
            extractor="local-fallback",
            extractor_version="1.0",
            error="summarize tool not installed; used conservative local extraction",
        )

    # No tool and no raw HTML: preserve URL as pending
    return ExtractionResult(
        status="pending",
        extractor="none",
        extractor_version="",
        error="summarize tool not installed and no raw content available",
    )
