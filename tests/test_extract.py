"""Extraction must read summarize's real output shape, and say why it fell back.

The bug these tests pin: summarize nests its values under an ``extracted``
object, but the parser read them from the top level. Every lookup returned
None, the content check failed, and all extraction silently degraded to a
tag-stripping fallback that kept page chrome. ``extractor='summarize'`` appeared
zero times across the entire corpus while summarize was working perfectly.

The pre-existing test is kept at the bottom of this file, updated only where it
asserted the old behaviour.
"""

import json

import pytest

from edward.services.extract import (
    FALLBACK_PLACEHOLDER,
    FALLBACK_SCHEMA_MISMATCH,
    MIN_MEANINGFUL_CONTENT_CHARS,
    _is_placeholder_content,
    _payload_field,
    clean_html_simple,
    extract_content,
)
from edward.services.subprocess_runner import SubprocessResult

# A trimmed copy of real summarize output for an X post, including the nested
# shape that the parser previously missed.
REAL_SUMMARIZE_OUTPUT = {
    "input": {"kind": "url", "url": "https://t.co/8JaKk4HmQi", "format": "markdown"},
    "env": {"hasOpenAIKey": False, "hasFirecrawlKey": False},
    "extracted": {
        "url": "https://t.co/8JaKk4HmQi",
        "title": "Curious and Wondrous Travel Destinations - Atlas Obscura",
        "description": None,
        "siteName": "Atlas Obscura",
        "content": "Atlas Obscura is a magazine and travel company devoted to "
        + "sharing the world's hidden wonders. " * 20,
        "truncated": False,
        "totalCharacters": 9001,
    },
    "summary": None,
    "llm": None,
}

PLACEHOLDER_OUTPUT = {
    "input": {"kind": "url", "url": "https://t.co/abc", "format": "markdown"},
    "env": {},
    "extracted": {
        "url": "https://t.co/abc",
        "title": "http://atlasobscura.com",
        "description": None,
        "siteName": "t.co",
        "content": "http://atlasobscura.com",
        "truncated": False,
    },
    "summary": None,
}


@pytest.fixture
def fake_summarize(monkeypatch):
    """Install a stub summarize that returns a chosen payload."""
    from edward.services import extract as mod

    def _install(payload, exit_code=0, stderr=""):
        stdout = json.dumps(payload) if payload is not None else ""
        monkeypatch.setattr(mod, "is_tool_available", lambda tool: True)
        monkeypatch.setattr(
            mod,
            "run_tool",
            lambda *a, **k: SubprocessResult(exit_code, stdout, stderr, 0.1),
        )
        return mod

    return _install


# --------------------------------------------------------------------------
# The field-reading bug
# --------------------------------------------------------------------------


def test_nested_extracted_fields_are_read():
    """The core regression: values live under 'extracted', not at top level."""
    assert _payload_field(REAL_SUMMARIZE_OUTPUT, "content") is not None
    assert _payload_field(REAL_SUMMARIZE_OUTPUT, "title") == (
        "Curious and Wondrous Travel Destinations - Atlas Obscura"
    )


def test_top_level_fields_still_work():
    """Older or alternate summarize shapes must keep working."""
    flat = {"content": "plain body text", "title": "A title"}

    assert _payload_field(flat, "content") == "plain body text"
    assert _payload_field(flat, "title") == "A title"


def test_nested_is_preferred_over_top_level():
    payload = {"content": "top level", "extracted": {"content": "nested wins"}}

    assert _payload_field(payload, "content") == "nested wins"


def test_missing_everywhere_returns_none():
    assert _payload_field({}, "content") is None
    assert _payload_field({"extracted": {}}, "content") is None
    assert _payload_field({"extracted": "not-a-dict"}, "content") is None


def test_blank_strings_are_treated_as_absent():
    assert _payload_field({"extracted": {"content": "   "}}, "content") is None


def test_real_shape_yields_summarize_not_fallback(fake_summarize):
    """The whole point: this must no longer degrade to local-fallback."""
    fake_summarize(REAL_SUMMARIZE_OUTPUT)

    result = extract_content("https://t.co/8JaKk4HmQi")

    assert result.extractor == "summarize"
    assert result.status == "completed"
    assert result.extractor != "local-fallback"
    assert result.title == "Curious and Wondrous Travel Destinations - Atlas Obscura"
    assert len(result.clean_text or "") > MIN_MEANINGFUL_CONTENT_CHARS
    assert result.error is None


def test_extracted_content_is_not_replaced_by_raw_html(fake_summarize, monkeypatch):
    """With a good summarize result, the chrome-laden fallback must not run."""
    mod = fake_summarize(REAL_SUMMARIZE_OUTPUT)
    called = []
    original = mod.clean_html_simple
    monkeypatch.setattr(
        mod, "clean_html_simple", lambda html: (called.append(html), original(html))[1]
    )

    result = mod.extract_content(
        "https://t.co/8JaKk4HmQi", raw_html="<html><body>nav junk</body></html>"
    )

    assert result.extractor == "summarize"
    assert called == [], "the fallback must not run when summarize succeeded"


# --------------------------------------------------------------------------
# Placeholder detection
# --------------------------------------------------------------------------


def test_a_bare_url_is_not_content():
    assert _is_placeholder_content("http://atlasobscura.com", "https://t.co/x") is True


def test_long_real_content_is_not_a_placeholder():
    body = "A genuine article sentence about travel destinations. " * 10

    assert _is_placeholder_content(body, "https://t.co/x") is False


def test_empty_content_is_a_placeholder():
    assert _is_placeholder_content("", "https://t.co/x") is True
    assert _is_placeholder_content(None, "https://t.co/x") is True


def test_placeholder_result_is_labelled_honestly(fake_summarize):
    """A URL-echo result must not claim summarize was simply unavailable."""
    fake_summarize(PLACEHOLDER_OUTPUT)

    result = extract_content("https://t.co/abc")

    assert result.error == FALLBACK_PLACEHOLDER
    assert "no text" not in (result.error or ""), (
        "the reason must describe a placeholder, not a missing-text failure"
    )


def test_placeholder_falls_back_to_raw_html_when_available(fake_summarize):
    fake_summarize(PLACEHOLDER_OUTPUT)

    result = extract_content(
        "https://t.co/abc", raw_html="<html><title>T</title><body>real body</body></html>"
    )

    assert result.extractor == "local-fallback"
    assert "real body" in (result.clean_text or "")


# --------------------------------------------------------------------------
# Honest failure reporting
# --------------------------------------------------------------------------


def test_unreadable_output_shape_is_named(fake_summarize):
    """If summarize's format changes again, the error must say so."""
    fake_summarize({"some": "future", "shape": True})

    result = extract_content("https://example.com/page")

    assert result.error == FALLBACK_SCHEMA_MISMATCH


def test_tool_failure_names_the_tool_error(fake_summarize):
    """A real summarize failure must still be reported as one."""
    mod = fake_summarize(None, exit_code=1, stderr="Failed to fetch HTML document (status 403)")

    result = mod.extract_content("https://example.com/page")

    assert result.status == "failed"
    assert "403" in (result.error or "")
    assert result.extractor == "summarize"


def test_no_raw_html_and_no_content_is_a_real_failure(fake_summarize):
    """Nothing usable and nothing to fall back to must not be 'completed'."""
    fake_summarize(PLACEHOLDER_OUTPUT)

    result = extract_content("https://t.co/abc")

    assert result.status == "failed"
    assert result.clean_text in (None, "")


def test_fallback_reason_is_specific_not_generic():
    """Every fallback path must name its own cause."""
    assert FALLBACK_PLACEHOLDER != FALLBACK_SCHEMA_MISMATCH
    assert "placeholder" in FALLBACK_PLACEHOLDER
    assert "extracted" in FALLBACK_SCHEMA_MISMATCH


# --------------------------------------------------------------------------
# The fallback remains a last resort, documented as such
# --------------------------------------------------------------------------


def test_fallback_keeps_page_chrome_and_says_so():
    """clean_html_simple strips tags but not nav/footer — that is why it is last."""
    html = (
        "<html><body><nav>Menu Home About</nav><p>Body text</p>"
        "<footer>© 2026</footer></body></html>"
    )

    _title, clean = clean_html_simple(html)

    assert "Body text" in clean
    # Documented limitation: chrome survives, so this is not a reader.
    assert "Menu Home About" in clean
    assert "© 2026" in clean


def test_min_meaningful_chars_is_a_real_threshold():
    assert MIN_MEANINGFUL_CONTENT_CHARS >= 60


# --------------------------------------------------------------------------
# Persistence of the reason
# --------------------------------------------------------------------------


def test_extraction_note_is_persisted(test_db):
    from edward.services.resource import store_resource_content

    with test_db.transaction() as conn:
        conn.execute(
            """
            INSERT INTO resources (id, canonical_url, url_hash, identity_key, title,
                                   created_at, updated_at)
            VALUES ('res_x', 'https://t.co/abc', 'h', 'k', 'T', '2026-01-01', '2026-01-01');
            """
        )
        store_resource_content(
            conn,
            "res_x",
            "body text",
            extractor="local-fallback",
            extraction_note=FALLBACK_PLACEHOLDER,
        )

    with test_db.connection() as conn:
        row = conn.execute(
            "SELECT extractor, extraction_note FROM resource_contents WHERE resource_id='res_x';"
        ).fetchone()

    assert row["extractor"] == "local-fallback"
    assert row["extraction_note"] == FALLBACK_PLACEHOLDER, (
        "without the note, the only trace is an extractor name that "
        "cannot distinguish a failure from a placeholder"
    )


def test_extraction_note_defaults_to_null(test_db):
    from edward.services.resource import store_resource_content

    with test_db.transaction() as conn:
        conn.execute(
            """
            INSERT INTO resources (id, canonical_url, url_hash, identity_key, title,
                                   created_at, updated_at)
            VALUES ('res_y', 'https://example.com/a', 'h', 'k', 'T', '2026-01-01', '2026-01-01');
            """
        )
        store_resource_content(conn, "res_y", "body", extractor="summarize")

    with test_db.connection() as conn:
        note = conn.execute(
            "SELECT extraction_note FROM resource_contents WHERE resource_id='res_y';"
        ).fetchone()["extraction_note"]

    assert note is None, "a clean extraction needs no excuse"


# --------------------------------------------------------------------------
# Pre-existing behaviour, updated only where it asserted the old shape
# --------------------------------------------------------------------------


def test_empty_summarize_response_uses_saved_html(monkeypatch):
    """A title-only summarize response is not content, so saved HTML is used."""
    monkeypatch.setattr("edward.services.extract.is_tool_available", lambda tool: True)
    monkeypatch.setattr(
        "edward.services.extract.run_tool",
        lambda *args, **kwargs: SubprocessResult(0, '{"title":"Article"}', "", 0.1),
    )
    result = extract_content(
        "https://example.com/article",
        raw_html="<html><body><p>Relevant source detail.</p></body></html>",
    )
    assert result.clean_text == "Relevant source detail."
    assert result.extractor == "local-fallback"
    # The reason is now recorded rather than left to inference.
    assert result.error


def test_nested_title_only_response_still_falls_back(monkeypatch):
    """A real summarize shape carrying only a title has no body to store."""
    monkeypatch.setattr("edward.services.extract.is_tool_available", lambda tool: True)
    monkeypatch.setattr(
        "edward.services.extract.run_tool",
        lambda *args, **kwargs: SubprocessResult(
            0, '{"extracted":{"title":"Article","content":null}}', "", 0.1
        ),
    )
    result = extract_content(
        "https://example.com/article",
        raw_html="<html><body><p>Saved body.</p></body></html>",
    )
    assert result.extractor == "local-fallback"
    assert "Saved body." in (result.clean_text or "")
