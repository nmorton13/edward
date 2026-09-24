"""Shortlink resolution, and the backfill that depends on it.

The bug these pin: a shortener's snapshot is its own interstitial, not the
destination. Reading it stored page chrome as the resource's content, so the
archive's largest entries carried sign-in prompts instead of articles.
"""

import pytest

from edward.services.shortlinks import (
    SHORTLINK_HOSTS,
    is_shortlink,
    looks_like_shortlink_interstitial,
    resolve_shortlink,
)

# --------------------------------------------------------------------------
# is_shortlink
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "https://t.co/8JaKk4HmQi",
        "http://t.co/abc",
        "https://bit.ly/xyz",
        "https://lnkd.in/abc",
    ],
)
def test_known_shorteners_are_recognised(url):
    assert is_shortlink(url) is True


@pytest.mark.parametrize(
    "url",
    [
        "https://atlasobscura.com/",
        "https://x.com/user/status/123",
        # A short path is not evidence of a shortener.
        "https://example.com/a",
        # A host that merely starts with a shortener's name is not one.
        "https://t.com.co.evil.example/x",
        "https://notbit.ly/x",
    ],
)
def test_other_hosts_are_not_shortlinks(url):
    assert is_shortlink(url) is False


def test_none_and_empty_are_not_shortlinks():
    assert is_shortlink(None) is False
    assert is_shortlink("") is False


def test_shortlink_hosts_is_deliberately_narrow():
    """Guessing broadly would re-fetch pages that are already fine."""
    assert "t.co" in SHORTLINK_HOSTS
    assert len(SHORTLINK_HOSTS) < 12


# --------------------------------------------------------------------------
# resolve_shortlink
# --------------------------------------------------------------------------


def test_non_shortlink_is_not_resolved(monkeypatch):
    """Resolving a normal URL would waste a fetch and change nothing."""
    called = []
    monkeypatch.setattr(
        "edward.services.shortlinks.safe_fetch_url",
        lambda url, **kw: called.append(url),
    )

    assert resolve_shortlink("https://atlasobscura.com/") is None
    assert called == []


def test_resolution_returns_final_url(monkeypatch):
    class _Fetched:
        final_url = "https://aboutideasnow.com/"

    monkeypatch.setattr("edward.services.shortlinks.safe_fetch_url", lambda url, **kw: _Fetched())

    assert resolve_shortlink("https://t.co/bk8ZBuoSax") == "https://aboutideasnow.com/"


def test_unresolvable_link_returns_none(monkeypatch):
    """A deleted or private destination is a normal outcome, not an error."""

    def _boom(url, **kw):
        raise RuntimeError("SSRF blocked or network down")

    monkeypatch.setattr("edward.services.shortlinks.safe_fetch_url", _boom)

    assert resolve_shortlink("https://t.co/gone") is None


def test_chain_ending_where_it_started_is_none(monkeypatch):
    class _Fetched:
        final_url = "https://t.co/same"

    monkeypatch.setattr("edward.services.shortlinks.safe_fetch_url", lambda url, **kw: _Fetched())

    assert resolve_shortlink("https://t.co/same") is None


def test_missing_final_url_is_none(monkeypatch):
    class _Fetched:
        final_url = None

    monkeypatch.setattr("edward.services.shortlinks.safe_fetch_url", lambda url, **kw: _Fetched())

    assert resolve_shortlink("https://t.co/abc") is None


def test_blank_final_url_is_none(monkeypatch):
    class _Fetched:
        final_url = "   "

    monkeypatch.setattr("edward.services.shortlinks.safe_fetch_url", lambda url, **kw: _Fetched())

    assert resolve_shortlink("https://t.co/abc") is None


# --------------------------------------------------------------------------
# The backfill
# --------------------------------------------------------------------------


@pytest.fixture
def fallback_resource(test_db):
    """A resource whose stored content is a shortener's interstitial."""
    with test_db.transaction() as conn:
        conn.execute(
            """
            INSERT INTO resources (id, canonical_url, url_hash, identity_key, title,
                                   created_at, updated_at)
            VALUES ('res_sl', 'https://t.co/bk8ZBuoSax', 'h1', 'k1', 'Old title',
                    '2026-01-01', '2026-01-01');
            """
        )
        conn.execute(
            """
            INSERT INTO resource_contents (id, resource_id, content_hash, clean_text,
                                           char_count, extractor, extractor_version, created_at)
            VALUES ('rc_sl', 'res_sl', 'ch1', 'Log in or sign up for X Trending now',
                    36, 'local-fallback', '1.0', '2026-01-01');
            """
        )
    return "res_sl"


def test_backfill_dry_run_writes_nothing(test_db, fallback_resource, monkeypatch):
    from edward.services import lifecycle
    from edward.services.extract import ExtractionResult

    monkeypatch.setattr(
        "edward.services.lifecycle.resolve_shortlink",
        lambda url, **kw: "https://aboutideasnow.com/",
    )
    import edward.services.extract as extract_mod

    monkeypatch.setattr(
        extract_mod,
        "extract_content",
        lambda url, **kw: ExtractionResult(
            status="completed",
            clean_text="Real article text about personal sites. " * 20,
            title="About Ideas Now",
            extractor="summarize",
        ),
    )

    result = lifecycle.backfill_shortlink_extractions(test_db, dry_run=True)

    assert result["dry_run"] is True
    assert result["recovered"] == 1
    assert result["details"][0]["old_chars"] == 36

    with test_db.connection() as conn:
        text = conn.execute(
            "SELECT clean_text FROM resource_contents WHERE resource_id='res_sl';"
        ).fetchone()["clean_text"]

    assert text == "Log in or sign up for X Trending now", "dry run must not write"


def test_backfill_replaces_interstitial_with_real_content(test_db, fallback_resource, monkeypatch):
    from edward.services import lifecycle
    from edward.services.extract import ExtractionResult

    monkeypatch.setattr(
        "edward.services.lifecycle.resolve_shortlink",
        lambda url, **kw: "https://aboutideasnow.com/",
    )
    import edward.services.extract as extract_mod

    real = "Real article text about personal sites. " * 20
    monkeypatch.setattr(
        extract_mod,
        "extract_content",
        lambda url, **kw: ExtractionResult(
            status="completed",
            clean_text=real,
            title="About Ideas Now",
            extractor="summarize",
        ),
    )

    result = lifecycle.backfill_shortlink_extractions(test_db)

    assert result["recovered"] == 1
    with test_db.connection() as conn:
        row = conn.execute(
            """
            SELECT clean_text, extractor, extraction_note FROM resource_contents
            WHERE resource_id='res_sl' ORDER BY created_at DESC LIMIT 1;
            """
        ).fetchone()
        url = conn.execute("SELECT canonical_url FROM resources WHERE id='res_sl';").fetchone()[
            "canonical_url"
        ]

    assert row["extractor"] == "summarize"
    assert "Real article text" in row["clean_text"]
    assert "Log in or sign up" not in row["clean_text"]
    assert row["extraction_note"] and "backfilled from" in row["extraction_note"]
    assert url == "https://t.co/bk8ZBuoSax", (
        "identity must not change: annotations, intents, and projects hang off it"
    )


def test_backfill_leaves_unresolvable_resources_alone(test_db, fallback_resource, monkeypatch):
    from edward.services import lifecycle

    monkeypatch.setattr("edward.services.lifecycle.resolve_shortlink", lambda url, **kw: None)

    result = lifecycle.backfill_shortlink_extractions(test_db)

    assert result["unresolved"] == 1
    assert result["recovered"] == 0
    with test_db.connection() as conn:
        text = conn.execute(
            "SELECT clean_text FROM resource_contents WHERE resource_id='res_sl';"
        ).fetchone()["clean_text"]

    assert text == "Log in or sign up for X Trending now", "nothing to replace it with"


def test_backfill_skips_non_shortlink_fallbacks(test_db, monkeypatch):
    """A fallback that is not a shortlink is a different problem, not this one."""
    from edward.services import lifecycle

    with test_db.transaction() as conn:
        conn.execute(
            """
            INSERT INTO resources (id, canonical_url, url_hash, identity_key, title,
                                   created_at, updated_at)
            VALUES ('res_plain', 'https://example.com/blocked', 'h2', 'k2', 'T',
                    '2026-01-01', '2026-01-01');
            """
        )
        conn.execute(
            """
            INSERT INTO resource_contents (id, resource_id, content_hash, clean_text,
                                           char_count, extractor, extractor_version, created_at)
            VALUES ('rc_plain', 'res_plain', 'ch2', 'some chrome', 11,
                    'local-fallback', '1.0', '2026-01-01');
            """
        )

    called = []
    monkeypatch.setattr(
        "edward.services.lifecycle.resolve_shortlink",
        lambda url, **kw: called.append(url),
    )

    result = lifecycle.backfill_shortlink_extractions(test_db)

    assert result["skipped"] == 1
    assert called == [], "a non-shortlink must not be fetched"


def test_backfill_limit_caps_the_pass(test_db, monkeypatch):
    from edward.services import lifecycle

    with test_db.transaction() as conn:
        for i in range(3):
            conn.execute(
                """
                INSERT INTO resources (id, canonical_url, url_hash, identity_key, title,
                                       created_at, updated_at)
                VALUES (?, ?, ?, ?, 'T', '2026-01-01', '2026-01-01');
                """,
                (f"res_l{i}", f"https://t.co/l{i}", f"h{i}", f"k{i}"),
            )
            conn.execute(
                """
                INSERT INTO resource_contents (id, resource_id, content_hash, clean_text,
                                               char_count, extractor, extractor_version, created_at)
                VALUES (?, ?, ?, 'Log in or sign up for X Trending now', 36,
                        'local-fallback', '1.0', '2026-01-01');
                """,
                (f"rc_l{i}", f"res_l{i}", f"c{i}"),
            )

    monkeypatch.setattr("edward.services.lifecycle.resolve_shortlink", lambda url, **kw: None)

    result = lifecycle.backfill_shortlink_extractions(test_db, limit=2)

    assert result["candidates"] == 2


def test_backfill_reports_extraction_failure(test_db, fallback_resource, monkeypatch):
    from edward.services import lifecycle

    monkeypatch.setattr(
        "edward.services.lifecycle.resolve_shortlink",
        lambda url, **kw: "https://aboutideasnow.com/",
    )
    import edward.services.extract as extract_mod
    from edward.services.extract import ExtractionResult

    monkeypatch.setattr(
        extract_mod,
        "extract_content",
        lambda url, **kw: ExtractionResult(
            status="failed",
            clean_text=None,
            title=None,
            extractor="summarize",
            error="Failed to fetch HTML document (status 403)",
        ),
    )

    result = lifecycle.backfill_shortlink_extractions(test_db)

    assert result["recovered"] == 0
    assert len(result["failed"]) == 1
    assert "403" in result["failed"][0]["error"]


def test_backfill_skips_a_shortlink_that_already_holds_the_article(test_db, monkeypatch):
    """Only chrome is a defect; re-fetching good content risks replacing it."""
    from edward.services import lifecycle

    real = "A detailed article about data centers and their electricity use. " * 10
    with test_db.transaction() as conn:
        conn.execute(
            """
            INSERT INTO resources (id, canonical_url, url_hash, identity_key, title,
                                   created_at, updated_at)
            VALUES ('res_ok', 'https://t.co/goodlink', 'h3', 'k3', 'T',
                    '2026-01-01', '2026-01-01');
            """
        )
        conn.execute(
            """
            INSERT INTO resource_contents (id, resource_id, content_hash, clean_text,
                                           char_count, extractor, extractor_version, created_at)
            VALUES ('rc_ok', 'res_ok', 'ch3', ?, ?, 'local-fallback', '1.0', '2026-01-01');
            """,
            (real, len(real)),
        )

    called = []
    monkeypatch.setattr(
        "edward.services.lifecycle.resolve_shortlink", lambda url, **kw: called.append(url)
    )

    result = lifecycle.backfill_shortlink_extractions(test_db)

    assert result["skipped"] == 1
    assert result["candidates"] == 0
    assert called == [], "an already-good shortlink must not be fetched"


# --------------------------------------------------------------------------
# Interstitial detection
# --------------------------------------------------------------------------


def test_x_signin_shell_is_detected_as_interstitial():
    chrome = (
        "Log in or sign up for X Trending now Continue with Google "
        "Continue with Apple Email or username"
    )
    assert looks_like_shortlink_interstitial(chrome) is True


def test_real_article_text_is_not_interstitial():
    article = "A detailed article about data centers and their electricity use. " * 10
    assert looks_like_shortlink_interstitial(article) is False


def test_a_single_marker_is_not_enough():
    """One passing mention of a phrase must not condemn real content."""
    text = "The article discusses how a sign in to x flow can be improved. " * 5
    assert looks_like_shortlink_interstitial(text) is False


def test_empty_text_is_not_interstitial():
    assert looks_like_shortlink_interstitial("") is False
    assert looks_like_shortlink_interstitial(None) is False


# --------------------------------------------------------------------------
# Superseded chunks must not stay searchable
# --------------------------------------------------------------------------


def test_backfill_rechunks_so_the_interstitial_is_no_longer_searchable(
    test_db, fallback_resource, monkeypatch
):
    """Storing content is not enough: the old chunks stay indexed without this."""
    from edward.services import lifecycle
    from edward.services.embed import store_resource_chunks
    from edward.services.extract import ExtractionResult

    with test_db.transaction() as conn:
        rc = conn.execute(
            "SELECT id FROM resource_contents WHERE resource_id='res_sl';"
        ).fetchone()["id"]
        store_resource_chunks(conn, "res_sl", rc, "Log in or sign up for X Trending now")

    monkeypatch.setattr(
        "edward.services.lifecycle.resolve_shortlink",
        lambda url, **kw: "https://aboutideasnow.com/",
    )
    import edward.services.extract as extract_mod

    real = "Real article text about personal sites and their owners. " * 20
    monkeypatch.setattr(
        extract_mod,
        "extract_content",
        lambda url, **kw: ExtractionResult(
            status="completed",
            clean_text=real,
            title="About Ideas Now",
            extractor="summarize",
        ),
    )

    lifecycle.backfill_shortlink_extractions(test_db)

    with test_db.connection() as conn:
        rows = conn.execute(
            "SELECT text FROM resource_chunks WHERE resource_id='res_sl';"
        ).fetchall()
        latest = conn.execute(
            "SELECT id FROM resource_contents WHERE resource_id='res_sl' "
            "ORDER BY created_at DESC LIMIT 1;"
        ).fetchone()["id"]
        owners = {
            r["resource_content_id"]
            for r in conn.execute(
                "SELECT resource_content_id FROM resource_chunks WHERE resource_id='res_sl';"
            ).fetchall()
        }

    texts = " ".join(r["text"] for r in rows)
    assert "Real article text" in texts, "the article must be chunked"
    assert "Log in or sign up" not in texts, (
        "the superseded interstitial must not remain searchable"
    )
    assert owners == {latest}, "chunks must belong to the current content only"


def test_repair_superseded_chunks_fixes_a_stale_resource(test_db, monkeypatch):
    from edward.services import lifecycle
    from edward.services.embed import store_resource_chunks

    with test_db.transaction() as conn:
        conn.execute(
            """
            INSERT INTO resources (id, canonical_url, url_hash, identity_key, title,
                                   created_at, updated_at)
            VALUES ('res_st', 'https://example.com/st', 'h9', 'k9', 'T',
                    '2026-01-01', '2026-01-01');
            """
        )
        conn.execute(
            """
            INSERT INTO resource_contents (id, resource_id, content_hash, clean_text,
                                           char_count, extractor, extractor_version, created_at)
            VALUES ('rc_old', 'res_st', 'c9', 'old body text', 13,
                    'local-fallback', '1.0', '2026-01-01');
            """
        )
        store_resource_chunks(conn, "res_st", "rc_old", "old body text")
        # A newer extraction arrives without re-chunking it.
        conn.execute(
            """
            INSERT INTO resource_contents (id, resource_id, content_hash, clean_text,
                                           char_count, extractor, extractor_version, created_at)
            VALUES ('rc_new', 'res_st', 'c10', 'new body text', 13,
                    'summarize', '1.0', '2026-06-01');
            """
        )

    result = lifecycle.repair_superseded_chunks(test_db)

    assert result["repaired"] == 1
    with test_db.connection() as conn:
        owners = {
            r["resource_content_id"]
            for r in conn.execute(
                "SELECT resource_content_id FROM resource_chunks WHERE resource_id='res_st';"
            ).fetchall()
        }
    assert owners == {"rc_new"}


def test_repair_superseded_chunks_is_idempotent(test_db):
    from edward.services import lifecycle

    with test_db.transaction() as conn:
        conn.execute(
            """
            INSERT INTO resources (id, canonical_url, url_hash, identity_key, title,
                                   created_at, updated_at)
            VALUES ('res_ok2', 'https://example.com/ok', 'h10', 'k10', 'T',
                    '2026-01-01', '2026-01-01');
            """
        )
        conn.execute(
            """
            INSERT INTO resource_contents (id, resource_id, content_hash, clean_text,
                                           char_count, extractor, extractor_version, created_at)
            VALUES ('rc_ok2', 'res_ok2', 'c11', 'body', 4, 'summarize', '1.0', '2026-01-01');
            """
        )

    first = lifecycle.repair_superseded_chunks(test_db)
    second = lifecycle.repair_superseded_chunks(test_db)

    assert first["repaired"] == 0, "a resource with no chunks is not this repair's job"
    assert second["repaired"] == 0
