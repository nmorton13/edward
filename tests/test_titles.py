"""Title derivation for source resources.

The rule under test: an X resource's title comes from the post text, and only a
title that exactly matches one of the resource's author identities is eligible
for repair.
"""

import json

import pytest

from edward.services.lifecycle import reindex_object_document, repair_author_derived_titles
from edward.services.titles import (
    derive_post_title,
    first_meaningful_line,
    is_author_derived_title,
    is_usable_line,
    normalize_identity,
    shorten,
)

# Set by `_blob_store` so the seed helpers can attach readable snapshot blobs.
_BLOB_STORE: list = []


@pytest.fixture(autouse=True)
def _blob_store(test_blob_store):
    """Make the active blob store available to the module-level seed helpers."""
    _BLOB_STORE[:] = [test_blob_store]
    yield
    _BLOB_STORE[:] = []


def test_title_comes_from_post_text_not_the_author():
    text = "AI Risk Hype and Reality\n\nThe loudest voices stoking fears..."
    assert derive_post_title(text, fallback="Andrew Ng") == "AI Risk Hype and Reality"


def test_short_opening_hook_is_skipped_for_the_substance_line():
    """X posts routinely open with a one-word hook on its own line."""
    text = "BREAKING: \n\n@OpenAI just dropped GPT-6 Sol. It is my new daily driver."
    title = derive_post_title(text, fallback="danshipper")
    assert title.startswith("@OpenAI just dropped GPT-6 Sol")


def test_bare_link_post_falls_back_to_the_author():
    assert derive_post_title("https://t.co/qAUP2dNJsi", fallback="jack") == "jack"


def test_chrome_and_navigation_lines_are_not_usable_titles():
    for line in (
        "Skip to Main Content",
        "Most Popular Opinion",
        "- Most Popular Opinion",
        "Sign in",
        "Subscribe",
        "@handle",
        "2102140576498065758",
        "https://example.com/some/long/path",
    ):
        assert not is_usable_line(line), line


def test_markdown_links_and_urls_are_stripped_from_titles():
    text = "Cloudflare Quick Tunnels\n\nExpose your local server: https://t.co/abc"
    assert derive_post_title(text).startswith("Cloudflare Quick Tunnels")


def test_first_meaningful_line_returns_none_for_empty_or_linkonly_text():
    assert first_meaningful_line("") is None
    assert first_meaningful_line("https://t.co/x\n\nhttps://t.co/y") is None


def test_shorten_marks_truncation_at_a_word_boundary():
    assert shorten("short line") == "short line"
    long = "word " * 40
    result = shorten(long, limit=40)
    assert result.endswith("\u2026")
    assert len(result) <= 41
    assert not result[:-1].endswith(" ")


def test_normalize_identity_ignores_case_spacing_and_punctuation():
    assert normalize_identity("Andrew Ng") == "andrewng"
    assert normalize_identity("@levelsio") == "levelsio"
    assert normalize_identity("Cr\u00e9mieux") == "cremieux"
    assert normalize_identity(None) == ""


def test_author_derived_title_is_exact_not_fuzzy():
    """A title that only nearly matches an author may still be real content."""
    assert is_author_derived_title("Andrew Ng", "Andrew Ng", "AndrewYNg")
    assert is_author_derived_title("@levelsio", "levelsio")
    assert not is_author_derived_title("AI Risk Hype and Reality", "Andrew Ng", "AndrewYNg")
    assert not is_author_derived_title("", "jack")
    # 'nader dabit' vs 'dabit3' is a near-miss: leave it alone.
    assert not is_author_derived_title("nader dabit", "dabit3")


def _seed_resource(conn, *, rid, title, author, url, text, snapshot_authors=None):
    conn.execute(
        """INSERT INTO resources (id, canonical_url, url_hash, identity_key, title, author,
                                  primary_form, review_state, is_deleted, created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, 'x-post', 'unreviewed', 0, '2026-01-01', '2026-01-01');""",
        (rid, url, f"hash-{rid}", f"url:{url}", title, author),
    )
    conn.execute(
        """INSERT INTO resource_contents (id, resource_id, content_hash, clean_text,
                                          char_count, extractor, extractor_version, created_at)
           VALUES (?, ?, ?, ?, ?, 'bird', 'v1', '2026-01-01');""",
        (f"rc-{rid}", rid, f"c-{rid}", text, len(text)),
    )
    if snapshot_authors:
        payload = {
            "birdclaw": {
                "author_name": snapshot_authors[0],
                "author_handle": snapshot_authors[1] if len(snapshot_authors) > 1 else None,
            }
        }
        _store_snapshot(conn, rid, payload)


def _store_snapshot(conn, rid, payload):
    """Attach a real, readable snapshot blob so the repair can resolve a display name."""
    blob_store = _BLOB_STORE[0]
    assert blob_store is not None, "tests using snapshots must set the blob store"
    content_hash, path = blob_store.store_bytes(json.dumps(payload).encode())
    conn.execute(
        """INSERT OR REPLACE INTO source_snapshots (id, resource_id, content_hash, headers_json,
                                                   blob_path, size_bytes, created_at)
           VALUES (?, ?, ?, '{}', ?, 0, '2026-01-01');""",
        (
            f"snp-{rid}",
            rid,
            content_hash,
            f"{content_hash[:2]}/{path.name}",
        ),
    )


def _titles(db):
    with db.connection() as conn:
        return {
            row["id"]: row["title"]
            for row in conn.execute("SELECT id, title FROM resources;").fetchall()
        }


def test_repair_rewrites_author_titles_and_leaves_real_ones(test_db, test_blob_store):
    with test_db.transaction() as conn:
        _seed_resource(
            conn,
            rid="res_author",
            title="Andrew Ng",
            author="AndrewYNg",
            url="https://x.com/i/status/1",
            text="AI Risk Hype and Reality\n\nThe loudest voices stoking fears...",
            snapshot_authors=("Andrew Ng", "AndrewYNg"),
        )
        _seed_resource(
            conn,
            rid="res_real",
            title="A title that already came from the source",
            author="someone",
            url="https://x.com/i/status/2",
            text="Body text that is long enough to be a title candidate.",
            snapshot_authors=("Someone Else", "someone"),
        )

    result = repair_author_derived_titles(test_db, test_blob_store)
    titles = _titles(test_db)

    assert result["repaired"] == 1
    assert result["resource_ids"] == ["res_author"]
    assert titles["res_author"] == "AI Risk Hype and Reality"
    assert titles["res_real"] == "A title that already came from the source"


def test_repair_uses_the_display_name_from_the_snapshot(test_db, test_blob_store):
    """The stored title is the display name; the handle column alone does not match it."""
    with test_db.transaction() as conn:
        _seed_resource(
            conn,
            rid="res_author",
            title="Andrew Ng",
            author="AndrewYNg",
            url="https://x.com/i/status/1",
            text="AI Risk Hype and Reality\n\nThe loudest voices stoking fears...",
            snapshot_authors=("Andrew Ng", "AndrewYNg"),
        )

    result = repair_author_derived_titles(test_db, test_blob_store)

    assert result["repaired"] == 1
    assert _titles(test_db)["res_author"] == "AI Risk Hype and Reality"


def test_repair_is_idempotent(test_db, test_blob_store):
    with test_db.transaction() as conn:
        _seed_resource(
            conn,
            rid="res_author",
            title="Andrew Ng",
            author="AndrewYNg",
            url="https://x.com/i/status/1",
            text="AI Risk Hype and Reality\n\nThe loudest voices stoking fears...",
            snapshot_authors=("Andrew Ng", "AndrewYNg"),
        )

    first = repair_author_derived_titles(test_db, test_blob_store)
    second = repair_author_derived_titles(test_db, test_blob_store)

    assert first["repaired"] == 1
    assert second["repaired"] == 0


def test_repair_skips_resources_with_no_stored_text(test_db, test_blob_store):
    with test_db.transaction() as conn:
        conn.execute(
            """INSERT INTO resources (id, canonical_url, url_hash, identity_key, title, author,
                                      primary_form, review_state, is_deleted, created_at, updated_at)
               VALUES ('res_notext', 'https://x.com/i/status/9', 'h9', 'url:9', 'jack', 'jack',
                       'x-post', 'unreviewed', 0, '2026-01-01', '2026-01-01');"""
        )

    result = repair_author_derived_titles(test_db, test_blob_store)

    assert result["repaired"] == 0
    assert result["skipped"] == 1


def test_repair_updates_the_search_projection(test_db, test_blob_store):
    with test_db.transaction() as conn:
        _seed_resource(
            conn,
            rid="res_author",
            title="Andrew Ng",
            author="AndrewYNg",
            url="https://x.com/i/status/1",
            text="AI Risk Hype and Reality\n\nThe loudest voices stoking fears...",
            snapshot_authors=("Andrew Ng", "AndrewYNg"),
        )
        reindex_object_document(conn, "resource", "res_author")

    repair_author_derived_titles(test_db, test_blob_store)

    with test_db.connection() as conn:
        row = conn.execute(
            """SELECT title FROM search_documents
               WHERE object_type='resource' AND object_id='res_author';"""
        ).fetchone()

    assert row is not None
    assert row["title"] == "AI Risk Hype and Reality"


def test_repair_survives_an_unreadable_snapshot_blob(test_db, test_blob_store):
    """A missing blob must not raise — snapshots are advisory for this repair."""
    with test_db.transaction() as conn:
        _seed_resource(
            conn,
            rid="res_author",
            title="Andrew Ng",
            author="AndrewYNg",
            url="https://x.com/i/status/1",
            text="AI Risk Hype and Reality\n\nThe loudest voices stoking fears...",
        )
        conn.execute(
            """INSERT INTO source_snapshots (id, resource_id, content_hash, headers_json,
                                             blob_path, size_bytes, created_at)
               VALUES ('snp-missing', 'res_author', 'de' || printf('%062d', 0), '{}',
                       'de/deadbeef', 0, '2026-01-01');"""
        )

    result = repair_author_derived_titles(test_db, test_blob_store)

    # The blob is unreadable, so no snapshot author resolves; nothing is provably
    # author-derived and nothing is destroyed.
    assert result["repaired"] == 0
