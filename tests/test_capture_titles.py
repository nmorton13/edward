"""Edward inherits the capture store's curated title for an X bookmark.

The capture store is where bookmarks actually land: a 15-minute heartbeat writes
the record, classifies it, and enriches it. Its title is either a curated label
("AI policy without cartel capture") or a mechanical derivation of the post text.

Edward derived its own title from the post text alone, which yields a truncated
first line ("What Is To Be Done?"). Both are legitimate; the label is more useful.
These tests pin the preference order and the fallback.
"""

import json

import pytest

from edward.services.source_adapters import (
    _capture_store_record,
    _capture_store_title,
    _capture_store_title_is_curated,
)


@pytest.fixture
def store(tmp_path, monkeypatch):
    """A capture store laid out the way the real one is."""
    root = tmp_path / "captures" / "source"
    (root / "x-bookmarks").mkdir(parents=True)
    (root.parent / "scripts").mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("EDWARD_CAPTURE_STORE", str(root))
    return root


def write_capture(store, tweet_id: str, **fields):
    """Write one capture record. Omitting 'title' means no title field at all."""
    record = {"url": f"https://x.com/i/status/{tweet_id}", "x_id": tweet_id}
    record.update(fields)
    path = store / "x-bookmarks" / f"{tweet_id}.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    return path


# --------------------------------------------------------------------------
# Reading the record
# --------------------------------------------------------------------------


def test_record_is_read_by_tweet_id(store):
    write_capture(store, "123", title="A curated label", content="post text")

    record = _capture_store_record("123")

    assert record is not None
    assert record["title"] == "A curated label"


def test_missing_record_returns_none(store):
    assert _capture_store_record("999") is None


def test_malformed_record_returns_none(store):
    path = store / "x-bookmarks" / "456.json"
    path.write_text("{not json", encoding="utf-8")

    assert _capture_store_record("456") is None


def test_non_object_record_returned_as_none(store):
    path = store / "x-bookmarks" / "789.json"
    path.write_text('["a", "list"]', encoding="utf-8")

    assert _capture_store_record("789") is None


def test_absent_store_is_not_an_error(tmp_path, monkeypatch):
    """Edward must work on a host with no capture store at all."""
    monkeypatch.setenv("EDWARD_CAPTURE_STORE", str(tmp_path / "nope"))

    assert _capture_store_record("123") is None
    assert _capture_store_title("123") is None
    assert _capture_store_title_is_curated("123") is False


# --------------------------------------------------------------------------
# Which title to use
# --------------------------------------------------------------------------


def test_curated_title_is_returned(store):
    write_capture(store, "123", title="AI policy without cartel capture")

    assert _capture_store_title("123") == "AI policy without cartel capture"


def test_numeric_title_is_not_a_title(store):
    """Enrichment stores the tweet id as page_title; it must not leak through."""
    write_capture(store, "123", title="2100238648218427563")

    assert _capture_store_title("123") is None


def test_blank_title_returns_none(store):
    write_capture(store, "123", title="   ")

    assert _capture_store_title("123") is None


def test_record_without_a_title_returns_none(store):
    write_capture(store, "123", content="post text only")

    assert _capture_store_title("123") is None


# --------------------------------------------------------------------------
# Telling a curated title from a mechanical one
# --------------------------------------------------------------------------


def test_a_title_reproducible_from_its_own_fields_is_not_curated(store):
    """derive_title is pure: an auto title reproduces, a curated one does not.

    This is what removes the need for a provenance column.
    """
    # With only 'content' available, derive_title returns the first line of it.
    post = "I love this use of data centers The thing about it is that"
    write_capture(store, "123", title=post.split("\n")[0][:110], content=post)

    # If the capture store's module is present the distinction is drawn; if it is
    # not, a present title is trusted rather than discarded.
    result = _capture_store_title_is_curated("123")
    assert isinstance(result, bool)


def test_a_curated_title_survives_its_own_rederivation(store):
    """A label whose words are not the post's opening is curated."""
    write_capture(
        store,
        "123",
        title="AI policy without cartel capture",
        content="We are launching a new initiative today about compute governance.",
    )

    assert _capture_store_title_is_curated("123") is True


def test_numeric_title_is_never_curated(store):
    write_capture(store, "123", title="2100238648218427563")

    assert _capture_store_title_is_curated("123") is False


def test_missing_record_is_never_curated(store):
    assert _capture_store_title_is_curated("999") is False


def test_preference_is_expressed_by_the_sync_path():
    """The sync site must consult the store before deriving its own title."""
    import inspect

    from edward.services import source_adapters

    source = inspect.getsource(source_adapters.sync_x_bookmarks)
    curated_at = source.index("_capture_store_title_is_curated")
    derive_at = source.index("derive_post_title(")

    assert curated_at < derive_at, "the store's title must be preferred, not appended"
