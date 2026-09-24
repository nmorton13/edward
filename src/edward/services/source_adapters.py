"""Read-only native adapters for the user's X bookmarks and self-sent Gmail."""

import datetime
import importlib.util
import json
import logging
import mimetypes
import os
import re
import sqlite3
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from edward.blobs import BlobStore
from edward.db import Database
from edward.models import CaptureInput, generate_id, make_job_key
from edward.services.capture import capture_item
from edward.services.extract import clean_html_simple, extract_content
from edward.services.lifecycle import reindex_object_document
from edward.services.network import safe_fetch_url
from edward.services.ocr import enqueue_ocr_extraction, is_image_attachment
from edward.services.pdf import enqueue_pdf_extraction, is_pdf_attachment
from edward.services.resource import (
    get_cached_content,
    get_or_create_url_resource,
    store_resource_content,
)
from edward.services.search import index_document
from edward.services.subprocess_runner import SubprocessError, is_tool_available, run_tool
from edward.services.titles import derive_post_title

BOOKMARK_KIND = "bookmarks"
BIRDCLAW_SOURCE = "bird"
SELF_EMAIL_QUERY = "from:me to:me"
URL_PATTERN = re.compile(r"https?://[^\s<>\"']+", re.IGNORECASE)

# An attached image is a screenshot, not a document; this caps a single fetch.
MEDIA_FETCH_MAX_BYTES = 20 * 1024 * 1024

logger = logging.getLogger(__name__)
EMAIL_PATTERN = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.IGNORECASE)


class SourceAdapterError(Exception):
    """Raised when a source cannot be read completely and safely."""


def _json_object(raw: str, tool_name: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SourceAdapterError(f"{tool_name} returned invalid JSON: {exc}") from exc


def _run_json_tool(args: list[str], tool_name: str, *, allow_plain_text: bool = False) -> Any:
    try:
        result = run_tool(args, timeout=60.0)
    except SubprocessError as exc:
        raise SourceAdapterError(str(exc)) from exc
    if result.exit_code != 0:
        message = result.stderr.strip() or f"{tool_name} exited with code {result.exit_code}"
        raise SourceAdapterError(message)
    if not result.stdout.strip():
        raise SourceAdapterError(f"{tool_name} returned no output")
    try:
        return _json_object(result.stdout, tool_name)
    except SourceAdapterError:
        if allow_plain_text:
            return {"text": result.stdout.strip()}
        raise


def _dicts_with_keys(value: Any, keys: set[str]):
    if isinstance(value, dict):
        if keys.intersection(value):
            yield value
        for child in value.values():
            yield from _dicts_with_keys(child, keys)
    elif isinstance(value, list):
        for child in value:
            yield from _dicts_with_keys(child, keys)


def _first_text(value: Any, keys: tuple[str, ...]) -> str | None:
    if isinstance(value, dict):
        for key in keys:
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
        for child in value.values():
            candidate = _first_text(child, keys)
            if candidate:
                return candidate
    elif isinstance(value, list):
        for child in value:
            candidate = _first_text(child, keys)
            if candidate:
                return candidate
    return None


def _decode_json_column(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return default


def _birdclaw_database_path() -> Path:
    configured = os.environ.get("EDWARD_BIRDCLAW_DB", "").strip()
    return (
        Path(configured).expanduser()
        if configured
        else Path.home() / ".birdclaw" / "birdclaw.sqlite"
    )


def _read_birdclaw_bookmarks() -> list[dict[str, Any]]:
    path = _birdclaw_database_path()
    if not path.is_file():
        raise SourceAdapterError(f"Birdclaw archive not found: {path}")

    uri = f"{path.resolve().as_uri()}?mode=ro"
    try:
        conn = sqlite3.connect(uri, uri=True)
        conn.row_factory = sqlite3.Row
        table_names = {
            row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table';")
        }
        required = {"tweet_collections", "tweets", "profiles"}
        if not required.issubset(table_names):
            missing = ", ".join(sorted(required - table_names))
            raise SourceAdapterError(f"Birdclaw archive is missing required tables: {missing}")

        rows = conn.execute(
            """
            SELECT tc.account_id, tc.tweet_id, tc.collected_at, tc.source,
                   tc.raw_json AS collection_json,
                   t.author_profile_id, t.text, t.created_at AS published_at,
                   t.entities_json, t.media_json, t.deleted_at,
                   p.handle AS author_handle, p.display_name AS author_name
            FROM tweet_collections AS tc
            JOIN tweets AS t ON t.id = tc.tweet_id
            LEFT JOIN profiles AS p ON p.id = t.author_profile_id
            WHERE lower(tc.kind) = ? AND lower(tc.source) = ?
            ORDER BY tc.collected_at DESC, tc.tweet_id DESC;
            """,
            (BOOKMARK_KIND, BIRDCLAW_SOURCE),
        ).fetchall()
        return [dict(row) for row in rows]
    except sqlite3.Error as exc:
        raise SourceAdapterError(f"Could not read Birdclaw archive: {exc}") from exc
    finally:
        if "conn" in locals():
            conn.close()


def _x_reader_payload(tweet_id: str) -> tuple[Any | None, str | None, str | None]:
    failures: list[str] = []
    if is_tool_available("xurl"):
        try:
            return (
                _run_json_tool(["xurl", "read", tweet_id], "xurl", allow_plain_text=True),
                "xurl",
                None,
            )
        except SourceAdapterError as exc:
            failures.append(f"xurl: {exc}")

    if is_tool_available("bird"):
        try:
            return (
                _run_json_tool(["bird", "read", tweet_id, "--json"], "bird"),
                "bird",
                "; ".join(failures) or None,
            )
        except SourceAdapterError as exc:
            failures.append(f"bird: {exc}")

    if failures:
        return None, None, "; ".join(failures)
    return None, None, "xurl and bird are not installed"


def _extract_author_thread(
    tweets: list[dict[str, Any]],
    bookmarked_tweet_id: str,
) -> list[dict[str, Any]]:
    """Walk an author's self-reply chain up to root and down to leaf.

    Given a list of tweets from a conversation thread (e.g. from `bird thread --json`),
    filters strictly for self-replies by the same author following inReplyToStatusId links.
    Returns the complete linear thread in chronological order (root -> leaf).
    """
    by_id = {str(t.get("id")): t for t in tweets if t.get("id")}
    bookmarked = by_id.get(str(bookmarked_tweet_id))
    if not bookmarked:
        return []

    author = bookmarked.get("author", {}).get("username")
    author_id = bookmarked.get("authorId")

    def is_same_author(t: dict[str, Any] | None) -> bool:
        if not t:
            return False
        t_author = t.get("author", {}).get("username")
        t_author_id = t.get("authorId")
        if author and t_author and t_author.lower() == author.lower():
            return True
        if author_id and t_author_id and str(t_author_id) == str(author_id):
            return True
        return False

    # 1. Walk UP to the thread start
    curr = bookmarked
    visited_up: set[str] = {str(curr.get("id"))}
    while True:
        parent_id = curr.get("inReplyToStatusId")
        if not parent_id or str(parent_id) in visited_up:
            break
        parent = by_id.get(str(parent_id))
        if parent and is_same_author(parent):
            curr = parent
            visited_up.add(str(curr.get("id")))
        else:
            break
    root = curr

    # 2. Walk DOWN from root to the end of the author chain
    chain = [root]
    curr = root
    visited_down: set[str] = {str(root.get("id"))}
    while True:
        curr_id = str(curr.get("id"))
        child = None
        for t in tweets:
            t_id = str(t.get("id"))
            if (
                t_id not in visited_down
                and str(t.get("inReplyToStatusId")) == curr_id
                and is_same_author(t)
            ):
                child = t
                break
        if child:
            chain.append(child)
            visited_down.add(str(child.get("id")))
            curr = child
        else:
            break

    return chain


def _unroll_x_thread(tweet_id: str) -> list[dict[str, Any]]:
    """Unroll an author's self-reply thread via bird thread --json.

    If bird is available, fetches the conversation containing tweet_id and follows
    the author's self-reply chain up to root and down to leaf.
    Returns the linear list of tweets in chronological order.
    Returns [] if not a thread, tool unavailable, or on failure.
    """
    if not is_tool_available("bird"):
        return []
    try:
        payload = _run_json_tool(["bird", "thread", tweet_id, "--json"], "bird thread")
        if not isinstance(payload, list) or len(payload) < 2:
            return []
        chain = _extract_author_thread(payload, tweet_id)
        return chain if len(chain) >= 2 else []
    except Exception as exc:
        logger.debug("Failed to unroll X thread %s: %s", tweet_id, exc)
        return []


def _format_thread_text(chain: list[dict[str, Any]]) -> str:
    """Stitch an unrolled thread into clean continuous text."""
    parts = []
    for t in chain:
        text = _tweet_text(t)
        if text:
            parts.append(text)
    return "\n\n---\n\n".join(parts)


def _collect_thread_media_and_links(
    chain: list[dict[str, Any]],
    thread_text: str,
) -> tuple[list[str], list[str]]:
    """Gather all media URLs and outgoing external links across every tweet in the thread."""
    all_media: list[str] = []
    all_links: list[str] = []
    seen_media: set[str] = set()
    seen_links: set[str] = set()

    for t in chain:
        media_list = t.get("media") or []
        entities = t.get("entities") or {}
        text = _tweet_text(t) or ""
        media_urls = _tweet_media_urls(media_list, entities, text)
        for m in media_urls:
            if m not in seen_media:
                seen_media.add(m)
                all_media.append(m)

        links = _tweet_urls(entities, text)
        for link in links:
            t_id = str(t.get("id"))
            if f"/status/{t_id}" not in link and link not in seen_media and link not in seen_links:
                seen_links.add(link)
                all_links.append(link)

    return all_media, all_links


def _tweet_text(payload: Any) -> str | None:
    return _first_text(payload, ("text", "full_text", "note_tweet_text"))


def _tweet_urls(entities: Any, text: str) -> list[str]:
    candidates: list[str] = []
    if isinstance(entities, dict):
        for item in entities.get("urls", []):
            if not isinstance(item, dict):
                continue
            url = item.get("expanded_url") or item.get("expandedUrl") or item.get("url")
            if isinstance(url, str):
                candidates.append(url)
    candidates.extend(URL_PATTERN.findall(text))

    unique: list[str] = []
    seen: set[str] = set()
    for value in candidates:
        cleaned = value.rstrip(".,!?;:)")
        parsed = urlparse(cleaned)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            continue
        host = parsed.netloc.lower().split(":", 1)[0]
        if host == "pic.x.com":
            continue
        if cleaned not in seen:
            seen.add(cleaned)
            unique.append(cleaned)
    return unique


TWITTER_MEDIA_HOSTS = (
    "pbs.twimg.com",
    "video.twimg.com",
    "ton.twimg.com",
)

# Only these asset paths are post media. A bare extension check would also
# catch profile avatars, which are not attached to any post.
TWITTER_MEDIA_PATH_PREFIXES = (
    "/media/",
    "/amplify_video_thumb/",
    "/amplify_video/",
    "/ext_tw_video_thumb/",
    "/ext_tw_video/",
    "/tweet_video_thumb/",
    "/tweet_video/",
)


def is_twitter_media_url(url: str) -> bool:
    """True for an X/Twitter post image or video asset, not a linked page."""
    try:
        parsed = urlparse(url)
    except ValueError:
        return False
    host = (parsed.netloc or "").lower().split(":", 1)[0]
    if host not in TWITTER_MEDIA_HOSTS:
        return False
    path = (parsed.path or "").lower()
    return any(path.startswith(prefix) for prefix in TWITTER_MEDIA_PATH_PREFIXES)


def _tweet_media_urls(media: Any, entities: Any, text: str) -> list[str]:
    """Collect the media asset URLs a post carries.

    birdclaw records these both in ``media_json`` and — unhelpfully — inside
    ``entities_json.urls``, which is why they used to become resources.
    """
    found: list[str] = []

    def _add(value: Any) -> None:
        if isinstance(value, str) and value.startswith(("http://", "https://")):
            if is_twitter_media_url(value) and value not in found:
                found.append(value)

    if isinstance(media, list):
        for item in media:
            if isinstance(item, dict):
                _add(item.get("url"))
                _add(item.get("thumbnailUrl"))
                _add(item.get("thumbnail_url"))
                for variant in item.get("variants") or []:
                    if isinstance(variant, dict):
                        _add(variant.get("url"))
    if isinstance(entities, dict):
        for item in entities.get("urls", []):
            if isinstance(item, dict):
                _add(item.get("expandedUrl") or item.get("expanded_url") or item.get("url"))
    for candidate in URL_PATTERN.findall(text or ""):
        _add(candidate)
    return found


def _capture_tweet_media(
    conn: sqlite3.Connection,
    blob_store: BlobStore,
    *,
    resource_id: str,
    media_urls: list[str],
    now_iso: str,
) -> int:
    """Persist a post's images as attachments of that post and queue their OCR.

    The image is not a source and never becomes a resource; it is evidence
    attached to the post that carried it. Text read out of it later joins the
    post's searchable body under a provenance marker.
    """
    saved = 0
    for url in media_urls:
        file_name = Path(urlparse(url).path).name or "image"
        if not is_image_attachment(file_name, ""):
            # Video assets are recorded but not downloaded: OCR does not apply,
            # and there is no video pipeline yet.
            continue
        try:
            fetched = safe_fetch_url(url, max_bytes=MEDIA_FETCH_MAX_BYTES)
        except Exception as exc:  # network/SSRF/size — attachment is optional
            logger.warning("Could not fetch media %s: %s", url, exc)
            continue
        if not fetched.body:
            continue

        content_hash, _path = blob_store.store_bytes(fetched.body)
        mime_type = (
            fetched.headers.get("content-type", "").split(";", 1)[0].strip()
            or mimetypes.guess_type(file_name)[0]
            or "application/octet-stream"
        )
        attachment_id = generate_id("att")
        conn.execute(
            """
            INSERT INTO attachments (
                id, object_type, object_id, file_name, mime_type, content_hash,
                size_bytes, blob_path, created_at
            ) VALUES (?, 'resource', ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                attachment_id,
                resource_id,
                file_name,
                mime_type,
                content_hash,
                len(fetched.body),
                f"{content_hash[:2]}/{content_hash}",
                now_iso,
            ),
        )
        if is_image_attachment(file_name, mime_type):
            enqueue_ocr_extraction(
                conn,
                _capture_id_for_resource(conn, resource_id) or resource_id,
                attachment_id,
                now_iso=now_iso,
            )
        saved += 1
    return saved


def _capture_id_for_resource(conn: sqlite3.Connection, resource_id: str) -> str | None:
    row = conn.execute(
        """
        SELECT capture_id FROM capture_resources
        WHERE resource_id = ? ORDER BY created_at ASC LIMIT 1;
        """,
        (resource_id,),
    ).fetchone()
    return row["capture_id"] if row else None


def _queue_link_resource(
    conn: sqlite3.Connection,
    capture_id: str,
    url: str,
    now_iso: str,
) -> bool:
    try:
        resource_id, _created = get_or_create_url_resource(conn, url, title=url)
    except ValueError:
        return False

    link = conn.execute(
        """
        INSERT OR IGNORE INTO capture_resources (capture_id, resource_id, relationship_type, created_at)
        VALUES (?, ?, 'referenced', ?);
        """,
        (capture_id, resource_id, now_iso),
    )
    cached = get_cached_content(conn, resource_id)
    if not cached:
        index_document(conn, "resource", resource_id, title=url, body=url)
        job_id = generate_id("job")
        job_key = make_job_key("fetch", resource_id)
        conn.execute(
            """
            INSERT INTO processing_jobs (
                id, job_key, capture_id, resource_id, stage, status,
                available_at, attempts, created_at, updated_at
            ) VALUES (?, ?, ?, ?, 'resource-fetch', 'pending', ?, 0, ?, ?)
            ON CONFLICT(job_key) DO UPDATE SET
                capture_id = COALESCE(excluded.capture_id, processing_jobs.capture_id),
                updated_at = excluded.updated_at;
            """,
            (job_id, job_key, capture_id, resource_id, now_iso, now_iso, now_iso),
        )
    return link.rowcount > 0


def _capture_exists(conn: sqlite3.Connection, origin: str, origin_id: str) -> bool:
    return (
        conn.execute(
            "SELECT 1 FROM captures WHERE origin_namespace = ? AND origin_id = ? LIMIT 1;",
            (origin, origin_id),
        ).fetchone()
        is not None
    )


def _capture_store_title(tweet_id: str) -> str | None:
    """Return the curated title the capture store holds for an X post, if any.

    The capture store is where bookmarks actually land, and its title is either a
    curated label ("AI policy without cartel capture") or a mechanical
    derivation. Edward's own derivation runs on the post text alone and produces
    a truncated first line ("What Is To Be Done?"), so inheriting the store's
    title makes both tools agree and gives the reader something that says what
    the item is about.

    A record whose title is still an auto-derivation of its own fields is not
    treated as curated, so a mechanical title is never preferred over Edward's
    (which has the post text available and re-derives on repair).

    Absent, unreadable, or malformed records return None, and the caller falls
    back to deriving from the post.
    """
    record = _capture_store_record(tweet_id)
    if not record:
        return None
    title = (record.get("title") or "").strip()
    if not title:
        return None
    # A numeric-only title is the tweet id the enrichment step stored, not a title.
    if title.isdigit():
        return None
    return title


def _capture_store_record(tweet_id: str) -> dict[str, Any] | None:
    """Read one capture-store record by tweet id, or None."""
    path = _capture_store_path() / "x-bookmarks" / f"{tweet_id}.json"
    try:
        with open(path, encoding="utf-8") as handle:
            loaded = json.load(handle)
    except (FileNotFoundError, NotADirectoryError, PermissionError, OSError):
        return None
    except (ValueError, UnicodeDecodeError):
        return None
    return loaded if isinstance(loaded, dict) else None


def _capture_store_path() -> Path:
    """Locate the capture store, honouring an override for tests and other hosts."""
    override = os.environ.get("EDWARD_CAPTURE_STORE")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".nathan.local" / "captures" / "source"


def _capture_store_title_is_curated(tweet_id: str) -> bool:
    """True when the capture store's title is worth preferring over derivation.

    ``derive_title`` in the capture store is pure: re-running it on the record's
    own fields reproduces an auto title exactly, while a title a human or the
    agent wrote does not reproduce. That makes the distinction checkable without
    a provenance column.
    """
    record = _capture_store_record(tweet_id)
    if not record:
        return False
    title = (record.get("title") or "").strip()
    if not title or title.isdigit():
        return False
    module = _capture_common_module()
    if module is None:
        # Without the shared module the distinction cannot be drawn, so treat a
        # present title as curated rather than discarding the store's work.
        return True
    try:
        derived = (module.derive_title(record) or "").strip()
    except Exception:
        return True
    return title != derived


def _capture_common_module() -> Any | None:
    """Load the capture store's shared title/classify module, or None."""
    path = _capture_store_path().parent / "scripts" / "capture_common.py"
    if not path.is_file():
        return None
    try:
        spec = importlib.util.spec_from_file_location("edward_capture_common", path)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception:
        return None
    return module


def sync_x_bookmarks(
    database: Database,
    blob_store: BlobStore,
    *,
    limit: int | None = 25,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Import new Birdclaw bookmarks, use xurl/bird when available, and queue linked pages."""
    source_rows = _read_birdclaw_bookmarks()
    with database.connection() as conn:
        existing_ids = {
            row[0]
            for row in conn.execute(
                "SELECT origin_id FROM captures WHERE origin_namespace = 'x' AND origin_id IS NOT NULL;"
            ).fetchall()
        }
    new_rows = [row for row in source_rows if str(row["tweet_id"]) not in existing_ids]
    selected = new_rows if limit is None else new_rows[:limit]
    result: dict[str, Any] = {
        "source": "birdclaw",
        "available": True,
        "bookmarks_found": len(source_rows),
        "already_imported": len(source_rows) - len(new_rows),
        "eligible": len(new_rows),
        "selected": len(selected),
        "captured": 0,
        "linked_urls": 0,
        "media_attached": 0,
        "reader_errors": [],
        "dry_run": dry_run,
        "complete_scan": True,
    }
    if dry_run:
        return result

    run_id = generate_id("run")
    for row in selected:
        tweet_id = str(row["tweet_id"])
        archived_text = (row.get("text") or "").strip()
        archived_entities = _decode_json_column(row.get("entities_json"), {})
        archived_media = _decode_json_column(row.get("media_json"), [])
        raw_archive = {
            "tweet_id": tweet_id,
            "account_id": row.get("account_id"),
            "source": row.get("source"),
            "collection": _decode_json_column(row.get("collection_json"), {}),
            "collected_at": row.get("collected_at"),
            "author_handle": row.get("author_handle"),
            "author_name": row.get("author_name"),
            "published_at": row.get("published_at"),
            "text": archived_text,
            "entities": archived_entities,
            "media": archived_media,
            "deleted_at": row.get("deleted_at"),
        }

        post_url = f"https://x.com/i/status/{tweet_id}"
        thread_chain = _unroll_x_thread(tweet_id)
        is_thread = len(thread_chain) >= 2
        reader_error = None
        if is_thread:
            reader_payload = thread_chain
            reader_name = "bird-thread"
            text = _format_thread_text(thread_chain)
            raw_archive["is_thread"] = True
            raw_archive["thread_length"] = len(thread_chain)
            raw_archive["thread_tweet_ids"] = [str(t.get("id")) for t in thread_chain]
            media_urls, links = _collect_thread_media_and_links(thread_chain, text)
        else:
            reader_payload, reader_name, reader_error = _x_reader_payload(tweet_id)
            text = _tweet_text(reader_payload) if reader_payload else None
            if not text:
                text = archived_text
            if not text and is_tool_available("summarize"):
                post_url = f"https://x.com/i/status/{tweet_id}"
                extracted = extract_content(post_url)
                text = extracted.clean_text
                if text:
                    reader_name = "summarize"
                    reader_error = None
            if not text:
                result["reader_errors"].append(
                    {"tweet_id": tweet_id, "error": "No readable post text is available"}
                )
                continue

            entities = archived_entities
            media_urls = _tweet_media_urls(archived_media, entities, text)
            media_set = set(media_urls)
            links = [
                url
                for url in _tweet_urls(entities, text)
                if f"/status/{tweet_id}" not in url and url not in media_set
            ]

        payload_bytes = json.dumps(
            {"birdclaw": raw_archive, "reader": reader_payload, "reader_name": reader_name},
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")

        now_iso = datetime.datetime.now(datetime.UTC).isoformat()
        with database.transaction() as conn:
            if _capture_exists(conn, "x", tweet_id):
                result["already_imported"] += 1
                continue
            captured = capture_item(
                conn,
                CaptureInput(
                    url=post_url,
                    # The capture records how the post was saved; the linked
                    # resource owns the post text and its embedding.
                    text=None,
                    origin_namespace="x",
                    origin_id=tweet_id,
                    collection_channel="birdclaw",
                    collector="edward-birdclaw-adapter",
                    collector_run_id=run_id,
                    acquisition_method="birdclaw-sqlite",
                ),
            )
            if not captured["resource_id"]:
                raise SourceAdapterError(f"Edward did not create an X resource for post {tweet_id}")

            content_hash, _blob_path = blob_store.store_bytes(payload_bytes)
            conn.execute(
                """
                INSERT INTO source_snapshots (
                    id, resource_id, content_hash, headers_json, blob_path, size_bytes, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    generate_id("snp"),
                    captured["resource_id"],
                    content_hash,
                    json.dumps({"source": "birdclaw", "reader": reader_name}),
                    f"{content_hash[:2]}/{content_hash}",
                    len(payload_bytes),
                    now_iso,
                ),
            )
            # The post text IS the source, so the title is derived from it. The
            # author belongs in its own column — using it as the title makes
            # every retrieved item read as a person's name rather than the post.
            #
            # The capture store is where the bookmark actually landed, and its
            # title may be a curated label ("AI policy without cartel capture")
            # where derivation yields a bare first line ("What Is To Be Done?").
            # Prefer the curated one; fall back to deriving from the post.
            post_title = None
            if _capture_store_title_is_curated(tweet_id):
                post_title = _capture_store_title(tweet_id)
            if not post_title:
                post_title = derive_post_title(
                    text, fallback=row.get("author_name") or row.get("author_handle") or post_url
                )
            conn.execute(
                "UPDATE resources SET title = ?, author = ?, published_at = ? WHERE id = ?;",
                (
                    post_title,
                    row.get("author_handle"),
                    row.get("published_at"),
                    captured["resource_id"],
                ),
            )
            cached_content = get_cached_content(conn, captured["resource_id"])
            if cached_content:
                # Keep richer content already associated with the post URL.
                reindex_object_document(conn, "resource", captured["resource_id"])
            else:
                store_resource_content(
                    conn,
                    captured["resource_id"],
                    text,
                    extractor=reader_name or "birdclaw",
                    extractor_version="source-adapter-v1",
                    title=post_title,
                    capture_id=captured["capture_id"],
                )
                conn.execute(
                    """
                    UPDATE processing_jobs
                    SET status = 'completed', completed_at = ?, updated_at = ?
                    WHERE job_key = ? AND stage = 'resource-fetch' AND status = 'pending';
                    """,
                    (now_iso, now_iso, make_job_key("fetch", captured["resource_id"])),
                )
                classify_job_id = generate_id("job")
                classify_job_key = make_job_key("classify", captured["resource_id"])
                conn.execute(
                    """
                    INSERT INTO processing_jobs (
                        id, job_key, capture_id, resource_id, stage, status,
                        available_at, attempts, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, 'classify', 'pending', ?, 0, ?, ?)
                    ON CONFLICT(job_key) DO UPDATE SET
                        capture_id = COALESCE(excluded.capture_id, processing_jobs.capture_id),
                        updated_at = excluded.updated_at;
                    """,
                    (
                        classify_job_id,
                        classify_job_key,
                        captured["capture_id"],
                        captured["resource_id"],
                        now_iso,
                        now_iso,
                        now_iso,
                    ),
                )
            for url in links:
                result["linked_urls"] += int(
                    _queue_link_resource(conn, captured["capture_id"], url, now_iso)
                )
            result["media_attached"] += _capture_tweet_media(
                conn,
                blob_store,
                resource_id=captured["resource_id"],
                media_urls=media_urls,
                now_iso=now_iso,
            )
        if reader_error:
            result["reader_errors"].append({"tweet_id": tweet_id, "error": reader_error})
        result["captured"] += 1

    return result


def _addresses(value: Any) -> set[str]:
    if isinstance(value, list):
        found: set[str] = set()
        for item in value:
            found.update(_addresses(item))
        return found
    if isinstance(value, dict):
        return _addresses(" ".join(str(v) for v in value.values()))
    if isinstance(value, str):
        return {address.lower() for address in EMAIL_PATTERN.findall(value)}
    return set()


def _headers(message: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    raw = message.get("headers")
    if isinstance(raw, dict):
        result.update({str(key).lower(): value for key, value in raw.items()})
    elif isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict) and item.get("name"):
                result[str(item["name"]).lower()] = item.get("value", "")
    for key in ("from", "to", "cc", "bcc", "subject", "date", "message-id"):
        if key not in result:
            for variant in (key, key.replace("-", "_"), key.title()):
                if variant in message:
                    result[key] = message[variant]
                    break
    return result


def _gmail_text_body(message: dict[str, Any]) -> tuple[str, str | None]:
    for key in ("textBody", "text_body", "plainText", "plain_text", "bodyText", "body_text"):
        value = message.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip(), None
    body = message.get("body")
    if isinstance(body, str):
        return body.strip(), None
    if isinstance(body, dict):
        for key in ("text", "plain", "text/plain", "content"):
            value = body.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip(), None
        for key in ("html", "text/html"):
            value = body.get(key)
            if isinstance(value, str) and value.strip():
                _title, cleaned = clean_html_simple(value)
                return cleaned, value

    html_body = message.get("htmlBody") or message.get("html_body")
    if isinstance(html_body, str) and html_body.strip():
        _title, cleaned = clean_html_simple(html_body)
        return cleaned, html_body

    parts = message.get("parts") or message.get("payload")
    if isinstance(parts, (dict, list)):
        text = _first_text(parts, ("text/plain", "textBody", "plainText", "bodyText"))
        if text:
            return text, None
        html_part = _first_text(parts, ("text/html", "htmlBody", "html"))
        if html_part:
            _title, cleaned = clean_html_simple(html_part)
            return cleaned, html_part
    snippet = message.get("snippet")
    return (str(snippet).strip(), None) if snippet else ("", None)


def _attachment_metadata(message: dict[str, Any]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for item in _dicts_with_keys(
        message, {"filename", "fileName", "attachmentId", "attachment_id"}
    ):
        name = item.get("filename") or item.get("fileName")
        attachment_id = item.get("attachmentId") or item.get("attachment_id")
        if name or attachment_id:
            output.append(
                {
                    "file_name": name or "attachment",
                    "attachment_id": attachment_id,
                    "mime_type": item.get("mimeType")
                    or item.get("mime_type")
                    or "application/octet-stream",
                    "size_bytes": item.get("size") or item.get("size_bytes"),
                }
            )
    return output


def _gmail_message_date(
    message: dict[str, Any], headers: dict[str, Any]
) -> datetime.datetime | None:
    value = headers.get("date") or message.get("internalDate") or message.get("internal_date")
    if not value:
        return None
    if isinstance(value, (int, float)) or (isinstance(value, str) and value.isdigit()):
        return datetime.datetime.fromtimestamp(int(value) / 1000, tz=datetime.UTC)
    if isinstance(value, str):
        try:
            return datetime.datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(
                datetime.UTC
            )
        except ValueError:
            from email.utils import parsedate_to_datetime

            try:
                parsed = parsedate_to_datetime(value)
                return (
                    parsed.astimezone(datetime.UTC)
                    if parsed.tzinfo
                    else parsed.replace(tzinfo=datetime.UTC)
                )
            except (TypeError, ValueError):
                return None
    return None


def _gmail_thread_ids(payload: Any) -> list[str]:
    found: list[str] = []
    for item in _dicts_with_keys(payload, {"threadId", "thread_id", "id"}):
        thread_id = item.get("threadId") or item.get("thread_id")
        if not thread_id and "messages" not in item:
            thread_id = item.get("id")
        if isinstance(thread_id, str) and thread_id not in found:
            found.append(thread_id)
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, str) and item not in found:
                found.append(item)
    return found


def _gmail_messages(payload: Any) -> list[dict[str, Any]]:
    for item in _dicts_with_keys(payload, {"messages"}):
        messages = item.get("messages")
        if isinstance(messages, list):
            return [message for message in messages if isinstance(message, dict)]
    if isinstance(payload, list):
        return [message for message in payload if isinstance(message, dict)]
    if isinstance(payload, dict) and ("id" in payload or "messageId" in payload):
        return [payload]
    return []


def _gmail_search_query(database: Database) -> str:
    with database.connection() as conn:
        row = conn.execute(
            "SELECT cursor_value FROM source_cursors WHERE source = 'gmail-self';"
        ).fetchone()
    if not row:
        return SELF_EMAIL_QUERY
    try:
        checkpoint = datetime.datetime.fromisoformat(row["cursor_value"])
    except ValueError:
        return SELF_EMAIL_QUERY
    overlap = checkpoint - datetime.timedelta(days=7)
    return f"{SELF_EMAIL_QUERY} after:{overlap:%Y/%m/%d}"


def sync_self_sent_gmail(
    database: Database,
    blob_store: BlobStore,
    *,
    limit: int | None = None,
    dry_run: bool = False,
    download_attachments: bool = False,
) -> dict[str, Any]:
    """Read self-sent Gmail threads through gog, verify message headers, and capture links."""
    if not is_tool_available("gog"):
        raise SourceAdapterError("gog is not installed")

    query = _gmail_search_query(database)
    search_args = ["gog", "gmail", "search", query, "--json", "--readonly", "--no-input"]
    if limit is None:
        search_args.append("--all")
    else:
        search_args.extend(["--max", str(limit)])
    search_payload = _run_json_tool(search_args, "gog gmail search")
    thread_ids = _gmail_thread_ids(search_payload)
    if limit is not None:
        thread_ids = thread_ids[:limit]

    result: dict[str, Any] = {
        "source": "gmail",
        "query": query,
        "threads_found": len(thread_ids),
        "captured": 0,
        "already_imported": 0,
        "rejected_not_self_sent": 0,
        "linked_urls": 0,
        "attachments_saved": 0,
        "dry_run": dry_run,
        "complete_scan": limit is None,
    }
    if dry_run:
        return result

    run_id = generate_id("run")
    latest_date: datetime.datetime | None = None
    for thread_id in thread_ids:
        thread_args = [
            "gog",
            "gmail",
            "thread",
            "get",
            thread_id,
            "--json",
            "--readonly",
            "--no-input",
            "--full",
        ]
        download_dir: tempfile.TemporaryDirectory[str] | None = None
        if download_attachments:
            download_dir = tempfile.TemporaryDirectory(prefix="edward-gmail-")
            thread_args.extend(["--download", "--out-dir", download_dir.name])
        try:
            thread_payload = _run_json_tool(thread_args, "gog gmail thread get")
        except Exception:
            if download_dir:
                download_dir.cleanup()
            raise
        messages = _gmail_messages(thread_payload)
        if not messages:
            if download_dir:
                download_dir.cleanup()
            raise SourceAdapterError(f"gog returned no messages for Gmail thread {thread_id}")

        for message in messages:
            headers = _headers(message)
            sender_addresses = _addresses(headers.get("from"))
            recipient_addresses = set().union(
                *(_addresses(headers.get(key)) for key in ("to", "cc", "bcc"))
            )
            if not sender_addresses or not sender_addresses.intersection(recipient_addresses):
                result["rejected_not_self_sent"] += 1
                continue

            message_date = _gmail_message_date(message, headers)
            if message_date and (latest_date is None or message_date > latest_date):
                latest_date = message_date

            message_id = str(
                message.get("id") or message.get("messageId") or message.get("message_id") or ""
            )
            if not message_id:
                if download_dir:
                    download_dir.cleanup()
                raise SourceAdapterError(
                    f"Gmail thread {thread_id} contains a message without an ID"
                )

            body, html_body = _gmail_text_body(message)
            subject = str(headers.get("subject") or "(no subject)")
            sender = str(headers.get("from") or "")
            recipients = str(headers.get("to") or "")
            attachment_items = _attachment_metadata(message)
            attachment_lines = [
                f"- {item['file_name']} ({item['mime_type']}; "
                f"{item['size_bytes'] or 'size unknown'} bytes; "
                f"provider ID {item['attachment_id'] or 'unavailable'}; "
                f"{'retrieval attempted' if download_attachments else 'not downloaded'})"
                for item in attachment_items
            ]
            raw_content = (
                f"Subject: {subject}\nFrom: {sender}\nTo: {recipients}\n"
                f"Date: {headers.get('date') or (message_date.isoformat() if message_date else '')}\n"
                f"Thread ID: {thread_id}\nMessage ID: {message_id}\n\n{body}"
            )
            if attachment_lines:
                raw_content += "\n\nAttachments:\n" + "\n".join(attachment_lines)
            if html_body:
                raw_content += (
                    "\n\n[HTML body preserved in sanitized form]\n"
                    + clean_html_simple(html_body)[1]
                )

            with database.transaction() as conn:
                if _capture_exists(conn, "gmail", message_id):
                    result["already_imported"] += 1
                    continue
                captured = capture_item(
                    conn,
                    CaptureInput(
                        text=raw_content,
                        note=body or subject,
                        origin_namespace="gmail",
                        origin_id=message_id,
                        collection_channel="gog",
                        collector="edward-gog-adapter",
                        collector_run_id=run_id,
                        acquisition_method="gog-readonly",
                    ),
                )
                now_iso = datetime.datetime.now(datetime.UTC).isoformat()
                links = _tweet_urls({}, raw_content)
                for url in links:
                    result["linked_urls"] += int(
                        _queue_link_resource(conn, captured["capture_id"], url, now_iso)
                    )

                if download_dir and attachment_items:
                    result["attachments_saved"] += _save_downloaded_gmail_attachments(
                        download_dir.name,
                        attachment_items,
                        captured["capture_id"],
                        blob_store,
                        conn,
                    )
            result["captured"] += 1

        if download_dir:
            download_dir.cleanup()

    if result["complete_scan"] and latest_date:
        with database.transaction() as conn:
            conn.execute(
                """
                INSERT INTO source_cursors (source, cursor_value, updated_at)
                VALUES ('gmail-self', ?, ?)
                ON CONFLICT(source) DO UPDATE SET
                    cursor_value = excluded.cursor_value,
                    updated_at = excluded.updated_at;
                """,
                (latest_date.isoformat(), datetime.datetime.now(datetime.UTC).isoformat()),
            )
    return result


def _save_downloaded_gmail_attachments(
    download_dir: str,
    attachment_items: list[dict[str, Any]],
    capture_id: str,
    blob_store: BlobStore,
    conn: sqlite3.Connection,
) -> int:
    """Save files downloaded by gog when the caller explicitly opted in."""
    files = [path for path in Path(download_dir).rglob("*") if path.is_file()]
    remaining = {path: path.name.casefold() for path in files}
    saved = 0
    for item in attachment_items:
        name = str(item["file_name"])
        match = next((path for path, base in remaining.items() if base == name.casefold()), None)
        if match is None:
            continue
        content_hash, _path = blob_store.store_file(match)
        attachment_id = generate_id("att")
        mime_type = item["mime_type"] or mimetypes.guess_type(name)[0] or "application/octet-stream"
        now_iso = datetime.datetime.now(datetime.UTC).isoformat()
        conn.execute(
            """
            INSERT INTO attachments (
                id, object_type, object_id, file_name, mime_type, content_hash, size_bytes,
                blob_path, created_at
            ) VALUES (?, 'capture', ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                attachment_id,
                capture_id,
                name,
                mime_type,
                content_hash,
                match.stat().st_size,
                f"{content_hash[:2]}/{content_hash}",
                now_iso,
            ),
        )
        if is_pdf_attachment(name, mime_type):
            enqueue_pdf_extraction(conn, capture_id, attachment_id, now_iso=now_iso)
        remaining.pop(match)
        saved += 1
    return saved


def repair_x_threads(
    database: Database,
    blob_store: BlobStore,
    *,
    limit: int | None = None,
    dry_run: bool = False,
    force: bool = False,
    on_progress: Any = None,
) -> dict[str, Any]:
    """Scan existing X bookmarks, unroll threads using bird thread, and update resources."""
    from edward.services.embed import store_resource_chunks

    if not is_tool_available("bird"):
        raise SourceAdapterError("bird CLI is required for thread unrolling")

    with database.connection() as conn:
        rows = conn.execute(
            """
            SELECT c.id AS capture_id, c.origin_id AS tweet_id, cr.resource_id,
                   r.title, rc.id AS content_id, rc.clean_text, rc.char_count, rc.extractor
            FROM captures c
            JOIN capture_resources cr ON cr.capture_id = c.id AND cr.relationship_type = 'primary'
            JOIN resources r ON r.id = cr.resource_id
            LEFT JOIN resource_contents rc ON rc.id = (
                SELECT id FROM resource_contents
                WHERE resource_id = r.id
                ORDER BY created_at DESC LIMIT 1
            )
            WHERE c.origin_namespace = 'x' AND c.origin_id IS NOT NULL AND c.is_deleted = 0
            ORDER BY c.created_at DESC;
            """
        ).fetchall()

    result: dict[str, Any] = {
        "captures_checked": 0,
        "threads_found": 0,
        "threads_expanded": 0,
        "media_attached": 0,
        "linked_urls_queued": 0,
        "dry_run": dry_run,
        "details": [],
    }

    selected = rows if limit is None else rows[:limit]
    total_to_check = len(selected)
    seen_tweet_ids: set[str] = set()

    for idx, row in enumerate(selected, start=1):
        tweet_id = str(row["tweet_id"])
        if tweet_id in seen_tweet_ids:
            continue
        seen_tweet_ids.add(tweet_id)
        result["captures_checked"] += 1

        if not force and row["extractor"] == "bird-thread":
            continue

        resource_id = row["resource_id"]
        current_text = row["clean_text"] or ""

        if on_progress:
            on_progress(idx, total_to_check, f"Checking post {tweet_id}")

        chain = _unroll_x_thread(tweet_id)
        if len(chain) < 2:
            continue

        result["threads_found"] += 1
        new_text = _format_thread_text(chain)
        if len(new_text) <= len(current_text) and "\n\n---\n\n" in current_text:
            continue

        new_title = derive_post_title(
            new_text, fallback=row["title"] or f"https://x.com/i/status/{tweet_id}"
        )
        thread_media, thread_links = _collect_thread_media_and_links(chain, new_text)

        if on_progress:
            on_progress(idx, total_to_check, f"Found thread for {tweet_id} ({len(chain)} tweets)")

        result["details"].append(
            {
                "tweet_id": tweet_id,
                "resource_id": resource_id,
                "old_title": row["title"],
                "new_title": new_title,
                "thread_length": len(chain),
                "old_char_count": len(current_text),
                "new_char_count": len(new_text),
                "new_media_count": len(thread_media),
            }
        )

        if dry_run:
            continue

        now_iso = datetime.datetime.now(datetime.UTC).isoformat()
        with database.transaction() as conn:
            content_id, _ = store_resource_content(
                conn,
                resource_id=resource_id,
                clean_text=new_text,
                extractor="bird-thread",
                extractor_version="source-adapter-v1",
                title=new_title,
                capture_id=row["capture_id"],
                extraction_note=f"Unrolled {len(chain)}-tweet thread",
            )
            store_resource_chunks(
                conn,
                resource_id=resource_id,
                resource_content_id=content_id,
                text=new_text,
            )
            if new_title:
                conn.execute(
                    "UPDATE resources SET title = ?, updated_at = ? WHERE id = ?;",
                    (new_title, now_iso, resource_id),
                )

            if thread_media:
                saved_media = _capture_tweet_media(
                    conn,
                    blob_store,
                    resource_id=resource_id,
                    media_urls=thread_media,
                    now_iso=now_iso,
                )
                result["media_attached"] += saved_media

            for url in thread_links:
                result["linked_urls_queued"] += int(
                    _queue_link_resource(conn, row["capture_id"], url, now_iso)
                )

        result["threads_expanded"] += 1

    return result
