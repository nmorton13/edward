"""Migration of legacy media-asset resources into post attachments.

Before this migration, every image URL in a bookmarked post's entities became
its own resource: 243 standalone "sources" whose entire content was a JPEG that
no reader or classifier could use. The migration must rehome those bytes onto
the post that carried them and retire the bogus rows — without losing an image.
"""

import json

from edward.services.lifecycle import migrate_media_resources_to_attachments

CAPTURE_ID = "cap_test"
POST_RESOURCE = "res_post"
MEDIA_RESOURCE = "res_media"
MEDIA_URL = "https://pbs.twimg.com/media/HSwAPAfbMAAa1fq.jpg"


def _seed_post_with_media_attachment(conn, blob_store, *, store_blob: bool = True):
    """A post resource, plus a media resource wrongly created from its image."""
    conn.execute(
        """
        INSERT INTO captures (
            id, origin_namespace, origin_id, collection_channel, collector,
            acquisition_method, retrieved_at, raw_content, created_at, updated_at
        ) VALUES (?, 'x', 'tweet-1', 'birdclaw', 'edward-birdclaw-adapter',
                  'birdclaw-sqlite', '2026-01-01', 'post text', '2026-01-01', '2026-01-01');
        """,
        (CAPTURE_ID,),
    )
    for rid, title, url in (
        (POST_RESOURCE, "built a malware scanner using laya", "https://x.com/i/status/1"),
        (MEDIA_RESOURCE, MEDIA_URL, MEDIA_URL),
    ):
        conn.execute(
            """
            INSERT INTO resources (id, canonical_url, url_hash, identity_key, title,
                                   created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, '2026-01-01', '2026-01-01');
            """,
            (rid, url, f"h_{rid}", f"k_{rid}", title),
        )
    conn.execute(
        """
        INSERT INTO capture_resources (capture_id, resource_id, relationship_type, created_at)
        VALUES (?, ?, 'primary', '2026-01-01'), (?, ?, 'referenced', '2026-01-01');
        """,
        (CAPTURE_ID, POST_RESOURCE, CAPTURE_ID, MEDIA_RESOURCE),
    )

    content_hash = None
    if store_blob:
        jpeg = b"\xff\xd8\xff\xe0" + b"\x00" * 512
        content_hash, _path = blob_store.store_bytes(jpeg)
        conn.execute(
            """
            INSERT INTO source_snapshots (
                id, resource_id, content_hash, headers_json, blob_path, size_bytes, created_at
            ) VALUES ('snp_media', ?, ?, '{}', ?, ?, '2026-01-01');
            """,
            (MEDIA_RESOURCE, content_hash, f"{content_hash[:2]}/{content_hash}", len(jpeg)),
        )
    return content_hash


def _counts(db):
    with db.connection() as conn:
        return {
            "resources": conn.execute("SELECT count(*) c FROM resources;").fetchone()["c"],
            "attachments": conn.execute("SELECT count(*) c FROM attachments;").fetchone()["c"],
            "ocr_jobs": conn.execute(
                "SELECT count(*) c FROM processing_jobs WHERE stage='attachment-ocr';"
            ).fetchone()["c"],
        }


def test_media_resource_becomes_an_attachment_of_the_post(test_db, test_blob_store):
    with test_db.transaction() as conn:
        _seed_post_with_media_attachment(conn, test_blob_store, store_blob=True)

    result = migrate_media_resources_to_attachments(test_db, test_blob_store)

    assert result["converted"] == 1
    assert result["failed"] == []
    counts = _counts(test_db)
    # The bogus resource is gone; the real post remains.
    assert counts["resources"] == 1
    assert counts["attachments"] == 1
    with test_db.connection() as conn:
        attachment = conn.execute("SELECT * FROM attachments;").fetchone()
    assert attachment["object_type"] == "resource"
    assert attachment["object_id"] == POST_RESOURCE
    assert attachment["file_name"] == "HSwAPAfbMAAa1fq.jpg"


def test_migration_queues_ocr_for_the_rehomed_image(test_db, test_blob_store):
    with test_db.transaction() as conn:
        _seed_post_with_media_attachment(conn, test_blob_store, store_blob=True)

    migrate_media_resources_to_attachments(test_db, test_blob_store)

    assert _counts(test_db)["ocr_jobs"] == 1


def test_image_bytes_survive_the_migration(test_db, test_blob_store):
    """The attachment must reference a blob that is still readable."""
    with test_db.transaction() as conn:
        _seed_post_with_media_attachment(conn, test_blob_store, store_blob=True)

    migrate_media_resources_to_attachments(test_db, test_blob_store)

    with test_db.connection() as conn:
        attachment = conn.execute("SELECT content_hash FROM attachments;").fetchone()
    assert test_blob_store.exists(attachment["content_hash"])
    data = test_blob_store.read_bytes(attachment["content_hash"])
    assert data[:2] == b"\xff\xd8"


def test_migration_does_not_touch_ordinary_resources(test_db, test_blob_store):
    """A genuinely interesting link must not be treated as media."""
    with test_db.transaction() as conn:
        conn.execute(
            """
            INSERT INTO resources (id, canonical_url, url_hash, identity_key, title,
                                   created_at, updated_at)
            VALUES ('res_article', 'https://example.com/data-centers', 'h1', 'k1',
                    'Have Data Centers Raised Your Electric Bill?', '2026-01-01', '2026-01-01');
            """
        )

    result = migrate_media_resources_to_attachments(test_db, test_blob_store)

    assert result["converted"] == 0
    assert _counts(test_db)["resources"] == 1


def test_migration_is_idempotent(test_db, test_blob_store):
    with test_db.transaction() as conn:
        _seed_post_with_media_attachment(conn, test_blob_store, store_blob=True)

    first = migrate_media_resources_to_attachments(test_db, test_blob_store)
    second = migrate_media_resources_to_attachments(test_db, test_blob_store)

    assert first["converted"] == 1
    assert second["converted"] == 0
    assert _counts(test_db)["attachments"] == 1
    assert _counts(test_db)["ocr_jobs"] == 1


def test_missing_blob_still_retires_the_bogus_resource(test_db, test_blob_store):
    """No bytes to preserve means nothing is lost by removing the empty row."""
    with test_db.transaction() as conn:
        _seed_post_with_media_attachment(conn, test_blob_store, store_blob=False)
        conn.execute(
            """
            INSERT INTO source_snapshots (
                id, resource_id, content_hash, headers_json, blob_path, size_bytes, created_at
            ) VALUES ('snp_gone', ?, ?, '{}', 'ab/absent', 100, '2026-01-01');
            """,
            (MEDIA_RESOURCE, "ab" + "0" * 62),
        )

    result = migrate_media_resources_to_attachments(test_db, test_blob_store)

    assert result["converted"] == 1
    assert result["failed"] == []
    assert _counts(test_db)["resources"] == 1


def test_migration_removes_the_media_row_from_search(test_db, test_blob_store):
    """The retired resource must stop appearing in search results."""
    with test_db.transaction() as conn:
        _seed_post_with_media_attachment(conn, test_blob_store, store_blob=False)
        conn.execute(
            """
            INSERT INTO search_documents (object_type, object_id, title, body, labels, entities)
            VALUES ('resource', ?, ?, ?, '', '');
            """,
            (MEDIA_RESOURCE, MEDIA_URL, MEDIA_URL),
        )

    migrate_media_resources_to_attachments(test_db, test_blob_store)

    with test_db.connection() as conn:
        remaining = conn.execute(
            "SELECT count(*) c FROM search_documents WHERE object_id = ?;", (MEDIA_RESOURCE,)
        ).fetchone()["c"]
    assert remaining == 0


def test_result_reports_failures_without_aborting_the_run(test_db, test_blob_store):
    """One bad row must not stop the rest of the migration."""
    with test_db.transaction() as conn:
        _seed_post_with_media_attachment(conn, test_blob_store, store_blob=False)

    result = migrate_media_resources_to_attachments(test_db, test_blob_store)

    assert set(result.keys()) >= {"converted", "skipped", "failed", "converted_ids"}
    assert isinstance(result["converted_ids"], list)
    assert json.dumps(result)  # must be JSON-serializable for --json output
