from edward.blobs import BlobStore
from edward.db import Database
from edward.models import CaptureInput
from edward.services.capture import capture_item
from edward.services.lifecycle import (
    add_annotation,
    add_intent,
    get_annotations,
    prune_unreferenced_blobs,
    purge_object,
    remove_intent,
    restore,
    set_review_state,
    soft_delete,
)
from edward.services.search import search_lexical


def test_review_state_transitions(test_db: Database):
    with test_db.transaction() as conn:
        res = capture_item(conn, CaptureInput(text="Sample knowledge"))
        cap_id = res["capture_id"]

        set_review_state(conn, "capture", cap_id, "approved", actor="nathan")

    with test_db.connection() as conn:
        row = conn.execute("SELECT review_state FROM captures WHERE id = ?;", (cap_id,)).fetchone()
        assert row["review_state"] == "approved"


def test_soft_delete_and_restore(test_db: Database):
    with test_db.transaction() as conn:
        res = capture_item(conn, CaptureInput(text="Deleteable knowledge"))
        cap_id = res["capture_id"]

    with test_db.connection() as conn:
        # Document should be searchable initially
        assert search_lexical(conn, "Deleteable").count >= 1

    # Soft delete
    with test_db.transaction() as conn:
        soft_delete(conn, "capture", cap_id)

    with test_db.connection() as conn:
        # Not searchable after soft delete
        assert search_lexical(conn, "Deleteable").count == 0
        row = conn.execute(
            "SELECT is_deleted, deleted_at FROM captures WHERE id = ?;", (cap_id,)
        ).fetchone()
        assert row["is_deleted"] == 1
        assert row["deleted_at"] is not None

    # Restore
    with test_db.transaction() as conn:
        restore(conn, "capture", cap_id)

    with test_db.connection() as conn:
        # Searchable again
        assert search_lexical(conn, "Deleteable").count >= 1
        row = conn.execute("SELECT is_deleted FROM captures WHERE id = ?;", (cap_id,)).fetchone()
        assert row["is_deleted"] == 0


def test_intent_addition_and_removal(test_db: Database):
    with test_db.transaction() as conn:
        res = capture_item(
            conn, CaptureInput(url="https://example.com/essay", note="Essay idea on AI memory")
        )
        res_id = res["resource_id"]

        add_intent(conn, "resource", res_id, "essay-seed")

    with test_db.connection() as conn:
        row = conn.execute(
            "SELECT is_active FROM intents WHERE object_id = ? AND intent = 'essay-seed';",
            (res_id,),
        ).fetchone()
        assert row["is_active"] == 1

    # Remove intent
    with test_db.transaction() as conn:
        remove_intent(conn, "resource", res_id, "essay-seed")

    with test_db.connection() as conn:
        # Historical record remains with is_active = 0
        row = conn.execute(
            "SELECT is_active FROM intents WHERE object_id = ? AND intent = 'essay-seed';",
            (res_id,),
        ).fetchone()
        assert row["is_active"] == 0


def test_permanent_purge(test_db: Database):
    with test_db.transaction() as conn:
        res = capture_item(conn, CaptureInput(text="Purgeable entry"))
        cap_id = res["capture_id"]

    with test_db.transaction() as conn:
        purge_object(conn, "capture", cap_id)

    with test_db.connection() as conn:
        assert conn.execute("SELECT 1 FROM captures WHERE id = ?;", (cap_id,)).fetchone() is None
        assert search_lexical(conn, "Purgeable").count == 0


def test_annotations_append_and_audit(test_db: Database):
    with test_db.transaction() as conn:
        res = capture_item(conn, CaptureInput(text="Base content for annotation"))
        cap_id = res["capture_id"]

        # Append two annotations
        ann1 = add_annotation(
            conn, "capture", cap_id, "First insight on performance", author="alice"
        )
        ann2 = add_annotation(conn, "capture", cap_id, "Second insight on scaling", author="bob")

    with test_db.connection() as conn:
        # Retrieve annotations - should have both in order
        anns = get_annotations(conn, "capture", cap_id)
        assert len(anns) == 2
        assert anns[0].id == ann1.id
        assert anns[0].content == "First insight on performance"
        assert anns[0].author == "alice"
        assert anns[1].id == ann2.id
        assert anns[1].content == "Second insight on scaling"
        assert anns[1].author == "bob"

        # Search projection should include content from annotations
        s_res1 = search_lexical(conn, "performance")
        assert any(r.id == cap_id for r in s_res1.results)

        s_res2 = search_lexical(conn, "scaling")
        assert any(r.id == cap_id for r in s_res2.results)

        # Audit events must be recorded
        events = conn.execute(
            "SELECT event_type, actor FROM audit_events WHERE object_id = ? AND event_type = 'annotation.added' ORDER BY created_at ASC;",
            (cap_id,),
        ).fetchall()
        assert len(events) == 2
        assert events[0]["actor"] == "alice"
        assert events[1]["actor"] == "bob"


def test_prune_unreferenced_blobs(test_db: Database, test_blob_store: BlobStore):
    # Store 2 blobs
    b1_hash, b1_path = test_blob_store.store_bytes(b"Active attachment content")
    b2_hash, b2_path = test_blob_store.store_bytes(b"Orphaned blob content")

    # Reference only b1 in SQLite
    with test_db.transaction() as conn:
        res = capture_item(conn, CaptureInput(text="Item with attachment"))
        conn.execute(
            """
            INSERT INTO attachments (id, object_type, object_id, file_name, mime_type, content_hash, size_bytes, blob_path, created_at)
            VALUES ('att_test', 'capture', ?, 'active.txt', 'text/plain', ?, 24, ?, datetime('now'));
            """,
            (res["capture_id"], b1_hash, str(b1_path)),
        )

    # Prune unreferenced blobs
    pruned = prune_unreferenced_blobs(test_db, test_blob_store)
    assert b2_hash in pruned
    assert b1_hash not in pruned

    # Verify disk state
    assert b1_path.exists()
    assert not b2_path.exists()
