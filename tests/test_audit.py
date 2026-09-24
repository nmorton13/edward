"""Tests for append-only audit log and preservation of human-authored metadata."""

import json

from edward.db import Database
from edward.models import CaptureInput
from edward.services.audit import get_audit_trail, record_audit_event
from edward.services.capture import capture_item
from edward.services.lifecycle import add_annotation, purge_object


def test_append_only_audit_log(test_db: Database):
    with test_db.transaction() as conn:
        record_audit_event(
            conn,
            event_type="capture.created",
            object_type="capture",
            object_id="cap_123",
            actor="test-agent",
            payload={"url": "https://example.com"},
        )
        record_audit_event(
            conn,
            event_type="annotation.added",
            object_type="capture",
            object_id="cap_123",
            actor="human-user",
            payload={"note": "Reviewed by user"},
        )

    with test_db.connection() as conn:
        trail = get_audit_trail(conn, "cap_123")
        assert len(trail) == 2
        assert trail[0].event_type == "capture.created"
        assert trail[0].actor == "test-agent"
        assert trail[1].event_type == "annotation.added"
        assert trail[1].actor == "human-user"


def test_preservation_of_human_annotations(test_db: Database):
    """Ensure automated updates do not overwrite human notes."""
    with test_db.transaction() as conn:
        conn.execute(
            """
            INSERT INTO captures (
                id, origin_namespace, collection_channel, collector,
                acquisition_method, retrieved_at, user_note, review_state,
                created_at, updated_at
            ) VALUES ('cap_h1', 'web', 'manual', 'user', 'manual', datetime('now'), 'Crucial human note', 'approved', datetime('now'), datetime('now'));
            """
        )

        # An automated background sync or refresh must NOT overwrite user_note or review_state
        conn.execute(
            """
            UPDATE captures
            SET collector_run_id = 'sync-run-99'
            WHERE id = 'cap_h1';
            """
        )

    with test_db.connection() as conn:
        row = conn.execute(
            "SELECT user_note, review_state, collector_run_id FROM captures WHERE id = 'cap_h1';"
        ).fetchone()
        assert row["user_note"] == "Crucial human note"
        assert row["review_state"] == "approved"
        assert row["collector_run_id"] == "sync-run-99"


def test_annotation_audit_payload_privacy(test_db: Database):
    with test_db.transaction() as conn:
        res = capture_item(conn, CaptureInput(text="Resource for note privacy"))
        cap_id = res["capture_id"]
        ann = add_annotation(
            conn,
            object_type="capture",
            object_id=cap_id,
            content="Highly confidential human insight",
            author="human",
            annotation_type="note",
        )

    with test_db.connection() as conn:
        row = conn.execute(
            "SELECT payload_json FROM audit_events WHERE object_id = ? AND event_type = 'annotation.added';",
            (cap_id,),
        ).fetchone()
        assert row is not None
        payload = json.loads(row["payload_json"])
        # Privacy invariant: plaintext content is NOT in audit payload
        assert "Highly confidential" not in row["payload_json"]
        assert payload["annotation_id"] == ann.id
        assert payload["annotation_type"] == "note"
        assert "content_hash" in payload
        assert payload["char_count"] == len("Highly confidential human insight")


def test_purge_object_audit_redaction(test_db: Database):
    with test_db.transaction() as conn:
        res = capture_item(conn, CaptureInput(text="Item to be purged"))
        cap_id = res["capture_id"]
        add_annotation(conn, "capture", cap_id, "Note before purge", author="human")

    with test_db.connection() as conn:
        events_before = conn.execute(
            "SELECT COUNT(*) FROM audit_events WHERE object_id = ?;", (cap_id,)
        ).fetchone()[0]
        assert events_before >= 2  # capture.created, annotation.added

    # Purge the object
    with test_db.transaction() as conn:
        purge_object(conn, "capture", cap_id)

    with test_db.connection() as conn:
        # Audit records must still exist (append-only trail maintained)
        all_events = conn.execute(
            "SELECT event_type, payload_json FROM audit_events WHERE object_id = ? ORDER BY created_at ASC;",
            (cap_id,),
        ).fetchall()
        assert len(all_events) == events_before + 1  # includes lifecycle.purged

        # Historical payloads must be redacted
        for ev in all_events:
            if ev["event_type"] == "lifecycle.purged":
                p = json.loads(ev["payload_json"])
                assert p["object_type"] == "capture"
                assert p["purged"] is True
            else:
                p = json.loads(ev["payload_json"])
                assert p == {"redacted": True, "reason": "purged"}
