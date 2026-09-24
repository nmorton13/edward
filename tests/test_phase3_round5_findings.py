"""Regression coverage for final Phase 3 review remediation."""

import datetime

from edward.services.answer import synthesize_answer


def test_unreviewed_valid_citation_returns_calibrated_tier3_without_auto_judgment(test_db):
    """Unchecked Level 4 support is surfaced without an unsafe second model call."""
    now = datetime.datetime.now(datetime.UTC).isoformat()
    with test_db.transaction() as conn:
        conn.execute(
            """
            INSERT INTO resources (
                id, identity_key, canonical_url, title, review_state,
                is_deleted, created_at, updated_at
            ) VALUES ('res_calibrated', 'url:calibrated', 'https://example.com/calibrated',
                      'Calibrated', 'unreviewed', 0, ?, ?);
            """,
            (now, now),
        )
        conn.execute(
            """
            INSERT INTO resource_contents (
                id, resource_id, content_hash, clean_text, extractor,
                extractor_version, char_count, created_at
            ) VALUES ('rc_calibrated', 'res_calibrated', 'hash_calibrated',
                      'The source contains a calibrated statement.', 'test', '1', 43, ?);
            """,
            (now,),
        )

    packet = {
        "query": "What does the source say?",
        "items": [
            {
                "id": "res_calibrated",
                "kind": "resource",
                "text": "The source contains a calibrated statement.",
                "review_state": "unreviewed",
                "source": {
                    "id": "res_calibrated",
                    "url": "https://example.com/calibrated",
                    "origin_namespace": "web",
                },
            }
        ],
    }

    class LocalClient:
        provider = "local"
        location = "local"
        base_url = "http://localhost:11434/v1"
        model = "test-local"
        calls = 0

        def chat_completion(self, messages, **kwargs):
            self.calls += 1
            return "The source contains a calibrated statement [#res_calibrated].", None

    client = LocalClient()
    with test_db.connection() as conn:
        result = synthesize_answer(conn, packet["query"], packet, client)
        judgment_count = conn.execute(
            "SELECT COUNT(*) FROM judgments WHERE object_id = 'res_calibrated';"
        ).fetchone()[0]

    assert client.calls == 1
    assert result["tier"] == 3
    assert result["status"] == "support_unchecked"
    assert judgment_count == 0
