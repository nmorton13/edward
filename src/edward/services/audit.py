"""Append-only audit logging service for Edward."""

import datetime
import json
import sqlite3
from typing import Any

from edward.models import AuditEvent, generate_id


def parse_timestamp(val: Any) -> datetime.datetime:
    """Safely parse SQLite timestamp string or datetime into datetime.datetime."""
    if isinstance(val, datetime.datetime):
        return val
    try:
        return datetime.datetime.fromisoformat(str(val))
    except Exception:
        return datetime.datetime.now(datetime.UTC)


def record_audit_event(
    conn: sqlite3.Connection,
    event_type: str,
    object_type: str,
    object_id: str,
    actor: str,
    payload: dict[str, Any] | None = None,
) -> AuditEvent:
    """Record an immutable audit event in the database."""
    event_id = generate_id("evt")
    now = datetime.datetime.now(datetime.UTC)
    payload_json = json.dumps(payload) if payload is not None else None

    conn.execute(
        """
        INSERT INTO audit_events (id, event_type, object_type, object_id, actor, payload_json, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?);
        """,
        (event_id, event_type, object_type, object_id, actor, payload_json, now.isoformat()),
    )

    return AuditEvent(
        id=event_id,
        event_type=event_type,
        object_type=object_type,
        object_id=object_id,
        actor=actor,
        payload_json=payload_json,
        created_at=now,
    )


def get_audit_trail(
    conn: sqlite3.Connection,
    object_id: str,
    limit: int = 50,
) -> list[AuditEvent]:
    """Retrieve audit history for a specific object in chronological order."""
    cursor = conn.execute(
        """
        SELECT id, event_type, object_type, object_id, actor, payload_json, created_at
        FROM audit_events
        WHERE object_id = ?
        ORDER BY created_at ASC
        LIMIT ?;
        """,
        (object_id, limit),
    )
    rows = cursor.fetchall()
    return [
        AuditEvent(
            id=row["id"],
            event_type=row["event_type"],
            object_type=row["object_type"],
            object_id=row["object_id"],
            actor=row["actor"],
            payload_json=row["payload_json"],
            created_at=parse_timestamp(row["created_at"]),
        )
        for row in rows
    ]
