"""Resource summarization service (model-backed, opt-in).

Summarizes extracted resource clean text into 2-3 sentence overviews using an
OpenAI-compatible model client. Follows Edward's strict privacy rules, preserves
existing bundle or human summaries, and maintains idempotency on content hash.
"""

import datetime
import logging
import sqlite3
from typing import Any

from pydantic import BaseModel, Field

from edward.models import generate_id, make_job_key
from edward.services import llm
from edward.services.llm import LLMClient
from edward.services.privacy import (
    DataClass,
    assert_transmission_permitted,
)

logger = logging.getLogger(__name__)

SUMMARIZE_STAGE = "summarize"


class SummarizerDisabledError(Exception):
    """Raised when summarizer operation is attempted without a configured model."""

    pass


class ResourceSummaryPayload(BaseModel):
    """Structured response payload for resource summarization."""

    summary: str = Field(
        ...,
        description="A concise 2-3 sentence factual plain-text summary of the content.",
        min_length=1,
    )


def generate_summary(
    text: str,
    client: LLMClient | Any | None = None,
    data_class: DataClass = "public_web",
) -> str:
    """Generate a 2-3 sentence plain text summary of text using a configured LLM client.

    Strictly enforces privacy rules BEFORE dispatch.
    """
    clean = text.strip()
    if not clean:
        return ""

    resolved_client = client or llm.get_summarizer_client()
    if resolved_client is None:
        raise SummarizerDisabledError(
            "Summarizer model is not configured or disabled. "
            "Set EDWARD_SUMMARIZER_MODE=enabled (and endpoint/model if needed)."
        )

    # Privacy check MUST run and halt before prompt construction or network dispatch
    assert_transmission_permitted(
        provider_name=resolved_client.provider,
        data_class=data_class,
        declared_location=resolved_client.location,
        base_url=resolved_client.base_url,
    )

    system_prompt = (
        "You are an objective research assistant that writes concise, factual summaries of text. "
        "Summarize the provided content in 2-3 sentences of clear, plain text. "
        "Do not invent facts or extrapolate beyond what is stated in the text. "
        "Output strictly valid JSON conforming to the schema."
    )
    user_prompt = (
        f'TEXT:\n{clean[:8000]}\n\nProvide a 2-3 sentence summary as JSON: {{"summary": "..."}}'
    )

    _, parsed = resolved_client.chat_completion(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_model=ResourceSummaryPayload,
        data_class=data_class,
        temperature=0.2,
    )

    if parsed is not None and parsed.summary.strip():
        return parsed.summary.strip()

    raise ValueError("Summarizer model returned empty summary")


def enqueue_missing_summarize_jobs(
    conn: sqlite3.Connection,
    capture_id: str | None = None,
) -> int:
    """Enqueue pending summarize jobs for resources with extracted content that lack them.

    Resources with existing summaries are marked completed with their current content hash
    so they are not re-processed.
    """
    now_iso = datetime.datetime.now(datetime.UTC).isoformat()

    if capture_id:
        rows = conn.execute(
            """
            SELECT r.id AS resource_id, rc.content_hash, rc.summary, rc.summary_source, cr.capture_id
            FROM resources r
            JOIN resource_contents rc ON r.id = rc.resource_id
            JOIN capture_resources cr ON r.id = cr.resource_id
            WHERE cr.capture_id = ?
              AND r.is_deleted = 0
              AND length(trim(rc.clean_text)) > 0
              AND rc.id = (
                  SELECT id FROM resource_contents
                  WHERE resource_id = r.id
                  ORDER BY created_at DESC LIMIT 1
              );
            """,
            (capture_id,),
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT r.id AS resource_id, rc.content_hash, rc.summary, rc.summary_source,
                   (SELECT cr.capture_id FROM capture_resources cr
                    WHERE cr.resource_id = r.id ORDER BY cr.created_at DESC LIMIT 1) AS capture_id
            FROM resources r
            JOIN resource_contents rc ON r.id = rc.resource_id
            WHERE r.is_deleted = 0
              AND length(trim(rc.clean_text)) > 0
              AND rc.id = (
                  SELECT id FROM resource_contents
                  WHERE resource_id = r.id
                  ORDER BY created_at DESC LIMIT 1
              );
            """
        ).fetchall()

    queued = 0
    for r in rows:
        res_id = r["resource_id"]
        c_hash = r["content_hash"]
        summary_val = (r["summary"] or "").strip()
        source = r["summary_source"]
        cap_id = r["capture_id"]

        is_legacy = bool(summary_val and source == "legacy")
        has_final_summary = bool(summary_val and source in ("bundle", "human", "model"))

        job_key = make_job_key(SUMMARIZE_STAGE, res_id)
        job_id = generate_id("job")

        if has_final_summary:
            # Already has final summary (from bundle, human, or model): mark completed
            conn.execute(
                """
                INSERT INTO processing_jobs (
                    id, job_key, capture_id, resource_id, stage, status, input_hash,
                    attempts, available_at, completed_at, created_at, updated_at
                ) VALUES (?, ?, ?, ?, 'summarize', 'completed', ?, 0, ?, ?, ?, ?)
                ON CONFLICT(job_key) DO UPDATE SET
                    input_hash = CASE
                        WHEN processing_jobs.status = 'completed' AND processing_jobs.input_hash IS NULL
                        THEN excluded.input_hash
                        ELSE processing_jobs.input_hash
                    END,
                    updated_at = excluded.updated_at;
                """,
                (job_id, job_key, cap_id, res_id, c_hash, now_iso, now_iso, now_iso, now_iso),
            )
        else:
            # Needs a summary (missing or legacy summary): insert or update pending
            cursor = conn.execute(
                """
                INSERT INTO processing_jobs (
                    id, job_key, capture_id, resource_id, stage, status, input_hash,
                    attempts, available_at, created_at, updated_at
                ) VALUES (?, ?, ?, ?, 'summarize', 'pending', NULL, 0, ?, ?, ?)
                ON CONFLICT(job_key) DO UPDATE SET
                    capture_id = COALESCE(excluded.capture_id, processing_jobs.capture_id),
                    status = CASE
                        WHEN (processing_jobs.status = 'completed' AND (processing_jobs.input_hash != ? OR ? = 1))
                             OR processing_jobs.status = 'failed'
                        THEN 'pending'
                        ELSE processing_jobs.status
                    END,
                    available_at = CASE
                        WHEN (processing_jobs.status = 'completed' AND (processing_jobs.input_hash != ? OR ? = 1))
                             OR processing_jobs.status = 'failed'
                        THEN excluded.available_at
                        ELSE processing_jobs.available_at
                    END,
                    attempts = CASE
                        WHEN (processing_jobs.status = 'completed' AND (processing_jobs.input_hash != ? OR ? = 1))
                             OR processing_jobs.status = 'failed'
                        THEN 0
                        ELSE processing_jobs.attempts
                    END,
                    last_error = CASE
                        WHEN (processing_jobs.status = 'completed' AND (processing_jobs.input_hash != ? OR ? = 1))
                             OR processing_jobs.status = 'failed'
                        THEN NULL
                        ELSE processing_jobs.last_error
                    END,
                    completed_at = CASE
                        WHEN (processing_jobs.status = 'completed' AND (processing_jobs.input_hash != ? OR ? = 1))
                             OR processing_jobs.status = 'failed'
                        THEN NULL
                        ELSE processing_jobs.completed_at
                    END,
                    input_hash = CASE
                        WHEN (processing_jobs.status = 'completed' AND ? = 1)
                        THEN NULL
                        ELSE processing_jobs.input_hash
                    END,
                    updated_at = excluded.updated_at;
                """,
                (
                    job_id,
                    job_key,
                    cap_id,
                    res_id,
                    now_iso,
                    now_iso,
                    now_iso,
                    c_hash,
                    1 if is_legacy else 0,
                    c_hash,
                    1 if is_legacy else 0,
                    c_hash,
                    1 if is_legacy else 0,
                    c_hash,
                    1 if is_legacy else 0,
                    c_hash,
                    1 if is_legacy else 0,
                    1 if is_legacy else 0,
                ),
            )
            if cursor.rowcount > 0:
                queued += 1

    return queued
