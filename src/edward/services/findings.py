"""Finding and entity extraction service.

Extracts first-class findings (claims, quotations, questions, observations)
with exact supporting passages and candidate entities, maintaining provenance
and updating the FTS5 search projection.
"""

import datetime
import hashlib
import json
import logging
import re
import sqlite3
from typing import Any, Literal

from pydantic import BaseModel, Field

from edward.models import generate_id, make_job_key
from edward.services.lifecycle import reindex_object_document
from edward.services.llm import LLMClient
from edward.services.privacy import classify_content_data_class

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic Schemas for Model Extraction
# ---------------------------------------------------------------------------


class ExtractedPassage(BaseModel):
    passage: str
    locator: dict[str, Any] = Field(default_factory=dict)


class ExtractedFinding(BaseModel):
    statement: str
    assertion_role: Literal[
        "source-claim",
        "direct-quotation",
        "agent-conclusion",
        "personal-belief",
        "personal-observation",
        "question",
        "hypothesis",
        "connection",
    ] = "source-claim"
    confidence: float = 0.85
    supporting_passages: list[ExtractedPassage] = Field(default_factory=list)


class ExtractedEntity(BaseModel):
    name: str
    entity_type: str | None = None
    confidence: float = 0.85


class ExtractionPayload(BaseModel):
    findings: list[ExtractedFinding] = Field(default_factory=list)
    entities: list[ExtractedEntity] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Deterministic Rule-Based Fallback Extractor
# ---------------------------------------------------------------------------


def extract_heuristically(text: str) -> ExtractionPayload:
    """Extract findings and entities deterministically from markdown structure and punctuation."""
    findings: list[ExtractedFinding] = []
    entities_map: dict[str, str] = {}

    lines = [line.strip() for line in text.splitlines() if line.strip()]

    # 1. Look for blockquotes (direct-quotation)
    for line in lines:
        if line.startswith(">"):
            quote_text = line.lstrip("> ").strip()
            if len(quote_text) > 15:
                findings.append(
                    ExtractedFinding(
                        statement=quote_text,
                        assertion_role="direct-quotation",
                        confidence=0.95,
                        supporting_passages=[ExtractedPassage(passage=quote_text)],
                    )
                )

    # 2. Look for questions
    sentences = re.split(r"(?<=[.?!])\s+", text)
    for s in sentences:
        s_clean = s.strip()
        if s_clean.endswith("?") and len(s_clean) > 20:
            findings.append(
                ExtractedFinding(
                    statement=s_clean,
                    assertion_role="question",
                    confidence=0.90,
                    supporting_passages=[ExtractedPassage(passage=s_clean)],
                )
            )

    # 3. Look for bullet claims (- or * or numbered)
    for line in lines:
        m = re.match(r"^[-*•]\s+(.*)$", line)
        if m:
            claim = m.group(1).strip()
            if len(claim) > 25 and not claim.endswith("?"):
                findings.append(
                    ExtractedFinding(
                        statement=claim,
                        assertion_role="source-claim",
                        confidence=0.80,
                        supporting_passages=[ExtractedPassage(passage=claim)],
                    )
                )

    # 4. Extract candidate entities: technical terms, model names, capitalized phrases
    known_tech = [
        ("Apple Silicon", "hardware"),
        ("sqlite-vec", "software"),
        ("SQLite", "software"),
        ("FTS5", "software"),
        ("Ollama", "software"),
        ("llama.cpp", "software"),
        ("PyTorch", "software"),
        ("OpenRouter", "service"),
        ("TypeSafe", "service"),
        ("Jev", "service"),
        ("Birdclaw", "software"),
    ]
    for tech, etype in known_tech:
        if tech.lower() in text.lower():
            entities_map[tech] = etype

    # Capitalized 2-3 word sequences (potential named entities)
    caps_matches = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\b", text)
    for match in caps_matches[:10]:
        if len(match) > 4 and match not in entities_map:
            entities_map[match] = "entity"

    entity_list = [
        ExtractedEntity(name=name, entity_type=etype, confidence=0.75)
        for name, etype in entities_map.items()
    ]

    return ExtractionPayload(findings=findings, entities=entity_list)


# ---------------------------------------------------------------------------
# Storage of Findings and Entities
# ---------------------------------------------------------------------------


def normalize_entity_name(name: str) -> str:
    """Normalize an entity name for indexing and alias matching."""
    norm = name.strip().lower()
    norm = re.sub(r"[^\w\s-]", "", norm)
    return re.sub(r"\s+", " ", norm).strip()


def store_extracted_payload(
    conn: sqlite3.Connection,
    resource_id: str,
    payload: ExtractionPayload,
    source_content_hash: str,
    extractor: str = "extractor",
    extractor_version: str = "1.0",
) -> dict[str, Any]:
    """Store extracted findings, supporting passages, and entities in the database."""
    now_iso = datetime.datetime.now(datetime.UTC).isoformat()
    stored_findings: list[str] = []
    stored_entities: list[str] = []

    # 0. Mark obsolete extractor-owned unreviewed findings for this resource as superseded
    old_f_rows = conn.execute(
        """
        SELECT id FROM findings
        WHERE resource_id = ?
          AND (source_content_hash IS NULL OR source_content_hash != ?)
          AND review_state = 'unreviewed'
          AND extractor IS NOT NULL
          AND extractor != 'human';
        """,
        (resource_id, source_content_hash),
    ).fetchall()
    if old_f_rows:
        old_ids = [r["id"] for r in old_f_rows]
        ph = ",".join("?" * len(old_ids))
        conn.execute(
            f"DELETE FROM embeddings WHERE object_type = 'finding' AND object_id IN ({ph});",
            old_ids,
        )
        conn.execute(
            f"DELETE FROM search_documents WHERE object_type = 'finding' AND object_id IN ({ph});",
            old_ids,
        )
        conn.execute(
            f"UPDATE findings SET review_state = 'superseded', updated_at = ? WHERE id IN ({ph});",
            [now_iso] + old_ids,
        )

    conn.execute(
        """
        DELETE FROM object_entities
        WHERE object_type = 'resource' AND object_id = ?
          AND source_content_hash != ? AND review_state = 'unreviewed';
        """,
        (resource_id, source_content_hash),
    )

    # Derive privacy data class from resource's provenance
    # 1. Store findings and finding_support
    rc_row = conn.execute(
        "SELECT clean_text FROM resource_contents WHERE resource_id = ? ORDER BY created_at DESC LIMIT 1;",
        (resource_id,),
    ).fetchone()
    canonical_text = (rc_row["clean_text"] if rc_row and rc_row["clean_text"] else "").strip()

    for f in payload.findings:
        f_id = generate_id("fnd")
        conn.execute(
            """
            INSERT INTO findings (
                id, resource_id, statement, assertion_role, agent_confidence,
                extractor, extractor_version, source_content_hash,
                review_state, is_deleted, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'unreviewed', 0, ?, ?);
            """,
            (
                f_id,
                resource_id,
                f.statement,
                f.assertion_role,
                f.confidence,
                extractor,
                extractor_version,
                source_content_hash,
                now_iso,
                now_iso,
            ),
        )
        stored_findings.append(f_id)

        for p in f.supporting_passages:
            p_text = p.passage.strip()
            if not p_text:
                continue
            # Verbatim verification: must exist in canonical source clean_text
            if canonical_text and p_text not in canonical_text:
                logger.debug(
                    "Dropping ungrounded supporting passage not found in canonical source text: %s",
                    p_text[:60],
                )
                continue

            sp_id = generate_id("fsp")
            p_hash = hashlib.sha256(p_text.encode("utf-8")).hexdigest()
            conn.execute(
                """
                INSERT INTO finding_support (
                    id, finding_id, passage, locator_json, content_hash, created_at
                ) VALUES (?, ?, ?, ?, ?, ?);
                """,
                (
                    sp_id,
                    f_id,
                    p_text,
                    json.dumps(p.locator),
                    p_hash,
                    now_iso,
                ),
            )

        # Update FTS projection for new finding
        reindex_object_document(conn, "finding", f_id)

    # 2. Store entities and object_entities
    for e in payload.entities:
        clean_name = e.name.strip()
        if not clean_name:
            continue
        norm_name = normalize_entity_name(clean_name)
        if not norm_name:
            continue

        # Look for existing entity by normalized name
        row = conn.execute(
            "SELECT id FROM entities WHERE normalized_name = ?;", (norm_name,)
        ).fetchone()

        if row:
            ent_id = row["id"]
        else:
            ent_id = generate_id("ent")
            conn.execute(
                """
                INSERT INTO entities (id, name, normalized_name, entity_type, created_at)
                VALUES (?, ?, ?, ?, ?);
                """,
                (ent_id, clean_name, norm_name, e.entity_type, now_iso),
            )

        # Link to resource
        oe_id = generate_id("oen")
        conn.execute(
            """
            INSERT OR IGNORE INTO object_entities (
                id, object_type, object_id, entity_id, extractor,
                extractor_version, source_content_hash, confidence,
                review_state, created_at
            ) VALUES (?, 'resource', ?, ?, ?, ?, ?, ?, 'unreviewed', ?);
            """,
            (
                oe_id,
                resource_id,
                ent_id,
                extractor,
                extractor_version,
                source_content_hash,
                e.confidence,
                now_iso,
            ),
        )
        stored_entities.append(ent_id)

    # Update FTS projection for the parent resource
    reindex_object_document(conn, "resource", resource_id)

    # Queue embed job for the resource so newly created/updated findings are embedded
    if stored_findings:
        emb_job_id = generate_id("job")
        emb_job_key = make_job_key("embed", resource_id)
        conn.execute(
            """
            INSERT INTO processing_jobs (
                id, job_key, capture_id, resource_id, stage, status,
                available_at, attempts, created_at, updated_at
            ) VALUES (?, ?, NULL, ?, 'embed', 'pending', ?, 0, ?, ?)
            ON CONFLICT(job_key) DO UPDATE SET
                status = CASE
                    WHEN processing_jobs.status = 'running' AND (processing_jobs.lease_expires_at IS NULL OR processing_jobs.lease_expires_at > ?)
                    THEN processing_jobs.status
                    ELSE 'pending'
                END,
                lease_owner = CASE
                    WHEN processing_jobs.status = 'running' AND (processing_jobs.lease_expires_at IS NULL OR processing_jobs.lease_expires_at > ?)
                    THEN processing_jobs.lease_owner
                    ELSE NULL
                END,
                lease_expires_at = CASE
                    WHEN processing_jobs.status = 'running' AND (processing_jobs.lease_expires_at IS NULL OR processing_jobs.lease_expires_at > ?)
                    THEN processing_jobs.lease_expires_at
                    ELSE NULL
                END,
                available_at = CASE
                    WHEN processing_jobs.status = 'running' AND (processing_jobs.lease_expires_at IS NULL OR processing_jobs.lease_expires_at > ?)
                    THEN processing_jobs.available_at
                    ELSE excluded.available_at
                END,
                updated_at = excluded.updated_at;
            """,
            (
                emb_job_id,
                emb_job_key,
                resource_id,
                now_iso,
                now_iso,
                now_iso,
                now_iso,
                now_iso,
                now_iso,
                now_iso,
            ),
        )

    return {
        "resource_id": resource_id,
        "findings_count": len(stored_findings),
        "entities_count": len(stored_entities),
        "finding_ids": stored_findings,
        "entity_ids": stored_entities,
    }


def extract_findings_for_resource(
    conn: sqlite3.Connection,
    resource_id: str,
    llm_client: LLMClient | None = None,
) -> dict[str, Any]:
    """Execute finding and entity extraction for a resource and persist results."""
    # 1. Load resource content
    content_row = conn.execute(
        """
        SELECT clean_text, summary, content_hash FROM resource_contents
        WHERE resource_id = ? ORDER BY created_at DESC LIMIT 1;
        """,
        (resource_id,),
    ).fetchone()

    if not content_row:
        return {"resource_id": resource_id, "findings_count": 0, "entities_count": 0}

    text = (content_row["clean_text"] or content_row["summary"] or "").strip()
    if not text:
        return {"resource_id": resource_id, "findings_count": 0, "entities_count": 0}

    content_hash = content_row["content_hash"] or hashlib.sha256(text.encode("utf-8")).hexdigest()

    # Determine privacy data class
    res_row = conn.execute(
        "SELECT canonical_url FROM resources WHERE id = ?;", (resource_id,)
    ).fetchone()
    data_class = classify_content_data_class(
        canonical_url=res_row["canonical_url"] if res_row else None
    )

    # Only a caller-supplied client is ever used. Edward is memory, not a mind: this
    # stage does not construct one, so extraction stays deterministic by default and
    # a calling agent that wants model-based extraction passes its own client.
    client = llm_client
    payload: ExtractionPayload | None = None
    extractor_name = "heuristic-extractor"

    # 2. If LLM is available and allowed, attempt generative extraction
    if client is not None:
        prompt = (
            "You are an expert research analyst. Read the following text and extract:\n"
            "1. Key findings: empirical claims, direct quotations, unanswered questions, or conclusions.\n"
            "   For each finding, provide the exact verbatim supporting passage from the text.\n"
            "2. Notable named entities: models, hardware, software tools, organizations, or products.\n\n"
            f"TEXT:\n{text[:6000]}\n\n"
            "Output strictly valid JSON matching the ExtractionPayload schema."
        )
        try:
            _, parsed = client.chat_completion(
                messages=[
                    {
                        "role": "system",
                        "content": "You extract structured research evidence into JSON.",
                    },
                    {"role": "user", "content": prompt},
                ],
                response_model=ExtractionPayload,
                data_class=data_class,
                temperature=0.1,
            )
            if parsed is not None and (parsed.findings or parsed.entities):
                payload = parsed
                extractor_name = f"llm-{client.model}"
        except Exception as e:
            logger.info("LLM extraction failed or was bypassed: %s. Using heuristic extractor.", e)

    # 3. Fallback to deterministic heuristic extraction if needed
    if payload is None:
        payload = extract_heuristically(text)

    # 4. Persist
    return store_extracted_payload(
        conn=conn,
        resource_id=resource_id,
        payload=payload,
        source_content_hash=content_hash,
        extractor=extractor_name,
    )
