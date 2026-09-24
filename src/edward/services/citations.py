"""Citation extraction and multi-level calibration validator for Edward synthesis.

Validates that generated answers cite real, supplied objects and verifies
passage veracity without ungrounded hallucinations.

Validation Levels:
- Level 1: ID exists in database.
- Level 2: ID was present in the supplied Evidence Packet.
- Level 3: Quoted text matches stored source passage verbatim.
- Level 4: Claim support state (support_unchecked, support_likely, human_verified).
"""

import datetime
import hashlib
import json
import logging
import re
import sqlite3
from typing import Any, Literal

from pydantic import BaseModel, Field

from edward.models import generate_id
from edward.services.privacy import DataClass

logger = logging.getLogger(__name__)

SupportStatus = Literal["support_unchecked", "support_likely", "human_verified", "unsupported"]


class CitationCheckResult(BaseModel):
    """Result of validating a single citation token."""

    citation_token: str
    object_id: str
    level1_id_exists: bool = False
    level2_in_packet: bool = False
    level3_passage_match: bool | None = None
    level4_support_status: SupportStatus = "support_unchecked"
    level4_evaluator: str | None = None
    details: str = ""


class CitationValidationReport(BaseModel):
    """Aggregated validation report for all citations in an answer."""

    total_citations: int = 0
    valid_citations: int = 0
    all_levels_passed: bool = False
    citations: list[CitationCheckResult] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)


def extract_citation_ids(text: str) -> list[tuple[str, str]]:
    """Extract citation tokens and object IDs from text.

    Matches patterns like [#fnd_123], [#res_456], [fnd_123], [res_456], or [#1].
    Returns list of (token, object_id).
    """
    # Match [#id] or [id] where id starts with fnd_, res_, cap_, chk_, or is alphanumeric
    pattern = re.compile(r"\[#?([a-zA-Z0-9_-]{3,})\]")
    results: list[tuple[str, str]] = []
    seen = set()

    for m in pattern.finditer(text):
        token = m.group(0)
        obj_id = m.group(1)
        # Avoid common markdown links like [text](url)
        if obj_id.lower() in ("http", "https", "image", "ref"):
            continue
        if obj_id not in seen:
            seen.add(obj_id)
            results.append((token, obj_id))

    return results


def extract_quotes_for_citations(text: str) -> dict[str, str]:
    """Extract quoted text adjacent to citation tokens in the answer."""
    quotes: dict[str, str] = {}
    # Matches "quoted text" [#obj_id]
    pat1 = re.compile(r'["“\']([^"”\']{8,})["”\']\s*\[#?([a-zA-Z0-9_-]{3,})\]')
    for m in pat1.finditer(text):
        quotes[m.group(2)] = m.group(1).strip()

    # Matches [#obj_id] "quoted text"
    pat2 = re.compile(r'\[#?([a-zA-Z0-9_-]{3,})\]\s*["“\']([^"”\']{8,})["”\']')
    for m in pat2.finditer(text):
        if m.group(1) not in quotes:
            quotes[m.group(1)] = m.group(2).strip()

    return quotes


def check_id_exists_in_db(conn: sqlite3.Connection, object_id: str) -> bool:
    """Level 1 check: Verify that the object ID exists in Edward tables."""
    queries = [
        "SELECT 1 FROM findings WHERE id = ? AND is_deleted = 0 LIMIT 1;",
        "SELECT 1 FROM resources WHERE id = ? AND is_deleted = 0 LIMIT 1;",
        "SELECT 1 FROM resource_chunks WHERE id = ? LIMIT 1;",
        "SELECT 1 FROM captures WHERE id = ? AND is_deleted = 0 LIMIT 1;",
    ]
    for q in queries:
        row = conn.execute(q, (object_id,)).fetchone()
        if row:
            return True
    return False


def verify_passage_verbatim(
    conn: sqlite3.Connection,
    object_id: str,
    quoted_text: str,
) -> bool:
    """Level 3 check: Verify that quoted text appears verbatim in source content or verified chunks.

    Case-sensitive and strictly checks canonical clean text, chunks, or capture content, never summaries or ungrounded claims.
    """
    clean_quote = quoted_text.strip()
    if not clean_quote:
        return False

    # 1. Determine target resource_id or capture_id
    target_res_id = object_id
    f_row = conn.execute("SELECT resource_id FROM findings WHERE id = ?;", (object_id,)).fetchone()
    if f_row and f_row["resource_id"]:
        target_res_id = f_row["resource_id"]

    # 2. Check canonical clean_text of the resource (never summary)
    r_rows = conn.execute(
        """
        SELECT clean_text FROM resource_contents
        WHERE resource_id = ? ORDER BY created_at DESC LIMIT 1;
        """,
        (target_res_id,),
    ).fetchall()
    for r in r_rows:
        body = r["clean_text"] or ""
        if clean_quote in body:
            return True

    # 3. Check resource_chunks
    c_rows = conn.execute(
        "SELECT text FROM resource_chunks WHERE id = ? OR resource_id = ?;",
        (object_id, target_res_id),
    ).fetchall()
    for r in c_rows:
        if clean_quote in (r["text"] or ""):
            return True

    # 4. Check capture raw_content or user_note
    cap_rows = conn.execute(
        "SELECT raw_content, user_note FROM captures WHERE id = ?;",
        (object_id,),
    ).fetchall()
    for cap in cap_rows:
        if clean_quote in (cap["raw_content"] or "") or clean_quote in (cap["user_note"] or ""):
            return True

    return False


def _is_positive_support_answer(answer_raw: str) -> bool:
    """Validate that judgment answer_json indicates a positive claim support evaluation."""
    if not answer_raw:
        return False
    try:
        val = json.loads(answer_raw)
    except Exception:
        val = answer_raw.strip().strip('"').lower()

    if isinstance(val, bool):
        return val is True
    if isinstance(val, (int, float)):
        return float(val) >= 0.7
    if isinstance(val, str):
        return val.strip().lower() in ("true", "yes", "supported", "support_likely", "support")
    if isinstance(val, dict):
        for k in (
            "supported",
            "support",
            "claim_supported",
            "result",
            "answer",
            "verdict",
            "noul",
            "probability",
        ):
            if k in val:
                v = val[k]
                if isinstance(v, bool):
                    return v is True
                if isinstance(v, str) and v.strip().lower() in ("true", "yes", "supported"):
                    return True
                if isinstance(v, (int, float)) and float(v) >= 0.7:
                    return True
    return False


def _get_object_canonical_hashes(conn: sqlite3.Connection, object_id: str) -> set[str]:
    """Retrieve canonical content and text hashes associated with an object."""
    hashes = set()
    # Check findings
    f_row = conn.execute(
        "SELECT statement, source_content_hash FROM findings WHERE id = ?;", (object_id,)
    ).fetchone()
    if f_row:
        if f_row["statement"]:
            hashes.add(hashlib.sha256(f_row["statement"].encode("utf-8")).hexdigest())
        if f_row["source_content_hash"]:
            hashes.add(f_row["source_content_hash"])
        fs_rows = conn.execute(
            "SELECT passage, content_hash FROM finding_support WHERE finding_id = ?;", (object_id,)
        ).fetchall()
        for fs in fs_rows:
            if fs["content_hash"]:
                hashes.add(fs["content_hash"])
            if fs["passage"]:
                hashes.add(hashlib.sha256(fs["passage"].encode("utf-8")).hexdigest())

    # Check resources
    rc_rows = conn.execute(
        "SELECT content_hash, clean_text FROM resource_contents WHERE resource_id = ?;",
        (object_id,),
    ).fetchall()
    for rc in rc_rows:
        if rc["content_hash"]:
            hashes.add(rc["content_hash"])
        if rc["clean_text"]:
            hashes.add(hashlib.sha256(rc["clean_text"].encode("utf-8")).hexdigest())

    # Check chunks
    chk_row = conn.execute(
        "SELECT text FROM resource_chunks WHERE id = ?;", (object_id,)
    ).fetchone()
    if chk_row and chk_row["text"]:
        hashes.add(hashlib.sha256(chk_row["text"].encode("utf-8")).hexdigest())

    # Check captures
    cap_row = conn.execute(
        "SELECT raw_content, user_note FROM captures WHERE id = ?;", (object_id,)
    ).fetchone()
    if cap_row:
        if cap_row["raw_content"]:
            hashes.add(hashlib.sha256(cap_row["raw_content"].encode("utf-8")).hexdigest())
        if cap_row["user_note"]:
            hashes.add(hashlib.sha256(cap_row["user_note"].encode("utf-8")).hexdigest())

    return hashes


def validate_citations(
    conn: sqlite3.Connection,
    answer_text: str,
    evidence_packet: dict[str, Any] | None = None,
    quoted_passages: dict[str, str] | None = None,
) -> CitationValidationReport:
    """Validate all citations in answer_text across Levels 1 through 4."""
    extracted = extract_citation_ids(answer_text)
    quotes = dict(quoted_passages or {})
    extracted_quotes = extract_quotes_for_citations(answer_text)
    for q_id, q_text in extracted_quotes.items():
        if q_id not in quotes:
            quotes[q_id] = q_text

    # Extract allowed IDs from evidence packet (Level 2)
    packet_ids: set[str] = set()
    if evidence_packet and "items" in evidence_packet:
        for item in evidence_packet["items"]:
            if isinstance(item, dict) and "id" in item:
                packet_ids.add(item["id"])
                # Also include source id if present
                if "source" in item and isinstance(item["source"], dict) and "id" in item["source"]:
                    packet_ids.add(item["source"]["id"])

    report = CitationValidationReport()
    valid_count = 0

    for token, obj_id in extracted:
        # Level 1: Database existence
        l1_pass = check_id_exists_in_db(conn, obj_id)

        # Level 2: Evidence Packet presence
        l2_pass = (obj_id in packet_ids) if evidence_packet is not None else l1_pass

        # Level 3: Verbatim passage match (if a quote was provided or extracted for this ID)
        l3_pass = None
        if obj_id in quotes:
            l3_pass = verify_passage_verbatim(conn, obj_id, quotes[obj_id])

        # Level 4: Claim support state
        support_status: SupportStatus = "support_unchecked"
        evaluator: str | None = None
        if not l1_pass or not l2_pass:
            support_status = "unsupported"
            evaluator = "validator"
        else:
            # 1. Check for explicit human review on this object or parent resource
            is_human_verified = False
            f_row = conn.execute(
                "SELECT resource_id, review_state FROM findings WHERE id = ?;", (obj_id,)
            ).fetchone()
            if f_row:
                if f_row["review_state"] in ("approved", "reviewed"):
                    is_human_verified = True
                    evaluator = "human"
                elif f_row["resource_id"]:
                    p_row = conn.execute(
                        "SELECT review_state FROM resources WHERE id = ?;", (f_row["resource_id"],)
                    ).fetchone()
                    if p_row and p_row["review_state"] in ("approved", "reviewed"):
                        is_human_verified = True
                        evaluator = "human/parent_resource"
            else:
                r_row = conn.execute(
                    "SELECT review_state FROM resources WHERE id = ?;", (obj_id,)
                ).fetchone()
                if r_row:
                    if r_row["review_state"] in ("approved", "reviewed"):
                        is_human_verified = True
                        evaluator = "human"
                else:
                    c_row = conn.execute(
                        "SELECT review_state FROM captures WHERE id = ?;", (obj_id,)
                    ).fetchone()
                    if c_row:
                        if c_row["review_state"] in ("approved", "reviewed"):
                            is_human_verified = True
                            evaluator = "human"
                    else:
                        chk_row = conn.execute(
                            """
                            SELECT r.review_state
                            FROM resource_chunks rc
                            JOIN resources r ON rc.resource_id = r.id
                            WHERE rc.id = ?;
                            """,
                            (obj_id,),
                        ).fetchone()
                        if chk_row and chk_row["review_state"] in ("approved", "reviewed"):
                            is_human_verified = True
                            evaluator = "human/parent_resource"

            if is_human_verified:
                support_status = "human_verified"
            else:
                # 2. Check for explicit completed claim-support judgment with provenance
                valid_hashes = _get_object_canonical_hashes(conn, obj_id)
                j_rows = conn.execute(
                    """
                    SELECT provider, resolved_model, confidence, probability,
                           answer_json, input_content_hash, family, label_or_question_id
                    FROM judgments
                    WHERE object_id = ?
                      AND status = 'completed'
                      AND label_or_question_id IN ('claim-support', 'claim_support')
                      AND family IN ('jev', 'claim-support', 'evaluation', 'custom')
                      AND (confidence >= 0.7 OR probability >= 0.7)
                    ORDER BY created_at DESC;
                    """,
                    (obj_id,),
                ).fetchall()

                found_support = False
                for j_row in j_rows:
                    if not _is_positive_support_answer(j_row["answer_json"]):
                        continue
                    j_hash = j_row["input_content_hash"]
                    if valid_hashes and j_hash and j_hash not in valid_hashes:
                        continue
                    support_status = "support_likely"
                    evaluator = f"model:{j_row['resolved_model'] or j_row['provider']}"
                    found_support = True
                    break

                if not found_support:
                    support_status = "support_unchecked"
                    evaluator = None

        is_item_valid = (
            l1_pass
            and l2_pass
            and (l3_pass is None or l3_pass)
            and support_status in ("human_verified", "support_likely")
        )
        if is_item_valid:
            valid_count += 1
        else:
            report.unsupported_claims.append(f"Citation {token} ({obj_id}) failed validation.")

        details = []
        if not l1_pass:
            details.append("ID not found in database")
        if not l2_pass:
            details.append("ID not present in evidence packet")
        if l3_pass is False:
            details.append("Quoted passage does not match source verbatim")
        if support_status not in ("human_verified", "support_likely"):
            details.append(f"Level 4 support status is {support_status} ({evaluator})")

        report.citations.append(
            CitationCheckResult(
                citation_token=token,
                object_id=obj_id,
                level1_id_exists=l1_pass,
                level2_in_packet=l2_pass,
                level3_passage_match=l3_pass,
                level4_support_status=support_status,
                level4_evaluator=evaluator,
                details="; ".join(details) if details else "Valid",
            )
        )

    report.total_citations = len(extracted)
    report.valid_citations = valid_count
    # Factual synthesis strictly requires at least one valid citation; 0 citations cannot pass
    report.all_levels_passed = (len(extracted) > 0) and (len(extracted) == valid_count)

    return report


def evaluate_claim_support(
    conn: sqlite3.Connection,
    object_id: str,
    claim_text: str | None = None,
    llm_client: Any | None = None,
    data_class: DataClass = "personal_notes",
) -> dict[str, Any]:
    """Evaluate Level 4 claim support for an object and record a completed judgment.

    Enforces evaluator/model provenance and matching canonical input content hash.
    """
    now_iso = datetime.datetime.now(datetime.UTC).isoformat()

    # 1. Determine canonical text and primary input hash
    canonical_text = ""
    primary_hash: str | None = None
    obj_type = "resource"

    f_row = conn.execute(
        "SELECT resource_id, statement FROM findings WHERE id = ?;", (object_id,)
    ).fetchone()
    if f_row:
        obj_type = "finding"
        canonical_text = f_row["statement"] or ""
        supp = conn.execute(
            "SELECT passage, content_hash FROM finding_support WHERE finding_id = ? LIMIT 1;",
            (object_id,),
        ).fetchone()
        if supp:
            canonical_text = supp["passage"] or canonical_text
            primary_hash = supp["content_hash"] or primary_hash
    else:
        rc_row = conn.execute(
            "SELECT clean_text, content_hash FROM resource_contents WHERE resource_id = ? ORDER BY created_at DESC LIMIT 1;",
            (object_id,),
        ).fetchone()
        if rc_row:
            canonical_text = rc_row["clean_text"] or ""
            primary_hash = rc_row["content_hash"] or primary_hash
        else:
            chk_row = conn.execute(
                "SELECT text FROM resource_chunks WHERE id = ?;", (object_id,)
            ).fetchone()
            if chk_row:
                obj_type = "resource_chunk"
                canonical_text = chk_row["text"] or ""
                primary_hash = hashlib.sha256(canonical_text.encode("utf-8")).hexdigest()
            else:
                cap_row = conn.execute(
                    "SELECT raw_content, user_note FROM captures WHERE id = ?;", (object_id,)
                ).fetchone()
                if cap_row:
                    obj_type = "capture"
                    canonical_text = (cap_row["user_note"] or "") + (
                        "\n\n" + (cap_row["raw_content"] or "")
                    )
                    primary_hash = hashlib.sha256(
                        canonical_text.strip().encode("utf-8")
                    ).hexdigest()

    if not primary_hash:
        primary_hash = hashlib.sha256((canonical_text or "").encode("utf-8")).hexdigest()

    # 2. Check if an existing completed positive claim-support judgment already exists with matching hash
    existing = conn.execute(
        """
        SELECT id, confidence, answer_json, resolved_model, provider
        FROM judgments
        WHERE object_id = ?
          AND status = 'completed'
          AND label_or_question_id IN ('claim-support', 'claim_support')
          AND family IN ('jev', 'claim-support', 'evaluation', 'custom')
          AND input_content_hash = ?
          AND (confidence >= 0.7 OR probability >= 0.7)
        ORDER BY created_at DESC LIMIT 1;
        """,
        (object_id, primary_hash),
    ).fetchone()

    if existing and _is_positive_support_answer(existing["answer_json"]):
        return {
            "status": "completed",
            "judgment_id": existing["id"],
            "supported": True,
            "confidence": existing["confidence"],
            "model": existing["resolved_model"],
            "provider": existing["provider"],
        }

    # 3. Evaluate claim support using llm_client if provided
    eval_supported = False
    eval_confidence = 0.0
    eval_model = "grounded-passage-v1"
    eval_provider = "rule-based"

    target_claim = (claim_text or "").strip()
    if llm_client is not None and hasattr(llm_client, "chat_completion"):
        try:
            eval_model = getattr(llm_client, "model", "llm")
            eval_provider = getattr(llm_client, "provider", "llm")
            prompt = (
                "You are evaluating claim support.\n"
                f"SOURCE TEXT: {canonical_text[:2000]}\n"
                f"CLAIM: {target_claim or canonical_text[:500]}\n\n"
                "Does the source text substantiate the claim? Answer strictly with JSON: "
                '{"supported": true, "confidence": 0.95}'
            )
            resp, _ = llm_client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                data_class=data_class,
            )
            clean_resp = resp.strip()
            if "{" in clean_resp and "}" in clean_resp:
                start = clean_resp.find("{")
                end = clean_resp.rfind("}") + 1
                data = json.loads(clean_resp[start:end])
                eval_supported = bool(data.get("supported", False))
                eval_confidence = float(data.get("confidence", 0.85 if eval_supported else 0.0))
            elif "yes" in clean_resp.lower():
                eval_supported = True
                eval_confidence = 0.85
        except Exception as e:
            logger.warning(
                "Model claim-support evaluation error: %s. Falling back to rule-based.", e
            )

    if not eval_supported:
        # Rule-based fallback: check if target_claim is verbatim in canonical text,
        # or if target_claim is empty and canonical text exists
        if target_claim and target_claim in canonical_text:
            eval_supported = True
            eval_confidence = 0.95
            eval_model = "grounded-passage-v1"
            eval_provider = "rule-based"

    jdg_id = generate_id("jdg")
    answer_payload = json.dumps({"supported": eval_supported, "confidence": eval_confidence})
    conn.execute(
        """
        INSERT INTO judgments (
            id, object_type, object_id, family, label_or_question_id,
            primitive, answer_json, requested_model, resolved_model,
            provider, question_registry_version, threshold_policy_version,
            input_content_hash, confidence, status, created_at
        ) VALUES (?, ?, ?, 'claim-support', 'claim-support', 'noul', ?, ?, ?, ?, '1.0', '1.0', ?, ?, 'completed', ?);
        """,
        (
            jdg_id,
            obj_type,
            object_id,
            answer_payload,
            eval_model,
            eval_model,
            eval_provider,
            primary_hash,
            eval_confidence,
            now_iso,
        ),
    )

    return {
        "status": "completed",
        "judgment_id": jdg_id,
        "supported": eval_supported,
        "confidence": eval_confidence,
        "model": eval_model,
        "provider": eval_provider,
    }
