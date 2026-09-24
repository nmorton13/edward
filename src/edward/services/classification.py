"""Classification pipeline orchestrator, deterministic form detection, and judgment persistence."""

import datetime
import hashlib
import json
import os
import sqlite3
from pathlib import Path
from typing import Any

from edward.classifiers.base import (
    BaseClassifier,
    ClassificationRequest,
    ClassificationResult,
)
from edward.classifiers.dry_run import DryRunClassifier
from edward.classifiers.local import LocalClassifier
from edward.models import generate_id
from edward.services.lifecycle import reindex_object_document


def _now_iso() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat()


def detect_primary_form(
    url: str | None = None,
    text: str | None = None,
    mime_type: str | None = None,
) -> str:
    """Heuristically determine primary object form from URL patterns, MIME types, and text characteristics."""
    if url:
        u_lower = url.lower()
        if "x.com" in u_lower or "twitter.com" in u_lower:
            return "x-post"
        if "github.com" in u_lower or "gitlab.com" in u_lower:
            return "repository"
        if "arxiv.org" in u_lower:
            return "paper"

    if mime_type:
        m_lower = mime_type.lower()
        if "pdf" in m_lower:
            return "paper"

    if not url and text and len(text.strip()) < 1000:
        return "personal-note"

    if url:
        return "article"

    return "other"


def get_classifier(provider: str | None = None) -> BaseClassifier | None:
    """Instantiate the active classifier based on parameter or environment configuration."""
    prov = (provider or os.environ.get("EDWARD_CLASSIFIER_PROVIDER", "disabled")).lower().strip()
    if prov in ("disabled", "none", "off", ""):
        return None
    elif prov in ("dry-run", "dry_run"):
        return DryRunClassifier()
    elif prov == "local":
        return LocalClassifier()
    elif prov == "typesafe":
        from edward.classifiers.jev import JevClassifier
        from edward.classifiers.providers.typesafe import TypeSafeProvider

        return JevClassifier(TypeSafeProvider(), provider_name="typesafe")
    elif prov == "openrouter":
        from edward.classifiers.jev import JevClassifier
        from edward.classifiers.providers.openrouter import OpenRouterProvider

        return JevClassifier(OpenRouterProvider(), provider_name="openrouter")
    elif prov == "jev":
        from edward.classifiers.jev import JevClassifier

        underlying = os.environ.get("EDWARD_CLASSIFIER_TRANSPORT", "typesafe").strip().lower()
        if underlying == "openrouter":
            from edward.classifiers.providers.openrouter import OpenRouterProvider

            return JevClassifier(OpenRouterProvider(), provider_name="openrouter")
        else:
            from edward.classifiers.providers.typesafe import TypeSafeProvider

            return JevClassifier(TypeSafeProvider(), provider_name="typesafe")
    else:
        raise ValueError(
            f"Unknown classifier provider: '{prov}'. Supported: 'disabled', 'dry-run', 'local', 'typesafe', 'openrouter', 'jev'."
        )


def load_thresholds(policy_name: str = "balanced-precision") -> dict[str, float]:
    """Load versioned decision thresholds from registries."""
    registry_path = Path(__file__).parent.parent / "registries" / "thresholds-v1.json"
    if registry_path.exists():
        try:
            data = json.loads(registry_path.read_text(encoding="utf-8"))
            policies = data.get("policies", {})
            default_pol = data.get("default_policy", "balanced-precision")
            policy_data = policies.get(policy_name) or policies.get(default_pol) or {}
            return policy_data.get("thresholds", {})
        except Exception:
            pass
    return {}


def resolve_threshold(
    thresholds: dict[str, float],
    family: str,
    label_id: str,
    default: float = 0.70,
) -> float:
    """Resolve decision threshold for a label across multiple namespace styles."""
    clean_id = label_id.replace("/", "-")
    leaf_id = label_id.split("/")[-1]
    candidates = [
        label_id,
        f"{family}-{clean_id}",
        clean_id,
        leaf_id,
        f"{family}-{leaf_id}",
        family,
    ]
    for cand in candidates:
        if cand in thresholds:
            return thresholds[cand]
    return default


def make_canonical_classifier_text(
    clean_text: str | None = None,
    summary: str | None = None,
    title: str | None = None,
    url: str | None = None,
) -> str:
    """Build canonical text representation used for classification inference and hashing."""
    clean = (clean_text or "").strip()
    summ = (summary or "").strip()
    if summ and clean and summ != clean:
        text = f"{summ}\n\n{clean}"
    else:
        text = clean or summ
    if not text.strip():
        text = (title or url or "").strip()
    if not text.strip():
        text = "Untitled object"
    return text


def compute_classification_input_hash(
    clean_text: str | None = None,
    summary: str | None = None,
    title: str | None = None,
    url: str | None = None,
) -> str:
    """Compute the SHA-256 hash of the canonical classifier input text."""
    text = make_canonical_classifier_text(
        clean_text=clean_text, summary=summary, title=title, url=url
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_classification_target(
    conn: sqlite3.Connection,
    object_type: str,
    object_id: str,
) -> dict[str, Any] | None:
    """Load object text, URL, and metadata from the database for classification."""
    table = (
        "captures"
        if object_type == "capture"
        else ("resources" if object_type == "resource" else "findings")
    )
    row = conn.execute(f"SELECT * FROM {table} WHERE id = ?;", (object_id,)).fetchone()
    if not row:
        return None

    url: str | None = None
    metadata: dict[str, Any] = {}
    has_content = False
    if object_type == "resource":
        url = row["canonical_url"]
        content_row = conn.execute(
            "SELECT clean_text, summary, content_hash FROM resource_contents WHERE resource_id = ? ORDER BY created_at DESC LIMIT 1;",
            (object_id,),
        ).fetchone()
        clean = (content_row["clean_text"] or "") if content_row else ""
        summary = (content_row["summary"] or "") if content_row else ""
        has_content = bool(clean.strip() or summary.strip())
        text = make_canonical_classifier_text(
            clean_text=clean,
            summary=summary,
            title=row["title"],
            url=url,
        )
        cap_link = conn.execute(
            """
            SELECT c.origin_namespace, c.origin_id
            FROM capture_resources cr
            JOIN captures c ON cr.capture_id = c.id
            WHERE cr.resource_id = ?
            ORDER BY cr.created_at DESC LIMIT 1;
            """,
            (object_id,),
        ).fetchone()
        origin_ns = cap_link["origin_namespace"] if cap_link else "web"
        origin_id = cap_link["origin_id"] if cap_link else ""
        metadata = {
            "url": url,
            "canonical_url": url,
            "origin_namespace": origin_ns,
            "origin_id": origin_id,
            "form": row["primary_form"],
            "title": row["title"],
        }
    elif object_type == "capture":
        parts = [p.strip() for p in (row["user_note"], row["raw_content"]) if p and p.strip()]
        has_content = bool(parts)
        text = "\n\n".join(parts)
        res_link = conn.execute(
            "SELECT r.canonical_url FROM resources r JOIN capture_resources cr ON cr.resource_id = r.id WHERE cr.capture_id = ? LIMIT 1;",
            (object_id,),
        ).fetchone()
        if res_link:
            url = res_link["canonical_url"]
        if not text.strip():
            text = url or "Untitled object"
        metadata = {
            "url": url,
            "canonical_url": url,
            "origin_namespace": row["origin_namespace"],
            "origin_id": row["origin_id"] or "",
            "collection_channel": row["collection_channel"],
        }
    else:
        text = (row["statement"] or "").strip()
        has_content = bool(text)
        if not text:
            text = "Untitled object"
        if row["resource_id"]:
            r_row = conn.execute(
                "SELECT canonical_url FROM resources WHERE id = ?;", (row["resource_id"],)
            ).fetchone()
            if r_row:
                url = r_row["canonical_url"]
        metadata = {
            "url": url,
            "canonical_url": url,
            "origin_namespace": "edward",
        }

    content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

    return {
        "object_type": object_type,
        "object_id": object_id,
        "text": text,
        "url": url,
        "content_hash": content_hash,
        "metadata": metadata,
        "has_content": has_content,
    }


def classify_target(
    text: str,
    url: str | None,
    record_id: str,
    object_type: str,
    provider: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> tuple[str, ClassificationResult | None]:
    """Execute heuristic form detection and classifier inference outside database transactions."""
    detected_form = detect_primary_form(url, text)
    classifier = get_classifier(provider)
    if classifier is None:
        return detected_form, None

    req_meta = {"url": url, "canonical_url": url}
    if metadata:
        req_meta.update(metadata)

    request = ClassificationRequest(
        record_id=record_id,
        object_type=object_type,  # type: ignore[arg-type]
        text=text,
        metadata=req_meta,
    )
    result = classifier.classify(request)
    return detected_form, result


def persist_classification_result(
    conn: sqlite3.Connection,
    object_type: str,
    object_id: str,
    detected_form: str | None,
    result: ClassificationResult | None,
    text: str,
) -> list[dict]:
    """Persist detected form, raw judgments, threshold decisions, and update FTS projection."""
    if object_type == "resource" and detected_form:
        conn.execute(
            """
            UPDATE resources
            SET primary_form = ?
            WHERE id = ? AND (primary_form IS NULL OR primary_form = 'other');
            """,
            (detected_form, object_id),
        )

    if result is None:
        reindex_object_document(conn, object_type, object_id)
        return []

    content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    thresholds = load_thresholds()
    default_threshold = 0.70
    now_iso = datetime.datetime.now(datetime.UTC).isoformat()
    saved_judgments: list[dict] = []

    for item in result.judgments:
        judgment_id = generate_id("jdg")
        ans_json = json.dumps(item.answer)

        req_model = result.requested_model or result.model
        res_model = result.resolved_model or result.model

        conn.execute(
            """
            INSERT INTO judgments (
                id, object_type, object_id, family, label_or_question_id, primitive,
                answer_json, probability, confidence, requested_model, resolved_model,
                provider, provider_request_id, question_registry_version, label_registry_version,
                threshold_policy_version, input_content_hash, usage_input_tokens,
                usage_output_tokens, cost, status, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'completed', ?);
            """,
            (
                judgment_id,
                object_type,
                object_id,
                item.family,
                item.label_or_question_id,
                item.primitive,
                ans_json,
                item.probability,
                item.confidence,
                req_model,
                res_model,
                result.provider,
                result.provider_request_id,
                "1.0",
                "1.0",
                "1.0",
                content_hash,
                result.input_tokens,
                result.output_tokens,
                result.cost,
                now_iso,
            ),
        )
        saved_judgments.append(
            {"id": judgment_id, "family": item.family, "label": item.label_or_question_id}
        )

        # Apply Thresholds & Write to object_labels (without overwriting human labels)
        should_label = False
        target_label = item.metadata.get("target_label")
        label_to_apply = target_label or item.label_or_question_id

        if item.primitive == "choice":
            selected = item.answer.get("selected") or item.answer.get("choice")
            if selected and selected != "other":
                label_to_apply = selected
                should_label = True
                # If resource, update primary_form if not set or generic 'other'
                if object_type == "resource":
                    conn.execute(
                        "UPDATE resources SET primary_form = ? WHERE id = ? AND (primary_form IS NULL OR primary_form = 'other');",
                        (selected, object_id),
                    )
        elif item.primitive == "noul" and item.probability is not None:
            # Check threshold against target_label (e.g. ai/local-models) or question_id (e.g. topic-local-ai)
            threshold = resolve_threshold(
                thresholds, item.family, label_to_apply, default=default_threshold
            )
            if threshold == default_threshold and item.label_or_question_id != label_to_apply:
                threshold = resolve_threshold(
                    thresholds, item.family, item.label_or_question_id, default=default_threshold
                )
            if item.probability >= threshold:
                should_label = True

        if should_label:
            conn.execute(
                "INSERT OR IGNORE INTO label_families (id, description, created_at) VALUES (?, ?, ?);",
                (item.family, f"Taxonomic {item.family} labels", now_iso),
            )
            conn.execute(
                "INSERT OR IGNORE INTO labels (id, family, description, active, version, created_at) VALUES (?, ?, ?, 1, '1.0', ?);",
                (label_to_apply, item.family, f"Derived {label_to_apply}", now_iso),
            )
            lbl_obj_id = f"lbl_{object_id}_{label_to_apply}_classifier"
            conn.execute(
                """
                INSERT OR IGNORE INTO object_labels (id, object_type, object_id, label_id, source, confidence, created_at)
                VALUES (?, ?, ?, ?, 'classifier', ?, ?);
                """,
                (lbl_obj_id, object_type, object_id, label_to_apply, item.confidence, now_iso),
            )

    # Reproject FTS index
    reindex_object_document(conn, object_type, object_id)

    # Auto-complete any pending background classify job for this object
    conn.execute(
        """
        UPDATE processing_jobs
        SET status = 'completed', completed_at = ?, updated_at = ?
        WHERE stage = 'classify' AND status = 'pending'
          AND (resource_id = ? OR capture_id = ?);
        """,
        (now_iso, now_iso, object_id, object_id),
    )

    return saved_judgments


def run_classification_pipeline(
    conn: sqlite3.Connection,
    object_type: str,
    object_id: str,
    provider: str | None = None,
) -> list[dict]:
    """Run classifier on an object, persist raw judgments, apply thresholds, and update FTS projection."""
    target = load_classification_target(conn, object_type, object_id)
    if not target:
        return []

    form, result = classify_target(
        text=target["text"],
        url=target["url"],
        record_id=object_id,
        object_type=object_type,
        provider=provider,
        metadata=target.get("metadata"),
    )
    return persist_classification_result(conn, object_type, object_id, form, result, target["text"])
