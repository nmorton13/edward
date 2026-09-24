"""Project workspaces, evidence membership, and versioned evidence-linked outlines."""

import datetime
import json
import re
import sqlite3
from typing import Any

from edward.models import OutlineProposalInput, Project, generate_id
from edward.services.audit import parse_timestamp, record_audit_event
from edward.services.hybrid import build_evidence_item
from edward.services.llm import LLMClient
from edward.services.privacy import (
    DataClass,
    assert_transmission_permitted,
    classify_content_data_class,
)
from edward.services.search import search_lexical


class ProjectError(Exception):
    """Base project workspace error."""


class ProjectNotFoundError(ProjectError):
    """Raised when a project is not active or does not exist."""


class ProjectConflictError(ProjectError):
    """Raised when a workspace operation conflicts with existing state."""


VALID_RELATIONSHIPS = {
    "evidence",
    "supporting",
    "counterargument",
    "question",
    "gap",
    "background",
}
VALID_MEMBERSHIP_STATUSES = {"candidate", "accepted", "rejected"}
VALID_OUTLINE_STATUSES = {"proposal", "accepted", "rejected", "superseded"}
VALID_EVIDENCE_RELATIONSHIPS = {"supporting", "counterevidence", "qualification"}

# Membership relationships describe how an object relates to a *workspace*.
# Outline evidence relationships describe how a link relates to a *section's claim*.
# They are deliberately different vocabularies, so a stored membership value must be
# mapped before it is sent to a model: an outline proposal is validated against the
# narrower evidence vocabulary, and an echoed 'evidence'/'counterargument' fails it.
_MEMBERSHIP_TO_EVIDENCE_RELATIONSHIP = {
    "counterargument": "counterevidence",
    "supporting": "supporting",
    "evidence": "supporting",
    "background": "qualification",
    "question": "qualification",
    "gap": "qualification",
}


def to_evidence_relationship(membership_relationship: str | None) -> str:
    """Map a stored membership relationship to a valid outline evidence relationship."""
    mapped = _MEMBERSHIP_TO_EVIDENCE_RELATIONSHIP.get((membership_relationship or "").strip())
    return mapped if mapped in VALID_EVIDENCE_RELATIONSHIPS else "supporting"


def _now() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat()


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "project"


def _active_project_row(conn: sqlite3.Connection, project_id: str) -> sqlite3.Row:
    row = conn.execute(
        "SELECT * FROM projects WHERE id = ? AND is_deleted = 0;", (project_id,)
    ).fetchone()
    if not row:
        raise ProjectNotFoundError(f"Project with ID '{project_id}' not found")
    return row


def _object_type(conn: sqlite3.Connection, object_id: str) -> str | None:
    if conn.execute(
        "SELECT 1 FROM captures WHERE id = ? AND is_deleted = 0;", (object_id,)
    ).fetchone():
        return "capture"
    if conn.execute(
        "SELECT 1 FROM resources WHERE id = ? AND is_deleted = 0;", (object_id,)
    ).fetchone():
        return "resource"
    if conn.execute(
        "SELECT 1 FROM findings WHERE id = ? AND is_deleted = 0;", (object_id,)
    ).fetchone():
        return "finding"
    return None


def create_project(
    conn: sqlite3.Connection,
    title: str,
    brief: str | None = None,
    slug: str | None = None,
    actor: str = "human",
) -> Project:
    """Create an active writing/research project with a stable unique slug."""
    title = title.strip()
    if not title:
        raise ValueError("Project title must not be empty")
    clean_slug = _slugify(slug or title)
    if conn.execute("SELECT 1 FROM projects WHERE slug = ?;", (clean_slug,)).fetchone():
        raise ProjectConflictError(f"Project slug '{clean_slug}' already exists")

    project_id = generate_id("prj")
    now = _now()
    clean_brief = brief.strip() if brief and brief.strip() else None
    conn.execute(
        """
        INSERT INTO projects (id, title, slug, description, status, is_deleted, created_at, updated_at)
        VALUES (?, ?, ?, ?, 'active', 0, ?, ?);
        """,
        (project_id, title, clean_slug, clean_brief, now, now),
    )
    record_audit_event(
        conn,
        event_type="project.created",
        object_type="project",
        object_id=project_id,
        actor=actor,
        payload={"title": title, "slug": clean_slug},
    )
    return Project(
        id=project_id,
        title=title,
        slug=clean_slug,
        brief=clean_brief,
        status="active",
        created_at=parse_timestamp(now),
        updated_at=parse_timestamp(now),
    )


def delete_project(
    conn: sqlite3.Connection,
    project_id: str,
    actor: str = "human",
) -> dict[str, Any]:
    """Soft-delete an active project workspace and record an audit event."""
    project = _active_project_row(conn, project_id)
    now = _now()
    conn.execute(
        """
        UPDATE projects
        SET is_deleted = 1, updated_at = ?
        WHERE id = ?;
        """,
        (now, project_id),
    )
    record_audit_event(
        conn,
        event_type="project.deleted",
        object_type="project",
        object_id=project_id,
        actor=actor,
        payload={"title": project["title"], "slug": project["slug"]},
    )
    return {
        "status": "deleted",
        "id": project_id,
        "title": project["title"],
    }


def add_project_object(
    conn: sqlite3.Connection,
    project_id: str,
    object_id: str,
    relationship: str = "evidence",
    membership_status: str = "accepted",
    added_by: str = "human",
    relevance_note: str | None = None,
) -> dict[str, Any]:
    """Add or explicitly transition an object membership without mutating corpus data."""
    _active_project_row(conn, project_id)
    if relationship not in VALID_RELATIONSHIPS:
        raise ValueError(f"Invalid project relationship: {relationship}")
    if membership_status not in VALID_MEMBERSHIP_STATUSES:
        raise ValueError(f"Invalid membership status: {membership_status}")
    object_type = _object_type(conn, object_id)
    if object_type is None:
        raise ValueError(f"Object with ID '{object_id}' not found")

    now = _now()
    existing = conn.execute(
        """
        SELECT id, added_by FROM project_objects
        WHERE project_id = ? AND object_type = ? AND object_id = ?;
        """,
        (project_id, object_type, object_id),
    ).fetchone()
    if existing:
        # Human decisions are durable: automated suggestions cannot demote or rewrite them.
        if existing["added_by"] == "human" and added_by != "human":
            row = conn.execute(
                "SELECT * FROM project_objects WHERE id = ?;", (existing["id"],)
            ).fetchone()
            return dict(row)
        conn.execute(
            """
            UPDATE project_objects
            SET relationship = ?, membership_status = ?, added_by = ?, relevance_note = ?
            WHERE id = ?;
            """,
            (relationship, membership_status, added_by, relevance_note, existing["id"]),
        )
        membership_id = existing["id"]
        event_type = "project.membership_updated"
    else:
        membership_id = generate_id("pob")
        conn.execute(
            """
            INSERT INTO project_objects
                (id, project_id, object_type, object_id, relationship, created_at,
                 membership_status, added_by, relevance_note)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                membership_id,
                project_id,
                object_type,
                object_id,
                relationship,
                now,
                membership_status,
                added_by,
                relevance_note,
            ),
        )
        event_type = "project.membership_added"

    conn.execute("UPDATE projects SET updated_at = ? WHERE id = ?;", (now, project_id))
    record_audit_event(
        conn,
        event_type=event_type,
        object_type="project",
        object_id=project_id,
        actor=added_by,
        payload={
            "membership_id": membership_id,
            "target_type": object_type,
            "target_id": object_id,
            "relationship": relationship,
            "membership_status": membership_status,
        },
    )
    row = conn.execute("SELECT * FROM project_objects WHERE id = ?;", (membership_id,)).fetchone()
    return dict(row)


def add_project_note(
    conn: sqlite3.Connection,
    project_id: str,
    content: str,
    kind: str = "note",
    author: str = "human",
) -> dict[str, Any]:
    """Append a workspace-local note, question, or gap."""
    _active_project_row(conn, project_id)
    if kind not in {"note", "question", "gap", "counterargument"}:
        raise ValueError(f"Invalid project note kind: {kind}")
    clean = content.strip()
    if not clean:
        raise ValueError("Project note content must not be empty")
    note_id = generate_id("ann")
    now = _now()
    conn.execute(
        """
        INSERT INTO annotations (id, object_type, object_id, annotation_type, content, author, created_at)
        VALUES (?, 'project', ?, ?, ?, ?, ?);
        """,
        (note_id, project_id, kind, clean, author, now),
    )
    conn.execute("UPDATE projects SET updated_at = ? WHERE id = ?;", (now, project_id))
    record_audit_event(
        conn,
        event_type="project.note_added",
        object_type="project",
        object_id=project_id,
        actor=author,
        payload={"annotation_id": note_id, "kind": kind},
    )
    return {
        "id": note_id,
        "project_id": project_id,
        "kind": kind,
        "content": clean,
        "author": author,
        "created_at": now,
    }


def suggest_project_evidence(
    conn: sqlite3.Connection, project_id: str, limit: int = 10, persist: bool = True
) -> list[dict[str, Any]]:
    """Retrieve lexical candidates from the project brief without invoking a model."""
    project = _active_project_row(conn, project_id)
    query = (project["description"] or project["title"]).strip()
    stopwords = {
        "about",
        "after",
        "before",
        "from",
        "have",
        "into",
        "that",
        "their",
        "there",
        "these",
        "they",
        "this",
        "what",
        "when",
        "where",
        "which",
        "with",
    }
    terms = [
        term.lower()
        for term in re.findall(r"[A-Za-z0-9_-]+", query)
        if len(term) >= 4 and term.lower() not in stopwords
    ]
    # FTS whitespace is conjunctive. Search the complete brief first, then its
    # significant terms so natural-language briefs do not produce false empties.
    ranked_items: list[Any] = []
    seen_search_ids: set[str] = set()
    for candidate_query in [query, *dict.fromkeys(terms)]:
        response = search_lexical(conn, query=candidate_query, limit=max(limit * 2, limit))
        for item in response.results:
            if item.id not in seen_search_ids:
                seen_search_ids.add(item.id)
                ranked_items.append(item)

    candidates: list[dict[str, Any]] = []
    for item in ranked_items:
        if len(candidates) >= limit:
            break
        if _object_type(conn, item.id) is None:
            continue
        existing = conn.execute(
            "SELECT membership_status FROM project_objects WHERE project_id = ? AND object_id = ?;",
            (project_id, item.id),
        ).fetchone()
        if existing:
            continue
        evidence = build_evidence_item(conn, item.id, relevance_score=item.score)
        if evidence is None:
            continue
        if persist:
            add_project_object(
                conn,
                project_id,
                item.id,
                relationship="evidence",
                membership_status="candidate",
                added_by="system",
                relevance_note="Suggested from lexical retrieval against the project brief.",
            )
        candidates.append(evidence)
    return candidates


def get_project_context(
    conn: sqlite3.Connection,
    project_id: str,
    include_rejected: bool = False,
) -> dict[str, Any]:
    """Return brief, evidence membership, notes, gaps, and latest outline."""
    project = _active_project_row(conn, project_id)
    rows = conn.execute(
        """
        SELECT * FROM project_objects
        WHERE project_id = ?
          AND (? = 1 OR membership_status != 'rejected')
        ORDER BY CASE membership_status WHEN 'accepted' THEN 0 WHEN 'candidate' THEN 1 ELSE 2 END,
                 created_at ASC;
        """,
        (project_id, 1 if include_rejected else 0),
    ).fetchall()
    memberships: list[dict[str, Any]] = []
    for row in rows:
        evidence = build_evidence_item(conn, row["object_id"])
        if evidence is None:
            continue
        memberships.append(
            {
                "membership_id": row["id"],
                "status": row["membership_status"],
                "relationship": row["relationship"],
                "added_by": row["added_by"],
                "relevance_note": row["relevance_note"],
                "item": evidence,
            }
        )

    note_rows = conn.execute(
        """
        SELECT id, annotation_type, content, author, created_at
        FROM annotations WHERE object_type = 'project' AND object_id = ?
        ORDER BY created_at ASC;
        """,
        (project_id,),
    ).fetchall()
    notes = [dict(row) for row in note_rows]
    gaps: list[dict[str, Any]] = [
        {"kind": row["annotation_type"], "description": row["content"], "id": row["id"]}
        for row in note_rows
        if row["annotation_type"] in {"gap", "question", "counterargument"}
    ]
    accepted = [m for m in memberships if m["status"] == "accepted"]
    if not accepted:
        gaps.append(
            {"kind": "gap", "description": "No evidence has been accepted for this project."}
        )
    if not any(
        m["relationship"] == "counterargument"
        or m["item"].get("assertion_role") == "counterargument"
        for m in memberships
    ):
        gaps.append(
            {
                "kind": "gap",
                "description": "No counterargument or conflicting evidence is represented yet.",
            }
        )
    if accepted and all(m["item"].get("review_state") == "unreviewed" for m in accepted):
        gaps.append({"kind": "uncertainty", "description": "All accepted evidence is unreviewed."})

    outline = get_outline(conn, project_id, version=None, required=False)
    return {
        "project": {
            "id": project["id"],
            "title": project["title"],
            "slug": project["slug"],
            "brief": project["description"],
            "status": project["status"],
            "created_at": project["created_at"],
            "updated_at": project["updated_at"],
        },
        "memberships": memberships,
        "accepted_evidence": [m for m in memberships if m["status"] == "accepted"],
        "candidate_evidence": [m for m in memberships if m["status"] == "candidate"],
        "notes": notes,
        "gaps_and_counterarguments": gaps,
        "latest_outline": outline,
    }


def deterministic_outline(context: dict[str, Any]) -> OutlineProposalInput:
    """Build a useful offline outline proposal from workspace state.

    The offline path has no model, so it cannot know what a source *says*. What it does
    know is the relationship each membership was filed under, and that is the only
    honest signal available for assignment. Route by that signal rather than by slice
    position: a flat list sliced positionally files unrelated items together and, worse,
    silently drops every membership that is not supporting or countering.

    What this deliberately does NOT do: infer an item's stance from its text. Keyword
    matching over source prose would be guesswork dressed up as analysis, and a wrong
    assignment is harder to notice than a missing one. Choosing which claim a source
    serves is the calling agent's job (`--input`); this keeps evidence reachable and
    honestly labelled until then.
    """
    project = context["project"]
    all_evidence = context["accepted_evidence"] + context["candidate_evidence"]
    gaps = [g["description"] for g in context["gaps_and_counterarguments"]]

    # Group by the *evidence* vocabulary, so a stored 'background'/'gap'/'question'
    # membership (which maps to 'qualification') is placed instead of silently dropped.
    grouped: dict[str, list[dict[str, Any]]] = {
        "supporting": [],
        "counterevidence": [],
        "qualification": [],
    }
    for membership in all_evidence:
        relationship = to_evidence_relationship(membership["relationship"])
        grouped.setdefault(relationship, []).append(membership)

    def links(items: list[dict[str, Any]], relationship: str) -> list[dict[str, Any]]:
        return [
            {
                "object_id": m["item"]["id"],
                "relationship": relationship,
                "relevance_note": m.get("relevance_note"),
            }
            for m in items
        ]

    supporting = grouped["supporting"]
    countering = grouped["counterevidence"]
    qualifying = grouped["qualification"]

    # Every need appears exactly once: at the first section that can host one, plus a
    # trailing pointer when there are more sections wanting needs than needs. Repeating
    # the same list in every section is noise that buries the signal.
    def needs_for(index: int) -> list[str]:
        host_sections = 2  # Counterarguments, Implications
        if not gaps:
            return []
        if index >= host_sections:
            return []
        chunk = -(-len(gaps) // host_sections)  # ceil
        return gaps[index * chunk : (index + 1) * chunk]

    return OutlineProposalInput.model_validate(
        {
            "title": project["title"],
            "premise": project["brief"],
            "sections": [
                {
                    "heading": "Premise and stakes",
                    "purpose": "Introduce the question, scope, and why it matters.",
                    "claim": project["brief"],
                    "evidence": links(supporting[:2], "supporting"),
                },
                {
                    "heading": "What the evidence shows",
                    "purpose": "Develop the strongest evidence-backed parts of the argument.",
                    "evidence": links(supporting[2:], "supporting"),
                },
                {
                    "heading": "Counterarguments and qualifications",
                    "purpose": "Test the premise against conflicts, limitations, and alternative views.",
                    "unresolved_research_needs": needs_for(0),
                    "evidence": links(countering, "counterevidence")
                    + links(qualifying, "qualification"),
                },
                {
                    "heading": "Implications and open questions",
                    "purpose": "Synthesize implications without overstating the evidence.",
                    "unresolved_research_needs": needs_for(1),
                    "evidence": [],
                },
            ],
        }
    )


def collect_project_context_data_classes(context: dict[str, Any]) -> set[DataClass]:
    """Collect every data class represented in the project workspace (brief, notes, and evidence)."""
    classes: set[DataClass] = {"personal_notes"}
    for m in context.get("memberships", []):
        it = m.get("item") or {}
        src = it.get("source") or {}
        ns = (it.get("origin_namespace") or src.get("origin_namespace") or "").lower()
        kind = it.get("kind") or ""
        url = src.get("url") or it.get("canonical_url") or ""
        form = it.get("form") or ""
        meta = it.get("metadata") or src.get("metadata") or {}

        # 1. Classify underlying source content independently
        source_class: DataClass
        if ns == "gmail" or meta.get("source_type") == "gmail":
            source_class = "gmail"
        elif (
            ns in ("documents", "document", "file", "attachment")
            or ns.startswith("file:")
            or meta.get("source_type") in ("file", "attachment", "document")
            or url.lower().startswith(("file://", "/"))
        ):
            source_class = "documents"
        elif (
            ns in ("personal_notes", "notes", "manual") or kind == "note" or form == "personal-note"
        ):
            source_class = "personal_notes"
        else:
            source_class = classify_content_data_class(
                origin_namespace=ns,
                canonical_url=url,
                form=form,
                metadata=meta,
            )
        classes.add(source_class)

        # 2. Separately add personal_notes when human notes/annotations are attached
        notes = it.get("user_notes") or []
        user_note = it.get("user_note")
        relevance_note = m.get("relevance_note")
        if notes or user_note or relevance_note:
            classes.add("personal_notes")

    return classes


def generate_outline_proposal(
    context: dict[str, Any],
    llm_client: LLMClient | None = None,
    revision_instructions: str | None = None,
) -> OutlineProposalInput:
    """Generate a strict outline with a configured model, or deterministically offline."""
    if llm_client is None:
        return deterministic_outline(context)

    # Privacy check MUST run and halt before prompt construction or payload serialization.
    # Every data class represented across brief, notes, and evidence is evaluated independently.
    data_classes = collect_project_context_data_classes(context)
    for dc in sorted(data_classes):
        assert_transmission_permitted(
            provider_name=llm_client.provider,
            data_class=dc,
            declared_location=llm_client.location,
            base_url=llm_client.base_url,
        )

    compact = {
        "project": context["project"],
        "evidence": [
            {
                # Name the field as the schema requires. A model echoes what it is given,
                # so a bare `id` here produces an evidence link the schema rejects.
                "object_id": m["item"]["id"],
                "object_type": m["item"].get("kind"),
                "text": m["item"].get("text", "")[:1200],
                "assertion_role": m["item"].get("assertion_role"),
                "review_state": m["item"].get("review_state"),
                "membership_status": m["status"],
                # Map to the outline evidence vocabulary so an echoed value validates.
                "relationship": to_evidence_relationship(m["relationship"]),
            }
            for m in context["memberships"][:20]
        ],
        "workspace_notes": context["notes"],
        "gaps": context["gaps_and_counterarguments"],
        "latest_outline": context.get("latest_outline"),
        "revision_instructions": revision_instructions,
    }
    system = (
        "Create a rigorous writing outline using only the supplied Edward workspace. "
        "Return JSON matching the requested schema. Cite evidence only by object_id values "
        "present in the supplied evidence. Keep gaps as unresolved_research_needs; do not invent facts."
    )
    _, parsed = llm_client.chat_completion(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(compact, ensure_ascii=False)},
        ],
        response_model=OutlineProposalInput,
        data_class=data_classes,
        temperature=0.2,
    )
    if parsed is None:
        raise ProjectError("Outline model returned no structured proposal")
    return parsed


def save_outline(
    conn: sqlite3.Connection,
    project_id: str,
    proposal: OutlineProposalInput | dict[str, Any],
    author_type: str = "human",
    author_id: str | None = None,
    parent_outline_id: str | None = None,
    revision_instructions: str | None = None,
) -> dict[str, Any]:
    """Persist an immutable outline version after validating every evidence link."""
    _active_project_row(conn, project_id)
    parsed = (
        proposal
        if isinstance(proposal, OutlineProposalInput)
        else OutlineProposalInput.model_validate(proposal)
    )
    allowed_rows = conn.execute(
        """
        SELECT object_id FROM project_objects
        WHERE project_id = ? AND membership_status IN ('accepted', 'candidate');
        """,
        (project_id,),
    ).fetchall()
    allowed_ids = {row["object_id"] for row in allowed_rows}
    for section in parsed.sections:
        for evidence in section.evidence:
            if evidence.object_id not in allowed_ids:
                raise ValueError(
                    f"Outline evidence '{evidence.object_id}' is not accepted or candidate project evidence"
                )

    if parent_outline_id:
        parent = conn.execute(
            "SELECT id FROM outlines WHERE id = ? AND project_id = ?;",
            (parent_outline_id, project_id),
        ).fetchone()
        if not parent:
            raise ValueError("Parent outline does not belong to this project")

    version = conn.execute(
        "SELECT COALESCE(MAX(version), 0) + 1 FROM outlines WHERE project_id = ?;",
        (project_id,),
    ).fetchone()[0]
    outline_id = generate_id("out")
    now = _now()
    conn.execute(
        """
        INSERT INTO outlines
            (id, project_id, version, title, premise, author_type, author_id, created_at,
             status, parent_outline_id, revision_instructions)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'proposal', ?, ?);
        """,
        (
            outline_id,
            project_id,
            version,
            parsed.title,
            parsed.premise,
            author_type,
            author_id,
            now,
            parent_outline_id,
            revision_instructions,
        ),
    )
    for order, section in enumerate(parsed.sections, start=1):
        section_id = generate_id("sec")
        conn.execute(
            """
            INSERT INTO outline_sections
                (id, outline_id, section_order, heading, content, notes, created_at,
                 purpose, claim, unresolved_needs_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                section_id,
                outline_id,
                order,
                section.heading,
                section.content,
                section.notes,
                now,
                section.purpose,
                section.claim,
                json.dumps(section.unresolved_research_needs),
            ),
        )
        for evidence in section.evidence:
            object_type = _object_type(conn, evidence.object_id)
            finding_id = evidence.object_id if object_type == "finding" else None
            resource_id = evidence.object_id if object_type == "resource" else None
            conn.execute(
                """
                INSERT INTO outline_section_evidence
                    (id, section_id, finding_id, resource_id, relevance_note, created_at,
                     relationship, object_type, object_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    generate_id("ose"),
                    section_id,
                    finding_id,
                    resource_id,
                    evidence.relevance_note,
                    now,
                    evidence.relationship,
                    object_type,
                    evidence.object_id,
                ),
            )

    conn.execute("UPDATE projects SET updated_at = ? WHERE id = ?;", (now, project_id))
    record_audit_event(
        conn,
        event_type="project.outline_created",
        object_type="project",
        object_id=project_id,
        actor=author_id or author_type,
        payload={
            "outline_id": outline_id,
            "version": version,
            "parent_outline_id": parent_outline_id,
        },
    )
    result = get_outline(conn, project_id, version=version)
    assert result is not None
    return result


def get_outline(
    conn: sqlite3.Connection,
    project_id: str,
    version: int | None = None,
    required: bool = True,
) -> dict[str, Any] | None:
    """Load a complete outline version; latest is returned when version is omitted."""
    if version is None:
        outline = conn.execute(
            "SELECT * FROM outlines WHERE project_id = ? ORDER BY version DESC LIMIT 1;",
            (project_id,),
        ).fetchone()
    else:
        outline = conn.execute(
            "SELECT * FROM outlines WHERE project_id = ? AND version = ?;",
            (project_id, version),
        ).fetchone()
    if not outline:
        if required:
            raise ProjectNotFoundError(f"No outline found for project '{project_id}'")
        return None

    sections = []
    for section in conn.execute(
        "SELECT * FROM outline_sections WHERE outline_id = ? ORDER BY section_order ASC;",
        (outline["id"],),
    ).fetchall():
        evidence_rows = conn.execute(
            """
            SELECT id, finding_id, resource_id, relevance_note, relationship,
                   object_type, object_id
            FROM outline_section_evidence WHERE section_id = ? ORDER BY created_at ASC;
            """,
            (section["id"],),
        ).fetchall()
        evidence = []
        for row in evidence_rows:
            evidence.append(
                {
                    "id": row["id"],
                    "object_id": row["object_id"] or row["finding_id"] or row["resource_id"],
                    "object_type": row["object_type"],
                    "relationship": row["relationship"],
                    "relevance_note": row["relevance_note"],
                }
            )
        sections.append(
            {
                "id": section["id"],
                "order": section["section_order"],
                "heading": section["heading"],
                "purpose": section["purpose"],
                "claim": section["claim"],
                "content": section["content"],
                "notes": section["notes"],
                "unresolved_research_needs": json.loads(section["unresolved_needs_json"] or "[]"),
                "evidence": evidence,
            }
        )
    data = dict(outline)
    data["sections"] = sections
    return data


def set_outline_status(
    conn: sqlite3.Connection,
    project_id: str,
    status: str,
    version: int | None = None,
    actor: str = "human",
) -> dict[str, Any]:
    """Accept or reject an outline proposal while retaining every version."""
    if status not in {"accepted", "rejected"}:
        raise ValueError("Outline status must be 'accepted' or 'rejected'")
    outline = get_outline(conn, project_id, version=version)
    assert outline is not None
    if status == "accepted":
        conn.execute(
            """
            UPDATE outlines SET status = 'superseded'
            WHERE project_id = ? AND status = 'accepted' AND id != ?;
            """,
            (project_id, outline["id"]),
        )
    conn.execute("UPDATE outlines SET status = ? WHERE id = ?;", (status, outline["id"]))
    record_audit_event(
        conn,
        event_type=f"project.outline_{status}",
        object_type="project",
        object_id=project_id,
        actor=actor,
        payload={"outline_id": outline["id"], "version": outline["version"]},
    )
    result = get_outline(conn, project_id, version=outline["version"])
    assert result is not None
    return result
