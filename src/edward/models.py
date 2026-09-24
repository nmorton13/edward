"""Pydantic domain models and validation schemas for Edward."""

import datetime
import uuid
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


def generate_id(prefix: str) -> str:
    """Generate a prefixed random identifier."""
    # Use timestamp prefix + uuid hex for natural chronologic sorting
    ts = int(datetime.datetime.now(datetime.UTC).timestamp() * 1000)
    rand = uuid.uuid4().hex[:12]
    return f"{prefix}_{ts}_{rand}"


AssertionRole = Literal[
    "source-claim",
    "direct-quotation",
    "agent-conclusion",
    "personal-belief",
    "personal-observation",
    "question",
    "hypothesis",
    "connection",
]

ReviewState = Literal[
    "unreviewed",
    "reviewed",
    "approved",
    "corrected",
    "disputed",
    "dismissed",
    "superseded",
]


class CaptureInput(BaseModel):
    """Input payload for capturing an item into Edward."""

    url: str | None = None
    text: str | None = None
    note: str | None = None
    intent: str | None = None
    origin_namespace: str = "manual"
    origin_id: str | None = None
    collection_channel: str = "manual"
    collector: str = "edward-cli"
    collector_run_id: str | None = None
    acquisition_method: str = "direct-input"
    idempotency_key: str | None = None


class Capture(BaseModel):
    """Persisted capture event."""

    id: str
    origin_namespace: str
    origin_id: str | None = None
    collection_channel: str
    collector: str
    collector_run_id: str | None = None
    acquisition_method: str
    retrieved_at: datetime.datetime
    raw_content: str | None = None
    user_note: str | None = None
    review_state: ReviewState = "unreviewed"
    is_deleted: bool = False
    deleted_at: datetime.datetime | None = None
    created_at: datetime.datetime
    updated_at: datetime.datetime


class Resource(BaseModel):
    """Deduplicated primary resource."""

    id: str
    canonical_url: str | None = None
    url_hash: str | None = None
    identity_key: str | None = None
    title: str | None = None
    primary_form: str | None = None
    author: str | None = None
    published_at: datetime.datetime | None = None
    latest_content_hash: str | None = None
    review_state: ReviewState = "unreviewed"
    is_deleted: bool = False
    deleted_at: datetime.datetime | None = None
    created_at: datetime.datetime
    updated_at: datetime.datetime


class Annotation(BaseModel):
    """Append-only contextual note or correction on an object."""

    id: str
    object_type: str
    object_id: str
    annotation_type: str = "note"
    content: str
    author: str = "human"
    created_at: datetime.datetime


class Finding(BaseModel):
    """Extracted atomic finding."""

    id: str
    resource_id: str | None = None
    statement: str
    assertion_role: AssertionRole = "source-claim"
    agent_confidence: float | None = None
    extractor: str | None = None
    extractor_version: str | None = None
    source_content_hash: str | None = None
    review_state: ReviewState = "unreviewed"
    is_deleted: bool = False
    deleted_at: datetime.datetime | None = None
    created_at: datetime.datetime
    updated_at: datetime.datetime


class SearchItem(BaseModel):
    """Single item returned in search."""

    id: str
    object_type: str
    title: str | None = None
    body: str | None = None
    snippet: str | None = None
    score: float | None = None
    labels: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    created_at: str | None = None


class SearchResponse(BaseModel):
    """Complete search query response."""

    query: str
    count: int
    results: list[SearchItem]


class AuditEvent(BaseModel):
    """Append-only audit log entry."""

    id: str
    event_type: str
    object_type: str
    object_id: str
    actor: str
    payload_json: str | None = None
    created_at: datetime.datetime


class ProcessingJob(BaseModel):
    """Background queue processing job."""

    id: str
    job_key: str
    capture_id: str | None = None
    resource_id: str | None = None
    stage: str
    depends_on: str | None = None
    status: Literal["pending", "running", "completed", "failed"] = "pending"
    available_at: datetime.datetime
    started_at: datetime.datetime | None = None
    completed_at: datetime.datetime | None = None
    lease_owner: str | None = None
    lease_expires_at: datetime.datetime | None = None
    attempts: int = 0
    max_attempts: int = 3
    input_hash: str | None = None
    last_error: str | None = None
    created_at: datetime.datetime
    updated_at: datetime.datetime


class SourceItem(BaseModel):
    """Source item in research bundle."""

    model_config = ConfigDict(extra="allow")

    source_id: str | None = None
    identity_key: str | None = None
    origin: str = Field(min_length=1)
    origin_id: str | None = None
    url: str | None = None
    title: str | None = None
    retrieved_at: str | None = None
    extracted_text: str | None = None
    content_hash: str | None = None
    snapshot_path: str | None = None
    snapshot: str | None = None
    snapshot_hash: str | None = None
    acquired_by: dict[str, Any] | None = None

    @model_validator(mode="after")
    def ensure_source_identity(self) -> "SourceItem":
        if self.snapshot_path:
            raise ValueError(
                "Field 'snapshot_path' is rejected for security reasons; provide inline 'snapshot' string or 'extracted_text' instead."
            )
        if (
            not self.identity_key
            and not self.url
            and not self.source_id
            and not (self.origin and self.origin_id)
        ):
            raise ValueError(
                "SourceItem must specify at least one of 'identity_key', 'url', 'source_id', or ('origin' and 'origin_id')"
            )
        if not self.identity_key:
            if self.url:
                self.identity_key = f"url:{self.url}"
            elif self.source_id:
                self.identity_key = self.source_id
            elif self.origin and self.origin_id:
                self.identity_key = f"{self.origin}:{self.origin_id}"
        return self


def make_job_key(stage: str, target_id: str) -> str:
    """Construct a canonical deterministic job key for background processing."""
    return f"{stage}:{target_id}"


class FindingItem(BaseModel):
    """Finding item in research bundle."""

    model_config = ConfigDict(extra="allow")

    statement: str = Field(min_length=1)
    assertion_role: AssertionRole = "source-claim"
    source_url: str | None = None
    supporting_passage: str | None = None
    locator: dict[str, Any] | None = None
    agent_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    labels: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)


class ResearchBundle(BaseModel):
    """Structured research bundle schema v1."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["research-bundle"] = "research-bundle"
    schema_version: Literal["1"] = "1"
    bundle_id: str = Field(min_length=1)
    idempotency_key: str | None = None
    title: str = Field(min_length=1)
    brief: str | None = None
    agent: dict[str, Any] | None = None
    summary: str | None = None
    sources: list[SourceItem] = Field(min_length=1)
    findings: list[FindingItem] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    suggested_intents: list[str] = Field(default_factory=list)


class EvidenceItem(BaseModel):
    """Individual item in bounded evidence packet."""

    id: str
    kind: Literal["finding", "resource", "capture", "note"]
    text: str
    assertion_role: AssertionRole | None = None
    review_state: ReviewState = "unreviewed"
    confidence: float | None = None
    relevance_score: float | None = None
    source: dict[str, Any] | None = None
    supporting_passages: list[dict[str, Any]] = Field(default_factory=list)
    user_notes: list[str] = Field(default_factory=list)
    labels: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)


class EvidencePacket(BaseModel):
    """Bounded evidence packet schema v1."""

    type: Literal["evidence-packet"] = "evidence-packet"
    schema_version: Literal["1"] = "1"
    query: str
    created_at: str
    parameters: dict[str, Any] | None = None
    items: list[EvidenceItem] = Field(default_factory=list)
    gaps_and_disagreements: list[dict[str, Any]] = Field(default_factory=list)


ProjectMembershipStatus = Literal["candidate", "accepted", "rejected"]
ProjectRelationship = Literal[
    "evidence", "supporting", "counterargument", "question", "gap", "background"
]
OutlineStatus = Literal["proposal", "accepted", "rejected", "superseded"]
OutlineEvidenceRelationship = Literal["supporting", "counterevidence", "qualification"]


class OutlineEvidenceInput(BaseModel):
    """Evidence link proposed for an outline section."""

    object_id: str
    relationship: OutlineEvidenceRelationship = "supporting"
    relevance_note: str | None = None


class OutlineSectionInput(BaseModel):
    """A section in a proposed or revised outline."""

    heading: str = Field(min_length=1)
    purpose: str | None = None
    claim: str | None = None
    content: str | None = None
    notes: str | None = None
    unresolved_research_needs: list[str] = Field(default_factory=list)
    evidence: list[OutlineEvidenceInput] = Field(default_factory=list)


class OutlineProposalInput(BaseModel):
    """Strict structured output accepted from a model or calling agent."""

    title: str = Field(min_length=1)
    premise: str | None = None
    sections: list[OutlineSectionInput] = Field(min_length=1)


class Project(BaseModel):
    """Active research and writing workspace."""

    id: str
    title: str
    slug: str
    brief: str | None = None
    status: str = "active"
    is_deleted: bool = False
    created_at: datetime.datetime
    updated_at: datetime.datetime
