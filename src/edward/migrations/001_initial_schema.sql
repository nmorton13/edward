-- 001_initial_schema.sql
-- Master schema for Edward: Personal Research Memory and Writing Workspace

-- Schema migrations tracker
CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    applied_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Captures: Individual capture events preserving contextual notes and provenance
CREATE TABLE captures (
    id TEXT PRIMARY KEY,
    origin_namespace TEXT NOT NULL,
    origin_id TEXT,
    collection_channel TEXT NOT NULL,
    collector TEXT NOT NULL,
    collector_run_id TEXT,
    acquisition_method TEXT NOT NULL,
    retrieved_at TIMESTAMP NOT NULL,
    raw_content TEXT,
    user_note TEXT,
    review_state TEXT NOT NULL DEFAULT 'unreviewed',
    is_deleted INTEGER NOT NULL DEFAULT 0,
    deleted_at TIMESTAMP,
    created_at TIMESTAMP NOT NULL,
    updated_at TIMESTAMP NOT NULL
);

CREATE INDEX idx_captures_origin ON captures(origin_namespace, origin_id);
CREATE INDEX idx_captures_created ON captures(created_at DESC);
CREATE INDEX idx_captures_deleted ON captures(is_deleted);

-- Resources: Deduplicated underlying assets (web pages, repositories, papers, local files, notes)
CREATE TABLE resources (
    id TEXT PRIMARY KEY,
    identity_key TEXT NOT NULL UNIQUE,
    canonical_url TEXT,
    url_hash TEXT,
    title TEXT,
    primary_form TEXT,
    author TEXT,
    published_at TIMESTAMP,
    latest_content_hash TEXT,
    review_state TEXT NOT NULL DEFAULT 'unreviewed',
    is_deleted INTEGER NOT NULL DEFAULT 0,
    deleted_at TIMESTAMP,
    created_at TIMESTAMP NOT NULL,
    updated_at TIMESTAMP NOT NULL
);

CREATE UNIQUE INDEX idx_resources_canonical_url ON resources(canonical_url) WHERE canonical_url IS NOT NULL;
CREATE UNIQUE INDEX idx_resources_url_hash ON resources(url_hash) WHERE url_hash IS NOT NULL;
CREATE INDEX idx_resources_deleted ON resources(is_deleted);

-- Join table linking contextual captures to underlying resources
CREATE TABLE capture_resources (
    capture_id TEXT NOT NULL REFERENCES captures(id) ON DELETE CASCADE,
    resource_id TEXT NOT NULL REFERENCES resources(id) ON DELETE CASCADE,
    relationship_type TEXT NOT NULL DEFAULT 'primary',
    created_at TIMESTAMP NOT NULL,
    PRIMARY KEY (capture_id, resource_id)
);

CREATE INDEX idx_capture_resources_resource ON capture_resources(resource_id);

-- Raw source snapshots with strict header allowlist
CREATE TABLE source_snapshots (
    id TEXT PRIMARY KEY,
    resource_id TEXT NOT NULL REFERENCES resources(id) ON DELETE CASCADE,
    content_hash TEXT NOT NULL,
    headers_json TEXT,
    blob_path TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    created_at TIMESTAMP NOT NULL
);

CREATE INDEX idx_source_snapshots_resource ON source_snapshots(resource_id);
CREATE INDEX idx_source_snapshots_hash ON source_snapshots(content_hash);

-- Extracted clean text and summaries
CREATE TABLE resource_contents (
    id TEXT PRIMARY KEY,
    resource_id TEXT NOT NULL REFERENCES resources(id) ON DELETE CASCADE,
    content_hash TEXT NOT NULL,
    clean_text TEXT NOT NULL,
    summary TEXT,
    char_count INTEGER NOT NULL,
    extractor TEXT NOT NULL,
    extractor_version TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL
);

CREATE INDEX idx_resource_contents_resource ON resource_contents(resource_id);
CREATE INDEX idx_resource_contents_hash ON resource_contents(content_hash);

-- Searchable and citeable text chunks
CREATE TABLE resource_chunks (
    id TEXT PRIMARY KEY,
    resource_content_id TEXT NOT NULL REFERENCES resource_contents(id) ON DELETE CASCADE,
    resource_id TEXT NOT NULL REFERENCES resources(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    text TEXT NOT NULL,
    locator_json TEXT,
    token_count INTEGER,
    created_at TIMESTAMP NOT NULL
);

CREATE INDEX idx_resource_chunks_content ON resource_chunks(resource_content_id);
CREATE INDEX idx_resource_chunks_resource ON resource_chunks(resource_id);

-- Content-addressed binary attachments
CREATE TABLE attachments (
    id TEXT PRIMARY KEY,
    object_type TEXT NOT NULL,
    object_id TEXT NOT NULL,
    file_name TEXT NOT NULL,
    mime_type TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    blob_path TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL
);

CREATE INDEX idx_attachments_object ON attachments(object_type, object_id);
CREATE INDEX idx_attachments_hash ON attachments(content_hash);

-- Extracted atomic findings
CREATE TABLE findings (
    id TEXT PRIMARY KEY,
    resource_id TEXT REFERENCES resources(id) ON DELETE SET NULL,
    statement TEXT NOT NULL,
    assertion_role TEXT NOT NULL DEFAULT 'source-claim',
    agent_confidence REAL,
    extractor TEXT,
    extractor_version TEXT,
    source_content_hash TEXT,
    review_state TEXT NOT NULL DEFAULT 'unreviewed',
    is_deleted INTEGER NOT NULL DEFAULT 0,
    deleted_at TIMESTAMP,
    created_at TIMESTAMP NOT NULL,
    updated_at TIMESTAMP NOT NULL
);

CREATE INDEX idx_findings_resource ON findings(resource_id);
CREATE INDEX idx_findings_assertion_role ON findings(assertion_role);
CREATE INDEX idx_findings_deleted ON findings(is_deleted);

-- Passages and locators supporting findings
CREATE TABLE finding_support (
    id TEXT PRIMARY KEY,
    finding_id TEXT NOT NULL REFERENCES findings(id) ON DELETE CASCADE,
    passage TEXT NOT NULL,
    locator_json TEXT,
    content_hash TEXT,
    created_at TIMESTAMP NOT NULL
);

CREATE INDEX idx_finding_support_finding ON finding_support(finding_id);

-- Controlled taxonomic label families and labels
CREATE TABLE label_families (
    id TEXT PRIMARY KEY,
    description TEXT,
    created_at TIMESTAMP NOT NULL
);

CREATE TABLE labels (
    id TEXT PRIMARY KEY,
    family TEXT NOT NULL REFERENCES label_families(id),
    description TEXT,
    parent TEXT,
    active INTEGER NOT NULL DEFAULT 1,
    version TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL
);

CREATE INDEX idx_labels_family ON labels(family);

-- Object to label assignments with provenance
CREATE TABLE object_labels (
    id TEXT PRIMARY KEY,
    object_type TEXT NOT NULL,
    object_id TEXT NOT NULL,
    label_id TEXT NOT NULL REFERENCES labels(id),
    source TEXT NOT NULL,
    confidence REAL,
    created_at TIMESTAMP NOT NULL,
    UNIQUE (object_type, object_id, label_id, source)
);

CREATE INDEX idx_object_labels_object ON object_labels(object_type, object_id);
CREATE INDEX idx_object_labels_label ON object_labels(label_id);

-- Entities (people, companies, models, hardware, software)
CREATE TABLE entities (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    normalized_name TEXT NOT NULL,
    entity_type TEXT,
    canonical_id TEXT REFERENCES entities(id),
    created_at TIMESTAMP NOT NULL
);

CREATE INDEX idx_entities_normalized ON entities(normalized_name);

-- Object to entity associations
CREATE TABLE object_entities (
    id TEXT PRIMARY KEY,
    object_type TEXT NOT NULL,
    object_id TEXT NOT NULL,
    entity_id TEXT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    extractor TEXT NOT NULL,
    extractor_version TEXT,
    source_content_hash TEXT,
    confidence REAL,
    review_state TEXT NOT NULL DEFAULT 'unreviewed',
    created_at TIMESTAMP NOT NULL,
    UNIQUE (object_type, object_id, entity_id)
);

CREATE INDEX idx_object_entities_object ON object_entities(object_type, object_id);
CREATE INDEX idx_object_entities_entity ON object_entities(entity_id);

-- Intents (essay-seed, reference, follow-up, etc.)
CREATE TABLE intents (
    id TEXT PRIMARY KEY,
    object_type TEXT NOT NULL,
    object_id TEXT NOT NULL,
    intent TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT 'human',
    is_active INTEGER NOT NULL DEFAULT 1,
    created_at TIMESTAMP NOT NULL,
    updated_at TIMESTAMP NOT NULL,
    UNIQUE (object_type, object_id, intent)
);

CREATE INDEX idx_intents_object ON intents(object_type, object_id);
CREATE INDEX idx_intents_active ON intents(intent, is_active);

-- Append-only contextual annotations and notes
CREATE TABLE annotations (
    id TEXT PRIMARY KEY,
    object_type TEXT NOT NULL,
    object_id TEXT NOT NULL,
    annotation_type TEXT NOT NULL DEFAULT 'note',
    content TEXT NOT NULL,
    author TEXT NOT NULL DEFAULT 'human',
    created_at TIMESTAMP NOT NULL
);

CREATE INDEX idx_annotations_object ON annotations(object_type, object_id);
CREATE INDEX idx_annotations_created ON annotations(created_at ASC);

-- Projects and writing workspaces
CREATE TABLE projects (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    slug TEXT NOT NULL UNIQUE,
    description TEXT,
    status TEXT NOT NULL DEFAULT 'active',
    is_deleted INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMP NOT NULL,
    updated_at TIMESTAMP NOT NULL
);

CREATE TABLE project_objects (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    object_type TEXT NOT NULL,
    object_id TEXT NOT NULL,
    relationship TEXT NOT NULL DEFAULT 'evidence',
    created_at TIMESTAMP NOT NULL,
    UNIQUE (project_id, object_type, object_id)
);

CREATE INDEX idx_project_objects_project ON project_objects(project_id);
CREATE INDEX idx_project_objects_target ON project_objects(object_type, object_id);

-- Outlines and outline sections with evidence links
CREATE TABLE outlines (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    version INTEGER NOT NULL,
    title TEXT NOT NULL,
    premise TEXT,
    author_type TEXT NOT NULL,
    author_id TEXT,
    created_at TIMESTAMP NOT NULL,
    UNIQUE (project_id, version)
);

CREATE TABLE outline_sections (
    id TEXT PRIMARY KEY,
    outline_id TEXT NOT NULL REFERENCES outlines(id) ON DELETE CASCADE,
    section_order INTEGER NOT NULL,
    heading TEXT NOT NULL,
    content TEXT,
    notes TEXT,
    created_at TIMESTAMP NOT NULL
);

CREATE INDEX idx_outline_sections_outline ON outline_sections(outline_id);

CREATE TABLE outline_section_evidence (
    id TEXT PRIMARY KEY,
    section_id TEXT NOT NULL REFERENCES outline_sections(id) ON DELETE CASCADE,
    finding_id TEXT REFERENCES findings(id) ON DELETE CASCADE,
    resource_id TEXT REFERENCES resources(id) ON DELETE CASCADE,
    relevance_note TEXT,
    created_at TIMESTAMP NOT NULL
);

CREATE INDEX idx_outline_section_evidence_section ON outline_section_evidence(section_id);

-- Structured classifier judgments (Jev / local)
CREATE TABLE judgments (
    id TEXT PRIMARY KEY,
    object_type TEXT NOT NULL,
    object_id TEXT NOT NULL,
    family TEXT NOT NULL,
    label_or_question_id TEXT NOT NULL,
    primitive TEXT NOT NULL,
    answer_json TEXT NOT NULL,
    probability REAL,
    confidence REAL,
    requested_model TEXT NOT NULL,
    resolved_model TEXT NOT NULL,
    provider TEXT NOT NULL,
    provider_request_id TEXT,
    question_registry_version TEXT NOT NULL,
    label_registry_version TEXT,
    threshold_policy_version TEXT NOT NULL,
    input_content_hash TEXT NOT NULL,
    usage_input_tokens INTEGER,
    usage_output_tokens INTEGER,
    cost REAL,
    status TEXT NOT NULL DEFAULT 'completed',
    error TEXT,
    created_at TIMESTAMP NOT NULL
);

CREATE INDEX idx_judgments_object ON judgments(object_type, object_id);
CREATE INDEX idx_judgments_question ON judgments(label_or_question_id);

-- Optional vector embeddings
CREATE TABLE embeddings (
    id TEXT PRIMARY KEY,
    object_type TEXT NOT NULL,
    object_id TEXT NOT NULL,
    model TEXT NOT NULL,
    dimensions INTEGER NOT NULL,
    embedding_blob BLOB NOT NULL,
    input_hash TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL,
    UNIQUE (object_type, object_id, model)
);

CREATE INDEX idx_embeddings_object ON embeddings(object_type, object_id);

-- Scaffolding for research runs and tasks
CREATE TABLE research_runs (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    brief TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    created_at TIMESTAMP NOT NULL,
    completed_at TIMESTAMP
);

CREATE TABLE research_tasks (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES research_runs(id) ON DELETE CASCADE,
    task_order INTEGER NOT NULL,
    task_type TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    result_json TEXT,
    created_at TIMESTAMP NOT NULL
);

-- Idempotency keys with strict hash conflict detection
CREATE TABLE idempotency_keys (
    id TEXT PRIMARY KEY,
    namespace TEXT NOT NULL,
    operation TEXT NOT NULL,
    key TEXT NOT NULL,
    result_object_id TEXT NOT NULL,
    request_hash TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL,
    UNIQUE (namespace, operation, key)
);

CREATE INDEX idx_idempotency_lookup ON idempotency_keys(namespace, operation, key);

-- Background and offline processing jobs with lease recovery
CREATE TABLE processing_jobs (
    id TEXT PRIMARY KEY,
    job_key TEXT NOT NULL UNIQUE,
    capture_id TEXT REFERENCES captures(id) ON DELETE CASCADE,
    resource_id TEXT REFERENCES resources(id) ON DELETE CASCADE,
    stage TEXT NOT NULL,
    depends_on TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    available_at TIMESTAMP NOT NULL,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    lease_owner TEXT,
    lease_expires_at TIMESTAMP,
    attempts INTEGER NOT NULL DEFAULT 0,
    max_attempts INTEGER NOT NULL DEFAULT 3,
    input_hash TEXT,
    last_error TEXT,
    created_at TIMESTAMP NOT NULL,
    updated_at TIMESTAMP NOT NULL
);

CREATE INDEX idx_processing_jobs_status_available ON processing_jobs(status, available_at);
CREATE INDEX idx_processing_jobs_lease ON processing_jobs(status, lease_expires_at);

-- Append-only audit events log
CREATE TABLE audit_events (
    id TEXT PRIMARY KEY,
    event_type TEXT NOT NULL,
    object_type TEXT NOT NULL,
    object_id TEXT NOT NULL,
    actor TEXT NOT NULL,
    payload_json TEXT,
    created_at TIMESTAMP NOT NULL
);

CREATE INDEX idx_audit_events_object ON audit_events(object_type, object_id);
CREATE INDEX idx_audit_events_created ON audit_events(created_at DESC);

-- Source adapter synchronization cursors
CREATE TABLE source_cursors (
    source TEXT PRIMARY KEY,
    cursor_value TEXT NOT NULL,
    updated_at TIMESTAMP NOT NULL
);

-- Full-Text Search 5 virtual table (unicode61 tokenization)
CREATE VIRTUAL TABLE search_documents USING fts5(
    object_type UNINDEXED,
    object_id UNINDEXED,
    title,
    body,
    labels,
    entities,
    tokenize = 'unicode61 remove_diacritics 2'
);
