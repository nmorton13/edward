-- Phase 4: project workspace state and versioned outline metadata

ALTER TABLE project_objects ADD COLUMN membership_status TEXT NOT NULL DEFAULT 'accepted'
    CHECK (membership_status IN ('candidate', 'accepted', 'rejected'));
ALTER TABLE project_objects ADD COLUMN added_by TEXT NOT NULL DEFAULT 'human';
ALTER TABLE project_objects ADD COLUMN relevance_note TEXT;

ALTER TABLE outlines ADD COLUMN status TEXT NOT NULL DEFAULT 'proposal'
    CHECK (status IN ('proposal', 'accepted', 'rejected', 'superseded'));
ALTER TABLE outlines ADD COLUMN parent_outline_id TEXT REFERENCES outlines(id) ON DELETE SET NULL;
ALTER TABLE outlines ADD COLUMN revision_instructions TEXT;

ALTER TABLE outline_sections ADD COLUMN purpose TEXT;
ALTER TABLE outline_sections ADD COLUMN claim TEXT;
ALTER TABLE outline_sections ADD COLUMN unresolved_needs_json TEXT;

ALTER TABLE outline_section_evidence ADD COLUMN relationship TEXT NOT NULL DEFAULT 'supporting'
    CHECK (relationship IN ('supporting', 'counterevidence', 'qualification'));
ALTER TABLE outline_section_evidence ADD COLUMN object_type TEXT;
ALTER TABLE outline_section_evidence ADD COLUMN object_id TEXT;

CREATE INDEX idx_project_objects_status ON project_objects(project_id, membership_status);
CREATE INDEX idx_outlines_project_status ON outlines(project_id, status, version DESC);
