-- Which captures have been checked against which project, and with what result.
--
-- Every capture is evaluated against a project once per brief: a row here means
-- "already considered", so later passes only look at new captures. Changing the
-- brief changes brief_hash, which makes every capture eligible again.

CREATE TABLE project_matches (
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    capture_id TEXT NOT NULL REFERENCES captures(id) ON DELETE CASCADE,
    brief_hash TEXT NOT NULL,
    object_id TEXT,
    similarity REAL NOT NULL,
    rank INTEGER,
    relevance REAL,
    outcome TEXT NOT NULL
        CHECK (outcome IN ('suggested', 'not-relevant', 'below-rank', 'member')),
    evaluated_at TIMESTAMP NOT NULL,
    PRIMARY KEY (project_id, capture_id)
);

CREATE INDEX idx_project_matches_outcome ON project_matches(project_id, outcome);
