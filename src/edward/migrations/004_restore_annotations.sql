-- Restore the annotations table for databases created before it was included in the initial schema.
CREATE TABLE IF NOT EXISTS annotations (
    id TEXT PRIMARY KEY,
    object_type TEXT NOT NULL,
    object_id TEXT NOT NULL,
    annotation_type TEXT NOT NULL DEFAULT 'note',
    content TEXT NOT NULL,
    author TEXT NOT NULL DEFAULT 'human',
    created_at TIMESTAMP NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_annotations_object ON annotations(object_type, object_id);
CREATE INDEX IF NOT EXISTS idx_annotations_created ON annotations(created_at ASC);
