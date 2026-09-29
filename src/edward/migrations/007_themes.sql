-- Themes: named groups of captures derived from their embeddings.
--
-- Derived data: membership and centroids can always be rebuilt from embeddings.
-- The one exception is a human-assigned name (name_source = 'human'), which
-- automated rebuilds and renaming must never overwrite.

CREATE TABLE themes (
    id TEXT PRIMARY KEY,
    name TEXT,
    description TEXT,
    name_source TEXT CHECK (name_source IN ('model', 'human', 'fallback')),
    embedding_model TEXT NOT NULL,
    centroid_blob BLOB NOT NULL,
    member_count INTEGER NOT NULL DEFAULT 0,
    named_members_hash TEXT,
    created_at TIMESTAMP NOT NULL,
    updated_at TIMESTAMP NOT NULL
);

-- One theme per capture.
CREATE TABLE theme_members (
    capture_id TEXT PRIMARY KEY REFERENCES captures(id) ON DELETE CASCADE,
    theme_id TEXT NOT NULL REFERENCES themes(id) ON DELETE CASCADE,
    similarity REAL NOT NULL,
    assigned_at TIMESTAMP NOT NULL
);

CREATE INDEX idx_theme_members_theme ON theme_members(theme_id);
