-- Restore resource identity keys for databases created before the column was present.
ALTER TABLE resources ADD COLUMN identity_key TEXT;

UPDATE resources AS resource
SET identity_key = CASE
    WHEN resource.canonical_url IS NOT NULL
        AND NOT EXISTS (
            SELECT 1 FROM resources AS earlier
            WHERE earlier.canonical_url = resource.canonical_url
              AND earlier.id < resource.id
        )
        THEN 'url:' || resource.canonical_url
    WHEN resource.latest_content_hash IS NOT NULL
        AND NOT EXISTS (
            SELECT 1 FROM resources AS earlier
            WHERE earlier.latest_content_hash = resource.latest_content_hash
              AND earlier.id < resource.id
        )
        THEN 'blob:' || resource.latest_content_hash
    ELSE 'resource:' || resource.id
END
WHERE resource.identity_key IS NULL;

CREATE UNIQUE INDEX IF NOT EXISTS idx_resources_identity_key
    ON resources(identity_key);
