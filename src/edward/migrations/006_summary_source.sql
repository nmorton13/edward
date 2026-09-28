-- Explicit provenance for resource summaries: bundle | human | model | legacy.
-- Distinguishes low-quality legacy summaries (e.g. page lead text extracted by early summarize CLI)
-- from model-generated summaries, bundle imports, and human-authored summaries.

ALTER TABLE resource_contents ADD COLUMN summary_source TEXT
    CHECK (summary_source IN ('bundle', 'human', 'model', 'legacy'));

-- 1. bundle: written by import-research (extractor = 'markdown-report')
UPDATE resource_contents
SET summary_source = 'bundle'
WHERE summary IS NOT NULL AND length(trim(summary)) > 0
  AND extractor = 'markdown-report';

-- 2. human: explicitly human-authored summaries if any
UPDATE resource_contents
SET summary_source = 'human'
WHERE summary IS NOT NULL AND length(trim(summary)) > 0
  AND extractor = 'human';

-- 3. legacy: early summaries produced by summarize CLI extracting page lead text
UPDATE resource_contents
SET summary_source = 'legacy'
WHERE summary IS NOT NULL AND length(trim(summary)) > 0
  AND extractor = 'summarize';

-- 4. model: any other pre-existing summaries (such as runs from board prep)
UPDATE resource_contents
SET summary_source = 'model'
WHERE summary IS NOT NULL AND length(trim(summary)) > 0
  AND summary_source IS NULL;

-- Reset processing_jobs for legacy summaries to pending so they can be re-summarized by the model
UPDATE processing_jobs
SET status = 'pending',
    completed_at = NULL,
    attempts = 0,
    last_error = NULL,
    available_at = CURRENT_TIMESTAMP,
    updated_at = CURRENT_TIMESTAMP
WHERE stage = 'summarize'
  AND status = 'completed'
  AND resource_id IN (
      SELECT resource_id FROM resource_contents
      WHERE summary_source = 'legacy'
  );
