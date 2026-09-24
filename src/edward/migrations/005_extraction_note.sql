-- Record why an extraction produced what it did.
--
-- fallback rows used to be indistinguishable: the extractor name was the only
-- trace, and 'local-fallback' could mean a tool failure, a placeholder result,
-- or an unreadable output shape. That ambiguity sent an investigation chasing
-- the wrong suspect. The reason is now stored next to the content.

ALTER TABLE resource_contents ADD COLUMN extraction_note TEXT;
