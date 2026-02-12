-- Add unique constraint on content_hash for ON CONFLICT support
ALTER TABLE scripts DROP CONSTRAINT IF EXISTS unique_content_hash;
ALTER TABLE scripts ADD CONSTRAINT unique_content_hash UNIQUE (content_hash);