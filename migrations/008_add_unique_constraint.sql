-- Add unique constraint on content_hash for ON CONFLICT support
ALTER TABLE scripts ADD CONSTRAINT unique_content_hash UNIQUE (content_hash);