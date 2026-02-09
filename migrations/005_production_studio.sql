-- Animus Production Studio Migration
-- Supports dynamic library, community engagement, and enhanced metadata

-- 1. Wisdom Library for the Librarian worker
CREATE TABLE wisdom_library (
    id SERIAL PRIMARY KEY,
    author TEXT,
    title TEXT,
    content_chunk TEXT NOT NULL,
    thematic_tags TEXT[],
    source_url TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_wisdom_library_tags ON wisdom_library USING GIN (thematic_tags);

-- 2. Comment Interactions for the Community worker
CREATE TABLE comment_interactions (
    id SERIAL PRIMARY KEY,
    comment_id VARCHAR(100) UNIQUE NOT NULL,
    video_id UUID REFERENCES videos(id),
    youtube_video_id VARCHAR(50) NOT NULL,
    author_name TEXT,
    comment_text TEXT,
    reply_text TEXT,
    status VARCHAR(20) DEFAULT 'pending', -- pending, replied, ignored
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    replied_at TIMESTAMPTZ
);

-- 3. Update videos table with production and recycling fields
ALTER TABLE videos ADD COLUMN shorts_path TEXT;
ALTER TABLE videos ADD COLUMN performance_score FLOAT4;
ALTER TABLE videos ADD COLUMN production_metadata JSONB; -- Stores mood, SFX triggers, etc.
ALTER TABLE videos ADD COLUMN performance_updated_at TIMESTAMPTZ;
