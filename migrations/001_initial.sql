-- Animus Initial Schema
-- Videos table with all intermediate state for crash recovery

CREATE TABLE videos (
    id UUID PRIMARY KEY,
    status VARCHAR(20) NOT NULL DEFAULT 'scheduled',
    
    -- Intermediate state (stored as JSONB for flexibility)
    topic_brief JSONB,
    script JSONB,
    audio_timing JSONB,
    asset_manifest JSONB,
    seo_metadata JSONB,
    
    -- Output paths (S3 keys)
    video_path TEXT,
    thumbnail_path TEXT,
    
    -- YouTube result
    youtube_id VARCHAR(50),
    youtube_url TEXT,
    
    -- Scheduling
    scheduled_at TIMESTAMPTZ,
    published_at TIMESTAMPTZ,
    
    -- Error tracking
    error_message TEXT,
    failed_at_stage VARCHAR(50),
    
    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes for common queries
CREATE INDEX idx_videos_status ON videos(status);
CREATE INDEX idx_videos_scheduled ON videos(scheduled_at) WHERE scheduled_at IS NOT NULL;
CREATE INDEX idx_videos_created ON videos(created_at DESC);

-- Auto-update updated_at trigger
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER videos_updated_at
    BEFORE UPDATE ON videos
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();
