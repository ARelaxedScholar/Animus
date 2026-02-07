-- Video Performance Tracking for DSPy Optimization Loop
-- Tracks YouTube analytics data for each published video
-- Used to train the Judge (predictor) and optimize the Writer (generator)

-- Performance snapshots from YouTube Analytics API
-- Multiple snapshots per video allow tracking performance over time
CREATE TABLE video_performance (
    id SERIAL PRIMARY KEY,
    video_id UUID NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    
    -- Raw metrics from YouTube
    view_count BIGINT NOT NULL DEFAULT 0,
    like_count BIGINT NOT NULL DEFAULT 0,
    comment_count BIGINT NOT NULL DEFAULT 0,
    
    -- Retention metrics (the most valuable signal)
    average_view_duration_seconds REAL,          -- How long people actually watch
    average_view_percentage REAL,                -- AVD / total duration
    
    -- Detailed retention curve (JSONB for flexibility)
    -- Format: {"0": 100, "30": 85, "60": 70, ...} (seconds -> % retained)
    retention_curve JSONB,
    
    -- Engagement ratios (computed for convenience)
    like_ratio REAL,                             -- likes / views
    comment_ratio REAL,                          -- comments / views
    
    -- When this snapshot was taken
    fetched_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    -- How many hours after publish this was fetched
    hours_since_publish INTEGER
);

-- Index for finding latest snapshot per video
CREATE INDEX idx_video_performance_video_id ON video_performance(video_id);
CREATE INDEX idx_video_performance_fetched ON video_performance(fetched_at DESC);

-- Composite index for "get latest snapshot for video"
CREATE INDEX idx_video_performance_latest ON video_performance(video_id, fetched_at DESC);


-- Add performance_score to videos table
-- This is the "label" for DSPy training: the harmonic mean of normalized metrics
ALTER TABLE videos ADD COLUMN performance_score REAL;

-- Track when performance was last updated
ALTER TABLE videos ADD COLUMN performance_updated_at TIMESTAMPTZ;

-- Index for finding videos that need performance updates
-- (published but no score, or score is stale)
CREATE INDEX idx_videos_needs_performance ON videos(published_at) 
    WHERE status = 'published' AND performance_score IS NULL;


-- Channel baseline metrics for normalization
-- Stores rolling averages to normalize individual video performance
CREATE TABLE channel_baselines (
    id SERIAL PRIMARY KEY,
    
    -- Rolling averages (updated weekly)
    avg_views_7d REAL,                           -- Average views per video in last 7 days
    avg_likes_7d REAL,
    avg_retention_7d REAL,                       -- Average retention percentage
    
    avg_views_30d REAL,
    avg_likes_30d REAL,
    avg_retention_30d REAL,
    
    -- Video count used for these averages
    video_count_7d INTEGER,
    video_count_30d INTEGER,
    
    -- When this baseline was computed
    computed_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Only keep the most recent baseline (or last N for history)
CREATE INDEX idx_channel_baselines_latest ON channel_baselines(computed_at DESC);


-- Comments for documentation
COMMENT ON TABLE video_performance IS 'YouTube Analytics snapshots for DSPy training loop';
COMMENT ON COLUMN video_performance.retention_curve IS 'JSON object mapping seconds to retention percentage';
COMMENT ON COLUMN videos.performance_score IS 'Harmonic mean of normalized views, likes, and retention (0-1 scale)';
COMMENT ON TABLE channel_baselines IS 'Rolling channel averages for metric normalization';
