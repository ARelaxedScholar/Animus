-- Seed queue for pending video topics
-- Allows users to queue seed topics that will be processed FIFO
-- Persists across daemon restarts for crash recovery

CREATE TABLE seed_queue (
    id SERIAL PRIMARY KEY,
    seed_topic TEXT NOT NULL,
    source_focus VARCHAR(50),  -- NULL = auto-rotate based on video count
    queued_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Index for FIFO ordering (oldest first)
CREATE INDEX idx_seed_queue_queued_at ON seed_queue(queued_at ASC);

-- Comment for documentation
COMMENT ON TABLE seed_queue IS 'FIFO queue for seed topics awaiting video production';
COMMENT ON COLUMN seed_queue.seed_topic IS 'User-provided topic/theme for video generation';
COMMENT ON COLUMN seed_queue.source_focus IS 'Wisdom source override (Bible, Stoicism, Philosophy, Biography, Psychology) or NULL for auto-rotation';
