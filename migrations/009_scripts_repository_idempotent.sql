-- Script Repository (idempotent version)
-- Combines logic from original 007 and 008 with proper IF NOT EXISTS patterns

-- Table (already idempotent with IF NOT EXISTS)
CREATE TABLE IF NOT EXISTS scripts (
    id SERIAL PRIMARY KEY,
    video_id UUID REFERENCES videos(id) ON DELETE SET NULL,
    content JSONB NOT NULL,
    topic TEXT NOT NULL,
    word_count INTEGER NOT NULL,
    quality_score REAL,
    content_hash VARCHAR(64) NOT NULL,
    exported_formats TEXT[] NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes with IF NOT EXISTS
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = 'idx_scripts_video_id') THEN
        CREATE INDEX idx_scripts_video_id ON scripts(video_id);
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = 'idx_scripts_topic') THEN
        CREATE INDEX idx_scripts_topic ON scripts(topic);
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = 'idx_scripts_word_count') THEN
        CREATE INDEX idx_scripts_word_count ON scripts(word_count);
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = 'idx_scripts_quality_score') THEN
        CREATE INDEX idx_scripts_quality_score ON scripts(quality_score DESC);
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = 'idx_scripts_content_hash') THEN
        CREATE INDEX idx_scripts_content_hash ON scripts(content_hash);
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = 'idx_scripts_created_at') THEN
        CREATE INDEX idx_scripts_created_at ON scripts(created_at DESC);
    END IF;
END
$$;

-- Unique constraint (from original 008)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.table_constraints 
        WHERE constraint_name = 'unique_content_hash' AND table_name = 'scripts'
    ) THEN
        ALTER TABLE scripts ADD CONSTRAINT unique_content_hash UNIQUE (content_hash);
    END IF;
END
$$;

-- Trigger function (CREATE OR REPLACE is idempotent)
CREATE OR REPLACE FUNCTION update_scripts_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger with proper existence check
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'scripts_updated_at') THEN
        CREATE TRIGGER scripts_updated_at
            BEFORE UPDATE ON scripts
            FOR EACH ROW
            EXECUTE FUNCTION update_scripts_updated_at();
    END IF;
END
$$;

-- Helper functions (CREATE OR REPLACE is idempotent)
CREATE OR REPLACE FUNCTION calculate_script_word_count(script_json JSONB)
RETURNS INTEGER AS $$
BEGIN
    RETURN (
        SELECT coalesce(
            array_length(regexp_split_to_array(script_json->>'full_text', '\s+'), 1),
            0
        )
    );
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION insert_script(
    p_video_id UUID,
    p_content JSONB,
    p_topic TEXT,
    p_quality_score REAL DEFAULT NULL
)
RETURNS INTEGER AS $$
DECLARE
    v_word_count INTEGER;
    v_content_hash VARCHAR(64);
BEGIN
    v_word_count := calculate_script_word_count(p_content);
    
    v_content_hash := encode(
        sha256(convert_to(p_content->>'full_text', 'UTF8')),
        'hex'
    );
    
    IF EXISTS (SELECT 1 FROM scripts WHERE content_hash = v_content_hash) THEN
        RAISE NOTICE 'Duplicate script detected with hash %', v_content_hash;
        RETURN (SELECT id FROM scripts WHERE content_hash = v_content_hash LIMIT 1);
    END IF;
    
    INSERT INTO scripts (video_id, content, topic, word_count, quality_score, content_hash)
    VALUES (p_video_id, p_content, p_topic, v_word_count, p_quality_score, v_content_hash)
    RETURNING id INTO v_word_count;
    
    RETURN v_word_count;
END;
$$ LANGUAGE plpgsql;

-- View (CREATE OR REPLACE is idempotent)
CREATE OR REPLACE VIEW script_exports AS
SELECT
    id,
    video_id,
    topic,
    word_count,
    quality_score,
    created_at,
    content->>'full_text' as full_text,
    content->'hook' as hook,
    content->'sections' as sections,
    content->'cta' as cta,
    content->'total_duration_seconds' as total_duration_seconds,
    exported_formats
FROM scripts;