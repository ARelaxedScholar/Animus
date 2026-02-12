-- Script Repository
-- Stores all generated scripts for reuse, analysis, and export

CREATE TABLE IF NOT EXISTS scripts (
    id SERIAL PRIMARY KEY,
    -- Optional link to a video (if this script was used for a video)
    video_id UUID REFERENCES videos(id) ON DELETE SET NULL,
    -- Script content as JSONB (full Script struct)
    content JSONB NOT NULL,
    -- Topic/subject (extracted from script or provided)
    topic TEXT NOT NULL,
    -- Word count (for filtering)
    word_count INTEGER NOT NULL,
    -- Quality score (1-10) from evaluation if available
    quality_score REAL,
    -- Hash of script content for deduplication (SHA256 of full_text)
    content_hash VARCHAR(64) NOT NULL,
    -- Export formats generated (JSON, markdown, text)
    exported_formats TEXT[] NOT NULL DEFAULT '{}',
    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes for fast lookup
CREATE INDEX IF NOT EXISTS idx_scripts_video_id ON scripts(video_id);
CREATE INDEX IF NOT EXISTS idx_scripts_topic ON scripts(topic);
CREATE INDEX IF NOT EXISTS idx_scripts_word_count ON scripts(word_count);
CREATE INDEX IF NOT EXISTS idx_scripts_quality_score ON scripts(quality_score DESC);
CREATE INDEX IF NOT EXISTS idx_scripts_content_hash ON scripts(content_hash);
CREATE INDEX IF NOT EXISTS idx_scripts_created_at ON scripts(created_at DESC);

-- Trigger to update updated_at
CREATE OR REPLACE FUNCTION update_scripts_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS scripts_updated_at ON scripts;
CREATE TRIGGER scripts_updated_at
    BEFORE UPDATE ON scripts
    FOR EACH ROW
    EXECUTE FUNCTION update_scripts_updated_at();

-- Function to calculate word count from script JSON
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

-- Insert a script with automatic word count calculation
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
    -- Calculate word count
    v_word_count := calculate_script_word_count(p_content);
    
    -- Calculate SHA256 hash of full_text (convert text to bytea)
    v_content_hash := encode(
        sha256(convert_to(p_content->>'full_text', 'UTF8')),
        'hex'
    );
    
    -- Check for duplicate by content_hash
    IF EXISTS (SELECT 1 FROM scripts WHERE content_hash = v_content_hash) THEN
        RAISE NOTICE 'Duplicate script detected with hash %', v_content_hash;
        RETURN (SELECT id FROM scripts WHERE content_hash = v_content_hash LIMIT 1);
    END IF;
    
    -- Insert new script
    INSERT INTO scripts (video_id, content, topic, word_count, quality_score, content_hash)
    VALUES (p_video_id, p_content, p_topic, v_word_count, p_quality_score, v_content_hash)
    RETURNING id INTO v_word_count; -- reuse variable for id
    
    RETURN v_word_count;
END;
$$ LANGUAGE plpgsql;

-- View for easy script export
CREATE OR REPLACE VIEW script_exports AS
SELECT
    id,
    video_id,
    topic,
    word_count,
    quality_score,
    created_at,
    -- Extract full text for plain text export
    content->>'full_text' as full_text,
    -- Extract sections for structured export
    content->'hook' as hook,
    content->'sections' as sections,
    content->'cta' as cta,
    content->'total_duration_seconds' as total_duration_seconds,
    -- Export formats availability
    exported_formats
FROM scripts;