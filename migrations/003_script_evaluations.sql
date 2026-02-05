-- Script evaluation records for self-improvement loop
-- Stores all candidate scripts and their judge feedback for analysis

CREATE TABLE script_evaluations (
    id SERIAL PRIMARY KEY,
    video_id UUID NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    iteration INTEGER NOT NULL,              -- 0 = initial candidates, 1+ = refinements
    candidate_index INTEGER,                 -- Which candidate (0, 1, 2) or NULL for refinement
    script_hash VARCHAR(64) NOT NULL,        -- SHA256 of script content for dedup
    overall_score REAL NOT NULL,
    criteria_scores JSONB NOT NULL,          -- Individual criterion scores
    strengths TEXT[] NOT NULL DEFAULT '{}',
    weaknesses TEXT[] NOT NULL DEFAULT '{}',
    ai_telltale_signs TEXT[] NOT NULL DEFAULT '{}',
    specific_improvements JSONB NOT NULL DEFAULT '[]',
    script_content JSONB,                    -- Optional: store the actual script for debugging
    selected BOOLEAN NOT NULL DEFAULT FALSE, -- Was this the final chosen script?
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_script_evaluations_video_id ON script_evaluations(video_id);
CREATE INDEX idx_script_evaluations_selected ON script_evaluations(selected) WHERE selected = TRUE;
CREATE INDEX idx_script_evaluations_score ON script_evaluations(overall_score DESC);
