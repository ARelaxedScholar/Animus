-- Migration 006: Multi-account support
CREATE TABLE youtube_accounts (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL UNIQUE, -- e.g. "Excelsior Academy"
    niche VARCHAR(100),                -- e.g. "stoicism", "fitness"
    
    -- OAuth Credentials
    client_id TEXT NOT NULL,
    client_secret TEXT NOT NULL,
    refresh_token TEXT NOT NULL,
    
    -- Channel Metadata
    channel_id VARCHAR(100),
    
    -- Status
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Add account_id to videos table to track which channel published which video
ALTER TABLE videos ADD COLUMN youtube_account_id INTEGER REFERENCES youtube_accounts(id);

-- Add account management to seed_queue (optionally target specific accounts)
ALTER TABLE seed_queue ADD COLUMN target_account_id INTEGER REFERENCES youtube_accounts(id);
