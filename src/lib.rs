//! Animus - Autonomous YouTube Content Farm
//!
//! Powered by Orichalcum, this system produces long-form motivation/self-help
//! videos from classic wisdom sources (Bible, Stoicism, Philosophy, Biographies).
//!
//! # Architecture
//!
//! The system is built as an Orichalcum AsyncFlow with the following nodes:
//!
//! ```text
//! Scheduler → Strategy → ScriptWriter → TTS → AssetCollector
//!                                               ↓
//!                                         VideoAssembler
//!                                               ↓
//!                               Thumbnail → SEO → Publisher
//! ```

pub mod api;
pub mod bridge;
pub mod config;
pub mod db;
pub mod flows;
pub mod nodes;
pub mod storage;

// Re-exports for convenience
pub use config::Settings;
pub use db::{Video, VideoStatus};
pub use flows::VideoProductionFlow;

/// Application-wide error type
#[derive(Debug, thiserror::Error)]
pub enum AnimusError {
    #[error("Configuration error: {0}")]
    Config(#[from] config::ConfigError),

    #[error("Database error: {0}")]
    Database(#[from] sqlx::Error),

    #[error("Storage error: {0}")]
    Storage(String),

    #[error("LLM error: {0}")]
    Llm(String),

    #[error("TTS error: {0}")]
    Tts(String),

    #[error("Video processing error: {0}")]
    Video(String),

    #[error("YouTube API error: {0}")]
    YouTube(String),

    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Node execution error: {0}")]
    Node(String),

    #[error("Cancelled")]
    Cancelled,
}

pub type Result<T> = std::result::Result<T, AnimusError>;

/// Shared state keys used across nodes
pub mod state_keys {
    pub const VIDEO_ID: &str = "video_id";
    pub const TOPIC_BRIEF: &str = "topic_brief";
    pub const SCRIPT: &str = "script";
    pub const AUDIO_PATH: &str = "audio_path";
    pub const AUDIO_TIMING: &str = "audio_timing";
    pub const ASSET_MANIFEST: &str = "asset_manifest";
    pub const VIDEO_PATH: &str = "video_path";
    pub const THUMBNAIL_PATH: &str = "thumbnail_path";
    pub const SEO_METADATA: &str = "seo_metadata";
    pub const YOUTUBE_VIDEO_ID: &str = "youtube_video_id";
    pub const ERROR: &str = "error";
    pub const CANCELLED: &str = "cancelled";
}
