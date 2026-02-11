//! Database models

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sqlx::FromRow;
use uuid::Uuid;
use std::str::FromStr;

/// Video production status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum VideoStatus {
    Scheduled,
    Producing,
    ReadyForReview,
    Published,
    Failed,
}

impl FromStr for VideoStatus {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s.to_lowercase().as_str() {
            "scheduled" => Self::Scheduled,
            "producing" => Self::Producing,
            "readyforreview" => Self::ReadyForReview,
            "published" => Self::Published,
            "failed" => Self::Failed,
            _ => Self::Scheduled,
        })
    }
}

impl VideoStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Scheduled => "scheduled",
            Self::Producing => "producing",
            Self::ReadyForReview => "readyforreview",
            Self::Published => "published",
            Self::Failed => "failed",
        }
    }
}

/// Video record in the database
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct Video {
    pub id: Uuid,
    /// Status stored as VARCHAR in DB, converted manually
    #[sqlx(rename = "status")]
    pub status_str: String,
    pub topic_brief: Option<serde_json::Value>,
    pub script: Option<serde_json::Value>,
    pub audio_timing: Option<serde_json::Value>,
    pub asset_manifest: Option<serde_json::Value>,
    pub seo_metadata: Option<serde_json::Value>,
    pub video_path: Option<String>,
    pub thumbnail_path: Option<String>,
    pub youtube_id: Option<String>,
    pub youtube_url: Option<String>,
    #[sqlx(skip)]
    pub youtube_account_id: Option<i32>,
    pub scheduled_at: Option<DateTime<Utc>>,
    pub published_at: Option<DateTime<Utc>>,
    pub error_message: Option<String>,
    pub failed_at_stage: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl Video {
    /// Get the status as an enum
    pub fn status(&self) -> VideoStatus {
        VideoStatus::from_str(&self.status_str).unwrap_or(VideoStatus::Scheduled)
    }

    /// Create a new video record for a production run
    pub fn new_production(
        id: Uuid,
        topic_brief: serde_json::Value,
        scheduled_at: Option<DateTime<Utc>>,
    ) -> Self {
        let now = Utc::now();
        Self {
            id,
            status_str: VideoStatus::Producing.as_str().to_string(),
            topic_brief: Some(topic_brief),
            script: None,
            audio_timing: None,
            asset_manifest: None,
            seo_metadata: None,
            video_path: None,
            thumbnail_path: None,
            youtube_id: None,
            youtube_url: None,
            youtube_account_id: None,
            scheduled_at,
            published_at: None,
            error_message: None,
            failed_at_stage: None,
            created_at: now,
            updated_at: now,
        }
    }
}
