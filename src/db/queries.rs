//! Database queries for video persistence

use super::models::{Video, VideoStatus};
use sqlx::PgPool;
use uuid::Uuid;

/// Get a video by ID
pub async fn get_video(pool: &PgPool, id: Uuid) -> Result<Option<Video>, sqlx::Error> {
    sqlx::query_as::<_, Video>("SELECT * FROM videos WHERE id = $1")
        .bind(id)
        .fetch_optional(pool)
        .await
}

/// List videos with optional status filter
pub async fn list_videos(
    pool: &PgPool,
    status: Option<VideoStatus>,
    limit: i64,
) -> Result<Vec<Video>, sqlx::Error> {
    match status {
        Some(s) => {
            sqlx::query_as::<_, Video>(
                "SELECT * FROM videos WHERE status = $1 ORDER BY created_at DESC LIMIT $2",
            )
            .bind(s.as_str())
            .bind(limit)
            .fetch_all(pool)
            .await
        }
        None => {
            sqlx::query_as::<_, Video>(
                "SELECT * FROM videos ORDER BY created_at DESC LIMIT $1",
            )
            .bind(limit)
            .fetch_all(pool)
            .await
        }
    }
}

/// Insert a new video record
pub async fn insert_video(pool: &PgPool, video: &Video) -> Result<(), sqlx::Error> {
    sqlx::query(
        r#"INSERT INTO videos 
           (id, status, topic_brief, script, audio_timing, asset_manifest, 
            seo_metadata, video_path, thumbnail_path, youtube_id, youtube_url,
            scheduled_at, published_at, error_message, failed_at_stage, 
            created_at, updated_at)
           VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17)"#,
    )
    .bind(&video.id)
    .bind(&video.status_str)
    .bind(&video.topic_brief)
    .bind(&video.script)
    .bind(&video.audio_timing)
    .bind(&video.asset_manifest)
    .bind(&video.seo_metadata)
    .bind(&video.video_path)
    .bind(&video.thumbnail_path)
    .bind(&video.youtube_id)
    .bind(&video.youtube_url)
    .bind(&video.scheduled_at)
    .bind(&video.published_at)
    .bind(&video.error_message)
    .bind(&video.failed_at_stage)
    .bind(&video.created_at)
    .bind(&video.updated_at)
    .execute(pool)
    .await?;
    Ok(())
}

/// Update video status
pub async fn update_video_status(
    pool: &PgPool,
    id: Uuid,
    status: VideoStatus,
) -> Result<(), sqlx::Error> {
    sqlx::query("UPDATE videos SET status = $1 WHERE id = $2")
        .bind(status.as_str())
        .bind(id)
        .execute(pool)
        .await?;
    Ok(())
}

/// Update a specific JSONB field on a video
pub async fn update_video_json_field(
    pool: &PgPool,
    id: Uuid,
    field: &str,
    value: serde_json::Value,
) -> Result<(), sqlx::Error> {
    // Validate field name to prevent SQL injection
    let allowed_fields = [
        "topic_brief",
        "script",
        "audio_timing",
        "asset_manifest",
        "seo_metadata",
    ];
    if !allowed_fields.contains(&field) {
        return Err(sqlx::Error::Protocol(format!("Invalid field: {}", field)));
    }

    let query = format!("UPDATE videos SET {} = $1 WHERE id = $2", field);
    sqlx::query(&query)
        .bind(value)
        .bind(id)
        .execute(pool)
        .await?;
    Ok(())
}

/// Update a specific text field on a video
pub async fn update_video_text_field(
    pool: &PgPool,
    id: Uuid,
    field: &str,
    value: &str,
) -> Result<(), sqlx::Error> {
    let allowed_fields = ["video_path", "thumbnail_path", "youtube_id", "youtube_url"];
    if !allowed_fields.contains(&field) {
        return Err(sqlx::Error::Protocol(format!("Invalid field: {}", field)));
    }

    let query = format!("UPDATE videos SET {} = $1 WHERE id = $2", field);
    sqlx::query(&query)
        .bind(value)
        .bind(id)
        .execute(pool)
        .await?;
    Ok(())
}

/// Mark a video as failed
pub async fn mark_video_failed(
    pool: &PgPool,
    id: Uuid,
    stage: &str,
    error: &str,
) -> Result<(), sqlx::Error> {
    sqlx::query(
        "UPDATE videos SET status = 'failed', failed_at_stage = $1, error_message = $2 WHERE id = $3",
    )
    .bind(stage)
    .bind(error)
    .bind(id)
    .execute(pool)
    .await?;
    Ok(())
}

/// Mark a video as published
pub async fn mark_video_published(
    pool: &PgPool,
    id: Uuid,
    youtube_id: &str,
    youtube_url: &str,
) -> Result<(), sqlx::Error> {
    sqlx::query(
        r#"UPDATE videos SET 
           status = 'published', 
           youtube_id = $1, 
           youtube_url = $2,
           published_at = NOW() 
           WHERE id = $3"#,
    )
    .bind(youtube_id)
    .bind(youtube_url)
    .bind(id)
    .execute(pool)
    .await?;
    Ok(())
}

// =============================================================================
// Seed Queue Functions
// =============================================================================

/// Queue a seed topic for future production
/// Returns the ID of the queued seed
pub async fn queue_seed(
    pool: &PgPool,
    seed_topic: &str,
    source_focus: Option<&str>,
) -> Result<i32, sqlx::Error> {
    let row: (i32,) = sqlx::query_as(
        "INSERT INTO seed_queue (seed_topic, source_focus) VALUES ($1, $2) RETURNING id",
    )
    .bind(seed_topic)
    .bind(source_focus)
    .fetch_one(pool)
    .await?;
    Ok(row.0)
}

/// Pop the oldest seed from the queue (FIFO)
/// Returns (id, seed_topic, source_focus) or None if queue is empty
pub async fn pop_seed(
    pool: &PgPool,
) -> Result<Option<(i32, String, Option<String>)>, sqlx::Error> {
    // Use a CTE to atomically select and delete the oldest entry
    let result: Option<(i32, String, Option<String>)> = sqlx::query_as(
        r#"WITH oldest AS (
            SELECT id, seed_topic, source_focus 
            FROM seed_queue 
            ORDER BY queued_at ASC 
            LIMIT 1
            FOR UPDATE SKIP LOCKED
        )
        DELETE FROM seed_queue 
        WHERE id = (SELECT id FROM oldest)
        RETURNING id, seed_topic, source_focus"#,
    )
    .fetch_optional(pool)
    .await?;
    Ok(result)
}

/// Get the number of seeds in the queue
pub async fn get_queue_length(pool: &PgPool) -> Result<i64, sqlx::Error> {
    let row: (i64,) = sqlx::query_as("SELECT COUNT(*) FROM seed_queue")
        .fetch_one(pool)
        .await?;
    Ok(row.0)
}

/// Peek at the queue without removing entries
/// Returns seeds ordered by queued_at (oldest first)
pub async fn peek_queue(
    pool: &PgPool,
    limit: i64,
) -> Result<Vec<(i32, String, Option<String>)>, sqlx::Error> {
    let rows: Vec<(i32, String, Option<String>)> = sqlx::query_as(
        "SELECT id, seed_topic, source_focus FROM seed_queue ORDER BY queued_at ASC LIMIT $1",
    )
    .bind(limit)
    .fetch_all(pool)
    .await?;
    Ok(rows)
}

/// Clear the entire seed queue (for admin/testing)
pub async fn clear_seed_queue(pool: &PgPool) -> Result<u64, sqlx::Error> {
    let result = sqlx::query("DELETE FROM seed_queue")
        .execute(pool)
        .await?;
    Ok(result.rows_affected())
}
