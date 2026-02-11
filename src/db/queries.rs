//! Database queries for video persistence

use super::models::{Video, VideoStatus};
use chrono::{DateTime, Utc};
use sqlx::PgPool;
use uuid::Uuid;

/// Get the most recent video with 'producing' status
pub async fn get_active_production(pool: &PgPool) -> Result<Option<Video>, sqlx::Error> {
    sqlx::query_as::<_, Video>("SELECT * FROM videos WHERE status = 'producing' ORDER BY updated_at DESC LIMIT 1")
        .fetch_optional(pool)
        .await
}

/// Get the latest scheduled time across all autonomous videos
pub async fn get_latest_scheduled_time(pool: &PgPool) -> Result<Option<DateTime<Utc>>, sqlx::Error> {
    let row: Option<(Option<DateTime<Utc>>,)> = sqlx::query_as(
        "SELECT scheduled_at FROM videos WHERE topic_brief->>'is_autonomous' = 'true' ORDER BY created_at DESC LIMIT 1",
    )
    .fetch_optional(pool)
    .await?;
    Ok(row.and_then(|r| r.0))
}

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
    .bind(video.id)
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
    .bind(video.scheduled_at)
    .bind(video.published_at)
    .bind(&video.error_message)
    .bind(&video.failed_at_stage)
    .bind(video.created_at)
    .bind(video.updated_at)
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

// =============================================================================
// Script Evaluation Functions (Self-Improvement Loop)
// =============================================================================

/// Insert a script evaluation record
/// Returns the ID of the inserted evaluation
#[allow(clippy::too_many_arguments)]
pub async fn insert_script_evaluation(
    pool: &PgPool,
    video_id: Uuid,
    iteration: i32,
    candidate_index: Option<i32>,
    script_hash: &str,
    overall_score: f32,
    criteria_scores: serde_json::Value,
    strengths: &[String],
    weaknesses: &[String],
    ai_telltale_signs: &[String],
    specific_improvements: serde_json::Value,
    script_content: Option<serde_json::Value>,
) -> Result<i32, sqlx::Error> {
    let row: (i32,) = sqlx::query_as(
        r#"INSERT INTO script_evaluations 
           (video_id, iteration, candidate_index, script_hash, overall_score, 
            criteria_scores, strengths, weaknesses, ai_telltale_signs, 
            specific_improvements, script_content, selected)
           VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, FALSE)
           RETURNING id"#,
    )
    .bind(video_id)
    .bind(iteration)
    .bind(candidate_index)
    .bind(script_hash)
    .bind(overall_score)
    .bind(criteria_scores)
    .bind(strengths)
    .bind(weaknesses)
    .bind(ai_telltale_signs)
    .bind(specific_improvements)
    .bind(script_content)
    .fetch_one(pool)
    .await?;
    Ok(row.0)
}

/// Mark a script evaluation as the selected (final) one
pub async fn mark_evaluation_selected(
    pool: &PgPool,
    evaluation_id: i32,
) -> Result<(), sqlx::Error> {
    sqlx::query("UPDATE script_evaluations SET selected = TRUE WHERE id = $1")
        .bind(evaluation_id)
        .execute(pool)
        .await?;
    Ok(())
}

/// Get all evaluations for a video (for debugging/analysis)
pub async fn get_video_evaluations(
    pool: &PgPool,
    video_id: Uuid,
) -> Result<Vec<(i32, i32, Option<i32>, f32, bool)>, sqlx::Error> {
    // Returns (id, iteration, candidate_index, overall_score, selected)
    let rows: Vec<(i32, i32, Option<i32>, f32, bool)> = sqlx::query_as(
        r#"SELECT id, iteration, candidate_index, overall_score, selected 
           FROM script_evaluations 
           WHERE video_id = $1 
           ORDER BY iteration ASC, candidate_index ASC NULLS LAST"#,
    )
    .bind(video_id)
    .fetch_all(pool)
    .await?;
    Ok(rows)
}

/// Get the best evaluation for a video (highest score)
pub async fn get_best_evaluation(
    pool: &PgPool,
    video_id: Uuid,
) -> Result<Option<(i32, f32)>, sqlx::Error> {
    // Returns (id, overall_score) of the best evaluation
    let row: Option<(i32, f32)> = sqlx::query_as(
        r#"SELECT id, overall_score 
           FROM script_evaluations 
           WHERE video_id = $1 
           ORDER BY overall_score DESC 
           LIMIT 1"#,
    )
    .bind(video_id)
    .fetch_optional(pool)
    .await?;
    Ok(row)
}

// =============================================================================
// Video Performance Functions (DSPy Training Loop)
// =============================================================================

/// Insert a performance snapshot for a video
#[allow(clippy::too_many_arguments)]
pub async fn insert_video_performance(
    pool: &PgPool,
    video_id: Uuid,
    view_count: i64,
    like_count: i64,
    comment_count: i64,
    average_view_duration_seconds: Option<f32>,
    average_view_percentage: Option<f32>,
    retention_curve: Option<serde_json::Value>,
    hours_since_publish: Option<i32>,
) -> Result<i32, sqlx::Error> {
    // Compute ratios
    let like_ratio = if view_count > 0 {
        Some(like_count as f32 / view_count as f32)
    } else {
        None
    };
    let comment_ratio = if view_count > 0 {
        Some(comment_count as f32 / view_count as f32)
    } else {
        None
    };

    let row: (i32,) = sqlx::query_as(
        r#"INSERT INTO video_performance 
           (video_id, view_count, like_count, comment_count,
            average_view_duration_seconds, average_view_percentage,
            retention_curve, like_ratio, comment_ratio, hours_since_publish)
           VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
           RETURNING id"#,
    )
    .bind(video_id)
    .bind(view_count)
    .bind(like_count)
    .bind(comment_count)
    .bind(average_view_duration_seconds)
    .bind(average_view_percentage)
    .bind(retention_curve)
    .bind(like_ratio)
    .bind(comment_ratio)
    .bind(hours_since_publish)
    .fetch_one(pool)
    .await?;
    Ok(row.0)
}

/// Get the latest performance snapshot for a video
pub async fn get_latest_video_performance(
    pool: &PgPool,
    video_id: Uuid,
) -> Result<Option<(i64, i64, Option<f32>, Option<f32>)>, sqlx::Error> {
    // Returns (view_count, like_count, avg_view_percentage, like_ratio)
    let row: Option<(i64, i64, Option<f32>, Option<f32>)> = sqlx::query_as(
        r#"SELECT view_count, like_count, average_view_percentage, like_ratio
           FROM video_performance
           WHERE video_id = $1
           ORDER BY fetched_at DESC
           LIMIT 1"#,
    )
    .bind(video_id)
    .fetch_optional(pool)
    .await?;
    Ok(row)
}

/// Update the performance score on a video (the harmonic mean)
pub async fn update_video_performance_score(
    pool: &PgPool,
    video_id: Uuid,
    score: f32,
) -> Result<(), sqlx::Error> {
    sqlx::query(
        r#"UPDATE videos 
           SET performance_score = $1, performance_updated_at = NOW() 
           WHERE id = $2"#,
    )
    .bind(score)
    .bind(video_id)
    .execute(pool)
    .await?;
    Ok(())
}

/// Get published videos that need performance updates
/// Returns videos published > N hours ago without a performance score
pub async fn get_videos_needing_performance_update(
    pool: &PgPool,
    min_hours_since_publish: i32,
    limit: i64,
) -> Result<Vec<(Uuid, String, DateTime<Utc>)>, sqlx::Error> {
    // Returns (video_id, youtube_id, published_at)
    let rows: Vec<(Uuid, String, DateTime<Utc>)> = sqlx::query_as(
        r#"SELECT id, youtube_id, published_at
           FROM videos
           WHERE status = 'published'
             AND youtube_id IS NOT NULL
             AND published_at IS NOT NULL
             AND published_at < NOW() - INTERVAL '1 hour' * $1
             AND performance_score IS NULL
           ORDER BY published_at ASC
           LIMIT $2"#,
    )
    .bind(min_hours_since_publish)
    .bind(limit)
    .fetch_all(pool)
    .await?;
    Ok(rows)
}

/// Get the latest channel baseline
pub async fn get_channel_baseline(
    pool: &PgPool,
) -> Result<Option<(f32, f32, f32)>, sqlx::Error> {
    // Returns (avg_views_30d, avg_likes_30d, avg_retention_30d)
    let row: Option<(Option<f32>, Option<f32>, Option<f32>)> = sqlx::query_as(
        r#"SELECT avg_views_30d, avg_likes_30d, avg_retention_30d
           FROM channel_baselines
           ORDER BY computed_at DESC
           LIMIT 1"#,
    )
    .fetch_optional(pool)
    .await?;
    
    // Unwrap options, defaulting to 1.0 to avoid division by zero
    Ok(row.map(|(v, l, r)| (v.unwrap_or(1.0), l.unwrap_or(1.0), r.unwrap_or(1.0))))
}

/// Insert or update channel baseline
#[allow(clippy::too_many_arguments)]
pub async fn upsert_channel_baseline(
    pool: &PgPool,
    avg_views_7d: f32,
    avg_likes_7d: f32,
    avg_retention_7d: f32,
    avg_views_30d: f32,
    avg_likes_30d: f32,
    avg_retention_30d: f32,
    video_count_7d: i32,
    video_count_30d: i32,
) -> Result<i32, sqlx::Error> {
    let row: (i32,) = sqlx::query_as(
        r#"INSERT INTO channel_baselines 
           (avg_views_7d, avg_likes_7d, avg_retention_7d,
            avg_views_30d, avg_likes_30d, avg_retention_30d,
            video_count_7d, video_count_30d)
           VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
           RETURNING id"#,
    )
    .bind(avg_views_7d)
    .bind(avg_likes_7d)
    .bind(avg_retention_7d)
    .bind(avg_views_30d)
    .bind(avg_likes_30d)
    .bind(avg_retention_30d)
    .bind(video_count_7d)
    .bind(video_count_30d)
    .fetch_one(pool)
    .await?;
    Ok(row.0)
}

/// Get all published videos with their scripts and performance scores
/// Used for DSPy training data export
pub async fn get_training_data(
    pool: &PgPool,
    min_score: Option<f32>,
    limit: i64,
) -> Result<Vec<(Uuid, serde_json::Value, serde_json::Value, f32)>, sqlx::Error> {
    // Returns (video_id, topic_brief, script, performance_score)
    let query = match min_score {
        Some(score) => {
            sqlx::query_as(
                r#"SELECT id, topic_brief, script, performance_score
                   FROM videos
                   WHERE status = 'published'
                     AND script IS NOT NULL
                     AND performance_score IS NOT NULL
                     AND performance_score >= $1
                   ORDER BY performance_score DESC
                   LIMIT $2"#,
            )
            .bind(score)
            .bind(limit)
        }
        None => {
            sqlx::query_as(
                r#"SELECT id, topic_brief, script, performance_score
                   FROM videos
                   WHERE status = 'published'
                     AND script IS NOT NULL
                     AND performance_score IS NOT NULL
                   ORDER BY performance_score DESC
                   LIMIT $1"#,
            )
            .bind(limit)
        }
    };
    
    #[allow(clippy::type_complexity)]
    let rows: Vec<(Uuid, Option<serde_json::Value>, Option<serde_json::Value>, Option<f32>)> = 
        query.fetch_all(pool).await?;
    
    // Filter out nulls and unwrap
    Ok(rows
        .into_iter()
        .filter_map(|(id, tb, s, ps)| {
            match (tb, s, ps) {
                (Some(topic), Some(script), Some(score)) => Some((id, topic, script, score)),
                _ => None,
            }
        })
        .collect())
}
