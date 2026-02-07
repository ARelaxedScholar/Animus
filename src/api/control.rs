//! HTTP Control API using Axum

use axum::{
    body::Body,
    extract::{Path, Query, State},
    http::{header, StatusCode},
    middleware,
    response::Response,
    routing::{delete, get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use sqlx::PgPool;
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::api::auth::auth_middleware;
use crate::db::{self, Video, VideoStatus};
use crate::storage::S3Client;

/// Daemon state exposed to the API
#[derive(Clone)]
pub struct AppState {
    /// Whether the daemon is paused
    pub paused: Arc<RwLock<bool>>,
    /// Shutdown signal sender
    pub shutdown_tx: tokio::sync::watch::Sender<bool>,
    /// Current production status
    pub current_status: Arc<RwLock<DaemonStatus>>,
    /// Shared state from main loop
    pub shared_state: Arc<RwLock<std::collections::HashMap<String, orichalcum::NodeValue>>>,
    /// Database pool
    pub db_pool: Arc<PgPool>,
    /// S3 client
    pub s3_client: Arc<S3Client>,
}

/// Current daemon status
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DaemonStatus {
    pub running: bool,
    pub paused: bool,
    pub current_video_id: Option<String>,
    pub current_stage: Option<String>,
    pub next_scheduled_video: Option<String>,
    pub hours_until_next: Option<i64>,
    pub videos_produced: u32,
    pub last_error: Option<String>,
}

/// API response wrapper
#[derive(Serialize)]
struct ApiResponse<T> {
    success: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    data: Option<T>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

impl<T> ApiResponse<T> {
    fn ok(data: T) -> Self {
        Self {
            success: true,
            data: Some(data),
            error: None,
        }
    }

    fn err(error: impl Into<String>) -> Self {
        Self {
            success: false,
            data: None,
            error: Some(error.into()),
        }
    }
}

/// Create the API router
pub fn create_router(state: AppState) -> Router {
    Router::new()
        // Health & Status
        .route("/health", get(health_check))
        .route("/status", get(get_status))
        .route("/stats", get(get_stats))
        // Daemon control
        .route("/pause", post(pause_daemon))
        .route("/resume", post(resume_daemon))
        .route("/shutdown", post(shutdown_daemon))
        // Video management
        .route("/videos", get(list_videos))
        .route("/videos/:id", get(get_video))
        .route("/videos/:id/retry", post(retry_video))
        .route("/videos/:id/download", get(download_video))
        // Queue management
        .route("/queue", get(list_queue))
        .route("/queue", post(add_to_queue))
        .route("/queue/:id", delete(remove_from_queue))
        .route("/queue/clear", post(clear_queue))
        // Legacy endpoints
        .route("/manual/script", post(upload_manual_script))
        .route("/manual/seed", post(queue_manual_seed))
        .layer(middleware::from_fn(auth_middleware))
        .with_state(state)
}

/// Health check endpoint
async fn health_check() -> &'static str {
    "OK"
}

/// Get daemon status
async fn get_status(
    State(state): State<AppState>,
) -> Json<ApiResponse<DaemonStatus>> {
    let status = state.current_status.read().await.clone();
    Json(ApiResponse::ok(status))
}

/// Upload a manual script directly to the processing folder
async fn upload_manual_script(
    State(_state): State<AppState>,
    Json(script): Json<serde_json::Value>,
) -> (StatusCode, Json<ApiResponse<String>>) {
    let path = format!("manual_scripts/api_upload_{}.json", uuid::Uuid::new_v4());
    match tokio::fs::write(&path, serde_json::to_string_pretty(&script).unwrap()).await {
        Ok(_) => (StatusCode::OK, Json(ApiResponse::ok(format!("Script uploaded to {}", path)))),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(ApiResponse::err(format!("Failed to write script: {}", e)))),
    }
}

/// Queue a manual seed topic
async fn queue_manual_seed(
    State(state): State<AppState>,
    Json(payload): Json<serde_json::Value>,
) -> (StatusCode, Json<ApiResponse<String>>) {
    // Injects into shared state so the main loop can pick it up
    let mut shared = state.shared_state.write().await;
    if let Some(topic) = payload.get("topic").and_then(|v| v.as_str()) {
        shared.insert("seed_topic".to_string(), serde_json::json!(topic));
        if let Some(source) = payload.get("source").and_then(|v| v.as_str()) {
            shared.insert("source_focus_override".to_string(), serde_json::json!(source));
        }
        (StatusCode::OK, Json(ApiResponse::ok("Topic queued for next cycle".to_string())))
    } else {
        (StatusCode::BAD_REQUEST, Json(ApiResponse::err("Missing 'topic' field")))
    }
}

/// Pause video production
async fn pause_daemon(
    State(state): State<AppState>,
) -> (StatusCode, Json<ApiResponse<String>>) {
    let mut paused = state.paused.write().await;
    *paused = true;
    
    let mut status = state.current_status.write().await;
    status.paused = true;
    
    (StatusCode::OK, Json(ApiResponse::ok("Daemon paused".to_string())))
}

/// Resume video production
async fn resume_daemon(
    State(state): State<AppState>,
) -> (StatusCode, Json<ApiResponse<String>>) {
    let mut paused = state.paused.write().await;
    *paused = false;
    
    let mut status = state.current_status.write().await;
    status.paused = false;
    
    (StatusCode::OK, Json(ApiResponse::ok("Daemon resumed".to_string())))
}

/// Graceful shutdown
async fn shutdown_daemon(
    State(state): State<AppState>,
) -> (StatusCode, Json<ApiResponse<String>>) {
    // Send shutdown signal
    let _ = state.shutdown_tx.send(true);
    
    let mut status = state.current_status.write().await;
    status.running = false;
    
    (StatusCode::OK, Json(ApiResponse::ok("Shutdown initiated".to_string())))
}

// =============================================================================
// Video Management Endpoints
// =============================================================================

/// Query parameters for listing videos
#[derive(Debug, Deserialize)]
pub struct ListVideosQuery {
    pub status: Option<String>,
    pub limit: Option<i64>,
    pub _offset: Option<i64>,
}

/// Video summary for list view
#[derive(Debug, Serialize)]
pub struct VideoSummary {
    pub id: String,
    pub status: String,
    pub title: Option<String>,
    pub youtube_url: Option<String>,
    pub created_at: String,
    pub failed_at_stage: Option<String>,
    pub error_message: Option<String>,
}

impl From<Video> for VideoSummary {
    fn from(v: Video) -> Self {
        let title = v.seo_metadata
            .as_ref()
            .and_then(|m| m.get("title"))
            .and_then(|t| t.as_str())
            .map(|s| s.to_string())
            .or_else(|| {
                v.topic_brief
                    .as_ref()
                    .and_then(|b| b.get("topic"))
                    .and_then(|t| t.as_str())
                    .map(|s| s.to_string())
            });
        
        Self {
            id: v.id.to_string(),
            status: v.status_str,
            title,
            youtube_url: v.youtube_url,
            created_at: v.created_at.to_rfc3339(),
            failed_at_stage: v.failed_at_stage,
            error_message: v.error_message,
        }
    }
}

/// List videos with optional filters
async fn list_videos(
    State(state): State<AppState>,
    Query(params): Query<ListVideosQuery>,
) -> (StatusCode, Json<ApiResponse<Vec<VideoSummary>>>) {
    let status = params.status.as_ref().map(|s| VideoStatus::from_str(s));
    let limit = params.limit.unwrap_or(50);
    
    match db::list_videos(&state.db_pool, status, limit).await {
        Ok(videos) => {
            let summaries: Vec<VideoSummary> = videos.into_iter().map(|v| v.into()).collect();
            (StatusCode::OK, Json(ApiResponse::ok(summaries)))
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ApiResponse::err(format!("Database error: {}", e))),
        ),
    }
}

/// Full video details
#[derive(Debug, Serialize)]
pub struct VideoDetails {
    pub id: String,
    pub status: String,
    pub title: Option<String>,
    pub topic_brief: Option<serde_json::Value>,
    pub script: Option<serde_json::Value>,
    pub seo_metadata: Option<serde_json::Value>,
    pub video_path: Option<String>,
    pub thumbnail_path: Option<String>,
    pub youtube_id: Option<String>,
    pub youtube_url: Option<String>,
    pub scheduled_at: Option<String>,
    pub published_at: Option<String>,
    pub failed_at_stage: Option<String>,
    pub error_message: Option<String>,
    pub created_at: String,
    pub updated_at: String,
}

impl From<Video> for VideoDetails {
    fn from(v: Video) -> Self {
        let title = v.seo_metadata
            .as_ref()
            .and_then(|m| m.get("title"))
            .and_then(|t| t.as_str())
            .map(|s| s.to_string());
        
        Self {
            id: v.id.to_string(),
            status: v.status_str,
            title,
            topic_brief: v.topic_brief,
            script: v.script,
            seo_metadata: v.seo_metadata,
            video_path: v.video_path,
            thumbnail_path: v.thumbnail_path,
            youtube_id: v.youtube_id,
            youtube_url: v.youtube_url,
            scheduled_at: v.scheduled_at.map(|t| t.to_rfc3339()),
            published_at: v.published_at.map(|t| t.to_rfc3339()),
            failed_at_stage: v.failed_at_stage,
            error_message: v.error_message,
            created_at: v.created_at.to_rfc3339(),
            updated_at: v.updated_at.to_rfc3339(),
        }
    }
}

/// Get a single video by ID
async fn get_video(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> (StatusCode, Json<ApiResponse<VideoDetails>>) {
    let uuid = match uuid::Uuid::parse_str(&id) {
        Ok(u) => u,
        Err(_) => return (StatusCode::BAD_REQUEST, Json(ApiResponse::err("Invalid UUID"))),
    };
    
    match db::get_video(&state.db_pool, uuid).await {
        Ok(Some(video)) => (StatusCode::OK, Json(ApiResponse::ok(video.into()))),
        Ok(None) => (StatusCode::NOT_FOUND, Json(ApiResponse::err("Video not found"))),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ApiResponse::err(format!("Database error: {}", e))),
        ),
    }
}

/// Retry a failed video (specifically for publisher failures)
async fn retry_video(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> (StatusCode, Json<ApiResponse<String>>) {
    let uuid = match uuid::Uuid::parse_str(&id) {
        Ok(u) => u,
        Err(_) => return (StatusCode::BAD_REQUEST, Json(ApiResponse::err("Invalid UUID"))),
    };
    
    // Get the video
    let video = match db::get_video(&state.db_pool, uuid).await {
        Ok(Some(v)) => v,
        Ok(None) => return (StatusCode::NOT_FOUND, Json(ApiResponse::err("Video not found"))),
        Err(e) => return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ApiResponse::err(format!("Database error: {}", e))),
        ),
    };
    
    // Check if it's a failed video
    if video.status() != VideoStatus::Failed {
        return (
            StatusCode::BAD_REQUEST,
            Json(ApiResponse::err("Only failed videos can be retried")),
        );
    }
    
    // Check if it failed at publisher stage (has video_path but no youtube_id)
    if video.video_path.is_none() {
        return (
            StatusCode::BAD_REQUEST,
            Json(ApiResponse::err("Video must have been assembled before retry. Failed at earlier stage.")),
        );
    }
    
    // Queue for retry by updating status and clearing error
    match db::update_video_status(&state.db_pool, uuid, VideoStatus::Producing).await {
        Ok(_) => {
            // Also inject into shared state for the main loop to pick up
            let mut shared = state.shared_state.write().await;
            shared.insert("retry_video_id".to_string(), serde_json::json!(id));
            (StatusCode::OK, Json(ApiResponse::ok("Video queued for retry".to_string())))
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ApiResponse::err(format!("Failed to update video: {}", e))),
        ),
    }
}

/// Download a video file from S3
async fn download_video(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Response<Body>, (StatusCode, Json<ApiResponse<String>>)> {
    let uuid = match uuid::Uuid::parse_str(&id) {
        Ok(u) => u,
        Err(_) => return Err((StatusCode::BAD_REQUEST, Json(ApiResponse::err("Invalid UUID")))),
    };
    
    // Get the video
    let video = match db::get_video(&state.db_pool, uuid).await {
        Ok(Some(v)) => v,
        Ok(None) => return Err((StatusCode::NOT_FOUND, Json(ApiResponse::err("Video not found")))),
        Err(e) => return Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ApiResponse::err(format!("Database error: {}", e))),
        )),
    };
    
    // Check if video has a path
    let video_path = match video.video_path {
        Some(p) => p,
        None => return Err((
            StatusCode::NOT_FOUND,
            Json(ApiResponse::err("Video file not available (not yet assembled)")),
        )),
    };
    
    // Download from S3
    let video_bytes = match state.s3_client.download_bytes(&video_path).await {
        Ok(bytes) => bytes,
        Err(e) => return Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ApiResponse::err(format!("S3 download error: {}", e))),
        )),
    };
    
    // Build filename from video title or ID
    let filename = video.seo_metadata
        .as_ref()
        .and_then(|m| m.get("title"))
        .and_then(|t| t.as_str())
        .map(|s| {
            // Sanitize filename
            s.chars()
                .map(|c| if c.is_alphanumeric() || c == ' ' || c == '-' || c == '_' { c } else { '_' })
                .collect::<String>()
        })
        .unwrap_or_else(|| id.clone());
    
    let response = Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "video/mp4")
        .header(
            header::CONTENT_DISPOSITION,
            format!("attachment; filename=\"{}.mp4\"", filename),
        )
        .header(header::CONTENT_LENGTH, video_bytes.len())
        .body(Body::from(video_bytes))
        .unwrap();
    
    Ok(response)
}

// =============================================================================
// Queue Management Endpoints
// =============================================================================

/// Seed queue item
#[derive(Debug, Serialize)]
pub struct QueueItem {
    pub id: i32,
    pub seed_topic: String,
    pub source_focus: Option<String>,
}

/// List the seed queue
async fn list_queue(
    State(state): State<AppState>,
) -> (StatusCode, Json<ApiResponse<Vec<QueueItem>>>) {
    match db::peek_queue(&state.db_pool, 100).await {
        Ok(items) => {
            let queue: Vec<QueueItem> = items
                .into_iter()
                .map(|(id, seed_topic, source_focus)| QueueItem {
                    id,
                    seed_topic,
                    source_focus,
                })
                .collect();
            (StatusCode::OK, Json(ApiResponse::ok(queue)))
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ApiResponse::err(format!("Database error: {}", e))),
        ),
    }
}

/// Add to queue request
#[derive(Debug, Deserialize)]
pub struct AddToQueueRequest {
    pub topic: String,
    pub source: Option<String>,
}

/// Add a topic to the seed queue
async fn add_to_queue(
    State(state): State<AppState>,
    Json(payload): Json<AddToQueueRequest>,
) -> (StatusCode, Json<ApiResponse<i32>>) {
    match db::queue_seed(&state.db_pool, &payload.topic, payload.source.as_deref()).await {
        Ok(id) => (StatusCode::CREATED, Json(ApiResponse::ok(id))),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ApiResponse::err(format!("Failed to queue: {}", e))),
        ),
    }
}

/// Remove an item from the queue
async fn remove_from_queue(
    State(state): State<AppState>,
    Path(id): Path<i32>,
) -> (StatusCode, Json<ApiResponse<String>>) {
    match sqlx::query("DELETE FROM seed_queue WHERE id = $1")
        .bind(id)
        .execute(state.db_pool.as_ref())
        .await
    {
        Ok(result) => {
            if result.rows_affected() > 0 {
                (StatusCode::OK, Json(ApiResponse::ok("Item removed".to_string())))
            } else {
                (StatusCode::NOT_FOUND, Json(ApiResponse::err("Item not found")))
            }
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ApiResponse::err(format!("Database error: {}", e))),
        ),
    }
}

/// Clear the entire queue
async fn clear_queue(
    State(state): State<AppState>,
) -> (StatusCode, Json<ApiResponse<u64>>) {
    match db::clear_seed_queue(&state.db_pool).await {
        Ok(count) => (StatusCode::OK, Json(ApiResponse::ok(count))),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ApiResponse::err(format!("Database error: {}", e))),
        ),
    }
}

// =============================================================================
// Statistics Endpoint
// =============================================================================

/// Aggregate statistics
#[derive(Debug, Serialize)]
pub struct Stats {
    pub total_videos: i64,
    pub published: i64,
    pub failed: i64,
    pub producing: i64,
    pub queue_length: i64,
    pub success_rate: f64,
    pub recent_failures: Vec<FailureSummary>,
}

#[derive(Debug, Serialize)]
pub struct FailureSummary {
    pub stage: String,
    pub count: i64,
}

/// Get aggregate statistics
async fn get_stats(
    State(state): State<AppState>,
) -> (StatusCode, Json<ApiResponse<Stats>>) {
    // Query counts
    let total: (i64,) = match sqlx::query_as("SELECT COUNT(*) FROM videos")
        .fetch_one(state.db_pool.as_ref())
        .await
    {
        Ok(r) => r,
        Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, Json(ApiResponse::err(format!("DB error: {}", e)))),
    };
    
    let published: (i64,) = match sqlx::query_as("SELECT COUNT(*) FROM videos WHERE status = 'published'")
        .fetch_one(state.db_pool.as_ref())
        .await
    {
        Ok(r) => r,
        Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, Json(ApiResponse::err(format!("DB error: {}", e)))),
    };
    
    let failed: (i64,) = match sqlx::query_as("SELECT COUNT(*) FROM videos WHERE status = 'failed'")
        .fetch_one(state.db_pool.as_ref())
        .await
    {
        Ok(r) => r,
        Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, Json(ApiResponse::err(format!("DB error: {}", e)))),
    };
    
    let producing: (i64,) = match sqlx::query_as("SELECT COUNT(*) FROM videos WHERE status = 'producing'")
        .fetch_one(state.db_pool.as_ref())
        .await
    {
        Ok(r) => r,
        Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, Json(ApiResponse::err(format!("DB error: {}", e)))),
    };
    
    let queue_length = match db::get_queue_length(&state.db_pool).await {
        Ok(l) => l,
        Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, Json(ApiResponse::err(format!("DB error: {}", e)))),
    };
    
    // Get failure breakdown
    let failures: Vec<(Option<String>, i64)> = match sqlx::query_as(
        "SELECT failed_at_stage, COUNT(*) FROM videos WHERE status = 'failed' GROUP BY failed_at_stage"
    )
        .fetch_all(state.db_pool.as_ref())
        .await
    {
        Ok(r) => r,
        Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, Json(ApiResponse::err(format!("DB error: {}", e)))),
    };
    
    let recent_failures: Vec<FailureSummary> = failures
        .into_iter()
        .map(|(stage, count)| FailureSummary {
            stage: stage.unwrap_or_else(|| "unknown".to_string()),
            count,
        })
        .collect();
    
    let success_rate = if total.0 > 0 {
        (published.0 as f64 / total.0 as f64) * 100.0
    } else {
        0.0
    };
    
    (StatusCode::OK, Json(ApiResponse::ok(Stats {
        total_videos: total.0,
        published: published.0,
        failed: failed.0,
        producing: producing.0,
        queue_length,
        success_rate,
        recent_failures,
    })))
}
