//! HTTP Control API using Axum

use axum::{
    extract::State,
    http::StatusCode,
    middleware,
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;
use crate::api::auth::auth_middleware;

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
        .route("/health", get(health_check))
        .route("/status", get(get_status))
        .route("/pause", post(pause_daemon))
        .route("/resume", post(resume_daemon))
        .route("/shutdown", post(shutdown_daemon))
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
