//! HTTP Control API using Axum

use axum::{
    extract::State,
    http::StatusCode,
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;

/// Daemon state exposed to the API
#[derive(Clone)]
pub struct AppState {
    /// Whether the daemon is paused
    pub paused: Arc<RwLock<bool>>,
    /// Shutdown signal sender
    pub shutdown_tx: tokio::sync::watch::Sender<bool>,
    /// Current production status
    pub current_status: Arc<RwLock<DaemonStatus>>,
}

/// Current daemon status
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DaemonStatus {
    pub running: bool,
    pub paused: bool,
    pub current_video_id: Option<String>,
    pub current_stage: Option<String>,
    pub videos_produced: u32,
    pub last_error: Option<String>,
}

/// API response wrapper
#[derive(Serialize)]
struct ApiResponse<T> {
    success: bool,
    data: Option<T>,
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

    fn err(error: impl Into<String>) -> ApiResponse<()> {
        ApiResponse {
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
