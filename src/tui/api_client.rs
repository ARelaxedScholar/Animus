//! HTTP API Client for the TUI
//!
//! Communicates with the Animus daemon via its REST API.

use reqwest::Client;
use serde::{Deserialize, Serialize};

/// API response wrapper
#[derive(Debug, Deserialize)]
pub struct ApiResponse<T> {
    pub success: bool,
    pub data: Option<T>,
    pub error: Option<String>,
}

/// Daemon status from /status endpoint
#[derive(Debug, Clone, Deserialize, Default)]
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

/// Video summary from /videos endpoint
#[derive(Debug, Clone, Deserialize)]
pub struct VideoSummary {
    pub id: String,
    pub status: String,
    pub title: Option<String>,
    pub youtube_url: Option<String>,
    pub created_at: String,
    pub failed_at_stage: Option<String>,
    pub error_message: Option<String>,
}

/// Queue item from /queue endpoint
#[derive(Debug, Clone, Deserialize)]
pub struct QueueItem {
    pub id: i32,
    pub seed_topic: String,
    pub source_focus: Option<String>,
}

/// Statistics from /stats endpoint
#[derive(Debug, Clone, Deserialize, Default)]
pub struct Stats {
    pub total_videos: i64,
    pub published: i64,
    pub failed: i64,
    pub producing: i64,
    pub queue_length: i64,
    pub success_rate: f64,
    pub recent_failures: Vec<FailureSummary>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct FailureSummary {
    pub stage: String,
    pub count: i64,
}

/// Add to queue request
#[derive(Debug, Serialize)]
pub struct AddToQueueRequest {
    pub topic: String,
    pub source: Option<String>,
}

/// API Client for the Animus daemon
#[derive(Clone)]
pub struct AnimusClient {
    client: Client,
    base_url: String,
    api_key: String,
}

impl AnimusClient {
    pub fn new(base_url: &str, api_key: &str) -> Self {
        Self {
            client: Client::new(),
            base_url: base_url.trim_end_matches('/').to_string(),
            api_key: api_key.to_string(),
        }
    }

    async fn get<T: for<'de> Deserialize<'de>>(&self, path: &str) -> Result<T, String> {
        let url = format!("{}{}", self.base_url, path);
        let response = self.client
            .get(&url)
            .header("X-API-Key", &self.api_key)
            .send()
            .await
            .map_err(|e| format!("Request failed: {}", e))?;

        if !response.status().is_success() {
            return Err(format!("HTTP {}", response.status()));
        }

        let api_response: ApiResponse<T> = response.json().await
            .map_err(|e| format!("Parse error: {}", e))?;

        if api_response.success {
            api_response.data.ok_or_else(|| "No data in response".to_string())
        } else {
            Err(api_response.error.unwrap_or_else(|| "Unknown error".to_string()))
        }
    }

    async fn post<T: for<'de> Deserialize<'de>>(&self, path: &str) -> Result<T, String> {
        let url = format!("{}{}", self.base_url, path);
        let response = self.client
            .post(&url)
            .header("X-API-Key", &self.api_key)
            .send()
            .await
            .map_err(|e| format!("Request failed: {}", e))?;

        if !response.status().is_success() {
            return Err(format!("HTTP {}", response.status()));
        }

        let api_response: ApiResponse<T> = response.json().await
            .map_err(|e| format!("Parse error: {}", e))?;

        if api_response.success {
            api_response.data.ok_or_else(|| "No data in response".to_string())
        } else {
            Err(api_response.error.unwrap_or_else(|| "Unknown error".to_string()))
        }
    }

    async fn post_json<B: Serialize, T: for<'de> Deserialize<'de>>(&self, path: &str, body: &B) -> Result<T, String> {
        let url = format!("{}{}", self.base_url, path);
        let response = self.client
            .post(&url)
            .header("X-API-Key", &self.api_key)
            .json(body)
            .send()
            .await
            .map_err(|e| format!("Request failed: {}", e))?;

        if !response.status().is_success() {
            return Err(format!("HTTP {}", response.status()));
        }

        let api_response: ApiResponse<T> = response.json().await
            .map_err(|e| format!("Parse error: {}", e))?;

        if api_response.success {
            api_response.data.ok_or_else(|| "No data in response".to_string())
        } else {
            Err(api_response.error.unwrap_or_else(|| "Unknown error".to_string()))
        }
    }

    async fn delete<T: for<'de> Deserialize<'de>>(&self, path: &str) -> Result<T, String> {
        let url = format!("{}{}", self.base_url, path);
        let response = self.client
            .delete(&url)
            .header("X-API-Key", &self.api_key)
            .send()
            .await
            .map_err(|e| format!("Request failed: {}", e))?;

        if !response.status().is_success() {
            return Err(format!("HTTP {}", response.status()));
        }

        let api_response: ApiResponse<T> = response.json().await
            .map_err(|e| format!("Parse error: {}", e))?;

        if api_response.success {
            api_response.data.ok_or_else(|| "No data in response".to_string())
        } else {
            Err(api_response.error.unwrap_or_else(|| "Unknown error".to_string()))
        }
    }

    // =========================================================================
    // Status & Health
    // =========================================================================

    pub async fn health(&self) -> Result<bool, String> {
        let url = format!("{}/health", self.base_url);
        let response = self.client
            .get(&url)
            .send()
            .await
            .map_err(|e| format!("Health check failed: {}", e))?;
        Ok(response.status().is_success())
    }

    pub async fn status(&self) -> Result<DaemonStatus, String> {
        self.get("/status").await
    }

    pub async fn stats(&self) -> Result<Stats, String> {
        self.get("/stats").await
    }

    // =========================================================================
    // Daemon Control
    // =========================================================================

    pub async fn pause(&self) -> Result<String, String> {
        self.post("/pause").await
    }

    pub async fn resume(&self) -> Result<String, String> {
        self.post("/resume").await
    }

    pub async fn shutdown(&self) -> Result<String, String> {
        self.post("/shutdown").await
    }

    // =========================================================================
    // Video Management
    // =========================================================================

    pub async fn list_videos(&self, status: Option<&str>, limit: Option<i64>) -> Result<Vec<VideoSummary>, String> {
        let mut path = "/videos".to_string();
        let mut params = vec![];
        if let Some(s) = status {
            params.push(format!("status={}", s));
        }
        if let Some(l) = limit {
            params.push(format!("limit={}", l));
        }
        if !params.is_empty() {
            path = format!("{}?{}", path, params.join("&"));
        }
        self.get(&path).await
    }

    pub async fn retry_video(&self, video_id: &str) -> Result<String, String> {
        self.post(&format!("/videos/{}/retry", video_id)).await
    }

    /// Download a video file to a local path
    pub async fn download_video(&self, video_id: &str, output_path: &str) -> Result<String, String> {
        let url = format!("{}/videos/{}/download", self.base_url, video_id);
        let response = self.client
            .get(&url)
            .header("X-API-Key", &self.api_key)
            .send()
            .await
            .map_err(|e| format!("Download request failed: {}", e))?;

        if !response.status().is_success() {
            // Try to parse error response
            if let Ok(api_response) = response.json::<ApiResponse<String>>().await {
                return Err(api_response.error.unwrap_or_else(|| "Download failed".to_string()));
            }
            return Err("Download failed".to_string());
        }

        // Get filename from Content-Disposition header if available
        let filename = response
            .headers()
            .get("content-disposition")
            .and_then(|v| v.to_str().ok())
            .and_then(|s| {
                s.split("filename=").nth(1).map(|f| f.trim_matches('"').to_string())
            })
            .unwrap_or_else(|| format!("{}.mp4", video_id));

        let bytes = response.bytes().await
            .map_err(|e| format!("Failed to read response: {}", e))?;

        // Ensure output directory exists
        let output_dir = std::path::Path::new(output_path);
        if !output_dir.exists() {
            std::fs::create_dir_all(output_dir)
                .map_err(|e| format!("Failed to create output directory: {}", e))?;
        }

        let file_path = output_dir.join(&filename);
        std::fs::write(&file_path, &bytes)
            .map_err(|e| format!("Failed to write file: {}", e))?;

        Ok(file_path.to_string_lossy().to_string())
    }

    // =========================================================================
    // Queue Management
    // =========================================================================

    pub async fn list_queue(&self) -> Result<Vec<QueueItem>, String> {
        self.get("/queue").await
    }

    pub async fn add_to_queue(&self, topic: &str, source: Option<&str>) -> Result<i32, String> {
        let req = AddToQueueRequest {
            topic: topic.to_string(),
            source: source.map(|s| s.to_string()),
        };
        self.post_json("/queue", &req).await
    }

    pub async fn remove_from_queue(&self, id: i32) -> Result<String, String> {
        self.delete(&format!("/queue/{}", id)).await
    }

    pub async fn clear_queue(&self) -> Result<u64, String> {
        self.post("/queue/clear").await
    }
}
