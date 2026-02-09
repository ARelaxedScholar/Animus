//! Publisher Node
//!
//! Uploads videos to YouTube and manages the publishing process.

use async_trait::async_trait;
use orichalcum::{AsyncNodeLogic, NodeValue};
use reqwest::Client as HttpClient;
use serde::{Deserialize, Serialize};
use sqlx::PgPool;
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{error, info, warn};

use crate::db;
use crate::nodes::SEOMetadata;
use crate::state_keys;
use crate::storage::S3Client;

/// Configuration for YouTube publishing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PublisherConfig {
    /// Default YouTube account ID to use (if not specified in shared state)
    pub default_account_id: Option<i32>,
    /// Default category ID (22 = People & Blogs, 27 = Education)
    pub default_category_id: String,
    /// Default privacy status
    pub default_privacy: String,
}

impl Default for PublisherConfig {
    fn default() -> Self {
        Self {
            default_account_id: None,
            default_category_id: "27".to_string(), // Education
            default_privacy: "private".to_string(), // Start private for review
        }
    }
}

/// YouTube OAuth token response
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct TokenResponse {
    access_token: String,
    expires_in: u64,
}

/// YouTube video insert response
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct YouTubeVideoResponse {
    id: String,
    snippet: Option<YouTubeSnippet>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct YouTubeSnippet {
    title: String,
    #[serde(rename = "publishedAt")]
    published_at: Option<String>,
}

/// The publisher node logic
#[derive(Clone)]
pub struct PublisherLogic {
    pub config: PublisherConfig,
    pub http_client: Arc<HttpClient>,
    pub s3_client: Arc<S3Client>,
    pub db_pool: Arc<PgPool>,
}

impl PublisherLogic {
    pub fn new(config: PublisherConfig, s3_client: Arc<S3Client>, db_pool: Arc<PgPool>) -> Self {
        Self {
            config,
            http_client: Arc::new(HttpClient::new()),
            s3_client,
            db_pool,
        }
    }

    /// Retry publishing a video that failed at the publisher stage.
    /// Downloads from S3 and re-uploads to YouTube.
    pub async fn retry_publish(&self, video_id: uuid::Uuid) -> Result<String, String> {
        info!("Publisher: Retrying upload for video {}", video_id);
        
        // Get video from database
        let video = db::get_video(&self.db_pool, video_id)
            .await
            .map_err(|e| format!("Database error: {}", e))?
            .ok_or_else(|| "Video not found".to_string())?;
        
        // Find which account to use
        let account_id = video.youtube_account_id
            .or(self.config.default_account_id)
            .ok_or_else(|| "No YouTube account associated with this video".to_string())?;

        let account = db::accounts::get_account(&self.db_pool, account_id)
            .await
            .map_err(|e| format!("Failed to fetch account: {}", e))?
            .ok_or_else(|| format!("Account {} not found", account_id))?;

        // Check if it has a video path
        let video_path = video.video_path
            .ok_or_else(|| "Video has no S3 path - not yet assembled".to_string())?;
        
        // Get SEO metadata
        let seo_metadata: SEOMetadata = video.seo_metadata
            .ok_or_else(|| "No SEO metadata".to_string())
            .and_then(|v| serde_json::from_value(v).map_err(|e| format!("Invalid SEO metadata: {}", e)))?;
        
        // Get access token
        let access_token = self.get_access_token(&account).await
            .map_err(|e| format!("Failed to get access token: {}", e))?;
        
        // Download video from S3
        info!("Publisher: Downloading video from S3...");
        let video_data = self.s3_client.download_bytes(&video_path).await
            .map_err(|e| format!("Failed to download video: {}", e))?;
        info!("Publisher: Downloaded {} bytes", video_data.len());
        
        // Upload to YouTube
        info!("Publisher: Uploading to YouTube - '{}'", seo_metadata.title);
        let youtube_video_id = self.upload_video(
            &account,
            &access_token,
            video_data,
            &seo_metadata,
            video.scheduled_at,
        ).await
            .map_err(|e| format!("YouTube upload failed: {}", e))?;
        
        let youtube_url = format!("https://www.youtube.com/watch?v={}", youtube_video_id);
        
        // Upload thumbnail if available
        if let Some(thumb_path) = video.thumbnail_path {
            if let Ok(thumb_data) = self.s3_client.download_bytes(&thumb_path).await {
                match self.set_thumbnail(&access_token, &youtube_video_id, thumb_data).await {
                    Ok(_) => info!("Publisher: Custom thumbnail uploaded successfully"),
                    Err(e) if e.contains("PERMISSION_DENIED") => {
                        warn!("Publisher: {}", e);
                    }
                    Err(e) => {
                        warn!("Failed to set thumbnail: {}", e);
                    }
                }
            }
        }
        
        // Mark as published in database
        db::mark_video_published(&self.db_pool, video_id, &youtube_video_id, &youtube_url)
            .await
            .map_err(|e| format!("Failed to update database: {}", e))?;
        
        info!("Publisher: Retry successful! URL: {}", youtube_url);
        Ok(youtube_url)
    }

    /// Refresh the OAuth access token
    async fn get_access_token(&self, account: &db::accounts::YouTubeAccount) -> Result<String, String> {
        let response = self.http_client
            .post("https://oauth2.googleapis.com/token")
            .form(&[
                ("client_id", &account.client_id),
                ("client_secret", &account.client_secret),
                ("refresh_token", &account.refresh_token),
                ("grant_type", &"refresh_token".to_string()),
            ])
            .send()
            .await
            .map_err(|e| format!("Token refresh failed: {}", e))?;

        if !response.status().is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(format!("Token refresh error: {}", error_text));
        }

        let token_response: TokenResponse = response.json().await
            .map_err(|e| format!("Failed to parse token response: {}", e))?;

        Ok(token_response.access_token)
    }

    /// Upload a video to YouTube using resumable upload
    async fn upload_video(
        &self,
        _account: &db::accounts::YouTubeAccount,
        access_token: &str,
        video_data: Vec<u8>,
        metadata: &SEOMetadata,
        scheduled_publish: Option<chrono::DateTime<chrono::Utc>>,
    ) -> Result<String, String> {
        // Prepare the video metadata
        let privacy_status = if scheduled_publish.is_some() {
            "private" // Will be made public at scheduled time
        } else {
            &self.config.default_privacy
        };

        let video_metadata = serde_json::json!({
            "snippet": {
                "title": metadata.title,
                "description": metadata.description,
                "tags": metadata.tags,
                "categoryId": self.config.default_category_id
            },
            "status": {
                "privacyStatus": privacy_status,
                "selfDeclaredMadeForKids": false
            }
        });

        // Initiate resumable upload
        let init_response = self.http_client
            .post("https://www.googleapis.com/upload/youtube/v3/videos")
            .query(&[
                ("uploadType", "resumable"),
                ("part", "snippet,status"),
            ])
            .header("Authorization", format!("Bearer {}", access_token))
            .header("Content-Type", "application/json")
            .header("X-Upload-Content-Type", "video/mp4")
            .header("X-Upload-Content-Length", video_data.len().to_string())
            .json(&video_metadata)
            .send()
            .await
            .map_err(|e| format!("Upload initiation failed: {}", e))?;

        if !init_response.status().is_success() {
            let error = init_response.text().await.unwrap_or_default();
            return Err(format!("Upload initiation error: {}", error));
        }

        let upload_url = init_response
            .headers()
            .get("Location")
            .and_then(|v| v.to_str().ok())
            .ok_or("No upload URL in response")?
            .to_string();

        // Upload the video data
        let upload_response = self.http_client
            .put(&upload_url)
            .header("Content-Type", "video/mp4")
            .header("Content-Length", video_data.len().to_string())
            .body(video_data)
            .send()
            .await
            .map_err(|e| format!("Video upload failed: {}", e))?;

        if !upload_response.status().is_success() {
            let error = upload_response.text().await.unwrap_or_default();
            return Err(format!("Video upload error: {}", error));
        }

        let youtube_response: YouTubeVideoResponse = upload_response.json().await
            .map_err(|e| format!("Failed to parse upload response: {}", e))?;

        Ok(youtube_response.id)
    }

    /// Set the thumbnail for a video
    async fn set_thumbnail(
        &self,
        access_token: &str,
        video_id: &str,
        thumbnail_data: Vec<u8>,
    ) -> Result<(), String> {
        let response = self.http_client
            .post("https://www.googleapis.com/upload/youtube/v3/thumbnails/set")
            .query(&[("videoId", video_id)])
            .header("Authorization", format!("Bearer {}", access_token))
            .header("Content-Type", "image/png")
            .body(thumbnail_data)
            .send()
            .await
            .map_err(|e| format!("Thumbnail upload failed: {}", e))?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            
            if status == reqwest::StatusCode::FORBIDDEN && error_text.contains("thumbnail") {
                return Err("PERMISSION_DENIED: Custom thumbnails are not enabled for this account. Visit youtube.com/verify to fix this.".to_string());
            }
            
            return Err(format!("Thumbnail upload error {}: {}", status, error_text));
        }

        Ok(())
    }
}

#[async_trait]
impl AsyncNodeLogic for PublisherLogic {
    async fn prep(
        &self,
        _params: &HashMap<String, NodeValue>,
        shared: &HashMap<String, NodeValue>,
    ) -> NodeValue {
        let video_path = shared.get(state_keys::VIDEO_PATH).cloned().unwrap_or(serde_json::json!(null));
        let thumbnail_path = shared.get(state_keys::THUMBNAIL_PATH).cloned().unwrap_or(serde_json::json!(null));
        let seo_metadata = shared.get(state_keys::SEO_METADATA).cloned().unwrap_or(serde_json::json!(null));
        let scheduled_publish = shared.get("scheduled_publish").cloned();
        let youtube_account_id = shared.get("youtube_account_id").cloned();

        serde_json::json!({
            "video_path": video_path,
            "thumbnail_path": thumbnail_path,
            "seo_metadata": seo_metadata,
            "scheduled_publish": scheduled_publish,
            "youtube_account_id": youtube_account_id,
            "is_autonomous": shared.get("is_autonomous").cloned().unwrap_or(serde_json::json!(true))
        })
    }

    async fn exec(&self, input: NodeValue) -> NodeValue {
        let is_autonomous = input.get("is_autonomous").and_then(|v| v.as_bool()).unwrap_or(true);
        
        let account_id = input.get("youtube_account_id")
            .and_then(|v| v.as_i64())
            .map(|id| id as i32)
            .or(self.config.default_account_id)
            .ok_or_else(|| "No YouTube account ID provided".to_string());

        let account_id = match account_id {
            Ok(id) => id,
            Err(e) => return serde_json::json!({ "error": e }),
        };

        // Fetch account from DB
        let account = match db::accounts::get_account(&self.db_pool, account_id).await {
            Ok(Some(a)) => a,
            Ok(None) => return serde_json::json!({ "error": format!("Account {} not found", account_id) }),
            Err(e) => return serde_json::json!({ "error": format!("Database error: {}", e) }),
        };

        let video_path = match input.get("video_path").and_then(|v| v.as_str()) {
            Some(p) => p.to_string(),
            None => return serde_json::json!({ "error": "No video path provided" }),
        };

        let seo_metadata: SEOMetadata = match input.get("seo_metadata")
            .and_then(|v| serde_json::from_value(v.clone()).ok()) {
            Some(m) => m,
            None => return serde_json::json!({ "error": "No SEO metadata provided" }),
        };

        let thumbnail_path = input.get("thumbnail_path").and_then(|v| v.as_str());

        let scheduled_publish: Option<chrono::DateTime<chrono::Utc>> = input.get("scheduled_publish")
            .and_then(|v| v.as_str())
            .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
            .map(|dt| dt.with_timezone(&chrono::Utc));

        info!("Publisher: Uploading video to YouTube account '{}' - '{}'", account.name, seo_metadata.title);

        // Get access token
        let access_token = match self.get_access_token(&account).await {
            Ok(t) => t,
            Err(e) => return serde_json::json!({ "error": format!("Failed to get access token: {}", e) }),
        };

        // Download video from S3
        let video_data = match self.s3_client.download_bytes(&video_path).await {
            Ok(d) => d,
            Err(e) => return serde_json::json!({ "error": format!("Failed to download video: {}", e) }),
        };

        // Upload main video to YouTube
        let youtube_video_id = match self.upload_video(
            &account,
            &access_token,
            video_data,
            &seo_metadata,
            scheduled_publish,
        ).await {
            Ok(id) => id,
            Err(e) => return serde_json::json!({ "error": format!("YouTube upload failed: {}", e) }),
        };

        // --- SHORTS UPLOAD ---
        if let Some(shorts_path) = input.get("shorts_path").and_then(|v| v.as_str()) {
            info!("Publisher: Uploading Short to YouTube...");
            if let Ok(shorts_data) = self.s3_client.download_bytes(shorts_path).await {
                let mut shorts_metadata = seo_metadata.clone();
                shorts_metadata.title = format!("{} #shorts", shorts_metadata.title);
                
                let _ = self.upload_video(
                    &account,
                    &access_token,
                    shorts_data,
                    &shorts_metadata,
                    None, // Upload shorts immediately
                ).await;
            }
        }

        // Upload thumbnail if available
        if let Some(thumb_path) = thumbnail_path {
            if let Ok(thumb_data) = self.s3_client.download_bytes(thumb_path).await {
                match self.set_thumbnail(&access_token, &youtube_video_id, thumb_data).await {
                    Ok(_) => info!("Publisher: Custom thumbnail uploaded successfully"),
                    Err(e) if e.contains("PERMISSION_DENIED") => {
                        warn!("Publisher: {}", e);
                        info!("Publisher: Continuing without custom thumbnail...");
                    }
                    Err(e) => {
                        warn!("Failed to set thumbnail: {}", e);
                        // Continue anyway - thumbnail isn't critical
                    }
                }
            }
        }

        serde_json::json!({
            "success": true,
            "youtube_video_id": youtube_video_id,
            "youtube_url": format!("https://www.youtube.com/watch?v={}", youtube_video_id),
            "title": seo_metadata.title,
            "youtube_account_id": account_id,
            "is_autonomous": is_autonomous
        })
    }

    async fn post(
        &self,
        shared: &mut HashMap<String, NodeValue>,
        _prep_res: NodeValue,
        exec_res: NodeValue,
    ) -> Option<String> {
        if let Some(error) = exec_res.get("error").and_then(|v| v.as_str()) {
            error!("Publisher failed: {}", error);
            shared.insert(state_keys::ERROR.to_string(), serde_json::json!(error));
            
            // Mark video as failed in database
            if let Some(vid) = shared.get(state_keys::VIDEO_ID).and_then(|v| v.as_str()) {
                if let Ok(video_id) = uuid::Uuid::parse_str(vid) {
                    let _ = db::mark_video_failed(&self.db_pool, video_id, "publisher", error).await;
                }
            }
            
            return Some("error".to_string());
        }

        let youtube_video_id = exec_res.get("youtube_video_id")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        let youtube_url = exec_res.get("youtube_url")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        let title = exec_res.get("title")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        info!(
            "Publisher: Video published successfully!\n  Title: {}\n  URL: {}",
            title, youtube_url
        );

        shared.insert(
            state_keys::YOUTUBE_VIDEO_ID.to_string(),
            serde_json::json!(youtube_video_id),
        );

        // Mark production as complete
        shared.insert("production_in_progress".to_string(), serde_json::json!(false));

        let is_autonomous = exec_res.get("is_autonomous").and_then(|v| v.as_bool()).unwrap_or(true);
        
        if is_autonomous {
            shared.insert(
                "last_publish_time".to_string(),
                serde_json::json!(chrono::Utc::now().to_rfc3339()),
            );

            // Increment video count
            let current_count = shared.get("total_video_count")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            shared.insert("total_video_count".to_string(), serde_json::json!(current_count + 1));
        }

        // Mark video as published in database
        if let Some(vid) = shared.get(state_keys::VIDEO_ID).and_then(|v| v.as_str()) {
            if let Ok(video_id) = uuid::Uuid::parse_str(vid) {
                // Update account ID if we have it
                if let Some(acc_id) = exec_res.get("youtube_account_id").and_then(|v| v.as_i64()) {
                    let _ = sqlx::query!(
                        "UPDATE videos SET youtube_account_id = $1 WHERE id = $2",
                        acc_id as i32,
                        video_id
                    ).execute(&*self.db_pool).await;
                }

                if let Err(e) = db::mark_video_published(
                    &self.db_pool,
                    video_id,
                    youtube_video_id,
                    youtube_url,
                ).await {
                    error!("Failed to mark video as published in database: {}", e);
                }
            }
        }

        // Flow complete
        None
    }

    fn clone_box(&self) -> Box<dyn AsyncNodeLogic> {
        Box::new(self.clone())
    }
}
