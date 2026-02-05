//! Asset Collector Node
//!
//! Gathers B-roll footage and images from Pexels and Stable Diffusion
//! based on the visual suggestions in the script.

use async_trait::async_trait;
use orichalcum::{AsyncNodeLogic, NodeValue};
use reqwest::Client as HttpClient;
use serde::{Deserialize, Serialize};
use sqlx::PgPool;
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{error, info, warn};

use crate::db;
use crate::nodes::{AssetFile, AssetManifest, Script, SectionAssets};
use crate::state_keys;
use crate::storage::S3Client;

/// Configuration for asset collection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssetCollectorConfig {
    /// Pexels API key
    pub pexels_api_key: String,
    /// Stable Diffusion API URL (optional)
    pub sd_api_url: Option<String>,
    /// Minimum clips per section
    pub min_clips_per_section: u32,
}

impl Default for AssetCollectorConfig {
    fn default() -> Self {
        Self {
            pexels_api_key: String::new(),
            sd_api_url: None,
            min_clips_per_section: 3,
        }
    }
}

/// Pexels video search response
#[derive(Debug, Deserialize)]
struct PexelsVideoResponse {
    videos: Vec<PexelsVideo>,
}

#[derive(Debug, Deserialize)]
struct PexelsVideo {
    id: u64,
    duration: u32,
    video_files: Vec<PexelsVideoFile>,
}

#[derive(Debug, Deserialize)]
struct PexelsVideoFile {
    link: String,
    quality: String,
    width: u32,
    height: u32,
}

/// The asset collector node logic
#[derive(Clone)]
pub struct AssetCollectorLogic {
    pub config: AssetCollectorConfig,
    pub http_client: Arc<HttpClient>,
    pub s3_client: Arc<S3Client>,
    pub db_pool: Arc<PgPool>,
}

impl AssetCollectorLogic {
    pub fn new(config: AssetCollectorConfig, s3_client: Arc<S3Client>, db_pool: Arc<PgPool>) -> Self {
        Self {
            config,
            http_client: Arc::new(HttpClient::new()),
            s3_client,
            db_pool,
        }
    }

    /// Search Pexels for videos matching a query
    async fn search_pexels_videos(&self, query: &str, per_page: u32) -> Result<Vec<PexelsVideo>, String> {
        let url = format!(
            "https://api.pexels.com/videos/search?query={}&per_page={}&orientation=landscape",
            urlencoding::encode(query),
            per_page
        );

        let response = self.http_client
            .get(&url)
            .header("Authorization", &self.config.pexels_api_key)
            .send()
            .await
            .map_err(|e| format!("Pexels request failed: {}", e))?;

        if !response.status().is_success() {
            return Err(format!("Pexels API error: {}", response.status()));
        }

        let pexels_response: PexelsVideoResponse = response.json().await
            .map_err(|e| format!("Failed to parse Pexels response: {}", e))?;

        Ok(pexels_response.videos)
    }

    /// Download a video and upload to S3
    async fn download_and_upload_video(
        &self,
        video: &PexelsVideo,
        video_id: &str,
        section_index: usize,
        clip_index: usize,
    ) -> Result<AssetFile, String> {
        // Find the best quality video file (prefer HD)
        let video_file = video.video_files.iter()
            .filter(|f| f.width >= 1280)
            .max_by_key(|f| f.width)
            .or_else(|| video.video_files.first())
            .ok_or("No video files available")?;

        // Download the video
        let video_bytes = self.http_client
            .get(&video_file.link)
            .send()
            .await
            .map_err(|e| format!("Failed to download video: {}", e))?
            .bytes()
            .await
            .map_err(|e| format!("Failed to read video bytes: {}", e))?;

        // Upload to S3
        let key = format!(
            "assets/{}/section_{}/clip_{}.mp4",
            video_id, section_index, clip_index
        );

        self.s3_client.upload_bytes(&key, video_bytes.to_vec(), "video/mp4").await
            .map_err(|e| format!("Failed to upload to S3: {}", e))?;

        Ok(AssetFile {
            path: key,
            source: "pexels".to_string(),
            duration_seconds: Some(video.duration as f64),
            description: format!("Pexels video {}", video.id),
        })
    }
}

#[async_trait]
impl AsyncNodeLogic for AssetCollectorLogic {
    async fn prep(
        &self,
        _params: &HashMap<String, NodeValue>,
        shared: &HashMap<String, NodeValue>,
    ) -> NodeValue {
        let script = shared.get(state_keys::SCRIPT).cloned().unwrap_or(serde_json::json!(null));
        serde_json::json!({ "script": script })
    }

    async fn exec(&self, input: NodeValue) -> NodeValue {
        let script: Script = match input.get("script").and_then(|s| serde_json::from_value(s.clone()).ok()) {
            Some(s) => s,
            None => return serde_json::json!({ "error": "No script provided" }),
        };

        info!("AssetCollector: Gathering assets for video {}", script.video_id);

        let mut section_assets: Vec<SectionAssets> = Vec::new();
        let video_id = script.video_id.to_string();

        // Collect all sections (hook + main sections + cta)
        let mut all_sections = vec![&script.hook];
        all_sections.extend(script.sections.iter());
        all_sections.push(&script.cta);

        for (section_idx, section) in all_sections.iter().enumerate() {
            let mut video_clips: Vec<AssetFile> = Vec::new();

            for (suggestion_idx, suggestion) in section.visual_suggestions.iter().enumerate() {
                // Search Pexels for this visual suggestion
                match self.search_pexels_videos(suggestion, 3).await {
                    Ok(videos) => {
                        for (clip_idx, video) in videos.iter().take(2).enumerate() {
                            match self.download_and_upload_video(
                                video,
                                &video_id,
                                section_idx,
                                suggestion_idx * 10 + clip_idx,
                            ).await {
                                Ok(asset) => video_clips.push(asset),
                                Err(e) => warn!("Failed to download clip: {}", e),
                            }
                        }
                    }
                    Err(e) => warn!("Pexels search failed for '{}': {}", suggestion, e),
                }

                // Rate limiting
                tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;
            }

            section_assets.push(SectionAssets {
                section_title: section.title.clone(),
                video_clips,
                images: vec![], // Could add Stable Diffusion images here
            });
        }

        let manifest = AssetManifest {
            video_id: script.video_id,
            background_music: None, // TODO: Add royalty-free music
            section_assets,
        };

        serde_json::json!({
            "success": true,
            "manifest": manifest
        })
    }

    async fn post(
        &self,
        shared: &mut HashMap<String, NodeValue>,
        _prep_res: NodeValue,
        exec_res: NodeValue,
    ) -> Option<String> {
        if let Some(error) = exec_res.get("error").and_then(|v| v.as_str()) {
            error!("AssetCollector failed: {}", error);
            shared.insert(state_keys::ERROR.to_string(), serde_json::json!(error));
            
            // Mark video as failed in database
            if let Some(vid) = shared.get(state_keys::VIDEO_ID).and_then(|v| v.as_str()) {
                if let Ok(video_id) = uuid::Uuid::parse_str(vid) {
                    let _ = db::mark_video_failed(&self.db_pool, video_id, "asset_collector", error).await;
                }
            }
            
            return Some("error".to_string());
        }

        if let Some(manifest) = exec_res.get("manifest") {
            shared.insert(state_keys::ASSET_MANIFEST.to_string(), manifest.clone());
            info!("AssetCollector: Gathered assets successfully");
            
            // Persist asset_manifest to database
            if let Some(vid) = shared.get(state_keys::VIDEO_ID).and_then(|v| v.as_str()) {
                if let Ok(video_id) = uuid::Uuid::parse_str(vid) {
                    if let Err(e) = db::update_video_json_field(
                        &self.db_pool,
                        video_id,
                        "asset_manifest",
                        manifest.clone(),
                    ).await {
                        error!("Failed to persist asset_manifest to database: {}", e);
                    }
                }
            }
        }

        Some("default".to_string())
    }

    fn clone_box(&self) -> Box<dyn AsyncNodeLogic> {
        Box::new(self.clone())
    }
}
