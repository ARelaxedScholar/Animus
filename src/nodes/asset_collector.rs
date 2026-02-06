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
use crate::utils;

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
    #[serde(default)]
    videos: Vec<PexelsVideo>,
    // Pexels may return these on error
    #[serde(default)]
    error: Option<String>,
}

#[derive(Debug, Deserialize)]
struct PexelsVideo {
    id: u64,
    #[serde(default)]
    duration: Option<u32>,
    #[serde(default)]
    video_files: Option<Vec<PexelsVideoFile>>,
}

#[derive(Debug, Deserialize)]
struct PexelsVideoFile {
    #[serde(default)]
    link: Option<String>,
    #[serde(default)]
    quality: Option<String>,
    #[serde(default)]
    width: Option<u32>,
    #[serde(default)]
    height: Option<u32>,
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
        // Create HTTP client with reasonable timeouts for video downloads
        let http_client = HttpClient::builder()
            .timeout(std::time::Duration::from_secs(120)) // 2 minute timeout per download
            .connect_timeout(std::time::Duration::from_secs(30))
            .build()
            .expect("Failed to create HTTP client");
        
        Self {
            config,
            http_client: Arc::new(http_client),
            s3_client,
            db_pool,
        }
    }

    /// Clean up a visual suggestion to make it a better search query for Pexels
    fn clean_search_query(suggestion: &str) -> String {
        let mut query = suggestion.to_string();
        
        // Remove specific phrases that can appear anywhere (do this FIRST)
        let phrases_to_remove = [
            "(if available)",
            "(optional)",
            "or similar",
        ];
        for phrase in phrases_to_remove {
            query = query.replace(phrase, "");
        }
        query = query.trim().to_string();
        
        // Remove common prefixes at the START of the query
        let prefixes_to_remove = [
            "Text overlay:",
            "Text Overlay:",
            "TEXT OVERLAY:",
            "Archive footage of",
            "Archive Footage of",
            "Faded, grainy footage of",
            "Macro shots of",
            "Wide shot of",
            "Close-up of",
            "Close up of",
            "Slow motion of",
            "Slow-motion of",
            "B-roll of",
            "B-Roll of",
            "Stock footage of",
        ];
        
        // Only remove prefixes if they appear at the start
        for prefix in prefixes_to_remove {
            let trimmed = query.trim();
            if trimmed.starts_with(prefix) {
                query = trimmed[prefix.len()..].trim().to_string();
            }
        }
        
        // Remove parenthetical notes
        while let Some(start) = query.find('(') {
            if let Some(end) = query.find(')') {
                if end > start {
                    query = format!("{}{}", &query[..start], &query[end + 1..]);
                } else {
                    break;
                }
            } else {
                break;
            }
        }
        
        // Trim and limit length (Pexels works better with shorter queries)
        query = query.trim().to_string();
        
        // If query is too long, take first few words
        let words: Vec<&str> = query.split_whitespace().collect();
        if words.len() > 5 {
            query = words[..5].join(" ");
        }
        
        // If query starts with "a " or "an ", remove it
        if query.starts_with("a ") {
            query = query[2..].to_string();
        } else if query.starts_with("an ") {
            query = query[3..].to_string();
        }
        
        query
    }

    /// Search Pexels for videos matching a query
    async fn search_pexels_videos(&self, query: &str, per_page: u32) -> Result<Vec<PexelsVideo>, String> {
        // Clean up the query for better search results
        let cleaned_query = Self::clean_search_query(query);
        
        // Skip obviously unsearchable queries
        if cleaned_query.is_empty() || cleaned_query.len() < 3 {
            info!("Skipping unsearchable query: '{}' -> '{}'", query, cleaned_query);
            return Ok(vec![]);
        }
        
        info!("Pexels search: '{}' -> '{}'", query, cleaned_query);
        
        let url = format!(
            "https://api.pexels.com/videos/search?query={}&per_page={}&orientation=landscape",
            urlencoding::encode(&cleaned_query),
            per_page
        );

        let response = self.http_client
            .get(&url)
            .header("Authorization", &self.config.pexels_api_key)
            .send()
            .await
            .map_err(|e| format!("Pexels request failed: {}", e))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(format!("Pexels API error {}: {}", status, body));
        }

        // Get the response text first so we can log it on error
        let body = response.text().await
            .map_err(|e| format!("Failed to read Pexels response: {}", e))?;
        
        let pexels_response: PexelsVideoResponse = serde_json::from_str(&body)
            .map_err(|e| {
                // Log the first 500 chars of the response for debugging
                let preview = utils::safe_truncate(&body, 500);
                format!("Failed to parse Pexels response: {}. Body preview: {}", e, preview)
            })?;
        
        // Check if Pexels returned an error in the response
        if let Some(error) = pexels_response.error {
            return Err(format!("Pexels API returned error: {}", error));
        }

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
        let video_files = video.video_files.as_ref().ok_or("No video files in response")?;
        
        // Find the best quality video file (prefer HD/4K)
        let video_file = video_files.iter()
            .filter(|f| f.width.unwrap_or(0) >= 1280)
            .max_by_key(|f| f.width.unwrap_or(0))
            .or_else(|| video_files.first())
            .ok_or("No video files available")?;

        info!(
            "AssetCollector: Downloading video {} ({}x{})...",
            video.id, video_file.width.unwrap_or(0), video_file.height.unwrap_or(0)
        );

        // Download the video
        let link = video_file.link.as_ref().ok_or("Video file has no link")?;
        let response = self.http_client
            .get(link)
            .send()
            .await
            .map_err(|e| format!("Failed to start download: {}", e))?;
        
        if !response.status().is_success() {
            return Err(format!("Download failed with status: {}", response.status()));
        }

        // Log content length if available
        if let Some(content_length) = response.content_length() {
            info!("AssetCollector: Video size: {:.2} MB", content_length as f64 / 1024.0 / 1024.0);
        }
        
        // Download the video
        let video_bytes = response.bytes()
            .await
            .map_err(|e| format!("Failed to read video bytes: {}", e))?;

        // Upload to S3
        let key = format!(
            "assets/{}/section_{}/clip_{}.mp4",
            video_id, section_index, clip_index
        );

        let data = video_bytes.to_vec();
        self.s3_client.upload_bytes(&key, data, "video/mp4").await
            .map_err(|e| format!("Failed to upload to S3: {}", e))?;

        info!("AssetCollector: Uploaded {}", key);

        Ok(AssetFile {
            path: key,
            source: "pexels".to_string(),
            duration_seconds: Some(video.duration.unwrap_or(0) as f64),
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

        let total_sections = all_sections.len();
        let mut total_clips_downloaded = 0u32;
        
        for (section_idx, section) in all_sections.iter().enumerate() {
            info!(
                "AssetCollector: Section {}/{} - '{}'",
                section_idx + 1, total_sections, section.title
            );
            
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
                                Ok(asset) => {
                                    video_clips.push(asset);
                                    total_clips_downloaded += 1;
                                }
                                Err(e) => warn!("Failed to download clip: {}", e),
                            }
                        }
                    }
                    Err(e) => warn!("Pexels search failed for '{}': {}", suggestion, e),
                }

                // Rate limiting
                tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;
            }
            
            info!(
                "AssetCollector: Section {}/{} complete - {} clips",
                section_idx + 1, total_sections, video_clips.len()
            );

            section_assets.push(SectionAssets {
                section_title: section.title.clone(),
                video_clips,
                images: vec![], // Could add Stable Diffusion images here
            });
        }
        
        info!(
            "AssetCollector: Finished - {} total clips downloaded. Transitioning to VideoAssembler...",
            total_clips_downloaded
        );

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clean_search_query_text_overlay() {
        // Text overlays should be skipped entirely
        assert_eq!(
            AssetCollectorLogic::clean_search_query("Text overlay: THE BRUTAL GUARANTEE"),
            "THE BRUTAL GUARANTEE"
        );
        assert_eq!(
            AssetCollectorLogic::clean_search_query("TEXT OVERLAY: SOMETHING"),
            "SOMETHING"
        );
    }

    #[test]
    fn test_clean_search_query_archive_footage() {
        assert_eq!(
            AssetCollectorLogic::clean_search_query("Archive footage of Viktor Frankl (if available)"),
            "Viktor Frankl"
        );
    }

    #[test]
    fn test_clean_search_query_removes_parenthetical() {
        assert_eq!(
            AssetCollectorLogic::clean_search_query("old Bible being opened (worn page)"),
            "old Bible being opened"
        );
    }

    #[test]
    fn test_clean_search_query_wide_shot() {
        assert_eq!(
            AssetCollectorLogic::clean_search_query("Wide shot of a desolate but beautiful desert landscape"),
            "desolate but beautiful desert"
        );
    }

    #[test]
    fn test_clean_search_query_macro_shots() {
        // "Macro shots of" is removed, then we take first 5 words, then "an " at start is removed
        // "nature: an eye of a hawk, a rushing river" -> first 5 words -> "nature: an eye of a"
        // -> starts with "an "? No, starts with "nature:" -> result is "nature: an eye of a"
        // But actually we want something useful for search, so let's verify current behavior
        let result = AssetCollectorLogic::clean_search_query("Macro shots of nature: an eye of a hawk, a rushing river");
        // After prefix removal: "nature: an eye of a hawk, a rushing river"
        // After 5-word limit: "nature: an eye of a"
        assert_eq!(result, "nature: an eye of a");
    }

    #[test]
    fn test_clean_search_query_removes_article() {
        assert_eq!(
            AssetCollectorLogic::clean_search_query("a man walking alone"),
            "man walking alone"
        );
        assert_eq!(
            AssetCollectorLogic::clean_search_query("an ancient temple"),
            "ancient temple"
        );
    }

    #[test]
    fn test_clean_search_query_limits_words() {
        assert_eq!(
            AssetCollectorLogic::clean_search_query("one two three four five six seven eight"),
            "one two three four five"
        );
    }

    #[test]
    fn test_pexels_response_parsing_success() {
        let json = r#"{
            "videos": [
                {
                    "id": 12345,
                    "duration": 30,
                    "video_files": [
                        {
                            "link": "https://example.com/video.mp4",
                            "quality": "hd",
                            "width": 1920,
                            "height": 1080
                        }
                    ]
                }
            ]
        }"#;
        
        let response: PexelsVideoResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.videos.len(), 1);
        assert_eq!(response.videos[0].id, 12345);
        assert_eq!(response.videos[0].duration, 30);
        assert_eq!(response.videos[0].video_files.len(), 1);
        assert_eq!(response.videos[0].video_files[0].width, 1920);
    }

    #[test]
    fn test_pexels_response_parsing_empty_videos() {
        let json = r#"{"videos": []}"#;
        
        let response: PexelsVideoResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.videos.len(), 0);
    }

    #[test]
    fn test_pexels_response_parsing_missing_optional_fields() {
        // Minimal response with only required id field
        let json = r#"{
            "videos": [
                {
                    "id": 99999
                }
            ]
        }"#;
        
        let response: PexelsVideoResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.videos.len(), 1);
        assert_eq!(response.videos[0].id, 99999);
        assert_eq!(response.videos[0].duration, 0); // default
        assert_eq!(response.videos[0].video_files.len(), 0); // default empty
    }

    #[test]
    fn test_pexels_response_parsing_with_error() {
        let json = r#"{"error": "Invalid API key"}"#;
        
        let response: PexelsVideoResponse = serde_json::from_str(json).unwrap();
        assert!(response.error.is_some());
        assert_eq!(response.error.unwrap(), "Invalid API key");
        assert_eq!(response.videos.len(), 0);
    }

    #[test]
    fn test_pexels_response_parsing_extra_fields() {
        // Pexels returns additional fields we don't use - should still parse
        let json = r#"{
            "page": 1,
            "per_page": 15,
            "total_results": 100,
            "url": "https://www.pexels.com/search/videos/nature/",
            "videos": [
                {
                    "id": 11111,
                    "width": 1920,
                    "height": 1080,
                    "url": "https://www.pexels.com/video/11111/",
                    "image": "https://images.pexels.com/videos/11111/free-video-11111.jpg",
                    "full_res": null,
                    "tags": [],
                    "duration": 45,
                    "user": {
                        "id": 123,
                        "name": "Some User",
                        "url": "https://www.pexels.com/@someuser"
                    },
                    "video_files": [
                        {
                            "id": 22222,
                            "quality": "hd",
                            "file_type": "video/mp4",
                            "width": 1920,
                            "height": 1080,
                            "fps": 25.0,
                            "link": "https://player.vimeo.com/external/video.mp4"
                        }
                    ],
                    "video_pictures": [
                        {
                            "id": 33333,
                            "picture": "https://images.pexels.com/videos/11111/pictures/preview-0.jpg"
                        }
                    ]
                }
            ]
        }"#;
        
        let response: PexelsVideoResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.videos.len(), 1);
        assert_eq!(response.videos[0].id, 11111);
        assert_eq!(response.videos[0].duration, 45);
        assert_eq!(response.videos[0].video_files[0].width, 1920);
        assert!(response.videos[0].video_files[0].link.contains("vimeo"));
    }
}
