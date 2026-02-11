//! Thumbnail Generator Node
//!
//! Creates eye-catching thumbnails using HTML templates and optional AI generation.

use async_trait::async_trait;
use orichalcum::llm::{Client, Enabled, Providers};
use orichalcum::{AsyncNodeLogic, NodeValue};
use serde::{Deserialize, Serialize};
use sqlx::PgPool;
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{error, info, warn};

use crate::db;
use crate::nodes::TopicBrief;
use crate::state_keys;
use crate::storage::S3Client;

/// Configuration for thumbnail generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThumbnailConfig {
    /// Template directory path
    pub template_dir: String,
    /// Output width
    pub width: u32,
    /// Output height
    pub height: u32,
    /// Prompt prefix for Imagen
    pub prompt_prefix: String,
}

impl Default for ThumbnailConfig {
    fn default() -> Self {
        Self {
            template_dir: "templates/thumbnails".to_string(),
            width: 1280,
            height: 720,
            prompt_prefix: "A high-quality, cinematic YouTube thumbnail for a video titled:".to_string(),
        }
    }
}

/// The thumbnail generator node logic
#[derive(Clone)]
pub struct ThumbnailLogic {
    pub config: ThumbnailConfig,
    pub llm_client: Arc<Client<Providers<orichalcum::llm::Disabled, Enabled, Enabled>>>,
    pub s3_client: Arc<S3Client>,
    pub db_pool: Arc<PgPool>,
}

impl ThumbnailLogic {
    pub fn new(
        config: ThumbnailConfig,
        llm_client: Arc<Client<Providers<orichalcum::llm::Disabled, Enabled, Enabled>>>,
        s3_client: Arc<S3Client>,
        db_pool: Arc<PgPool>,
    ) -> Self {
        Self {
            config,
            llm_client,
            s3_client,
            db_pool,
        }
    }

    /// Generate a thumbnail using Gemini/Imagen
    async fn generate_imagen_thumbnail(
        &self,
        title: &str,
        _video_id: &str,
    ) -> Result<Vec<u8>, String> {
        info!("Thumbnail: Calling Gemini/Imagen for '{}'", title);

        // TODO: Implement Gemini Imagen generation
        Err("Imagen generation not implemented".to_string())
    }

    /// Fallback: Generate a simple gradient thumbnail with text overlay
    async fn generate_simple_thumbnail(
        &self,
        _title: &str,
        _video_id: &str,
    ) -> Result<Vec<u8>, String> {
        warn!("Thumbnail: Using fallback gradient thumbnail");
        use image::{Rgb, RgbImage};
        use std::io::Cursor;

        let width = self.config.width;
        let height = self.config.height;

        // Create a gradient background
        let mut img = RgbImage::new(width, height);
        
        for y in 0..height {
            for x in 0..width {
                // Dark blue to purple gradient
                let r = (30.0 + (x as f32 / width as f32) * 40.0) as u8;
                let g = (20.0 + (y as f32 / height as f32) * 30.0) as u8;
                let b = (80.0 + (x as f32 / width as f32) * 60.0) as u8;
                img.put_pixel(x, y, Rgb([r, g, b]));
            }
        }

        // Note: For proper text rendering, we'd use a library like `imageproc` with font support
        // or render HTML with Puppeteer. This is a placeholder.

        // Encode to PNG
        let mut buffer = Cursor::new(Vec::new());
        img.write_to(&mut buffer, image::ImageFormat::Png)
            .map_err(|e| format!("Failed to encode thumbnail: {}", e))?;

        Ok(buffer.into_inner())
    }
}

#[async_trait]
impl AsyncNodeLogic for ThumbnailLogic {
    async fn prep(
        &self,
        _params: &HashMap<String, NodeValue>,
        shared: &HashMap<String, NodeValue>,
    ) -> NodeValue {
        let topic_brief = shared.get(state_keys::TOPIC_BRIEF).cloned().unwrap_or(serde_json::json!(null));
        let video_id = shared.get(state_keys::VIDEO_ID).cloned().unwrap_or(serde_json::json!(null));
        
        serde_json::json!({
            "topic_brief": topic_brief,
            "video_id": video_id
        })
    }

    async fn exec(&self, input: NodeValue) -> NodeValue {
        let video_id = input.get("video_id")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
            .to_string();

        let topic_brief: Option<TopicBrief> = input.get("topic_brief")
            .and_then(|v| serde_json::from_value(v.clone()).ok());

        let title = topic_brief
            .as_ref()
            .map(|t| t.topic.clone())
            .unwrap_or_else(|| "Untitled".to_string());

        info!("Thumbnail: Generating thumbnail for '{}'", title);

        // Try Imagen first, fallback to simple gradient
        let thumbnail_bytes = match self.generate_imagen_thumbnail(&title, &video_id).await {
            Ok(bytes) => bytes,
            Err(e) => {
                error!("Thumbnail: Imagen failed, falling back: {}", e);
                match self.generate_simple_thumbnail(&title, &video_id).await {
                    Ok(bytes) => bytes,
                    Err(e) => return serde_json::json!({ "error": format!("Thumbnail generation failed: {}", e) }),
                }
            }
        };

        // Upload to S3
        let thumbnail_path = format!("thumbnails/{}/thumbnail.png", video_id);
        if let Err(e) = self.s3_client.upload_bytes(&thumbnail_path, thumbnail_bytes, "image/png").await {
            return serde_json::json!({ "error": format!("Failed to upload thumbnail: {}", e) });
        }

        serde_json::json!({
            "success": true,
            "thumbnail_path": thumbnail_path,
            "video_id": video_id
        })
    }

    async fn post(
        &self,
        shared: &mut HashMap<String, NodeValue>,
        _prep_res: NodeValue,
        exec_res: NodeValue,
    ) -> Option<String> {
        if let Some(error) = exec_res.get("error").and_then(|v| v.as_str()) {
            error!("Thumbnail generation failed: {}", error);
            shared.insert(state_keys::ERROR.to_string(), serde_json::json!(error));
            
            // Mark video as failed in database
            if let Some(vid) = shared.get(state_keys::VIDEO_ID).and_then(|v| v.as_str()) {
                if let Ok(video_id) = uuid::Uuid::parse_str(vid) {
                    let _ = db::mark_video_failed(&self.db_pool, video_id, "thumbnail", error).await;
                }
            }
            
            return Some("error".to_string());
        }

        if let Some(path) = exec_res.get("thumbnail_path") {
            shared.insert(state_keys::THUMBNAIL_PATH.to_string(), path.clone());
            info!("Thumbnail: Generated successfully");
            
            // Persist thumbnail_path to database
            if let Some(vid) = exec_res.get("video_id").and_then(|v| v.as_str()) {
                if let Ok(video_id) = uuid::Uuid::parse_str(vid) {
                    if let Some(path_str) = path.as_str() {
                        if let Err(e) = db::update_video_text_field(
                            &self.db_pool,
                            video_id,
                            "thumbnail_path",
                            path_str,
                        ).await {
                            error!("Failed to persist thumbnail_path to database: {}", e);
                        }
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
