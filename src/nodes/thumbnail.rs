//! Thumbnail Generator Node
//!
//! Creates eye-catching thumbnails using Leonardo AI with gradient fallback.

use async_trait::async_trait;
use orichalcum::llm::{Client, Enabled, Providers};
use orichalcum::{AsyncNodeLogic, NodeValue};
use serde::{Deserialize, Serialize};
use sqlx::PgPool;
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{error, info, warn};

use ab_glyph::{Font, FontArc, PxScale};
use image::{Rgb, RgbImage, Rgba, RgbaImage};
use imageproc::drawing::{draw_text_mut, text_size};
use std::io::Cursor;

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
    /// Prompt prefix for Leonardo
    pub prompt_prefix: String,
    /// Show watermark on fallback thumbnails
    pub show_watermark: bool,
}

impl Default for ThumbnailConfig {
    fn default() -> Self {
        Self {
            template_dir: "templates/thumbnails".to_string(),
            width: 1280,
            height: 720,
            prompt_prefix: "Professional YouTube thumbnail".to_string(),
            show_watermark: false,
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

    /// Generate a thumbnail using Leonardo AI
    async fn generate_leonardo_thumbnail(
        &self,
        title: &str,
        _video_id: &str,
    ) -> Result<Vec<u8>, String> {
        info!("Thumbnail: Calling Leonardo AI for '{}'", title);

        let api_key = std::env::var("LEONARDO_API_KEY")
            .map_err(|_| "LEONARDO_API_KEY not found in environment".to_string())?;

        let client = reqwest::Client::new();

        let prompt = format!(
            "Professional YouTube thumbnail, bold white text matching the title '{}', \
            ancient design, philosophical, secret knowledge, stoic, gnostic \
            high contrast, eye-catching, 16:9 aspect ratio, \
            cinematic lighting, premium quality",
            title
        );

        // Step 1: Generate the image
        let request_body = serde_json::json!({
            "prompt": prompt,
            "modelId": "b24e16ff-06e3-43eb-8d33-4416c2d75876", // Phoenix model
            "width": self.config.width,
            "height": self.config.height,
            "num_images": 1,
            "alchemy": true,
            "promptMagic": true,
            "promptMagicVersion": "v3",
            "contrast": 3.5,
        });

        let generation_response = client
            .post("https://cloud.leonardo.ai/api/rest/v1/generations")
            .header("Authorization", format!("Bearer {}", api_key))
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await
            .map_err(|e| format!("Failed to call Leonardo API: {}", e))?;

        if !generation_response.status().is_success() {
            let error_text = generation_response.text().await.unwrap_or_default();
            return Err(format!("Leonardo API error: {}", error_text));
        }

        let generation_data: serde_json::Value = generation_response
            .json()
            .await
            .map_err(|e| format!("Failed to parse generation response: {}", e))?;

        let generation_id = generation_data["sdGenerationJob"]["generationId"]
            .as_str()
            .ok_or("No generation ID in response")?;

        info!("Thumbnail: Generation ID: {}", generation_id);

        // Step 2: Poll until complete
        let image_url = self
            .poll_leonardo_completion(&client, &api_key, generation_id)
            .await?;

        info!("Thumbnail: Image ready at {}", image_url);

        // Step 3: Download the generated image
        let image_response = client
            .get(&image_url)
            .send()
            .await
            .map_err(|e| format!("Failed to download image: {}", e))?;

        if !image_response.status().is_success() {
            return Err(format!(
                "Failed to download image: {}",
                image_response.status()
            ));
        }

        let image_bytes = image_response
            .bytes()
            .await
            .map_err(|e| format!("Failed to read image bytes: {}", e))?
            .to_vec();

        info!(
            "Thumbnail: Generated {} bytes for '{}'",
            image_bytes.len(),
            title
        );

        Ok(image_bytes)
    }

    /// Poll Leonardo API until generation is complete
    async fn poll_leonardo_completion(
        &self,
        client: &reqwest::Client,
        api_key: &str,
        generation_id: &str,
    ) -> Result<String, String> {
        use tokio::time::{sleep, Duration};

        let max_attempts = 30; // 30 seconds max
        let poll_interval = Duration::from_secs(1);

        for attempt in 1..=max_attempts {
            let response = client
                .get(format!(
                    "https://cloud.leonardo.ai/api/rest/v1/generations/{}",
                    generation_id
                ))
                .header("Authorization", format!("Bearer {}", api_key))
                .send()
                .await
                .map_err(|e| format!("Failed to poll generation status: {}", e))?;

            let data: serde_json::Value = response
                .json()
                .await
                .map_err(|e| format!("Failed to parse status response: {}", e))?;

            let status = data["generations_by_pk"]["status"]
                .as_str()
                .ok_or("No status in response")?;

            match status {
                "COMPLETE" => {
                    let image_url = data["generations_by_pk"]["generated_images"][0]["url"]
                        .as_str()
                        .ok_or("No image URL in response")?;
                    return Ok(image_url.to_string());
                }
                "FAILED" => {
                    return Err("Leonardo generation failed".to_string());
                }
                "PENDING" => {
                    info!(
                        "Thumbnail: Generation pending (attempt {}/{})",
                        attempt, max_attempts
                    );
                    sleep(poll_interval).await;
                }
                _ => {
                    warn!("Thumbnail: Unknown status '{}', continuing to poll", status);
                    sleep(poll_interval).await;
                }
            }
        }

        Err("Leonardo generation timed out".to_string())
    }

    /// Fallback: Generate a simple gradient thumbnail with text overlay
    async fn generate_simple_thumbnail(
        &self,
        title: &str,
        video_id: &str,
    ) -> Result<Vec<u8>, String> {
        warn!("Thumbnail: Using fallback gradient thumbnail");

        let width = self.config.width;
        let height = self.config.height;

        // Create base image with alpha channel
        let mut img = RgbaImage::new(width, height);

        // Create gradient (dark blue to deep purple)
        for y in 0..height {
            for x in 0..width {
                let x_ratio = x as f32 / width as f32;
                let y_ratio = y as f32 / height as f32;

                let r = (25.0 + x_ratio * 45.0 + y_ratio * 20.0) as u8;
                let g = (15.0 + y_ratio * 40.0) as u8;
                let b = (70.0 + x_ratio * 80.0 - y_ratio * 20.0) as u8;

                img.put_pixel(x, y, Rgba([r, g, b, 255]));
            }
        }

        // Add vignette effect
        let center_x = width as f32 / 2.0;
        let center_y = height as f32 / 2.0;
        let max_dist = ((center_x.powi(2) + center_y.powi(2)) as f32).sqrt();

        for y in 0..height {
            for x in 0..width {
                let dx = x as f32 - center_x;
                let dy = y as f32 - center_y;
                let dist = (dx * dx + dy * dy).sqrt();
                let vignette = 1.0 - (dist / max_dist * 0.4);

                let pixel = img.get_pixel_mut(x, y);
                pixel[0] = (pixel[0] as f32 * vignette) as u8;
                pixel[1] = (pixel[1] as f32 * vignette) as u8;
                pixel[2] = (pixel[2] as f32 * vignette) as u8;
            }
        }

        // Load embedded font
        let font_data = include_bytes!("../assets/fonts/Roboto-Bold.ttf");
        let font = FontArc::try_from_slice(font_data as &[u8])
            .map_err(|e| format!("Failed to load font: {}", e))?;

        // Text settings
        let max_width = (width as f32 * 0.85) as u32;
        let font_size = (height as f32 * 0.12) as f32;
        let scale = PxScale::from(font_size);

        let wrapped_lines = wrap_text(title, &font, scale, max_width);
        let line_height = (font_size * 1.3) as i32;

        // Center text vertically
        let total_text_height = wrapped_lines.len() as i32 * line_height;
        let mut y_offset = (height as i32 - total_text_height) / 2;

        // Draw text with shadow
        let text_color = Rgba([255u8, 255u8, 255u8, 255u8]);
        let shadow_color = Rgba([0u8, 0u8, 0u8, 180u8]);

        for line in wrapped_lines {
            let (text_width, _) = text_size(scale, &font, &line);
            let x_offset = ((width as i32 - text_width as i32) / 2).max(20);

            // Shadow
            draw_text_mut(
                &mut img,
                shadow_color,
                x_offset + 3,
                y_offset + 3,
                scale,
                &font,
                &line,
            );

            // Main text
            draw_text_mut(&mut img, text_color, x_offset, y_offset, scale, &font, &line);

            y_offset += line_height;
        }

        // Add decorative corner
        add_decorative_corner(&mut img, width, height);

        // Optional watermark
        if self.config.show_watermark {
            let small_scale = PxScale::from(font_size * 0.3);
            let watermark = format!("ID: {}", video_id);
            draw_text_mut(
                &mut img,
                Rgba([255u8, 255u8, 255u8, 150u8]),
                10,
                height as i32 - 30,
                small_scale,
                &font,
                &watermark,
            );
        }

        // Convert to RGB and encode
        let rgb_img: RgbImage = RgbImage::from_fn(width, height, |x, y| {
            let pixel = img.get_pixel(x, y);
            Rgb([pixel[0], pixel[1], pixel[2]])
        });

        let mut buffer = Cursor::new(Vec::new());
        rgb_img
            .write_to(&mut buffer, image::ImageFormat::Png)
            .map_err(|e| format!("Failed to encode thumbnail: {}", e))?;

        Ok(buffer.into_inner())
    }
}

// Helper functions (moved outside impl block)
fn wrap_text(text: &str, font: &impl Font, scale: PxScale, max_width: u32) -> Vec<String> {
    let words: Vec<&str> = text.split_whitespace().collect();
    let mut lines = Vec::new();
    let mut current_line = String::new();

    for word in words {
        let test_line = if current_line.is_empty() {
            word.to_string()
        } else {
            format!("{} {}", current_line, word)
        };

        let (width, _) = text_size(scale, font, &test_line);

        if width <= max_width {
            current_line = test_line;
        } else {
            if !current_line.is_empty() {
                lines.push(current_line);
            }
            current_line = word.to_string();
        }
    }

    if !current_line.is_empty() {
        lines.push(current_line);
    }

    lines.truncate(4);
    lines
}

fn add_decorative_corner(img: &mut RgbaImage, width: u32, height: u32) {
    let accent_color = Rgba([255u8, 200u8, 100u8, 120u8]);
    let corner_size = (width.min(height) / 8) as i32;

    for i in 0..corner_size {
        for j in 0..3 {
            if (i as u32) < width && (j as u32) < height {
                img.put_pixel(i as u32, j as u32, accent_color);
            }
        }
        for j in 0..3 {
            if (j as u32) < width && (i as u32) < height {
                img.put_pixel(j as u32, i as u32, accent_color);
            }
        }
    }
}

#[async_trait]
impl AsyncNodeLogic for ThumbnailLogic {
    async fn prep(
        &self,
        _params: &HashMap<String, NodeValue>,
        shared: &HashMap<String, NodeValue>,
    ) -> NodeValue {
        let topic_brief = shared
            .get(state_keys::TOPIC_BRIEF)
            .cloned()
            .unwrap_or(serde_json::json!(null));
        let video_id = shared
            .get(state_keys::VIDEO_ID)
            .cloned()
            .unwrap_or(serde_json::json!(null));

        serde_json::json!({
            "topic_brief": topic_brief,
            "video_id": video_id
        })
    }

    async fn exec(&self, input: NodeValue) -> NodeValue {
        let video_id = input
            .get("video_id")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
            .to_string();

        let topic_brief: Option<TopicBrief> = input
            .get("topic_brief")
            .and_then(|v| serde_json::from_value(v.clone()).ok());

        let title = topic_brief
            .as_ref()
            .map(|t| t.topic.clone())
            .unwrap_or_else(|| "Untitled".to_string());

        info!("Thumbnail: Generating thumbnail for '{}'", title);

        // Try Leonardo first, fallback to simple gradient
        let thumbnail_bytes = match self.generate_leonardo_thumbnail(&title, &video_id).await {
            Ok(bytes) => bytes,
            Err(e) => {
                error!("Thumbnail: Leonardo failed, falling back: {}", e);
                match self.generate_simple_thumbnail(&title, &video_id).await {
                    Ok(bytes) => bytes,
                    Err(e) => {
                        return serde_json::json!({
                            "error": format!("Thumbnail generation failed: {}", e)
                        })
                    }
                }
            }
        };

        // Upload to S3
        let thumbnail_path = format!("thumbnails/{}/thumbnail.png", video_id);
        if let Err(e) = self
            .s3_client
            .upload_bytes(&thumbnail_path, thumbnail_bytes, "image/png")
            .await
        {
            return serde_json::json!({
                "error": format!("Failed to upload thumbnail: {}", e)
            });
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

            if let Some(vid) = shared.get(state_keys::VIDEO_ID).and_then(|v| v.as_str()) {
                if let Ok(video_id) = uuid::Uuid::parse_str(vid) {
                    let _ =
                        db::mark_video_failed(&self.db_pool, video_id, "thumbnail", error).await;
                }
            }

            return Some("error".to_string());
        }

        if let Some(path) = exec_res.get("thumbnail_path") {
            shared.insert(state_keys::THUMBNAIL_PATH.to_string(), path.clone());
            info!("Thumbnail: Generated successfully");

            if let Some(vid) = exec_res.get("video_id").and_then(|v| v.as_str()) {
                if let Ok(video_id) = uuid::Uuid::parse_str(vid) {
                    if let Some(path_str) = path.as_str() {
                        if let Err(e) = db::update_video_text_field(
                            &self.db_pool,
                            video_id,
                            "thumbnail_path",
                            path_str,
                        )
                        .await
                        {
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
