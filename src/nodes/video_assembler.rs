//! Video Assembler Node
//!
//! Combines audio and visual assets into the final video using MoviePy via Python bridge.

use async_trait::async_trait;
use orichalcum::{AsyncNodeLogic, NodeValue};
use serde::{Deserialize, Serialize};
use sqlx::PgPool;
use std::collections::HashMap;
use std::process::Stdio;
use std::sync::Arc;
use tokio::process::Command;
use tracing::{error, info, warn};

use crate::db;
use crate::nodes::{AssetManifest, AudioTiming};
use crate::state_keys;
use crate::storage::S3Client;

/// Configuration for video assembly
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoAssemblerConfig {
    /// Path to the Python bridge script
    pub bridge_script_path: String,
    /// Output resolution
    pub output_width: u32,
    pub output_height: u32,
    /// Output FPS
    pub output_fps: u32,
    /// Local temp directory for processing
    pub temp_dir: String,
}

impl Default for VideoAssemblerConfig {
    fn default() -> Self {
        Self {
            bridge_script_path: "src/bridge/moviepy_bridge.py".to_string(),
            output_width: 1920,
            output_height: 1080,
            output_fps: 30,
            temp_dir: "/tmp/animus".to_string(),
        }
    }
}

/// Input to the MoviePy bridge
#[derive(Debug, Serialize)]
struct BridgeInput {
    video_id: String,
    audio_path: String,
    asset_manifest: AssetManifest,
    audio_timing: AudioTiming,
    output_path: String,
    mode: String, // "horizontal" or "short"
    config: BridgeConfig,
}

#[derive(Debug, Serialize)]
struct BridgeConfig {
    width: u32,
    height: u32,
    fps: u32,
}

/// Output from the MoviePy bridge
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct BridgeOutput {
    success: bool,
    output_path: Option<String>,
    duration_seconds: Option<f64>,
    error: Option<String>,
}

/// The video assembler node logic
#[derive(Clone)]
pub struct VideoAssemblerLogic {
    pub config: VideoAssemblerConfig,
    pub s3_client: Arc<S3Client>,
    pub db_pool: Arc<PgPool>,
}

impl VideoAssemblerLogic {
    pub fn new(config: VideoAssemblerConfig, s3_client: Arc<S3Client>, db_pool: Arc<PgPool>) -> Self {
        Self { config, s3_client, db_pool }
    }

    /// Download a file from S3 to local temp
    async fn download_from_s3(&self, s3_path: &str, local_path: &str) -> Result<(), String> {
        self.s3_client.download_to_file(s3_path, local_path).await
    }

    /// Upload a local file to S3
    async fn upload_to_s3(&self, local_path: &str, s3_path: &str) -> Result<(), String> {
        self.s3_client.upload_file(local_path, s3_path, "video/mp4").await
    }
}

#[async_trait]
impl AsyncNodeLogic for VideoAssemblerLogic {
    async fn prep(
        &self,
        _params: &HashMap<String, NodeValue>,
        shared: &HashMap<String, NodeValue>,
    ) -> NodeValue {
        let audio_path = shared.get(state_keys::AUDIO_PATH).cloned().unwrap_or(serde_json::json!(null));
        let audio_timing = shared.get(state_keys::AUDIO_TIMING).cloned().unwrap_or(serde_json::json!(null));
        let asset_manifest = shared.get(state_keys::ASSET_MANIFEST).cloned().unwrap_or(serde_json::json!(null));
        let video_id = shared.get(state_keys::VIDEO_ID).cloned().unwrap_or(serde_json::json!(null));

        serde_json::json!({
            "audio_path": audio_path,
            "audio_timing": audio_timing,
            "asset_manifest": asset_manifest,
            "video_id": video_id
        })
    }

    async fn exec(&self, input: NodeValue) -> NodeValue {
        info!("VideoAssembler: Initializing...");
        let video_id = input.get("video_id")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
            .to_string();

        let audio_path: String = match input.get("audio_path").and_then(|v| v.as_str()) {
            Some(p) => p.to_string(),
            None => return serde_json::json!({ "error": "No audio path provided" }),
        };

        let audio_timing: AudioTiming = match input.get("audio_timing")
            .and_then(|v| serde_json::from_value(v.clone()).ok()) {
            Some(t) => t,
            None => return serde_json::json!({ "error": "No audio timing provided" }),
        };

        let asset_manifest: AssetManifest = match input.get("asset_manifest")
            .and_then(|v| serde_json::from_value(v.clone()).ok()) {
            Some(m) => m,
            None => return serde_json::json!({ "error": "No asset manifest provided" }),
        };

        info!("VideoAssembler: Assembling video {}", video_id);

        // Create temp directory
        let temp_dir = format!("{}/{}", self.config.temp_dir, video_id);
        if let Err(e) = tokio::fs::create_dir_all(&temp_dir).await {
            return serde_json::json!({ "error": format!("Failed to create temp dir: {}", e) });
        }

        // Download audio from S3
        let local_audio_path = format!("{}/audio.mp3", temp_dir);
        if let Err(e) = self.download_from_s3(&audio_path, &local_audio_path).await {
            return serde_json::json!({ "error": format!("Failed to download audio: {}", e) });
        }

        // Download all video assets
        let mut local_manifest = asset_manifest.clone();
        for section_assets in &mut local_manifest.section_assets {
            for clip in &mut section_assets.video_clips {
                let local_path = format!("{}/{}", temp_dir, clip.path.replace('/', "_"));
                if let Err(e) = self.download_from_s3(&clip.path, &local_path).await {
                    warn!("Failed to download clip {}: {}", clip.path, e);
                    continue;
                }
                clip.path = local_path;
            }
        }

        // Prepare bridge input
        let output_path = format!("{}/output.mp4", temp_dir);
        let bridge_input = BridgeInput {
            video_id: video_id.clone(),
            audio_path: local_audio_path.clone(),
            asset_manifest: local_manifest.clone(),
            audio_timing: audio_timing.clone(),
            output_path: output_path.clone(),
            mode: "horizontal".to_string(),
            config: BridgeConfig {
                width: self.config.output_width,
                height: self.config.output_height,
                fps: self.config.output_fps,
            },
        };

        // Call Python bridge for main video
        let main_result = self.run_bridge(&bridge_input).await?;
        
        // Upload final video to S3
        let s3_video_path = format!("videos/{}/{}.mp4", video_id, video_id);
        if let Err(e) = self.upload_to_s3(&output_path, &s3_video_path).await {
            return serde_json::json!({ "error": format!("Failed to upload video: {}", e) });
        }

        // --- SHORTS RENDERING ---
        let mut shorts_path = None;
        if let Some(script) = input.get("script").and_then(|v| serde_json::from_value::<crate::nodes::Script>(v.clone()).ok()) {
            if let Some(idx) = script.shorts_candidate_index {
                info!("VideoAssembler: Rendering vertical Short from section {}...", idx);
                let shorts_output_path = format!("{}/short.mp4", temp_dir);
                
                // Construct a manifest for just the Short
                let mut short_manifest = local_manifest.clone();
                let section_assets = short_manifest.section_assets.get(idx).cloned().unwrap_or(SectionAssets {
                    section_title: "Short".to_string(),
                    video_clips: vec![],
                    images: vec![],
                });
                short_manifest.section_assets = vec![section_assets];

                let short_bridge_input = BridgeInput {
                    video_id: format!("{}_short", video_id),
                    audio_path: local_audio_path,
                    asset_manifest: short_manifest,
                    audio_timing, // Ideally we'd slice this too
                    output_path: shorts_output_path.clone(),
                    mode: "short".to_string(),
                    config: BridgeConfig {
                        width: 1080,
                        height: 1920,
                        fps: self.config.output_fps,
                    },
                };

                if let Ok(_) = self.run_bridge(&short_bridge_input).await {
                    let s3_short_path = format!("videos/{}/short.mp4", video_id);
                    if let Ok(_) = self.upload_to_s3(&shorts_output_path, &s3_short_path).await {
                        shorts_path = Some(s3_short_path);
                        info!("VideoAssembler: Vertical Short uploaded successfully");
                    }
                }
            }
        }

        // Clean up temp directory
        let _ = tokio::fs::remove_dir_all(&temp_dir).await;

        serde_json::json!({
            "success": true,
            "video_path": s3_video_path,
            "shorts_path": shorts_path,
            "duration_seconds": main_result.duration_seconds,
            "video_id": video_id
        })
    }

    /// Run the Python bridge and parse output
    async fn run_bridge(&self, input: &BridgeInput) -> Result<BridgeOutput, String> {
        let input_json = serde_json::to_string(input).map_err(|e| e.to_string())?;
        
        let mut child = Command::new("python3")
            .arg(&self.config.bridge_script_path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|e| format!("Failed to spawn Python: {}", e))?;

        if let Some(mut stdin) = child.stdin.take() {
            use tokio::io::AsyncWriteExt;
            stdin.write_all(input_json.as_bytes()).await.map_err(|e| e.to_string())?;
            std::mem::drop(stdin);
        }

        let output = child.wait_with_output().await.map_err(|e| e.to_string())?;
        let stdout = String::from_utf8_lossy(&output.stdout);
        
        if !output.status.success() {
            return Err(format!("Python bridge failed: {}", stdout));
        }

        serde_json::from_str(&stdout).map_err(|e| format!("Failed to parse output: {}", e))
    }

    async fn post(
        &self,
        shared: &mut HashMap<String, NodeValue>,
        _prep_res: NodeValue,
        exec_res: NodeValue,
    ) -> Option<String> {
        if let Some(error) = exec_res.get("error").and_then(|v| v.as_str()) {
            error!("VideoAssembler failed: {}", error);
            shared.insert(state_keys::ERROR.to_string(), serde_json::json!(error));
            
            // Mark video as failed in database
            if let Some(vid) = shared.get(state_keys::VIDEO_ID).and_then(|v| v.as_str()) {
                if let Ok(video_id) = uuid::Uuid::parse_str(vid) {
                    let _ = db::mark_video_failed(&self.db_pool, video_id, "video_assembler", error).await;
                }
            }
            
            return Some("error".to_string());
        }

        if let Some(video_path) = exec_res.get("video_path") {
            shared.insert(state_keys::VIDEO_PATH.to_string(), video_path.clone());
            info!("VideoAssembler: Video assembled successfully");
            
            if let Some(shorts_path) = exec_res.get("shorts_path") {
                shared.insert("shorts_path".to_string(), shorts_path.clone());
            }

            // Persist video_path and shorts_path to database
            if let Some(vid) = exec_res.get("video_id").and_then(|v| v.as_str()) {
                if let Ok(video_id) = uuid::Uuid::parse_str(vid) {
                    if let Some(path_str) = video_path.as_str() {
                        let _ = db::update_video_text_field(&self.db_pool, video_id, "video_path", path_str).await;
                    }
                    if let Some(shorts_str) = exec_res.get("shorts_path").and_then(|v| v.as_str()) {
                        // We need a helper for shorts_path or use a custom query
                        let _ = sqlx::query("UPDATE videos SET shorts_path = $1 WHERE id = $2")
                            .bind(shorts_str)
                            .bind(video_id)
                            .execute(&*self.db_pool).await;
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
