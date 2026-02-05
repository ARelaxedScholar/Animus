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
            audio_path: local_audio_path,
            asset_manifest: local_manifest,
            audio_timing,
            output_path: output_path.clone(),
            config: BridgeConfig {
                width: self.config.output_width,
                height: self.config.output_height,
                fps: self.config.output_fps,
            },
        };

        // Call Python bridge
        let input_json = match serde_json::to_string(&bridge_input) {
            Ok(j) => j,
            Err(e) => return serde_json::json!({ "error": format!("Failed to serialize bridge input: {}", e) }),
        };

        let mut child = match Command::new("python3")
            .arg(&self.config.bridge_script_path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn() {
            Ok(c) => c,
            Err(e) => return serde_json::json!({ "error": format!("Failed to spawn Python: {}", e) }),
        };

        // Write input to stdin
        if let Some(mut stdin) = child.stdin.take() {
            use tokio::io::AsyncWriteExt;
            if let Err(e) = stdin.write_all(input_json.as_bytes()).await {
                return serde_json::json!({ "error": format!("Failed to write to Python: {}", e) });
            }
        }

        // Wait for completion
        let output = match child.wait_with_output().await {
            Ok(o) => o,
            Err(e) => return serde_json::json!({ "error": format!("Python process failed: {}", e) }),
        };

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let stdout = String::from_utf8_lossy(&output.stdout);
            let exit_code = output.status.code().map(|c| c.to_string()).unwrap_or_else(|| "unknown".to_string());
            
            error!("Python bridge failed. Exit code: {}, stdout: {}, stderr: {}", exit_code, stdout, stderr);
            
            // Try to parse stdout as JSON error response (the bridge writes errors to stdout as JSON)
            if let Ok(bridge_output) = serde_json::from_str::<BridgeOutput>(&stdout) {
                if let Some(err) = bridge_output.error {
                    return serde_json::json!({ "error": err });
                }
            }
            
            return serde_json::json!({ 
                "error": format!(
                    "Python process failed (exit {}). stderr: {}. stdout: {}", 
                    exit_code, 
                    if stderr.is_empty() { "<empty>" } else { stderr.as_ref() },
                    if stdout.is_empty() { "<empty>" } else { stdout.as_ref() }
                ) 
            });
        }

        // Parse output
        let stdout = String::from_utf8_lossy(&output.stdout);
        let bridge_output: BridgeOutput = match serde_json::from_str(&stdout) {
            Ok(o) => o,
            Err(e) => return serde_json::json!({ 
                "error": format!("Failed to parse Python output: {}. Output: {}", e, stdout) 
            }),
        };

        if !bridge_output.success {
            return serde_json::json!({ 
                "error": bridge_output.error.unwrap_or_else(|| "Unknown error".to_string()) 
            });
        }

        // Upload final video to S3
        let s3_video_path = format!("videos/{}/{}.mp4", video_id, video_id);
        if let Err(e) = self.upload_to_s3(&output_path, &s3_video_path).await {
            return serde_json::json!({ "error": format!("Failed to upload video: {}", e) });
        }

        // Clean up temp directory
        let _ = tokio::fs::remove_dir_all(&temp_dir).await;

        serde_json::json!({
            "success": true,
            "video_path": s3_video_path,
            "duration_seconds": bridge_output.duration_seconds,
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
            
            // Persist video_path to database
            if let Some(vid) = exec_res.get("video_id").and_then(|v| v.as_str()) {
                if let Ok(video_id) = uuid::Uuid::parse_str(vid) {
                    if let Some(path_str) = video_path.as_str() {
                        if let Err(e) = db::update_video_text_field(
                            &self.db_pool,
                            video_id,
                            "video_path",
                            path_str,
                        ).await {
                            error!("Failed to persist video_path to database: {}", e);
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
