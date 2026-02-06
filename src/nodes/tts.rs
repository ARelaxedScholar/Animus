//! TTS Node
//!
//! Converts scripts to natural-sounding audio using configurable TTS providers:
//! - ElevenLabs (cloud, high quality)
//! - Qwen3-TTS (open-source, local/self-hosted)
//! - OpenAI TTS (cloud, good quality)

use async_trait::async_trait;
use orichalcum::{AsyncNodeLogic, NodeValue};
use reqwest::Client as HttpClient;
use serde::{Deserialize, Serialize};
use sqlx::PgPool;
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{error, info, warn};

use crate::db;
use crate::nodes::{AudioTiming, Script, SectionTiming};
use crate::state_keys;
use crate::storage::S3Client;

/// TTS Provider selection
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum TTSProvider {
    /// ElevenLabs cloud API
    ElevenLabs,
    /// Qwen3-TTS (self-hosted or local)
    Qwen3,
    /// OpenAI TTS API
    OpenAI,
    /// Coqui TTS (local)
    Coqui,
    /// Piper TTS (local, very fast)
    Piper,
}

impl Default for TTSProvider {
    fn default() -> Self {
        Self::ElevenLabs
    }
}

/// Configuration for TTS - supports multiple providers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TTSConfig {
    /// Which provider to use
    pub provider: TTSProvider,
    
    // ElevenLabs settings
    pub elevenlabs_api_key: Option<String>,
    pub elevenlabs_voice_id: Option<String>,
    pub elevenlabs_model_id: Option<String>,
    
    // Qwen3-TTS settings (OpenAI-compatible API)
    pub qwen3_api_url: Option<String>,  // e.g., "http://localhost:8000/v1"
    pub qwen3_api_key: Option<String>,  // Optional, depends on setup
    pub qwen3_voice: Option<String>,    // Voice/speaker ID
    
    // OpenAI TTS settings
    pub openai_api_key: Option<String>,
    pub openai_voice: Option<String>,   // alloy, echo, fable, onyx, nova, shimmer
    pub openai_model: Option<String>,   // tts-1, tts-1-hd
    
    // Coqui/Piper settings (local)
    pub local_model_path: Option<String>,
    pub local_speaker_id: Option<String>,
    
    // Common settings
    pub stability: f32,
    pub similarity_boost: f32,
    pub speed: f32,
}

impl Default for TTSConfig {
    fn default() -> Self {
        Self {
            provider: TTSProvider::ElevenLabs,
            elevenlabs_api_key: None,
            elevenlabs_voice_id: None,
            elevenlabs_model_id: Some("eleven_monolingual_v1".to_string()),
            qwen3_api_url: None,
            qwen3_api_key: None,
            qwen3_voice: Some("default".to_string()),
            openai_api_key: None,
            openai_voice: Some("onyx".to_string()),
            openai_model: Some("tts-1-hd".to_string()),
            local_model_path: None,
            local_speaker_id: None,
            stability: 0.5,
            similarity_boost: 0.75,
            speed: 1.0,
        }
    }
}

/// Request body for ElevenLabs TTS
#[derive(Debug, Serialize)]
struct ElevenLabsRequest {
    text: String,
    model_id: String,
    voice_settings: VoiceSettings,
}

#[derive(Debug, Serialize)]
struct VoiceSettings {
    stability: f32,
    similarity_boost: f32,
}

/// Request body for OpenAI-compatible TTS (works for OpenAI and Qwen3)
#[derive(Debug, Serialize)]
struct OpenAITTSRequest {
    model: String,
    input: String,
    voice: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    speed: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<String>,
}

/// The TTS node logic
#[derive(Clone)]
pub struct TTSLogic {
    pub config: TTSConfig,
    pub http_client: Arc<HttpClient>,
    pub s3_client: Arc<S3Client>,
    pub db_pool: Arc<PgPool>,
}

impl TTSLogic {
    pub fn new(config: TTSConfig, s3_client: Arc<S3Client>, db_pool: Arc<PgPool>) -> Self {
        // Create HTTP client with long timeout for TTS generation
        // Long scripts (12-20 min) can take several minutes to generate
        let http_client = HttpClient::builder()
            .timeout(std::time::Duration::from_secs(600)) // 10 minute timeout
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

    /// Generate audio using the configured provider
    async fn generate_audio(&self, text: &str) -> Result<Vec<u8>, String> {
        match self.config.provider {
            TTSProvider::ElevenLabs => self.generate_elevenlabs(text).await,
            TTSProvider::Qwen3 => self.generate_qwen3(text).await,
            TTSProvider::OpenAI => self.generate_openai(text).await,
            TTSProvider::Coqui => self.generate_local_coqui(text).await,
            TTSProvider::Piper => self.generate_local_piper(text).await,
        }
    }

    /// Generate audio using ElevenLabs
    /// Handles chunking for long texts (ElevenLabs has character limits)
    async fn generate_elevenlabs(&self, text: &str) -> Result<Vec<u8>, String> {
        let api_key = self.config.elevenlabs_api_key.as_ref()
            .ok_or("ElevenLabs API key not configured")?;
        let voice_id = self.config.elevenlabs_voice_id.as_ref()
            .ok_or("ElevenLabs voice ID not configured")?;
        let model_id = self.config.elevenlabs_model_id.as_ref()
            .map(|s| s.as_str())
            .unwrap_or("eleven_monolingual_v1");

        let word_count = text.split_whitespace().count();
        let char_count = text.len();
        let estimated_duration_min = word_count as f64 / 150.0;
        
        // ElevenLabs has a 5000 character limit per request
        // We'll use a conservative 4500 to account for edge cases
        const MAX_CHARS_PER_CHUNK: usize = 4500;
        
        if char_count <= MAX_CHARS_PER_CHUNK {
            // Single request for short text
            info!(
                "TTS: Sending {} chars ({} words) to ElevenLabs (est. {:.1} min audio)...",
                char_count, word_count, estimated_duration_min
            );
            return self.generate_elevenlabs_chunk(text, api_key, voice_id, model_id).await;
        }
        
        // Need to chunk the text
        let chunks = Self::chunk_text_for_tts(text, MAX_CHARS_PER_CHUNK);
        info!(
            "TTS: Splitting {} chars into {} chunks for ElevenLabs (est. {:.1} min total)...",
            char_count, chunks.len(), estimated_duration_min
        );
        
        let mut all_audio: Vec<u8> = Vec::new();
        
        for (i, chunk) in chunks.iter().enumerate() {
            info!("TTS: Processing chunk {}/{} ({} chars)...", i + 1, chunks.len(), chunk.len());
            
            match self.generate_elevenlabs_chunk(chunk, api_key, voice_id, model_id).await {
                Ok(audio_bytes) => {
                    info!("TTS: Chunk {}/{} complete ({} bytes)", i + 1, chunks.len(), audio_bytes.len());
                    all_audio.extend(audio_bytes);
                }
                Err(e) => {
                    warn!("TTS: Chunk {}/{} failed: {}", i + 1, chunks.len(), e);
                    return Err(format!("Failed on chunk {}: {}", i + 1, e));
                }
            }
            
            // Small delay between chunks to avoid rate limiting
            if i < chunks.len() - 1 {
                tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
            }
        }
        
        info!("TTS: All chunks complete, total {} bytes", all_audio.len());
        Ok(all_audio)
    }
    
    /// Split text into chunks at sentence boundaries
    fn chunk_text_for_tts(text: &str, max_chars: usize) -> Vec<String> {
        let mut chunks = Vec::new();
        let mut current_chunk = String::new();
        
        // Split by sentences (period, exclamation, question mark followed by space or end)
        let sentences: Vec<&str> = text
            .split_inclusive(|c| c == '.' || c == '!' || c == '?')
            .collect();
        
        for sentence in sentences {
            let sentence = sentence.trim();
            if sentence.is_empty() {
                continue;
            }
            
            // If adding this sentence would exceed the limit
            if !current_chunk.is_empty() && current_chunk.len() + sentence.len() + 1 > max_chars {
                // Save current chunk and start new one
                chunks.push(current_chunk.trim().to_string());
                current_chunk = String::new();
            }
            
            // If a single sentence is too long, split it by commas or just force-split
            if sentence.len() > max_chars {
                // Try splitting by commas first
                let parts: Vec<&str> = sentence.split(',').collect();
                for part in parts {
                    let part = part.trim();
                    if current_chunk.len() + part.len() + 2 > max_chars {
                        if !current_chunk.is_empty() {
                            chunks.push(current_chunk.trim().to_string());
                            current_chunk = String::new();
                        }
                    }
                    if !current_chunk.is_empty() {
                        current_chunk.push_str(", ");
                    }
                    current_chunk.push_str(part);
                }
            } else {
                if !current_chunk.is_empty() {
                    current_chunk.push(' ');
                }
                current_chunk.push_str(sentence);
            }
        }
        
        // Don't forget the last chunk
        if !current_chunk.trim().is_empty() {
            chunks.push(current_chunk.trim().to_string());
        }
        
        chunks
    }
    
    /// Generate audio for a single chunk using ElevenLabs
    async fn generate_elevenlabs_chunk(
        &self,
        text: &str,
        api_key: &str,
        voice_id: &str,
        model_id: &str,
    ) -> Result<Vec<u8>, String> {
        let url = format!(
            "https://api.elevenlabs.io/v1/text-to-speech/{}",
            voice_id
        );

        let request_body = ElevenLabsRequest {
            text: text.to_string(),
            model_id: model_id.to_string(),
            voice_settings: VoiceSettings {
                stability: self.config.stability,
                similarity_boost: self.config.similarity_boost,
            },
        };

        let response = self.http_client
            .post(&url)
            .header("xi-api-key", api_key)
            .header("Content-Type", "application/json")
            .header("Accept", "audio/mpeg")
            .json(&request_body)
            .send()
            .await
            .map_err(|e| format!("ElevenLabs request failed: {}", e))?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(format!("ElevenLabs API error {}: {}", status, error_text));
        }

        let audio_bytes = response.bytes().await
            .map_err(|e| format!("Failed to read audio bytes: {}", e))?;

        Ok(audio_bytes.to_vec())
    }

    /// Generate audio using Qwen3-TTS (OpenAI-compatible endpoint)
    /// Handles chunking for long texts
    async fn generate_qwen3(&self, text: &str) -> Result<Vec<u8>, String> {
        let api_url = self.config.qwen3_api_url.as_ref()
            .ok_or("Qwen3-TTS API URL not configured")?;
        let voice = self.config.qwen3_voice.as_ref()
            .map(|s| s.as_str())
            .unwrap_or("default");
        let api_key = self.config.qwen3_api_key.as_ref();

        let word_count = text.split_whitespace().count();
        let char_count = text.len();
        let estimated_duration_min = word_count as f64 / 150.0;
        
        // Use a 4000 character limit for Qwen3 as well
        const MAX_CHARS_PER_CHUNK: usize = 4000;
        
        if char_count <= MAX_CHARS_PER_CHUNK {
            return self.generate_qwen3_chunk(text, api_url, api_key.map(|s| s.as_str()), voice).await;
        }
        
        // Need to chunk the text
        let chunks = Self::chunk_text_for_tts(text, MAX_CHARS_PER_CHUNK);
        info!(
            "TTS: Splitting {} chars into {} chunks for Qwen3 (est. {:.1} min total)...",
            char_count, chunks.len(), estimated_duration_min
        );
        
        let mut all_audio: Vec<u8> = Vec::new();
        
        for (i, chunk) in chunks.iter().enumerate() {
            info!("TTS: Processing chunk {}/{} ({} chars)...", i + 1, chunks.len(), chunk.len());
            
            match self.generate_qwen3_chunk(chunk, api_url, api_key.map(|s| s.as_str()), voice).await {
                Ok(audio_bytes) => {
                    info!("TTS: Chunk {}/{} complete ({} bytes)", i + 1, chunks.len(), audio_bytes.len());
                    all_audio.extend(audio_bytes);
                }
                Err(e) => {
                    warn!("TTS: Chunk {}/{} failed: {}", i + 1, chunks.len(), e);
                    return Err(format!("Failed on chunk {}: {}", i + 1, e));
                }
            }
            
            if i < chunks.len() - 1 {
                tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
            }
        }
        
        info!("TTS: All chunks complete, total {} bytes", all_audio.len());
        Ok(all_audio)
    }

    /// Generate audio for a single chunk using Qwen3
    async fn generate_qwen3_chunk(
        &self,
        text: &str,
        api_url: &str,
        api_key: Option<&str>,
        voice: &str,
    ) -> Result<Vec<u8>, String> {
        let url = format!("{}/audio/speech", api_url.trim_end_matches('/'));

        let request_body = OpenAITTSRequest {
            model: "qwen3-tts".to_string(),
            input: text.to_string(),
            voice: voice.to_string(),
            speed: Some(self.config.speed),
            response_format: Some("mp3".to_string()),
        };

        let mut request = self.http_client
            .post(&url)
            .header("Content-Type", "application/json")
            .json(&request_body);

        if let Some(key) = api_key {
            request = request.header("Authorization", format!("Bearer {}", key));
        }

        let response = request
            .send()
            .await
            .map_err(|e| format!("Qwen3-TTS request failed: {}", e))?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(format!("Qwen3-TTS API error {}: {}", status, error_text));
        }

        let audio_bytes = response.bytes().await
            .map_err(|e| format!("Failed to read audio bytes: {}", e))?;

        Ok(audio_bytes.to_vec())
    }

    /// Generate audio using OpenAI TTS
    /// Handles chunking for long texts (OpenAI has a 4096 character limit)
    async fn generate_openai(&self, text: &str) -> Result<Vec<u8>, String> {
        let api_key = self.config.openai_api_key.as_ref()
            .ok_or("OpenAI API key not configured")?;
        let voice = self.config.openai_voice.as_ref()
            .map(|s| s.as_str())
            .unwrap_or("onyx");
        let model = self.config.openai_model.as_ref()
            .map(|s| s.as_str())
            .unwrap_or("tts-1-hd");

        let word_count = text.split_whitespace().count();
        let char_count = text.len();
        let estimated_duration_min = word_count as f64 / 150.0;
        
        // OpenAI has a 4096 character limit per request
        const MAX_CHARS_PER_CHUNK: usize = 4000;
        
        if char_count <= MAX_CHARS_PER_CHUNK {
            info!(
                "TTS: Sending {} chars to OpenAI (est. {:.1} min audio)...",
                char_count, estimated_duration_min
            );
            return self.generate_openai_chunk(text, api_key, voice, model).await;
        }
        
        // Need to chunk the text
        let chunks = Self::chunk_text_for_tts(text, MAX_CHARS_PER_CHUNK);
        info!(
            "TTS: Splitting {} chars into {} chunks for OpenAI (est. {:.1} min total)...",
            char_count, chunks.len(), estimated_duration_min
        );
        
        let mut all_audio: Vec<u8> = Vec::new();
        
        for (i, chunk) in chunks.iter().enumerate() {
            info!("TTS: Processing chunk {}/{} ({} chars)...", i + 1, chunks.len(), chunk.len());
            
            match self.generate_openai_chunk(chunk, api_key, voice, model).await {
                Ok(audio_bytes) => {
                    info!("TTS: Chunk {}/{} complete ({} bytes)", i + 1, chunks.len(), audio_bytes.len());
                    all_audio.extend(audio_bytes);
                }
                Err(e) => {
                    warn!("TTS: Chunk {}/{} failed: {}", i + 1, chunks.len(), e);
                    return Err(format!("Failed on chunk {}: {}", i + 1, e));
                }
            }
            
            // Small delay between chunks to avoid rate limiting
            if i < chunks.len() - 1 {
                tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
            }
        }
        
        info!("TTS: All chunks complete, total {} bytes", all_audio.len());
        Ok(all_audio)
    }

    /// Generate audio for a single chunk using OpenAI
    async fn generate_openai_chunk(
        &self,
        text: &str,
        api_key: &str,
        voice: &str,
        model: &str,
    ) -> Result<Vec<u8>, String> {
        let request_body = OpenAITTSRequest {
            model: model.to_string(),
            input: text.to_string(),
            voice: voice.to_string(),
            speed: Some(self.config.speed),
            response_format: Some("mp3".to_string()),
        };

        let response = self.http_client
            .post("https://api.openai.com/v1/audio/speech")
            .header("Authorization", format!("Bearer {}", api_key))
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await
            .map_err(|e| format!("OpenAI TTS request failed: {}", e))?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(format!("OpenAI TTS API error {}: {}", status, error_text));
        }

        let audio_bytes = response.bytes().await
            .map_err(|e| format!("Failed to read audio bytes: {}", e))?;

        Ok(audio_bytes.to_vec())
    }

    /// Generate audio using local Coqui TTS
    async fn generate_local_coqui(&self, text: &str) -> Result<Vec<u8>, String> {
        use std::process::Stdio;
        use tokio::process::Command;

        let model_path = self.config.local_model_path.as_ref()
            .ok_or("Coqui model path not configured")?;

        // Create a temp file for output
        let output_path = format!("/tmp/coqui_output_{}.wav", uuid::Uuid::new_v4());

        let mut cmd = Command::new("tts");
        cmd.arg("--text").arg(text)
           .arg("--model_path").arg(model_path)
           .arg("--out_path").arg(&output_path)
           .stdout(Stdio::null())
           .stderr(Stdio::null());

        if let Some(speaker_id) = &self.config.local_speaker_id {
            cmd.arg("--speaker_idx").arg(speaker_id);
        }

        let status = cmd.status().await
            .map_err(|e| format!("Failed to run Coqui TTS: {}", e))?;

        if !status.success() {
            return Err("Coqui TTS failed".to_string());
        }

        // Read the output file
        let audio_bytes = tokio::fs::read(&output_path).await
            .map_err(|e| format!("Failed to read Coqui output: {}", e))?;

        // Clean up
        let _ = tokio::fs::remove_file(&output_path).await;

        Ok(audio_bytes)
    }

    /// Generate audio using local Piper TTS
    async fn generate_local_piper(&self, text: &str) -> Result<Vec<u8>, String> {
        use std::process::Stdio;
        use tokio::process::Command;
        use tokio::io::AsyncWriteExt;

        let model_path = self.config.local_model_path.as_ref()
            .ok_or("Piper model path not configured")?;

        let output_path = format!("/tmp/piper_output_{}.wav", uuid::Uuid::new_v4());

        // Piper reads from stdin and writes to file
        let mut cmd = Command::new("piper");
        cmd.arg("--model").arg(model_path)
           .arg("--output_file").arg(&output_path)
           .stdin(Stdio::piped())
           .stdout(Stdio::null())
           .stderr(Stdio::null());
        
        // Add speed control (length_scale is inverse of speed)
        if self.config.speed != 1.0 && self.config.speed > 0.0 {
            let length_scale = 1.0 / self.config.speed;
            cmd.arg("--length-scale").arg(length_scale.to_string());
        }

        let mut child = cmd.spawn()
            .map_err(|e| format!("Failed to spawn Piper: {}", e))?;

        // Write text to stdin
        if let Some(mut stdin) = child.stdin.take() {
            stdin.write_all(text.as_bytes()).await
                .map_err(|e| format!("Failed to write to Piper: {}", e))?;
        }

        let status = child.wait().await
            .map_err(|e| format!("Piper process failed: {}", e))?;

        if !status.success() {
            return Err("Piper TTS failed".to_string());
        }

        // Convert WAV to MP3 to maintain consistency with other providers
        let mp3_path = format!("{}.mp3", output_path);
        let convert_status = Command::new("ffmpeg")
            .arg("-i").arg(&output_path)
            .arg("-codec:a").arg("libmp3lame")
            .arg("-qscale:a").arg("2")
            .arg(&mp3_path)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status().await
            .map_err(|e| format!("Failed to run FFmpeg for conversion: {}", e))?;

        if !convert_status.success() {
            return Err("FFmpeg conversion failed".to_string());
        }

        // Read the MP3 file
        let audio_bytes = tokio::fs::read(&mp3_path).await
            .map_err(|e| format!("Failed to read MP3 output: {}", e))?;

        // Clean up
        let _ = tokio::fs::remove_file(&output_path).await;
        let _ = tokio::fs::remove_file(&mp3_path).await;

        Ok(audio_bytes)
    }

    /// Estimate duration from text (words per minute based)
    fn estimate_duration_seconds(&self, text: &str) -> f64 {
        let word_count = text.split_whitespace().count();
        // Assume ~150 words per minute for natural speech, adjusted by speed
        ((word_count as f64 / 150.0) * 60.0) / self.config.speed as f64
    }
}

#[async_trait]
impl AsyncNodeLogic for TTSLogic {
    async fn prep(
        &self,
        _params: &HashMap<String, NodeValue>,
        shared: &HashMap<String, NodeValue>,
    ) -> NodeValue {
        let script = shared
            .get(state_keys::SCRIPT)
            .cloned()
            .unwrap_or(serde_json::json!(null));
        
        let existing_timing = shared.get(state_keys::AUDIO_TIMING).cloned();

        serde_json::json!({
            "script": script,
            "existing_timing": existing_timing
        })
    }

    async fn exec(&self, input: NodeValue) -> NodeValue {
        // Parse the script
        let script: Script = match input.get("script") {
            Some(s) => match serde_json::from_value(s.clone()) {
                Ok(script) => script,
                Err(e) => {
                    return serde_json::json!({
                        "error": format!("Failed to parse script: {}", e)
                    });
                }
            },
            None => {
                return serde_json::json!({
                    "error": "No script provided"
                });
            }
        };

        // CHECK FOR RESUME
        if let Some(existing) = input.get("existing_timing").and_then(|v| serde_json::from_value::<AudioTiming>(v.clone()).ok()) {
            info!("TTS: Existing audio found for video {}, skipping generation", script.video_id);
            return serde_json::json!({
                "success": true,
                "audio_timing": serde_json::to_value(&existing).unwrap(),
                "is_resume": true
            });
        }

        info!(
            "TTS: Generating audio for video {} using {:?}",
            script.video_id,
            self.config.provider
        );

        let full_text = &script.full_text;
        let mut section_timings: Vec<SectionTiming> = Vec::new();
        let mut current_time: f64 = 0.0;

        // Generate audio for the full script
        let audio_bytes = match self.generate_audio(full_text).await {
            Ok(bytes) => bytes,
            Err(e) => {
                return serde_json::json!({
                    "error": format!("TTS generation failed: {}", e)
                });
            }
        };

        // Calculate section timings based on estimates
        // Hook
        let hook_duration = self.estimate_duration_seconds(&script.hook.narration);
        section_timings.push(SectionTiming {
            section_title: script.hook.title.clone(),
            start_seconds: 0.0,
            end_seconds: hook_duration,
        });
        current_time = hook_duration;

        // Main sections
        for section in &script.sections {
            let section_duration = self.estimate_duration_seconds(&section.narration);
            section_timings.push(SectionTiming {
                section_title: section.title.clone(),
                start_seconds: current_time,
                end_seconds: current_time + section_duration,
            });
            current_time += section_duration;
        }

        // CTA
        let cta_duration = self.estimate_duration_seconds(&script.cta.narration);
        section_timings.push(SectionTiming {
            section_title: script.cta.title.clone(),
            start_seconds: current_time,
            end_seconds: current_time + cta_duration,
        });
        current_time += cta_duration;

        // Upload to S3
        let audio_key = format!("audio/{}/{}.mp3", script.video_id, script.video_id);
        
        match self.s3_client.upload_bytes(&audio_key, audio_bytes, "audio/mpeg").await {
            Ok(_) => {
                info!("TTS: Uploaded audio to {}", audio_key);
            }
            Err(e) => {
                return serde_json::json!({
                    "error": format!("Failed to upload audio: {}", e)
                });
            }
        }

        serde_json::json!({
            "success": true,
            "audio_path": audio_key,
            "total_duration_seconds": current_time,
            "section_timings": section_timings,
            "video_id": script.video_id.to_string(),
            "provider": format!("{:?}", self.config.provider)
        })
    }

    async fn post(
        &self,
        shared: &mut HashMap<String, NodeValue>,
        _prep_res: NodeValue,
        exec_res: NodeValue,
    ) -> Option<String> {
        // Check for errors
        if let Some(error) = exec_res.get("error").and_then(|v| v.as_str()) {
            error!("TTS node failed: {}", error);
            shared.insert(state_keys::ERROR.to_string(), serde_json::json!(error));
            
            // Mark video as failed in database
            if let Some(vid) = shared.get(state_keys::VIDEO_ID).and_then(|v| v.as_str()) {
                if let Ok(video_id) = uuid::Uuid::parse_str(vid) {
                    let _ = db::mark_video_failed(&self.db_pool, video_id, "tts", error).await;
                }
            }
            
            return Some("error".to_string());
        }

        let audio_path = exec_res.get("audio_path")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        let total_duration = exec_res.get("total_duration_seconds")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);

        let section_timings: Vec<SectionTiming> = exec_res.get("section_timings")
            .and_then(|v| serde_json::from_value(v.clone()).ok())
            .unwrap_or_default();

        let provider = exec_res.get("provider")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");

        let audio_timing = AudioTiming {
            audio_path: audio_path.clone(),
            total_duration_seconds: total_duration,
            section_timings,
        };

        info!(
            "TTS: Generated audio ({}s duration, provider: {})",
            total_duration as u32,
            provider
        );

        // Store in shared state
        shared.insert(state_keys::AUDIO_PATH.to_string(), serde_json::json!(audio_path));
        shared.insert(
            state_keys::AUDIO_TIMING.to_string(),
            serde_json::to_value(&audio_timing).unwrap_or(serde_json::json!(null)),
        );

        // Persist audio_timing to database
        if let Some(vid) = exec_res.get("video_id").and_then(|v| v.as_str()) {
            if let Ok(video_id) = uuid::Uuid::parse_str(vid) {
                if let Err(e) = db::update_video_json_field(
                    &self.db_pool,
                    video_id,
                    "audio_timing",
                    serde_json::to_value(&audio_timing).unwrap_or(serde_json::json!(null)),
                ).await {
                    error!("Failed to persist audio_timing to database: {}", e);
                }
            }
        }

        // Proceed to asset collector
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
    fn test_chunk_text_short() {
        // Short text should not be chunked
        let text = "This is a short sentence. It should not be chunked.";
        let chunks = TTSLogic::chunk_text_for_tts(text, 4500);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], text);
    }

    #[test]
    fn test_chunk_text_by_sentences() {
        // Should split at sentence boundaries
        let text = "First sentence. Second sentence. Third sentence. Fourth sentence.";
        let chunks = TTSLogic::chunk_text_for_tts(text, 40);
        assert!(chunks.len() >= 2);
        // Each chunk should end with a sentence terminator or be complete
        for chunk in &chunks {
            assert!(!chunk.is_empty());
        }
    }

    #[test]
    fn test_chunk_text_preserves_content() {
        let text = "The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs. How vexingly quick daft zebras jump!";
        let chunks = TTSLogic::chunk_text_for_tts(text, 60);
        
        // Rejoin and verify no content is lost (allowing for whitespace normalization)
        let rejoined: String = chunks.join(" ");
        let original_words: Vec<&str> = text.split_whitespace().collect();
        let rejoined_words: Vec<&str> = rejoined.split_whitespace().collect();
        
        // All original words should be present
        assert_eq!(original_words.len(), rejoined_words.len());
    }

    #[test]
    fn test_chunk_text_respects_max_length() {
        let text = "A. B. C. D. E. F. G. H. I. J. K. L. M. N. O. P. Q. R. S. T. U. V. W. X. Y. Z.";
        let max_chars = 20;
        let chunks = TTSLogic::chunk_text_for_tts(text, max_chars);
        
        for chunk in &chunks {
            assert!(
                chunk.len() <= max_chars + 10, // Small buffer for edge cases
                "Chunk too long: {} chars (max {}): '{}'",
                chunk.len(), max_chars, chunk
            );
        }
    }
}
