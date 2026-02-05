//! SEO Optimizer Node
//!
//! Generates optimized titles, descriptions, tags, and chapter markers.

use async_trait::async_trait;
use orichalcum::{AsyncNodeLogic, NodeValue};
use orichalcum::llm::{Client, Enabled, Providers};
use serde::{Deserialize, Serialize};
use sqlx::PgPool;
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{error, info};

use crate::db;
use crate::nodes::{Chapter, Script, SEOMetadata, TopicBrief};
use crate::state_keys;

/// Configuration for SEO optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SEOConfig {
    /// Channel name for descriptions
    pub channel_name: String,
    /// Max title length
    pub max_title_length: usize,
    /// Max tags count
    pub max_tags: usize,
}

impl Default for SEOConfig {
    fn default() -> Self {
        Self {
            channel_name: "Excelsior Academy".to_string(),
            max_title_length: 70,
            max_tags: 15,
        }
    }
}

/// The SEO optimizer node logic
#[derive(Clone)]
pub struct SEOOptimizerLogic {
    pub config: SEOConfig,
    pub llm_client: Arc<Client<Providers<orichalcum::llm::Disabled, Enabled, Enabled>>>,
    pub db_pool: Arc<PgPool>,
}

impl SEOOptimizerLogic {
    pub fn new(
        config: SEOConfig,
        llm_client: Arc<Client<Providers<orichalcum::llm::Disabled, Enabled, Enabled>>>,
        db_pool: Arc<PgPool>,
    ) -> Self {
        Self { config, llm_client, db_pool }
    }

    fn build_system_prompt(&self) -> String {
        format!(
            r#"You are an SEO expert for the YouTube channel "{}".

Your task is to optimize video metadata for maximum discoverability while maintaining authenticity.

Guidelines:
1. TITLES: Use curiosity gaps, power words, and emotional triggers. Keep under 70 characters.
2. DESCRIPTIONS: Front-load keywords, include timestamps, add value beyond the video.
3. TAGS: Mix broad and specific terms, include common misspellings.

The content is thoughtful wisdom/philosophy - avoid clickbait that overpromises.

Respond in JSON format."#,
            self.config.channel_name
        )
    }

    fn build_user_prompt(&self, topic_brief: &TopicBrief, script: &Script) -> String {
        // Build chapter timestamps from script sections
        let sections_info: Vec<String> = script.sections.iter()
            .map(|s| format!("- {}: {} seconds", s.title, s.duration_seconds))
            .collect();

        format!(
            r#"Generate SEO-optimized metadata for this video:

TOPIC: {}
DESCRIPTION: {}
TARGET KEYWORDS: {}
HOOK ANGLE: {}

SCRIPT SECTIONS:
{}

Return a JSON object:
{{
    "title": "Optimized title (max 70 chars, curiosity-inducing)",
    "description": "Full description with timestamps, keywords, and call to action (1500+ chars)",
    "tags": ["tag1", "tag2", ...up to 15 tags],
    "chapters": [
        {{"title": "Intro", "timestamp_seconds": 0}},
        ...
    ]
}}"#,
            topic_brief.topic,
            topic_brief.description,
            topic_brief.target_keywords.join(", "),
            topic_brief.hook_angle,
            sections_info.join("\n")
        )
    }
}

#[async_trait]
impl AsyncNodeLogic for SEOOptimizerLogic {
    async fn prep(
        &self,
        _params: &HashMap<String, NodeValue>,
        shared: &HashMap<String, NodeValue>,
    ) -> NodeValue {
        let topic_brief = shared.get(state_keys::TOPIC_BRIEF).cloned().unwrap_or(serde_json::json!(null));
        let script = shared.get(state_keys::SCRIPT).cloned().unwrap_or(serde_json::json!(null));
        let thumbnail_path = shared.get(state_keys::THUMBNAIL_PATH).cloned().unwrap_or(serde_json::json!(null));

        serde_json::json!({
            "topic_brief": topic_brief,
            "script": script,
            "thumbnail_path": thumbnail_path
        })
    }

    async fn exec(&self, input: NodeValue) -> NodeValue {
        let topic_brief: TopicBrief = match input.get("topic_brief")
            .and_then(|v| serde_json::from_value(v.clone()).ok()) {
            Some(t) => t,
            None => return serde_json::json!({ "error": "No topic brief provided" }),
        };

        let script: Script = match input.get("script")
            .and_then(|v| serde_json::from_value(v.clone()).ok()) {
            Some(s) => s,
            None => return serde_json::json!({ "error": "No script provided" }),
        };

        let thumbnail_path = input.get("thumbnail_path")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        info!("SEO: Optimizing metadata for video {}", topic_brief.video_id);

        let system_prompt = self.build_system_prompt();
        let user_prompt = self.build_user_prompt(&topic_brief, &script);

        let response = match self.llm_client.deepseek_complete(
            "deepseek-chat",
            &system_prompt,
            &user_prompt,
            Some(0.6),
            Some(2000),
        ).await {
            Ok(text) => text,
            Err(e) => return serde_json::json!({ "error": format!("LLM call failed: {}", e) }),
        };

        // Parse response
        let json_str = response.trim()
            .trim_start_matches("```json")
            .trim_start_matches("```")
            .trim_end_matches("```")
            .trim();

        let seo_data: serde_json::Value = match serde_json::from_str(json_str) {
            Ok(d) => d,
            Err(e) => return serde_json::json!({ 
                "error": format!("Failed to parse SEO response: {}", e),
                "raw_response": response
            }),
        };

        serde_json::json!({
            "success": true,
            "seo_data": seo_data,
            "video_id": topic_brief.video_id.to_string(),
            "thumbnail_path": thumbnail_path
        })
    }

    async fn post(
        &self,
        shared: &mut HashMap<String, NodeValue>,
        _prep_res: NodeValue,
        exec_res: NodeValue,
    ) -> Option<String> {
        if let Some(error) = exec_res.get("error").and_then(|v| v.as_str()) {
            error!("SEO optimization failed: {}", error);
            shared.insert(state_keys::ERROR.to_string(), serde_json::json!(error));
            
            // Mark video as failed in database
            if let Some(vid) = shared.get(state_keys::VIDEO_ID).and_then(|v| v.as_str()) {
                if let Ok(video_id) = uuid::Uuid::parse_str(vid) {
                    let _ = db::mark_video_failed(&self.db_pool, video_id, "seo_optimizer", error).await;
                }
            }
            
            return Some("error".to_string());
        }

        let seo_data = match exec_res.get("seo_data") {
            Some(d) => d.clone(),
            None => return Some("error".to_string()),
        };

        let video_id = exec_res.get("video_id")
            .and_then(|v| v.as_str())
            .and_then(|s| uuid::Uuid::parse_str(s).ok())
            .unwrap_or_else(uuid::Uuid::new_v4);

        let thumbnail_paths: Vec<String> = exec_res.get("thumbnail_path")
            .and_then(|v| v.as_str())
            .map(|s| vec![s.to_string()])
            .unwrap_or_default();

        let chapters: Vec<Chapter> = seo_data.get("chapters")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter().filter_map(|c| {
                    Some(Chapter {
                        title: c.get("title")?.as_str()?.to_string(),
                        timestamp_seconds: c.get("timestamp_seconds")?.as_u64()? as u32,
                    })
                }).collect()
            })
            .unwrap_or_default();

        let tags: Vec<String> = seo_data.get("tags")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|t| t.as_str().map(|s| s.to_string()))
                    .take(self.config.max_tags)
                    .collect()
            })
            .unwrap_or_default();

        let metadata = SEOMetadata {
            video_id,
            title: seo_data.get("title").and_then(|v| v.as_str()).unwrap_or("").to_string(),
            description: seo_data.get("description").and_then(|v| v.as_str()).unwrap_or("").to_string(),
            tags,
            chapters,
            thumbnail_paths,
        };

        info!("SEO: Optimized metadata - Title: '{}'", metadata.title);

        shared.insert(
            state_keys::SEO_METADATA.to_string(),
            serde_json::to_value(&metadata).unwrap_or(serde_json::json!(null)),
        );

        // Persist seo_metadata to database
        if let Err(e) = db::update_video_json_field(
            &self.db_pool,
            video_id,
            "seo_metadata",
            serde_json::to_value(&metadata).unwrap_or(serde_json::json!(null)),
        ).await {
            error!("Failed to persist seo_metadata to database: {}", e);
        }

        Some("default".to_string())
    }

    fn clone_box(&self) -> Box<dyn AsyncNodeLogic> {
        Box::new(self.clone())
    }
}
