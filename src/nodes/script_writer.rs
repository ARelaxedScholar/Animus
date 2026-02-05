//! Script Writer Node
//!
//! Generates engaging 12-20 minute scripts with hooks, storytelling, and CTAs.
//! Uses LLM to create scripts optimized for watch time and engagement.

use async_trait::async_trait;
use orichalcum::{AsyncNodeLogic, NodeValue};
use orichalcum::llm::{Client, Enabled, Providers};
use serde::{Deserialize, Serialize};
use sqlx::PgPool;
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{error, info};

use crate::db;
use crate::nodes::{Script, ScriptSection, TopicBrief};
use crate::state_keys;

/// Configuration for script writing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScriptWriterConfig {
    /// Words per minute for duration estimation (average speaking rate)
    pub words_per_minute: u32,
    /// Channel persona description
    pub persona: String,
    /// Channel name
    pub channel_name: String,
}

impl Default for ScriptWriterConfig {
    fn default() -> Self {
        Self {
            words_per_minute: 150, // Slightly slower for thoughtful content
            persona: "An experienced traveler on life's journey, sharing wisdom with his past self. \
                     Speak as if you're sitting with a younger version of yourself, sharing hard-won \
                     insights with warmth and understanding. Neither preachy nor casual - thoughtful, \
                     genuine, and deeply human.".to_string(),
            channel_name: "Excelsior Academy".to_string(),
        }
    }
}

/// The script writer node logic
#[derive(Clone)]
pub struct ScriptWriterLogic {
    pub config: ScriptWriterConfig,
    pub llm_client: Arc<Client<Providers<orichalcum::llm::Disabled, Enabled, Enabled>>>,
    pub db_pool: Arc<PgPool>,
}

impl ScriptWriterLogic {
    pub fn new(
        config: ScriptWriterConfig,
        llm_client: Arc<Client<Providers<orichalcum::llm::Disabled, Enabled, Enabled>>>,
        db_pool: Arc<PgPool>,
    ) -> Self {
        Self { config, llm_client, db_pool }
    }

    /// Build the system prompt for script writing
    fn build_system_prompt(&self) -> String {
        format!(
            r#"You are a master scriptwriter for the YouTube channel "{}".

Your voice and persona: {}

CRITICAL RULES FOR SCRIPT WRITING:
1. HOOK (First 30 seconds): Start with a provocative question, surprising fact, or relatable struggle. The viewer should feel "this video is for ME" immediately.

2. STRUCTURE: Use the "Promise → Story → Wisdom → Application" framework:
   - Promise what they'll learn
   - Share a story or example that illustrates the struggle
   - Reveal the wisdom (from the source material)
   - Give practical application

3. PACING: Vary sentence length. Short punchy lines for impact. Longer flowing passages for story. Never let the energy flatten.

4. RETENTION: Every 2-3 minutes, create a "retention hook" - a teaser of what's coming, a surprising turn, or a powerful question.

5. AUTHENTICITY: Speak like a real person reflecting on life, not a motivational speaker performing. Include moments of vulnerability and honest uncertainty.

6. CITATIONS: When quoting wisdom sources, set them up with context. Don't just drop quotes - weave them into the narrative.

7. VISUAL CUES: Include [B-ROLL SUGGESTIONS] in brackets throughout for the video editor.

8. CTA: End with a genuine call to action that feels earned, not salesy.

Write scripts that people will FINISH watching because they genuinely want to hear what comes next."#,
            self.config.channel_name,
            self.config.persona
        )
    }

    /// Build the user prompt for a specific topic
    fn build_user_prompt(&self, topic_brief: &TopicBrief) -> String {
        let sources_desc = format!(
            "Primary source: {} - {}\nSecondary sources: {}",
            topic_brief.primary_source.category_name(),
            match &topic_brief.primary_source {
                crate::nodes::WisdomSource::Bible { book } => book.clone(),
                crate::nodes::WisdomSource::Stoicism { author } => author.clone(),
                crate::nodes::WisdomSource::Philosophy { author } => author.clone(),
                crate::nodes::WisdomSource::Biography { subject } => subject.clone(),
                crate::nodes::WisdomSource::Psychology { author } => author.clone(),
            },
            topic_brief.secondary_sources
                .iter()
                .map(|s| format!("{}", s.category_name()))
                .collect::<Vec<_>>()
                .join(", ")
        );

        let target_words = topic_brief.target_duration_minutes * self.config.words_per_minute;

        format!(
            r#"Write a complete YouTube script for this video:

TOPIC: {}

DESCRIPTION: {}

HOOK ANGLE: {}

TARGET DURATION: {} minutes (approximately {} words)

WISDOM SOURCES:
{}

TARGET KEYWORDS (weave naturally): {}

Return a JSON object with this structure:
{{
    "hook": {{
        "title": "Hook",
        "narration": "The complete hook script (first 30-45 seconds)...",
        "duration_seconds": 30,
        "visual_suggestions": ["suggestion1", "suggestion2"]
    }},
    "sections": [
        {{
            "title": "Section Title",
            "narration": "The complete section narration...",
            "duration_seconds": 180,
            "visual_suggestions": ["suggestion1", "suggestion2"]
        }}
    ],
    "cta": {{
        "title": "Call to Action",
        "narration": "The closing CTA...",
        "duration_seconds": 30,
        "visual_suggestions": ["suggestion1"]
    }}
}}

IMPORTANT:
- The total narration should be approximately {} words
- Include 4-6 main sections plus hook and CTA
- Each section should be 2-4 minutes of content
- Make visual suggestions specific and evocative"#,
            topic_brief.topic,
            topic_brief.description,
            topic_brief.hook_angle,
            topic_brief.target_duration_minutes,
            target_words,
            sources_desc,
            topic_brief.target_keywords.join(", "),
            target_words
        )
    }

    /// Calculate duration from word count
    fn estimate_duration_seconds(&self, text: &str) -> u32 {
        let word_count = text.split_whitespace().count() as u32;
        (word_count * 60) / self.config.words_per_minute
    }
}

#[async_trait]
impl AsyncNodeLogic for ScriptWriterLogic {
    async fn prep(
        &self,
        _params: &HashMap<String, NodeValue>,
        shared: &HashMap<String, NodeValue>,
    ) -> NodeValue {
        // Get the topic brief from shared state
        let topic_brief = shared
            .get(state_keys::TOPIC_BRIEF)
            .cloned()
            .unwrap_or(serde_json::json!(null));

        serde_json::json!({
            "topic_brief": topic_brief
        })
    }

    async fn exec(&self, input: NodeValue) -> NodeValue {
        // Parse the topic brief
        let topic_brief: TopicBrief = match input.get("topic_brief") {
            Some(tb) => match serde_json::from_value(tb.clone()) {
                Ok(brief) => brief,
                Err(e) => {
                    return serde_json::json!({
                        "error": format!("Failed to parse topic brief: {}", e)
                    });
                }
            },
            None => {
                return serde_json::json!({
                    "error": "No topic brief provided"
                });
            }
        };

        let system_prompt = self.build_system_prompt();
        let user_prompt = self.build_user_prompt(&topic_brief);

        info!("ScriptWriter: Generating script for '{}'", topic_brief.topic);

        // Call LLM for script generation (use Gemini for longer context)
        let response = match self.llm_client.gemini_complete(
            "gemini-3-flash-preview",
            &system_prompt,
            &user_prompt,
            Some(0.7),
            Some(8000), // Allow for long scripts
        ).await {
            Ok(text) => text,
            Err(e) => {
                error!("ScriptWriter LLM call failed: {}", e);
                return serde_json::json!({
                    "error": format!("LLM call failed: {}", e)
                });
            }
        };

        // Parse the JSON response
        let json_str = response
            .trim()
            .trim_start_matches("```json")
            .trim_start_matches("```")
            .trim_end_matches("```")
            .trim();

        match serde_json::from_str::<serde_json::Value>(json_str) {
            Ok(parsed) => {
                serde_json::json!({
                    "success": true,
                    "script_data": parsed,
                    "video_id": topic_brief.video_id.to_string()
                })
            }
            Err(e) => {
                error!("Failed to parse script response: {}. Response: {}", e, &response[..500.min(response.len())]);
                serde_json::json!({
                    "error": format!("Failed to parse script response: {}", e),
                    "raw_response": response
                })
            }
        }
    }

    async fn post(
        &self,
        shared: &mut HashMap<String, NodeValue>,
        _prep_res: NodeValue,
        exec_res: NodeValue,
    ) -> Option<String> {
        // Check for errors
        if let Some(error) = exec_res.get("error").and_then(|v| v.as_str()) {
            error!("ScriptWriter node failed: {}", error);
            shared.insert(state_keys::ERROR.to_string(), serde_json::json!(error));

            // Mark video as failed in database
            if let Some(vid) = shared.get(state_keys::VIDEO_ID).and_then(|v| v.as_str()) {
                if let Ok(video_id) = uuid::Uuid::parse_str(vid) {
                    let _ = db::mark_video_failed(&self.db_pool, video_id, "script_writer", error).await;
                }
            }

            return Some("error".to_string());
        }

        let script_data = match exec_res.get("script_data") {
            Some(data) => data.clone(),
            None => {
                error!("No script data in response");
                return Some("error".to_string());
            }
        };

        let video_id = exec_res
            .get("video_id")
            .and_then(|v| v.as_str())
            .and_then(|s| uuid::Uuid::parse_str(s).ok())
            .unwrap_or_else(uuid::Uuid::new_v4);

        // Parse sections
        let parse_section = |v: &serde_json::Value| -> Option<ScriptSection> {
            Some(ScriptSection {
                title: v.get("title")?.as_str()?.to_string(),
                narration: v.get("narration")?.as_str()?.to_string(),
                duration_seconds: v.get("duration_seconds")?.as_u64()? as u32,
                visual_suggestions: v.get("visual_suggestions")?
                    .as_array()?
                    .iter()
                    .filter_map(|s| s.as_str().map(|s| s.to_string()))
                    .collect(),
            })
        };

        let hook = script_data.get("hook")
            .and_then(parse_section)
            .unwrap_or_else(|| ScriptSection {
                title: "Hook".to_string(),
                narration: String::new(),
                duration_seconds: 30,
                visual_suggestions: vec![],
            });

        let sections: Vec<ScriptSection> = script_data.get("sections")
            .and_then(|v| v.as_array())
            .map(|arr| arr.iter().filter_map(parse_section).collect())
            .unwrap_or_default();

        let cta = script_data.get("cta")
            .and_then(parse_section)
            .unwrap_or_else(|| ScriptSection {
                title: "Call to Action".to_string(),
                narration: String::new(),
                duration_seconds: 30,
                visual_suggestions: vec![],
            });

        // Build full text
        let mut full_text = hook.narration.clone();
        for section in &sections {
            full_text.push_str("\n\n");
            full_text.push_str(&section.narration);
        }
        full_text.push_str("\n\n");
        full_text.push_str(&cta.narration);

        // Calculate total duration
        let total_duration = hook.duration_seconds
            + sections.iter().map(|s| s.duration_seconds).sum::<u32>()
            + cta.duration_seconds;

        let script = Script {
            video_id,
            hook,
            sections,
            cta,
            total_duration_seconds: total_duration,
            full_text,
        };

        info!(
            "ScriptWriter: Generated script for video {} ({} seconds, {} words)",
            video_id,
            total_duration,
            script.full_text.split_whitespace().count()
        );

        // Store in shared state
        shared.insert(
            state_keys::SCRIPT.to_string(),
            serde_json::to_value(&script).unwrap_or(serde_json::json!(null)),
        );

        // Persist script to database
        if let Err(e) = db::update_video_json_field(
            &self.db_pool,
            video_id,
            "script",
            serde_json::to_value(&script).unwrap_or(serde_json::json!(null)),
        ).await {
            error!("Failed to persist script to database: {}", e);
        }

        // Proceed to TTS node
        Some("default".to_string())
    }

    fn clone_box(&self) -> Box<dyn AsyncNodeLogic> {
        Box::new(self.clone())
    }
}
