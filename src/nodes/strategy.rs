//! Strategy Node
//!
//! Analyzes trends, selects topics from wisdom sources, and creates detailed topic briefs.
//! Uses LLM to generate compelling topic ideas based on the wisdom source catalog.

use async_trait::async_trait;
use orichalcum::{AsyncNodeLogic, NodeValue};
use orichalcum::llm::{Client, Enabled, Providers};
use serde::{Deserialize, Serialize};
use sqlx::PgPool;
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{error, info};
use uuid::Uuid;

use crate::db::{self, Video};
use crate::nodes::{TopicBrief, WisdomSource};
use crate::state_keys;

/// Configuration for the strategy node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyConfig {
    /// Target video duration range
    pub target_duration_min: u32,
    pub target_duration_max: u32,
    /// Channel persona description
    pub persona: String,
    /// Channel name
    pub channel_name: String,
}

impl Default for StrategyConfig {
    fn default() -> Self {
        Self {
            target_duration_min: 12,
            target_duration_max: 20,
            persona: "An experienced traveler on life's journey, sharing wisdom with his past self. \
                     Neither a mentor nor professor, but a thoughtful companion who has walked \
                     the path and learned from both triumphs and failures.".to_string(),
            channel_name: "Excelsior Academy".to_string(),
        }
    }
}

/// Wisdom source catalog entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WisdomCatalogEntry {
    pub source: WisdomSource,
    pub key_themes: Vec<String>,
    pub notable_quotes: Vec<String>,
    pub use_count: u32,
}

/// The strategy node logic
#[derive(Clone)]
pub struct StrategyLogic {
    pub config: StrategyConfig,
    pub llm_client: Arc<Client<Providers<orichalcum::llm::Disabled, Enabled, Enabled>>>,
    pub db_pool: Arc<PgPool>,
    pub wisdom_catalog: Vec<WisdomCatalogEntry>,
}

impl StrategyLogic {
    pub fn new(
        config: StrategyConfig,
        llm_client: Arc<Client<Providers<orichalcum::llm::Disabled, Enabled, Enabled>>>,
        db_pool: Arc<PgPool>,
    ) -> Self {
        Self {
            config,
            llm_client,
            db_pool,
            wisdom_catalog: Self::default_wisdom_catalog(),
        }
    }

    /// Initialize the default wisdom catalog
    fn default_wisdom_catalog() -> Vec<WisdomCatalogEntry> {
        vec![
            // Bible - Proverbs
            WisdomCatalogEntry {
                source: WisdomSource::Bible { book: "Proverbs".to_string() },
                key_themes: vec![
                    "Wisdom vs foolishness".to_string(),
                    "Discipline and correction".to_string(),
                    "The fear of the Lord".to_string(),
                    "Speech and silence".to_string(),
                    "Wealth and poverty".to_string(),
                ],
                notable_quotes: vec![
                    "The fear of the Lord is the beginning of wisdom".to_string(),
                    "Trust in the Lord with all your heart".to_string(),
                    "Iron sharpens iron".to_string(),
                ],
                use_count: 0,
            },
            // Bible - Ecclesiastes
            WisdomCatalogEntry {
                source: WisdomSource::Bible { book: "Ecclesiastes".to_string() },
                key_themes: vec![
                    "The meaning of life".to_string(),
                    "Vanity and purpose".to_string(),
                    "Time and seasons".to_string(),
                    "Joy in simple things".to_string(),
                    "Death and legacy".to_string(),
                ],
                notable_quotes: vec![
                    "Vanity of vanities, all is vanity".to_string(),
                    "To everything there is a season".to_string(),
                    "Remember your Creator in the days of your youth".to_string(),
                ],
                use_count: 0,
            },
            // Bible - Job
            WisdomCatalogEntry {
                source: WisdomSource::Bible { book: "Job".to_string() },
                key_themes: vec![
                    "Suffering without explanation".to_string(),
                    "Faith through darkness".to_string(),
                    "God's sovereignty".to_string(),
                    "Human limitations".to_string(),
                ],
                notable_quotes: vec![
                    "The Lord gave, and the Lord has taken away".to_string(),
                    "Though he slay me, yet will I trust in him".to_string(),
                ],
                use_count: 0,
            },
            // Bible - Romans
            WisdomCatalogEntry {
                source: WisdomSource::Bible { book: "Romans".to_string() },
                key_themes: vec![
                    "Suffering produces perseverance".to_string(),
                    "Grace and redemption".to_string(),
                    "Transformation of mind".to_string(),
                    "Purpose in trials".to_string(),
                ],
                notable_quotes: vec![
                    "Suffering produces perseverance; perseverance, character; and character, hope".to_string(),
                    "All things work together for good".to_string(),
                ],
                use_count: 0,
            },
            // Stoicism - Marcus Aurelius
            WisdomCatalogEntry {
                source: WisdomSource::Stoicism { author: "Marcus Aurelius".to_string() },
                key_themes: vec![
                    "Control what you can".to_string(),
                    "Memento mori".to_string(),
                    "Duty and responsibility".to_string(),
                    "Inner fortress".to_string(),
                    "Present moment".to_string(),
                ],
                notable_quotes: vec![
                    "You have power over your mind, not outside events".to_string(),
                    "The obstacle is the way".to_string(),
                    "Waste no more time arguing about what a good man should be. Be one.".to_string(),
                ],
                use_count: 0,
            },
            // Stoicism - Seneca
            WisdomCatalogEntry {
                source: WisdomSource::Stoicism { author: "Seneca".to_string() },
                key_themes: vec![
                    "Time and mortality".to_string(),
                    "Anger and emotions".to_string(),
                    "Friendship".to_string(),
                    "Adversity as opportunity".to_string(),
                ],
                notable_quotes: vec![
                    "It is not that we have a short time to live, but that we waste a lot of it".to_string(),
                    "Difficulties strengthen the mind".to_string(),
                ],
                use_count: 0,
            },
            // Stoicism - Epictetus
            WisdomCatalogEntry {
                source: WisdomSource::Stoicism { author: "Epictetus".to_string() },
                key_themes: vec![
                    "Dichotomy of control".to_string(),
                    "Freedom through acceptance".to_string(),
                    "External vs internal".to_string(),
                ],
                notable_quotes: vec![
                    "It's not what happens to you, but how you react to it that matters".to_string(),
                    "First say to yourself what you would be; then do what you have to do".to_string(),
                ],
                use_count: 0,
            },
            // Psychology - Viktor Frankl
            WisdomCatalogEntry {
                source: WisdomSource::Psychology { author: "Viktor Frankl".to_string() },
                key_themes: vec![
                    "Meaning in suffering".to_string(),
                    "Purpose as survival".to_string(),
                    "Choosing attitude".to_string(),
                    "Logotherapy".to_string(),
                ],
                notable_quotes: vec![
                    "He who has a why to live can bear almost any how".to_string(),
                    "Everything can be taken from a man but the last of human freedoms".to_string(),
                ],
                use_count: 0,
            },
            // Philosophy - Nietzsche
            WisdomCatalogEntry {
                source: WisdomSource::Philosophy { author: "Friedrich Nietzsche".to_string() },
                key_themes: vec![
                    "Will to power".to_string(),
                    "Amor fati".to_string(),
                    "Becoming who you are".to_string(),
                    "Overcoming".to_string(),
                ],
                notable_quotes: vec![
                    "What does not kill me makes me stronger".to_string(),
                    "He who fights with monsters should look to it that he himself does not become a monster".to_string(),
                ],
                use_count: 0,
            },
            // Biography
            WisdomCatalogEntry {
                source: WisdomSource::Biography { subject: "Abraham Lincoln".to_string() },
                key_themes: vec![
                    "Failure and persistence".to_string(),
                    "Leadership through adversity".to_string(),
                    "Moral courage".to_string(),
                    "Depression and purpose".to_string(),
                ],
                notable_quotes: vec![
                    "I am a slow walker, but I never walk back".to_string(),
                    "The best way to predict the future is to create it".to_string(),
                ],
                use_count: 0,
            },
        ]
    }

    /// Build the system prompt for topic generation
    fn build_system_prompt(&self) -> String {
        format!(
            r#"You are a content strategist for a YouTube channel called "{}".

The channel's persona: {}

Your task is to generate compelling video topic ideas that:
1. Draw from classic wisdom sources (the Bible, Stoic philosophy, great biographies, psychology classics)
2. Address timeless human struggles and questions
3. Offer genuine insight, not superficial motivation
4. Appeal to thoughtful viewers seeking depth over entertainment
5. Have strong "watch time" potential through storytelling and revelation

The target audience seeks meaning, purpose, and practical wisdom for navigating life's challenges.
Videos should be {}-{} minutes, allowing for depth without losing engagement.

Respond in JSON format."#,
            self.config.channel_name,
            self.config.persona,
            self.config.target_duration_min,
            self.config.target_duration_max
        )
    }

    /// Build the user prompt for a specific source focus
    fn build_user_prompt(&self, source_focus: &str, seed_topic: Option<&str>) -> String {
        let seed_section = seed_topic
            .map(|t| format!("\n\nThe user has specifically requested this seed topic: \"{}\".\nBuild the topic brief around this theme.", t))
            .unwrap_or_default();

        format!(
            r#"Generate a compelling video topic brief focusing on {} wisdom.{}

Select themes that resonate with modern struggles while drawing from timeless wisdom.

Return a JSON object with this structure:
{{
    "topic": "A compelling, curiosity-inducing title idea (not the final title, but the core concept)",
    "description": "A 2-3 sentence description of what the video will cover and why it matters",
    "primary_source": {{
        "category": "Bible|Stoicism|Philosophy|Biography|Psychology",
        "specific": "e.g., 'Ecclesiastes' or 'Marcus Aurelius'"
    }},
    "secondary_sources": [
        {{"category": "...", "specific": "..."}}
    ],
    "target_keywords": ["keyword1", "keyword2", "keyword3"],
    "hook_angle": "The specific angle or question that will hook viewers in the first 30 seconds"
}}"#,
            source_focus,
            seed_section
        )
    }
}

#[async_trait]
impl AsyncNodeLogic for StrategyLogic {
    async fn prep(
        &self,
        _params: &HashMap<String, NodeValue>,
        shared: &HashMap<String, NodeValue>,
    ) -> NodeValue {
        let video_id = shared
            .get(state_keys::VIDEO_ID)
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .unwrap_or_else(|| Uuid::new_v4().to_string());

        let source_focus = shared
            .get("source_focus")
            .and_then(|v| v.as_str())
            .unwrap_or("Bible")
            .to_string();

        let seed_topic = shared
            .get("seed_topic")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        let scheduled_publish = shared
            .get("scheduled_publish")
            .cloned();

        serde_json::json!({
            "video_id": video_id,
            "source_focus": source_focus,
            "seed_topic": seed_topic,
            "scheduled_publish": scheduled_publish
        })
    }

    async fn exec(&self, input: NodeValue) -> NodeValue {
        let source_focus = input.get("source_focus")
            .and_then(|v| v.as_str())
            .unwrap_or("Bible");

        let seed_topic = input.get("seed_topic")
            .and_then(|v| v.as_str());

        let system_prompt = self.build_system_prompt();
        let user_prompt = self.build_user_prompt(source_focus, seed_topic);

        // Call DeepSeek for topic generation
        let response = match self.llm_client.deepseek_complete(
            "deepseek-chat",
            &system_prompt,
            &user_prompt,
            Some(0.8), // Higher temperature for creativity
            Some(1000),
        ).await {
            Ok(text) => text,
            Err(e) => {
                error!("Strategy LLM call failed: {}", e);
                return serde_json::json!({
                    "error": format!("LLM call failed: {}", e)
                });
            }
        };

        // Parse the JSON response
        // Try to extract JSON from the response (it might have markdown code blocks)
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
                    "topic_data": parsed,
                    "video_id": input.get("video_id"),
                    "scheduled_publish": input.get("scheduled_publish")
                })
            }
            Err(e) => {
                error!("Failed to parse LLM response: {}. Response: {}", e, response);
                serde_json::json!({
                    "error": format!("Failed to parse topic response: {}", e),
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
            error!("Strategy node failed: {}", error);
            shared.insert(state_keys::ERROR.to_string(), serde_json::json!(error));
            
            // Try to mark video as failed in database if we have an ID
            if let Some(vid) = shared.get(state_keys::VIDEO_ID).and_then(|v| v.as_str()) {
                if let Ok(video_id) = Uuid::parse_str(vid) {
                    let _ = db::mark_video_failed(&self.db_pool, video_id, "strategy", error).await;
                }
            }
            
            return Some("error".to_string());
        }

        // Extract topic data
        let topic_data = match exec_res.get("topic_data") {
            Some(data) => data.clone(),
            None => {
                error!("No topic data in response");
                return Some("error".to_string());
            }
        };

        let video_id = exec_res
            .get("video_id")
            .and_then(|v| v.as_str())
            .and_then(|s| Uuid::parse_str(s).ok())
            .unwrap_or_else(Uuid::new_v4);

        // Build the topic brief
        let primary_source = topic_data
            .get("primary_source")
            .map(|ps| {
                let category = ps.get("category").and_then(|v| v.as_str()).unwrap_or("");
                let specific = ps.get("specific").and_then(|v| v.as_str()).unwrap_or("");
                match category {
                    "Bible" => WisdomSource::Bible { book: specific.to_string() },
                    "Stoicism" => WisdomSource::Stoicism { author: specific.to_string() },
                    "Philosophy" => WisdomSource::Philosophy { author: specific.to_string() },
                    "Biography" => WisdomSource::Biography { subject: specific.to_string() },
                    "Psychology" => WisdomSource::Psychology { author: specific.to_string() },
                    _ => WisdomSource::Bible { book: specific.to_string() },
                }
            })
            .unwrap_or(WisdomSource::Bible { book: "Proverbs".to_string() });

        let secondary_sources: Vec<WisdomSource> = topic_data
            .get("secondary_sources")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|s| {
                        let category = s.get("category")?.as_str()?;
                        let specific = s.get("specific")?.as_str()?;
                        Some(match category {
                            "Bible" => WisdomSource::Bible { book: specific.to_string() },
                            "Stoicism" => WisdomSource::Stoicism { author: specific.to_string() },
                            "Philosophy" => WisdomSource::Philosophy { author: specific.to_string() },
                            "Biography" => WisdomSource::Biography { subject: specific.to_string() },
                            "Psychology" => WisdomSource::Psychology { author: specific.to_string() },
                            _ => return None,
                        })
                    })
                    .collect()
            })
            .unwrap_or_default();

        let target_keywords: Vec<String> = topic_data
            .get("target_keywords")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_default();

        let scheduled_publish = exec_res
            .get("scheduled_publish")
            .and_then(|v| v.as_str())
            .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
            .map(|dt| dt.with_timezone(&chrono::Utc));

        let topic_brief = TopicBrief {
            video_id,
            topic: topic_data.get("topic").and_then(|v| v.as_str()).unwrap_or("").to_string(),
            description: topic_data.get("description").and_then(|v| v.as_str()).unwrap_or("").to_string(),
            target_duration_minutes: (self.config.target_duration_min + self.config.target_duration_max) / 2,
            primary_source,
            secondary_sources,
            target_keywords,
            hook_angle: topic_data.get("hook_angle").and_then(|v| v.as_str()).unwrap_or("").to_string(),
            scheduled_publish,
        };

        info!(
            "Strategy: Generated topic brief for video {}: {}",
            video_id,
            topic_brief.topic
        );

        // Store in shared state
        shared.insert(
            state_keys::TOPIC_BRIEF.to_string(),
            serde_json::to_value(&topic_brief).unwrap_or(serde_json::json!(null)),
        );
        shared.insert(
            state_keys::VIDEO_ID.to_string(),
            serde_json::json!(video_id.to_string()),
        );

        // Persist to database - create new video record
        let video = Video::new_production(
            video_id,
            serde_json::to_value(&topic_brief).unwrap_or(serde_json::json!(null)),
        );
        if let Err(e) = db::insert_video(&self.db_pool, &video).await {
            error!("Failed to persist video to database: {}", e);
            // Continue anyway - in-memory state still works
        } else {
            info!("Strategy: Video {} persisted to database", video_id);
        }

        // Proceed to script writer
        Some("default".to_string())
    }

    fn clone_box(&self) -> Box<dyn AsyncNodeLogic> {
        Box::new(self.clone())
    }
}
