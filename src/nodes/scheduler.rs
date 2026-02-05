//! Scheduler Node
//!
//! Manages the content calendar and triggers video production at appropriate times.
//! Supports seed topics via environment variables or database queue.
//! Runs on a timer and decides when to produce the next video.

use async_trait::async_trait;
use chrono::{DateTime, Datelike, Utc, Weekday};
use orichalcum::{AsyncNodeLogic, NodeValue};
use serde::{Deserialize, Serialize};
use sqlx::PgPool;
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{info, warn};
use uuid::Uuid;

use crate::db;
use crate::state_keys;

/// Valid wisdom source focus values
const VALID_SOURCES: [&str; 5] = ["Bible", "Stoicism", "Philosophy", "Biography", "Psychology"];

/// Normalize a source focus string (case-insensitive matching)
/// Returns Some(normalized) if valid, None if invalid
fn normalize_source_focus(input: &str) -> Option<String> {
    let trimmed = input.trim();
    VALID_SOURCES
        .iter()
        .find(|&s| s.eq_ignore_ascii_case(trimmed))
        .map(|s| s.to_string())
}

/// Configuration for the scheduler
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerConfig {
    /// Target number of videos per week
    pub videos_per_week: u32,
    /// Preferred publish days (0 = Sunday, 6 = Saturday)
    pub preferred_days: Vec<u32>,
    /// Preferred publish hour (0-23, in UTC)
    pub preferred_hour_utc: u32,
    /// Minimum hours between videos
    pub min_hours_between: u32,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            videos_per_week: 4,
            // Tuesday, Thursday, Saturday, Sunday
            preferred_days: vec![0, 2, 4, 6],
            preferred_hour_utc: 14, // 2 PM UTC
            min_hours_between: 36,
        }
    }
}

/// The scheduler node logic
#[derive(Clone)]
pub struct SchedulerLogic {
    pub config: SchedulerConfig,
    pub db_pool: Arc<PgPool>,
}

impl SchedulerLogic {
    pub fn new(config: SchedulerConfig, db_pool: Arc<PgPool>) -> Self {
        Self { config, db_pool }
    }

    /// Build the preferred weekdays list
    fn get_preferred_weekdays(&self) -> Vec<Weekday> {
        self.config
            .preferred_days
            .iter()
            .filter_map(|&d| match d {
                0 => Some(Weekday::Sun),
                1 => Some(Weekday::Mon),
                2 => Some(Weekday::Tue),
                3 => Some(Weekday::Wed),
                4 => Some(Weekday::Thu),
                5 => Some(Weekday::Fri),
                6 => Some(Weekday::Sat),
                _ => None,
            })
            .collect()
    }

    /// Calculate the next optimal publish time
    /// Fixed: Now considers TODAY if it's a preferred day and before preferred hour
    fn calculate_next_publish_time(
        &self,
        now: DateTime<Utc>,
        last_publish: Option<DateTime<Utc>>,
    ) -> DateTime<Utc> {
        use chrono::Duration;

        let preferred_weekdays = self.get_preferred_weekdays();

        // Check if today is a preferred day and we haven't passed the preferred hour
        let today_at_preferred = now
            .date_naive()
            .and_hms_opt(self.config.preferred_hour_utc, 0, 0)
            .unwrap()
            .and_utc();

        let mut candidate = if preferred_weekdays.contains(&now.weekday()) && now < today_at_preferred {
            // Today is a preferred day and we haven't passed the publish time yet
            today_at_preferred
        } else {
            // Start from tomorrow at the preferred hour
            now.date_naive()
                .succ_opt()
                .unwrap()
                .and_hms_opt(self.config.preferred_hour_utc, 0, 0)
                .unwrap()
                .and_utc()
        };

        // If we have a last publish time, ensure minimum gap
        if let Some(last) = last_publish {
            let min_next = last + Duration::hours(self.config.min_hours_between as i64);
            if candidate < min_next {
                candidate = min_next;
            }
        }

        // Advance to the next preferred day if needed
        for _ in 0..7 {
            if preferred_weekdays.contains(&candidate.weekday()) {
                break;
            }
            candidate = candidate + Duration::days(1);
        }

        candidate
    }

    /// Determine the wisdom source focus area for rotation
    fn select_source_focus(&self, video_number: u32) -> String {
        VALID_SOURCES[(video_number as usize) % VALID_SOURCES.len()].to_string()
    }
}

#[async_trait]
impl AsyncNodeLogic for SchedulerLogic {
    async fn prep(
        &self,
        _params: &HashMap<String, NodeValue>,
        shared: &HashMap<String, NodeValue>,
    ) -> NodeValue {
        // Check for existing production in progress
        let in_progress = shared
            .get("production_in_progress")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        // Get last publish time if available
        let last_publish: Option<DateTime<Utc>> = shared
            .get("last_publish_time")
            .and_then(|v| v.as_str())
            .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
            .map(|dt| dt.with_timezone(&Utc));

        // Get video count for source rotation
        let video_count: u32 = shared
            .get("total_video_count")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as u32;

        // Get seed topic from shared state (injected by main.rs from env var)
        let seed_topic = shared
            .get("seed_topic")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        // Get source focus override from shared state (injected by main.rs from env var)
        let source_focus_override = shared
            .get("source_focus_override")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        serde_json::json!({
            "in_progress": in_progress,
            "last_publish": last_publish.map(|dt| dt.to_rfc3339()),
            "video_count": video_count,
            "seed_topic": seed_topic,
            "source_focus_override": source_focus_override,
        })
    }

    async fn exec(&self, input: NodeValue) -> NodeValue {
        let in_progress = input
            .get("in_progress")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let video_count = input
            .get("video_count")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as u32;

        let seed_topic = input
            .get("seed_topic")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        let source_focus_override = input
            .get("source_focus_override")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        // Validate source_focus_override if provided
        let normalized_source = if let Some(ref override_str) = source_focus_override {
            match normalize_source_focus(override_str) {
                Some(normalized) => Some(normalized),
                None => {
                    warn!(
                        "Invalid SOURCE_FOCUS '{}'. Valid options: Bible, Stoicism, Philosophy, Biography, Psychology",
                        override_str
                    );
                    // Return wait action - don't proceed with invalid input
                    return serde_json::json!({
                        "should_produce": false,
                        "reason": format!(
                            "Invalid SOURCE_FOCUS '{}'. Valid options: {}",
                            override_str,
                            VALID_SOURCES.join(", ")
                        ),
                        "consume_inputs": true  // Still consume to avoid infinite loop
                    });
                }
            }
        } else {
            None
        };

        // CASE 1: We have a seed topic
        if let Some(ref topic) = seed_topic {
            if in_progress {
                // Production in progress - queue the seed for later
                let source_ref = normalized_source.as_deref();
                match db::queue_seed(&self.db_pool, topic, source_ref).await {
                    Ok(id) => {
                        let queue_len = db::get_queue_length(&self.db_pool).await.unwrap_or(0);
                        info!(
                            "Scheduler: Queued seed topic (id={}) for after current production. {} item(s) in queue",
                            id, queue_len
                        );
                    }
                    Err(e) => {
                        warn!("Scheduler: Failed to queue seed topic: {}", e);
                    }
                }
                return serde_json::json!({
                    "should_produce": false,
                    "reason": "queued",
                    "consume_inputs": true
                });
            }

            // Not in progress - use the seed topic immediately
            let source_focus = normalized_source
                .unwrap_or_else(|| self.select_source_focus(video_count));

            info!(
                "Scheduler: Using seed topic with source focus '{}'",
                source_focus
            );

            return serde_json::json!({
                "should_produce": true,
                "video_id": Uuid::new_v4().to_string(),
                "scheduled_publish": Utc::now().to_rfc3339(),
                "source_focus": source_focus,
                "seed_topic": topic,
                "video_number": video_count + 1,
                "hours_until_publish": 0,
                "consume_inputs": true
            });
        }

        // CASE 2: No seed topic from env, but check the database queue
        if !in_progress {
            if let Ok(Some((id, queued_topic, queued_source))) = db::pop_seed(&self.db_pool).await {
                let source_focus = queued_source
                    .unwrap_or_else(|| self.select_source_focus(video_count));

                info!(
                    "Scheduler: Using queued seed topic (id={}) with source focus '{}'",
                    id, source_focus
                );

                return serde_json::json!({
                    "should_produce": true,
                    "video_id": Uuid::new_v4().to_string(),
                    "scheduled_publish": Utc::now().to_rfc3339(),
                    "source_focus": source_focus,
                    "seed_topic": queued_topic,
                    "video_number": video_count + 1,
                    "hours_until_publish": 0,
                    "consume_inputs": false  // Already consumed from DB
                });
            }
        }

        // CASE 3: Production in progress, no seed topic
        if in_progress {
            return serde_json::json!({
                "should_produce": false,
                "reason": "Production already in progress"
            });
        }

        // CASE 4: Normal scheduling based on timing
        let last_publish: Option<DateTime<Utc>> = input
            .get("last_publish")
            .and_then(|v| v.as_str())
            .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
            .map(|dt| dt.with_timezone(&Utc));

        let now = Utc::now();
        let next_publish = self.calculate_next_publish_time(now, last_publish);
        let source_focus = self.select_source_focus(video_count);

        // Calculate hours until next publish
        let hours_until = (next_publish - now).num_hours();

        // We should start production if there's enough time
        // Assume production takes ~24 hours to allow for review
        let production_lead_time_hours = 24;
        let should_produce = hours_until <= production_lead_time_hours + 12;

        serde_json::json!({
            "should_produce": should_produce,
            "video_id": Uuid::new_v4().to_string(),
            "scheduled_publish": next_publish.to_rfc3339(),
            "source_focus": source_focus,
            "video_number": video_count + 1,
            "hours_until_publish": hours_until
        })
    }

    async fn post(
        &self,
        shared: &mut HashMap<String, NodeValue>,
        _prep_res: NodeValue,
        exec_res: NodeValue,
    ) -> Option<String> {
        // Consume inputs if flagged (removes env-based seed_topic and source_focus_override)
        if exec_res
            .get("consume_inputs")
            .and_then(|v| v.as_bool())
            .unwrap_or(false)
        {
            shared.remove("seed_topic");
            shared.remove("source_focus_override");
        }

        let should_produce = exec_res
            .get("should_produce")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        if !should_produce {
            let reason = exec_res
                .get("reason")
                .and_then(|v| v.as_str())
                .unwrap_or("No production needed");
            info!("Scheduler: {}", reason);
            return Some("wait".to_string());
        }

        // Set up production context in shared state
        if let Some(video_id) = exec_res.get("video_id").and_then(|v| v.as_str()) {
            shared.insert(
                state_keys::VIDEO_ID.to_string(),
                serde_json::json!(video_id),
            );
        }

        if let Some(scheduled) = exec_res.get("scheduled_publish") {
            shared.insert("scheduled_publish".to_string(), scheduled.clone());
        }

        if let Some(source_focus) = exec_res.get("source_focus") {
            shared.insert("source_focus".to_string(), source_focus.clone());
        }

        // Pass through seed_topic if present (for Strategy node to use)
        if let Some(seed_topic) = exec_res.get("seed_topic") {
            shared.insert("seed_topic".to_string(), seed_topic.clone());
        }

        shared.insert(
            "production_in_progress".to_string(),
            serde_json::json!(true),
        );

        info!(
            "Scheduler: Starting production for video {}, scheduled for {}",
            exec_res
                .get("video_id")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown"),
            exec_res
                .get("scheduled_publish")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown")
        );

        // Proceed to strategy node
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
    fn test_scheduler_config_default() {
        let config = SchedulerConfig::default();
        assert_eq!(config.videos_per_week, 4);
        assert_eq!(config.preferred_days.len(), 4);
    }

    #[test]
    fn test_source_rotation() {
        // Note: Can't easily test SchedulerLogic without a db_pool
        // Just test the VALID_SOURCES constant
        assert_eq!(VALID_SOURCES.len(), 5);
        assert!(VALID_SOURCES.contains(&"Bible"));
        assert!(VALID_SOURCES.contains(&"Stoicism"));
    }

    #[test]
    fn test_normalize_source_focus() {
        // Valid inputs (case insensitive)
        assert_eq!(normalize_source_focus("Bible"), Some("Bible".to_string()));
        assert_eq!(normalize_source_focus("bible"), Some("Bible".to_string()));
        assert_eq!(normalize_source_focus("STOICISM"), Some("Stoicism".to_string()));
        assert_eq!(normalize_source_focus("  Philosophy  "), Some("Philosophy".to_string()));

        // Invalid inputs
        assert_eq!(normalize_source_focus("Christianity"), None);
        assert_eq!(normalize_source_focus("invalid"), None);
        assert_eq!(normalize_source_focus(""), None);
    }
}
