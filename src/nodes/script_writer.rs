//! Script Writer Node
//!
//! Generates engaging 12-20 minute scripts with hooks, storytelling, and CTAs.
//! Uses LLM to create scripts optimized for watch time and engagement.
//!
//! Features a self-improvement loop:
//! 1. Generate N candidate scripts in parallel (Gemini)
//! 2. Judge each candidate (DeepSeek)
//! 3. If best score >= threshold, accept
//! 4. Otherwise, refine best candidate with feedback and re-evaluate
//! 5. Repeat until threshold met or max iterations reached

use async_trait::async_trait;
use orichalcum::{AsyncNodeLogic, NodeValue};
use orichalcum::llm::{Client, Enabled, Providers};
use serde::{Deserialize, Serialize};
use sha2::{Sha256, Digest};
use sqlx::PgPool;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::{error, info, warn};

use crate::config::ScriptImprovementConfig;
use crate::db;
use crate::nodes::{
    Script, ScriptSection, TopicBrief,
    ScriptEvaluation, CriteriaScores, CriterionScore, SpecificImprovement, ScoredScript
};
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
    pub improvement_config: ScriptImprovementConfig,
    pub llm_client: Arc<Client<Providers<orichalcum::llm::Disabled, Enabled, Enabled>>>,
    pub db_pool: Arc<PgPool>,
}

impl ScriptWriterLogic {
    pub fn new(
        config: ScriptWriterConfig,
        improvement_config: ScriptImprovementConfig,
        llm_client: Arc<Client<Providers<orichalcum::llm::Disabled, Enabled, Enabled>>>,
        db_pool: Arc<PgPool>,
    ) -> Self {
        Self { config, improvement_config, llm_client, db_pool }
    }

    // =========================================================================
    // Prompt Builders
    // =========================================================================

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

AVOID AI PATTERNS:
- Don't use "journey", "delve", "tapestry", "navigate", "it's important to note"
- Don't make every paragraph the same length
- Don't use lists of three for everything
- Don't hedge excessively ("It's worth noting that...", "One might argue...")
- Use specific, visceral examples instead of generic ones
- Include genuine uncertainty and rough edges
- Avoid too many semicolons and em-dashes

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

    /// Build the judge system prompt
    fn build_judge_system_prompt(&self) -> String {
        r#"You are a ruthless YouTube script quality evaluator. Your job is to identify weaknesses that will hurt watch time, engagement, and authenticity.

You evaluate scripts on these criteria (1-10 scale):

1. HOOK STRENGTH (15%): Does the first 30 seconds create an irresistible urge to keep watching? Does the viewer feel "this is for ME"?

2. PACING & RETENTION (15%): Does the rhythm vary? Are there retention hooks every 2-3 minutes? Any flat spots where attention drifts?

3. WISDOM INTEGRATION (15%): Are sources woven naturally into the narrative? Or just dropped as quotes? Does the wisdom feel earned?

4. AUTHENTICITY (15%): Does this sound like a real person sharing hard-won insights? Or a motivational speaker performing? Is there genuine vulnerability?

5. DURATION ACCURACY (10%): Is the word count close to target? (~150 words per minute)

6. CTA QUALITY (10%): Does the closing feel earned? Or salesy/abrupt?

7. AI DETECTION (20%): This is CRITICAL. You must identify telltale signs of AI-generated content:
   - Overused phrases: "journey", "delve", "tapestry", "navigate", "it's important to note", "in conclusion"
   - Predictable structure (every paragraph same length, every section same format)
   - Lists of three everything
   - Excessive hedging/qualifiers ("It's worth noting that...", "One might argue...")
   - Generic examples instead of specific, visceral moments
   - Overly balanced perspectives ("on one hand... on the other")
   - Lack of genuine uncertainty, rough edges, or imperfection
   - Too many semicolons and em-dashes
   - Abstract nouns where concrete imagery would work better
   - Smooth transitions everywhere (real speech has some roughness)

BE HARSH. A score of 8+ should mean "this could genuinely compete with top human-written scripts."
A score of 5-7 means "decent but clearly has issues."
Below 5 means "significant problems."

Respond in JSON format only."#.to_string()
    }

    /// Build the judge user prompt
    fn build_judge_user_prompt(&self, script: &Script, topic_brief: &TopicBrief) -> String {
        format!(r#"Evaluate this YouTube script:

TOPIC: {}
TARGET DURATION: {} minutes (~{} words)
ACTUAL WORD COUNT: {}

=== SCRIPT ===
{}
=== END SCRIPT ===

Respond with this exact JSON structure:
{{
  "overall_score": 7.2,
  "criteria": {{
    "hook_strength": {{ "score": 8.0, "notes": "..." }},
    "pacing_retention": {{ "score": 6.5, "notes": "..." }},
    "wisdom_integration": {{ "score": 7.0, "notes": "..." }},
    "authenticity": {{ "score": 7.5, "notes": "..." }},
    "duration_accuracy": {{ "score": 9.0, "notes": "..." }},
    "cta_quality": {{ "score": 6.0, "notes": "..." }},
    "ai_detection": {{ "score": 7.0, "notes": "..." }}
  }},
  "strengths": ["strength 1", "strength 2"],
  "weaknesses": ["weakness 1", "weakness 2"],
  "ai_telltale_signs": ["specific sign 1", "specific sign 2"],
  "specific_improvements": [
    {{
      "location": "Section 2, paragraph 3",
      "issue": "What's wrong",
      "suggestion": "How to fix it"
    }}
  ]
}}"#,
            topic_brief.topic,
            topic_brief.target_duration_minutes,
            topic_brief.target_duration_minutes * 150,
            script.full_text.split_whitespace().count(),
            script.full_text
        )
    }

    /// Build the refinement prompt
    fn build_refinement_prompt(
        &self,
        script: &Script,
        topic_brief: &TopicBrief,
        evaluation: &ScriptEvaluation,
        force_dramatic_changes: bool,
    ) -> String {
        let improvements = evaluation.specific_improvements
            .iter()
            .map(|i| format!("- {}: {} → {}", i.location, i.issue, i.suggestion))
            .collect::<Vec<_>>()
            .join("\n");

        let ai_fixes = evaluation.ai_telltale_signs
            .iter()
            .map(|s| format!("- {}", s))
            .collect::<Vec<_>>()
            .join("\n");

        let target_words = topic_brief.target_duration_minutes * self.config.words_per_minute;

        let dramatic_instruction = if force_dramatic_changes {
            r#"
⚠️ CRITICAL: Previous refinements have NOT improved the score. You MUST make DRAMATIC changes:
- COMPLETELY REWRITE weak sections from scratch - do not just tweak words
- Use a DIFFERENT narrative structure (if using chronological, try thematic; if using problem-solution, try story-based)
- CHANGE the hook entirely - new angle, new opening line, new emotional appeal
- REPLACE at least 50% of the examples and anecdotes with completely different ones
- VARY sentence patterns dramatically - if you've been using short punchy sentences, use longer flowing ones (or vice versa)
- ADD unexpected elements: rhetorical questions, direct challenges to the viewer, moments of silence/pause
- BREAK conventional patterns: start a section with the conclusion, use a single powerful word as a transition

DO NOT make superficial changes. The evaluator has seen similar variations and scored them the same.
You need to take creative risks and try something genuinely different.
"#
        } else {
            ""
        };

        format!(r#"Revise this script based on the feedback below.
{dramatic_instruction}
CRITICAL: Maintain what's working (the strengths) while fixing the weaknesses.

=== CURRENT SCRIPT ===
{script_text}
=== END SCRIPT ===

=== FEEDBACK ===

STRENGTHS TO PRESERVE:
{strengths}

WEAKNESSES TO FIX:
{weaknesses}

AI PATTERNS TO ELIMINATE:
{ai_fixes}

SPECIFIC CHANGES REQUIRED:
{improvements}

=== END FEEDBACK ===

Return the complete revised script in the same JSON format:
{{
    "hook": {{
        "title": "Hook",
        "narration": "...",
        "duration_seconds": 30,
        "visual_suggestions": ["..."]
    }},
    "sections": [
        {{
            "title": "Section Title",
            "narration": "...",
            "duration_seconds": 180,
            "visual_suggestions": ["..."]
        }}
    ],
    "cta": {{
        "title": "Call to Action",
        "narration": "...",
        "duration_seconds": 30,
        "visual_suggestions": ["..."]
    }}
}}

IMPORTANT:
- Target approximately {target_words} words total
- Do NOT explain your changes. Just return the improved script JSON."#,
            dramatic_instruction = dramatic_instruction,
            script_text = script.full_text,
            strengths = evaluation.strengths.iter().map(|s| format!("- {}", s)).collect::<Vec<_>>().join("\n"),
            weaknesses = evaluation.weaknesses.iter().map(|s| format!("- {}", s)).collect::<Vec<_>>().join("\n"),
            ai_fixes = ai_fixes,
            improvements = improvements,
            target_words = target_words
        )
    }

    // =========================================================================
    // Core Methods
    // =========================================================================

    /// Calculate SHA256 hash of script content
    fn script_hash(script: &Script) -> String {
        let mut hasher = Sha256::new();
        hasher.update(script.full_text.as_bytes());
        format!("{:x}", hasher.finalize())
    }

    /// Parse a script from JSON response
    fn parse_script(&self, json_str: &str, video_id: uuid::Uuid) -> Result<Script, String> {
        let cleaned = json_str
            .trim()
            .trim_start_matches("```json")
            .trim_start_matches("```")
            .trim_end_matches("```")
            .trim();

        let parsed: serde_json::Value = serde_json::from_str(cleaned)
            .map_err(|e| format!("Failed to parse script JSON: {}", e))?;

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

        let hook = parsed.get("hook")
            .and_then(parse_section)
            .unwrap_or_else(|| ScriptSection {
                title: "Hook".to_string(),
                narration: String::new(),
                duration_seconds: 30,
                visual_suggestions: vec![],
            });

        let sections: Vec<ScriptSection> = parsed.get("sections")
            .and_then(|v| v.as_array())
            .map(|arr| arr.iter().filter_map(parse_section).collect())
            .unwrap_or_default();

        let cta = parsed.get("cta")
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

        Ok(Script {
            video_id,
            hook,
            sections,
            cta,
            total_duration_seconds: total_duration,
            full_text,
        })
    }

    /// Parse evaluation from JSON response
    fn parse_evaluation(&self, json_str: &str) -> Result<ScriptEvaluation, String> {
        let cleaned = json_str
            .trim()
            .trim_start_matches("```json")
            .trim_start_matches("```")
            .trim_end_matches("```")
            .trim();

        let parsed: serde_json::Value = serde_json::from_str(cleaned)
            .map_err(|e| format!("Failed to parse evaluation JSON: {}", e))?;

        let parse_criterion = |name: &str| -> CriterionScore {
            parsed.get("criteria")
                .and_then(|c| c.get(name))
                .map(|v| CriterionScore {
                    score: v.get("score").and_then(|s| s.as_f64()).unwrap_or(5.0) as f32,
                    notes: v.get("notes").and_then(|s| s.as_str()).unwrap_or("").to_string(),
                })
                .unwrap_or(CriterionScore { score: 5.0, notes: String::new() })
        };

        let criteria = CriteriaScores {
            hook_strength: parse_criterion("hook_strength"),
            pacing_retention: parse_criterion("pacing_retention"),
            wisdom_integration: parse_criterion("wisdom_integration"),
            authenticity: parse_criterion("authenticity"),
            duration_accuracy: parse_criterion("duration_accuracy"),
            cta_quality: parse_criterion("cta_quality"),
            ai_detection: parse_criterion("ai_detection"),
        };

        let parse_string_array = |key: &str| -> Vec<String> {
            parsed.get(key)
                .and_then(|v| v.as_array())
                .map(|arr| arr.iter().filter_map(|s| s.as_str().map(|s| s.to_string())).collect())
                .unwrap_or_default()
        };

        let specific_improvements: Vec<SpecificImprovement> = parsed.get("specific_improvements")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter().filter_map(|item| {
                    Some(SpecificImprovement {
                        location: item.get("location")?.as_str()?.to_string(),
                        issue: item.get("issue")?.as_str()?.to_string(),
                        suggestion: item.get("suggestion")?.as_str()?.to_string(),
                    })
                }).collect()
            })
            .unwrap_or_default();

        Ok(ScriptEvaluation {
            overall_score: parsed.get("overall_score").and_then(|s| s.as_f64()).unwrap_or(5.0) as f32,
            criteria,
            strengths: parse_string_array("strengths"),
            weaknesses: parse_string_array("weaknesses"),
            ai_telltale_signs: parse_string_array("ai_telltale_signs"),
            specific_improvements,
        })
    }

    /// Generate a single script using Gemini
    async fn generate_script(&self, topic_brief: &TopicBrief) -> Result<Script, String> {
        let system_prompt = self.build_system_prompt();
        let user_prompt = self.build_user_prompt(topic_brief);

        let response = self.llm_client.gemini_complete(
            "gemini-3-flash-preview",
            &system_prompt,
            &user_prompt,
            Some(0.8), // Higher temperature for diversity
            Some(8000),
        ).await.map_err(|e| format!("Gemini call failed: {}", e))?;

        self.parse_script(&response, topic_brief.video_id)
    }

    /// Evaluate a script using DeepSeek
    async fn evaluate_script(&self, script: &Script, topic_brief: &TopicBrief) -> Result<ScriptEvaluation, String> {
        let system_prompt = self.build_judge_system_prompt();
        let user_prompt = self.build_judge_user_prompt(script, topic_brief);

        let response = self.llm_client.deepseek_complete(
            "deepseek-chat",
            &system_prompt,
            &user_prompt,
            Some(0.3), // Lower temperature for consistency
            Some(2000),
        ).await.map_err(|e| format!("DeepSeek call failed: {}", e))?;

        self.parse_evaluation(&response)
    }

    /// Refine a script based on feedback using Gemini
    async fn refine_script(
        &self,
        script: &Script,
        topic_brief: &TopicBrief,
        evaluation: &ScriptEvaluation,
        temperature: f32,
        force_dramatic_changes: bool,
    ) -> Result<Script, String> {
        let system_prompt = self.build_system_prompt();
        let user_prompt = self.build_refinement_prompt(script, topic_brief, evaluation, force_dramatic_changes);

        let response = self.llm_client.gemini_complete(
            "gemini-3-flash-preview",
            &system_prompt,
            &user_prompt,
            Some(temperature),
            Some(8000),
        ).await.map_err(|e| format!("Gemini refinement failed: {}", e))?;

        self.parse_script(&response, topic_brief.video_id)
    }

    /// Persist evaluation to database
    async fn persist_evaluation(
        &self,
        video_id: uuid::Uuid,
        iteration: i32,
        candidate_index: Option<i32>,
        script: &Script,
        evaluation: &ScriptEvaluation,
    ) -> Option<i32> {
        let script_hash = Self::script_hash(script);
        let criteria_json = serde_json::to_value(&evaluation.criteria).unwrap_or(serde_json::json!({}));
        let improvements_json = serde_json::to_value(&evaluation.specific_improvements).unwrap_or(serde_json::json!([]));
        let script_json = serde_json::to_value(script).ok();

        match db::insert_script_evaluation(
            &self.db_pool,
            video_id,
            iteration,
            candidate_index,
            &script_hash,
            evaluation.overall_score,
            criteria_json,
            &evaluation.strengths,
            &evaluation.weaknesses,
            &evaluation.ai_telltale_signs,
            improvements_json,
            script_json,
        ).await {
            Ok(id) => Some(id),
            Err(e) => {
                warn!("Failed to persist evaluation: {}", e);
                None
            }
        }
    }

    /// Run the self-improvement loop
    async fn generate_with_improvement(&self, topic_brief: &TopicBrief) -> Result<Script, String> {
        let config = &self.improvement_config;
        let deadline = Instant::now() + Duration::from_secs(config.timeout_seconds as u64);
        let video_id = topic_brief.video_id;

        info!(
            "ScriptWriter: Starting self-improvement loop (candidates={}, threshold={}, max_iter={})",
            config.candidate_count, config.quality_threshold, config.max_iterations
        );

        // Phase 1: Generate initial candidates
        info!("ScriptWriter: Generating {} candidates...", config.candidate_count);
        let mut candidates: Vec<Script> = Vec::new();

        for i in 0..config.candidate_count {
            if Instant::now() > deadline {
                warn!("ScriptWriter: Timeout during candidate generation");
                break;
            }

            match self.generate_script(topic_brief).await {
                Ok(script) => {
                    info!("ScriptWriter: Candidate {} generated ({} words)", i + 1, script.full_text.split_whitespace().count());
                    candidates.push(script);
                }
                Err(e) => {
                    warn!("ScriptWriter: Failed to generate candidate {}: {}", i + 1, e);
                }
            }
        }

        if candidates.is_empty() {
            return Err("All script candidates failed to generate".to_string());
        }

        // Phase 2: Evaluate all candidates
        info!("ScriptWriter: Evaluating {} candidates...", candidates.len());
        let mut best: Option<ScoredScript> = None;

        for (idx, script) in candidates.iter().enumerate() {
            if Instant::now() > deadline {
                warn!("ScriptWriter: Timeout during evaluation");
                break;
            }

            match self.evaluate_script(script, topic_brief).await {
                Ok(evaluation) => {
                    let eval_id = self.persist_evaluation(
                        video_id,
                        0,
                        Some(idx as i32),
                        script,
                        &evaluation,
                    ).await;

                    info!(
                        "ScriptWriter: Candidate {} scored {:.1}",
                        idx + 1, evaluation.overall_score
                    );

                    let scored = ScoredScript {
                        script: script.clone(),
                        evaluation,
                        iteration: 0,
                        candidate_index: Some(idx as u32),
                        evaluation_id: eval_id,
                    };

                    if best.is_none() || scored.evaluation.overall_score > best.as_ref().unwrap().evaluation.overall_score {
                        best = Some(scored);
                    }
                }
                Err(e) => {
                    warn!("ScriptWriter: Failed to evaluate candidate {}: {}", idx + 1, e);
                }
            }
        }

        let mut best = best.ok_or("No candidates could be evaluated")?;
        info!(
            "ScriptWriter: Best initial score: {:.1} (threshold: {:.1})",
            best.evaluation.overall_score, config.quality_threshold
        );

        // Phase 3: Refinement loop with stagnation detection
        let mut iteration = 0u32;
        let mut stagnant_iterations = 0u32;
        const STAGNATION_THRESHOLD: u32 = 3; // Break if no improvement for 3 iterations
        const BASE_TEMPERATURE: f32 = 0.6;
        const MAX_TEMPERATURE: f32 = 1.0;
        
        while best.evaluation.overall_score < config.quality_threshold
            && iteration < config.max_iterations
            && Instant::now() < deadline
            && stagnant_iterations < STAGNATION_THRESHOLD
        {
            iteration += 1;
            
            // Escalate temperature based on stagnation (0.6 -> 0.7 -> 0.8 -> 0.9 -> 1.0)
            let temperature = (BASE_TEMPERATURE + (stagnant_iterations as f32 * 0.1)).min(MAX_TEMPERATURE);
            let force_dramatic = stagnant_iterations >= 2;
            
            info!(
                "ScriptWriter: Refinement {}/{} (score: {:.1}, temp: {:.1}, stagnant: {})",
                iteration, config.max_iterations, best.evaluation.overall_score, temperature, stagnant_iterations
            );

            // Log top issues being addressed
            if !best.evaluation.weaknesses.is_empty() {
                info!("  Addressing: {}", best.evaluation.weaknesses.first().unwrap_or(&String::new()));
            }
            if !best.evaluation.ai_telltale_signs.is_empty() {
                info!("  AI pattern: {}", best.evaluation.ai_telltale_signs.first().unwrap_or(&String::new()));
            }
            if force_dramatic {
                info!("  Mode: DRAMATIC CHANGES (stagnation detected)");
            }

            // Refine based on feedback
            match self.refine_script(&best.script, topic_brief, &best.evaluation, temperature, force_dramatic).await {
                Ok(refined_script) => {
                    // Evaluate refined script
                    match self.evaluate_script(&refined_script, topic_brief).await {
                        Ok(evaluation) => {
                            let eval_id = self.persist_evaluation(
                                video_id,
                                iteration as i32,
                                None,
                                &refined_script,
                                &evaluation,
                            ).await;

                            // Only accept if it's actually better
                            if evaluation.overall_score > best.evaluation.overall_score {
                                info!(
                                    "ScriptWriter: Refinement improved {:.1} -> {:.1}",
                                    best.evaluation.overall_score, evaluation.overall_score
                                );
                                best = ScoredScript {
                                    script: refined_script,
                                    evaluation,
                                    iteration,
                                    candidate_index: None,
                                    evaluation_id: eval_id,
                                };
                                // Reset stagnation counter on improvement
                                stagnant_iterations = 0;
                            } else {
                                info!(
                                    "ScriptWriter: Refinement did not improve ({:.1} vs {:.1})",
                                    evaluation.overall_score, best.evaluation.overall_score
                                );
                                stagnant_iterations += 1;
                            }
                        }
                        Err(e) => {
                            warn!("ScriptWriter: Failed to evaluate refinement: {}", e);
                            stagnant_iterations += 1;
                        }
                    }
                }
                Err(e) => {
                    warn!("ScriptWriter: Failed to refine: {}", e);
                    stagnant_iterations += 1;
                }
            }
        }

        // Log reason for stopping
        if stagnant_iterations >= STAGNATION_THRESHOLD {
            warn!(
                "ScriptWriter: Stopping refinement due to stagnation ({} iterations without improvement)",
                stagnant_iterations
            );
        }

        // Mark final as selected
        if let Some(eval_id) = best.evaluation_id {
            if let Err(e) = db::mark_evaluation_selected(&self.db_pool, eval_id).await {
                warn!("ScriptWriter: Failed to mark evaluation as selected: {}", e);
            }
        }

        // Log final result
        if best.evaluation.overall_score >= config.quality_threshold {
            info!(
                "ScriptWriter: Accepted script with score {:.1} after {} iterations",
                best.evaluation.overall_score, iteration
            );
        } else {
            warn!(
                "ScriptWriter: Accepting below-threshold script ({:.1}) after {} iterations",
                best.evaluation.overall_score, iteration
            );
        }

        Ok(best.script)
    }

    /// Simple single-shot generation (when improvement loop is disabled)
    async fn generate_single(&self, topic_brief: &TopicBrief) -> Result<Script, String> {
        info!("ScriptWriter: Single-shot generation (improvement loop disabled)");
        self.generate_script(topic_brief).await
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

        info!("ScriptWriter: Generating script for '{}'", topic_brief.topic);

        // Check if improvement loop is enabled
        let improvement_enabled = self.improvement_config.enabled
            && std::env::var("SCRIPT_IMPROVEMENT_ENABLED")
                .map(|v| v.to_lowercase() != "false" && v != "0")
                .unwrap_or(true);

        let result = if improvement_enabled {
            self.generate_with_improvement(&topic_brief).await
        } else {
            self.generate_single(&topic_brief).await
        };

        match result {
            Ok(script) => {
                serde_json::json!({
                    "success": true,
                    "script": serde_json::to_value(&script).unwrap_or(serde_json::json!(null)),
                    "video_id": topic_brief.video_id.to_string()
                })
            }
            Err(e) => {
                error!("ScriptWriter failed: {}", e);
                serde_json::json!({
                    "error": e
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

        // Get the script from exec result
        let script: Script = match exec_res.get("script") {
            Some(s) => match serde_json::from_value(s.clone()) {
                Ok(script) => script,
                Err(e) => {
                    error!("Failed to parse script from exec result: {}", e);
                    return Some("error".to_string());
                }
            },
            None => {
                error!("No script in exec result");
                return Some("error".to_string());
            }
        };

        let video_id = exec_res
            .get("video_id")
            .and_then(|v| v.as_str())
            .and_then(|s| uuid::Uuid::parse_str(s).ok())
            .unwrap_or_else(uuid::Uuid::new_v4);

        info!(
            "ScriptWriter: Script ready for video {} ({} seconds, {} words)",
            video_id,
            script.total_duration_seconds,
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
