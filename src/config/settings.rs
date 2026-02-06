//! Application settings loaded from environment variables

use serde::Deserialize;

/// Root configuration structure
#[derive(Debug, Clone, Deserialize)]
pub struct Settings {
    pub database: DatabaseConfig,
    pub s3: S3Config,
    pub llm: LlmConfig,
    pub tts: TtsConfig,
    pub assets: AssetConfig,
    pub youtube: YouTubeConfig,
    pub channel: ChannelConfig,
    pub content_strategy: ContentStrategyConfig,
    pub script_improvement: ScriptImprovementConfig,
    pub notifications: NotificationConfig,
    pub control_api_port: u16,
}

use crate::nodes::tts::TTSProvider;

#[derive(Debug, Clone, Deserialize)]
pub struct DatabaseConfig {
    pub url: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct S3Config {
    pub endpoint: String,
    pub access_key: String,
    pub secret_key: String,
    pub bucket: String,
    pub region: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct LlmConfig {
    pub deepseek_api_key: String,
    pub deepseek_base_url: String,
    pub gemini_api_key: String,
    /// Which provider to use by default: "deepseek" or "gemini"
    pub default_provider: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TtsConfig {
    /// TTS provider: elevenlabs, qwen3, openai, coqui, piper
    pub provider: TTSProvider,
    
    // ElevenLabs settings
    pub elevenlabs_api_key: Option<String>,
    pub elevenlabs_voice_id: Option<String>,
    pub elevenlabs_model_id: Option<String>,
    
    // Qwen3-TTS settings (self-hosted, OpenAI-compatible)
    pub qwen3_api_url: Option<String>,
    pub qwen3_api_key: Option<String>,
    pub qwen3_voice: Option<String>,
    
    // OpenAI TTS settings
    pub openai_api_key: Option<String>,
    pub openai_voice: Option<String>,
    pub openai_model: Option<String>,
    
    // Local TTS settings (Coqui/Piper)
    pub local_model_path: Option<String>,
    pub local_speaker_id: Option<String>,
    
    // Common settings
    pub stability: f32,
    pub similarity_boost: f32,
    pub speed: f32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct YouTubeConfig {
    pub client_id: String,
    pub client_secret: String,
    pub refresh_token: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ChannelConfig {
    pub name: String,
    pub tagline: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ContentStrategyConfig {
    pub target_duration_min: u32,
    pub target_duration_max: u32,
    pub videos_per_week: u32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct NotificationConfig {
    pub discord_webhook_url: Option<String>,
    pub smtp_host: Option<String>,
    pub smtp_port: Option<u16>,
    pub smtp_user: Option<String>,
    pub smtp_pass: Option<String>,
    pub notification_email: Option<String>,
}

/// Pexels and Stable Diffusion asset config
#[derive(Debug, Clone, Deserialize)]
pub struct AssetConfig {
    pub pexels_api_key: String,
    pub sd_api_url: Option<String>,
    pub sd_api_key: Option<String>,
}

/// Script self-improvement loop configuration
#[derive(Debug, Clone, Deserialize)]
pub struct ScriptImprovementConfig {
    /// Enable the self-improvement loop
    pub enabled: bool,
    /// Number of initial script candidates to generate in parallel
    pub candidate_count: u32,
    /// Minimum score (1.0-10.0) to accept without refinement
    pub quality_threshold: f32,
    /// Maximum refinement iterations before accepting best available
    pub max_iterations: u32,
    /// Total timeout for the entire self-improvement process (seconds)
    pub timeout_seconds: u32,
}

impl Settings {
    /// Load settings from environment variables
    pub fn from_env() -> Result<Self, config::ConfigError> {
        dotenvy::dotenv().ok();

        let settings = config::Config::builder()
            // Database
            .set_default("database.url", "")?
            .set_override_option(
                "database.url",
                std::env::var("DATABASE_URL").ok(),
            )?
            // S3
            .set_default("s3.endpoint", "http://localhost:9000")?
            .set_override_option("s3.endpoint", std::env::var("S3_ENDPOINT").ok())?
            .set_default("s3.access_key", "minioadmin")?
            .set_override_option("s3.access_key", std::env::var("S3_ACCESS_KEY").ok())?
            .set_default("s3.secret_key", "minioadmin")?
            .set_override_option("s3.secret_key", std::env::var("S3_SECRET_KEY").ok())?
            .set_default("s3.bucket", "animus-assets")?
            .set_override_option("s3.bucket", std::env::var("S3_BUCKET").ok())?
            .set_default("s3.region", "us-east-1")?
            .set_override_option("s3.region", std::env::var("S3_REGION").ok())?
            // LLM
            .set_default("llm.deepseek_api_key", "")?
            .set_override_option(
                "llm.deepseek_api_key",
                std::env::var("DEEPSEEK_API_KEY").ok(),
            )?
            .set_default("llm.deepseek_base_url", "https://api.deepseek.com")?
            .set_override_option(
                "llm.deepseek_base_url",
                std::env::var("DEEPSEEK_BASE_URL").ok(),
            )?
            .set_default("llm.gemini_api_key", "")?
            .set_override_option("llm.gemini_api_key", std::env::var("GEMINI_API_KEY").ok())?
            .set_default("llm.default_provider", "deepseek")?
            .set_override_option(
                "llm.default_provider",
                std::env::var("LLM_DEFAULT_PROVIDER").ok(),
            )?
            // TTS
            .set_default("tts.provider", "elevenlabs")?
            .set_override_option("tts.provider", std::env::var("TTS_PROVIDER").ok())?
            // ElevenLabs
            .set_default("tts.elevenlabs_api_key", Option::<String>::None)?
            .set_override_option(
                "tts.elevenlabs_api_key",
                std::env::var("ELEVENLABS_API_KEY").ok(),
            )?
            .set_default("tts.elevenlabs_voice_id", Option::<String>::None)?
            .set_override_option(
                "tts.elevenlabs_voice_id",
                std::env::var("ELEVENLABS_VOICE_ID").ok(),
            )?
            .set_default("tts.elevenlabs_model_id", Option::<String>::None)?
            .set_override_option(
                "tts.elevenlabs_model_id",
                std::env::var("ELEVENLABS_MODEL_ID").ok(),
            )?
            // Qwen3-TTS
            .set_default("tts.qwen3_api_url", Option::<String>::None)?
            .set_override_option("tts.qwen3_api_url", std::env::var("QWEN3_API_URL").ok())?
            .set_default("tts.qwen3_api_key", Option::<String>::None)?
            .set_override_option("tts.qwen3_api_key", std::env::var("QWEN3_API_KEY").ok())?
            .set_default("tts.qwen3_voice", Option::<String>::None)?
            .set_override_option("tts.qwen3_voice", std::env::var("QWEN3_VOICE").ok())?
            // OpenAI TTS
            .set_default("tts.openai_api_key", Option::<String>::None)?
            .set_override_option("tts.openai_api_key", std::env::var("OPENAI_TTS_API_KEY").ok())?
            .set_default("tts.openai_voice", Option::<String>::None)?
            .set_override_option("tts.openai_voice", std::env::var("OPENAI_TTS_VOICE").ok())?
            .set_default("tts.openai_model", Option::<String>::None)?
            .set_override_option("tts.openai_model", std::env::var("OPENAI_TTS_MODEL").ok())?
            // Local TTS
            .set_default("tts.local_model_path", Option::<String>::None)?
            .set_override_option("tts.local_model_path", std::env::var("LOCAL_TTS_MODEL_PATH").ok())?
            .set_default("tts.local_speaker_id", Option::<String>::None)?
            .set_override_option("tts.local_speaker_id", std::env::var("LOCAL_TTS_SPEAKER_ID").ok())?
            // Common TTS settings
            .set_default("tts.stability", 0.5)?
            .set_default("tts.similarity_boost", 0.75)?
            .set_default("tts.speed", 1.0)?
            .set_override_option(
                "tts.speed",
                std::env::var("TTS_SPEED").ok().and_then(|v| v.parse::<f64>().ok()),
            )?
            // Assets (Pexels, Stable Diffusion)
            .set_default("assets.pexels_api_key", "")?
            .set_override_option(
                "assets.pexels_api_key",
                std::env::var("PEXELS_API_KEY").ok(),
            )?
            .set_default("assets.sd_api_url", Option::<String>::None)?
            .set_override_option("assets.sd_api_url", std::env::var("SD_API_URL").ok())?
            .set_default("assets.sd_api_key", Option::<String>::None)?
            .set_override_option("assets.sd_api_key", std::env::var("SD_API_KEY").ok())?
            // YouTube
            .set_default("youtube.client_id", "")?
            .set_override_option(
                "youtube.client_id",
                std::env::var("YOUTUBE_CLIENT_ID").ok(),
            )?
            .set_default("youtube.client_secret", "")?
            .set_override_option(
                "youtube.client_secret",
                std::env::var("YOUTUBE_CLIENT_SECRET").ok(),
            )?
            .set_default("youtube.refresh_token", "")?
            .set_override_option(
                "youtube.refresh_token",
                std::env::var("YOUTUBE_REFRESH_TOKEN").ok(),
            )?
            // Channel
            .set_default("channel.name", "Excelsior Academy")?
            .set_override_option("channel.name", std::env::var("CHANNEL_NAME").ok())?
            .set_default("channel.tagline", "Wisdom for the journey upward")?
            .set_override_option("channel.tagline", std::env::var("CHANNEL_TAGLINE").ok())?
            // Content Strategy
            .set_default("content_strategy.target_duration_min", 12)?
            .set_override_option(
                "content_strategy.target_duration_min",
                std::env::var("TARGET_DURATION_MIN")
                    .ok()
                    .and_then(|v| v.parse::<i64>().ok()),
            )?
            .set_default("content_strategy.target_duration_max", 20)?
            .set_override_option(
                "content_strategy.target_duration_max",
                std::env::var("TARGET_DURATION_MAX")
                    .ok()
                    .and_then(|v| v.parse::<i64>().ok()),
            )?
            .set_default("content_strategy.videos_per_week", 4)?
            .set_override_option(
                "content_strategy.videos_per_week",
                std::env::var("VIDEOS_PER_WEEK")
                    .ok()
                    .and_then(|v| v.parse::<i64>().ok()),
            )?
            // Script Self-Improvement Loop
            .set_default("script_improvement.enabled", true)?
            .set_override_option(
                "script_improvement.enabled",
                std::env::var("SCRIPT_IMPROVEMENT_ENABLED")
                    .ok()
                    .map(|v| v.to_lowercase() == "true" || v == "1"),
            )?
            .set_default("script_improvement.candidate_count", 3)?
            .set_override_option(
                "script_improvement.candidate_count",
                std::env::var("SCRIPT_IMPROVEMENT_CANDIDATES")
                    .ok()
                    .and_then(|v| v.parse::<i64>().ok()),
            )?
            .set_default("script_improvement.quality_threshold", 8.0)?
            .set_override_option(
                "script_improvement.quality_threshold",
                std::env::var("SCRIPT_IMPROVEMENT_THRESHOLD")
                    .ok()
                    .and_then(|v| v.parse::<f64>().ok()),
            )?
            .set_default("script_improvement.max_iterations", 10)?
            .set_override_option(
                "script_improvement.max_iterations",
                std::env::var("SCRIPT_IMPROVEMENT_MAX_ITERATIONS")
                    .ok()
                    .and_then(|v| v.parse::<i64>().ok()),
            )?
            .set_default("script_improvement.timeout_seconds", 1800)?
            .set_override_option(
                "script_improvement.timeout_seconds",
                std::env::var("SCRIPT_IMPROVEMENT_TIMEOUT")
                    .ok()
                    .and_then(|v| v.parse::<i64>().ok()),
            )?
            // Notifications
            .set_default("notifications.discord_webhook_url", Option::<String>::None)?
            .set_override_option(
                "notifications.discord_webhook_url",
                std::env::var("DISCORD_WEBHOOK_URL").ok(),
            )?
            .set_default("notifications.smtp_host", Option::<String>::None)?
            .set_override_option("notifications.smtp_host", std::env::var("SMTP_HOST").ok())?
            .set_default("notifications.smtp_port", Option::<u16>::None)?
            .set_override_option(
                "notifications.smtp_port",
                std::env::var("SMTP_PORT")
                    .ok()
                    .and_then(|v| v.parse::<i64>().ok()),
            )?
            .set_default("notifications.smtp_user", Option::<String>::None)?
            .set_override_option("notifications.smtp_user", std::env::var("SMTP_USER").ok())?
            .set_default("notifications.smtp_pass", Option::<String>::None)?
            .set_override_option("notifications.smtp_pass", std::env::var("SMTP_PASS").ok())?
            .set_default("notifications.notification_email", Option::<String>::None)?
            .set_override_option(
                "notifications.notification_email",
                std::env::var("NOTIFICATION_EMAIL").ok(),
            )?
            // Control API
            .set_default("control_api_port", 8080)?
            .set_override_option(
                "control_api_port",
                std::env::var("CONTROL_API_PORT")
                    .ok()
                    .and_then(|v| v.parse::<i64>().ok()),
            )?
            .build()?;

        settings.try_deserialize()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_settings() {
        // This will use defaults since no .env is loaded in tests
        std::env::set_var("DATABASE_URL", "postgres://test:test@localhost/test");
        let settings = Settings::from_env();
        assert!(settings.is_ok());
    }
}
