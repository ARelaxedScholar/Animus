//! Configuration management for Animus

mod settings;

pub use settings::{
    AssetConfig, ChannelConfig, ContentStrategyConfig, DatabaseConfig, FactCheckerConfig, LlmConfig,
    NotificationConfig, S3Config, ScriptImprovementConfig, Settings, TtsConfig, YouTubeConfig,
};

pub use config::ConfigError;
