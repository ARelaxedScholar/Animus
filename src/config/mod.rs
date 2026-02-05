//! Configuration management for Animus

mod settings;

pub use settings::{
    ChannelConfig, ContentStrategyConfig, DatabaseConfig, LlmConfig, NotificationConfig,
    S3Config, Settings, TtsConfig, YouTubeConfig,
};

pub use config::ConfigError;
