//! Video Production Flow
//!
//! Wires together all nodes into a complete production pipeline.

use orichalcum::llm::{Client, Enabled, Providers};
use orichalcum::{AsyncFlow, AsyncNode, Executable};
use sqlx::PgPool;
use std::sync::Arc;

use crate::config::Settings;
use crate::nodes::asset_collector::{AssetCollectorConfig, AssetCollectorLogic};
use crate::nodes::publisher::{PublisherConfig, PublisherLogic};
use crate::nodes::scheduler::{SchedulerConfig, SchedulerLogic};
use crate::nodes::script_writer::{ScriptWriterConfig, ScriptWriterLogic};
use crate::nodes::seo_optimizer::{SEOConfig, SEOOptimizerLogic};
use crate::nodes::strategy::{StrategyConfig, StrategyLogic};
use crate::nodes::thumbnail::{ThumbnailConfig, ThumbnailLogic};
use crate::nodes::tts::{TTSConfig, TTSLogic};
use crate::nodes::video_assembler::{VideoAssemblerConfig, VideoAssemblerLogic};
use crate::storage::S3Client;

/// The main video production flow
pub struct VideoProductionFlow {
    flow: AsyncFlow,
}

impl VideoProductionFlow {
    /// Create a new video production flow from settings
    pub fn new(
        settings: &Settings,
        llm_client: Arc<Client<Providers<orichalcum::llm::Disabled, Enabled, Enabled>>>,
        s3_client: Arc<S3Client>,
        db_pool: Arc<PgPool>,
    ) -> Self {
        // Create all node configurations
        let scheduler_config = SchedulerConfig {
            videos_per_week: settings.content_strategy.videos_per_week,
            ..Default::default()
        };

        let strategy_config = StrategyConfig {
            target_duration_min: settings.content_strategy.target_duration_min,
            target_duration_max: settings.content_strategy.target_duration_max,
            channel_name: settings.channel.name.clone(),
            ..Default::default()
        };

        let script_writer_config = ScriptWriterConfig {
            channel_name: settings.channel.name.clone(),
            ..Default::default()
        };

        let tts_config = TTSConfig {
            provider: settings.tts.provider.clone(),
            elevenlabs_api_key: settings.tts.elevenlabs_api_key.clone(),
            elevenlabs_voice_id: settings.tts.elevenlabs_voice_id.clone(),
            elevenlabs_model_id: settings.tts.elevenlabs_model_id.clone(),
            openai_api_key: settings.tts.openai_api_key.clone(),
            openai_voice: settings.tts.openai_voice.clone(),
            openai_model: settings.tts.openai_model.clone(),
            qwen3_api_url: settings.tts.qwen3_api_url.clone(),
            qwen3_api_key: settings.tts.qwen3_api_key.clone(),
            qwen3_voice: settings.tts.qwen3_voice.clone(),
            local_model_path: settings.tts.local_model_path.clone(),
            local_speaker_id: settings.tts.local_speaker_id.clone(),
            stability: settings.tts.stability,
            similarity_boost: settings.tts.similarity_boost,
            speed: settings.tts.speed,
        };

        let asset_collector_config = AssetCollectorConfig {
            pexels_api_key: settings.assets.pexels_api_key.clone(),
            leonardo_api_key: settings.assets.leonardo_api_key.clone(),
            freesound_api_key: settings.assets.freesound_api_key.clone(),
            sd_api_url: settings.assets.sd_api_url.clone(),
            sd_api_key: settings.assets.sd_api_key.clone(),
            min_clips_per_section: settings.assets.min_clips_per_section,
        };
        let video_assembler_config = VideoAssemblerConfig::default();
        let thumbnail_config = ThumbnailConfig::default();

        let seo_config = SEOConfig {
            channel_name: settings.channel.name.clone(),
            ..Default::default()
        };

        let publisher_config = PublisherConfig {
            client_id: settings.youtube.client_id.clone(),
            client_secret: settings.youtube.client_secret.clone(),
            refresh_token: settings.youtube.refresh_token.clone(),
            ..Default::default()
        };

        // Create nodes with database pool for persistence
        let scheduler_node = AsyncNode::new(SchedulerLogic::new(scheduler_config, db_pool.clone()));
        let strategy_node = AsyncNode::new(StrategyLogic::new(
            strategy_config,
            llm_client.clone(),
            db_pool.clone(),
        ));
        let script_writer_node = AsyncNode::new(ScriptWriterLogic::new(
            script_writer_config,
            settings.script_improvement.clone(),
            llm_client.clone(),
            db_pool.clone(),
        ));
        let tts_node = AsyncNode::new(TTSLogic::new(
            tts_config,
            s3_client.clone(),
            db_pool.clone(),
        ));
        let asset_collector_node = AsyncNode::new(AssetCollectorLogic::new(
            asset_collector_config,
            llm_client.clone(),
            s3_client.clone(),
            db_pool.clone(),
        ));
        let video_assembler_node = AsyncNode::new(VideoAssemblerLogic::new(
            video_assembler_config,
            s3_client.clone(),
            db_pool.clone(),
        ));
        let thumbnail_node = AsyncNode::new(ThumbnailLogic::new(
            thumbnail_config,
            llm_client.clone(),
            s3_client.clone(),
            db_pool.clone(),
        ));
        let seo_node = AsyncNode::new(SEOOptimizerLogic::new(
            seo_config,
            llm_client.clone(),
            db_pool.clone(),
        ));
        let publisher_node = AsyncNode::new(PublisherLogic::new(
            publisher_config,
            s3_client.clone(),
            db_pool.clone(),
        ));

        // Wire the flow:
        // Scheduler -> Strategy -> ScriptWriter -> TTS -> AssetCollector -> VideoAssembler -> Thumbnail -> SEO -> Publisher

        let publisher = publisher_node; // End of chain

        let seo = seo_node.next(Executable::Async(publisher));
        let thumbnail = thumbnail_node.next(Executable::Async(seo));
        let video_assembler = video_assembler_node.next(Executable::Async(thumbnail));
        let asset_collector = asset_collector_node.next(Executable::Async(video_assembler));
        let tts = tts_node.next(Executable::Async(asset_collector));
        let script_writer = script_writer_node.next(Executable::Async(tts));
        let strategy = strategy_node.next(Executable::Async(script_writer));

        // Scheduler can go to strategy (default) or wait (if no production needed)
        let scheduler = scheduler_node.next(Executable::Async(strategy));

        // Create the flow starting with scheduler
        let flow = AsyncFlow::new(Executable::Async(scheduler));

        Self { flow }
    }

    /// Get a mutable reference to the underlying flow
    pub fn inner_mut(&mut self) -> &mut AsyncFlow {
        &mut self.flow
    }

    /// Get a reference to the underlying flow
    pub fn inner(&self) -> &AsyncFlow {
        &self.flow
    }
}

impl std::ops::Deref for VideoProductionFlow {
    type Target = AsyncFlow;

    fn deref(&self) -> &Self::Target {
        &self.flow
    }
}

impl std::ops::DerefMut for VideoProductionFlow {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.flow
    }
}
