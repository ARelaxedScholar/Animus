//! Animus - Autonomous YouTube Content Farm
//!
//! A daemon that produces long-form motivation/self-help videos from classic wisdom sources.

use animus::api::{create_router, AppState, DaemonStatus};
use animus::config::Settings;
use animus::db;
use animus::flows::VideoProductionFlow;
use animus::storage::S3Client;
use futures::FutureExt;
use orichalcum::NodeValue;
use orichalcum::llm::Client as LlmClient;
use sqlx::PgPool;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{error, info, warn};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "animus=info,orichalcum=info".into()),
        )
        .init();

    info!("🎬 Animus - YouTube Content Farm");
    info!("   Channel: Excelsior Academy");
    info!("   Wisdom for the journey upward");
    info!("");

    // Load configuration
    let settings = Settings::from_env().map_err(|e| anyhow::anyhow!("Config error: {}", e))?;
    info!("Configuration loaded");

    // Initialize S3 client
    let s3_client = Arc::new(
        S3Client::new(
            &settings.s3.endpoint,
            &settings.s3.access_key,
            &settings.s3.secret_key,
            &settings.s3.bucket,
            &settings.s3.region,
        )
        .await
        .map_err(|e| anyhow::anyhow!("S3 init error: {}", e))?,
    );
    info!("S3 client initialized (bucket: {})", settings.s3.bucket);

    // Initialize LLM client
    let llm_client = Arc::new(
        LlmClient::new()
            .with_deepseek(&settings.llm.deepseek_api_key, Some(settings.llm.deepseek_base_url.clone()))
            .with_gemini(&settings.llm.gemini_api_key, None),
    );
    info!("LLM client initialized (DeepSeek + Gemini)");

    // Initialize database pool
    let db_pool = Arc::new(
        db::create_pool(&settings.database.url)
            .await
            .map_err(|e| anyhow::anyhow!("Database connection failed: {}", e))?,
    );
    info!("Database connected");

    // Run migrations
    db::run_migrations(&db_pool)
        .await
        .map_err(|e| anyhow::anyhow!("Migration failed: {}", e))?;
    info!("Database migrations applied");

    // Log queued seeds from previous session
    match db::get_queue_length(&db_pool).await {
        Ok(count) if count > 0 => {
            info!("Found {} seed(s) queued from previous session", count);
        }
        Ok(_) => {}
        Err(e) => {
            warn!("Failed to check seed queue: {}", e);
        }
    }

    // Create the production flow
    let mut flow = VideoProductionFlow::new(&settings, llm_client, s3_client, db_pool.clone());
    info!("Production flow initialized");

    // Set up shutdown signaling
    let (shutdown_tx, mut shutdown_rx) = tokio::sync::watch::channel(false);

    // Create API state
    let app_state = AppState {
        paused: Arc::new(RwLock::new(false)),
        shutdown_tx,
        current_status: Arc::new(RwLock::new(DaemonStatus {
            running: true,
            paused: false,
            current_video_id: None,
            current_stage: None,
            videos_produced: 0,
            last_error: None,
        })),
    };

    // Start the control API server
    let api_router = create_router(app_state.clone());
    let api_port = settings.control_api_port;
    let api_handle = tokio::spawn(async move {
        let listener = tokio::net::TcpListener::bind(format!("0.0.0.0:{}", api_port))
            .await
            .expect("Failed to bind API port");
        info!("Control API listening on http://0.0.0.0:{}", api_port);
        axum::serve(listener, api_router.into_make_service()).await.ok()
    });

    // Initialize shared state
    let mut shared_state: HashMap<String, NodeValue> = HashMap::new();
    
    // Check for seed topic (from environment variable)
    if let Ok(seed_topic) = std::env::var("SEED_TOPIC") {
        if !seed_topic.trim().is_empty() {
            info!("Seed topic configured: {}", seed_topic);
            shared_state.insert("seed_topic".to_string(), serde_json::json!(seed_topic));
        }
    }

    // Check for source focus override (from environment variable)
    if let Ok(source_focus) = std::env::var("SOURCE_FOCUS") {
        if !source_focus.trim().is_empty() {
            info!("Source focus override: {}", source_focus);
            shared_state.insert("source_focus_override".to_string(), serde_json::json!(source_focus));
        }
    }

    info!("");
    info!("🚀 Daemon starting main loop");
    info!("   Control API: http://localhost:{}", settings.control_api_port);
    info!("   POST /pause    - Pause production");
    info!("   POST /resume   - Resume production");
    info!("   POST /shutdown - Graceful shutdown");
    info!("");
    info!("   Env vars: SEED_TOPIC, SOURCE_FOCUS (Bible, Stoicism, Philosophy, Biography, Psychology)");
    info!("");

    // Main production loop
    loop {
        // Check for shutdown signal
        if *shutdown_rx.borrow() {
            info!("Shutdown signal received, exiting gracefully...");
            break;
        }

        // Check if paused
        {
            let paused = app_state.paused.read().await;
            if *paused {
                tokio::time::sleep(tokio::time::Duration::from_secs(10)).await;
                continue;
            }
        }

        // Run one production cycle
        info!("Starting production cycle...");
        
        {
            let mut status = app_state.current_status.write().await;
            status.current_stage = Some("scheduler".to_string());
        }

        let flow_result = std::panic::AssertUnwindSafe(flow.inner_mut().run(&mut shared_state)).catch_unwind().await;
        
        match flow_result {
            Ok(Some(action)) => {
                if action == "wait" {
                    info!("Scheduler: No production needed, waiting...");
                    // Wait before checking again (1 hour)
                    for _ in 0..360 {
                        if *shutdown_rx.borrow() {
                            break;
                        }
                        tokio::time::sleep(tokio::time::Duration::from_secs(10)).await;
                    }
                } else if action == "error" {
                    let error = shared_state
                        .get("error")
                        .and_then(|v: &NodeValue| v.as_str())
                        .unwrap_or("Unknown error");
                    error!("Production cycle failed: {}", error);
                    
                    {
                        let mut status = app_state.current_status.write().await;
                        status.last_error = Some(error.to_string());
                    }
                    
                    // Wait before retrying (5 minutes)
                    tokio::time::sleep(tokio::time::Duration::from_secs(300)).await;
                } else {
                    info!("Production cycle completed with action: {}", action);
                    
                    {
                        let mut status = app_state.current_status.write().await;
                        status.videos_produced += 1;
                        status.current_video_id = None;
                        status.current_stage = None;
                    }
                }
            }
            Ok(None) => {
                info!("Production cycle completed successfully!");
                
                {
                    let mut status = app_state.current_status.write().await;
                    status.videos_produced += 1;
                    status.current_video_id = None;
                    status.current_stage = None;
                }
                
                // Small delay before next cycle
                tokio::time::sleep(tokio::time::Duration::from_secs(60)).await;
            }
            Err(e) => {
                error!("CRITICAL: Production flow panicked! {:?}", e);
                // Wait before retrying to avoid rapid crash loop
                tokio::time::sleep(tokio::time::Duration::from_secs(60)).await;
            }
        }

        // Clear any one-time seed topic
        shared_state.remove("seed_topic");
    }

    // Clean shutdown
    api_handle.abort();
    info!("Animus daemon stopped");

    Ok(())
}
