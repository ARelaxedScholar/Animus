//! Animus - Autonomous YouTube Content Farm
//!
//! A daemon that produces long-form motivation/self-help videos from classic wisdom sources.

use animus::api::{create_router, AppState, DaemonStatus};
use animus::config::Settings;
use animus::db;
use animus::flows::VideoProductionFlow;
use animus::storage::S3Client;
use futures::FutureExt;
use orichalcum::llm::Client as LlmClient;
use orichalcum::NodeValue;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{error, info, warn};

fn resolve_api_url_file_path() -> PathBuf {
    if let Ok(path) = std::env::var("ANIMUS_API_URL_FILE") {
        return PathBuf::from(path);
    }

    if let Ok(runtime_dir) = std::env::var("XDG_RUNTIME_DIR") {
        return PathBuf::from(runtime_dir).join("animus_api_url");
    }

    std::env::temp_dir().join("animus_api_url")
}

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
    let llm_client_base = LlmClient::new();

    let llm_client_with_ds = if settings.llm.deepseek_base_url.is_empty() {
        llm_client_base.with_deepseek(&settings.llm.deepseek_api_key)
    } else {
        llm_client_base.with_deepseek_at(
            &settings.llm.deepseek_api_key,
            &settings.llm.deepseek_base_url,
        )
    };

    let llm_client = Arc::new(llm_client_with_ds.with_gemini(&settings.llm.gemini_api_key));
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

    // CLEANUP: Reset any 'producing' videos that aren't actually running
    // (We'll do hydration right after this)

    // Initialize shared state
    let mut shared_state_map: HashMap<String, NodeValue> = HashMap::new();

    // Check for seed topic (from environment variable)
    if let Ok(seed_topic) = std::env::var("SEED_TOPIC") {
        if !seed_topic.trim().is_empty() {
            info!("Seed topic configured: {}", seed_topic);
            shared_state_map.insert("seed_topic".to_string(), serde_json::json!(seed_topic));
        }
    }

    // Check for source focus override (from environment variable)
    if let Ok(source_focus) = std::env::var("SOURCE_FOCUS") {
        if !source_focus.trim().is_empty() {
            info!("Source focus override: {}", source_focus);
            shared_state_map.insert(
                "source_focus_override".to_string(),
                serde_json::json!(source_focus),
            );
        }
    }

    // HYDRATION: Check for active production to resume
    match db::get_active_production(&db_pool).await {
        Ok(Some(active_video)) => {
            info!("🔄 Resuming active production: {}", active_video.id);
            shared_state_map.insert(
                animus::state_keys::VIDEO_ID.to_string(),
                serde_json::json!(active_video.id.to_string()),
            );

            if let Some(brief) = active_video.topic_brief {
                shared_state_map.insert(animus::state_keys::TOPIC_BRIEF.to_string(), brief);
            }
            if let Some(script) = active_video.script {
                shared_state_map.insert(animus::state_keys::SCRIPT.to_string(), script);
            }
            if let Some(timing) = active_video.audio_timing {
                if let Some(path) = timing.get("audio_path").cloned() {
                    shared_state_map.insert(animus::state_keys::AUDIO_PATH.to_string(), path);
                }
                shared_state_map.insert(animus::state_keys::AUDIO_TIMING.to_string(), timing);
            }
            if let Some(manifest) = active_video.asset_manifest {
                shared_state_map.insert(animus::state_keys::ASSET_MANIFEST.to_string(), manifest);
            }

            shared_state_map.insert(
                "production_in_progress".to_string(),
                serde_json::json!(true),
            );
        }
        Ok(None) => {
            // Clean up any stale videos that might have been marked as producing but were lost
            if let Err(e) = sqlx::query!(
                "UPDATE videos SET status = 'failed', error_message = 'Stale production cleared at startup', failed_at_stage = 'unknown' WHERE status = 'producing'"
            ).execute(&*db_pool).await {
                warn!("Failed to cleanup stale 'producing' videos: {}", e);
            }
        }
        Err(e) => warn!("Failed to check for active production: {}", e),
    }

    let shared_state_arc = Arc::new(RwLock::new(shared_state_map));

    // Create the production flow
    let mut flow =
        VideoProductionFlow::new(&settings, llm_client, s3_client.clone(), db_pool.clone());
    info!("Production flow initialized");

    // Set up shutdown signaling
    let (shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);

    // Create API state
    let app_state = AppState {
        paused: Arc::new(RwLock::new(false)),
        shutdown_tx,
        current_status: Arc::new(RwLock::new(DaemonStatus {
            running: true,
            paused: false,
            current_video_id: None,
            current_stage: None,
            next_scheduled_video: None,
            hours_until_next: None,
            videos_produced: 0,
            last_error: None,
        })),
        shared_state: shared_state_arc.clone(),
        db_pool: db_pool.clone(),
        s3_client: s3_client.clone(),
    };

    // Start the control API server
    let api_router = create_router(app_state.clone());
    let api_port = settings.control_api_port;
    let api_url_file_path = resolve_api_url_file_path();
    let api_handle = tokio::spawn(async move {
        let requested_addr = format!("0.0.0.0:{}", api_port);
        let listener = match tokio::net::TcpListener::bind(&requested_addr).await {
            Ok(listener) => listener,
            Err(e) => {
                warn!(
                    "Control API failed to bind on {}: {}; falling back to an available port",
                    requested_addr, e
                );

                match tokio::net::TcpListener::bind("0.0.0.0:0").await {
                    Ok(listener) => listener,
                    Err(fallback_error) => {
                        error!(
                            "Control API failed to bind fallback address 0.0.0.0:0: {}",
                            fallback_error
                        );
                        return;
                    }
                }
            }
        };

        match listener.local_addr() {
            Ok(local_addr) => {
                let discovered_url = format!("http://127.0.0.1:{}", local_addr.port());
                info!("Control API listening on http://{}", local_addr);

                if let Some(parent_dir) = api_url_file_path.parent() {
                    if parent_dir.as_os_str().is_empty() {
                        // Relative file in current directory has no parent to create.
                    } else if let Err(e) = tokio::fs::create_dir_all(parent_dir).await {
                        warn!(
                            "Failed to create API URL discovery directory {}: {}",
                            parent_dir.display(),
                            e
                        );
                    }
                }

                match tokio::fs::write(&api_url_file_path, format!("{}\n", discovered_url)).await {
                    Ok(()) => info!(
                        "Wrote discovered API URL {} to {}",
                        discovered_url,
                        api_url_file_path.display()
                    ),
                    Err(e) => warn!(
                        "Failed to write discovered API URL to {}: {}",
                        api_url_file_path.display(),
                        e
                    ),
                }
            }
            Err(e) => warn!("Control API bound but failed to read local address: {}", e),
        }

        if let Err(e) = axum::serve(listener, api_router.into_make_service()).await {
            error!("Control API server exited with error: {}", e);
        }
    });

    info!("");
    info!("🚀 Daemon starting main loop");
    info!(
        "   Control API requested: http://localhost:{} (fallback to any free port if unavailable)",
        settings.control_api_port
    );
    info!("   POST /pause    - Pause production");
    info!("   POST /resume   - Resume production");
    info!("   POST /shutdown - Graceful shutdown");
    info!("");

    // Start background analytics worker
    let analytics_pool = db_pool.clone();
    tokio::spawn(async move {
        animus::analytics::start_analytics_worker(analytics_pool, 12).await;
    });

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

            // Update next scheduled time from database
            if let Ok(Some(next)) = db::get_latest_scheduled_time(&db_pool).await {
                let now = chrono::Utc::now();
                status.next_scheduled_video = Some(next.to_rfc3339());
                status.hours_until_next = Some((next - now).num_hours());
            }
        }

        // Get exclusive access to shared state for this run
        let mut shared_state = shared_state_arc.write().await;

        // Ensure per-run state is clean IF not resuming
        let is_resuming = shared_state
            .get("production_in_progress")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        if !is_resuming {
            shared_state.remove("error");
            shared_state.remove("manual_script");
            // Note: seed_topic and source_focus_override are cleared at the end of the loop
            // but we ensure production_in_progress is false if we are not resuming
            shared_state.insert(
                "production_in_progress".to_string(),
                serde_json::json!(false),
            );
        }

        let flow_result = std::panic::AssertUnwindSafe(flow.inner_mut().run(&mut shared_state))
            .catch_unwind()
            .await;

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
                        status.current_video_id = None;
                        status.current_stage = None;
                    }

                    // Reset resume flags
                    shared_state.insert(
                        "production_in_progress".to_string(),
                        serde_json::json!(false),
                    );

                    // Wait before retrying (10 minutes)
                    tokio::time::sleep(tokio::time::Duration::from_secs(600)).await;
                } else {
                    info!("Production cycle completed with action: {}", action);

                    {
                        let mut status = app_state.current_status.write().await;
                        status.videos_produced += 1;
                        status.current_video_id = None;
                        status.current_stage = None;
                    }

                    // Clear state for next run
                    shared_state.insert(
                        "production_in_progress".to_string(),
                        serde_json::json!(false),
                    );

                    // Small delay before next cycle
                    tokio::time::sleep(tokio::time::Duration::from_secs(300)).await;
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

                // Clear state
                shared_state.insert(
                    "production_in_progress".to_string(),
                    serde_json::json!(false),
                );

                // Small delay before next cycle
                tokio::time::sleep(tokio::time::Duration::from_secs(300)).await;
            }
            Err(e) => {
                error!("CRITICAL: Production flow panicked! {:?}", e);
                // Reset state
                shared_state.insert(
                    "production_in_progress".to_string(),
                    serde_json::json!(false),
                );
                // Wait before retrying to avoid rapid crash loop
                tokio::time::sleep(tokio::time::Duration::from_secs(600)).await;
            }
        }

        // Clear one-time inputs
        shared_state.remove("seed_topic");
        shared_state.remove("source_focus_override");
        shared_state.remove("manual_script");
    }

    // Clean shutdown
    api_handle.abort();
    info!("Animus daemon stopped");

    Ok(())
}
