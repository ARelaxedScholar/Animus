//! Background worker for YouTube Analytics and performance tracking
//!
//! Periodically calls the Python analytics worker to update metrics
//! for published videos and maintain channel baselines.

use std::process::Stdio;
use std::sync::Arc;
use std::time::Duration;
use sqlx::PgPool;
use tokio::process::Command;
use tracing::{debug, error, info};

/// Background task that runs analytics updates periodically
pub async fn start_analytics_worker(_db_pool: Arc<PgPool>, interval_hours: u64) {
    info!("Starting background analytics worker (interval: {} hours)", interval_hours);
    
    let python_executable = std::env::var("PYTHON_EXECUTABLE")
        .unwrap_or_else(|_| "python3".to_string());
    
    let script_path = "scripts/analytics_worker.py";
    
    // Initial delay to let the daemon settle
    tokio::time::sleep(Duration::from_secs(60)).await;

    loop {
        info!("Running scheduled analytics update...");
        
        match Command::new(&python_executable)
            .arg(script_path)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
        {
            Ok(child) => {
                let output = child.wait_with_output().await;
                match output {
                    Ok(out) => {
                        if out.status.success() {
                            info!("Analytics update completed successfully");
                            let stdout = String::from_utf8_lossy(&out.stdout);
                            for line in stdout.lines() {
                                if line.contains("Processed") || line.contains("Baseline") {
                                    debug!("Analytics: {}", line.trim());
                                }
                            }
                        } else {
                            let stderr = String::from_utf8_lossy(&out.stderr);
                            error!("Analytics worker failed: {}", stderr);
                        }
                    }
                    Err(e) => error!("Failed to read analytics worker output: {}", e),
                }
            }
            Err(e) => error!("Failed to spawn analytics worker: {}", e),
        }

        // Wait for next interval
        tokio::time::sleep(Duration::from_secs(interval_hours * 3600)).await;
    }
}
