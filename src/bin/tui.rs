//! Animus TUI Dashboard
//!
//! A terminal-based dashboard for monitoring and controlling the Animus daemon.

use animus::tui::{App, AppAction, AnimusClient};
use animus::tui::ui;

use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::prelude::*;
use std::io;
use std::time::Duration;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load environment variables
    dotenvy::dotenv().ok();

    // Get configuration from environment
    let api_url = std::env::var("ANIMUS_API_URL")
        .unwrap_or_else(|_| "http://localhost:8080".to_string());
    let api_key = std::env::var("ANIMUS_API_KEY")
        .unwrap_or_else(|_| "animus_dev_key".to_string());

    // Setup terminal
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    // Create app
    let client = AnimusClient::new(&api_url, &api_key);
    let mut app = App::new(client);

    // Initial data fetch
    app.refresh_data().await;

    // Main loop
    let result = run_app(&mut terminal, &mut app).await;

    // Restore terminal
    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture
    )?;
    terminal.show_cursor()?;

    if let Err(err) = result {
        eprintln!("Error: {}", err);
    }

    Ok(())
}

async fn run_app<B: Backend>(
    terminal: &mut Terminal<B>,
    app: &mut App,
) -> io::Result<()> {
    loop {
        // Render
        terminal.draw(|f| ui::render(f, app))?;

        // Check if we should quit
        if app.should_quit {
            return Ok(());
        }

        // Handle events with timeout for refresh
        let timeout = app.refresh_interval
            .checked_sub(app.last_refresh.elapsed())
            .unwrap_or(Duration::ZERO);

        if event::poll(timeout)? {
            if let Event::Key(key) = event::read()? {
                if let Some(action) = app.handle_key_event(key) {
                    execute_action(app, action).await;
                }
            }
        }

        // Refresh data if needed
        if app.last_refresh.elapsed() >= app.refresh_interval {
            app.refresh_data().await;
        }
    }
}

async fn execute_action(app: &mut App, action: AppAction) {
    app.busy = true;
    match action {
        AppAction::Refresh => {
            app.refresh_data().await;
        }
        AppAction::TogglePause => {
            let result = if app.status.paused {
                app.client.resume().await
            } else {
                app.client.pause().await
            };
            match result {
                Ok(msg) => app.log(msg),
                Err(e) => app.log(format!("Error: {}", e)),
            }
            app.refresh_data().await;
        }
        AppAction::Shutdown => {
            match app.client.shutdown().await {
                Ok(msg) => app.log(msg),
                Err(e) => app.log(format!("Error: {}", e)),
            }
            app.refresh_data().await;
        }
        AppAction::AddToQueue(topic, source) => {
            match app.client.add_to_queue(&topic, source.as_deref()).await {
                Ok(id) => app.log(format!("Added to queue: {} (id: {})", topic, id)),
                Err(e) => app.log(format!("Error adding to queue: {}", e)),
            }
            app.refresh_data().await;
        }
        AppAction::RemoveFromQueue(id) => {
            match app.client.remove_from_queue(id).await {
                Ok(_) => app.log(format!("Removed queue item {}", id)),
                Err(e) => app.log(format!("Error removing from queue: {}", e)),
            }
            app.refresh_data().await;
        }
        AppAction::ClearQueue => {
            match app.client.clear_queue().await {
                Ok(count) => app.log(format!("Cleared {} items from queue", count)),
                Err(e) => app.log(format!("Error clearing queue: {}", e)),
            }
            app.refresh_data().await;
        }
        AppAction::RetryVideo(video_id) => {
            app.retry_in_progress = true;
            app.retry_result = None;
            app.log(format!("Retrying video {}", video_id));
            
            let result = app.client.retry_video(&video_id).await;
            app.retry_in_progress = false;
            
            match &result {
                Ok(msg) => app.log(format!("Retry success: {}", msg)),
                Err(e) => app.log(format!("Retry failed: {}", e)),
            }
            app.retry_result = Some(result);
            app.refresh_data().await;
        }
        AppAction::DownloadVideo(video_id) => {
            app.log(format!("Downloading video {}...", video_id));
            let result = app.client.download_video(&video_id, "downloads").await;
            match result {
                Ok(path) => app.log(format!("Video downloaded to: {}", path)),
                Err(e) => app.log(format!("Download failed: {}", e)),
            }
        }
    }
    app.busy = false;
}
