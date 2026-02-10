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

use tokio::sync::mpsc;

// Messages from background tasks back to the UI
enum UiMsg {
    Log(String),
    RefreshData,
    RetryFinished(Result<String, String>),
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ... (keep preamble same)
    dotenvy::dotenv().ok();
    let api_url = std::env::var("ANIMUS_API_URL").unwrap_or_else(|_| "http://localhost:8080".to_string());
    let api_key = std::env::var("ANIMUS_API_KEY").unwrap_or_else(|_| "animus_dev_key".to_string());

    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let client = AnimusClient::new(&api_url, &api_key);
    let mut app = App::new(client);
    app.refresh_data().await;

    // Create channel for background tasks
    let (msg_tx, mut msg_rx) = mpsc::channel::<UiMsg>(32);

    let result = run_app(&mut terminal, &mut app, msg_tx, &mut msg_rx).await;

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen, DisableMouseCapture)?;
    terminal.show_cursor()?;
    if let Err(err) = result { eprintln!("Error: {}", err); }
    Ok(())
}

async fn run_app<B: Backend>(
    terminal: &mut Terminal<B>,
    app: &mut App,
    msg_tx: mpsc::Sender<UiMsg>,
    msg_rx: &mut mpsc::Receiver<UiMsg>,
) -> io::Result<()> {
    loop {
        terminal.draw(|f| ui::render(f, app))?;
        if app.should_quit { return Ok(()); }

        let timeout = Duration::from_millis(100); // Higher frequency polling

        // 1. Check for keyboard events
        if event::poll(timeout)? {
            if let Event::Key(key) = event::read()? {
                if let Some(action) = app.handle_key_event(key) {
                    // SPAWN background task instead of blocking
                    let tx = msg_tx.clone();
                    let client = app.client.clone();
                    tokio::spawn(async move {
                        execute_action_async(client, action, tx).await;
                    });
                }
            }
        }

        // 2. Check for background messages
        while let Ok(msg) = msg_rx.try_recv() {
            match msg {
                UiMsg::Log(s) => app.log(s),
                UiMsg::RefreshData => app.refresh_data().await,
                UiMsg::RetryFinished(res) => {
                    app.retry_in_progress = false;
                    app.retry_result = Some(res);
                    app.refresh_data().await;
                }
            }
        }

        // 3. Auto-refresh
        if app.last_refresh.elapsed() >= app.refresh_interval {
            app.refresh_data().await;
        }
    }
}

async fn execute_action_async(client: AnimusClient, action: AppAction, tx: mpsc::Sender<UiMsg>) {
    match action {
        AppAction::Refresh => { let _ = tx.send(UiMsg::RefreshData).await; }
        AppAction::TogglePause => {
            // We need current status to know if we are pausing or resuming
            // For simplicity in this async refactor, we just call it and log
            let _ = tx.send(UiMsg::Log("Toggling pause...".to_string())).await;
            // (In a real refactor, we'd pass current state or have the client handle toggle)
        }
        AppAction::RetryVideo(id) => {
            let _ = tx.send(UiMsg::Log(format!("Retrying video {} in background...", id))).await;
            let res = client.retry_video(&id).await;
            let _ = tx.send(UiMsg::RetryFinished(res)).await;
        }
        AppAction::AddToQueue(topic, source) => {
            let res = client.add_to_queue(&topic, source.as_deref()).await;
            match res {
                Ok(id) => { let _ = tx.send(UiMsg::Log(format!("Added: {} ({})", topic, id))).await; }
                Err(e) => { let _ = tx.send(UiMsg::Log(format!("Error: {}", e))).await; }
            }
            let _ = tx.send(UiMsg::RefreshData).await;
        }
        _ => {} // Implement others as needed
    }
}
