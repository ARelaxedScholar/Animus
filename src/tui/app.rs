//! TUI Application State and Event Loop

use crossterm::event::{self, KeyCode, KeyModifiers};
use std::time::{Duration, Instant};

use crate::tui::api_client::{AnimusClient, DaemonStatus, QueueItem, Stats, VideoSummary};

/// Active tab in the TUI
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tab {
    Dashboard,
    Videos,
    Queue,
    Retry,
    Settings,
}

impl Tab {
    pub fn all() -> &'static [Tab] {
        &[
            Tab::Dashboard,
            Tab::Videos,
            Tab::Queue,
            Tab::Retry,
            Tab::Settings,
        ]
    }

    pub fn title(&self) -> &'static str {
        match self {
            Tab::Dashboard => "Dashboard",
            Tab::Videos => "Videos",
            Tab::Queue => "Queue",
            Tab::Retry => "Retry",
            Tab::Settings => "Settings",
        }
    }

    pub fn key(&self) -> &'static str {
        match self {
            Tab::Dashboard => "F1",
            Tab::Videos => "F2",
            Tab::Queue => "F3",
            Tab::Retry => "F4",
            Tab::Settings => "F5",
        }
    }

    pub fn next(&self) -> Tab {
        match self {
            Tab::Dashboard => Tab::Videos,
            Tab::Videos => Tab::Queue,
            Tab::Queue => Tab::Retry,
            Tab::Retry => Tab::Settings,
            Tab::Settings => Tab::Dashboard,
        }
    }

    pub fn prev(&self) -> Tab {
        match self {
            Tab::Dashboard => Tab::Settings,
            Tab::Videos => Tab::Dashboard,
            Tab::Queue => Tab::Videos,
            Tab::Retry => Tab::Queue,
            Tab::Settings => Tab::Retry,
        }
    }
}

/// Input mode for the app
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputMode {
    Normal,
    Editing,
}

/// Main application state
pub struct App {
    pub client: AnimusClient,
    pub active_tab: Tab,
    pub input_mode: InputMode,
    pub should_quit: bool,
    pub last_refresh: Instant,
    pub refresh_interval: Duration,

    // Connection state
    pub connected: bool,
    pub last_error: Option<String>,

    // Dashboard data
    pub status: DaemonStatus,
    pub stats: Stats,

    // Videos tab
    pub videos: Vec<VideoSummary>,
    pub videos_selected: usize,
    pub videos_filter: Option<String>,
    pub videos_scroll: usize,

    // Queue tab
    pub queue: Vec<QueueItem>,
    pub queue_selected: usize,
    pub queue_input: String,
    pub queue_source_input: String,
    pub queue_editing_source: bool,

    // Retry tab
    pub retry_videos: Vec<VideoSummary>,
    pub retry_selected: usize,
    pub retry_in_progress: bool,
    pub retry_result: Option<Result<String, String>>,

    // Activity log
    pub activity_log: Vec<String>,
    pub busy: bool,
}

impl App {
    pub fn new(client: AnimusClient) -> Self {
        Self {
            client,
            active_tab: Tab::Dashboard,
            input_mode: InputMode::Normal,
            should_quit: false,
            last_refresh: Instant::now() - Duration::from_secs(10), // Force immediate refresh
            refresh_interval: Duration::from_secs(2),

            connected: false,
            last_error: None,

            status: DaemonStatus::default(),
            stats: Stats::default(),

            videos: Vec::new(),
            videos_selected: 0,
            videos_filter: None,
            videos_scroll: 0,

            queue: Vec::new(),
            queue_selected: 0,
            queue_input: String::new(),
            queue_source_input: String::new(),
            queue_editing_source: false,

            retry_videos: Vec::new(),
            retry_selected: 0,
            retry_in_progress: false,
            retry_result: None,

            activity_log: vec!["TUI started".to_string()],
            busy: false,
        }
    }

    pub fn log(&mut self, msg: impl Into<String>) {
        let timestamp = chrono::Local::now().format("%H:%M:%S").to_string();
        self.activity_log
            .push(format!("[{}] {}", timestamp, msg.into()));
        if self.activity_log.len() > 100 {
            self.activity_log.remove(0);
        }
    }

    pub async fn refresh_data(&mut self) {
        // Check connection
        match self.client.health().await {
            Ok(true) => {
                if !self.connected {
                    self.log("Connected to daemon");
                }
                self.connected = true;
                self.last_error = None;
            }
            Ok(false) | Err(_) => {
                if self.connected {
                    self.log("Lost connection to daemon");
                }
                self.connected = false;
                return;
            }
        }

        // Refresh based on active tab
        match self.active_tab {
            Tab::Dashboard => {
                if let Ok(status) = self.client.status().await {
                    self.status = status;
                }
                if let Ok(stats) = self.client.stats().await {
                    self.stats = stats;
                }
            }
            Tab::Videos => {
                if let Ok(videos) = self
                    .client
                    .list_videos(self.videos_filter.as_deref(), Some(100))
                    .await
                {
                    self.videos = videos;
                }
            }
            Tab::Queue => {
                if let Ok(queue) = self.client.list_queue().await {
                    self.queue = queue;
                }
            }
            Tab::Retry => {
                if let Ok(videos) = self.client.list_videos(Some("failed"), Some(50)).await {
                    // Filter to only those that failed at publisher stage (have video_path)
                    self.retry_videos = videos
                        .into_iter()
                        .filter(|v| v.failed_at_stage.as_deref() == Some("publisher"))
                        .collect();
                }
            }
            Tab::Settings => {
                if let Ok(status) = self.client.status().await {
                    self.status = status;
                }
            }
        }

        self.last_refresh = Instant::now();
    }

    pub fn handle_key_event(&mut self, key: event::KeyEvent) -> Option<AppAction> {
        // Global keys
        match (key.modifiers, key.code) {
            (KeyModifiers::CONTROL, KeyCode::Char('c')) => {
                self.should_quit = true;
                return None;
            }
            (_, KeyCode::Char('q')) if self.input_mode == InputMode::Normal => {
                self.should_quit = true;
                return None;
            }
            (_, KeyCode::F(1)) => {
                self.active_tab = Tab::Dashboard;
                return None;
            }
            (_, KeyCode::F(2)) => {
                self.active_tab = Tab::Videos;
                return None;
            }
            (_, KeyCode::F(3)) => {
                self.active_tab = Tab::Queue;
                return None;
            }
            (_, KeyCode::F(4)) => {
                self.active_tab = Tab::Retry;
                return None;
            }
            (_, KeyCode::F(5)) => {
                self.active_tab = Tab::Settings;
                return None;
            }
            (_, KeyCode::Tab) if self.input_mode == InputMode::Normal => {
                self.active_tab = self.active_tab.next();
                return None;
            }
            (KeyModifiers::SHIFT, KeyCode::BackTab) => {
                self.active_tab = self.active_tab.prev();
                return None;
            }
            _ => {}
        }

        // Tab-specific keys
        match self.active_tab {
            Tab::Dashboard => self.handle_dashboard_key(key),
            Tab::Videos => self.handle_videos_key(key),
            Tab::Queue => self.handle_queue_key(key),
            Tab::Retry => self.handle_retry_key(key),
            Tab::Settings => self.handle_settings_key(key),
        }
    }

    fn handle_dashboard_key(&mut self, key: event::KeyEvent) -> Option<AppAction> {
        match key.code {
            KeyCode::Char('r') => Some(AppAction::Refresh),
            _ => None,
        }
    }

    fn handle_videos_key(&mut self, key: event::KeyEvent) -> Option<AppAction> {
        match key.code {
            KeyCode::Up | KeyCode::Char('k') => {
                if self.videos_selected > 0 {
                    self.videos_selected -= 1;
                }
                None
            }
            KeyCode::Down | KeyCode::Char('j') => {
                if self.videos_selected < self.videos.len().saturating_sub(1) {
                    self.videos_selected += 1;
                }
                None
            }
            KeyCode::Char('f') => {
                // Cycle through filters
                self.videos_filter = match self.videos_filter.as_deref() {
                    None => Some("published".to_string()),
                    Some("published") => Some("failed".to_string()),
                    Some("failed") => Some("producing".to_string()),
                    _ => None,
                };
                self.videos_selected = 0;
                Some(AppAction::Refresh)
            }
            KeyCode::Char('d') => {
                // Download selected video
                if !self.videos.is_empty() {
                    let video_id = self.videos[self.videos_selected].id.clone();
                    return Some(AppAction::DownloadVideo(video_id));
                }
                None
            }
            KeyCode::Char('r') => Some(AppAction::Refresh),
            _ => None,
        }
    }

    fn handle_queue_key(&mut self, key: event::KeyEvent) -> Option<AppAction> {
        match self.input_mode {
            InputMode::Normal => match key.code {
                KeyCode::Up | KeyCode::Char('k') => {
                    if self.queue_selected > 0 {
                        self.queue_selected -= 1;
                    }
                    None
                }
                KeyCode::Down | KeyCode::Char('j') => {
                    if self.queue_selected < self.queue.len().saturating_sub(1) {
                        self.queue_selected += 1;
                    }
                    None
                }
                KeyCode::Char('a') => {
                    self.input_mode = InputMode::Editing;
                    self.queue_input.clear();
                    self.queue_source_input.clear();
                    self.queue_editing_source = false;
                    None
                }
                KeyCode::Char('d') | KeyCode::Delete => {
                    if !self.queue.is_empty() {
                        let id = self.queue[self.queue_selected].id;
                        return Some(AppAction::RemoveFromQueue(id));
                    }
                    None
                }
                KeyCode::Char('c') => Some(AppAction::ClearQueue),
                KeyCode::Char('r') => Some(AppAction::Refresh),
                _ => None,
            },
            InputMode::Editing => match key.code {
                KeyCode::Enter => {
                    if self.queue_editing_source {
                        // Submit
                        let topic = self.queue_input.clone();
                        let source = if self.queue_source_input.is_empty() {
                            None
                        } else {
                            Some(self.queue_source_input.clone())
                        };
                        self.input_mode = InputMode::Normal;
                        self.queue_input.clear();
                        self.queue_source_input.clear();
                        return Some(AppAction::AddToQueue(topic, source));
                    } else {
                        // Move to source input
                        self.queue_editing_source = true;
                    }
                    None
                }
                KeyCode::Esc => {
                    self.input_mode = InputMode::Normal;
                    self.queue_input.clear();
                    self.queue_source_input.clear();
                    None
                }
                KeyCode::Backspace => {
                    if self.queue_editing_source {
                        self.queue_source_input.pop();
                    } else {
                        self.queue_input.pop();
                    }
                    None
                }
                KeyCode::Tab => {
                    self.queue_editing_source = !self.queue_editing_source;
                    None
                }
                KeyCode::Char(c) => {
                    if self.queue_editing_source {
                        self.queue_source_input.push(c);
                    } else {
                        self.queue_input.push(c);
                    }
                    None
                }
                _ => None,
            },
        }
    }

    fn handle_retry_key(&mut self, key: event::KeyEvent) -> Option<AppAction> {
        match key.code {
            KeyCode::Up | KeyCode::Char('k') => {
                if self.retry_selected > 0 {
                    self.retry_selected -= 1;
                }
                None
            }
            KeyCode::Down | KeyCode::Char('j') => {
                if self.retry_selected < self.retry_videos.len().saturating_sub(1) {
                    self.retry_selected += 1;
                }
                None
            }
            KeyCode::Enter | KeyCode::Char('r') => {
                if !self.retry_videos.is_empty() && !self.retry_in_progress {
                    let video_id = self.retry_videos[self.retry_selected].id.clone();
                    return Some(AppAction::RetryVideo(video_id));
                }
                None
            }
            _ => None,
        }
    }

    fn handle_settings_key(&mut self, key: event::KeyEvent) -> Option<AppAction> {
        match key.code {
            KeyCode::Char('p') => Some(AppAction::TogglePause),
            KeyCode::Char('s') => Some(AppAction::Shutdown),
            KeyCode::Char('r') => Some(AppAction::Refresh),
            _ => None,
        }
    }
}

/// Actions that require async execution
pub enum AppAction {
    Refresh,
    TogglePause,
    Shutdown,
    AddToQueue(String, Option<String>),
    RemoveFromQueue(i32),
    ClearQueue,
    RetryVideo(String),
    DownloadVideo(String),
}
