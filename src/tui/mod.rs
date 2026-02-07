//! TUI Dashboard Module
//!
//! A terminal-based dashboard for monitoring and controlling the Animus daemon.

pub mod api_client;
pub mod app;
pub mod ui;

pub use api_client::AnimusClient;
pub use app::{App, AppAction, Tab};
