//! Control API for the daemon

pub mod auth;
mod control;

pub use control::{create_router, AppState, DaemonStatus};
