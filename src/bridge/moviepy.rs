//! MoviePy bridge utilities
//! 
//! The actual processing is done by the Python script at bridge/moviepy_bridge.py
//! This module provides Rust utilities for interacting with it.

use std::path::Path;

/// Check if the MoviePy bridge script exists
pub fn bridge_script_exists(path: &str) -> bool {
    Path::new(path).exists()
}

/// Default path to the bridge script
pub const DEFAULT_BRIDGE_PATH: &str = "src/bridge/moviepy_bridge.py";
