//! DSPy bridge for the optimized Judge module
//!
//! This module provides a Rust interface to the Python DSPy bridge,
//! which handles script evaluation using an optimized predictor.
//!
//! The Judge learns to predict real-world performance (views, retention, likes)
//! from historical (script, performance_score) data.

use serde::{Deserialize, Serialize};
use std::path::Path;
use std::process::Stdio;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::process::Command;
use tracing::{debug, error, info, warn};

/// Default path to the DSPy bridge script
pub const DEFAULT_BRIDGE_PATH: &str = "src/bridge/dspy_bridge.py";

/// Check if the DSPy bridge script exists
pub fn bridge_script_exists(path: &str) -> bool {
    Path::new(path).exists()
}

/// Result from the DSPy Judge evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DspyEvaluationResult {
    pub overall_score: f32,
    pub criteria: serde_json::Value,
    pub strengths: Vec<String>,
    pub weaknesses: Vec<String>,
    pub ai_telltale_signs: Vec<String>,
    pub specific_improvements: Vec<serde_json::Value>,
    #[serde(default)]
    pub dspy_metadata: Option<DspyMetadata>,
}

/// Metadata from the DSPy evaluation
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DspyMetadata {
    #[serde(default)]
    pub predicted_score: Option<f32>,
    #[serde(default)]
    pub confidence: Option<f32>,
    #[serde(default)]
    pub reasoning: Option<String>,
    #[serde(default)]
    pub is_compiled: bool,
    #[serde(default)]
    pub is_fallback: bool,
}

/// Response wrapper from the bridge
#[derive(Debug, Deserialize)]
struct BridgeResponse {
    success: bool,
    #[serde(default)]
    evaluation: Option<DspyEvaluationResult>,
    #[serde(default)]
    error: Option<String>,
}

/// Client for the DSPy bridge
#[derive(Clone)]
pub struct DspyBridge {
    bridge_path: String,
    python_executable: String,
}

impl DspyBridge {
    /// Create a new DSPy bridge client
    pub fn new() -> Self {
        Self {
            bridge_path: DEFAULT_BRIDGE_PATH.to_string(),
            python_executable: std::env::var("PYTHON_EXECUTABLE")
                .unwrap_or_else(|_| "python3".to_string()),
        }
    }

    /// Create with custom paths
    pub fn with_paths(bridge_path: String, python_executable: String) -> Self {
        Self {
            bridge_path,
            python_executable,
        }
    }

    /// Check if the bridge is available
    pub fn is_available(&self) -> bool {
        bridge_script_exists(&self.bridge_path)
    }

    /// Evaluate a script using the DSPy Judge
    ///
    /// This calls the Python bridge via subprocess, passing the script and topic_brief
    /// as JSON via stdin, and receiving the evaluation as JSON via stdout.
    pub async fn evaluate_script(
        &self,
        script: &serde_json::Value,
        topic_brief: &serde_json::Value,
    ) -> Result<DspyEvaluationResult, String> {
        if !self.is_available() {
            return Err(format!("DSPy bridge not found at {}", self.bridge_path));
        }

        let input = serde_json::json!({
            "action": "evaluate",
            "script": script,
            "topic_brief": topic_brief,
        });

        let input_json = serde_json::to_string(&input)
            .map_err(|e| format!("Failed to serialize input: {}", e))?;

        debug!("Calling DSPy bridge for script evaluation");

        let mut child = Command::new(&self.python_executable)
            .arg(&self.bridge_path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| format!("Failed to spawn DSPy bridge: {}", e))?;

        // Write input to stdin
        if let Some(mut stdin) = child.stdin.take() {
            stdin
                .write_all(input_json.as_bytes())
                .await
                .map_err(|e| format!("Failed to write to bridge stdin: {}", e))?;
        }

        // Wait for completion and read output
        let output = child
            .wait_with_output()
            .await
            .map_err(|e| format!("Bridge execution failed: {}", e))?;

        // Log stderr for debugging
        if !output.stderr.is_empty() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            for line in stderr.lines() {
                debug!("DSPy bridge: {}", line);
            }
        }

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(format!("DSPy bridge failed: {}", stderr));
        }

        // Parse response
        let stdout = String::from_utf8_lossy(&output.stdout);
        let response: BridgeResponse = serde_json::from_str(&stdout)
            .map_err(|e| format!("Failed to parse bridge response: {} (raw: {})", e, stdout))?;

        if !response.success {
            return Err(response.error.unwrap_or_else(|| "Unknown error".to_string()));
        }

        response
            .evaluation
            .ok_or_else(|| "No evaluation in response".to_string())
    }

    /// Check the health of the DSPy bridge
    pub async fn health_check(&self) -> Result<DspyHealthStatus, String> {
        if !self.is_available() {
            return Ok(DspyHealthStatus {
                available: false,
                dspy_installed: false,
                is_compiled: false,
                model: None,
                error: Some(format!("Bridge not found at {}", self.bridge_path)),
            });
        }

        let input = serde_json::json!({
            "action": "health",
        });

        let input_json = serde_json::to_string(&input)
            .map_err(|e| format!("Failed to serialize input: {}", e))?;

        let mut child = Command::new(&self.python_executable)
            .arg(&self.bridge_path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| format!("Failed to spawn DSPy bridge: {}", e))?;

        if let Some(mut stdin) = child.stdin.take() {
            stdin
                .write_all(input_json.as_bytes())
                .await
                .map_err(|e| format!("Failed to write to bridge stdin: {}", e))?;
        }

        let output = child
            .wait_with_output()
            .await
            .map_err(|e| format!("Bridge execution failed: {}", e))?;

        if !output.status.success() {
            return Ok(DspyHealthStatus {
                available: true,
                dspy_installed: false,
                is_compiled: false,
                model: None,
                error: Some(String::from_utf8_lossy(&output.stderr).to_string()),
            });
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        
        #[derive(Deserialize)]
        struct HealthResponse {
            success: bool,
            #[serde(default)]
            dspy_available: bool,
            #[serde(default)]
            is_compiled: bool,
            #[serde(default)]
            model: Option<String>,
        }

        let response: HealthResponse = serde_json::from_str(&stdout)
            .map_err(|e| format!("Failed to parse health response: {}", e))?;

        Ok(DspyHealthStatus {
            available: true,
            dspy_installed: response.dspy_available,
            is_compiled: response.is_compiled,
            model: response.model,
            error: None,
        })
    }
}

impl Default for DspyBridge {
    fn default() -> Self {
        Self::new()
    }
}

/// Health status of the DSPy bridge
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DspyHealthStatus {
    pub available: bool,
    pub dspy_installed: bool,
    pub is_compiled: bool,
    pub model: Option<String>,
    pub error: Option<String>,
}
