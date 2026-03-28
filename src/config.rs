use std::path::PathBuf;

use anyhow::{Context, Result};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct Config {
    pub confidence_drop_threshold: f32,
    pub confidence_auto_accept_threshold: f32,
    pub extraction_model: String,
    pub curation_model: String,
    pub embedding_model: String,
    pub embedding_dimensions: usize,
    pub reconcile: ReconcileConfig,
}

#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct ReconcileConfig {
    /// Project paths to exclude from extraction.
    /// Matched against the original project path decoded from the JSONL directory name.
    pub exclude_projects: Vec<String>,
    /// Minimum number of human turns for a session to be processed.
    pub min_human_turns: usize,
    /// Approximate max tokens per chunk sent to the extraction model.
    pub chunk_max_tokens: usize,
    /// Token overlap between chunks.
    pub chunk_overlap_tokens: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            confidence_drop_threshold: 0.6,
            confidence_auto_accept_threshold: 0.85,
            extraction_model: "gemma-3-4b-it-Q6_K.gguf".into(),
            curation_model: "qwen3-8b-q4_k_m.gguf".into(),
            embedding_model: "nomic-embed-text-v1.5.Q8_0.gguf".into(),
            embedding_dimensions: 768,
            reconcile: ReconcileConfig::default(),
        }
    }
}

impl Default for ReconcileConfig {
    fn default() -> Self {
        Self {
            exclude_projects: vec![],
            min_human_turns: 4,
            chunk_max_tokens: 4000,
            chunk_overlap_tokens: 200,
        }
    }
}

impl Config {
    pub fn load() -> Result<Self> {
        let path = Self::config_path();
        if path.exists() {
            let contents =
                std::fs::read_to_string(&path).context("failed to read config.toml")?;
            toml::from_str(&contents).context("failed to parse config.toml")
        } else {
            Ok(Self::default())
        }
    }

    pub fn config_path() -> PathBuf {
        dirs::config_dir()
            .unwrap_or_else(|| PathBuf::from("~/.config"))
            .join("engram")
            .join("config.toml")
    }

    pub fn data_dir() -> PathBuf {
        dirs::data_dir()
            .unwrap_or_else(|| PathBuf::from("~/.local/share"))
            .join("engram")
    }

    pub fn db_path() -> PathBuf {
        Self::data_dir().join("db")
    }

    pub fn models_dir() -> PathBuf {
        Self::data_dir().join("models")
    }

    pub fn state_dir() -> PathBuf {
        Self::data_dir().join("state")
    }

    pub fn logs_dir() -> PathBuf {
        Self::data_dir().join("logs")
    }

    /// Path to Claude Code's project sessions directory.
    pub fn claude_projects_dir() -> PathBuf {
        dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("~"))
            .join(".claude")
            .join("projects")
    }

    /// Ensure all data directories exist.
    pub fn ensure_dirs() -> Result<()> {
        for dir in [
            Self::db_path(),
            Self::models_dir(),
            Self::state_dir(),
            Self::logs_dir(),
        ] {
            std::fs::create_dir_all(&dir)
                .with_context(|| format!("failed to create directory: {}", dir.display()))?;
        }
        Ok(())
    }
}
