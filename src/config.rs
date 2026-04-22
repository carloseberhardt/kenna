use std::path::PathBuf;

use anyhow::{Context, Result};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct Config {
    /// Default log level: "error", "warn", "info", "debug", "trace".
    /// Overridden by RUST_LOG environment variable if set.
    pub log_level: String,
    pub confidence_drop_threshold: f32,
    pub confidence_auto_accept_threshold: f32,
    pub extraction_model: String,
    pub curation_model: String,
    pub embedding_model: String,
    pub embedding_dimensions: usize,
    /// Cosine similarity above which two memories are considered duplicates.
    pub dedup_cosine_threshold: f32,
    /// Cosine similarity range for supersession (same entity, updated claim).
    /// Min: lower bound (below this, facts coexist). Max: upper bound (above this, it's a duplicate).
    pub supersession_cosine_min: f32,
    pub supersession_cosine_max: f32,
    /// Scope confidence below which personal-scoped items are demoted to project.
    pub scope_demotion_threshold: f32,
    /// Model for settling synthesis. Defaults to curation_model if not set.
    pub settling_model: Option<String>,
    /// Minimum free VRAM (GB) required before reconcile will start.
    /// Checked via rocm-smi. If rocm-smi is unavailable, the check is skipped.
    /// Reconcile loads Gemma (~3GB) then Qwen (~5GB) sequentially, so peak need is ~6GB.
    pub reconcile_min_free_vram_gb: f64,
    /// Minimum free VRAM (GB) required before settle will start.
    /// Default is higher than reconcile to leave room for larger synthesis models.
    pub settle_min_free_vram_gb: f64,
    pub reconcile: ReconcileConfig,
    pub settle: SettleConfig,
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
            log_level: "warn".into(),
            confidence_drop_threshold: 0.6,
            confidence_auto_accept_threshold: 0.85,
            extraction_model: "gemma-3-4b-it-Q6_K.gguf".into(),
            curation_model: "qwen3-8b-q4_k_m.gguf".into(),
            embedding_model: "nomic-embed-text-v1.5.Q8_0.gguf".into(),
            embedding_dimensions: 768,
            settling_model: None,
            reconcile_min_free_vram_gb: 7.0,
            settle_min_free_vram_gb: 7.0,
            dedup_cosine_threshold: 0.85,
            supersession_cosine_min: 0.7,
            supersession_cosine_max: 0.85,
            scope_demotion_threshold: 0.75,
            reconcile: ReconcileConfig::default(),
            settle: SettleConfig::default(),
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

#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct SettleConfig {
    /// Minimum distinct projects for a project-scoped pattern to be promoted to personal.
    pub min_projects_for_promotion: usize,
    /// Minimum non-superseded memories under an entity to trigger synthesis.
    pub min_memories_for_synthesis: usize,
    /// Cosine similarity threshold for cross-project clustering.
    pub cluster_cosine_threshold: f32,
    /// Maximum cluster size to prevent chaining artifacts.
    pub max_cluster_size: usize,
}

impl Default for SettleConfig {
    fn default() -> Self {
        Self {
            min_projects_for_promotion: 3,
            min_memories_for_synthesis: 3,
            cluster_cosine_threshold: 0.75,
            max_cluster_size: 10,
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
            .join("kenna")
            .join("config.toml")
    }

    pub fn data_dir() -> PathBuf {
        dirs::data_dir()
            .unwrap_or_else(|| PathBuf::from("~/.local/share"))
            .join("kenna")
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
