use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::config::Config;

/// Tracks which session files have been processed, so reconcile
/// only processes new or modified sessions.
///
/// Keyed by the session file path (relative to ~/.claude/projects/).
/// Value is the file's last-modified timestamp at the time we processed it.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct Cursor {
    /// Map of session file path → last-modified unix timestamp (seconds).
    processed: HashMap<String, u64>,
}

impl Cursor {
    /// Load cursor state from disk, or return empty if not found.
    pub fn load() -> Result<Self> {
        let path = Self::path();
        if !path.exists() {
            return Ok(Self::default());
        }
        let contents = std::fs::read_to_string(&path)
            .context("failed to read cursor.json")?;
        serde_json::from_str(&contents).context("failed to parse cursor.json")
    }

    /// Save cursor state to disk.
    pub fn save(&self) -> Result<()> {
        let path = Self::path();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(&path, json).context("failed to write cursor.json")
    }

    /// Check if a session file needs processing.
    /// Returns true if the file is new or has been modified since last processing.
    pub fn needs_processing(&self, session_path: &Path) -> bool {
        let key = self.path_key(session_path);
        let recorded_mtime = match self.processed.get(&key) {
            Some(t) => *t,
            None => return true, // never processed
        };

        let current_mtime = match file_mtime_secs(session_path) {
            Some(t) => t,
            None => return true, // can't read mtime, process to be safe
        };

        current_mtime > recorded_mtime
    }

    /// Mark a session file as processed at its current mtime.
    pub fn mark_processed(&mut self, session_path: &Path) {
        let key = self.path_key(session_path);
        if let Some(mtime) = file_mtime_secs(session_path) {
            self.processed.insert(key, mtime);
        }
    }

    /// Number of tracked sessions.
    pub fn len(&self) -> usize {
        self.processed.len()
    }

    fn path() -> PathBuf {
        Config::state_dir().join("cursor.json")
    }

    /// Convert an absolute path to a stable key relative to the projects dir.
    fn path_key(&self, path: &Path) -> String {
        let projects_dir = Config::claude_projects_dir();
        path.strip_prefix(&projects_dir)
            .unwrap_or(path)
            .to_string_lossy()
            .to_string()
    }
}

fn file_mtime_secs(path: &Path) -> Option<u64> {
    path.metadata()
        .ok()?
        .modified()
        .ok()?
        .duration_since(SystemTime::UNIX_EPOCH)
        .ok()
        .map(|d| d.as_secs())
}
