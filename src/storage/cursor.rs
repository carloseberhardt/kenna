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
/// Stores both mtime and byte offset — JSONL files are append-only,
/// so on re-runs we skip to the byte offset and only process new content.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct Cursor {
    /// Map of session file path → processing state.
    /// Supports legacy format (bare u64 mtime) via custom deserialization.
    processed: HashMap<String, SessionState>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionState {
    /// File mtime (unix seconds) when last processed.
    pub mtime: u64,
    /// Byte offset up to which the file has been processed.
    /// On next run, preprocessing starts reading from this offset.
    pub byte_offset: u64,
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

        // Try new format first, fall back to legacy (bare u64 mtimes)
        if let Ok(cursor) = serde_json::from_str::<Cursor>(&contents) {
            return Ok(cursor);
        }

        // Legacy migration: old format was HashMap<String, u64>
        if let Ok(legacy) = serde_json::from_str::<HashMap<String, u64>>(&contents) {
            let processed = legacy.into_iter()
                .map(|(k, mtime)| (k, SessionState { mtime, byte_offset: 0 }))
                .collect();
            return Ok(Cursor { processed });
        }

        Ok(Self::default())
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
    /// Returns true if the file is new or has grown since last processing.
    pub fn needs_processing(&self, session_path: &Path) -> bool {
        let key = self.path_key(session_path);
        let state = match self.processed.get(&key) {
            Some(s) => s,
            None => return true, // never processed
        };

        let current_mtime = match file_mtime_secs(session_path) {
            Some(t) => t,
            None => return true,
        };

        if current_mtime > state.mtime {
            return true; // file modified
        }

        // Also check if file has grown (mtime might not change on some filesystems)
        let current_size = file_size(session_path).unwrap_or(0);
        current_size > state.byte_offset
    }

    /// Get the byte offset to resume processing from.
    /// Returns 0 for new/unknown files.
    pub fn byte_offset(&self, session_path: &Path) -> u64 {
        let key = self.path_key(session_path);
        self.processed.get(&key)
            .map(|s| s.byte_offset)
            .unwrap_or(0)
    }

    /// Mark a session file as processed up to the given byte offset.
    pub fn mark_processed(&mut self, session_path: &Path) {
        let key = self.path_key(session_path);
        let mtime = file_mtime_secs(session_path).unwrap_or(0);
        let byte_offset = file_size(session_path).unwrap_or(0);
        self.processed.insert(key, SessionState { mtime, byte_offset });
    }

    /// Number of tracked sessions.
    #[allow(dead_code)]
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

fn file_size(path: &Path) -> Option<u64> {
    path.metadata().ok().map(|m| m.len())
}
