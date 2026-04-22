use std::path::PathBuf;

use anyhow::{Context, Result};

use crate::config::Config;

/// A discovered session file with metadata.
#[derive(Debug)]
pub struct SessionFile {
    /// Path to the JSONL file.
    pub path: PathBuf,
    /// The project directory name as Claude Code encodes it.
    /// e.g. "-home-you-projects-foo"
    pub project_dir_name: String,
    /// Session ID extracted from filename.
    pub session_id: String,
    /// Whether this is an aside_question subagent (/btw command).
    /// These bypass the minimum turn count since they contain
    /// high-signal personal content in just 1-2 turns.
    pub is_aside: bool,
}

/// Discover all session JSONL files, applying exclusion filters.
///
/// Returns sessions sorted by modification time (oldest first).
/// Includes aside_question subagent files which contain /btw content.
pub fn discover_sessions(
    exclude_projects: &[String],
    limit: Option<usize>,
) -> Result<Vec<SessionFile>> {
    let projects_dir = Config::claude_projects_dir();
    if !projects_dir.exists() {
        return Ok(vec![]);
    }

    // Encode exclusion paths to match directory names
    let encoded_exclusions: Vec<String> = exclude_projects
        .iter()
        .map(|p| encode_project_path(p))
        .collect();

    let mut sessions = Vec::new();

    let project_dirs = std::fs::read_dir(&projects_dir)
        .context("failed to read Claude projects directory")?;

    for entry in project_dirs {
        let entry = entry?;
        let dir_name = entry.file_name().to_string_lossy().to_string();

        if !entry.file_type()?.is_dir() {
            continue;
        }

        // Apply exclusion filter on encoded directory names
        if is_excluded(&dir_name, &encoded_exclusions) {
            tracing::debug!("Excluding project: {dir_name}");
            continue;
        }

        // Find top-level JSONL session files
        let jsonl_files = std::fs::read_dir(entry.path())?;
        for file_entry in jsonl_files {
            let file_entry = file_entry?;
            let file_name = file_entry.file_name().to_string_lossy().to_string();

            if !file_name.ends_with(".jsonl") {
                continue;
            }

            let session_id = file_name.trim_end_matches(".jsonl").to_string();

            sessions.push(SessionFile {
                path: file_entry.path(),
                project_dir_name: dir_name.clone(),
                session_id,
                is_aside: false,
            });
        }

        // Scan for aside_question subagent files in session subdirectories.
        // Structure: <project_dir>/<session_uuid>/subagents/agent-aside_question-*.jsonl
        discover_aside_subagents(&entry.path(), &dir_name, &mut sessions)?;
    }

    // Sort by modification time (oldest first)
    sessions.sort_by_key(|s| {
        s.path
            .metadata()
            .ok()
            .and_then(|m| m.modified().ok())
            .unwrap_or(std::time::SystemTime::UNIX_EPOCH)
    });

    if let Some(limit) = limit {
        sessions.truncate(limit);
    }

    Ok(sessions)
}

/// Find aside_question subagent files within a project directory.
fn discover_aside_subagents(
    project_dir: &std::path::Path,
    project_dir_name: &str,
    sessions: &mut Vec<SessionFile>,
) -> Result<()> {
    // Each session may have a subdirectory with subagents/
    let entries = match std::fs::read_dir(project_dir) {
        Ok(e) => e,
        Err(_) => return Ok(()),
    };

    for entry in entries {
        let entry = entry?;
        if !entry.file_type()?.is_dir() {
            continue;
        }

        let subagents_dir = entry.path().join("subagents");
        if !subagents_dir.exists() {
            continue;
        }

        let subagent_files = match std::fs::read_dir(&subagents_dir) {
            Ok(f) => f,
            Err(_) => continue,
        };

        for sa_entry in subagent_files {
            let sa_entry = sa_entry?;
            let sa_name = sa_entry.file_name().to_string_lossy().to_string();

            // Only pick up aside_question subagents (/btw commands)
            if !sa_name.starts_with("agent-aside_question") || !sa_name.ends_with(".jsonl") {
                continue;
            }

            let session_id = sa_name.trim_end_matches(".jsonl").to_string();

            sessions.push(SessionFile {
                path: sa_entry.path(),
                project_dir_name: project_dir_name.to_string(),
                session_id,
                is_aside: true,
            });
        }
    }

    Ok(())
}

/// Encode a project path to the directory name format Claude Code uses.
/// "/home/alice/projects/foo" → "-home-alice-projects-foo"
fn encode_project_path(path: &str) -> String {
    path.replace('/', "-")
}

/// Check if a directory name matches any encoded exclusion.
fn is_excluded(dir_name: &str, encoded_exclusions: &[String]) -> bool {
    encoded_exclusions.iter().any(|excl| dir_name == excl)
}

/// Print a summary of discovered sessions.
pub fn print_discovery_summary(sessions: &[SessionFile], excluded_count: usize) {
    let mut project_counts: std::collections::HashMap<&str, usize> =
        std::collections::HashMap::new();
    let mut aside_count = 0usize;
    for s in sessions {
        *project_counts.entry(&s.project_dir_name).or_insert(0) += 1;
        if s.is_aside {
            aside_count += 1;
        }
    }

    println!(
        "Discovered {} sessions across {} projects ({} aside/btw):",
        sessions.len(),
        project_counts.len(),
        aside_count,
    );
    let mut projects: Vec<_> = project_counts.into_iter().collect();
    projects.sort_by(|a, b| b.1.cmp(&a.1));
    for (project, count) in &projects {
        println!("  {project}: {count} sessions");
    }
    if excluded_count > 0 {
        println!("  ({excluded_count} project(s) excluded by config)");
    }
}

/// Count how many project directories are excluded.
pub fn count_excluded_projects(exclude_projects: &[String]) -> Result<usize> {
    let projects_dir = Config::claude_projects_dir();
    if !projects_dir.exists() {
        return Ok(0);
    }

    let encoded_exclusions: Vec<String> = exclude_projects
        .iter()
        .map(|p| encode_project_path(p))
        .collect();

    let mut count = 0;
    for entry in std::fs::read_dir(&projects_dir)? {
        let entry = entry?;
        if !entry.file_type()?.is_dir() {
            continue;
        }
        let dir_name = entry.file_name().to_string_lossy().to_string();
        if is_excluded(&dir_name, &encoded_exclusions) {
            count += 1;
        }
    }
    Ok(count)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_project_path() {
        assert_eq!(
            encode_project_path("/home/alice/projects/foo"),
            "-home-alice-projects-foo"
        );
    }

    #[test]
    fn test_is_excluded() {
        let exclusions = vec!["-home-alice-projects-foo".to_string()];
        assert!(is_excluded(
            "-home-alice-projects-foo",
            &exclusions
        ));
        assert!(!is_excluded("-home-alice-projects-bar", &exclusions));
    }
}
