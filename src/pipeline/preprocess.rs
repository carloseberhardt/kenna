use std::io::BufRead;
use std::path::Path;

use anyhow::{Context, Result};
use serde_json::Value;

use crate::config::ReconcileConfig;

/// A preprocessed conversation chunk ready for extraction.
#[derive(Debug)]
pub struct ConversationChunk {
    /// The session ID this chunk came from.
    pub session_id: String,
    /// The project directory name.
    pub project_dir_name: String,
    /// Chunk index within the session (0-based).
    pub chunk_index: usize,
    /// The preprocessed conversation text.
    pub text: String,
    /// Approximate token count (word-based estimate).
    pub approx_tokens: usize,
}

/// A single conversation turn extracted from JSONL.
struct Turn {
    role: String, // "user" or "assistant"
    text: String,
}

/// Preprocess a session JSONL file into conversation chunks.
///
/// Returns an empty vec if the session doesn't meet the minimum turn count.
/// For aside sessions (is_aside=true), the turn minimum is bypassed and
/// only the aside-specific turns are extracted (not the replayed context).
pub fn preprocess_session(
    path: &Path,
    session_id: &str,
    project_dir_name: &str,
    config: &ReconcileConfig,
    is_aside: bool,
    byte_offset: u64,
) -> Result<Vec<ConversationChunk>> {
    let mut file = std::fs::File::open(path)
        .with_context(|| format!("failed to open session: {}", path.display()))?;

    // Skip to the byte offset — only process new content since last run.
    // JSONL is append-only, so everything before the offset has been processed.
    if byte_offset > 0 {
        use std::io::Seek;
        file.seek(std::io::SeekFrom::Start(byte_offset))?;
    }

    let reader = std::io::BufReader::new(file);

    let mut turns: Vec<Turn> = Vec::new();
    let mut human_turn_count = 0usize;

    for line in reader.lines() {
        let line = line?;
        if line.is_empty() {
            continue;
        }

        let record: Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(_) => continue, // skip malformed lines
        };

        let record_type = record.get("type").and_then(|t| t.as_str()).unwrap_or("");

        match record_type {
            "user" => {
                if let Some(text) = extract_user_text(&record) {
                    if !text.is_empty() {
                        human_turn_count += 1;
                        turns.push(Turn {
                            role: "Human".into(),
                            text,
                        });
                    }
                }
            }
            "assistant" => {
                if let Some(text) = extract_assistant_text(&record) {
                    if !text.is_empty() {
                        turns.push(Turn {
                            role: "Assistant".into(),
                            text,
                        });
                    }
                }
            }
            _ => continue, // skip system, progress, file-history-snapshot, etc.
        }
    }

    // Gate by minimum human turns (aside sessions bypass this)
    // Skip sessions with too few human turns — but only on first processing.
    // Incremental runs (byte_offset > 0) may have fewer turns in the new content;
    // the session already passed the gate on the initial run.
    if !is_aside && byte_offset == 0 && human_turn_count < config.min_human_turns {
        return Ok(vec![]);
    }

    let turns = if is_aside {
        // For aside/btw sessions, the JSONL contains the full parent session
        // context replayed, then the aside-specific turns at the end.
        // Extract only the aside-specific content: the last user message
        // containing <system-reminder>...side question... and everything after.
        extract_aside_turns(turns)
    } else {
        turns
    };

    if turns.is_empty() {
        return Ok(vec![]);
    }

    // Merge consecutive same-role turns
    let merged = merge_consecutive_turns(turns);

    // Skip if no meaningful content remains
    if merged.iter().all(|t| t.text.trim().is_empty()) {
        return Ok(vec![]);
    }

    // Format as conversation text
    let full_text = format_conversation(&merged);

    // Chunk if needed
    let chunks = chunk_text(
        &full_text,
        session_id,
        project_dir_name,
        config.chunk_max_tokens,
        config.chunk_overlap_tokens,
    );

    Ok(chunks)
}

/// Extract displayable text from a user message.
fn extract_user_text(record: &Value) -> Option<String> {
    let content = record.get("message")?.get("content")?;

    match content {
        Value::String(s) => {
            // Skip command messages (e.g. <command-message>init</command-message>)
            if s.contains("<command-message>") || s.contains("<command-name>") {
                return None;
            }
            Some(strip_xml_noise(s))
        }
        Value::Array(blocks) => {
            let mut parts = Vec::new();
            for block in blocks {
                let block_type = block.get("type").and_then(|t| t.as_str()).unwrap_or("");
                match block_type {
                    "text" => {
                        if let Some(text) = block.get("text").and_then(|t| t.as_str()) {
                            // Skip system-reminder blocks embedded in user messages
                            if text.contains("<system-reminder>") {
                                continue;
                            }
                            let cleaned = strip_xml_noise(text);
                            if !cleaned.trim().is_empty() {
                                parts.push(cleaned);
                            }
                        }
                    }
                    "tool_result" => {
                        // Summarize tool results — don't include full content
                        let tool_id = block
                            .get("tool_use_id")
                            .and_then(|t| t.as_str())
                            .unwrap_or("unknown");
                        let content_size = block
                            .get("content")
                            .map(|c| c.to_string().len())
                            .unwrap_or(0);
                        if content_size > 500 {
                            parts.push(format!(
                                "[tool result: {tool_id}, {content_size} bytes]"
                            ));
                        }
                        // Skip small tool results entirely — they're noise
                    }
                    _ => continue,
                }
            }
            if parts.is_empty() {
                None
            } else {
                Some(parts.join("\n"))
            }
        }
        _ => None,
    }
}

/// Extract displayable text from an assistant message.
fn extract_assistant_text(record: &Value) -> Option<String> {
    let content = record.get("message")?.get("content")?;

    let blocks = content.as_array()?;
    let mut parts = Vec::new();

    for block in blocks {
        let block_type = block.get("type").and_then(|t| t.as_str()).unwrap_or("");
        match block_type {
            "text" => {
                if let Some(text) = block.get("text").and_then(|t| t.as_str()) {
                    parts.push(truncate_code_blocks(text));
                }
            }
            "tool_use" => {
                // Summarize tool usage
                let tool_name = block
                    .get("name")
                    .and_then(|n| n.as_str())
                    .unwrap_or("unknown");
                parts.push(format!("[used tool: {tool_name}]"));
            }
            // Skip "thinking" blocks entirely
            _ => continue,
        }
    }

    if parts.is_empty() {
        None
    } else {
        Some(parts.join("\n"))
    }
}

/// Extract only the aside-specific turns from a subagent session.
///
/// Aside subagent files contain the full parent session replayed as context,
/// followed by the aside-specific user message and the assistant's response.
/// We take the last user turn and any assistant turns after it.
/// The system-reminder noise in the aside user message has already been
/// stripped by extract_user_text → strip_xml_noise.
fn extract_aside_turns(turns: Vec<Turn>) -> Vec<Turn> {
    // Find the last Human turn — this is the aside user message.
    // Everything before it is replayed parent context.
    let last_human = turns.iter().rposition(|t| t.role == "Human");

    let Some(start_idx) = last_human else {
        return vec![];
    };

    let mut aside_turns = Vec::new();

    let user_text = turns[start_idx].text.trim();
    if !user_text.is_empty() {
        aside_turns.push(Turn {
            role: "Human".into(),
            text: user_text.to_string(),
        });
    }

    // Include any assistant turns after the last human turn
    for turn in &turns[start_idx + 1..] {
        if turn.role == "Assistant" && !turn.text.trim().is_empty() {
            aside_turns.push(Turn {
                role: "Assistant".into(),
                text: turn.text.clone(),
            });
        }
    }

    aside_turns
}

/// Strip XML-style noise tags injected by Claude Code.
fn strip_xml_noise(text: &str) -> String {
    // These tags are Claude Code harness artifacts, not user conversation
    let noise_patterns = [
        "<local-command-caveat>",
        "</local-command-caveat>",
        "<local-command-stdout>",
        "</local-command-stdout>",
        "<local-command-stderr>",
        "</local-command-stderr>",
        "<bash-input>",
        "</bash-input>",
        "<bash-stdout>",
        "</bash-stdout>",
        "<bash-stderr>",
        "</bash-stderr>",
    ];

    let mut result = text.to_string();

    // Remove full XML tag blocks (tag + content + closing tag) for caveat/reminder
    for tag in ["local-command-caveat", "system-reminder"] {
        while let Some(start) = result.find(&format!("<{tag}>")) {
            if let Some(end) = result.find(&format!("</{tag}>")) {
                let end = end + format!("</{tag}>").len();
                result.replace_range(start..end, "");
            } else {
                break;
            }
        }
    }

    // Remove individual noise tags
    for pattern in &noise_patterns {
        result = result.replace(pattern, "");
    }

    result
}

/// Replace code blocks longer than 20 lines with a summary.
fn truncate_code_blocks(text: &str) -> String {
    let mut result = String::new();
    let mut in_code_block = false;
    let mut code_lines: Vec<String> = Vec::new();
    let mut lang = String::new();

    for line in text.lines() {
        if line.starts_with("```") {
            if in_code_block {
                // End of code block
                if code_lines.len() > 20 {
                    let l = if lang.is_empty() { "code" } else { &lang };
                    result.push_str(&format!(
                        "[code block: {l}, {} lines]\n",
                        code_lines.len()
                    ));
                } else {
                    result.push_str(&format!("```{lang}\n"));
                    for cl in &code_lines {
                        result.push_str(cl);
                        result.push('\n');
                    }
                    result.push_str("```\n");
                }
                code_lines.clear();
                lang.clear();
                in_code_block = false;
            } else {
                // Start of code block
                in_code_block = true;
                lang = line.trim_start_matches('`').to_string();
            }
        } else if in_code_block {
            code_lines.push(line.to_string());
        } else {
            result.push_str(line);
            result.push('\n');
        }
    }

    // Handle unclosed code block
    if in_code_block && !code_lines.is_empty() {
        if code_lines.len() > 20 {
            let l = if lang.is_empty() { "code" } else { &lang };
            result.push_str(&format!(
                "[code block: {l}, {} lines]\n",
                code_lines.len()
            ));
        } else {
            result.push_str(&format!("```{lang}\n"));
            for cl in &code_lines {
                result.push_str(cl);
                result.push('\n');
            }
        }
    }

    result
}

/// Merge consecutive turns from the same role.
fn merge_consecutive_turns(turns: Vec<Turn>) -> Vec<Turn> {
    let mut merged: Vec<Turn> = Vec::new();
    for turn in turns {
        if let Some(last) = merged.last_mut() {
            if last.role == turn.role {
                last.text.push('\n');
                last.text.push_str(&turn.text);
                continue;
            }
        }
        merged.push(turn);
    }
    merged
}

/// Format turns into a conversation string.
fn format_conversation(turns: &[Turn]) -> String {
    let mut text = String::new();
    for turn in turns {
        text.push_str(&turn.role);
        text.push_str(": ");
        text.push_str(&turn.text);
        text.push_str("\n\n");
    }
    text
}

/// Approximate token count using word-based heuristic (~1.3 tokens per word).
fn approx_token_count(text: &str) -> usize {
    // Use chars/4 as a conservative estimate. The word-based estimate (words * 1.3)
    // severely undercounts for paths, UUIDs, base64, and other non-prose content
    // that tokenizes into many subword pieces per whitespace-delimited "word".
    // chars/4 is the standard rough estimate and errs on the side of smaller chunks,
    // which is safer than overflowing the model's context.
    text.len() / 4
}

/// Split text into chunks of approximately max_tokens, with overlap.
fn chunk_text(
    text: &str,
    session_id: &str,
    project_dir_name: &str,
    max_tokens: usize,
    overlap_tokens: usize,
) -> Vec<ConversationChunk> {
    let total_tokens = approx_token_count(text);

    if total_tokens <= max_tokens {
        return vec![ConversationChunk {
            session_id: session_id.to_string(),
            project_dir_name: project_dir_name.to_string(),
            chunk_index: 0,
            text: text.to_string(),
            approx_tokens: total_tokens,
        }];
    }

    // Split on double-newlines (turn boundaries) for natural chunk boundaries
    let paragraphs: Vec<&str> = text.split("\n\n").collect();
    let mut chunks = Vec::new();
    let mut current_chunk = String::new();
    let mut current_tokens = 0usize;
    let mut chunk_index = 0usize;
    // Track paragraphs for overlap
    let mut recent_paragraphs: Vec<String> = Vec::new();

    for para in &paragraphs {
        let para_tokens = approx_token_count(para);

        // Guard: if a single paragraph exceeds max_tokens, force-split it
        // on single newlines. This handles huge tool outputs, base64 blobs, etc.
        if para_tokens > max_tokens {
            // Emit current chunk first if non-empty
            if !current_chunk.is_empty() {
                chunks.push(ConversationChunk {
                    session_id: session_id.to_string(),
                    project_dir_name: project_dir_name.to_string(),
                    chunk_index,
                    text: current_chunk.clone(),
                    approx_tokens: current_tokens,
                });
                chunk_index += 1;
                current_chunk.clear();
                current_tokens = 0;
            }

            // Split the oversized paragraph on single newlines
            let lines: Vec<&str> = para.split('\n').collect();
            for line in &lines {
                let line_tokens = approx_token_count(line);
                if current_tokens + line_tokens > max_tokens && !current_chunk.is_empty() {
                    chunks.push(ConversationChunk {
                        session_id: session_id.to_string(),
                        project_dir_name: project_dir_name.to_string(),
                        chunk_index,
                        text: current_chunk.clone(),
                        approx_tokens: current_tokens,
                    });
                    chunk_index += 1;
                    current_chunk.clear();
                    current_tokens = 0;
                }
                current_chunk.push_str(line);
                current_chunk.push('\n');
                current_tokens += line_tokens;
            }
            recent_paragraphs.clear();
            continue;
        }

        if current_tokens + para_tokens > max_tokens && !current_chunk.is_empty() {
            // Emit current chunk
            chunks.push(ConversationChunk {
                session_id: session_id.to_string(),
                project_dir_name: project_dir_name.to_string(),
                chunk_index,
                text: current_chunk.clone(),
                approx_tokens: current_tokens,
            });
            chunk_index += 1;

            // Start next chunk with overlap from recent paragraphs
            current_chunk.clear();
            current_tokens = 0;
            for rp in &recent_paragraphs {
                let rp_tokens = approx_token_count(rp);
                if current_tokens + rp_tokens > overlap_tokens {
                    break;
                }
                current_chunk.push_str(rp);
                current_chunk.push_str("\n\n");
                current_tokens += rp_tokens;
            }
        }

        current_chunk.push_str(para);
        current_chunk.push_str("\n\n");
        current_tokens += para_tokens;

        recent_paragraphs.push(para.to_string());
        // Keep only enough recent paragraphs for overlap
        while recent_paragraphs.len() > 5 {
            recent_paragraphs.remove(0);
        }
    }

    // Emit final chunk
    if !current_chunk.is_empty() {
        chunks.push(ConversationChunk {
            session_id: session_id.to_string(),
            project_dir_name: project_dir_name.to_string(),
            chunk_index,
            text: current_chunk,
            approx_tokens: current_tokens,
        });
    }

    chunks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_truncate_code_blocks_short() {
        let input = "Here:\n```rust\nfn main() {}\n```\nDone.";
        let result = truncate_code_blocks(input);
        assert!(result.contains("fn main()"));
    }

    #[test]
    fn test_truncate_code_blocks_long() {
        let mut input = "Here:\n```python\n".to_string();
        for i in 0..25 {
            input.push_str(&format!("line {i}\n"));
        }
        input.push_str("```\nDone.");
        let result = truncate_code_blocks(&input);
        assert!(result.contains("[code block: python, 25 lines]"));
        assert!(!result.contains("line 10"));
    }

    #[test]
    fn test_approx_token_count() {
        let text = "This is a test with about ten words here now";
        let tokens = approx_token_count(text);
        assert!(tokens > 10);
        assert!(tokens < 20);
    }
}
