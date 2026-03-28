use anyhow::Result;

use crate::inference::InferenceBackend;
use super::extract::ExtractedCandidate;

/// Curation prompt for the thinking model. It receives both the original
/// conversation text and the extracted candidates, so it can verify claims
/// against the source.
const CURATION_SYSTEM_PROMPT: &str = r#"You are a quality filter for extracted knowledge about a user. You will receive:
1. The original conversation text
2. A list of candidate facts extracted from it

Your job is to verify each candidate against the conversation and decide what to keep.

KEEP items that are:
- Directly supported by something the USER said or did in the conversation
- Durable — would still be true next week
- Distinctive — not something true of most developers

DROP items that are:
- NOT supported by the conversation text (hallucinated or inferred)
- Trivial or generic ("uses the terminal", "runs commands", "has a computer")
- Ephemeral state ("keyboard is working", "build succeeded")
- Duplicates or near-duplicates of other items in the batch

FIX SCOPE: if an item is labeled "personal" but is really about the project, change to "keep_as_project".

For each item, respond with one of:
- "keep" — item is verified and worth storing
- "keep_as_project" — item is valid but should be project scope
- "drop" — item should not be stored (hallucinated, trivial, or ephemeral)
- "merge:N" — merge with item N (0-indexed) as they say the same thing

Respond with ONLY a JSON array: [{"index": 0, "action": "keep"}, {"index": 1, "action": "drop"}, ...]"#;

/// Run the curation pass on a batch of extraction candidates.
/// The conversation_text is provided so the curation model can verify
/// claims against the source material.
pub fn curate_candidates(
    backend: &dyn InferenceBackend,
    candidates: &[ExtractedCandidate],
    conversation_text: &str,
    max_tokens: u32,
) -> Result<Vec<ExtractedCandidate>> {
    if candidates.is_empty() {
        return Ok(vec![]);
    }

    // Format candidates as numbered list
    let mut items = String::new();
    for (i, c) in candidates.iter().enumerate() {
        items.push_str(&format!(
            "{}. [scope={}, category={}] {}\n",
            i, c.scope, c.category, c.content,
        ));
    }

    // Truncate conversation text if very long — the curation model needs
    // enough to verify claims but doesn't need every detail
    let max_conv_chars = 8000;
    let conv_text = if conversation_text.len() > max_conv_chars {
        &conversation_text[..max_conv_chars]
    } else {
        conversation_text
    };

    let user_prompt = format!(
        "CONVERSATION:\n{}\n\nCANDIDATE EXTRACTIONS ({} items):\n{}",
        conv_text,
        candidates.len(),
        items,
    );

    let response = backend.generate_chat(CURATION_SYSTEM_PROMPT, &user_prompt, max_tokens)?;

    let actions = parse_curation_response(&response, candidates.len());

    // Apply actions
    let mut kept = Vec::new();
    let mut merge_targets: std::collections::HashSet<usize> = std::collections::HashSet::new();

    // First pass: identify merge targets
    for action in &actions {
        if let CurationAction::Merge(_) = action.action {
            merge_targets.insert(action.index);
        }
    }

    // Second pass: apply keep/drop/scope changes
    for action in &actions {
        if action.index >= candidates.len() {
            continue;
        }

        match action.action {
            CurationAction::Keep => {
                if !merge_targets.contains(&action.index) {
                    kept.push(candidates[action.index].clone());
                }
            }
            CurationAction::KeepAsProject => {
                if !merge_targets.contains(&action.index) {
                    let mut c = candidates[action.index].clone();
                    tracing::info!(
                        "Curation: demoting to project: {}",
                        &c.content[..c.content.len().min(80)],
                    );
                    c.scope = "project".to_string();
                    kept.push(c);
                }
            }
            CurationAction::Drop => {
                let c = &candidates[action.index];
                tracing::info!(
                    "Curation: dropping: {}",
                    &c.content[..c.content.len().min(80)],
                );
            }
            CurationAction::Merge(target) => {
                let c = &candidates[action.index];
                tracing::info!(
                    "Curation: merging into item {}: {}",
                    target,
                    &c.content[..c.content.len().min(80)],
                );
            }
        }
    }

    let dropped = candidates.len() - kept.len();
    if dropped > 0 {
        tracing::info!(
            "Curation: kept {}/{} candidates ({} dropped)",
            kept.len(),
            candidates.len(),
            dropped,
        );
    }

    Ok(kept)
}

#[derive(Debug, Clone)]
enum CurationAction {
    Keep,
    KeepAsProject,
    Drop,
    Merge(usize),
}

#[derive(Debug)]
struct CurationDecision {
    index: usize,
    action: CurationAction,
}

/// Parse the curation model's response. Falls back to keeping everything
/// if the response can't be parsed.
fn parse_curation_response(response: &str, num_candidates: usize) -> Vec<CurationDecision> {
    let trimmed = response.trim();

    let parsed = try_parse_curation(trimmed)
        .or_else(|| {
            let stripped = super::extract::strip_code_fences(trimmed);
            try_parse_curation(&stripped)
        })
        .or_else(|| {
            let start = trimmed.find('[')?;
            let end = trimmed.rfind(']')?;
            try_parse_curation(&trimmed[start..=end])
        });

    match parsed {
        Some(decisions) => decisions,
        None => {
            tracing::warn!(
                "Curation response unparseable, keeping all candidates: {}",
                &trimmed[..trimmed.len().min(200)]
            );
            (0..num_candidates)
                .map(|i| CurationDecision {
                    index: i,
                    action: CurationAction::Keep,
                })
                .collect()
        }
    }
}

fn try_parse_curation(text: &str) -> Option<Vec<CurationDecision>> {
    let arr: Vec<serde_json::Value> = serde_json::from_str(text).ok()?;

    let mut decisions = Vec::new();
    for item in &arr {
        let index = item.get("index")?.as_u64()? as usize;
        let action_str = item.get("action")?.as_str()?;

        let action = if action_str == "keep" {
            CurationAction::Keep
        } else if action_str == "keep_as_project" {
            CurationAction::KeepAsProject
        } else if action_str == "drop" {
            CurationAction::Drop
        } else if action_str.starts_with("merge:") {
            let target: usize = action_str[6..].parse().ok()?;
            CurationAction::Merge(target)
        } else {
            CurationAction::Keep
        };

        decisions.push(CurationDecision { index, action });
    }

    Some(decisions)
}
