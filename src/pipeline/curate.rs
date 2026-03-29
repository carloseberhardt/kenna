use anyhow::Result;

use crate::inference::InferenceBackend;
use super::extract::ExtractedCandidate;

/// Curation prompt for the thinking model. It receives both the original
/// conversation text and the extracted candidates, so it can verify claims
/// against the source.
const CURATION_SYSTEM_PROMPT: &str = r#"You are a quality filter for extracted knowledge about a user. These items will be stored in a long-term memory system. When a different AI assistant meets this user for the first time in a new project, these are the things it will know about them.

For each candidate, think carefully: would knowing this actually change how you interact with this person? Would it help you make better assumptions, avoid mistakes, or tailor your approach? If not, drop it — even if it's technically true and supported.

You will receive:
1. The original conversation text
2. A list of candidate facts extracted from it

Your job is to verify each candidate against the conversation AND judge whether it's worth remembering.

KEEP items that are:
- Explicitly evidenced by a specific USER statement or action in the conversation. If you cannot point to a concrete user statement that evidences the claim, it is hallucinated — drop it. Absence of contradiction is NOT evidence of support.
- Durable — would still be true next week
- Genuinely useful — would change how an assistant works with this person

DROP items that are:
- NOT supported by a specific user statement (hallucinated or inferred from general context)
- Pure project documentation: paths, filenames, config values, build steps, error messages, environment variables. These belong in the project, not in memory about the person.
- Actions or choices without reasoning. A bare choice tells you nothing about the person — drop it. A choice with reasoning reveals how they think — keep it at project scope. The reasoning is the valuable signal, not the choice itself.
- Trivial or generic — things true of nearly all developers or that anyone could guess. If you would say the same about a random developer, drop it.
- Too vague or partial to be useful. If the claim lacks specificity that makes it actionable or interesting, drop it.
- Ephemeral state (current status, temporary conditions, in-progress work)
- Session events — things that happened during the conversation but don't reflect a durable trait or preference. Status updates, debugging outcomes, what was removed or added, test results. These are project history, not knowledge about the person.
- Duplicates or near-duplicates of other items in the batch

FIX SCOPE: if an item is labeled "personal" but is really project-specific, change to "keep_as_project". Personal means true about the user regardless of which project they are working in. Project-specific UI preferences, architecture decisions, and tooling choices for a particular codebase are project scope. When in doubt, demote to keep_as_project. False-project is cheap; false-personal is expensive.

For each item, respond with one of:
- "keep" — item is verified and worth storing
- "keep_as_project" — item is valid but should be project scope
- "drop" — item should not be stored
- "merge:N" — merge with item N (0-indexed) as they say the same thing

Include a short reason for every decision to aid debugging.

Respond with ONLY a JSON array of objects with "index", "action", and "reason" fields."#;

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
    // enough to verify claims but doesn't need every detail.
    // 24000 chars ≈ 6000 tokens, leaving room for candidates + response
    // within Qwen3 8B's context window.
    let max_conv_chars = 24000;
    let conv_text = if conversation_text.len() > max_conv_chars {
        // Take from the end — candidates are more likely to reference recent text,
        // and the first 8000 chars of a long session are often setup/boilerplate.
        &conversation_text[conversation_text.len() - max_conv_chars..]
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

    // First pass: resolve merges — for each merge pair, keep the higher-confidence one.
    // Build a set of indices that should be skipped (the losing side of each merge).
    for action in &actions {
        if let CurationAction::Merge(target) = action.action {
            if action.index < candidates.len() && target < candidates.len() {
                let source = &candidates[action.index];
                let dest = &candidates[target];
                // Keep the higher-confidence one, drop the other
                let (keep_idx, drop_idx) = if source.confidence >= dest.confidence {
                    (action.index, target)
                } else {
                    (target, action.index)
                };
                merge_targets.insert(drop_idx);
                tracing::info!(
                    "Curation: merge — keeping \"{}\" (conf={:.2}), dropping \"{}\" (conf={:.2})",
                    truncate_str(&candidates[keep_idx].content, 50),
                    candidates[keep_idx].confidence,
                    truncate_str(&candidates[drop_idx].content, 50),
                    candidates[drop_idx].confidence,
                );
            }
        }
    }

    // Second pass: apply keep/drop/scope changes
    for action in &actions {
        if action.index >= candidates.len() {
            continue;
        }
        // Skip items that lost a merge
        if merge_targets.contains(&action.index) {
            continue;
        }

        match action.action {
            CurationAction::Keep | CurationAction::Merge(_) => {
                let c = &candidates[action.index];
                tracing::info!(
                    "Curation: keeping: {} (reason: {})",
                    truncate_str(&c.content, 80),
                    action.reason.as_deref().unwrap_or("none"),
                );
                kept.push(candidates[action.index].clone());
            }
            CurationAction::KeepAsProject => {
                let mut c = candidates[action.index].clone();
                tracing::info!(
                    "Curation: demoting to project: {} (reason: {})",
                    truncate_str(&c.content, 80),
                    action.reason.as_deref().unwrap_or("none"),
                );
                c.scope = "project".to_string();
                kept.push(c);
            }
            CurationAction::Drop => {
                let c = &candidates[action.index];
                tracing::info!(
                    "Curation: dropping: {} (reason: {})",
                    truncate_str(&c.content, 80),
                    action.reason.as_deref().unwrap_or("none"),
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
    reason: Option<String>,
}

/// Truncate a string to at most `max` bytes at a char boundary.
pub fn truncate_str(s: &str, max: usize) -> &str {
    if s.len() <= max {
        return s;
    }
    let mut end = max;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

/// Strip Qwen3-style `<think>...</think>` blocks from the response.
/// The thinking model generates chain-of-thought before the JSON output,
/// and this block often contains brackets that confuse the JSON parser.
fn strip_thinking_block(text: &str) -> &str {
    // Find the end of the thinking block
    if let Some(end) = text.find("</think>") {
        let after = &text[end + "</think>".len()..];
        after.trim()
    } else if text.starts_with("<think>") {
        // Thinking block started but never closed — model hit token limit
        // during thinking. Try to find JSON after any obvious boundary.
        text
    } else {
        text
    }
}

/// Parse the curation model's response. Falls back to keeping everything
/// if the response can't be parsed.
fn parse_curation_response(response: &str, num_candidates: usize) -> Vec<CurationDecision> {
    let trimmed = response.trim();

    // Strip thinking block first — Qwen3 emits <think>...</think> before JSON
    let without_thinking = strip_thinking_block(trimmed);

    let parsed = try_parse_curation(without_thinking)
        .or_else(|| {
            let stripped = super::extract::strip_code_fences(without_thinking);
            try_parse_curation(&stripped)
        })
        .or_else(|| {
            let start = without_thinking.find('[')?;
            let end = without_thinking.rfind(']')?;
            try_parse_curation(&without_thinking[start..=end])
        });

    match parsed {
        Some(decisions) => {
            // Validate that indices are in range
            let valid: Vec<_> = decisions.into_iter()
                .filter(|d| {
                    if d.index >= num_candidates {
                        tracing::warn!(
                            "Curation returned out-of-range index {} (max {}), skipping",
                            d.index, num_candidates - 1,
                        );
                        false
                    } else {
                        true
                    }
                })
                .collect();

            // If we got decisions but they don't cover all candidates,
            // default uncovered ones to keep
            if valid.len() < num_candidates {
                let covered: std::collections::HashSet<usize> = valid.iter().map(|d| d.index).collect();
                let mut all = valid;
                for i in 0..num_candidates {
                    if !covered.contains(&i) {
                        tracing::debug!("Curation missing decision for index {}, keeping by default", i);
                        all.push(CurationDecision {
                            index: i,
                            action: CurationAction::Keep,
                            reason: Some("no curation decision for this item, keeping by default".to_string()),
                        });
                    }
                }
                all
            } else {
                valid
            }
        }
        None => {
            tracing::warn!(
                "Curation response unparseable ({} chars), keeping all {} candidates. First 300 chars: {}",
                trimmed.len(),
                num_candidates,
                &trimmed[..trimmed.len().min(300)]
            );
            (0..num_candidates)
                .map(|i| CurationDecision {
                    index: i,
                    action: CurationAction::Keep,
                    reason: Some("curation response unparseable, keeping by default".to_string()),
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
        let reason = item.get("reason")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

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

        decisions.push(CurationDecision { index, action, reason });
    }

    Some(decisions)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_truncate_str_ascii() {
        assert_eq!(truncate_str("hello world", 5), "hello");
        assert_eq!(truncate_str("hello", 10), "hello");
        assert_eq!(truncate_str("", 5), "");
    }

    #[test]
    fn test_truncate_str_multibyte() {
        // Em-dash is 3 bytes (UTF-8: e2 80 94)
        let s = "hello\u{2014}world";
        // Cutting at 6 would land inside the em-dash — should back up
        let result = truncate_str(s, 6);
        assert_eq!(result, "hello");
        // Cutting at 8 lands after the em-dash
        let result = truncate_str(s, 8);
        assert_eq!(result, "hello\u{2014}");
    }

    #[test]
    fn test_strip_thinking_block() {
        let input = "<think>some reasoning here</think>\n[{\"index\": 0}]";
        assert_eq!(strip_thinking_block(input), "[{\"index\": 0}]");
    }

    #[test]
    fn test_strip_thinking_block_no_thinking() {
        let input = "[{\"index\": 0}]";
        assert_eq!(strip_thinking_block(input), "[{\"index\": 0}]");
    }

    #[test]
    fn test_strip_thinking_block_unclosed() {
        // Thinking block started but never closed — model hit token limit
        let input = "<think>reasoning that never ends...";
        assert_eq!(strip_thinking_block(input), input);
    }

    #[test]
    fn test_parse_curation_basic() {
        let input = r#"[{"index": 0, "action": "keep", "reason": "good"}, {"index": 1, "action": "drop", "reason": "bad"}]"#;
        let decisions = parse_curation_response(input, 2);
        assert_eq!(decisions.len(), 2);
        assert!(matches!(decisions[0].action, CurationAction::Keep));
        assert!(matches!(decisions[1].action, CurationAction::Drop));
    }

    #[test]
    fn test_parse_curation_with_thinking() {
        let input = "<think>Let me analyze each candidate...</think>\n[{\"index\": 0, \"action\": \"keep\", \"reason\": \"valid\"}]";
        let decisions = parse_curation_response(input, 1);
        assert_eq!(decisions.len(), 1);
        assert!(matches!(decisions[0].action, CurationAction::Keep));
    }

    #[test]
    fn test_parse_curation_merge() {
        let input = r#"[{"index": 0, "action": "keep"}, {"index": 1, "action": "merge:0"}]"#;
        let decisions = parse_curation_response(input, 2);
        assert_eq!(decisions.len(), 2);
        assert!(matches!(decisions[1].action, CurationAction::Merge(0)));
    }

    #[test]
    fn test_parse_curation_scope_fix() {
        let input = r#"[{"index": 0, "action": "keep_as_project", "reason": "project-specific"}]"#;
        let decisions = parse_curation_response(input, 1);
        assert_eq!(decisions.len(), 1);
        assert!(matches!(decisions[0].action, CurationAction::KeepAsProject));
    }

    #[test]
    fn test_parse_curation_unparseable_keeps_all() {
        let input = "I can't parse this into JSON";
        let decisions = parse_curation_response(input, 3);
        assert_eq!(decisions.len(), 3);
        // All should default to keep
        for d in &decisions {
            assert!(matches!(d.action, CurationAction::Keep));
        }
    }
}
