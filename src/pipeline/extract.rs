use anyhow::{Result, bail};
use serde::Deserialize;

use crate::inference::InferenceBackend;
use super::preprocess::ConversationChunk;

fn default_scope_confidence() -> f32 {
    0.5
}

/// A raw extracted engram candidate from the LLM.
#[derive(Debug, Clone, Deserialize)]
pub struct ExtractedCandidate {
    pub content: String,
    pub scope: String,
    pub category: String,
    pub entity: Option<String>,
    /// How confident the model is that the content is durable and accurate.
    pub confidence: f32,
    /// How confident the model is in the scope classification.
    /// Low scope_confidence on a "personal" item → safer to demote to "project".
    #[serde(default = "default_scope_confidence")]
    pub scope_confidence: f32,
}

const EXTRACTION_SYSTEM_PROMPT: &str = r#"You know nothing about this user other than what appears in the conversation below. You extract durable knowledge about them from their conversations with an AI assistant.

RULES:
1. Only extract knowledge originating from or confirmed by the USER ("Human" turns). The assistant provides context but is not a source of facts about the user.
2. Never infer facts that are not explicitly stated or directly demonstrated by the user's own words or actions. If something is not evidenced in the conversation, it does not exist.
3. Only extract things that will still be true next week. Not: task progress, game state, debugging steps, what's being worked on right now, procedural exchanges, or generic advice the assistant provided.
4. For games, simulations, or fiction: extract what the user's engagement reveals about THEM (e.g., "enjoys RimWorld"), not the game state itself.
5. Write each item as something you know about this person — not a summary of dialogue, not a quote from the conversation, and not anything from this system prompt.
6. Be highly selective. Most chunks contain 0-3 extractable items. Only extract what is distinctive or meaningful about this specific person. "Uses the terminal" is not distinctive.

Extract things like: facts (role, employer, location, hardware, OS, people), preferences and opinions, decisions with rationale, work patterns, interests, humor, technical environment.

For each item, classify:
- scope: Default to "project". Use "personal" ONLY for things true about the user regardless of project: job, location, hobbies, hardware they own, cross-project language preferences, personality, people in their life. Everything else is "project". When in doubt, "project".
- category: "fact", "preference", "decision", "pattern", "interest", "humor", or "opinion"
- entity: short kebab-case key for the primary topic. Use consistent, general keys — prefer "home-workstation" over "ryzen-7600-build", prefer "keyboard" over "keychron-k4-he-bluetooth".
- confidence: 0.0-1.0 how certain the content is durable and accurate
- scope_confidence: 0.0-1.0 how certain you are about the scope. Low (< 0.7) if it could be either.
- content: concise, self-contained natural language statement

Respond with ONLY a JSON array. If nothing worth extracting, respond with []. Do not invent facts."#;

/// Extract engram candidates from a conversation chunk using the LLM.
pub fn extract_from_chunk(
    backend: &dyn InferenceBackend,
    chunk: &ConversationChunk,
    max_tokens: u32,
) -> Result<Vec<ExtractedCandidate>> {
    let system_prompt = format!(
        "{EXTRACTION_SYSTEM_PROMPT}\n\nIMPORTANT: Respond ONLY with the JSON array. No thinking, no explanation, no preamble."
    );

    let user_prompt = format!(
        "Project: {project}\nSession: {session}\n\n<conversation>\n{text}\n</conversation>",
        project = chunk.project_dir_name,
        session = chunk.session_id,
        text = chunk.text,
    );

    tracing::debug!(
        "Extracting from chunk {} of session {} (~{} tokens)",
        chunk.chunk_index,
        chunk.session_id,
        chunk.approx_tokens,
    );

    let response = backend.generate_chat(&system_prompt, &user_prompt, max_tokens)?;

    parse_extraction_response(&response)
}

/// Parse the LLM's JSON response, with fallback for common formatting issues.
fn parse_extraction_response(response: &str) -> Result<Vec<ExtractedCandidate>> {
    let trimmed = response.trim();

    // Empty or explicit empty array
    if trimmed.is_empty() || trimmed == "[]" {
        return Ok(vec![]);
    }

    // Try direct parse first
    if let Ok(candidates) = serde_json::from_str::<Vec<ExtractedCandidate>>(trimmed) {
        return Ok(candidates);
    }

    // Strip markdown code fences: ```json ... ```
    let stripped = strip_code_fences(trimmed);
    if let Ok(candidates) = serde_json::from_str::<Vec<ExtractedCandidate>>(&stripped) {
        return Ok(candidates);
    }

    // Try to find JSON array bounds: first '[' to last ']'
    if let Some(start) = stripped.find('[') {
        if let Some(end) = stripped.rfind(']') {
            let json_slice = &stripped[start..=end];
            if let Ok(candidates) = serde_json::from_str::<Vec<ExtractedCandidate>>(json_slice) {
                return Ok(candidates);
            }
        }
    }

    // Try to repair truncated JSON: find the last complete object and close the array
    if let Some(repaired) = repair_truncated_json(&stripped) {
        if let Ok(candidates) = serde_json::from_str::<Vec<ExtractedCandidate>>(&repaired) {
            tracing::debug!("Repaired truncated JSON ({} candidates salvaged)", candidates.len());
            return Ok(candidates);
        }
    }

    // If we still can't parse, log and skip
    tracing::warn!(
        "Failed to parse extraction response ({} chars): {}",
        response.len(),
        &response[..response.len().min(200)]
    );
    bail!("unparseable extraction response")
}

/// Attempt to repair a truncated JSON array by finding the last complete object.
/// e.g. `[{"a":1},{"b":2},{"c":3` → `[{"a":1},{"b":2}]`
fn repair_truncated_json(text: &str) -> Option<String> {
    let start = text.find('[')?;
    let inner = &text[start..];

    // If it already has a closing bracket, this isn't a truncation issue
    if inner.rfind(']').is_some() {
        return None;
    }

    // Find the last complete object by looking for the last `}`
    // followed by either `,` or whitespace (not inside a string)
    let last_close = inner.rfind('}')?;
    let candidate = format!("{}]", &inner[..=last_close]);

    // Verify it parses
    if serde_json::from_str::<serde_json::Value>(&candidate).is_ok() {
        Some(candidate)
    } else {
        None
    }
}

/// Strip markdown code fences from LLM output.
pub fn strip_code_fences(text: &str) -> String {
    let mut result = text.to_string();

    // Remove opening fence: ```json or ```
    if let Some(start) = result.find("```") {
        let end_of_fence = result[start + 3..]
            .find('\n')
            .map(|i| start + 3 + i + 1)
            .unwrap_or(start + 3);
        result = result[end_of_fence..].to_string();
    }

    // Remove closing fence
    if let Some(pos) = result.rfind("```") {
        result = result[..pos].to_string();
    }

    result.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_clean_json() {
        let input = r#"[
            {"content": "Uses Rust", "scope": "personal", "category": "preference", "entity": "lang", "confidence": 0.9}
        ]"#;
        let result = parse_extraction_response(input).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].content, "Uses Rust");
    }

    #[test]
    fn test_parse_with_fences() {
        let input = "```json\n[{\"content\": \"Test\", \"scope\": \"personal\", \"category\": \"fact\", \"entity\": \"test\", \"confidence\": 0.8}]\n```";
        let result = parse_extraction_response(input).unwrap();
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_parse_with_preamble() {
        let input = "Here are the extracted engrams:\n[{\"content\": \"Test\", \"scope\": \"personal\", \"category\": \"fact\", \"entity\": \"test\", \"confidence\": 0.8}]";
        let result = parse_extraction_response(input).unwrap();
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_parse_empty() {
        let result = parse_extraction_response("[]").unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_parse_garbage() {
        let result = parse_extraction_response("I don't understand the question");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_null_entity() {
        let input = r#"[{"content": "Test", "scope": "personal", "category": "fact", "entity": null, "confidence": 0.8}]"#;
        let result = parse_extraction_response(input).unwrap();
        assert_eq!(result.len(), 1);
        assert!(result[0].entity.is_none());
    }
}
