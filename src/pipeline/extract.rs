use anyhow::{Result, bail};
use serde::Deserialize;

use crate::inference::InferenceBackend;
use super::preprocess::ConversationChunk;

fn default_confidence() -> f32 {
    0.5
}

fn default_scope_confidence() -> f32 {
    0.5
}

/// A raw extracted memory candidate from the LLM.
#[derive(Debug, Clone, Deserialize)]
pub struct ExtractedCandidate {
    pub content: String,
    pub scope: String,
    pub category: String,
    pub entity: Option<String>,
    /// How confident the model is that the content is durable and accurate.
    #[serde(default = "default_confidence")]
    pub confidence: f32,
    /// How confident the model is in the scope classification.
    /// Low scope_confidence on a "personal" item → safer to demote to "project".
    #[serde(default = "default_scope_confidence")]
    pub scope_confidence: f32,
}

const EXTRACTION_SYSTEM_PROMPT: &str = r#"You know nothing about this user other than what appears in the conversation below. You extract durable knowledge about them from their conversations with an AI assistant.

Your goal: if a different AI assistant met this user for the first time in a new project, what would be genuinely useful to know? Extract ONLY things that would change how you'd interact with them or what you'd assume about them.

RULES:
1. Only extract knowledge originating from or confirmed by the USER ("Human" turns). The assistant provides context but is not a source of facts about the user.
2. Never infer facts that are not explicitly stated or directly demonstrated by the user's own words or actions. If something is not evidenced in the conversation, it does not exist.
3. Only extract things that will still be true next week. Not: task progress, debugging steps, what's being worked on right now, procedural exchanges, or generic advice the assistant provided.
4. For games, simulations, or fiction: extract what the user's engagement reveals about THEM, not the game/fiction state itself.
5. Write each item as something you know about this person — not a summary of dialogue, not a quote from the conversation, and not anything from this system prompt.

SELECTIVITY — this is critical:
- Most chunks contain ZERO extractable items. Returning [] is the correct answer most of the time.
- Ask yourself: "Would this help me work with this person better?" If not, skip it.
- DO NOT extract: paths, filenames, directory structures, config values, CLI flags, dependency versions, build steps, error messages, environment variables, or anything derivable from reading the project. That is project documentation, not knowledge about the user.
- DO NOT extract what was done — extract WHY and HOW they chose to do it. The action itself is project history. The reasoning behind it reveals how the user thinks — that is extractable as project scope. Even if a preference only appears in one project right now, it may recur across projects later and become a known personal pattern.
- DO NOT extract things true of any developer: uses a terminal, has a home directory, writes code, uses git, runs tests, has a keyboard, uses a laptop.
- DO extract: who they are (role, employer, location, hardware, OS), how they think (preferences, opinions, reasoning style, design philosophy, humor), what they care about (interests, values, strong reactions), and how they want to be worked with (communication style, review preferences, workflow).
- DO extract as project scope: decision rationale, design preferences, and thinking patterns observed in this project. These are valuable signals — when the same pattern appears across multiple projects, it reveals a personal trait.

For each item, classify:
- scope: Default to "project". Use "personal" ONLY for things true about the user regardless of project: job, location, hobbies, hardware they own, cross-project language preferences, personality, people in their life. Everything else is "project". When in doubt, "project".
- category: "fact", "preference", "decision", "pattern", "interest", "humor", or "opinion"
- entity: short kebab-case key for the primary topic. Use consistent, general keys (not product names or model numbers).
- confidence: 0.0-1.0 how certain the content is durable and accurate
- scope_confidence: 0.0-1.0 how certain you are about the scope. Low (< 0.7) if it could be either.
- content: concise, self-contained natural language statement. Must be specific enough to be useful on its own — vague or partial claims are not worth storing.

Respond with ONLY a JSON array. If nothing worth extracting, respond with []. Do not invent facts.

JSON OUTPUT RULE: Inside the "content" string, use single quotes (') for any embedded quotation, scare-quote, or term of art. Reserve double quotes (") for JSON field delimiters only. Example: write 'impressive demo', not "impressive demo"."#;

use crate::inference::ChatMessage;

/// Few-shot examples as prior conversation turns.
/// These teach the model grounding behavior without contaminating the system prompt.
/// Content is deliberately unrealistic so it can't be confused with real extractions.
const FEWSHOT_USER_1: &str = r#"Project: acme-widgets
Session: example-1

<conversation>
Human: nah, let's go with the simpler approach. I always regret overengineering things
Assistant: Makes sense. I'll go with the straightforward implementation.
Human: yeah. also can you fix those linting errors while you're at it
Assistant: Done, fixed 3 linting issues.
</conversation>"#;

const FEWSHOT_ASSISTANT_1: &str = r#"[{"content": "Prefers simpler approaches, has been burned by overengineering in the past", "scope": "project", "category": "preference", "entity": "design-philosophy", "confidence": 0.85, "scope_confidence": 0.5}]"#;

const FEWSHOT_USER_2: &str = r#"Project: acme-widgets
Session: example-2

<conversation>
Human: ok, run the tests again
Assistant: All 42 tests passing.
Human: great, ship it
Assistant: Pushed to main.
</conversation>"#;

const FEWSHOT_ASSISTANT_2: &str = "[]";

const FEWSHOT_USER_3: &str = r#"Project: acme-widgets
Session: example-3

<conversation>
Human: I want the demo to feel polished, not like some quick hack
Assistant: Makes sense, I'll focus on the end-to-end flow.
</conversation>"#;

// Demonstrates the single-quote rule for scare-quotes inside content.
const FEWSHOT_ASSISTANT_3: &str = r#"[{"content": "Values a polished end-to-end demo over a 'quick hack' prototype", "scope": "project", "category": "preference", "entity": "demo-quality", "confidence": 0.8, "scope_confidence": 0.6}]"#;

/// Extract memory candidates from a conversation chunk using the LLM.
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

    // Multi-turn: system + three few-shot examples + real conversation
    let messages = vec![
        ChatMessage { role: "system".into(), content: system_prompt },
        // Few-shot 1: preference extraction with rationale
        ChatMessage { role: "user".into(), content: FEWSHOT_USER_1.into() },
        ChatMessage { role: "assistant".into(), content: FEWSHOT_ASSISTANT_1.into() },
        // Few-shot 2: empty array for procedural talk
        ChatMessage { role: "user".into(), content: FEWSHOT_USER_2.into() },
        ChatMessage { role: "assistant".into(), content: FEWSHOT_ASSISTANT_2.into() },
        // Few-shot 3: demonstrates the single-quote rule for embedded quotes in content
        ChatMessage { role: "user".into(), content: FEWSHOT_USER_3.into() },
        ChatMessage { role: "assistant".into(), content: FEWSHOT_ASSISTANT_3.into() },
        // Real conversation
        ChatMessage { role: "user".into(), content: user_prompt },
    ];

    let response = backend.generate_chat_multi(&messages, max_tokens)?;

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

    // Bare object without array brackets: Gemma sometimes outputs {"content":...}
    // instead of [{"content":...}]. Wrap it in brackets and try.
    if stripped.starts_with('{') {
        let wrapped = format!("[{stripped}]");
        if let Ok(candidates) = serde_json::from_str::<Vec<ExtractedCandidate>>(&wrapped) {
            tracing::debug!("Parsed bare JSON object (wrapped in array)");
            return Ok(candidates);
        }
    }

    // Multiple arrays: Gemma sometimes outputs two or more JSON arrays back-to-back.
    // Use serde to find each valid array/object rather than bracket counting,
    // which breaks on brackets inside JSON string values.
    {
        let mut all_candidates = Vec::new();
        let mut remaining = stripped.trim();
        while !remaining.is_empty() {
            // Try parsing an array at current position
            if remaining.starts_with('[') {
                match serde_json::from_str::<Vec<ExtractedCandidate>>(remaining) {
                    Ok(candidates) => {
                        all_candidates.extend(candidates);
                        break; // consumed everything
                    }
                    Err(_) => {
                        // Try to find where this array ends by looking for ]\s*[
                        // or ]\s*{ boundaries
                        if let Some(boundary) = find_array_boundary(remaining) {
                            let slice = &remaining[..boundary];
                            if let Ok(candidates) = serde_json::from_str::<Vec<ExtractedCandidate>>(slice) {
                                all_candidates.extend(candidates);
                            }
                            remaining = remaining[boundary..].trim();
                        } else {
                            break;
                        }
                    }
                }
            } else if remaining.starts_with('{') {
                // Bare object
                let wrapped = format!("[{remaining}]");
                if let Ok(candidates) = serde_json::from_str::<Vec<ExtractedCandidate>>(&wrapped) {
                    all_candidates.extend(candidates);
                    break;
                } else {
                    break;
                }
            } else {
                // Skip non-JSON prefix (preamble text)
                if let Some(pos) = remaining.find('[').or_else(|| remaining.find('{')) {
                    remaining = &remaining[pos..];
                } else {
                    break;
                }
            }
        }
        if !all_candidates.is_empty() {
            tracing::debug!(
                "Parsed {} candidates from multiple JSON fragments",
                all_candidates.len()
            );
            return Ok(all_candidates);
        }
    }

    // Last resort: salvage individual objects from malformed JSON.
    // Handles missing braces, corrupted separators, etc. by splitting on
    // } boundaries and parsing each piece individually.
    {
        let mut salvaged = Vec::new();
        let inner = stripped.trim().trim_start_matches('[').trim_end_matches(']');
        for piece in split_json_objects(inner) {
            let piece = piece.trim();
            if piece.is_empty() {
                continue;
            }
            let obj_str = if piece.starts_with('{') {
                piece.to_string()
            } else {
                format!("{{{piece}")
            };
            if let Ok(candidate) = serde_json::from_str::<ExtractedCandidate>(&obj_str) {
                salvaged.push(candidate);
            }
        }
        if !salvaged.is_empty() {
            tracing::debug!(
                "Salvaged {} candidates from malformed JSON by object-level parsing",
                salvaged.len()
            );
            return Ok(salvaged);
        }
    }

    // Try to repair truncated JSON: find the last complete object and close the array
    if let Some(repaired) = repair_truncated_json(&stripped) {
        if let Ok(candidates) = serde_json::from_str::<Vec<ExtractedCandidate>>(&repaired) {
            tracing::debug!("Repaired truncated JSON ({} candidates salvaged)", candidates.len());
            return Ok(candidates);
        }
    }

    // If we still can't parse, dump to file for diagnosis and log summary
    let len = response.len();
    let dump_dir = crate::config::Config::data_dir().join("debug");
    let _ = std::fs::create_dir_all(&dump_dir);
    let dump_path = dump_dir.join(format!(
        "parse_fail_{}.json",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis()
    ));
    let _ = std::fs::write(&dump_path, response);

    let head = &response[..len.min(200)];
    tracing::warn!(
        "Failed to parse extraction response ({len} chars). Dumped to {}. HEAD: {head}",
        dump_path.display(),
    );
    bail!("unparseable extraction response")
}

/// Split a string containing JSON objects on `}` boundaries, respecting string context.
/// Returns pieces that should each start with (or be prefixable with) `{`.
fn split_json_objects(text: &str) -> Vec<&str> {
    let bytes = text.as_bytes();
    let mut in_string = false;
    let mut escape_next = false;
    let mut depth = 0;
    let mut pieces = Vec::new();
    let mut start = 0;

    for i in 0..bytes.len() {
        if escape_next {
            escape_next = false;
            continue;
        }
        match bytes[i] {
            b'\\' if in_string => escape_next = true,
            b'"' => in_string = !in_string,
            b'{' if !in_string => depth += 1,
            b'}' if !in_string => {
                depth -= 1;
                if depth == 0 {
                    pieces.push(&text[start..=i]);
                    start = i + 1;
                }
            }
            _ => {}
        }
    }
    // Capture any trailing fragment (incomplete object)
    let trailing = text[start..].trim().trim_start_matches(',').trim();
    if !trailing.is_empty() {
        pieces.push(trailing);
    }
    pieces
}

/// Find the boundary between two back-to-back JSON arrays in a string.
/// Looks for `]\s*[` or `]\s*{` patterns, skipping those inside strings.
/// Returns the byte offset of the start of the second array/object.
fn find_array_boundary(text: &str) -> Option<usize> {
    let bytes = text.as_bytes();
    let mut in_string = false;
    let mut escape_next = false;

    for i in 0..bytes.len() {
        if escape_next {
            escape_next = false;
            continue;
        }
        match bytes[i] {
            b'\\' if in_string => escape_next = true,
            b'"' => in_string = !in_string,
            b']' if !in_string => {
                // Look ahead past whitespace for '[' or '{'
                let mut j = i + 1;
                while j < bytes.len() && bytes[j].is_ascii_whitespace() {
                    j += 1;
                }
                if j < bytes.len() && (bytes[j] == b'[' || bytes[j] == b'{') {
                    // Found boundary: return position including the ']'
                    return Some(i + 1);
                }
            }
            _ => {}
        }
    }
    None
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
        let input = "Here are the extracted memories:\n[{\"content\": \"Test\", \"scope\": \"personal\", \"category\": \"fact\", \"entity\": \"test\", \"confidence\": 0.8}]";
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

    #[test]
    fn test_parse_multi_array() {
        // Gemma sometimes outputs two arrays back-to-back
        let input = r#"[{"content": "First", "scope": "personal", "category": "fact", "entity": "a", "confidence": 0.9}]
[{"content": "Second", "scope": "project", "category": "preference", "entity": "b", "confidence": 0.8}]"#;
        let result = parse_extraction_response(input).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].content, "First");
        assert_eq!(result[1].content, "Second");
    }

    #[test]
    fn test_parse_bare_object() {
        // Gemma sometimes outputs a single object without array brackets
        let input = r#"{"content": "Test", "scope": "personal", "category": "fact", "entity": "x", "confidence": 0.9}"#;
        let result = parse_extraction_response(input).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].content, "Test");
    }

    #[test]
    fn test_parse_missing_brace() {
        // Gemma drops opening brace on second object
        let input = r#"[{"content": "First", "scope": "personal", "category": "fact", "entity": "a", "confidence": 0.9}, "content": "Second", "scope": "project", "category": "fact", "entity": "b", "confidence": 0.8}]"#;
        let result = parse_extraction_response(input).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].content, "First");
        assert_eq!(result[1].content, "Second");
    }

    #[test]
    fn test_parse_truncated() {
        // Response cut off mid-object — should salvage the complete first object
        let input = r#"[{"content": "Complete", "scope": "personal", "category": "fact", "entity": "a", "confidence": 0.9}, {"content": "Trunc"#;
        let result = parse_extraction_response(input).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].content, "Complete");
    }

    #[test]
    fn test_split_json_objects_basic() {
        let input = r#"{"a": 1}, {"b": 2}"#;
        let pieces = split_json_objects(input);
        assert_eq!(pieces.len(), 2);
    }

    #[test]
    fn test_split_json_objects_brackets_in_strings() {
        // Brackets inside string values should not split
        let input = r#"{"content": "chose [A] over [B]", "val": 1}, {"content": "other", "val": 2}"#;
        let pieces = split_json_objects(input);
        assert_eq!(pieces.len(), 2);
        assert!(pieces[0].contains("[A] over [B]"));
    }

    #[test]
    fn test_find_array_boundary() {
        let input = r#"[{"a": 1}] [{"b": 2}]"#;
        let boundary = find_array_boundary(input).unwrap();
        assert_eq!(&input[..boundary], r#"[{"a": 1}]"#);
    }

    #[test]
    fn test_find_array_boundary_no_second() {
        let input = r#"[{"a": 1}]"#;
        assert!(find_array_boundary(input).is_none());
    }
}
