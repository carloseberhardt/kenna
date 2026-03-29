use std::collections::{HashMap, HashSet};

use anyhow::Result;
use chrono::Utc;
use serde::Deserialize;
use uuid::Uuid;

use crate::config::Config;
use crate::storage::db::EngramDb;
use crate::storage::models::{Category, Engram, Lifecycle, Scope};

use super::curate::{strip_thinking_block, truncate_str};
use super::extract::strip_code_fences;
use super::reconcile::cosine_similarity;

/// Sentinel value for source_session on settled engrams.
const SETTLING_SESSION: &str = "settling";

// ── Synthesis prompts ──

const PROMOTION_SYSTEM_PROMPT: &str = r#"You synthesize observations about a user from multiple projects into a single personal trait. You will receive a list of observations made across different projects.

First, discard any items that are:
- Trivial or generic (true of any developer)
- Misclassified (not actually about the stated topic)
- Project-specific implementation details rather than personal traits

Then write a single concise sentence that captures the underlying personal trait or preference from the remaining items. This should be true about the person regardless of which project they are working in.

Respond with ONLY a JSON object: {"content": "...", "category": "fact|preference|decision|pattern|interest|humor|opinion", "confidence": 0.0-1.0}"#;

const ENTITY_SYNTHESIS_SYSTEM_PROMPT: &str = r#"You consolidate multiple knowledge items about the same topic into a coherent summary.

First, discard any items that are:
- Trivial or generic (true of any developer)
- Misclassified (clearly about a different topic than the stated entity)
- Redundant (already covered by another item with higher confidence)
- Too vague to be useful on its own

Then write a single concise paragraph that captures the remaining important facts. Preserve the most confident and most recent claims. If items contradict, keep the most recent one.

If after filtering there is only one item left, return it as-is. If nothing survives filtering, respond with {"content": "", "confidence": 0.0}.

Respond with ONLY a JSON object: {"content": "...", "confidence": 0.0-1.0}"#;

// ── Report ──

#[derive(Debug, Default)]
pub struct SettleReport {
    pub promotions: Vec<PromotionCandidate>,
    pub syntheses: Vec<SynthesisCandidate>,
    pub skipped_already_settled: usize,
}

#[derive(Debug)]
pub struct PromotionCandidate {
    pub source_engrams: Vec<PromotionSource>,
    pub distinct_projects: usize,
    /// Set after synthesis (None in dry-run)
    pub synthesized_content: Option<String>,
    pub new_id: Option<Uuid>,
}

#[derive(Debug)]
pub struct PromotionSource {
    pub id: Uuid,
    pub content: String,
    pub source_project: String,
}

#[derive(Debug)]
pub struct SynthesisCandidate {
    pub entity: String,
    pub source_engrams: Vec<SynthesisSource>,
    /// Set after synthesis (None in dry-run)
    pub synthesized_content: Option<String>,
    pub new_id: Option<Uuid>,
}

#[derive(Debug)]
pub struct SynthesisSource {
    pub id: Uuid,
    pub content: String,
    pub confidence: f32,
}

// ── Synthesis response parsing ──

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct PromotionResponse {
    content: String,
    #[serde(default = "default_category")]
    category: String,
    #[serde(default = "default_confidence")]
    confidence: f32,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct SynthesisResponse {
    content: String,
    #[serde(default = "default_confidence")]
    confidence: f32,
}

fn default_category() -> String { "pattern".into() }
fn default_confidence() -> f32 { 0.8 }

// ── Union-Find for clustering ──

struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }

    fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            self.parent[x] = self.find(self.parent[x]);
        }
        self.parent[x]
    }

    fn union(&mut self, x: usize, y: usize) {
        let rx = self.find(x);
        let ry = self.find(y);
        if rx == ry { return; }
        if self.rank[rx] < self.rank[ry] {
            self.parent[rx] = ry;
        } else if self.rank[rx] > self.rank[ry] {
            self.parent[ry] = rx;
        } else {
            self.parent[ry] = rx;
            self.rank[rx] += 1;
        }
    }

    fn groups(&mut self) -> Vec<Vec<usize>> {
        let mut map: HashMap<usize, Vec<usize>> = HashMap::new();
        for i in 0..self.parent.len() {
            let root = self.find(i);
            map.entry(root).or_default().push(i);
        }
        map.into_values().collect()
    }
}

// ── Core logic ──

/// Run the settling pass: cross-project promotion + entity synthesis.
pub async fn run_settle(
    db: &EngramDb,
    config: &Config,
    dry_run: bool,
    generate_fn: Option<&dyn Fn(&str, &str) -> Result<String>>,
    embed_fn: Option<&dyn Fn(&str) -> Result<Vec<f32>>>,
) -> Result<SettleReport> {
    let mut report = SettleReport::default();

    // Load all non-superseded engrams
    let all_engrams = db
        .list(&crate::storage::db::ListFilters {
            exclude_superseded: true,
            limit: Some(100_000),
            ..Default::default()
        })
        .await?;

    tracing::info!("Settling: loaded {} active engrams", all_engrams.len());

    // ── Phase 1: Cross-project promotion ──
    let project_engrams: Vec<&Engram> = all_engrams.iter()
        .filter(|e| e.scope == Scope::Project)
        .collect();

    tracing::info!("Settling: {} project-scoped engrams for clustering", project_engrams.len());

    if project_engrams.len() >= 2 {
        let clusters = cluster_by_similarity(
            &project_engrams,
            config.settle.cluster_cosine_threshold,
            config.settle.max_cluster_size,
        );

        for cluster_indices in &clusters {
            // Count distinct source projects
            let projects: HashSet<&str> = cluster_indices.iter()
                .filter_map(|&i| project_engrams[i].source_project.as_deref())
                .collect();

            if projects.len() < config.settle.min_projects_for_promotion {
                continue;
            }

            // Idempotency: skip if any member is already superseded by a settling engram
            let any_already_settled = cluster_indices.iter().any(|&i| {
                project_engrams[i].superseded_by.is_some()
            });
            if any_already_settled {
                report.skipped_already_settled += 1;
                continue;
            }

            let sources: Vec<PromotionSource> = cluster_indices.iter().map(|&i| {
                let e = project_engrams[i];
                PromotionSource {
                    id: e.id,
                    content: e.content.clone(),
                    source_project: e.source_project.clone().unwrap_or_default(),
                }
            }).collect();

            report.promotions.push(PromotionCandidate {
                distinct_projects: projects.len(),
                source_engrams: sources,
                synthesized_content: None,
                new_id: None,
            });
        }
    }

    // ── Phase 2: Entity grouping ──
    // Reload to account for any engrams that might overlap with promotions
    let active_engrams: Vec<&Engram> = all_engrams.iter()
        .filter(|e| e.superseded_by.is_none())
        .collect();

    let mut entity_groups: HashMap<&str, Vec<&Engram>> = HashMap::new();
    for e in &active_engrams {
        if let Some(ref entity) = e.entity {
            entity_groups.entry(entity.as_str()).or_default().push(e);
        }
    }

    let promotion_ids: HashSet<Uuid> = report.promotions.iter()
        .flat_map(|p| p.source_engrams.iter().map(|s| s.id))
        .collect();

    for (entity, group) in &entity_groups {
        if group.len() < config.settle.min_engrams_for_synthesis {
            continue;
        }

        // Skip entities that are too broad — synthesis of a vague bucket
        // produces a vague paragraph that loses specificity.
        // These need reclassification first (future settling step).
        if group.len() > config.settle.max_cluster_size {
            tracing::info!(
                "Settling: skipping entity '{}' ({} engrams, exceeds max {})",
                entity, group.len(), config.settle.max_cluster_size,
            );
            continue;
        }

        // Skip single-session entities — if all engrams come from one session,
        // it's session-specific detail, not a pattern worth synthesizing.
        let distinct_sessions: HashSet<&str> = group.iter()
            .map(|e| e.source_session.as_str())
            .collect();
        if distinct_sessions.len() <= 1 {
            continue;
        }

        // Idempotency: skip if any member is already settled
        let any_already_settled = group.iter().any(|e| {
            e.source_session == SETTLING_SESSION
        });
        if any_already_settled {
            report.skipped_already_settled += 1;
            continue;
        }

        // Don't synthesize entities that are entirely part of a promotion cluster
        // (they'll be handled by promotion)
        let non_promoted: Vec<&&Engram> = group.iter()
            .filter(|e| !promotion_ids.contains(&e.id))
            .collect();
        if non_promoted.len() < config.settle.min_engrams_for_synthesis {
            continue;
        }

        let sources: Vec<SynthesisSource> = non_promoted.iter().map(|e| {
            SynthesisSource {
                id: e.id,
                content: e.content.clone(),
                confidence: e.confidence,
            }
        }).collect();

        report.syntheses.push(SynthesisCandidate {
            entity: entity.to_string(),
            source_engrams: sources,
            synthesized_content: None,
            new_id: None,
        });
    }

    if dry_run {
        return Ok(report);
    }

    let generate = generate_fn.expect("generate_fn required for non-dry-run");
    let embed = embed_fn.expect("embed_fn required for non-dry-run");

    // ── Phase 3: Synthesis ──

    // Cross-project promotions
    for promotion in &mut report.promotions {
        let items: Vec<String> = promotion.source_engrams.iter()
            .map(|s| format!("- [project: {}] {}", s.source_project, s.content))
            .collect();
        let user_prompt = format!(
            "The following observations were made across {} different projects:\n{}",
            promotion.distinct_projects,
            items.join("\n"),
        );

        match generate(PROMOTION_SYSTEM_PROMPT, &user_prompt) {
            Ok(response) => {
                if let Some(parsed) = parse_synthesis_response::<PromotionResponse>(&response) {
                    if parsed.content.is_empty() {
                        tracing::info!("Promotion: all items filtered during synthesis");
                    } else {
                        promotion.synthesized_content = Some(parsed.content);
                    }
                } else {
                    tracing::warn!("Failed to parse promotion synthesis response");
                }
            }
            Err(e) => {
                tracing::warn!("Promotion synthesis failed: {e}");
            }
        }
    }

    // Entity syntheses
    for synthesis in &mut report.syntheses {
        let items: Vec<String> = synthesis.source_engrams.iter()
            .map(|s| format!("- (conf={:.2}) {}", s.confidence, s.content))
            .collect();
        let user_prompt = format!(
            "Topic: \"{}\"\nItems to consolidate:\n{}",
            synthesis.entity,
            items.join("\n"),
        );

        match generate(ENTITY_SYNTHESIS_SYSTEM_PROMPT, &user_prompt) {
            Ok(response) => {
                if let Some(parsed) = parse_synthesis_response::<SynthesisResponse>(&response) {
                    if parsed.content.is_empty() {
                        tracing::info!("Entity '{}': all items filtered during synthesis", synthesis.entity);
                    } else {
                        synthesis.synthesized_content = Some(parsed.content);
                    }
                } else {
                    tracing::warn!("Failed to parse entity synthesis response for '{}'", synthesis.entity);
                }
            }
            Err(e) => {
                tracing::warn!("Entity synthesis failed for '{}': {e}", synthesis.entity);
            }
        }
    }

    // ── Phase 4 & 5: Embed + Persist ──

    let now = Utc::now();

    // Persist promotions
    for promotion in &mut report.promotions {
        let content = match &promotion.synthesized_content {
            Some(c) => c,
            None => continue, // synthesis failed, skip
        };

        let embedding = embed(content)?;
        let new_id = Uuid::new_v4();

        let engram = Engram {
            id: new_id,
            content: content.clone(),
            embedding,
            scope: Scope::Personal,
            category: most_common_category_from_projects(
                &promotion.source_engrams.iter().map(|s| s.id).collect::<Vec<_>>(),
                &all_engrams,
            ),
            entity: None, // personal traits don't need entity grouping
            source_project: None,
            source_session: SETTLING_SESSION.to_string(),
            source_timestamp: now,
            lifecycle: Lifecycle::Candidate,
            confidence: 0.8, // default for syntheses
            created_at: now,
            updated_at: now,
            accessed_at: None,
            supersedes: Some(promotion.source_engrams[0].id),
            superseded_by: None,
        };

        db.insert(vec![engram]).await?;

        for source in &promotion.source_engrams {
            let _ = db.mark_superseded(&source.id, &new_id).await;
        }

        promotion.new_id = Some(new_id);
        tracing::info!(
            "Promoted to personal ({}): {} (from {} projects)",
            &new_id.to_string()[..8],
            truncate_str(content, 60),
            promotion.distinct_projects,
        );
    }

    // Persist entity syntheses
    for synthesis in &mut report.syntheses {
        let content = match &synthesis.synthesized_content {
            Some(c) => c,
            None => continue,
        };

        let embedding = embed(content)?;
        let new_id = Uuid::new_v4();

        // Inherit scope and category from the majority of source engrams
        let source_ids: Vec<Uuid> = synthesis.source_engrams.iter().map(|s| s.id).collect();
        let (scope, category) = most_common_scope_and_category(&source_ids, &all_engrams);

        let engram = Engram {
            id: new_id,
            content: content.clone(),
            embedding,
            scope,
            category,
            entity: Some(synthesis.entity.clone()),
            source_project: None,
            source_session: SETTLING_SESSION.to_string(),
            source_timestamp: now,
            lifecycle: Lifecycle::Candidate,
            confidence: synthesis.source_engrams.iter()
                .map(|s| s.confidence)
                .fold(0.0f32, f32::max), // inherit highest confidence
            created_at: now,
            updated_at: now,
            accessed_at: None,
            supersedes: Some(synthesis.source_engrams[0].id),
            superseded_by: None,
        };

        db.insert(vec![engram]).await?;

        for source in &synthesis.source_engrams {
            let _ = db.mark_superseded(&source.id, &new_id).await;
        }

        synthesis.new_id = Some(new_id);
        tracing::info!(
            "Synthesized entity '{}' ({}): {} (from {} engrams)",
            synthesis.entity,
            &new_id.to_string()[..8],
            truncate_str(content, 60),
            synthesis.source_engrams.len(),
        );
    }

    Ok(report)
}

// ── Helpers ──

fn cluster_by_similarity(
    engrams: &[&Engram],
    threshold: f32,
    max_size: usize,
) -> Vec<Vec<usize>> {
    let n = engrams.len();
    let mut uf = UnionFind::new(n);

    for i in 0..n {
        for j in (i + 1)..n {
            let sim = cosine_similarity(&engrams[i].embedding, &engrams[j].embedding);
            if sim >= threshold {
                uf.union(i, j);
            }
        }
    }

    uf.groups().into_iter()
        .filter(|g| g.len() >= 2 && g.len() <= max_size)
        .collect()
}

fn parse_synthesis_response<T: serde::de::DeserializeOwned>(response: &str) -> Option<T> {
    let cleaned = strip_thinking_block(response.trim());
    let cleaned = strip_code_fences(cleaned);
    let trimmed = cleaned.trim();

    // Try direct parse
    if let Ok(parsed) = serde_json::from_str::<T>(trimmed) {
        return Some(parsed);
    }

    // Try to find JSON object bounds
    if let Some(start) = trimmed.find('{') {
        if let Some(end) = trimmed.rfind('}') {
            if let Ok(parsed) = serde_json::from_str::<T>(&trimmed[start..=end]) {
                return Some(parsed);
            }
        }
    }

    // Log the failure for debugging
    let preview = if trimmed.len() <= 500 { trimmed } else { &trimmed[..500] };
    tracing::debug!("Synthesis response unparseable: {preview}");
    None
}

fn most_common_category_from_projects(ids: &[Uuid], all: &[Engram]) -> Category {
    let mut counts: HashMap<&Category, usize> = HashMap::new();
    for e in all {
        if ids.contains(&e.id) {
            *counts.entry(&e.category).or_default() += 1;
        }
    }
    counts.into_iter()
        .max_by_key(|&(_, count)| count)
        .map(|(cat, _)| cat.clone())
        .unwrap_or(Category::Pattern)
}

fn most_common_scope_and_category(ids: &[Uuid], all: &[Engram]) -> (Scope, Category) {
    let mut scope_counts: HashMap<Scope, usize> = HashMap::new();
    let mut cat_counts: HashMap<&Category, usize> = HashMap::new();
    for e in all {
        if ids.contains(&e.id) {
            *scope_counts.entry(e.scope.clone()).or_default() += 1;
            *cat_counts.entry(&e.category).or_default() += 1;
        }
    }
    let scope = scope_counts.into_iter()
        .max_by_key(|&(_, count)| count)
        .map(|(s, _)| s)
        .unwrap_or(Scope::Project);
    let category = cat_counts.into_iter()
        .max_by_key(|&(_, count)| count)
        .map(|(c, _)| c.clone())
        .unwrap_or(Category::Pattern);
    (scope, category)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_union_find_basic() {
        let mut uf = UnionFind::new(5);
        uf.union(0, 1);
        uf.union(2, 3);
        uf.union(1, 3);
        let groups = uf.groups();
        // {0,1,2,3} and {4}
        assert_eq!(groups.len(), 2);
        let big_group = groups.iter().find(|g| g.len() == 4).unwrap();
        assert!(big_group.contains(&0));
        assert!(big_group.contains(&3));
    }

    #[test]
    fn test_union_find_all_separate() {
        let mut uf = UnionFind::new(3);
        let groups = uf.groups();
        assert_eq!(groups.len(), 3);
    }

    #[test]
    fn test_parse_promotion_response() {
        let input = r#"{"content": "Consistently prefers simpler solutions", "category": "preference", "confidence": 0.9}"#;
        let parsed: PromotionResponse = parse_synthesis_response(input).unwrap();
        assert_eq!(parsed.content, "Consistently prefers simpler solutions");
        assert_eq!(parsed.category, "preference");
    }

    #[test]
    fn test_parse_synthesis_response_with_thinking() {
        let input = r#"<think>Let me consider...</think>
{"content": "Uses Arch Linux with AMD GPU", "confidence": 0.95}"#;
        let parsed: SynthesisResponse = parse_synthesis_response(input).unwrap();
        assert_eq!(parsed.content, "Uses Arch Linux with AMD GPU");
    }
}
