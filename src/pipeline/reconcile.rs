use anyhow::Result;
use chrono::Utc;
use uuid::Uuid;

use crate::config::Config;
use crate::inference::InferenceBackend;
use crate::storage::db::MemoryDb;
use crate::storage::models::{Category, Memory, Lifecycle, Scope};

use super::extract::ExtractedCandidate;

/// Reconciliation outcome for a single candidate.
#[derive(Debug)]
pub enum ReconcileOutcome {
    /// Stored as new memory.
    Accepted(Uuid),
    /// Stored as candidate pending review.
    Candidate(Uuid),
    /// Dropped — confidence below threshold.
    DroppedLowConfidence(f32),
    /// Dropped — duplicate of existing memory.
    DroppedDuplicate(Uuid),
    /// Dropped — invalid scope/category.
    DroppedInvalid(String),
}

/// Reconcile a batch of extracted candidates against the existing store.
///
/// For each candidate:
/// 1. Validate scope/category
/// 2. Drop if confidence below threshold
/// 3. Embed the content
/// 4. Check for duplicates (cosine > 0.85)
/// 5. Check for supersession (same entity, different content)
/// 6. Persist with appropriate lifecycle
pub async fn reconcile_candidates(
    candidates: Vec<ExtractedCandidate>,
    session_id: &str,
    project_dir_name: &str,
    db: &MemoryDb,
    backend: &dyn InferenceBackend,
    config: &Config,
) -> Result<Vec<ReconcileOutcome>> {
    let mut outcomes = Vec::new();

    for candidate in candidates {
        let outcome = reconcile_one(
            candidate,
            session_id,
            project_dir_name,
            db,
            backend,
            config,
        )
        .await?;
        outcomes.push(outcome);
    }

    Ok(outcomes)
}

async fn reconcile_one(
    candidate: ExtractedCandidate,
    session_id: &str,
    project_dir_name: &str,
    db: &MemoryDb,
    backend: &dyn InferenceBackend,
    config: &Config,
) -> Result<ReconcileOutcome> {
    // 1. Validate scope and category
    let mut scope: Scope = match candidate.scope.parse() {
        Ok(s) => s,
        Err(_) => {
            return Ok(ReconcileOutcome::DroppedInvalid(format!(
                "invalid scope: {}",
                candidate.scope
            )));
        }
    };

    // Demote personal → project when scope confidence is low.
    // False-personal is expensive (wrong facts surfaced globally);
    // false-project is cheap (just means a fact stays project-local).
    if scope == Scope::Personal && candidate.scope_confidence < config.scope_demotion_threshold {
        tracing::info!(
            "Demoting to project scope (scope_confidence={:.2}): {}",
            candidate.scope_confidence,
            &candidate.content[..candidate.content.len().min(60)],
        );
        scope = Scope::Project;
    }
    let category: Category = match candidate.category.parse() {
        Ok(c) => c,
        Err(_) => {
            return Ok(ReconcileOutcome::DroppedInvalid(format!(
                "invalid category: {}",
                candidate.category
            )));
        }
    };

    // 2. Confidence gating
    if candidate.confidence < config.confidence_drop_threshold {
        return Ok(ReconcileOutcome::DroppedLowConfidence(
            candidate.confidence,
        ));
    }

    // 3. Embed the content
    let embedding = backend.embed(&candidate.content)?;

    // 4. Check for duplicates via vector search
    let similar = db.vector_search(embedding.clone(), 1, None).await?;
    if let Some(existing) = similar.first() {
        let similarity = cosine_similarity(&embedding, &existing.embedding);
        if similarity > config.dedup_cosine_threshold {
            tracing::debug!(
                "Duplicate detected (similarity={similarity:.3}): {:?} ≈ {:?}",
                &candidate.content[..candidate.content.len().min(50)],
                &existing.content[..existing.content.len().min(50)],
            );
            return Ok(ReconcileOutcome::DroppedDuplicate(existing.id));
        }
    }

    // 5. Check for supersession (same entity, newer content)
    // Similarity 0.85+ = duplicate (handled above in step 4).
    // 0.7-0.85 = very similar claim, different wording → supersession.
    //   e.g., "Prefers Vim" → "Switched to Neovim"
    // Below 0.7 = different facts about the same topic → coexist.
    //   e.g., "Values simplicity" and "Values testability" under design-philosophy.
    // Merge/combine of related-but-distinct memories is deferred to the settling pass.
    let mut supersedes_ids: Vec<Uuid> = Vec::new();
    if let Some(ref entity) = candidate.entity {
        let entity_matches = db
            .list(&crate::storage::db::ListFilters {
                entity: Some(entity.clone()),
                limit: Some(10),
                ..Default::default()
            })
            .await?;

        for existing in &entity_matches {
            // Skip already-superseded memories
            if existing.superseded_by.is_some() {
                continue;
            }
            let similarity = cosine_similarity(&embedding, &existing.embedding);
            if similarity >= config.supersession_cosine_min && similarity < config.supersession_cosine_max {
                tracing::info!(
                    "Superseding memory {} (entity={entity}, similarity={similarity:.3}): \"{}\" → \"{}\"",
                    &existing.id.to_string()[..8],
                    crate::pipeline::curate::truncate_str(&existing.content, 50),
                    crate::pipeline::curate::truncate_str(&candidate.content, 50),
                );
                supersedes_ids.push(existing.id);
            }
        }
    }

    // 6. Determine lifecycle and persist
    let now = Utc::now();
    let lifecycle = if candidate.confidence >= config.confidence_auto_accept_threshold {
        Lifecycle::Accepted
    } else {
        Lifecycle::Candidate
    };

    // If superseding, point to the most recent one we're replacing
    let supersedes = supersedes_ids.first().copied();

    let memory = Memory {
        id: Uuid::new_v4(),
        content: candidate.content,
        embedding,
        scope,
        category,
        entity: candidate.entity,
        source_project: Some(project_dir_name.to_string()),
        source_session: session_id.to_string(),
        source_timestamp: now,
        lifecycle,
        confidence: candidate.confidence,
        created_at: now,
        updated_at: now,
        accessed_at: None,
        supersedes,
        superseded_by: None,
    };

    let id = memory.id;
    db.insert(vec![memory]).await?;

    // Mark superseded memories with back-link to the new one
    for old_id in &supersedes_ids {
        if let Err(e) = db.mark_superseded(old_id, &id).await {
            tracing::warn!(
                "Failed to mark memory {} as superseded: {e}",
                &old_id.to_string()[..8],
            );
        }
    }

    match lifecycle {
        Lifecycle::Accepted => Ok(ReconcileOutcome::Accepted(id)),
        Lifecycle::Candidate => Ok(ReconcileOutcome::Candidate(id)),
    }
}

/// Compute cosine similarity between two vectors.
pub(crate) fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot / (norm_a * norm_b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        assert!(cosine_similarity(&a, &b).abs() < 0.001);
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        assert!((cosine_similarity(&a, &b) + 1.0).abs() < 0.001);
    }
}
