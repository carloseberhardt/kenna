use anyhow::Result;
use chrono::Utc;
use uuid::Uuid;

use crate::config::Config;
use crate::inference::InferenceBackend;
use crate::storage::db::MemoryDb;
use crate::storage::models::{Category, Memory, Lifecycle, Scope};
use crate::storage::vector::cosine_similarity;

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
/// When `commit` is false, every decision is computed (embedding, dedup,
/// supersession, lifecycle) but nothing is persisted — used by `--dry-run`.
pub async fn reconcile_candidates(
    candidates: Vec<ExtractedCandidate>,
    session_id: &str,
    project_dir_name: &str,
    db: &MemoryDb,
    backend: &dyn InferenceBackend,
    config: &Config,
    commit: bool,
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
            commit,
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
    commit: bool,
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

    // 4. Check for duplicates — scoped to this project (+ already-personal),
    // not the whole store. Cross-project near-duplicates are left to coexist so
    // the settling pass can count distinct projects and promote. See
    // dedup-scoping-issue.md.
    if let Some((similarity, existing)) = db
        .find_dedup_match(&embedding, scope, Some(project_dir_name))
        .await?
        && similarity > config.dedup_cosine_threshold
    {
        tracing::debug!(
            "Duplicate detected (similarity={similarity:.3}): {:?} ≈ {:?}",
            &candidate.content[..candidate.content.len().min(50)],
            &existing.content[..existing.content.len().min(50)],
        );
        return Ok(ReconcileOutcome::DroppedDuplicate(existing.id));
    }

    // 5. Check for supersession (same entity, newer content)
    // Similarity 0.85+ = duplicate (handled above in step 4).
    // 0.7-0.85 = very similar claim, different wording → supersession.
    //   e.g., "Prefers Vim" → "Switched to Neovim"
    // Below 0.7 = different facts about the same topic → coexist.
    //   e.g., "Values simplicity" and "Values testability" under design-philosophy.
    // Merge/combine of related-but-distinct memories is deferred to the settling pass.
    // Supersession is scoped to the same project too: a newer claim replaces an
    // older one within a project's history, but the same entity across different
    // projects coexists (again, so settle can see the cross-project signal).
    let mut supersedes_ids: Vec<Uuid> = Vec::new();
    if let Some(ref entity) = candidate.entity {
        let entity_matches = db
            .list(&crate::storage::db::ListFilters {
                entity: Some(entity.clone()),
                source_project: Some(project_dir_name.to_string()),
                // No cap: the result set is already narrowed by entity + project,
                // so it is small; an arbitrary cap could miss matches for a busy entity.
                limit: None,
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
    if commit {
        // Insert the new memory and supersede its sources atomically. A plain
        // insert followed by a best-effort per-source supersede loop could half-write
        // on partial failure (new row live, old rows never back-linked), leaving
        // both claims active and the supersession graph one-directional. Reconcile
        // is cursor-checkpointed, so a loud error here is safely re-runnable; an
        // empty supersedes_ids degenerates to a transactional single-row insert.
        db.apply_settlement(memory, &supersedes_ids).await?;
    }

    match lifecycle {
        Lifecycle::Accepted => Ok(ReconcileOutcome::Accepted(id)),
        Lifecycle::Candidate => Ok(ReconcileOutcome::Candidate(id)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::ChatMessage;
    use crate::storage::db::ListFilters;
    use std::collections::HashMap;

    /// Inference backend stub: `embed` looks the content up in a fixed map so a
    /// test can pin exact cosine relationships; `generate` is never called by
    /// `reconcile_one`.
    struct MockBackend {
        embeddings: HashMap<String, Vec<f32>>,
    }

    impl InferenceBackend for MockBackend {
        fn generate(&self, _prompt: &str, _max_tokens: u32) -> Result<String> {
            anyhow::bail!("MockBackend::generate is not used by reconcile")
        }
        fn generate_chat_multi(&self, _messages: &[ChatMessage], _max_tokens: u32) -> Result<String> {
            anyhow::bail!("MockBackend::generate_chat_multi is not used by reconcile")
        }
        fn embed(&self, text: &str) -> Result<Vec<f32>> {
            self.embeddings
                .get(text)
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("no mock embedding registered for {text:?}"))
        }
        fn embedding_dim(&self) -> usize {
            2
        }
    }

    async fn temp_db() -> (MemoryDb, std::path::PathBuf) {
        let path = std::env::temp_dir().join(format!("kenna-recon-test-{}.db", Uuid::new_v4()));
        let db = MemoryDb::open(&path).await.unwrap();
        (db, path)
    }

    fn cleanup(path: &std::path::Path) {
        let _ = std::fs::remove_file(path);
        let _ = std::fs::remove_file(format!("{}-wal", path.display()));
        let _ = std::fs::remove_file(format!("{}-shm", path.display()));
    }

    fn mk_stored(content: &str, embedding: Vec<f32>, entity: &str, project: &str) -> Memory {
        let now = Utc::now();
        Memory {
            id: Uuid::new_v4(),
            content: content.to_string(),
            embedding,
            scope: Scope::Project,
            category: Category::Preference,
            entity: Some(entity.to_string()),
            source_project: Some(project.to_string()),
            source_session: "seed".to_string(),
            source_timestamp: now,
            lifecycle: Lifecycle::Accepted,
            confidence: 0.9,
            created_at: now,
            updated_at: now,
            accessed_at: None,
            supersedes: None,
            superseded_by: None,
        }
    }

    fn candidate(content: &str, entity: Option<&str>, confidence: f32) -> ExtractedCandidate {
        ExtractedCandidate {
            content: content.to_string(),
            scope: "project".to_string(),
            category: "preference".to_string(),
            entity: entity.map(|e| e.to_string()),
            confidence,
            scope_confidence: 0.95,
        }
    }

    // Finding 1: the supersession persistence path is atomic — the new row and
    // the back-link on its source both land.
    #[tokio::test]
    async fn reconcile_supersession_persists_both_links() {
        let (db, path) = temp_db().await;

        // Seed memory A for entity "editor" in project "alpha".
        let a = mk_stored("prefers vim", vec![1.0, 0.0], "editor", "alpha");
        let a_id = a.id;
        db.insert(vec![a]).await.unwrap();

        // Candidate ≈ A at cosine 0.78 — inside the supersession band [0.70, 0.85),
        // below the dedup threshold (0.85), so it supersedes rather than dedups.
        let cand = candidate("switched to neovim", Some("editor"), 0.9);
        let mut embeddings = HashMap::new();
        embeddings.insert(cand.content.clone(), vec![0.78, 0.6258]);
        let backend = MockBackend { embeddings };
        let config = Config::default();

        let outcome = reconcile_one(cand, "sess", "alpha", &db, &backend, &config, true)
            .await
            .unwrap();
        let new_id = match outcome {
            ReconcileOutcome::Accepted(id) => id,
            other => panic!("expected Accepted, got {other:?}"),
        };

        // Both directions of the supersession graph were written in one shot.
        let new_row = db.get_by_id(&new_id).await.unwrap().unwrap();
        assert_eq!(new_row.supersedes, Some(a_id));
        let a_row = db.get_by_id(&a_id).await.unwrap().unwrap();
        assert_eq!(a_row.superseded_by, Some(new_id));
        cleanup(&path);
    }

    // Finding 2: a below-auto-accept fact reconciled twice yields a single row;
    // the second pass dedups against the pending Candidate stored by the first.
    #[tokio::test]
    async fn reconcile_dedups_against_pending_candidate() {
        let (db, path) = temp_db().await;
        let config = Config::default();

        // Confidence 0.7: above drop (0.6), below auto-accept (0.85) → Candidate.
        let make = || candidate("uses tabs not spaces", Some("indent"), 0.7);
        let mut embeddings = HashMap::new();
        embeddings.insert("uses tabs not spaces".to_string(), vec![1.0, 0.0]);
        let backend = MockBackend { embeddings };

        let first = reconcile_one(make(), "sess1", "alpha", &db, &backend, &config, true)
            .await
            .unwrap();
        assert!(matches!(first, ReconcileOutcome::Candidate(_)), "got {first:?}");

        let second = reconcile_one(make(), "sess2", "alpha", &db, &backend, &config, true)
            .await
            .unwrap();
        assert!(
            matches!(second, ReconcileOutcome::DroppedDuplicate(_)),
            "second identical candidate must dedup against the pending row, got {second:?}"
        );

        let all = db.list(&ListFilters::default()).await.unwrap();
        assert_eq!(all.len(), 1, "exactly one row should exist after dedup");
        cleanup(&path);
    }

    // Finding 3: a superseded (dead) row must not veto a new fact. After A→B,
    // a candidate matching dead A at >0.85 supersedes the *live* B and is not
    // dropped as a duplicate of A.
    #[tokio::test]
    async fn reconcile_does_not_drop_against_superseded_row() {
        let (db, path) = temp_db().await;
        let config = Config::default();

        // A (dead) and B (live), B supersedes A — set up via the atomic path.
        let a = mk_stored("prefers vim", vec![1.0, 0.0], "editor", "alpha");
        let a_id = a.id;
        db.insert(vec![a]).await.unwrap();
        let b = mk_stored("switched to neovim", vec![0.6, 0.8], "editor", "alpha");
        let b_id = b.id;
        db.apply_settlement(b, &[a_id]).await.unwrap();

        // Candidate ≈ A at 0.95 (would have dedup-dropped against the dead A row
        // under the old query) and ≈ B at ~0.82 (inside the supersession band).
        let cand = candidate("uses vim again", Some("editor"), 0.9);
        let mut embeddings = HashMap::new();
        embeddings.insert(cand.content.clone(), vec![0.95, 0.3122]);
        let backend = MockBackend { embeddings };

        let outcome = reconcile_one(cand, "sess", "alpha", &db, &backend, &config, true)
            .await
            .unwrap();
        let new_id = match outcome {
            ReconcileOutcome::Accepted(id) => id,
            other => panic!("expected Accepted (not a drop against dead A), got {other:?}"),
        };

        // The reversal supersedes the *live* B, not the corpse A.
        let new_row = db.get_by_id(&new_id).await.unwrap().unwrap();
        assert_eq!(new_row.supersedes, Some(b_id));
        cleanup(&path);
    }
}
