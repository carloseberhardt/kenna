use anyhow::Result;

use crate::config::Config;
use crate::inference::llama::LlamaBackend;
use crate::pipeline::curate::curate_candidates;
use crate::pipeline::discover::{
    count_excluded_projects, discover_sessions, print_discovery_summary,
};
use crate::pipeline::extract::extract_from_chunk;
use crate::pipeline::preprocess::{preprocess_session, ConversationChunk};
use crate::pipeline::reconcile::{reconcile_candidates, ReconcileOutcome};
use crate::storage::cursor::Cursor;
use crate::storage::db::MemoryDb;

use crate::pipeline::extract::ExtractedCandidate;

/// How deep a `--dry-run` runs the pipeline. Declaration order is depth order
/// (each level is a superset of the one above), so we can gate stages with
/// `stage >= DryRunStage::Curate`. Every level is read-only: nothing is written
/// to the DB and the cursor is never advanced.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, clap::ValueEnum)]
pub enum DryRunStage {
    /// Preprocessing only — show how sessions chunk. No model is loaded.
    Chunks,
    /// + Phase A: run extraction and print raw candidates.
    Extract,
    /// + Phase B: run curation and print survivors.
    Curate,
    /// + Phase C (read-only): print would-be reconcile outcomes against the live store.
    Reconcile,
}

/// Candidates extracted from a single conversation chunk, with source text.
struct ChunkExtraction {
    candidates: Vec<ExtractedCandidate>,
    /// The conversation text from the chunk that produced these candidates.
    source_text: String,
}

/// Collected extraction results for a session, pending curation.
struct SessionExtraction {
    session_id: String,
    project_dir_name: String,
    session_path: std::path::PathBuf,
    /// Per-chunk extractions — each batch paired with the text that produced it.
    chunks: Vec<ChunkExtraction>,
}

pub async fn run(dry: Option<DryRunStage>, limit: Option<usize>, model_override: Option<String>, curate_model_override: Option<String>, session_filter: Option<String>) -> Result<()> {
    let mut config = Config::load()?;
    if let Some(model) = model_override {
        config.extraction_model = model;
    }
    if let Some(model) = curate_model_override {
        config.curation_model = model;
    }

    // Control axes derived from the dry-run depth:
    //   commit         — write to the DB and advance the cursor (normal run only)
    //   run_curate     — run Phase B (curation)
    //   run_reconcile  — run Phase C (embed + reconcile; read-only when !commit)
    // A normal run (dry = None) does everything and commits.
    let commit = dry.is_none();
    let run_curate = dry.map_or(true, |s| s >= DryRunStage::Curate);
    let run_reconcile = dry.map_or(true, |s| s >= DryRunStage::Reconcile);

    // Discover sessions
    let excluded_count = count_excluded_projects(&config.reconcile.exclude_projects)?;
    let mut sessions = discover_sessions(&config.reconcile.exclude_projects, limit)?;

    // Filter to a specific session if requested
    if let Some(ref prefix) = session_filter {
        sessions.retain(|s| s.session_id.starts_with(prefix.as_str()));
        if sessions.is_empty() {
            println!("No sessions found matching prefix '{prefix}'.");
            return Ok(());
        }
        println!("Filtered to {} session(s) matching '{prefix}'.", sessions.len());
    }

    // Load cursor for incremental processing
    let mut cursor = Cursor::load()?;

    if dry.is_some() {
        print_discovery_summary(&sessions, excluded_count);
        println!();
    }

    if sessions.is_empty() {
        println!("No sessions to process.");
        return Ok(());
    }

    if matches!(dry, Some(DryRunStage::Chunks)) {
        // Chunks level: just show preprocessing output, no model load.
        let mut total_chunks = 0usize;
        let mut skipped = 0usize;
        for session in &sessions {
            let chunks = preprocess_session(
                &session.path,
                &session.session_id,
                &session.project_dir_name,
                &config.reconcile,
                session.is_aside,
                0, // dry-run always processes from start
            )?;
            if chunks.is_empty() {
                skipped += 1;
                continue;
            }
            print_session_chunks(&session.session_id, &session.project_dir_name, &chunks);
            total_chunks += chunks.len();
        }
        println!("─────────────────────────");
        println!(
            "Summary: {} chunks from {} sessions ({} skipped, < {} human turns)",
            total_chunks,
            sessions.len() - skipped,
            skipped,
            config.reconcile.min_human_turns,
        );
        return Ok(());
    }

    // ── Phase A: Extract with Gemma ──
    let extract_path = Config::models_dir().join(&config.extraction_model);
    let embed_path = Config::models_dir().join(&config.embedding_model);
    let curate_path = Config::models_dir().join(&config.curation_model);

    if !extract_path.exists() {
        anyhow::bail!("Extraction model not found at {}", extract_path.display());
    }
    if !embed_path.exists() {
        anyhow::bail!("Embedding model not found at {}", embed_path.display());
    }
    if !curate_path.exists() {
        anyhow::bail!("Curation model not found at {}", curate_path.display());
    }

    // Check GPU availability before loading any models. Soft exit if another
    // workload is using the VRAM — next scheduled run will try again.
    if let Err(msg) = crate::inference::gpu_check::ensure_free_vram(
        config.reconcile_min_free_vram_gb,
        "reconcile",
    ) {
        println!("{msg}");
        tracing::warn!("{msg}");
        return Ok(());
    }

    println!("Loading extraction model ({})...", config.extraction_model);
    let extract_backend = LlamaBackend::new(crate::inference::llama::LlamaConfig {
        generation_model_path: Some(extract_path.to_str().unwrap().to_string()),
        embedding_model_path: None,
        n_gpu_layers: 99,
        ..Default::default()
    })?;

    let total_sessions = sessions.len();
    let is_interactive = atty::is(atty::Stream::Stderr);
    let mut extractions: Vec<SessionExtraction> = Vec::new();
    let mut total_chunks = 0usize;
    let mut skipped_sessions = 0usize;
    let mut skipped_cursor = 0usize;
    let mut total_extracted = 0usize;
    let mut extract_errors = 0usize;

    for (session_idx, session) in sessions.iter().enumerate() {
        // Honor the cursor only on a committing run; dry runs always re-process
        // from the start so the same session can be previewed repeatedly.
        if commit && !cursor.needs_processing(&session.path) {
            skipped_cursor += 1;
            continue;
        }

        let byte_offset = if commit { cursor.byte_offset(&session.path) } else { 0 };
        let chunks = preprocess_session(
            &session.path,
            &session.session_id,
            &session.project_dir_name,
            &config.reconcile,
            session.is_aside,
            byte_offset,
        )?;

        if chunks.is_empty() {
            skipped_sessions += 1;
            if commit {
                cursor.mark_processed(&session.path);
                cursor.save()?;
            }
            continue;
        }

        if is_interactive {
            let short_id = if session.session_id.len() > 8 {
                &session.session_id[..8]
            } else {
                &session.session_id
            };
            eprint!(
                "\r[{}/{}] extracting {} ({} chunks)    ",
                session_idx + 1,
                total_sessions,
                short_id,
                chunks.len(),
            );
        }

        let mut chunk_extractions = Vec::new();

        for chunk in &chunks {
            total_chunks += 1;

            let candidates = match extract_from_chunk(&extract_backend, chunk, 2048) {
                Ok(c) => c,
                Err(e) => {
                    extract_errors += 1;
                    tracing::warn!(
                        "Extraction failed for chunk {} of {}: {e}",
                        chunk.chunk_index,
                        session.session_id,
                    );
                    continue;
                }
            };

            if !candidates.is_empty() {
                total_extracted += candidates.len();
                chunk_extractions.push(ChunkExtraction {
                    candidates,
                    source_text: chunk.text.clone(),
                });
            }
        }

        if !chunk_extractions.is_empty() {
            extractions.push(SessionExtraction {
                session_id: session.session_id.clone(),
                project_dir_name: session.project_dir_name.clone(),
                session_path: session.path.clone(),
                chunks: chunk_extractions,
            });
        } else if commit {
            cursor.mark_processed(&session.path);
            cursor.save()?;
        }
    }

    if is_interactive {
        eprintln!();
    }

    if extractions.is_empty() {
        println!("No candidates extracted.");
        return Ok(());
    }

    println!(
        "Extracted {} candidates from {} sessions.",
        total_extracted,
        extractions.len(),
    );

    // Stop here for `--dry-run=extract`: print raw candidates, load no further models.
    if !run_curate {
        print_extracted(&extractions, total_extracted, extract_errors);
        return Ok(());
    }

    // ── Phase B: Curate with Qwen3 ──
    drop(extract_backend); // free GPU memory

    println!("Loading curation model ({})...", config.curation_model);
    let curate_backend = LlamaBackend::new(crate::inference::llama::LlamaConfig {
        generation_model_path: Some(curate_path.to_str().unwrap().to_string()),
        embedding_model_path: None,
        n_gpu_layers: 99,
        ..Default::default()
    })?;

    let mut total_curated_dropped = 0usize;
    let num_extractions = extractions.len();

    for (i, extraction) in extractions.iter_mut().enumerate() {
        let total_candidates: usize = extraction.chunks.iter().map(|c| c.candidates.len()).sum();
        if is_interactive {
            let short_id = if extraction.session_id.len() > 8 {
                &extraction.session_id[..8]
            } else {
                &extraction.session_id
            };
            eprint!(
                "\r[{}/{}] curating {} ({} candidates, {} chunks)    ",
                i + 1,
                num_extractions,
                short_id,
                total_candidates,
                extraction.chunks.len(),
            );
        }

        // Curate each chunk's candidates against the chunk's own source text
        for chunk_ext in &mut extraction.chunks {
            let before = chunk_ext.candidates.len();
            match curate_candidates(
                &curate_backend,
                &chunk_ext.candidates,
                &chunk_ext.source_text,
                4096,
            ) {
                Ok(curated) => {
                    total_curated_dropped += before - curated.len();
                    chunk_ext.candidates = curated;
                }
                Err(e) => {
                    tracing::warn!(
                        "Curation failed for session {} chunk, keeping all: {e}",
                        extraction.session_id,
                    );
                }
            }
        }
    }

    if is_interactive {
        eprintln!();
    }

    println!(
        "Curation: kept {} of {} ({} dropped).",
        total_extracted - total_curated_dropped,
        total_extracted,
        total_curated_dropped,
    );

    // Stop here for `--dry-run=curate`: print survivors, skip the embed model.
    if !run_reconcile {
        print_curated(&extractions, total_extracted, total_curated_dropped);
        return Ok(());
    }

    // ── Phase C: Embed + Reconcile ──
    drop(curate_backend); // free GPU memory

    println!("Loading embedding model...");
    let embed_backend = LlamaBackend::embedding_only(
        embed_path.to_str().unwrap(),
        0,
        config.embedding_dimensions,
    )?;

    let db = MemoryDb::open(&Config::db_path()).await?;
    let mut total_accepted = 0usize;
    let mut total_candidates = 0usize;
    let mut total_dropped = 0usize;

    for extraction in &extractions {
        let all_candidates: Vec<ExtractedCandidate> = extraction
            .chunks
            .iter()
            .flat_map(|c| c.candidates.clone())
            .collect();

        if all_candidates.is_empty() {
            if commit {
                cursor.mark_processed(&extraction.session_path);
                cursor.save()?;
            }
            continue;
        }

        // On a dry run, keep each candidate's text so we can pair it with its
        // would-be outcome for display (the vec itself is consumed below).
        let dry_contents: Vec<String> = if commit {
            Vec::new()
        } else {
            all_candidates.iter().map(|c| c.content.clone()).collect()
        };

        let outcomes = reconcile_candidates(
            all_candidates,
            &extraction.session_id,
            &extraction.project_dir_name,
            &db,
            &embed_backend,
            &config,
            commit,
        )
        .await?;

        if !commit {
            print_reconcile_outcomes(&extraction.session_id, &dry_contents, &outcomes);
        }

        for outcome in &outcomes {
            match outcome {
                ReconcileOutcome::Accepted(id) => {
                    total_accepted += 1;
                    let short = &id.to_string()[..8];
                    tracing::info!("Accepted: {short}");
                }
                ReconcileOutcome::Candidate(id) => {
                    total_candidates += 1;
                    let short = &id.to_string()[..8];
                    tracing::info!("Candidate: {short}");
                }
                ReconcileOutcome::DroppedLowConfidence(c) => {
                    total_dropped += 1;
                    tracing::debug!("Dropped (low confidence: {c:.2})");
                }
                ReconcileOutcome::DroppedDuplicate(id) => {
                    total_dropped += 1;
                    let short = &id.to_string()[..8];
                    tracing::debug!("Dropped (duplicate of {short})");
                }
                ReconcileOutcome::DroppedInvalid(reason) => {
                    total_dropped += 1;
                    tracing::debug!("Dropped (invalid: {reason})");
                }
            }
        }

        if commit {
            cursor.mark_processed(&extraction.session_path);
            cursor.save()?;
        }
    }

    println!();
    println!("─────────────────────────");
    println!(
        "{}",
        if commit { "Reconciliation complete:" } else { "Reconciliation preview (read-only, nothing written):" }
    );
    println!("  Sessions found:     {total_sessions}");
    if skipped_cursor > 0 {
        println!("  Already processed:  {skipped_cursor}");
    }
    println!("  Sessions processed: {}", extractions.len());
    if skipped_sessions > 0 {
        println!(
            "  Skipped (< {} turns): {skipped_sessions}",
            config.reconcile.min_human_turns
        );
    }
    println!("  Chunks processed:   {total_chunks}");
    println!("  Extracted:          {total_extracted}");
    println!("  Curated out:        {total_curated_dropped}");
    println!("  Accepted:           {total_accepted}");
    println!("  Candidates:         {total_candidates}");
    println!("  Dropped (dedup):    {total_dropped}");

    Ok(())
}

fn print_session_chunks(
    session_id: &str,
    project_dir: &str,
    chunks: &[ConversationChunk],
) {
    let short_id = if session_id.len() > 8 {
        &session_id[..8]
    } else {
        session_id
    };

    println!("── {short_id} ({project_dir}) ── {} chunk(s)", chunks.len());
    for chunk in chunks {
        println!(
            "  chunk {}: ~{} tokens",
            chunk.chunk_index, chunk.approx_tokens
        );
        let preview: Vec<&str> = chunk.text.lines().take(3).collect();
        for line in &preview {
            let truncated = if line.len() > 100 {
                format!("{}…", &line[..100])
            } else {
                line.to_string()
            };
            println!("    | {truncated}");
        }
        if chunk.text.lines().count() > 3 {
            println!("    | ...");
        }
        println!();
    }
}

/// First 8 chars of a session id, for compact headers.
fn short_id(session_id: &str) -> &str {
    if session_id.len() > 8 { &session_id[..8] } else { session_id }
}

/// Print one candidate as an indented, single-line summary.
fn print_candidate(c: &ExtractedCandidate) {
    let entity = c.entity.as_deref().unwrap_or("-");
    println!(
        "  [{:<10} {:<8} conf {:.2} {}] {}",
        c.scope,
        c.category,
        c.confidence,
        entity,
        crate::pipeline::curate::truncate_str(&c.content, 100),
    );
}

/// List every session's candidates grouped by session. Shared by the
/// `--dry-run=extract` (raw) and `--dry-run=curate` (survivors) printers.
fn print_candidate_listing(extractions: &[SessionExtraction]) {
    for extraction in extractions {
        println!(
            "── {} ({}) ──",
            short_id(&extraction.session_id),
            extraction.project_dir_name,
        );
        for chunk in &extraction.chunks {
            for candidate in &chunk.candidates {
                print_candidate(candidate);
            }
        }
        println!();
    }
}

/// `--dry-run=extract`: raw candidates plus extraction-failure count.
fn print_extracted(extractions: &[SessionExtraction], total_extracted: usize, extract_errors: usize) {
    print_candidate_listing(extractions);
    println!("─────────────────────────");
    println!("Extraction preview (read-only, nothing written):");
    println!("  Extracted:       {total_extracted}");
    println!("  Extract errors:  {extract_errors}");
}

/// `--dry-run=curate`: surviving candidates plus kept/dropped tally.
fn print_curated(extractions: &[SessionExtraction], total_extracted: usize, dropped: usize) {
    print_candidate_listing(extractions);
    println!("─────────────────────────");
    println!("Curation preview (read-only, nothing written):");
    println!("  Extracted:    {total_extracted}");
    println!("  Curated out:  {dropped}");
    println!("  Kept:         {}", total_extracted - dropped);
}

/// `--dry-run=reconcile`: pair each candidate with its would-be outcome.
/// `contents` and `outcomes` are 1:1 and in candidate order.
fn print_reconcile_outcomes(session_id: &str, contents: &[String], outcomes: &[ReconcileOutcome]) {
    println!("── {} ──", short_id(session_id));
    for (content, outcome) in contents.iter().zip(outcomes.iter()) {
        let label = match outcome {
            ReconcileOutcome::Accepted(_) => "would accept   ".to_string(),
            ReconcileOutcome::Candidate(_) => "would candidate".to_string(),
            ReconcileOutcome::DroppedDuplicate(id) => {
                format!("duplicate of {}", &id.to_string()[..8])
            }
            ReconcileOutcome::DroppedLowConfidence(c) => format!("drop low-conf {c:.2}"),
            ReconcileOutcome::DroppedInvalid(reason) => format!("drop invalid: {reason}"),
        };
        println!("  [{label}] {}", crate::pipeline::curate::truncate_str(content, 90));
    }
    println!();
}
