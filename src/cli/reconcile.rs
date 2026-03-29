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
use crate::storage::db::EngramDb;

use crate::pipeline::extract::ExtractedCandidate;

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

pub async fn run(dry_run: bool, limit: Option<usize>, model_override: Option<String>, session_filter: Option<String>) -> Result<()> {
    let mut config = Config::load()?;
    if let Some(model) = model_override {
        config.extraction_model = model;
    }

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

    if dry_run {
        print_discovery_summary(&sessions, excluded_count);
        println!();
    }

    if sessions.is_empty() {
        println!("No sessions to process.");
        return Ok(());
    }

    if dry_run {
        // Dry run: just show preprocessing output
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

    println!("Loading extraction model...");
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

    for (session_idx, session) in sessions.iter().enumerate() {
        if !cursor.needs_processing(&session.path) {
            skipped_cursor += 1;
            continue;
        }

        let byte_offset = cursor.byte_offset(&session.path);
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
            cursor.mark_processed(&session.path);
            cursor.save()?;
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
        } else {
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

    // ── Phase B: Curate with Qwen3 ──
    drop(extract_backend); // free GPU memory

    println!("Loading curation model...");
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

    // ── Phase C: Embed + Reconcile ──
    drop(curate_backend); // free GPU memory

    println!("Loading embedding model...");
    let embed_backend = LlamaBackend::embedding_only(
        embed_path.to_str().unwrap(),
        0,
    )?;

    let db = EngramDb::open(&Config::db_path()).await?;
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
            cursor.mark_processed(&extraction.session_path);
            cursor.save()?;
            continue;
        }

        let outcomes = reconcile_candidates(
            all_candidates,
            &extraction.session_id,
            &extraction.project_dir_name,
            &db,
            &embed_backend,
            &config,
        )
        .await?;

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

        cursor.mark_processed(&extraction.session_path);
        cursor.save()?;
    }

    println!();
    println!("─────────────────────────");
    println!("Reconciliation complete:");
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
