use anyhow::Result;

use crate::config::Config;
use crate::inference::InferenceBackend;
use crate::inference::llama::LlamaBackend;
use crate::pipeline::settle::{run_settle, SettleReport};
use crate::storage::db::EngramDb;

pub async fn run(dry_run: bool) -> Result<()> {
    let config = Config::load()?;
    let db = EngramDb::open(&Config::db_path()).await?;

    if dry_run {
        println!("Running settle pass (dry run)...\n");
        let report = run_settle(&db, &config, true, None, None).await?;
        print_report(&report, true);
        return Ok(());
    }

    // Dry-run first to check if there's anything to do
    let preview = run_settle(&db, &config, true, None, None).await?;
    if preview.promotions.is_empty() && preview.syntheses.is_empty() {
        println!("Nothing to settle.");
        return Ok(());
    }
    println!(
        "Found {} promotion(s) and {} synthesis(es).",
        preview.promotions.len(),
        preview.syntheses.len(),
    );
    drop(preview);

    // Load both models in a single backend instance.
    // Generation (Qwen3) on GPU, embedding (nomic-embed) on CPU.
    let settling_model = config.settling_model.as_ref().unwrap_or(&config.curation_model);
    let gen_path = Config::models_dir().join(settling_model);
    let embed_path = Config::models_dir().join(&config.embedding_model);

    if !gen_path.exists() {
        anyhow::bail!("Settling model not found at {}", gen_path.display());
    }
    if !embed_path.exists() {
        anyhow::bail!("Embedding model not found at {}", embed_path.display());
    }

    // Check GPU availability before loading the synthesis model.
    if let Err(msg) = crate::inference::gpu_check::ensure_free_vram(
        config.settle_min_free_vram_gb,
        "settle",
    ) {
        println!("{msg}");
        tracing::warn!("{msg}");
        return Ok(());
    }

    println!("Loading models...");
    let backend = LlamaBackend::full(
        gen_path.to_str().unwrap(),
        embed_path.to_str().unwrap(),
        99, // GPU layers for generation model
    )?;

    let generate_fn = |system: &str, user: &str| -> Result<String> {
        // Settling synthesis needs more tokens than curation — Qwen3's <think>
        // block can consume 500+ tokens before producing the JSON output.
        backend.generate_chat(system, user, 4096)
    };
    let embed_fn = |text: &str| -> Result<Vec<f32>> {
        backend.embed(text)
    };

    let report = run_settle(
        &db,
        &config,
        false,
        Some(&generate_fn as &dyn Fn(&str, &str) -> Result<String>),
        Some(&embed_fn as &dyn Fn(&str) -> Result<Vec<f32>>),
    ).await?;

    drop(backend);

    println!();
    print_report(&report, false);

    Ok(())
}

fn print_report(report: &SettleReport, dry_run: bool) {
    let prefix = if dry_run { "Would promote" } else { "Promoted" };

    if !report.promotions.is_empty() {
        println!("── Cross-Project Promotions ──");
        for (i, promo) in report.promotions.iter().enumerate() {
            println!(
                "\n{} {} ({} projects, {} sources):",
                prefix,
                i + 1,
                promo.distinct_projects,
                promo.source_engrams.len(),
            );
            for src in &promo.source_engrams {
                println!("  [{:>20}] {}", src.source_project, src.content);
            }
            if let Some(ref content) = promo.synthesized_content {
                println!("  -> {content}");
            }
            if let Some(id) = promo.new_id {
                println!("  ID: {}", &id.to_string()[..8]);
            }
        }
    }

    let prefix = if dry_run { "Would synthesize" } else { "Synthesized" };

    if !report.syntheses.is_empty() {
        println!("\n── Entity Syntheses ──");
        for synthesis in &report.syntheses {
            println!(
                "\n{} entity '{}' ({} engrams):",
                prefix,
                synthesis.entity,
                synthesis.source_engrams.len(),
            );
            for src in &synthesis.source_engrams {
                println!("  (conf={:.2}) {}", src.confidence, src.content);
            }
            if let Some(ref content) = synthesis.synthesized_content {
                println!("  -> {content}");
            }
            if let Some(id) = synthesis.new_id {
                println!("  ID: {}", &id.to_string()[..8]);
            }
        }
    }

    if report.promotions.is_empty() && report.syntheses.is_empty() {
        println!("Nothing to settle.");
    }

    println!("\n─────────────────────────");
    println!("Promotions:      {}", report.promotions.len());
    println!("Syntheses:       {}", report.syntheses.len());
    println!("Already settled: {}", report.skipped_already_settled);
}
