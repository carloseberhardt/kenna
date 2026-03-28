mod export;
mod list;
mod manage;
mod reconcile;
mod search;
mod serve;
mod show;
mod stats;

use anyhow::Result;
use clap::{Parser, Subcommand};

use crate::config::Config;
use crate::inference::InferenceBackend;
use crate::inference::llama::LlamaBackend;
use crate::storage::db::EngramDb;
use crate::storage::models::{Category, Lifecycle, Scope};

#[derive(Parser)]
#[command(name = "engram", about = "Durable, implicit memory for Claude Code")]
pub struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Run the extraction pipeline on Claude Code session transcripts
    Reconcile {
        /// Preview what would be processed without running extraction
        #[arg(long)]
        dry_run: bool,
        /// Maximum number of sessions to process
        #[arg(long)]
        limit: Option<usize>,
        /// Override extraction model GGUF filename (in models dir)
        #[arg(long)]
        model: Option<String>,
    },
    /// Start the MCP server (stdio transport, for Claude Code integration)
    Serve,
    /// List engrams with optional filters
    List {
        /// Filter by lifecycle: show only pending candidates
        #[arg(long)]
        pending: bool,
        /// Filter by scope (personal | project)
        #[arg(long)]
        scope: Option<Scope>,
        /// Filter by category
        #[arg(long)]
        category: Option<Category>,
        /// Filter by entity key
        #[arg(long)]
        entity: Option<String>,
        /// Maximum number of results
        #[arg(short = 'n', long, default_value = "20")]
        limit: usize,
    },
    /// Show full details of a single engram
    Show {
        /// Engram ID (UUID, prefix match supported)
        id: String,
    },
    /// Search engrams (vector search if embedding model available, else keyword)
    Search {
        /// Search query
        query: String,
        /// Maximum number of results
        #[arg(short = 'n', long, default_value = "10")]
        limit: usize,
    },
    /// Promote a candidate engram to accepted
    Accept {
        /// Engram ID
        id: String,
    },
    /// Delete an engram permanently
    Delete {
        /// Engram ID
        id: String,
    },
    /// Promote an engram from project scope to personal scope
    Promote {
        /// Engram ID
        id: String,
    },
    /// Show statistics about stored engrams
    Stats,
    /// Export all engrams as JSON
    Export,
    /// Insert a test engram (development helper)
    #[command(name = "debug-insert")]
    DebugInsert {
        /// Content text
        content: String,
        /// Scope (personal | project)
        #[arg(long, default_value = "personal")]
        scope: Scope,
        /// Category
        #[arg(long, default_value = "fact")]
        category: Category,
        /// Entity key
        #[arg(long)]
        entity: Option<String>,
        /// Confidence score (0.0 - 1.0)
        #[arg(long, default_value = "0.9")]
        confidence: f32,
    },
}

impl Cli {
    pub async fn run(self) -> Result<()> {
        Config::ensure_dirs()?;
        let config = Config::load()?;
        let db = EngramDb::open(&Config::db_path()).await?;

        match self.command {
            Command::Reconcile { dry_run, limit, model } => {
                reconcile::run(dry_run, limit, model).await
            }
            Command::Serve => serve::run().await,
            Command::List {
                pending,
                scope,
                category,
                entity,
                limit,
            } => {
                let lifecycle = if pending {
                    Some(Lifecycle::Candidate)
                } else {
                    None
                };
                list::run(&db, scope, category, lifecycle, entity, limit).await
            }
            Command::Show { id } => show::run(&db, &id).await,
            Command::Search { query, limit } => {
                // Try to load embedding model for vector search
                let backend = try_load_embedding_backend(&config);
                let backend_ref: Option<&dyn InferenceBackend> =
                    backend.as_ref().map(|b| b as &dyn InferenceBackend);
                search::run(&db, &query, limit, backend_ref).await
            }
            Command::Accept { id } => manage::accept(&db, &id).await,
            Command::Delete { id } => manage::delete(&db, &id).await,
            Command::Promote { id } => manage::promote(&db, &id).await,
            Command::Stats => stats::run(&db).await,
            Command::Export => export::run(&db).await,
            Command::DebugInsert {
                content,
                scope,
                category,
                entity,
                confidence,
            } => {
                use crate::storage::models::Engram;
                let mut engram =
                    Engram::new_placeholder(content, scope, category, entity, confidence);
                // Use real embeddings if model available
                if let Some(backend) = try_load_embedding_backend(&config) {
                    let embedding = backend.embed(&engram.content)?;
                    engram.embedding = embedding;
                    println!("Inserting engram {} (with embedding)", engram.id);
                } else {
                    println!("Inserting engram {} (no embedding model)", engram.id);
                }
                db.insert(vec![engram]).await?;
                println!("Done.");
                Ok(())
            }
        }
    }
}

/// Try to load the embedding model. Returns None if the model file doesn't exist.
fn try_load_embedding_backend(config: &Config) -> Option<LlamaBackend> {
    let model_path = Config::models_dir().join(&config.embedding_model);
    if !model_path.exists() {
        tracing::debug!(
            "Embedding model not found at {}, falling back to keyword search",
            model_path.display()
        );
        return None;
    }

    // Use CPU for embedding model — small model, and Vulkan has issues with BERT
    match LlamaBackend::embedding_only(model_path.to_str()?, 0) {
        Ok(backend) => Some(backend),
        Err(e) => {
            tracing::warn!("Failed to load embedding model: {e}");
            None
        }
    }
}
