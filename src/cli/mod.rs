mod export;
mod list;
mod manage;
mod reconcile;
mod search;
mod serve;
mod settle;
mod show;
mod stats;

use anyhow::Result;
use clap::{Parser, Subcommand};

use crate::config::Config;
use crate::inference::InferenceBackend;
use crate::inference::llama::LlamaBackend;
use crate::storage::db::MemoryDb;
use crate::storage::models::{Category, Lifecycle, Scope};

#[derive(Parser)]
#[command(name = "kenna", about = "Durable, implicit memory for Claude Code")]
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
        /// Process only sessions whose ID starts with this prefix
        #[arg(long)]
        session: Option<String>,
    },
    /// Consolidate memories: cross-project promotion + entity synthesis
    Settle {
        /// Preview what would change without making modifications
        #[arg(long)]
        dry_run: bool,
    },
    /// Start the MCP server (stdio transport, for Claude Code integration)
    Serve,
    /// List memories with optional filters
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
    /// Show full details of a single memory
    Show {
        /// Memory ID (UUID, prefix match supported)
        id: String,
    },
    /// Search memories (vector search if embedding model available, else keyword)
    Search {
        /// Search query
        query: String,
        /// Maximum number of results
        #[arg(short = 'n', long, default_value = "10")]
        limit: usize,
    },
    /// Promote a candidate memory to accepted
    Accept {
        /// Memory ID
        id: String,
    },
    /// Delete a memory permanently
    Delete {
        /// Memory ID
        id: String,
    },
    /// Promote a memory from project scope to personal scope
    Promote {
        /// Memory ID
        id: String,
    },
    /// Show statistics about stored memories
    Stats,
    /// Export all memories as JSON
    Export,
    /// Insert a test memory (development helper)
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
        let db = MemoryDb::open(&Config::db_path()).await?;

        match self.command {
            Command::Reconcile { dry_run, limit, model, session } => {
                reconcile::run(dry_run, limit, model, session).await
            }
            Command::Settle { dry_run } => {
                settle::run(dry_run).await
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
                use crate::storage::models::Memory;
                let mut memory =
                    Memory::new_placeholder(content, scope, category, entity, confidence);
                // Use real embeddings if model available
                if let Some(backend) = try_load_embedding_backend(&config) {
                    let embedding = backend.embed(&memory.content)?;
                    memory.embedding = embedding;
                    println!("Inserting memory {} (with embedding)", memory.id);
                } else {
                    println!("Inserting memory {} (no embedding model)", memory.id);
                }
                db.insert(vec![memory]).await?;
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
