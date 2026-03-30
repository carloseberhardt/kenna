use std::sync::Arc;

use anyhow::Result;
use rmcp::handler::server::router::tool::ToolRouter;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{Implementation, ServerInfo};
use rmcp::{ServerHandler, ServiceExt, tool, tool_handler, tool_router};
use schemars::JsonSchema;
use serde::Deserialize;
use tokio::sync::Mutex;

use crate::config::Config;
use crate::inference::llama::LlamaBackend;
use crate::inference::InferenceBackend;
use crate::storage::db::EngramDb;
use crate::storage::models::Scope;

#[derive(Debug, Deserialize, JsonSchema)]
struct EngramRecallParams {
    /// What you want to recall — a natural language description of the context, topic, or question
    query: String,
    /// Optional scope filter. "personal" for knowledge true about the user regardless of project (hardware, preferences, interests). "project" for codebase-specific decisions and patterns. Omit to search both.
    #[serde(default)]
    scope: Option<String>,
    /// Maximum number of engrams to return
    #[serde(default = "default_limit")]
    limit: i32,
}

fn default_limit() -> i32 {
    5
}

#[derive(Clone)]
pub struct EngramServer {
    tool_router: ToolRouter<Self>,
    db: Arc<Mutex<EngramDb>>,
    backend: Arc<LlamaBackend>,
}

#[tool_router]
impl EngramServer {
    /// Recall what you know about the user from past interactions. Returns facts,
    /// preferences, decisions, interests, opinions, and patterns drawn from the
    /// user's history across all Claude Code sessions. Call this when context about
    /// the user would help you make better decisions or avoid assumptions — for
    /// example, their preferred tools, design philosophy, hardware setup, or how
    /// they like to work. Treat results as things you already know about this
    /// person, not as search results to present.
    ///
    /// The "project" scope filters to knowledge from specific codebases. Include
    /// the project directory name (e.g., "engram", "ralph-trader") in your query
    /// to find project-specific decisions and patterns. The "personal" scope
    /// returns knowledge true about the user regardless of project.
    #[tool(name = "engram_recall")]
    async fn engram_recall(
        &self,
        Parameters(params): Parameters<EngramRecallParams>,
    ) -> String {
        match self.do_recall(params).await {
            Ok(result) => result,
            Err(e) => format!("Error recalling engrams: {e}"),
        }
    }
}

#[tool_handler]
impl ServerHandler for EngramServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::default().with_server_info(
            Implementation::new("engram", env!("CARGO_PKG_VERSION"))
        )
    }
}

impl EngramServer {
    pub fn new(db: EngramDb, backend: LlamaBackend) -> Self {
        Self {
            tool_router: Self::tool_router(),
            db: Arc::new(Mutex::new(db)),
            backend: Arc::new(backend),
        }
    }

    async fn do_recall(&self, params: EngramRecallParams) -> Result<String> {
        let scope_filter = params
            .scope
            .as_deref()
            .map(|s| s.parse::<Scope>())
            .transpose()?;

        let limit = params.limit.max(1) as usize;

        // Embed the query
        let embedding = self.backend.embed(&params.query)?;

        // Search
        let db = self.db.lock().await;
        let results = db
            .vector_search(embedding, limit, scope_filter)
            .await?;

        // Filter out superseded engrams — only show the latest version
        let results: Vec<_> = results
            .into_iter()
            .filter(|e| e.superseded_by.is_none())
            .collect();

        if results.is_empty() {
            return Ok("No relevant engrams found.".to_string());
        }

        // Format results as readable text
        let mut output = String::new();
        for (i, engram) in results.iter().enumerate() {
            if i > 0 {
                output.push('\n');
            }
            let scope_tag = match engram.scope {
                Scope::Personal => "personal",
                Scope::Project => "project",
            };
            let entity_str = engram
                .entity
                .as_deref()
                .map(|e| format!(" [{e}]"))
                .unwrap_or_default();
            output.push_str(&format!(
                "- ({scope_tag}/{category}{entity_str}) {content}",
                category = engram.category,
                content = engram.content,
            ));
        }

        // Update accessed_at timestamps (best-effort, don't fail the recall)
        // TODO: batch update accessed_at for retrieved engrams

        Ok(output)
    }
}

/// Start the MCP server on stdio.
pub async fn run_server() -> Result<()> {
    let config = Config::load()?;
    Config::ensure_dirs()?;

    // Load embedding model for query embedding
    let embed_path = Config::models_dir().join(&config.embedding_model);
    if !embed_path.exists() {
        anyhow::bail!(
            "Embedding model not found at {}",
            embed_path.display()
        );
    }

    eprintln!("engram: loading embedding model...");
    let backend = LlamaBackend::embedding_only(embed_path.to_str().unwrap(), 0)?;

    let db = EngramDb::open(&Config::db_path()).await?;

    let server = EngramServer::new(db, backend);
    eprintln!("engram: MCP server ready");

    let (stdin, stdout) = rmcp::transport::stdio();
    let service = server
        .serve((stdin, stdout))
        .await?;

    service.waiting().await?;
    Ok(())
}
