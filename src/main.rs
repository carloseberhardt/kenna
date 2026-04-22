mod cli;
mod config;
pub mod inference;
mod mcp;
mod pipeline;
mod storage;

use anyhow::Result;
use clap::Parser;
use cli::Cli;

#[tokio::main]
async fn main() -> Result<()> {
    // Load config early for log level. If config fails, fall back to "warn".
    let log_level = config::Config::load()
        .map(|c| c.log_level)
        .unwrap_or_else(|_| "warn".into());

    // RUST_LOG env var takes precedence over config if set.
    let env_filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| {
            tracing_subscriber::EnvFilter::new(format!("kenna={log_level}"))
        });

    tracing_subscriber::fmt()
        .with_env_filter(env_filter)
        .init();

    let cli = Cli::parse();
    cli.run().await
}
