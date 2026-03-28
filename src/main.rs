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
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("engram=info".parse()?),
        )
        .init();

    let cli = Cli::parse();
    cli.run().await
}
