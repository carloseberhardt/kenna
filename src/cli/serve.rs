use anyhow::Result;

use crate::mcp::server::run_server;

pub async fn run() -> Result<()> {
    run_server().await
}
