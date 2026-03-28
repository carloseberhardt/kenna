use anyhow::Result;

use crate::storage::db::EngramDb;

pub async fn run(db: &EngramDb) -> Result<()> {
    let engrams = db.export_all().await?;
    let json = serde_json::to_string_pretty(&engrams)?;
    println!("{json}");
    Ok(())
}
