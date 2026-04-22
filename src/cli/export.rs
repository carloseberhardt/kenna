use anyhow::Result;

use crate::storage::db::MemoryDb;

pub async fn run(db: &MemoryDb) -> Result<()> {
    let memories = db.export_all().await?;
    let json = serde_json::to_string_pretty(&memories)?;
    println!("{json}");
    Ok(())
}
