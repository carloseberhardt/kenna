use anyhow::Result;

use super::show::find_by_prefix;
use crate::storage::db::MemoryDb;
use crate::storage::models::{Lifecycle, Scope};

pub async fn accept(db: &MemoryDb, id_prefix: &str) -> Result<()> {
    let memory = find_by_prefix(db, id_prefix).await?;
    if memory.lifecycle == Lifecycle::Accepted {
        println!("Memory {} is already accepted.", &memory.id.to_string()[..8]);
        return Ok(());
    }
    db.update_lifecycle(&memory.id, Lifecycle::Accepted).await?;
    println!("Accepted memory {}.", &memory.id.to_string()[..8]);
    Ok(())
}

pub async fn delete(db: &MemoryDb, id_prefix: &str) -> Result<()> {
    let memory = find_by_prefix(db, id_prefix).await?;
    db.delete(&memory.id).await?;
    println!("Deleted memory {}.", &memory.id.to_string()[..8]);
    Ok(())
}

pub async fn promote(db: &MemoryDb, id_prefix: &str) -> Result<()> {
    let memory = find_by_prefix(db, id_prefix).await?;
    if memory.scope == Scope::Personal {
        println!("Memory {} is already personal scope.", &memory.id.to_string()[..8]);
        return Ok(());
    }
    db.update_scope(&memory.id, Scope::Personal).await?;
    println!("Promoted memory {} to personal scope.", &memory.id.to_string()[..8]);
    Ok(())
}
