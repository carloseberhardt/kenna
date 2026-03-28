use anyhow::Result;

use super::show::find_by_prefix;
use crate::storage::db::EngramDb;
use crate::storage::models::{Lifecycle, Scope};

pub async fn accept(db: &EngramDb, id_prefix: &str) -> Result<()> {
    let engram = find_by_prefix(db, id_prefix).await?;
    if engram.lifecycle == Lifecycle::Accepted {
        println!("Engram {} is already accepted.", &engram.id.to_string()[..8]);
        return Ok(());
    }
    db.update_lifecycle(&engram.id, Lifecycle::Accepted).await?;
    println!("Accepted engram {}.", &engram.id.to_string()[..8]);
    Ok(())
}

pub async fn delete(db: &EngramDb, id_prefix: &str) -> Result<()> {
    let engram = find_by_prefix(db, id_prefix).await?;
    db.delete(&engram.id).await?;
    println!("Deleted engram {}.", &engram.id.to_string()[..8]);
    Ok(())
}

pub async fn promote(db: &EngramDb, id_prefix: &str) -> Result<()> {
    let engram = find_by_prefix(db, id_prefix).await?;
    if engram.scope == Scope::Personal {
        println!("Engram {} is already personal scope.", &engram.id.to_string()[..8]);
        return Ok(());
    }
    db.update_scope(&engram.id, Scope::Personal).await?;
    println!("Promoted engram {} to personal scope.", &engram.id.to_string()[..8]);
    Ok(())
}
