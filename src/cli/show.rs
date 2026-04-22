use anyhow::{Result, bail};

use crate::storage::db::MemoryDb;
use crate::storage::models::Memory;

pub async fn run(db: &MemoryDb, id_prefix: &str) -> Result<()> {
    let memory = find_by_prefix(db, id_prefix).await?;
    print_detail(&memory);
    Ok(())
}

pub async fn find_by_prefix(db: &MemoryDb, prefix: &str) -> Result<Memory> {
    // Try exact UUID parse first
    if let Ok(uuid) = prefix.parse() {
        if let Some(e) = db.get_by_id(&uuid).await? {
            return Ok(e);
        }
        bail!("memory not found: {prefix}");
    }

    // Prefix match: list all and filter
    let all = db
        .list(&crate::storage::db::ListFilters {
            limit: Some(100_000),
            ..Default::default()
        })
        .await?;

    let matches: Vec<_> = all
        .into_iter()
        .filter(|e| e.id.to_string().starts_with(prefix))
        .collect();

    match matches.len() {
        0 => bail!("no memory found matching prefix: {prefix}"),
        1 => Ok(matches.into_iter().next().unwrap()),
        n => bail!("ambiguous prefix '{prefix}': matches {n} memories"),
    }
}

fn print_detail(e: &Memory) {
    println!("ID:               {}", e.id);
    println!("Content:          {}", e.content);
    println!("Scope:            {}", e.scope);
    println!("Category:         {}", e.category);
    println!("Entity:           {}", e.entity.as_deref().unwrap_or("-"));
    println!("Lifecycle:        {}", e.lifecycle);
    println!("Confidence:       {:.2}", e.confidence);
    println!("Source project:   {}", e.source_project.as_deref().unwrap_or("-"));
    println!("Source session:   {}", e.source_session);
    println!("Source timestamp: {}", e.source_timestamp);
    println!("Created:          {}", e.created_at);
    println!("Updated:          {}", e.updated_at);
    println!(
        "Last accessed:    {}",
        e.accessed_at
            .map(|d| d.to_string())
            .unwrap_or_else(|| "-".into())
    );
    println!(
        "Supersedes:       {}",
        e.supersedes
            .map(|u| u.to_string())
            .unwrap_or_else(|| "-".into())
    );
    println!(
        "Superseded by:    {}",
        e.superseded_by
            .map(|u| u.to_string())
            .unwrap_or_else(|| "-".into())
    );
}
