use anyhow::{Result, bail};

use crate::storage::db::EngramDb;
use crate::storage::models::Engram;

pub async fn run(db: &EngramDb, id_prefix: &str) -> Result<()> {
    let engram = find_by_prefix(db, id_prefix).await?;
    print_detail(&engram);
    Ok(())
}

pub async fn find_by_prefix(db: &EngramDb, prefix: &str) -> Result<Engram> {
    // Try exact UUID parse first
    if let Ok(uuid) = prefix.parse() {
        if let Some(e) = db.get_by_id(&uuid).await? {
            return Ok(e);
        }
        bail!("engram not found: {prefix}");
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
        0 => bail!("no engram found matching prefix: {prefix}"),
        1 => Ok(matches.into_iter().next().unwrap()),
        n => bail!("ambiguous prefix '{prefix}': matches {n} engrams"),
    }
}

fn print_detail(e: &Engram) {
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
