use anyhow::Result;
use comfy_table::{Cell, Table};

use crate::inference::InferenceBackend;
use crate::storage::db::{MemoryDb, ListFilters};
use crate::storage::models::Memory;

/// Search memories. Uses vector search if an inference backend is provided,
/// otherwise falls back to keyword search.
pub async fn run(
    db: &MemoryDb,
    query: &str,
    limit: usize,
    backend: Option<&dyn InferenceBackend>,
) -> Result<()> {
    let matches = if let Some(backend) = backend {
        let embedding = backend.embed(query)?;
        db.vector_search(embedding, limit, None).await?
    } else {
        keyword_search(db, query, limit).await?
    };

    if matches.is_empty() {
        println!("No memories matching '{query}'.");
        return Ok(());
    }

    print_results(&matches);
    println!("\n{} result(s).", matches.len());
    Ok(())
}

async fn keyword_search(db: &MemoryDb, query: &str, limit: usize) -> Result<Vec<Memory>> {
    let all = db
        .list(&ListFilters {
            limit: Some(100_000),
            ..Default::default()
        })
        .await?;

    let query_lower = query.to_lowercase();
    Ok(all
        .into_iter()
        .filter(|e| {
            e.content.to_lowercase().contains(&query_lower)
                || e.entity
                    .as_ref()
                    .is_some_and(|ent| ent.to_lowercase().contains(&query_lower))
        })
        .take(limit)
        .collect())
}

fn print_results(matches: &[Memory]) {
    let mut table = Table::new();
    table.set_header(vec!["ID", "Scope", "Category", "Conf", "Content"]);

    for e in matches {
        let short_id = &e.id.to_string()[..8];
        let content_preview = if e.content.len() > 70 {
            format!("{}…", &e.content[..70])
        } else {
            e.content.clone()
        };
        table.add_row(vec![
            Cell::new(short_id),
            Cell::new(e.scope.to_string()),
            Cell::new(e.category.to_string()),
            Cell::new(format!("{:.2}", e.confidence)),
            Cell::new(content_preview),
        ]);
    }

    println!("{table}");
}
