use anyhow::Result;
use comfy_table::{Cell, Table};

use crate::storage::db::{EngramDb, ListFilters};
use crate::storage::models::{Category, Lifecycle, Scope};

pub async fn run(
    db: &EngramDb,
    scope: Option<Scope>,
    category: Option<Category>,
    lifecycle: Option<Lifecycle>,
    entity: Option<String>,
    limit: usize,
) -> Result<()> {
    let filters = ListFilters {
        scope,
        category,
        lifecycle,
        entity,
        limit: Some(limit),
        exclude_superseded: false, // we filter superseded in Rust below
    };

    let all_engrams = db.list(&filters).await?;

    // Hide superseded engrams by default — they've been replaced
    let engrams: Vec<_> = all_engrams
        .into_iter()
        .filter(|e| e.superseded_by.is_none())
        .collect();

    if engrams.is_empty() {
        println!("No engrams found.");
        return Ok(());
    }

    let mut table = Table::new();
    table.set_header(vec!["ID", "Scope", "Category", "Lifecycle", "Conf", "Content"]);

    for e in &engrams {
        let short_id = &e.id.to_string()[..8];
        let content_preview = if e.content.len() > 60 {
            format!("{}…", &e.content[..60])
        } else {
            e.content.clone()
        };
        table.add_row(vec![
            Cell::new(short_id),
            Cell::new(e.scope.to_string()),
            Cell::new(e.category.to_string()),
            Cell::new(e.lifecycle.to_string()),
            Cell::new(format!("{:.2}", e.confidence)),
            Cell::new(content_preview),
        ]);
    }

    println!("{table}");
    println!("\n{} engram(s) shown.", engrams.len());
    Ok(())
}
