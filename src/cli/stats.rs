use anyhow::Result;

use crate::storage::db::MemoryDb;

pub async fn run(db: &MemoryDb) -> Result<()> {
    let stats = db.count().await?;

    println!("Memory Statistics");
    println!("─────────────────");
    println!("Total:      {}", stats.total);
    println!("Personal:   {}", stats.personal);
    println!("Project:    {}", stats.project);
    println!("Accepted:   {}", stats.accepted);
    println!("Candidates: {}", stats.candidates);
    println!();
    println!("By category:");
    let mut cats: Vec<_> = stats.by_category.into_iter().collect();
    cats.sort_by(|a, b| b.1.cmp(&a.1));
    for (cat, count) in cats {
        println!("  {cat:<12} {count}");
    }

    Ok(())
}
