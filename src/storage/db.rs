use std::path::Path;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use chrono::Utc;
use turso::{Builder, Connection, Row, Value, params_from_iter};
use uuid::Uuid;

use super::models::{Category, Lifecycle, Memory, Scope};
use super::vector::{blob_to_embedding, cosine_similarity, embedding_to_blob, micros_to_dt};

/// Column list in a fixed order, shared by SELECT and decode (`row_to_memory`).
const COLS: &str = "id, content, embedding, embedding_dim, scope, category, entity, \
     source_project, source_session, source_timestamp, lifecycle, confidence, \
     created_at, updated_at, accessed_at, supersedes, superseded_by";

const INSERT_SQL: &str = "INSERT INTO memories (\
     id, content, embedding, embedding_dim, scope, category, entity, \
     source_project, source_session, source_timestamp, lifecycle, confidence, \
     created_at, updated_at, accessed_at, supersedes, superseded_by) \
     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17)";

const DDL: &str = "
CREATE TABLE IF NOT EXISTS memories (
    id               TEXT    PRIMARY KEY,
    content          TEXT    NOT NULL,
    embedding        BLOB    NOT NULL,
    embedding_dim    INTEGER NOT NULL,
    scope            TEXT    NOT NULL,
    category         TEXT    NOT NULL,
    entity           TEXT,
    source_project   TEXT,
    source_session   TEXT    NOT NULL,
    source_timestamp INTEGER NOT NULL,
    lifecycle        TEXT    NOT NULL,
    confidence       REAL    NOT NULL,
    created_at       INTEGER NOT NULL,
    updated_at       INTEGER NOT NULL,
    accessed_at      INTEGER,
    supersedes       TEXT,
    superseded_by    TEXT
);
CREATE INDEX IF NOT EXISTS idx_mem_lifecycle     ON memories(lifecycle);
CREATE INDEX IF NOT EXISTS idx_mem_scope         ON memories(scope);
CREATE INDEX IF NOT EXISTS idx_mem_entity        ON memories(entity);
CREATE INDEX IF NOT EXISTS idx_mem_superseded_by ON memories(superseded_by);
";

pub struct MemoryDb {
    db: turso::Database,
}

impl MemoryDb {
    /// Open (or create) the Turso store at `db_path` (a file) and run the DDL.
    pub async fn open(db_path: &Path) -> Result<Self> {
        if let Some(parent) = db_path.parent() {
            std::fs::create_dir_all(parent).with_context(|| {
                format!("failed to create db directory: {}", parent.display())
            })?;
        }
        let path = db_path
            .to_str()
            .with_context(|| format!("db path is not valid UTF-8: {}", db_path.display()))?;
        let db = Builder::new_local(path)
            .build()
            .await
            .context("failed to open Turso database")?;

        let this = Self { db };
        let conn = this.conn().await?;
        conn.execute_batch(DDL)
            .await
            .context("failed to run schema DDL")?;
        Ok(this)
    }

    /// Obtain a fresh connection with the busy timeout configured. Connections
    /// are cheap; we take one per operation so transactions (which need
    /// `&mut Connection`) compose with the `&self` public methods.
    async fn conn(&self) -> Result<Connection> {
        let conn = self.db.connect().context("failed to open Turso connection")?;
        conn.busy_timeout(Duration::from_secs(5))
            .context("failed to set busy_timeout")?;
        Ok(conn)
    }

    pub async fn insert(&self, memories: Vec<Memory>) -> Result<()> {
        if memories.is_empty() {
            return Ok(());
        }
        let mut conn = self.conn().await?;

        // Dimension consistency: all rows in the batch must agree, and must
        // match the dimension already in the store (if any). A silent mismatch
        // would make these rows unfindable (cosine returns 0.0 on length
        // mismatch), so reject it explicitly.
        let incoming_dim = memories[0].embedding.len();
        for m in &memories {
            if m.embedding.len() != incoming_dim {
                bail!(
                    "inconsistent embedding dimensions within insert batch: {} vs {}",
                    m.embedding.len(),
                    incoming_dim
                );
            }
        }
        if let Some(existing) = stored_dim(&conn).await?
            && existing != incoming_dim
        {
            bail!(
                "embedding dimension mismatch: store uses {existing}, incoming batch uses {incoming_dim}"
            );
        }

        let tx = conn.transaction().await?;
        for m in &memories {
            tx.execute(INSERT_SQL, params_from_iter(memory_to_values(m)))
                .await
                .context("failed to insert memory")?;
        }
        tx.commit().await.context("failed to commit insert")?;
        Ok(())
    }

    pub async fn list(&self, filters: &ListFilters) -> Result<Vec<Memory>> {
        let conn = self.conn().await?;

        let mut conditions: Vec<String> = Vec::new();
        let mut params: Vec<Value> = Vec::new();
        let mut idx = 1;
        if let Some(scope) = &filters.scope {
            conditions.push(format!("scope = ?{idx}"));
            params.push(Value::from(scope.to_string()));
            idx += 1;
        }
        if let Some(category) = &filters.category {
            conditions.push(format!("category = ?{idx}"));
            params.push(Value::from(category.to_string()));
            idx += 1;
        }
        if let Some(lifecycle) = &filters.lifecycle {
            conditions.push(format!("lifecycle = ?{idx}"));
            params.push(Value::from(lifecycle.to_string()));
            idx += 1;
        }
        if let Some(entity) = &filters.entity {
            conditions.push(format!("entity = ?{idx}"));
            params.push(Value::from(entity.clone()));
            idx += 1;
        }
        if let Some(source_project) = &filters.source_project {
            conditions.push(format!("source_project = ?{idx}"));
            params.push(Value::from(source_project.clone()));
            idx += 1;
        }
        if filters.exclude_superseded {
            conditions.push("superseded_by IS NULL".to_string());
        }

        let mut sql = format!("SELECT {COLS} FROM memories");
        if !conditions.is_empty() {
            sql.push_str(" WHERE ");
            sql.push_str(&conditions.join(" AND "));
        }
        if let Some(limit) = filters.limit {
            sql.push_str(&format!(" LIMIT ?{idx}"));
            params.push(Value::from(limit as i64));
        }

        let mut rows = conn
            .query(&sql, params_from_iter(params))
            .await
            .context("failed to query memories")?;
        collect_memories(&mut rows).await
    }

    pub async fn get_by_id(&self, id: &Uuid) -> Result<Option<Memory>> {
        let conn = self.conn().await?;
        let mut rows = conn
            .query(
                &format!("SELECT {COLS} FROM memories WHERE id = ?1"),
                params_from_iter(vec![Value::from(id.to_string())]),
            )
            .await?;
        match rows.next().await? {
            Some(row) => Ok(Some(row_to_memory(&row)?)),
            None => Ok(None),
        }
    }

    /// Vector similarity search: exact cosine over the accepted candidate set,
    /// ranked in-app. (Turso has no native vector type; at kenna's scale a full
    /// scan of accepted memories is sub-millisecond and exactness matters for
    /// the dedup path.)
    pub async fn vector_search(
        &self,
        query_embedding: Vec<f32>,
        limit: usize,
        scope_filter: Option<Scope>,
    ) -> Result<Vec<Memory>> {
        let conn = self.conn().await?;

        let mut sql = format!("SELECT {COLS} FROM memories WHERE lifecycle = ?1");
        let mut params = vec![Value::from(Lifecycle::Accepted.to_string())];
        if let Some(scope) = scope_filter {
            sql.push_str(" AND scope = ?2");
            params.push(Value::from(scope.to_string()));
        }

        let mut rows = conn
            .query(&sql, params_from_iter(params))
            .await
            .context("vector search candidate query failed")?;
        let candidates = collect_memories(&mut rows).await?;

        let mut scored: Vec<(f32, Memory)> = candidates
            .into_iter()
            .map(|m| (cosine_similarity(&query_embedding, &m.embedding), m))
            .collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        Ok(scored.into_iter().take(limit).map(|(_, m)| m).collect())
    }

    /// Find the most-similar accepted memory eligible to *dedup* a new memory of
    /// the given scope/project, returning it with its cosine score (or `None`
    /// if there are no eligible candidates).
    ///
    /// Scoping is deliberately narrow so the settling pass keeps the signal it
    /// needs: a project candidate is compared only against memories from the
    /// **same `source_project`** or ones already promoted to **personal** scope;
    /// a personal candidate is compared only against personal memories.
    /// Cross-project near-duplicates are intentionally *not* matched here — they
    /// are left to coexist as separate rows so `settle`'s promotion pass can
    /// count distinct projects. See `dedup-scoping-issue.md`.
    ///
    /// Unlike `vector_search` (recall, accepted-only), dedup considers **both**
    /// lifecycles — a pending `Candidate` is still "already in the store", and
    /// excluding it lets re-extraction pile up duplicate Candidate rows in the
    /// review queue. It also restricts to **live** rows (`superseded_by IS NULL`):
    /// a retired row must not be able to veto recording a genuine change/reversal.
    pub async fn find_dedup_match(
        &self,
        query_embedding: &[f32],
        scope: Scope,
        source_project: Option<&str>,
    ) -> Result<Option<(f32, Memory)>> {
        let conn = self.conn().await?;

        let mut sql = format!("SELECT {COLS} FROM memories WHERE superseded_by IS NULL");
        let mut params: Vec<Value> = Vec::new();
        match (scope, source_project) {
            // Project candidate with a known project: same project OR already-personal.
            (Scope::Project, Some(project)) => {
                sql.push_str(" AND (source_project = ?1 OR scope = ?2)");
                params.push(Value::from(project.to_string()));
                params.push(Value::from(Scope::Personal.to_string()));
            }
            // Personal candidate (or a project candidate with no project info):
            // dedup only against personal, never against another project.
            _ => {
                sql.push_str(" AND scope = ?1");
                params.push(Value::from(Scope::Personal.to_string()));
            }
        }

        let mut rows = conn
            .query(&sql, params_from_iter(params))
            .await
            .context("dedup candidate query failed")?;
        let candidates = collect_memories(&mut rows).await?;

        Ok(candidates
            .into_iter()
            .map(|m| (cosine_similarity(query_embedding, &m.embedding), m))
            .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)))
    }

    pub async fn delete(&self, id: &Uuid) -> Result<()> {
        let conn = self.conn().await?;
        conn.execute(
            "DELETE FROM memories WHERE id = ?1",
            params_from_iter(vec![Value::from(id.to_string())]),
        )
        .await?;
        Ok(())
    }

    pub async fn update_lifecycle(&self, id: &Uuid, lifecycle: Lifecycle) -> Result<()> {
        let conn = self.conn().await?;
        let n = conn
            .execute(
                "UPDATE memories SET lifecycle = ?1, updated_at = ?2 WHERE id = ?3",
                params_from_iter(vec![
                    Value::from(lifecycle.to_string()),
                    Value::from(Utc::now().timestamp_micros()),
                    Value::from(id.to_string()),
                ]),
            )
            .await?;
        if n == 0 {
            bail!("memory not found: {id}");
        }
        Ok(())
    }

    pub async fn update_scope(&self, id: &Uuid, scope: Scope) -> Result<()> {
        let conn = self.conn().await?;
        let n = conn
            .execute(
                "UPDATE memories SET scope = ?1, updated_at = ?2 WHERE id = ?3",
                params_from_iter(vec![
                    Value::from(scope.to_string()),
                    Value::from(Utc::now().timestamp_micros()),
                    Value::from(id.to_string()),
                ]),
            )
            .await?;
        if n == 0 {
            bail!("memory not found: {id}");
        }
        Ok(())
    }

    /// Atomically persist a settlement result: insert the new synthesized /
    /// promoted memory AND mark all source memories superseded, in one
    /// transaction. Either everything lands or nothing does.
    pub async fn apply_settlement(
        &self,
        new_memory: Memory,
        superseded_ids: &[Uuid],
    ) -> Result<()> {
        let mut conn = self.conn().await?;

        if let Some(existing) = stored_dim(&conn).await?
            && existing != new_memory.embedding.len()
        {
            bail!(
                "embedding dimension mismatch: store uses {existing}, new memory uses {}",
                new_memory.embedding.len()
            );
        }

        let now = Utc::now().timestamp_micros();
        let new_id = new_memory.id.to_string();

        let tx = conn.transaction().await?;
        tx.execute(INSERT_SQL, params_from_iter(memory_to_values(&new_memory)))
            .await
            .context("failed to insert settled memory")?;
        for sid in superseded_ids {
            tx.execute(
                "UPDATE memories SET superseded_by = ?1, updated_at = ?2 WHERE id = ?3",
                params_from_iter(vec![
                    Value::from(new_id.clone()),
                    Value::from(now),
                    Value::from(sid.to_string()),
                ]),
            )
            .await
            .context("failed to mark source memory superseded")?;
        }
        tx.commit().await.context("failed to commit settlement")?;
        Ok(())
    }

    pub async fn count(&self) -> Result<MemoryStats> {
        let conn = self.conn().await?;
        let mut stats = MemoryStats::default();

        let mut rows = conn.query("SELECT COUNT(*) FROM memories", ()).await?;
        if let Some(row) = rows.next().await? {
            stats.total = as_int(row.get_value(0)?, "count")? as usize;
        }

        for (scope, n) in group_counts(&conn, "SELECT scope, COUNT(*) FROM memories GROUP BY scope")
            .await?
        {
            match scope.parse::<Scope>() {
                Ok(Scope::Personal) => stats.personal = n as usize,
                Ok(Scope::Project) => stats.project = n as usize,
                Err(_) => {}
            }
        }

        for (lifecycle, n) in
            group_counts(&conn, "SELECT lifecycle, COUNT(*) FROM memories GROUP BY lifecycle")
                .await?
        {
            match lifecycle.parse::<Lifecycle>() {
                Ok(Lifecycle::Candidate) => stats.candidates = n as usize,
                Ok(Lifecycle::Accepted) => stats.accepted = n as usize,
                Err(_) => {}
            }
        }

        for (category, n) in
            group_counts(&conn, "SELECT category, COUNT(*) FROM memories GROUP BY category").await?
        {
            stats.by_category.insert(category, n as usize);
        }

        Ok(stats)
    }

    pub async fn export_all(&self) -> Result<Vec<Memory>> {
        self.list(&ListFilters::default()).await
    }

    /// Case-insensitive substring search over `content` and `entity`, pushed
    /// into SQL via a parameterized `LIKE` (no full-table load).
    pub async fn keyword_search(&self, query: &str, limit: usize) -> Result<Vec<Memory>> {
        let conn = self.conn().await?;
        let pattern = format!("%{}%", escape_like(query));
        let mut rows = conn
            .query(
                &format!(
                    "SELECT {COLS} FROM memories \
                     WHERE content LIKE ?1 ESCAPE '\\' OR entity LIKE ?1 ESCAPE '\\' \
                     LIMIT ?2"
                ),
                params_from_iter(vec![Value::from(pattern), Value::from(limit as i64)]),
            )
            .await
            .context("keyword search failed")?;
        collect_memories(&mut rows).await
    }
}

/// Escape `%`, `_`, and `\` so user input is treated as a literal substring
/// inside a `LIKE ... ESCAPE '\'` pattern.
fn escape_like(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        if matches!(c, '%' | '_' | '\\') {
            out.push('\\');
        }
        out.push(c);
    }
    out
}

#[derive(Debug, Default)]
pub struct ListFilters {
    pub scope: Option<Scope>,
    pub category: Option<Category>,
    pub lifecycle: Option<Lifecycle>,
    pub entity: Option<String>,
    pub source_project: Option<String>,
    /// `None` means no limit.
    pub limit: Option<usize>,
    /// If true, exclude memories that have been superseded (superseded_by IS NULL).
    pub exclude_superseded: bool,
}

#[derive(Debug, Default)]
pub struct MemoryStats {
    pub total: usize,
    pub personal: usize,
    pub project: usize,
    pub candidates: usize,
    pub accepted: usize,
    pub by_category: std::collections::HashMap<String, usize>,
}

// ── Row encode / decode ──

fn memory_to_values(m: &Memory) -> Vec<Value> {
    vec![
        Value::from(m.id.to_string()),
        Value::from(m.content.clone()),
        Value::from(embedding_to_blob(&m.embedding)),
        Value::from(m.embedding.len() as i64),
        Value::from(m.scope.to_string()),
        Value::from(m.category.to_string()),
        Value::from(m.entity.clone()),
        Value::from(m.source_project.clone()),
        Value::from(m.source_session.clone()),
        Value::from(m.source_timestamp.timestamp_micros()),
        Value::from(m.lifecycle.to_string()),
        Value::from(m.confidence),
        Value::from(m.created_at.timestamp_micros()),
        Value::from(m.updated_at.timestamp_micros()),
        Value::from(m.accessed_at.map(|d| d.timestamp_micros())),
        Value::from(m.supersedes.map(|u| u.to_string())),
        Value::from(m.superseded_by.map(|u| u.to_string())),
    ]
}

async fn collect_memories(rows: &mut turso::Rows) -> Result<Vec<Memory>> {
    let mut out = Vec::new();
    while let Some(row) = rows.next().await? {
        out.push(row_to_memory(&row)?);
    }
    Ok(out)
}

fn row_to_memory(row: &Row) -> Result<Memory> {
    let id = as_text(row.get_value(0)?, "id")?;
    let content = as_text(row.get_value(1)?, "content")?;
    let embedding = blob_to_embedding(&as_blob(row.get_value(2)?, "embedding")?)?;
    // column 3 (embedding_dim) is redundant with the decoded length; skipped.
    let scope = as_text(row.get_value(4)?, "scope")?;
    let category = as_text(row.get_value(5)?, "category")?;
    let entity = as_opt_text(row.get_value(6)?, "entity")?;
    let source_project = as_opt_text(row.get_value(7)?, "source_project")?;
    let source_session = as_text(row.get_value(8)?, "source_session")?;
    let source_timestamp = micros_to_dt(as_int(row.get_value(9)?, "source_timestamp")?)?;
    let lifecycle = as_text(row.get_value(10)?, "lifecycle")?;
    let confidence = as_real(row.get_value(11)?, "confidence")? as f32;
    let created_at = micros_to_dt(as_int(row.get_value(12)?, "created_at")?)?;
    let updated_at = micros_to_dt(as_int(row.get_value(13)?, "updated_at")?)?;
    let accessed_at = match as_opt_int(row.get_value(14)?, "accessed_at")? {
        Some(m) => Some(micros_to_dt(m)?),
        None => None,
    };
    let supersedes = as_opt_text(row.get_value(15)?, "supersedes")?;
    let superseded_by = as_opt_text(row.get_value(16)?, "superseded_by")?;

    Ok(Memory {
        id: id.parse().with_context(|| format!("invalid uuid in id: {id}"))?,
        content,
        embedding,
        scope: scope.parse()?,
        category: category.parse()?,
        entity,
        source_project,
        source_session,
        source_timestamp,
        lifecycle: lifecycle.parse()?,
        confidence,
        created_at,
        updated_at,
        accessed_at,
        supersedes: match supersedes {
            Some(s) => Some(s.parse().with_context(|| format!("invalid uuid in supersedes: {s}"))?),
            None => None,
        },
        superseded_by: match superseded_by {
            Some(s) => {
                Some(s.parse().with_context(|| format!("invalid uuid in superseded_by: {s}"))?)
            }
            None => None,
        },
    })
}

async fn stored_dim(conn: &Connection) -> Result<Option<usize>> {
    let mut rows = conn
        .query("SELECT embedding_dim FROM memories LIMIT 1", ())
        .await?;
    match rows.next().await? {
        Some(row) => Ok(Some(as_int(row.get_value(0)?, "embedding_dim")? as usize)),
        None => Ok(None),
    }
}

async fn group_counts(conn: &Connection, sql: &str) -> Result<Vec<(String, i64)>> {
    let mut rows = conn.query(sql, ()).await?;
    let mut out = Vec::new();
    while let Some(row) = rows.next().await? {
        let key = as_text(row.get_value(0)?, "group key")?;
        let count = as_int(row.get_value(1)?, "count")?;
        out.push((key, count));
    }
    Ok(out)
}

fn as_text(v: Value, col: &str) -> Result<String> {
    match v {
        Value::Text(s) => Ok(s),
        other => bail!("column {col}: expected TEXT, got {other:?}"),
    }
}

fn as_opt_text(v: Value, col: &str) -> Result<Option<String>> {
    match v {
        Value::Null => Ok(None),
        Value::Text(s) => Ok(Some(s)),
        other => bail!("column {col}: expected TEXT or NULL, got {other:?}"),
    }
}

fn as_int(v: Value, col: &str) -> Result<i64> {
    match v {
        Value::Integer(i) => Ok(i),
        other => bail!("column {col}: expected INTEGER, got {other:?}"),
    }
}

fn as_opt_int(v: Value, col: &str) -> Result<Option<i64>> {
    match v {
        Value::Null => Ok(None),
        Value::Integer(i) => Ok(Some(i)),
        other => bail!("column {col}: expected INTEGER or NULL, got {other:?}"),
    }
}

fn as_real(v: Value, col: &str) -> Result<f64> {
    match v {
        Value::Real(f) => Ok(f),
        Value::Integer(i) => Ok(i as f64),
        other => bail!("column {col}: expected REAL, got {other:?}"),
    }
}

fn as_blob(v: Value, col: &str) -> Result<Vec<u8>> {
    match v {
        Value::Blob(b) => Ok(b),
        other => bail!("column {col}: expected BLOB, got {other:?}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn temp_db() -> (MemoryDb, std::path::PathBuf) {
        let path = std::env::temp_dir().join(format!("kenna-test-{}.db", Uuid::new_v4()));
        let db = MemoryDb::open(&path).await.unwrap();
        (db, path)
    }

    fn cleanup(path: &std::path::Path) {
        let _ = std::fs::remove_file(path);
        let _ = std::fs::remove_file(format!("{}-wal", path.display()));
        let _ = std::fs::remove_file(format!("{}-shm", path.display()));
    }

    fn mk_memory(content: &str, embedding: Vec<f32>, lifecycle: Lifecycle) -> Memory {
        let now = Utc::now();
        Memory {
            id: Uuid::new_v4(),
            content: content.to_string(),
            embedding,
            scope: Scope::Personal,
            category: Category::Fact,
            entity: None,
            source_project: None,
            source_session: "test".to_string(),
            source_timestamp: now,
            lifecycle,
            confidence: 0.9,
            created_at: now,
            updated_at: now,
            accessed_at: None,
            supersedes: None,
            superseded_by: None,
        }
    }

    #[tokio::test]
    async fn insert_round_trips_all_fields() {
        let (db, path) = temp_db().await;
        let mut m = mk_memory("hello", vec![1.0, 2.0, 3.0], Lifecycle::Accepted);
        m.entity = Some("rust".to_string());
        m.source_project = Some("proj".to_string());
        m.confidence = 0.42;
        let id = m.id;
        db.insert(vec![m.clone()]).await.unwrap();

        let got = db.get_by_id(&id).await.unwrap().unwrap();
        assert_eq!(got.content, "hello");
        assert_eq!(got.embedding, vec![1.0, 2.0, 3.0]);
        assert_eq!(got.entity.as_deref(), Some("rust"));
        assert_eq!(got.source_project.as_deref(), Some("proj"));
        assert!((got.confidence - 0.42).abs() < 1e-6);
        assert_eq!(got.lifecycle, Lifecycle::Accepted);
        cleanup(&path);
    }

    #[tokio::test]
    async fn insert_dim_mismatch_errors() {
        let (db, path) = temp_db().await;
        db.insert(vec![mk_memory("a", vec![1.0, 0.0, 0.0, 0.0], Lifecycle::Accepted)])
            .await
            .unwrap();
        // A second insert with a different dimension must be rejected.
        let res = db
            .insert(vec![mk_memory("b", vec![1.0, 0.0, 0.0], Lifecycle::Accepted)])
            .await;
        assert!(res.is_err(), "dim mismatch should error");
        cleanup(&path);
    }

    #[tokio::test]
    async fn vector_search_matches_bruteforce_and_excludes_candidates() {
        let (db, path) = temp_db().await;
        let mems = vec![
            mk_memory("a", vec![1.0, 0.0, 0.0, 0.0], Lifecycle::Accepted),
            mk_memory("b", vec![0.0, 1.0, 0.0, 0.0], Lifecycle::Accepted),
            mk_memory("c", vec![0.9, 0.1, 0.0, 0.0], Lifecycle::Accepted),
            // Candidate must be excluded from vector_search.
            mk_memory("d", vec![1.0, 0.0, 0.0, 0.0], Lifecycle::Candidate),
        ];
        db.insert(mems.clone()).await.unwrap();

        let query = vec![1.0, 0.0, 0.0, 0.0];
        let results = db.vector_search(query.clone(), 3, None).await.unwrap();

        // Brute-force reference over the accepted set only.
        let mut expected: Vec<(f32, String)> = mems
            .iter()
            .filter(|m| m.lifecycle == Lifecycle::Accepted)
            .map(|m| (cosine_similarity(&query, &m.embedding), m.content.clone()))
            .collect();
        expected.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        let expected_order: Vec<String> =
            expected.into_iter().map(|(_, c)| c).take(3).collect();

        let got_order: Vec<String> = results.iter().map(|m| m.content.clone()).collect();
        assert_eq!(got_order, expected_order);
        assert!(!got_order.contains(&"d".to_string()));
        cleanup(&path);
    }

    #[tokio::test]
    async fn apply_settlement_supersedes_all_sources() {
        let (db, path) = temp_db().await;
        let s1 = mk_memory("src1", vec![1.0, 0.0], Lifecycle::Accepted);
        let s2 = mk_memory("src2", vec![0.0, 1.0], Lifecycle::Accepted);
        db.insert(vec![s1.clone(), s2.clone()]).await.unwrap();

        let new = mk_memory("synth", vec![0.5, 0.5], Lifecycle::Candidate);
        let new_id = new.id;
        db.apply_settlement(new, &[s1.id, s2.id]).await.unwrap();

        assert!(db.get_by_id(&new_id).await.unwrap().is_some());
        assert_eq!(
            db.get_by_id(&s1.id).await.unwrap().unwrap().superseded_by,
            Some(new_id)
        );
        assert_eq!(
            db.get_by_id(&s2.id).await.unwrap().unwrap().superseded_by,
            Some(new_id)
        );
        cleanup(&path);
    }

    #[tokio::test]
    async fn apply_settlement_empty_sources_is_plain_insert() {
        // Finding 1: apply_settlement with no superseded_ids degenerates to a
        // transactional single-row insert, touching nothing else. This is what
        // lets reconcile use it unconditionally (no branch on supersedes.is_empty()).
        let (db, path) = temp_db().await;
        let existing = mk_memory("untouched", vec![1.0, 0.0], Lifecycle::Accepted);
        let existing_id = existing.id;
        db.insert(vec![existing]).await.unwrap();

        let new = mk_memory("fresh", vec![0.0, 1.0], Lifecycle::Candidate);
        let new_id = new.id;
        db.apply_settlement(new, &[]).await.unwrap();

        // The new row landed...
        assert!(db.get_by_id(&new_id).await.unwrap().is_some());
        // ...and the pre-existing row is untouched (no stray supersession).
        let after = db.get_by_id(&existing_id).await.unwrap().unwrap();
        assert_eq!(after.superseded_by, None);
        assert_eq!(after.content, "untouched");
        cleanup(&path);
    }

    #[tokio::test]
    async fn apply_settlement_rolls_back_on_failure() {
        let (db, path) = temp_db().await;
        let s1 = mk_memory("src1", vec![1.0, 0.0], Lifecycle::Accepted);
        db.insert(vec![s1.clone()]).await.unwrap();

        // Force a mid-transaction failure: the new memory reuses an existing
        // primary key, so its INSERT inside the settlement transaction fails.
        let mut dup = mk_memory("dup", vec![0.5, 0.5], Lifecycle::Candidate);
        dup.id = s1.id;
        let res = db.apply_settlement(dup, &[s1.id]).await;
        assert!(res.is_err(), "settlement should fail on PK conflict");

        // The store must be unchanged: s1 is not superseded and its content is intact.
        let after = db.get_by_id(&s1.id).await.unwrap().unwrap();
        assert_eq!(after.superseded_by, None);
        assert_eq!(after.content, "src1");
        cleanup(&path);
    }

    fn mk_project_memory(content: &str, embedding: Vec<f32>, project: &str) -> Memory {
        let mut m = mk_memory(content, embedding, Lifecycle::Accepted);
        m.scope = Scope::Project;
        m.source_project = Some(project.to_string());
        m
    }

    #[tokio::test]
    async fn dedup_match_is_project_scoped() {
        let (db, path) = temp_db().await;
        let e = vec![1.0, 0.0, 0.0, 0.0];
        db.insert(vec![mk_project_memory("alpha mem", e.clone(), "alpha")])
            .await
            .unwrap();

        // An identical embedding from a *different* project is not a dedup match
        // — it must coexist so the settling pass can count distinct projects.
        let cross = db
            .find_dedup_match(&e, Scope::Project, Some("beta"))
            .await
            .unwrap();
        assert!(cross.is_none(), "cross-project near-dup must not match");

        // The same project, however, is a match.
        let (score, m) = db
            .find_dedup_match(&e, Scope::Project, Some("alpha"))
            .await
            .unwrap()
            .expect("same-project dup should match");
        assert!(score > 0.99);
        assert_eq!(m.content, "alpha mem");
        cleanup(&path);
    }

    #[tokio::test]
    async fn dedup_project_candidate_matches_personal() {
        let (db, path) = temp_db().await;
        let e = vec![1.0, 0.0, 0.0, 0.0];
        // An already-promoted personal memory.
        db.insert(vec![mk_memory("personal trait", e.clone(), Lifecycle::Accepted)])
            .await
            .unwrap();

        // A project candidate matching it dedups against the personal row, so we
        // don't re-hoard a trait that has already been promoted.
        let (score, got) = db
            .find_dedup_match(&e, Scope::Project, Some("beta"))
            .await
            .unwrap()
            .expect("project candidate should match personal");
        assert!(score > 0.99);
        assert_eq!(got.content, "personal trait");
        cleanup(&path);
    }

    #[tokio::test]
    async fn dedup_matches_candidate_lifecycle_rows() {
        // Finding 2: dedup must see pending Candidate rows too, otherwise
        // re-extraction of a below-auto-accept fact piles up duplicate rows.
        let (db, path) = temp_db().await;
        let e = vec![1.0, 0.0, 0.0, 0.0];
        let mut c = mk_project_memory("pending alpha mem", e.clone(), "alpha");
        c.lifecycle = Lifecycle::Candidate;
        db.insert(vec![c]).await.unwrap();

        let (score, got) = db
            .find_dedup_match(&e, Scope::Project, Some("alpha"))
            .await
            .unwrap()
            .expect("candidate-lifecycle row should be a dedup match");
        assert!(score > 0.99);
        assert_eq!(got.content, "pending alpha mem");
        cleanup(&path);
    }

    #[tokio::test]
    async fn dedup_ignores_superseded_rows() {
        // Finding 3: a retired (superseded) row must not veto a new fact. Set up
        // A→B via apply_settlement (the production supersession path), then a
        // candidate whose embedding ≈ A's must never match the dead A row.
        let (db, path) = temp_db().await;
        let a_emb = vec![1.0, 0.0, 0.0, 0.0];
        let b_emb = vec![0.0, 1.0, 0.0, 0.0];
        let a = mk_memory("prefers vim", a_emb.clone(), Lifecycle::Accepted);
        let a_id = a.id;
        db.insert(vec![a]).await.unwrap();

        // B supersedes A, atomically (A.superseded_by = B, both live in store).
        let b = mk_memory("switched to neovim", b_emb, Lifecycle::Accepted);
        let b_id = b.id;
        db.apply_settlement(b, &[a_id]).await.unwrap();

        // A candidate matching A exactly: the best (only) live match is B, never A.
        let matched_id = db
            .find_dedup_match(&a_emb, Scope::Personal, None)
            .await
            .unwrap()
            .map(|(_, m)| m.id);
        // A is definitively never returned; the only live match is B (or None if
        // B is below any threshold the caller applies).
        assert_ne!(matched_id, Some(a_id), "superseded row A must not be a dedup match");
        assert!(
            matched_id == Some(b_id) || matched_id.is_none(),
            "the only eligible live match is B"
        );
        cleanup(&path);
    }

    #[tokio::test]
    async fn list_filters_by_source_project() {
        let (db, path) = temp_db().await;
        let e = vec![1.0, 0.0];
        let mut a = mk_project_memory("a", e.clone(), "alpha");
        a.entity = Some("editor".to_string());
        let mut b = mk_project_memory("b", e.clone(), "beta");
        b.entity = Some("editor".to_string());
        db.insert(vec![a, b]).await.unwrap();

        let got = db
            .list(&ListFilters {
                entity: Some("editor".to_string()),
                source_project: Some("alpha".to_string()),
                ..Default::default()
            })
            .await
            .unwrap();
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].source_project.as_deref(), Some("alpha"));
        cleanup(&path);
    }
}
