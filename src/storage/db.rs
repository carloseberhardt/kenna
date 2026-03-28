use std::path::Path;

use anyhow::{Context, Result, bail};
use arrow_array::RecordBatch;
use chrono::{DateTime, TimeZone, Utc};
use lancedb::Connection;
use lancedb::query::{ExecutableQuery, QueryBase};
use uuid::Uuid;

use super::models::{
    Category, Engram, Lifecycle, Scope, arrow_schema,
};

const TABLE_NAME: &str = "engrams";

pub struct EngramDb {
    conn: Connection,
}

impl EngramDb {
    pub async fn open(db_path: &Path) -> Result<Self> {
        std::fs::create_dir_all(db_path)?;
        let conn = lancedb::connect(db_path.to_str().unwrap())
            .execute()
            .await
            .context("failed to connect to LanceDB")?;
        Ok(Self { conn })
    }

    async fn ensure_table(&self) -> Result<lancedb::Table> {
        let names = self.conn.table_names().execute().await?;
        if names.contains(&TABLE_NAME.to_string()) {
            Ok(self.conn.open_table(TABLE_NAME).execute().await?)
        } else {
            let schema = arrow_schema();
            Ok(self
                .conn
                .create_empty_table(TABLE_NAME, schema)
                .execute()
                .await?)
        }
    }

    pub async fn insert(&self, engrams: Vec<Engram>) -> Result<()> {
        if engrams.is_empty() {
            return Ok(());
        }
        let table = self.ensure_table().await?;
        let reader = Engram::to_record_batch_reader(engrams);
        table.add(reader).execute().await?;
        Ok(())
    }

    pub async fn list(&self, filters: &ListFilters) -> Result<Vec<Engram>> {
        let table = match self.open_table_if_exists().await? {
            Some(t) => t,
            None => return Ok(vec![]),
        };

        let mut conditions = Vec::new();
        if let Some(scope) = &filters.scope {
            conditions.push(format!("scope = '{scope}'"));
        }
        if let Some(category) = &filters.category {
            conditions.push(format!("category = '{category}'"));
        }
        if let Some(lifecycle) = &filters.lifecycle {
            conditions.push(format!("lifecycle = '{lifecycle}'"));
        }
        if let Some(entity) = &filters.entity {
            conditions.push(format!("entity = '{entity}'"));
        }

        let mut query = table.query();
        if !conditions.is_empty() {
            let filter = conditions.join(" AND ");
            query = query.only_if(filter);
        }
        let limit = filters.limit.unwrap_or(20);
        let batches = query
            .limit(limit)
            .execute()
            .await
            .context("failed to query engrams")?
            .try_collect::<Vec<RecordBatch>>()
            .await?;

        batches_to_engrams(&batches)
    }

    pub async fn get_by_id(&self, id: &Uuid) -> Result<Option<Engram>> {
        let table = match self.open_table_if_exists().await? {
            Some(t) => t,
            None => return Ok(None),
        };

        let filter = format!("id = '{id}'");
        let batches = table
            .query()
            .only_if(filter)
            .limit(1)
            .execute()
            .await?
            .try_collect::<Vec<RecordBatch>>()
            .await?;

        let engrams = batches_to_engrams(&batches)?;
        Ok(engrams.into_iter().next())
    }

    /// Vector similarity search using an embedding query.
    pub async fn vector_search(
        &self,
        query_embedding: Vec<f32>,
        limit: usize,
        scope_filter: Option<Scope>,
    ) -> Result<Vec<Engram>> {
        let table = match self.open_table_if_exists().await? {
            Some(t) => t,
            None => return Ok(vec![]),
        };

        let mut search = table
            .vector_search(query_embedding)
            .context("failed to create vector search")?
            .column("embedding")
            .limit(limit);

        // Only return accepted engrams by default
        let mut filter_parts = vec!["lifecycle = 'accepted'".to_string()];
        if let Some(scope) = scope_filter {
            filter_parts.push(format!("scope = '{scope}'"));
        }
        search = search.only_if(filter_parts.join(" AND "));

        let batches = search
            .execute()
            .await
            .context("vector search failed")?
            .try_collect::<Vec<RecordBatch>>()
            .await?;

        batches_to_engrams(&batches)
    }

    pub async fn delete(&self, id: &Uuid) -> Result<()> {
        let table = match self.open_table_if_exists().await? {
            Some(t) => t,
            None => bail!("no engrams table found"),
        };
        let filter = format!("id = '{id}'");
        table.delete(&filter).await?;
        Ok(())
    }

    pub async fn update_lifecycle(&self, id: &Uuid, lifecycle: Lifecycle) -> Result<()> {
        // LanceDB doesn't have a direct update API — we read, delete, and re-insert.
        let engram = self
            .get_by_id(id)
            .await?
            .context("engram not found")?;

        let mut updated = engram;
        updated.lifecycle = lifecycle;
        updated.updated_at = Utc::now();

        self.delete(id).await?;
        self.insert(vec![updated]).await?;
        Ok(())
    }

    pub async fn update_scope(&self, id: &Uuid, scope: Scope) -> Result<()> {
        let engram = self
            .get_by_id(id)
            .await?
            .context("engram not found")?;

        let mut updated = engram;
        updated.scope = scope;
        updated.updated_at = Utc::now();

        self.delete(id).await?;
        self.insert(vec![updated]).await?;
        Ok(())
    }

    pub async fn count(&self) -> Result<EngramStats> {
        let all = self
            .list(&ListFilters {
                limit: Some(100_000),
                ..Default::default()
            })
            .await?;

        let mut stats = EngramStats::default();
        for e in &all {
            stats.total += 1;
            match e.scope {
                Scope::Personal => stats.personal += 1,
                Scope::Project => stats.project += 1,
            }
            match e.lifecycle {
                Lifecycle::Candidate => stats.candidates += 1,
                Lifecycle::Accepted => stats.accepted += 1,
            }
            let cat = e.category.to_string();
            *stats.by_category.entry(cat).or_insert(0) += 1;
        }
        Ok(stats)
    }

    pub async fn export_all(&self) -> Result<Vec<Engram>> {
        self.list(&ListFilters {
            limit: Some(100_000),
            ..Default::default()
        })
        .await
    }

    async fn open_table_if_exists(&self) -> Result<Option<lancedb::Table>> {
        let names = self.conn.table_names().execute().await?;
        if names.contains(&TABLE_NAME.to_string()) {
            Ok(Some(self.conn.open_table(TABLE_NAME).execute().await?))
        } else {
            Ok(None)
        }
    }
}

#[derive(Debug, Default)]
pub struct ListFilters {
    pub scope: Option<Scope>,
    pub category: Option<Category>,
    pub lifecycle: Option<Lifecycle>,
    pub entity: Option<String>,
    pub limit: Option<usize>,
}

#[derive(Debug, Default)]
pub struct EngramStats {
    pub total: usize,
    pub personal: usize,
    pub project: usize,
    pub candidates: usize,
    pub accepted: usize,
    pub by_category: std::collections::HashMap<String, usize>,
}

use futures::TryStreamExt;

fn batches_to_engrams(batches: &[RecordBatch]) -> Result<Vec<Engram>> {
    use arrow_array::{
        Array, Float32Array, StringArray, TimestampMicrosecondArray,
    };

    let mut engrams = Vec::new();

    for batch in batches {
        let ids = batch.column_by_name("id").unwrap().as_any().downcast_ref::<StringArray>().unwrap();
        let contents = batch.column_by_name("content").unwrap().as_any().downcast_ref::<StringArray>().unwrap();
        let scopes = batch.column_by_name("scope").unwrap().as_any().downcast_ref::<StringArray>().unwrap();
        let categories = batch.column_by_name("category").unwrap().as_any().downcast_ref::<StringArray>().unwrap();
        let entities = batch.column_by_name("entity").unwrap().as_any().downcast_ref::<StringArray>().unwrap();
        let source_projects = batch.column_by_name("source_project").unwrap().as_any().downcast_ref::<StringArray>().unwrap();
        let source_sessions = batch.column_by_name("source_session").unwrap().as_any().downcast_ref::<StringArray>().unwrap();
        let source_timestamps = batch.column_by_name("source_timestamp").unwrap().as_any().downcast_ref::<TimestampMicrosecondArray>().unwrap();
        let lifecycles = batch.column_by_name("lifecycle").unwrap().as_any().downcast_ref::<StringArray>().unwrap();
        let confidences = batch.column_by_name("confidence").unwrap().as_any().downcast_ref::<Float32Array>().unwrap();
        let created_ats = batch.column_by_name("created_at").unwrap().as_any().downcast_ref::<TimestampMicrosecondArray>().unwrap();
        let updated_ats = batch.column_by_name("updated_at").unwrap().as_any().downcast_ref::<TimestampMicrosecondArray>().unwrap();
        let accessed_ats = batch.column_by_name("accessed_at").unwrap().as_any().downcast_ref::<TimestampMicrosecondArray>().unwrap();
        let supersedes_col = batch.column_by_name("supersedes").unwrap().as_any().downcast_ref::<StringArray>().unwrap();
        let superseded_by_col = batch.column_by_name("superseded_by").unwrap().as_any().downcast_ref::<StringArray>().unwrap();

        // Embedding column
        let embedding_col = batch
            .column_by_name("embedding")
            .unwrap()
            .as_any()
            .downcast_ref::<arrow_array::FixedSizeListArray>()
            .unwrap();

        for i in 0..batch.num_rows() {
            let embedding_values = embedding_col
                .value(i)
                .as_any()
                .downcast_ref::<Float32Array>()
                .unwrap()
                .values()
                .to_vec();

            engrams.push(Engram {
                id: ids.value(i).parse()?,
                content: contents.value(i).to_string(),
                embedding: embedding_values,
                scope: scopes.value(i).parse()?,
                category: categories.value(i).parse()?,
                entity: if entities.is_null(i) {
                    None
                } else {
                    Some(entities.value(i).to_string())
                },
                source_project: if source_projects.is_null(i) {
                    None
                } else {
                    Some(source_projects.value(i).to_string())
                },
                source_session: source_sessions.value(i).to_string(),
                source_timestamp: micros_to_dt(source_timestamps.value(i)),
                lifecycle: lifecycles.value(i).parse()?,
                confidence: confidences.value(i),
                created_at: micros_to_dt(created_ats.value(i)),
                updated_at: micros_to_dt(updated_ats.value(i)),
                accessed_at: if accessed_ats.is_null(i) {
                    None
                } else {
                    Some(micros_to_dt(accessed_ats.value(i)))
                },
                supersedes: if supersedes_col.is_null(i) {
                    None
                } else {
                    Some(supersedes_col.value(i).parse()?)
                },
                superseded_by: if superseded_by_col.is_null(i) {
                    None
                } else {
                    Some(superseded_by_col.value(i).parse()?)
                },
            });
        }
    }

    Ok(engrams)
}

fn micros_to_dt(micros: i64) -> DateTime<Utc> {
    Utc.timestamp_micros(micros).unwrap()
}
