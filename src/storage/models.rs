use std::fmt;
use std::sync::Arc;

use arrow_array::{
    ArrayRef, FixedSizeListArray, Float32Array, RecordBatch, RecordBatchReader, StringArray,
    TimestampMicrosecondArray,
};
use arrow_schema::{ArrowError, DataType, Field, Schema};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Memory {
    pub id: Uuid,
    pub content: String,
    #[serde(skip)]
    pub embedding: Vec<f32>,

    pub scope: Scope,
    pub category: Category,
    pub entity: Option<String>,

    pub source_project: Option<String>,
    pub source_session: String,
    pub source_timestamp: DateTime<Utc>,

    pub lifecycle: Lifecycle,
    pub confidence: f32,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub accessed_at: Option<DateTime<Utc>>,

    pub supersedes: Option<Uuid>,
    pub superseded_by: Option<Uuid>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Scope {
    Personal,
    Project,
}

impl fmt::Display for Scope {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Personal => write!(f, "personal"),
            Self::Project => write!(f, "project"),
        }
    }
}

impl std::str::FromStr for Scope {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "personal" => Ok(Self::Personal),
            "project" => Ok(Self::Project),
            _ => anyhow::bail!("invalid scope: {s}"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Category {
    Fact,
    Preference,
    Decision,
    Pattern,
    Context,
    Interest,
    Humor,
    Opinion,
}

impl fmt::Display for Category {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Fact => "fact",
            Self::Preference => "preference",
            Self::Decision => "decision",
            Self::Pattern => "pattern",
            Self::Context => "context",
            Self::Interest => "interest",
            Self::Humor => "humor",
            Self::Opinion => "opinion",
        };
        write!(f, "{s}")
    }
}

impl std::str::FromStr for Category {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "fact" => Ok(Self::Fact),
            "preference" => Ok(Self::Preference),
            "decision" => Ok(Self::Decision),
            "pattern" => Ok(Self::Pattern),
            "context" => Ok(Self::Context),
            "interest" => Ok(Self::Interest),
            "humor" => Ok(Self::Humor),
            "opinion" => Ok(Self::Opinion),
            _ => anyhow::bail!("invalid category: {s}"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Lifecycle {
    Candidate,
    Accepted,
}

impl fmt::Display for Lifecycle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Candidate => write!(f, "candidate"),
            Self::Accepted => write!(f, "accepted"),
        }
    }
}

impl std::str::FromStr for Lifecycle {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "candidate" => Ok(Self::Candidate),
            "accepted" => Ok(Self::Accepted),
            _ => anyhow::bail!("invalid lifecycle: {s}"),
        }
    }
}

/// The embedding dimension. Nomic-embed-text v1.5 produces 768-dim vectors.
pub const EMBEDDING_DIM: i32 = 768;

pub fn arrow_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("content", DataType::Utf8, false),
        Field::new(
            "embedding",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                EMBEDDING_DIM,
            ),
            false,
        ),
        Field::new("scope", DataType::Utf8, false),
        Field::new("category", DataType::Utf8, false),
        Field::new("entity", DataType::Utf8, true),
        Field::new("source_project", DataType::Utf8, true),
        Field::new("source_session", DataType::Utf8, false),
        Field::new(
            "source_timestamp",
            DataType::Timestamp(arrow_schema::TimeUnit::Microsecond, Some("UTC".into())),
            false,
        ),
        Field::new("lifecycle", DataType::Utf8, false),
        Field::new("confidence", DataType::Float32, false),
        Field::new(
            "created_at",
            DataType::Timestamp(arrow_schema::TimeUnit::Microsecond, Some("UTC".into())),
            false,
        ),
        Field::new(
            "updated_at",
            DataType::Timestamp(arrow_schema::TimeUnit::Microsecond, Some("UTC".into())),
            false,
        ),
        Field::new(
            "accessed_at",
            DataType::Timestamp(arrow_schema::TimeUnit::Microsecond, Some("UTC".into())),
            true,
        ),
        Field::new("supersedes", DataType::Utf8, true),
        Field::new("superseded_by", DataType::Utf8, true),
    ]))
}

fn dt_to_micros(dt: &DateTime<Utc>) -> i64 {
    dt.timestamp_micros()
}

fn opt_dt_to_micros(dt: &Option<DateTime<Utc>>) -> Option<i64> {
    dt.as_ref().map(|d| d.timestamp_micros())
}

fn opt_uuid_to_string(u: &Option<Uuid>) -> Option<String> {
    u.as_ref().map(|id| id.to_string())
}

impl Memory {
    /// Convert a batch of memories into a RecordBatchReader for LanceDB insertion.
    pub fn to_record_batch_reader(
        memories: Vec<Memory>,
    ) -> Box<dyn RecordBatchReader + Send> {
        let schema = arrow_schema();

        let ids: Vec<String> = memories.iter().map(|e| e.id.to_string()).collect();
        let contents: Vec<String> = memories.iter().map(|e| e.content.clone()).collect();
        let scopes: Vec<String> = memories.iter().map(|e| e.scope.to_string()).collect();
        let categories: Vec<String> = memories.iter().map(|e| e.category.to_string()).collect();
        let entities: Vec<Option<String>> = memories.iter().map(|e| e.entity.clone()).collect();
        let source_projects: Vec<Option<String>> =
            memories.iter().map(|e| e.source_project.clone()).collect();
        let source_sessions: Vec<String> =
            memories.iter().map(|e| e.source_session.clone()).collect();
        let source_timestamps: Vec<i64> =
            memories.iter().map(|e| dt_to_micros(&e.source_timestamp)).collect();
        let lifecycles: Vec<String> = memories.iter().map(|e| e.lifecycle.to_string()).collect();
        let confidences: Vec<f32> = memories.iter().map(|e| e.confidence).collect();
        let created_ats: Vec<i64> = memories.iter().map(|e| dt_to_micros(&e.created_at)).collect();
        let updated_ats: Vec<i64> = memories.iter().map(|e| dt_to_micros(&e.updated_at)).collect();
        let accessed_ats: Vec<Option<i64>> =
            memories.iter().map(|e| opt_dt_to_micros(&e.accessed_at)).collect();
        let supersedes_vec: Vec<Option<String>> =
            memories.iter().map(|e| opt_uuid_to_string(&e.supersedes)).collect();
        let superseded_by_vec: Vec<Option<String>> =
            memories.iter().map(|e| opt_uuid_to_string(&e.superseded_by)).collect();

        let n = memories.len();

        // Build embedding FixedSizeList
        let all_embeddings: Vec<f32> = memories.iter().flat_map(|e| e.embedding.iter().copied()).collect();
        assert_eq!(
            all_embeddings.len(),
            n * EMBEDDING_DIM as usize,
            "embedding total floats mismatch: got {} for {} memories (each should have {} dims)",
            all_embeddings.len(), n, EMBEDDING_DIM
        );
        let values = Arc::new(Float32Array::from(all_embeddings)) as ArrayRef;
        let field = Arc::new(Field::new("item", DataType::Float32, true));
        let embedding_array =
            FixedSizeListArray::try_new(field, EMBEDDING_DIM, values, None)
                .expect("embedding dimensions must match");
        let columns: Vec<(&str, ArrayRef)> = vec![
            ("id", Arc::new(StringArray::from(ids)) as ArrayRef),
            ("content", Arc::new(StringArray::from(contents))),
            ("embedding", Arc::new(embedding_array)),
            ("scope", Arc::new(StringArray::from(scopes))),
            ("category", Arc::new(StringArray::from(categories))),
            ("entity", Arc::new(StringArray::from(entities))),
            ("source_project", Arc::new(StringArray::from(source_projects))),
            ("source_session", Arc::new(StringArray::from(source_sessions))),
            ("source_timestamp", Arc::new(TimestampMicrosecondArray::from(source_timestamps).with_timezone("UTC"))),
            ("lifecycle", Arc::new(StringArray::from(lifecycles))),
            ("confidence", Arc::new(Float32Array::from(confidences))),
            ("created_at", Arc::new(TimestampMicrosecondArray::from(created_ats).with_timezone("UTC"))),
            ("updated_at", Arc::new(TimestampMicrosecondArray::from(updated_ats).with_timezone("UTC"))),
            ("accessed_at", Arc::new(TimestampMicrosecondArray::from(accessed_ats).with_timezone("UTC"))),
            ("supersedes", Arc::new(StringArray::from(supersedes_vec))),
            ("superseded_by", Arc::new(StringArray::from(superseded_by_vec))),
        ];

        let arrays: Vec<ArrayRef> = columns.into_iter().map(|(_, a)| a).collect();
        let batch = RecordBatch::try_new(schema.clone(), arrays)
            .expect("record batch construction should not fail");

        let batches: Vec<Result<RecordBatch, ArrowError>> = vec![Ok(batch)];
        Box::new(arrow_array::RecordBatchIterator::new(batches.into_iter(), schema))
    }

    /// Create a placeholder memory with zero embeddings (for Phase 1 testing).
    pub fn new_placeholder(
        content: String,
        scope: Scope,
        category: Category,
        entity: Option<String>,
        confidence: f32,
    ) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4(),
            content,
            embedding: vec![0.0; EMBEDDING_DIM as usize],
            scope,
            category,
            entity,
            source_project: None,
            source_session: Uuid::new_v4().to_string(),
            source_timestamp: now,
            lifecycle: if confidence >= 0.85 {
                Lifecycle::Accepted
            } else {
                Lifecycle::Candidate
            },
            confidence,
            created_at: now,
            updated_at: now,
            accessed_at: None,
            supersedes: None,
            superseded_by: None,
        }
    }
}
