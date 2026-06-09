use std::fmt;

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

/// Default embedding dimension when no model is loaded. Nomic-embed-text v1.5
/// produces 768-dim vectors. The store no longer enforces a fixed dimension at
/// the schema level — it records each row's dimension and guards against
/// mismatches at insert time (see `MemoryDb::insert`).
pub const DEFAULT_EMBEDDING_DIM: usize = 768;

impl Memory {
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
            embedding: vec![0.0; DEFAULT_EMBEDDING_DIM],
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
