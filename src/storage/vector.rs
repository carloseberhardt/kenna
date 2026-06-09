//! Shared helpers for the storage layer: in-app cosine similarity and the
//! serialization conversions between Rust values and their SQL representations.

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};

/// Compute cosine similarity between two vectors.
///
/// Returns `0.0` for empty or length-mismatched inputs. Length mismatch is
/// guarded against at insert time (see `MemoryDb::insert`) so it should not
/// occur for stored embeddings; this is a defensive fallback.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot / (norm_a * norm_b)
}

/// Serialize an embedding to a little-endian `f32` byte blob.
pub fn embedding_to_blob(v: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(v.len() * 4);
    for f in v {
        bytes.extend_from_slice(&f.to_le_bytes());
    }
    bytes
}

/// Deserialize a little-endian `f32` byte blob back into an embedding.
///
/// Errors if the byte length is not a multiple of 4.
pub fn blob_to_embedding(b: &[u8]) -> Result<Vec<f32>> {
    if !b.len().is_multiple_of(4) {
        anyhow::bail!(
            "embedding blob length {} is not a multiple of 4",
            b.len()
        );
    }
    Ok(b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

/// Convert microseconds-since-epoch (UTC) to a `DateTime<Utc>`.
pub fn micros_to_dt(micros: i64) -> Result<DateTime<Utc>> {
    DateTime::from_timestamp_micros(micros)
        .with_context(|| format!("timestamp out of range: {micros} micros"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blob_round_trip() {
        let v = vec![0.0_f32, 1.5, -2.25, 3.125, f32::MIN, f32::MAX];
        let blob = embedding_to_blob(&v);
        assert_eq!(blob.len(), v.len() * 4);
        let back = blob_to_embedding(&blob).unwrap();
        assert_eq!(v, back);
    }

    #[test]
    fn blob_bad_length_errors() {
        assert!(blob_to_embedding(&[0, 1, 2]).is_err());
    }

    #[test]
    fn micros_round_trip() {
        let dt = Utc::now();
        let micros = dt.timestamp_micros();
        let back = micros_to_dt(micros).unwrap();
        assert_eq!(dt.timestamp_micros(), back.timestamp_micros());
    }

    #[test]
    fn cosine_basics() {
        assert!((cosine_similarity(&[1.0, 0.0], &[1.0, 0.0]) - 1.0).abs() < 1e-6);
        assert!(cosine_similarity(&[1.0, 0.0], &[0.0, 1.0]).abs() < 1e-6);
        assert_eq!(cosine_similarity(&[1.0], &[1.0, 0.0]), 0.0);
        assert_eq!(cosine_similarity(&[], &[]), 0.0);
    }
}
