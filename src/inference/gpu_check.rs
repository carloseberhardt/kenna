use std::process::Command;

/// VRAM status for the primary GPU (card0).
#[derive(Debug, Clone, Copy)]
pub struct VramStatus {
    pub total_bytes: u64,
    pub used_bytes: u64,
}

impl VramStatus {
    pub fn free_bytes(&self) -> u64 {
        self.total_bytes.saturating_sub(self.used_bytes)
    }

    pub fn free_gb(&self) -> f64 {
        self.free_bytes() as f64 / 1_073_741_824.0
    }

    pub fn used_gb(&self) -> f64 {
        self.used_bytes as f64 / 1_073_741_824.0
    }

    pub fn total_gb(&self) -> f64 {
        self.total_bytes as f64 / 1_073_741_824.0
    }
}

/// Query VRAM status for card0 via `rocm-smi`.
/// Returns None if rocm-smi is unavailable or parsing fails — in that case
/// callers should proceed without the check (fail-open).
pub fn query_vram() -> Option<VramStatus> {
    let output = Command::new("rocm-smi")
        .args(["--showmeminfo", "vram", "--json"])
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let stdout = String::from_utf8(output.stdout).ok()?;

    // rocm-smi prints a warning line before the JSON on some systems.
    // Find the first '{' and parse from there.
    let json_start = stdout.find('{')?;
    let json = &stdout[json_start..];

    let parsed: serde_json::Value = serde_json::from_str(json).ok()?;
    let card0 = parsed.get("card0")?;

    let total_str = card0.get("VRAM Total Memory (B)")?.as_str()?;
    let used_str = card0.get("VRAM Total Used Memory (B)")?.as_str()?;

    let total_bytes: u64 = total_str.parse().ok()?;
    let used_bytes: u64 = used_str.parse().ok()?;

    Some(VramStatus {
        total_bytes,
        used_bytes,
    })
}

/// Check whether the GPU has enough free VRAM for a job.
/// Returns Ok(()) if the check passes or rocm-smi is unavailable.
/// Returns Err with a diagnostic message if VRAM is insufficient.
pub fn ensure_free_vram(min_free_gb: f64, job_name: &str) -> Result<(), String> {
    let status = match query_vram() {
        Some(s) => s,
        None => {
            // rocm-smi unavailable — proceed without the check
            tracing::debug!("rocm-smi unavailable, skipping VRAM check for {job_name}");
            return Ok(());
        }
    };

    let free = status.free_gb();
    if free >= min_free_gb {
        tracing::debug!(
            "VRAM check passed for {job_name}: {free:.1} GB free (need {min_free_gb:.1} GB)"
        );
        return Ok(());
    }

    Err(format!(
        "GPU busy: {free:.1} GB free of {:.1} GB total (need {min_free_gb:.1} GB for {job_name}). Skipping.",
        status.total_gb()
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vram_status_calculations() {
        let status = VramStatus {
            total_bytes: 21_474_836_480, // 20 GB
            used_bytes: 5_368_709_120,   // 5 GB
        };
        assert!((status.total_gb() - 20.0).abs() < 0.01);
        assert!((status.used_gb() - 5.0).abs() < 0.01);
        assert!((status.free_gb() - 15.0).abs() < 0.01);
    }

    #[test]
    fn test_free_bytes_saturating() {
        let status = VramStatus {
            total_bytes: 100,
            used_bytes: 200, // impossible but saturating should handle it
        };
        assert_eq!(status.free_bytes(), 0);
    }
}
