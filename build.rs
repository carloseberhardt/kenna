use std::process::Command;

/// Embed the current git short SHA (plus a `-dirty` marker when the working
/// tree has uncommitted changes) into the binary as `KENNA_GIT_SHA`, so
/// `kenna --version` can report exactly which commit a build came from.
fn main() {
    let sha = run_git(&["rev-parse", "--short", "HEAD"]).unwrap_or_else(|| "unknown".to_string());
    // Tracked changes only — untracked scratch files shouldn't flip the flag,
    // since they don't affect the build.
    let dirty = run_git(&["status", "--porcelain", "--untracked-files=no"]).is_some_and(|s| !s.is_empty());
    let suffix = if dirty { "-dirty" } else { "" };
    println!("cargo:rustc-env=KENNA_GIT_SHA={sha}{suffix}");

    // Re-embed when HEAD moves (commit / checkout) or the pointed-to ref changes.
    println!("cargo:rerun-if-changed=.git/HEAD");
    if let Ok(head) = std::fs::read_to_string(".git/HEAD")
        && let Some(reference) = head.strip_prefix("ref: ").map(str::trim)
    {
        println!("cargo:rerun-if-changed=.git/{reference}");
    }
    println!("cargo:rerun-if-changed=.git/packed-refs");
}

fn run_git(args: &[&str]) -> Option<String> {
    let out = Command::new("git").args(args).output().ok()?;
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).trim().to_string())
}
