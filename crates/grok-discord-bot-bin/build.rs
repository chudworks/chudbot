use std::path::Path;
use std::process::Command;

fn git(args: &[&str]) -> Option<String> {
    Command::new("git")
        .args(args)
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .filter(|s| !s.is_empty())
}

/// Register a rerun trigger only for paths that exist: cargo re-runs the
/// build script on every build if a `rerun-if-changed` path is missing,
/// which would needlessly defeat the fingerprint cache.
fn watch(path: &str) {
    if Path::new(path).exists() {
        println!("cargo:rerun-if-changed={path}");
    }
}

fn main() {
    let version =
        git(&["describe", "--tags", "--always", "--dirty"]).unwrap_or_else(|| "unknown".to_string());

    println!("cargo:rustc-env=GIT_VERSION={version}");

    if std::env::var("PROFILE").unwrap_or_default() == "distribute" {
        println!("cargo:rustc-cfg=distribute");
    }

    println!("cargo:rustc-check-cfg=cfg(distribute)");

    // Recompute the version whenever the checked-out commit changes.
    // Watching `.git/HEAD` alone is NOT enough: it only changes on a
    // branch switch (or detach). Committing on the current branch updates
    // the branch ref (`.git/refs/heads/<branch>` when loose, or
    // `.git/packed-refs` after `git gc`/`pack-refs`), so watch those too.
    if let Some(git_dir) = git(&["rev-parse", "--absolute-git-dir"]) {
        watch(&format!("{git_dir}/HEAD"));
        watch(&format!("{git_dir}/packed-refs"));
        if let Some(head_ref) = git(&["rev-parse", "--symbolic-full-name", "HEAD"]) {
            watch(&format!("{git_dir}/{head_ref}"));
        }
    }
}
