//! Build-time metadata wiring for the `chudbot-bin` executable.
//!
//! This script emits Cargo directives for the embedded Git version string,
//! the distribution-build cfg flag, and the Git files that should trigger a
//! rebuild when the version string can change.

use std::path::Path;
use std::process::Command;

/// Run a Git command and return trimmed stdout when it exits successfully.
///
/// Build scripts should degrade cleanly in source archives or other
/// non-checkout environments, so failures and empty output both become `None`.
fn git(args: &[&str]) -> Option<String> {
    Command::new("git")
        .args(args)
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_string())
        .filter(|value| !value.is_empty())
}

/// Ask Cargo to rerun the build script when an existing path changes.
///
/// Git worktrees do not always have every file this script cares about
/// (`packed-refs`, for example), and Cargo expects concrete paths rather than
/// optional filesystem state.
fn watch(path: &str) {
    if Path::new(path).exists() {
        println!("cargo:rerun-if-changed={path}");
    }
}

fn main() {
    // Embed a best-effort human-readable build identifier for runtime display.
    let version = git(&["describe", "--tags", "--always", "--dirty"])
        .unwrap_or_else(|| "unknown".to_string());
    println!("cargo:rustc-env=GIT_VERSION={version}");

    // Mirror Cargo's custom `distribute` profile into a cfg checked by rustc.
    if std::env::var("PROFILE").unwrap_or_default() == "distribute" {
        println!("cargo:rustc-cfg=distribute");
    }
    println!("cargo:rustc-check-cfg=cfg(distribute)");

    // Track the Git refs that can affect `git describe` without watching the
    // whole repository.
    if let Some(git_dir) = git(&["rev-parse", "--absolute-git-dir"]) {
        watch(&format!("{git_dir}/HEAD"));
        watch(&format!("{git_dir}/packed-refs"));
        if let Some(head_ref) = git(&["rev-parse", "--symbolic-full-name", "HEAD"]) {
            watch(&format!("{git_dir}/{head_ref}"));
        }
    }
}
