use std::path::Path;
use std::process::Command;

fn git(args: &[&str]) -> Option<String> {
    Command::new("git")
        .args(args)
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_string())
        .filter(|value| !value.is_empty())
}

fn watch(path: &str) {
    if Path::new(path).exists() {
        println!("cargo:rerun-if-changed={path}");
    }
}

fn main() {
    let version = git(&["describe", "--tags", "--always", "--dirty"])
        .unwrap_or_else(|| "unknown".to_string());
    println!("cargo:rustc-env=GIT_VERSION={version}");

    if std::env::var("PROFILE").unwrap_or_default() == "distribute" {
        println!("cargo:rustc-cfg=distribute");
    }
    println!("cargo:rustc-check-cfg=cfg(distribute)");

    if let Some(git_dir) = git(&["rev-parse", "--absolute-git-dir"]) {
        watch(&format!("{git_dir}/HEAD"));
        watch(&format!("{git_dir}/packed-refs"));
        if let Some(head_ref) = git(&["rev-parse", "--symbolic-full-name", "HEAD"]) {
            watch(&format!("{git_dir}/{head_ref}"));
        }
    }
}
