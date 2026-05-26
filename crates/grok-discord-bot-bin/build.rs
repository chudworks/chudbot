use std::process::Command;

fn main() {
    let output = Command::new("git")
        .args(["describe", "--tags", "--always", "--dirty"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|| "unknown".to_string());

    println!("cargo:rustc-env=GIT_VERSION={output}");

    if std::env::var("PROFILE").unwrap_or_default() == "distribute" {
        println!("cargo:rustc-cfg=distribute");
    }

    println!("cargo:rustc-check-cfg=cfg(distribute)");
    println!("cargo:rerun-if-changed=../../.git/HEAD");
}
