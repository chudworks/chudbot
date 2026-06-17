use std::fs;
use std::path::{Path, PathBuf};

fn main() {
    let migrations = Path::new("../../migrations");
    println!("cargo:rerun-if-changed={}", migrations.display());

    let mut files = migration_files(migrations);
    files.sort();
    for file in files {
        println!("cargo:rerun-if-changed={}", file.display());
    }
}

fn migration_files(dir: &Path) -> Vec<PathBuf> {
    let Ok(entries) = fs::read_dir(dir) else {
        return Vec::new();
    };

    entries
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.extension().is_some_and(|extension| extension == "sql"))
        .collect()
}
