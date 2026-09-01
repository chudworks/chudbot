use std::ffi::OsStr;
use std::path::{Component, Path, PathBuf};

use serde_json::Value;
use uuid::Uuid;

use crate::VibeError;

#[derive(Debug, Clone, Copy)]
pub struct ExportLimits {
    pub max_files: usize,
    pub max_file_bytes: u64,
    pub max_tree_bytes: u64,
}

#[derive(Debug, Clone)]
pub struct ExportedTree {
    pub root: PathBuf,
    pub files: usize,
    pub bytes: u64,
}

pub fn validate_relative_path(path: &Path) -> Result<(), VibeError> {
    if path.as_os_str().is_empty() {
        return Err(VibeError::InvalidSource("empty path".into()));
    }
    let text = path
        .to_str()
        .ok_or_else(|| VibeError::InvalidSource("paths must be UTF-8".into()))?;
    if text.contains('\0') || text.contains('\\') {
        return Err(VibeError::InvalidSource(
            "path contains a forbidden character".into(),
        ));
    }
    if text
        .split('/')
        .any(|segment| segment.is_empty() || matches!(segment, "." | ".."))
    {
        return Err(VibeError::InvalidSource(format!("invalid path `{text}`")));
    }
    for component in path.components() {
        let Component::Normal(segment) = component else {
            return Err(VibeError::InvalidSource(format!("invalid path `{text}`")));
        };
        if segment.is_empty() || segment == OsStr::new("..") {
            return Err(VibeError::InvalidSource(format!("invalid path `{text}`")));
        }
    }
    Ok(())
}

pub async fn validate_and_export(
    source: &Path,
    export_parent: &Path,
    limits: ExportLimits,
) -> Result<ExportedTree, VibeError> {
    let destination = export_parent.join(format!("export-{}", Uuid::new_v4()));
    tokio::fs::create_dir_all(&destination).await?;
    let result = copy_tree(source, &destination, limits).await;
    if result.is_err() {
        let _ = tokio::fs::remove_dir_all(&destination).await;
    }
    let (files, bytes) = result?;
    validate_dependencies(&destination).await?;
    Ok(ExportedTree {
        root: destination,
        files,
        bytes,
    })
}

async fn copy_tree(
    source: &Path,
    destination: &Path,
    limits: ExportLimits,
) -> Result<(usize, u64), VibeError> {
    let mut pending = vec![(source.to_path_buf(), PathBuf::new())];
    let mut files = 0usize;
    let mut bytes = 0u64;
    while let Some((directory, relative)) = pending.pop() {
        let mut entries = tokio::fs::read_dir(&directory).await?;
        while let Some(entry) = entries.next_entry().await? {
            let name = entry.file_name();
            let name_text = name
                .to_str()
                .ok_or_else(|| VibeError::InvalidSource("paths must be UTF-8".into()))?;
            if name_text == ".git" && relative.as_os_str().is_empty() {
                continue;
            }
            if matches!(name_text, "node_modules" | "dist") {
                continue;
            }
            if name_text == ".git"
                || name_text == ".gitmodules"
                || (relative.as_os_str().is_empty() && name_text == "__vibe")
            {
                return Err(VibeError::InvalidSource(format!(
                    "reserved source path `{name_text}`"
                )));
            }
            let child_relative = relative.join(&name);
            validate_relative_path(&child_relative)?;
            let metadata = tokio::fs::symlink_metadata(entry.path()).await?;
            let kind = metadata.file_type();
            if kind.is_symlink() {
                return Err(VibeError::InvalidSource(format!(
                    "symlink `{}` is not allowed",
                    child_relative.display()
                )));
            }
            if kind.is_dir() {
                tokio::fs::create_dir_all(destination.join(&child_relative)).await?;
                pending.push((entry.path(), child_relative));
                continue;
            }
            if !kind.is_file() {
                return Err(VibeError::InvalidSource(format!(
                    "special file `{}` is not allowed",
                    child_relative.display()
                )));
            }
            #[cfg(unix)]
            {
                use std::os::unix::fs::MetadataExt;
                if metadata.nlink() != 1 {
                    return Err(VibeError::InvalidSource(format!(
                        "hard link `{}` is not allowed",
                        child_relative.display()
                    )));
                }
            }
            if metadata.len() > limits.max_file_bytes {
                return Err(VibeError::InvalidSource(format!(
                    "file `{}` exceeds the per-file limit",
                    child_relative.display()
                )));
            }
            files = files
                .checked_add(1)
                .ok_or_else(|| VibeError::InvalidSource("file count overflow".into()))?;
            bytes = bytes
                .checked_add(metadata.len())
                .ok_or_else(|| VibeError::InvalidSource("source size overflow".into()))?;
            if files > limits.max_files {
                return Err(VibeError::InvalidSource(
                    "source contains too many files".into(),
                ));
            }
            if bytes > limits.max_tree_bytes {
                return Err(VibeError::InvalidSource(
                    "source tree exceeds the size limit".into(),
                ));
            }
            tokio::fs::copy(entry.path(), destination.join(&child_relative)).await?;
        }
    }
    Ok((files, bytes))
}

async fn validate_dependencies(root: &Path) -> Result<(), VibeError> {
    for forbidden in [".npmrc", "bunfig.toml"] {
        if tokio::fs::try_exists(root.join(forbidden)).await? {
            return Err(VibeError::InvalidSource(format!(
                "repository `{forbidden}` is not allowed"
            )));
        }
    }
    let package_path = root.join("package.json");
    let bytes = tokio::fs::read(&package_path)
        .await
        .map_err(|error| VibeError::InvalidSource(format!("package.json is required: {error}")))?;
    let package: Value = serde_json::from_slice(&bytes)
        .map_err(|error| VibeError::InvalidSource(format!("invalid package.json: {error}")))?;
    for section in [
        "dependencies",
        "devDependencies",
        "optionalDependencies",
        "peerDependencies",
    ] {
        let Some(dependencies) = package.get(section).and_then(Value::as_object) else {
            continue;
        };
        for (name, source) in dependencies {
            let Some(source) = source.as_str() else {
                return Err(VibeError::InvalidSource(format!(
                    "dependency `{name}` must use a registry version"
                )));
            };
            let lower = source.to_ascii_lowercase();
            if lower.starts_with("git")
                || lower.starts_with("ssh:")
                || lower.starts_with("github:")
                || lower.starts_with("gitlab:")
                || lower.starts_with("bitbucket:")
                || lower.starts_with("http:")
                || lower.starts_with("https:")
                || lower.starts_with("file:")
                || lower.starts_with("link:")
                || lower.contains("github.com/")
            {
                return Err(VibeError::InvalidSource(format!(
                    "dependency `{name}` uses a forbidden source"
                )));
            }
        }
    }
    Ok(())
}

pub async fn copy_built_artifact(
    source: &Path,
    destination: &Path,
    max_bytes: u64,
) -> Result<u64, VibeError> {
    if !tokio::fs::try_exists(source.join("index.html")).await? {
        return Err(VibeError::InvalidSource(
            "clean build did not produce dist/index.html".into(),
        ));
    }
    tokio::fs::create_dir_all(destination).await?;
    let limits = ExportLimits {
        max_files: usize::MAX,
        max_file_bytes: max_bytes,
        max_tree_bytes: max_bytes,
    };
    let (files, bytes) = copy_artifact_tree(source, destination, limits).await?;
    if files == 0 {
        return Err(VibeError::InvalidSource(
            "clean build output is empty".into(),
        ));
    }
    inject_sdk(&destination.join("index.html")).await?;
    Ok(bytes)
}

async fn copy_artifact_tree(
    source: &Path,
    destination: &Path,
    limits: ExportLimits,
) -> Result<(usize, u64), VibeError> {
    let mut pending = vec![(source.to_path_buf(), PathBuf::new())];
    let mut files = 0;
    let mut bytes = 0u64;
    while let Some((dir, relative)) = pending.pop() {
        let mut entries = tokio::fs::read_dir(dir).await?;
        while let Some(entry) = entries.next_entry().await? {
            let child = relative.join(entry.file_name());
            validate_relative_path(&child)?;
            let meta = tokio::fs::symlink_metadata(entry.path()).await?;
            if meta.file_type().is_symlink() || (!meta.is_file() && !meta.is_dir()) {
                return Err(VibeError::InvalidSource(
                    "build output contains a special file".into(),
                ));
            }
            if meta.is_dir() {
                tokio::fs::create_dir_all(destination.join(&child)).await?;
                pending.push((entry.path(), child));
            } else {
                files += 1;
                bytes = bytes
                    .checked_add(meta.len())
                    .ok_or_else(|| VibeError::InvalidSource("artifact size overflow".into()))?;
                if meta.len() > limits.max_file_bytes || bytes > limits.max_tree_bytes {
                    return Err(VibeError::InvalidSource(
                        "built output exceeds the size limit".into(),
                    ));
                }
                tokio::fs::copy(entry.path(), destination.join(child)).await?;
            }
        }
    }
    Ok((files, bytes))
}

async fn inject_sdk(index: &Path) -> Result<(), VibeError> {
    const TAG: &str = "<script defer src=\"/__vibe/sdk/v1/vibe.js\"></script>";
    let html = tokio::fs::read_to_string(index).await?;
    if html.contains(TAG) {
        return Ok(());
    }
    let updated = if let Some(position) = html.rfind("</body>") {
        format!("{}  {TAG}\n{}", &html[..position], &html[position..])
    } else {
        format!("{html}\n{TAG}\n")
    };
    tokio::fs::write(index, updated).await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_case::test_case;

    #[test_case("src/App.tsx", true)]
    #[test_case("../secret", false)]
    #[test_case("/etc/passwd", false)]
    #[test_case("a\\b", false)]
    #[test_case("", false)]
    fn path_rules(value: &str, accepted: bool) {
        assert_eq!(validate_relative_path(Path::new(value)).is_ok(), accepted);
    }

    fn limits() -> ExportLimits {
        ExportLimits {
            max_files: 8,
            max_file_bytes: 1024,
            max_tree_bytes: 4096,
        }
    }
    async fn fixture(package: &str) -> PathBuf {
        let root = std::env::temp_dir().join(format!("vibe-export-test-{}", Uuid::new_v4()));
        tokio::fs::create_dir_all(root.join("src")).await.unwrap();
        tokio::fs::write(root.join("package.json"), package)
            .await
            .unwrap();
        tokio::fs::write(root.join("src/App.tsx"), "export default 1")
            .await
            .unwrap();
        root
    }

    #[tokio::test]
    async fn export_rejects_forbidden_dependency_sources() {
        for source in [
            "git+https://example/x.git",
            "https://example/x.tgz",
            "file:../x",
            "link:../x",
        ] {
            let root = fixture(&format!(r#"{{"dependencies":{{"bad":"{source}"}}}}"#)).await;
            let result = validate_and_export(&root, &std::env::temp_dir(), limits()).await;
            assert!(
                matches!(result, Err(VibeError::InvalidSource(_))),
                "{source}"
            );
            let _ = tokio::fs::remove_dir_all(root).await;
        }
    }

    #[tokio::test]
    async fn export_rejects_reserved_nested_git_and_size_limits() {
        let root = fixture("{}").await;
        tokio::fs::create_dir_all(root.join("src/.git"))
            .await
            .unwrap();
        tokio::fs::write(root.join("large.bin"), vec![0u8; 1025])
            .await
            .unwrap();
        let result = validate_and_export(&root, &std::env::temp_dir(), limits()).await;
        assert!(matches!(result, Err(VibeError::InvalidSource(_))));
        let _ = tokio::fs::remove_dir_all(root).await;
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn export_rejects_symlinks_and_hard_links() {
        let root = fixture("{}").await;
        std::os::unix::fs::symlink("App.tsx", root.join("src/link.tsx")).unwrap();
        assert!(
            validate_and_export(&root, &std::env::temp_dir(), limits())
                .await
                .is_err()
        );
        tokio::fs::remove_file(root.join("src/link.tsx"))
            .await
            .unwrap();
        std::fs::hard_link(root.join("src/App.tsx"), root.join("src/hard.tsx")).unwrap();
        assert!(
            validate_and_export(&root, &std::env::temp_dir(), limits())
                .await
                .is_err()
        );
        let _ = tokio::fs::remove_dir_all(root).await;
    }
}
