use std::path::{Path, PathBuf};
use std::process::Stdio;

use chudbot_api::{ExternalId, VibeRevisionId, VibeSiteId};
use tokio::process::Command;

use crate::VibeError;

const GENERATED_GITIGNORE_RULES: [&str; 3] = ["node_modules/", "dist/", "*.tsbuildinfo"];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GitCommit {
    pub oid: String,
}

#[derive(Debug, Clone)]
pub struct VibeDiskStore {
    root: PathBuf,
    template: PathBuf,
}

impl VibeDiskStore {
    pub fn new(root: PathBuf, template: PathBuf) -> Self {
        Self { root, template }
    }
    pub fn root(&self) -> &Path {
        &self.root
    }
    pub fn repo_path(&self, site: VibeSiteId) -> PathBuf {
        self.root.join("repos").join(format!("{site}.git"))
    }
    pub fn workspace_path(&self, job: chudbot_api::VibeJobId) -> PathBuf {
        self.root.join("workspaces").join(job.to_string())
    }
    pub fn artifact_path(&self, site: VibeSiteId, revision: VibeRevisionId) -> PathBuf {
        self.root
            .join("artifacts")
            .join(site.to_string())
            .join(revision.to_string())
    }

    pub async fn initialize(&self) -> Result<(), VibeError> {
        for child in ["repos", "workspaces", "artifacts"] {
            tokio::fs::create_dir_all(self.root.join(child)).await?;
        }
        Ok(())
    }

    pub async fn clear_workspaces(&self) -> Result<(), VibeError> {
        let workspaces = self.root.join("workspaces");
        if tokio::fs::try_exists(&workspaces).await? {
            tokio::fs::remove_dir_all(&workspaces).await?;
        }
        tokio::fs::create_dir_all(workspaces).await?;
        Ok(())
    }

    pub async fn create_workspace(
        &self,
        site: VibeSiteId,
        job: chudbot_api::VibeJobId,
        commit: Option<&str>,
    ) -> Result<PathBuf, VibeError> {
        let workspace = self.workspace_path(job);
        tokio::fs::create_dir_all(&workspace).await?;
        match commit {
            Some(oid) => {
                validate_oid(oid)?;
                self.git(
                    &self.repo_path(site),
                    Some(&workspace),
                    ["checkout", "--force", oid, "--", "."],
                    "checkout",
                )
                .await?;
            }
            None => copy_template(&self.template, &workspace).await?,
        }
        ensure_generated_gitignore(&workspace).await?;
        initialize_workspace_git(&workspace).await?;
        Ok(workspace)
    }

    pub async fn ensure_repository(&self, site: VibeSiteId) -> Result<(), VibeError> {
        let repo = self.repo_path(site);
        if tokio::fs::try_exists(&repo).await? {
            return Ok(());
        }
        tokio::fs::create_dir_all(repo.parent().expect("repo parent")).await?;
        let output = Command::new("git")
            .args(["init", "--bare", "--initial-branch=main"])
            .arg(&repo)
            .env_clear()
            .env("PATH", "/usr/bin:/bin:/usr/local/bin")
            .env("GIT_CONFIG_GLOBAL", "/dev/null")
            .env("GIT_CONFIG_NOSYSTEM", "1")
            .stdin(Stdio::null())
            .output()
            .await?;
        ensure_git(output, "init")?;
        Ok(())
    }

    pub async fn commit_tree(
        &self,
        site: VibeSiteId,
        source: &Path,
        parent: Option<&str>,
        user_id: &ExternalId,
        message: &str,
    ) -> Result<GitCommit, VibeError> {
        self.ensure_repository(site).await?;
        if let Some(parent) = parent {
            validate_oid(parent)?;
        }
        let repo = self.repo_path(site);
        let index = self
            .root
            .join("workspaces")
            .join(format!("index-{}", uuid::Uuid::new_v4()));
        if let Some(parent) = parent {
            self.git_with_index(&repo, source, &index, ["read-tree", parent], "read-tree")
                .await?;
        }
        self.git_with_index(&repo, source, &index, ["add", "--all", "--", "."], "add")
            .await?;
        let tree = self
            .git_with_index(&repo, source, &index, ["write-tree"], "write-tree")
            .await?;
        let tree = tree.trim();
        validate_oid(tree)?;
        let mut command = self.git_command(&repo, Some(source));
        command.arg("commit-tree").arg(tree);
        if let Some(parent) = parent {
            command.arg("-p").arg(parent);
        }
        let email = format!("{}@users.noreply.discord.invalid", user_id.as_str());
        let output = command
            .args(["-m", message])
            .env("GIT_AUTHOR_NAME", "Discord user")
            .env("GIT_AUTHOR_EMAIL", &email)
            .env("GIT_COMMITTER_NAME", "Chudbot")
            .env("GIT_COMMITTER_EMAIL", "chudbot@localhost.invalid")
            .output()
            .await?;
        let oid = ensure_git(output, "commit-tree")?.trim().to_string();
        validate_oid(&oid)?;
        let _ = tokio::fs::remove_file(index).await;
        Ok(GitCommit { oid })
    }

    pub async fn source_tree_oid(
        &self,
        site: VibeSiteId,
        source: &Path,
    ) -> Result<String, VibeError> {
        self.ensure_repository(site).await?;
        let repo = self.repo_path(site);
        let index = self
            .root
            .join("workspaces")
            .join(format!("index-{}", uuid::Uuid::new_v4()));
        self.git_with_index(&repo, source, &index, ["add", "--all", "--", "."], "add")
            .await?;
        let tree = self
            .git_with_index(&repo, source, &index, ["write-tree"], "write-tree")
            .await?;
        let _ = tokio::fs::remove_file(index).await;
        let tree = tree.trim().to_string();
        validate_oid(&tree)?;
        Ok(tree)
    }

    pub async fn commit_tree_oid(
        &self,
        site: VibeSiteId,
        commit: &str,
    ) -> Result<String, VibeError> {
        validate_oid(commit)?;
        let spec = format!("{commit}^{{tree}}");
        let tree = self
            .git(
                &self.repo_path(site),
                None,
                ["rev-parse", &spec],
                "rev-parse",
            )
            .await?;
        let tree = tree.trim().to_string();
        validate_oid(&tree)?;
        Ok(tree)
    }

    pub async fn activate_main(&self, site: VibeSiteId, oid: &str) -> Result<(), VibeError> {
        validate_oid(oid)?;
        self.git(
            &self.repo_path(site),
            None,
            ["update-ref", "refs/heads/main", oid],
            "update-ref",
        )
        .await?;
        Ok(())
    }

    pub async fn repair_main(&self, site: VibeSiteId, oid: &str) -> Result<(), VibeError> {
        self.activate_main(site, oid).await
    }

    pub async fn main_oid(&self, site: VibeSiteId) -> Result<Option<String>, VibeError> {
        let output = self
            .git_command(&self.repo_path(site), None)
            .args(["rev-parse", "--verify", "refs/heads/main"])
            .output()
            .await?;
        if !output.status.success() {
            return Ok(None);
        }
        let oid = String::from_utf8(output.stdout)
            .map_err(|_| VibeError::Git {
                operation: "rev-parse",
                message: "git returned non-UTF-8 output".into(),
            })?
            .trim()
            .to_string();
        validate_oid(&oid)?;
        Ok(Some(oid))
    }

    pub async fn purge(&self, site: VibeSiteId) -> Result<(), VibeError> {
        let repo = self.repo_path(site);
        let artifacts = self.root.join("artifacts").join(site.to_string());
        if tokio::fs::try_exists(&repo).await? {
            tokio::fs::remove_dir_all(repo).await?;
        }
        if tokio::fs::try_exists(&artifacts).await? {
            tokio::fs::remove_dir_all(artifacts).await?;
        }
        Ok(())
    }

    /// List source paths in one trusted stored commit.
    pub async fn list_files(&self, site: VibeSiteId, oid: &str) -> Result<Vec<String>, VibeError> {
        validate_oid(oid)?;
        let output = self
            .git(
                &self.repo_path(site),
                None,
                ["ls-tree", "-r", "--name-only", oid, "--"],
                "ls-tree",
            )
            .await?;
        Ok(output
            .lines()
            .filter(|path| source_path_is_visible(path))
            .map(str::to_string)
            .collect())
    }

    /// Read a bounded UTF-8 source file from one trusted stored commit.
    pub async fn read_file(
        &self,
        site: VibeSiteId,
        oid: &str,
        path: &Path,
    ) -> Result<String, VibeError> {
        validate_oid(oid)?;
        crate::export::validate_relative_path(path)?;
        let path_text = path
            .to_str()
            .ok_or_else(|| VibeError::InvalidSource("paths must be UTF-8".into()))?;
        if !source_path_is_visible(path_text) {
            return Err(VibeError::InvalidSource(
                "generated source paths are hidden".into(),
            ));
        }
        let spec = format!("{oid}:{}", path.display());
        let output = self
            .git(&self.repo_path(site), None, ["show", &spec], "show")
            .await?;
        Ok(output.chars().take(1024 * 1024).collect())
    }

    /// Produce a bounded, no-external-driver patch between two stored commits.
    pub async fn diff(&self, site: VibeSiteId, from: &str, to: &str) -> Result<String, VibeError> {
        validate_oid(from)?;
        validate_oid(to)?;
        let output = self
            .git(
                &self.repo_path(site),
                None,
                [
                    "diff",
                    "--no-ext-diff",
                    "--no-textconv",
                    from,
                    to,
                    "--",
                    ".",
                    ":(glob,exclude)**/node_modules/**",
                    ":(glob,exclude)**/dist/**",
                    ":(glob,exclude)**/*.tsbuildinfo",
                ],
                "diff",
            )
            .await?;
        Ok(output.chars().take(2 * 1024 * 1024).collect())
    }

    fn git_command(&self, repo: &Path, work_tree: Option<&Path>) -> Command {
        let mut command = Command::new("git");
        command
            .env_clear()
            .env("PATH", "/usr/bin:/bin:/usr/local/bin")
            .env("GIT_CONFIG_GLOBAL", "/dev/null")
            .env("GIT_CONFIG_NOSYSTEM", "1")
            .env("GIT_OPTIONAL_LOCKS", "0")
            .args(["-c", "core.hooksPath=/dev/null"])
            .arg("--git-dir")
            .arg(repo);
        if let Some(work_tree) = work_tree {
            command.arg("--work-tree").arg(work_tree);
        }
        command
            .stdin(Stdio::null())
            .stderr(Stdio::piped())
            .stdout(Stdio::piped());
        command
    }

    async fn git<const N: usize>(
        &self,
        repo: &Path,
        work: Option<&Path>,
        args: [&str; N],
        operation: &'static str,
    ) -> Result<String, VibeError> {
        let output = self.git_command(repo, work).args(args).output().await?;
        ensure_git(output, operation)
    }
    async fn git_with_index<const N: usize>(
        &self,
        repo: &Path,
        work: &Path,
        index: &Path,
        args: [&str; N],
        operation: &'static str,
    ) -> Result<String, VibeError> {
        let output = self
            .git_command(repo, Some(work))
            .env("GIT_INDEX_FILE", index)
            .args(args)
            .output()
            .await?;
        ensure_git(output, operation)
    }
}

fn source_path_is_visible(path: &str) -> bool {
    !path
        .split('/')
        .any(|segment| matches!(segment, "node_modules" | "dist"))
        && !path
            .rsplit('/')
            .next()
            .is_some_and(|name| name.ends_with(".tsbuildinfo"))
}

fn validate_oid(oid: &str) -> Result<(), VibeError> {
    if (40..=64).contains(&oid.len()) && oid.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        Ok(())
    } else {
        Err(VibeError::Git {
            operation: "validate",
            message: "invalid object id".into(),
        })
    }
}

fn ensure_git(output: std::process::Output, operation: &'static str) -> Result<String, VibeError> {
    if output.status.success() {
        return String::from_utf8(output.stdout).map_err(|_| VibeError::Git {
            operation,
            message: "git returned non-UTF-8 output".into(),
        });
    }
    let message = String::from_utf8_lossy(&output.stderr)
        .chars()
        .take(2_000)
        .collect();
    Err(VibeError::Git { operation, message })
}

async fn copy_template(source: &Path, destination: &Path) -> Result<(), VibeError> {
    let mut pending = vec![(source.to_path_buf(), destination.to_path_buf())];
    while let Some((source, destination)) = pending.pop() {
        tokio::fs::create_dir_all(&destination).await?;
        let mut entries = tokio::fs::read_dir(source).await?;
        while let Some(entry) = entries.next_entry().await? {
            let target = destination.join(entry.file_name());
            let metadata = entry.metadata().await?;
            if metadata.is_dir() {
                pending.push((entry.path(), target));
            } else if metadata.is_file() {
                tokio::fs::copy(entry.path(), target).await?;
            } else {
                return Err(VibeError::InvalidSource(
                    "template contains a special file".into(),
                ));
            }
        }
    }
    Ok(())
}

async fn ensure_generated_gitignore(workspace: &Path) -> Result<(), VibeError> {
    let path = workspace.join(".gitignore");
    let mut contents = if tokio::fs::try_exists(&path).await? {
        tokio::fs::read_to_string(&path)
            .await
            .map_err(|_| VibeError::InvalidSource(".gitignore must be UTF-8".into()))?
    } else {
        String::new()
    };
    let missing = GENERATED_GITIGNORE_RULES
        .into_iter()
        .filter(|rule| !contents.lines().any(|line| line.trim() == *rule))
        .collect::<Vec<_>>();
    if missing.is_empty() {
        return Ok(());
    }
    if !contents.is_empty() && !contents.ends_with('\n') {
        contents.push('\n');
    }
    for rule in missing {
        contents.push_str(rule);
        contents.push('\n');
    }
    tokio::fs::write(path, contents).await?;
    Ok(())
}

async fn initialize_workspace_git(workspace: &Path) -> Result<(), VibeError> {
    let base = |command: &mut Command| {
        command
            .env_clear()
            .env("PATH", "/usr/bin:/bin:/usr/local/bin")
            .env("GIT_CONFIG_GLOBAL", "/dev/null")
            .env("GIT_CONFIG_NOSYSTEM", "1")
            .args(["-c", "core.hooksPath=/dev/null"])
            .current_dir(workspace)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
    };
    let mut init = Command::new("git");
    base(&mut init);
    let output = init
        .args(["init", "--initial-branch=main"])
        .output()
        .await?;
    ensure_git(output, "workspace init")?;
    let mut add = Command::new("git");
    base(&mut add);
    let output = add.args(["add", "--all", "--", "."]).output().await?;
    ensure_git(output, "workspace add")?;
    let mut commit = Command::new("git");
    base(&mut commit);
    let output = commit
        .args([
            "-c",
            "user.name=Chudbot",
            "-c",
            "user.email=chudbot@localhost.invalid",
            "commit",
            "--quiet",
            "-m",
            "Vibe workspace baseline",
        ])
        .output()
        .await?;
    ensure_git(output, "workspace commit")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use chudbot_api::{ExternalId, VibeJobId};

    #[tokio::test]
    async fn orphan_commit_is_harmless_and_main_can_be_repaired() {
        let root = std::env::temp_dir().join(format!("vibe-git-test-{}", uuid::Uuid::new_v4()));
        let template = root.join("template");
        tokio::fs::create_dir_all(&template).await.unwrap();
        tokio::fs::write(template.join("package.json"), "{}")
            .await
            .unwrap();
        let store = VibeDiskStore::new(root.clone(), template);
        store.initialize().await.unwrap();
        let site = VibeSiteId::new();
        let first_job = VibeJobId::new();
        let first = store.create_workspace(site, first_job, None).await.unwrap();
        let commit1 = store
            .commit_tree(site, &first, None, &ExternalId::new("1"), "Create site")
            .await
            .unwrap();
        store.activate_main(site, &commit1.oid).await.unwrap();
        let second = store
            .create_workspace(site, VibeJobId::new(), Some(&commit1.oid))
            .await
            .unwrap();
        tokio::fs::write(second.join("index.html"), "second")
            .await
            .unwrap();
        let commit2 = store
            .commit_tree(
                site,
                &second,
                Some(&commit1.oid),
                &ExternalId::new("1"),
                "Update site",
            )
            .await
            .unwrap();
        assert_eq!(
            store.main_oid(site).await.unwrap().as_deref(),
            Some(commit1.oid.as_str())
        );
        store.repair_main(site, &commit2.oid).await.unwrap();
        assert_eq!(
            store.main_oid(site).await.unwrap().as_deref(),
            Some(commit2.oid.as_str())
        );
        store.clear_workspaces().await.unwrap();
        assert!(
            tokio::fs::read_dir(root.join("workspaces"))
                .await
                .unwrap()
                .next_entry()
                .await
                .unwrap()
                .is_none()
        );
        let _ = tokio::fs::remove_dir_all(root).await;
    }

    #[tokio::test]
    async fn source_history_hides_legacy_dependencies_and_build_artifacts() {
        let root = std::env::temp_dir().join(format!(
            "vibe-git-hidden-source-test-{}",
            uuid::Uuid::new_v4()
        ));
        let template = root.join("template");
        tokio::fs::create_dir_all(&template).await.unwrap();
        tokio::fs::write(template.join("package.json"), "{}")
            .await
            .unwrap();
        tokio::fs::write(template.join(".gitignore"), "custom-cache/\n")
            .await
            .unwrap();
        let store = VibeDiskStore::new(root.clone(), template);
        store.initialize().await.unwrap();
        let site = VibeSiteId::new();
        let first = store
            .create_workspace(site, VibeJobId::new(), None)
            .await
            .unwrap();
        assert_eq!(
            tokio::fs::read_to_string(first.join(".gitignore"))
                .await
                .unwrap(),
            "custom-cache/\nnode_modules/\ndist/\n*.tsbuildinfo\n"
        );
        tokio::fs::remove_file(first.join(".gitignore"))
            .await
            .unwrap();
        tokio::fs::create_dir_all(first.join("src")).await.unwrap();
        tokio::fs::write(first.join("src/App.tsx"), "export default 1")
            .await
            .unwrap();
        tokio::fs::create_dir_all(first.join("node_modules/pkg"))
            .await
            .unwrap();
        tokio::fs::write(first.join("node_modules/pkg/index.js"), "dependency")
            .await
            .unwrap();
        tokio::fs::create_dir_all(first.join("dist/assets"))
            .await
            .unwrap();
        tokio::fs::write(first.join("dist/assets/index.js"), "artifact one")
            .await
            .unwrap();
        tokio::fs::write(first.join("tsconfig.app.tsbuildinfo"), "metadata one")
            .await
            .unwrap();
        let commit1 = store
            .commit_tree(site, &first, None, &ExternalId::new("1"), "Create site")
            .await
            .unwrap();

        assert_eq!(
            store.list_files(site, &commit1.oid).await.unwrap(),
            ["package.json", "src/App.tsx"]
        );
        for hidden in [
            "node_modules/pkg/index.js",
            "dist/assets/index.js",
            "tsconfig.app.tsbuildinfo",
        ] {
            assert!(
                store
                    .read_file(site, &commit1.oid, Path::new(hidden))
                    .await
                    .is_err(),
                "{hidden}"
            );
        }

        let second = store
            .create_workspace(site, VibeJobId::new(), Some(&commit1.oid))
            .await
            .unwrap();
        assert_eq!(
            tokio::fs::read_to_string(second.join(".gitignore"))
                .await
                .unwrap(),
            "node_modules/\ndist/\n*.tsbuildinfo\n"
        );
        tokio::fs::write(second.join("src/App.tsx"), "export default 2")
            .await
            .unwrap();
        tokio::fs::write(second.join("dist/assets/index.js"), "artifact two")
            .await
            .unwrap();
        tokio::fs::write(second.join("tsconfig.app.tsbuildinfo"), "metadata two")
            .await
            .unwrap();
        let commit2 = store
            .commit_tree(
                site,
                &second,
                Some(&commit1.oid),
                &ExternalId::new("1"),
                "Update site",
            )
            .await
            .unwrap();
        let diff = store.diff(site, &commit1.oid, &commit2.oid).await.unwrap();
        assert!(diff.contains("src/App.tsx"));
        assert!(!diff.contains("diff --git a/node_modules/"));
        assert!(!diff.contains("diff --git a/dist/"));
        assert!(!diff.contains("diff --git a/tsconfig.app.tsbuildinfo"));

        let _ = tokio::fs::remove_dir_all(root).await;
    }
}
