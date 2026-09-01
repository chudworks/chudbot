use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;

use chudbot_api::VibeJobId;
use tokio::process::Command;

use crate::config::VibeSandboxConfig;
use crate::{VibeError, export};

const MAX_COMMAND_OUTPUT: usize = 64 * 1024;
const VIBE_DOCKER_NETWORK: &str = "chudbot-vibe";

#[derive(Debug, Clone)]
pub struct VibeSandbox {
    config: VibeSandboxConfig,
}

#[derive(Debug)]
pub struct CodingContainer {
    name: String,
    workspace: PathBuf,
    sandbox: VibeSandbox,
    removed: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SandboxCommandOutput {
    pub stdout: String,
    pub stderr: String,
    pub exit_code: Option<i32>,
    pub timed_out: bool,
    pub truncated: bool,
}

#[derive(Debug, Clone)]
pub struct BuildOutput {
    pub image_id: String,
    pub log: String,
    pub artifact_bytes: u64,
}

impl VibeSandbox {
    pub fn new(config: VibeSandboxConfig) -> Self {
        Self { config }
    }
    pub fn config(&self) -> &VibeSandboxConfig {
        &self.config
    }

    pub async fn check_available(&self) -> Result<String, VibeError> {
        let output = self
            .docker()
            .args([
                "image",
                "inspect",
                "--format",
                "{{.Id}}",
                &self.config.image,
            ])
            .output()
            .await?;
        let image_id = ensure_docker(output, "inspect image")?;
        let output = self
            .docker()
            .args(["network", "inspect", VIBE_DOCKER_NETWORK])
            .output()
            .await?;
        ensure_docker(output, "inspect Vibe network")?;
        Ok(image_id)
    }

    pub async fn start_coding(
        &self,
        job: VibeJobId,
        workspace: &Path,
    ) -> Result<CodingContainer, VibeError> {
        let name = format!("chudbot-vibe-{job}-coding");
        let _ = self.force_remove(&name).await;
        let output = self
            .base_create(&name, workspace)
            .args([
                &self.config.image,
                "/bin/bash",
                "-lc",
                "exec sleep infinity",
            ])
            .output()
            .await?;
        if let Err(error) = ensure_docker(output, "create coding container") {
            let _ = self.force_remove(&name).await;
            return Err(error);
        }
        let output = self.docker().args(["start", &name]).output().await?;
        if let Err(error) = ensure_docker(output, "start coding container") {
            let _ = self.force_remove(&name).await;
            return Err(error);
        }
        Ok(CodingContainer {
            name,
            workspace: workspace.to_path_buf(),
            sandbox: self.clone(),
            removed: false,
        })
    }

    pub async fn clean_build(
        &self,
        job: VibeJobId,
        source: &Path,
        artifact: &Path,
        max_artifact_bytes: u64,
    ) -> Result<BuildOutput, VibeError> {
        let name = format!("chudbot-vibe-{job}-build");
        let _ = self.force_remove(&name).await;
        let output = self
            .base_create(&name, source)
            .args([
                &self.config.image,
                "/bin/bash",
                "-lc",
                "bun install --frozen-lockfile && bun run build",
            ])
            .output()
            .await?;
        if let Err(error) = ensure_docker(output, "create build container") {
            let _ = self.force_remove(&name).await;
            return Err(error);
        }
        let result = tokio::time::timeout(
            Duration::from_secs(self.config.build_timeout_seconds),
            async {
                self.docker()
                    .args(["start", "--attach", &name])
                    .output()
                    .await
            },
        )
        .await;
        let output = match result {
            Ok(Ok(output)) => output,
            Ok(Err(error)) => {
                let _ = self.force_remove(&name).await;
                return Err(error.into());
            }
            Err(_) => {
                let _ = self.force_remove(&name).await;
                return Err(VibeError::TimedOut);
            }
        };
        let log = bounded_combined_output(&output);
        if !output.status.success() {
            let _ = self.force_remove(&name).await;
            return Err(VibeError::Sandbox {
                operation: "clean build",
                message: log,
            });
        }
        let result = async {
            let image_id = self.check_available().await?;
            let artifact_bytes =
                export::copy_built_artifact(&source.join("dist"), artifact, max_artifact_bytes)
                    .await?;
            Ok(BuildOutput {
                image_id: image_id.trim().to_string(),
                log,
                artifact_bytes,
            })
        }
        .await;
        let cleanup = self.force_remove(&name).await;
        if result.is_err() {
            let _ = tokio::fs::remove_dir_all(artifact).await;
        }
        cleanup?;
        result
    }

    pub async fn cleanup_job(&self, job: VibeJobId) -> Result<(), VibeError> {
        for suffix in ["coding", "build"] {
            self.force_remove(&format!("chudbot-vibe-{job}-{suffix}"))
                .await?;
        }
        Ok(())
    }

    pub async fn cleanup_orphaned_containers(&self) -> Result<usize, VibeError> {
        let output = self
            .docker()
            .args([
                "ps",
                "--all",
                "--filter",
                "name=^/chudbot-vibe-",
                "--format",
                "{{.Names}}",
            ])
            .output()
            .await?;
        let names = ensure_docker(output, "list Vibe containers")?;
        let mut removed = 0;
        for name in names.lines().filter(|name| {
            name.starts_with("chudbot-vibe-")
                && name
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-')
        }) {
            self.force_remove(name).await?;
            removed += 1;
        }
        Ok(removed)
    }

    fn base_create(&self, name: &str, workspace: &Path) -> Command {
        let mut command = self.docker();
        command.args([
            "create",
            "--name",
            name,
            "--network",
            VIBE_DOCKER_NETWORK,
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges:true",
            "--user",
            "1000:1000",
            "--workdir",
            "/workspace",
            "--env",
            "BUN_INSTALL_CACHE_DIR=/tmp/bun-cache",
            "--env",
            "XDG_CACHE_HOME=/tmp/cache",
            "--pids-limit",
            &self.config.pids.to_string(),
            "--cpus",
            &self.config.cpus.to_string(),
            "--memory",
            &format!("{}m", self.config.memory_mebibytes),
            "--tmpfs",
            "/tmp:rw,nosuid,nodev,size=256m",
            "--mount",
            &format!("type=bind,src={},dst=/workspace", workspace.display()),
        ]);
        command
    }

    fn docker(&self) -> Command {
        let mut command = Command::new("docker");
        command
            .arg("--host")
            .arg(format!("unix://{}", self.config.docker_socket.display()))
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        command
    }

    async fn force_remove(&self, name: &str) -> Result<(), VibeError> {
        let output = self.docker().args(["rm", "--force", name]).output().await?;
        if output.status.success()
            || String::from_utf8_lossy(&output.stderr).contains("No such container")
        {
            Ok(())
        } else {
            Err(VibeError::Sandbox {
                operation: "remove container",
                message: "container cleanup failed".into(),
            })
        }
    }
}

impl CodingContainer {
    pub fn workspace(&self) -> &Path {
        &self.workspace
    }

    pub async fn shell(&mut self, command_text: &str) -> Result<SandboxCommandOutput, VibeError> {
        let result = tokio::time::timeout(
            Duration::from_secs(self.sandbox.config.command_timeout_seconds),
            self.sandbox
                .docker()
                .args(["exec", &self.name, "/bin/bash", "-lc", command_text])
                .output(),
        )
        .await;
        match result {
            Ok(output) => Ok(command_output(output?)),
            Err(_) => {
                self.sandbox.force_remove(&self.name).await?;
                self.removed = true;
                Ok(SandboxCommandOutput {
                    stdout: String::new(),
                    stderr: "command timed out; the sandbox was stopped".into(),
                    exit_code: None,
                    timed_out: true,
                    truncated: false,
                })
            }
        }
    }

    pub async fn remove(mut self) -> Result<(), VibeError> {
        if !self.removed {
            self.sandbox.force_remove(&self.name).await?;
            self.removed = true;
        }
        Ok(())
    }
}

fn command_output(output: std::process::Output) -> SandboxCommandOutput {
    let (stdout, stdout_truncated) = bounded_text(&output.stdout);
    let (stderr, stderr_truncated) = bounded_text(&output.stderr);
    SandboxCommandOutput {
        stdout,
        stderr,
        exit_code: output.status.code(),
        timed_out: false,
        truncated: stdout_truncated || stderr_truncated,
    }
}

fn bounded_text(bytes: &[u8]) -> (String, bool) {
    let truncated = bytes.len() > MAX_COMMAND_OUTPUT;
    let bytes = &bytes[..bytes.len().min(MAX_COMMAND_OUTPUT)];
    let mut text = String::from_utf8_lossy(bytes).into_owned();
    if truncated {
        text.push_str("\n[output truncated]");
    }
    (text, truncated)
}

fn bounded_combined_output(output: &std::process::Output) -> String {
    let mut bytes = output.stdout.clone();
    bytes.extend_from_slice(&output.stderr);
    bounded_text(&bytes).0
}

fn ensure_docker(
    output: std::process::Output,
    operation: &'static str,
) -> Result<String, VibeError> {
    if output.status.success() {
        return Ok(String::from_utf8_lossy(&output.stdout).into_owned());
    }
    // Docker often includes host mount paths and container ids in stderr. Keep
    // those details in neither model-facing results nor ordinary errors.
    Err(VibeError::Sandbox {
        operation,
        message: "Docker rejected the sandbox operation; inspect the host Docker logs".into(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn command_output_is_bounded_and_marks_nonzero() {
        #[cfg(unix)]
        {
            use std::os::unix::process::ExitStatusExt;
            let output = std::process::Output {
                status: std::process::ExitStatus::from_raw(7 << 8),
                stdout: vec![b'x'; MAX_COMMAND_OUTPUT + 10],
                stderr: b"failed".to_vec(),
            };
            let result = command_output(output);
            assert_eq!(result.exit_code, Some(7));
            assert!(result.truncated);
            assert!(result.stdout.ends_with("[output truncated]"));
            assert_eq!(result.stderr, "failed");
        }
    }

    #[test]
    fn container_arguments_include_every_isolation_limit() {
        let sandbox = VibeSandbox::new(VibeSandboxConfig::default());
        let command = sandbox.base_create("chudbot-vibe-test-coding", Path::new("/tmp/workspace"));
        let args = command
            .as_std()
            .get_args()
            .map(|arg| arg.to_string_lossy().into_owned())
            .collect::<Vec<_>>();
        for required in [
            "--read-only",
            "--cap-drop",
            "ALL",
            VIBE_DOCKER_NETWORK,
            "no-new-privileges:true",
            "--pids-limit",
            "256",
            "--cpus",
            "2",
            "--memory",
            "1024m",
        ] {
            assert!(args.iter().any(|arg| arg == required), "{required}");
        }
        let binds = args
            .iter()
            .filter(|arg| arg.starts_with("type=bind,"))
            .collect::<Vec<_>>();
        assert_eq!(binds.len(), 1);
        assert!(
            !binds
                .iter()
                .any(|arg| arg.contains("docker.sock") || arg.contains("config.toml"))
        );
    }

    #[test]
    fn docker_failures_hide_ids_and_host_paths() {
        #[cfg(unix)]
        {
            use std::os::unix::process::ExitStatusExt;
            let output = std::process::Output {
                status: std::process::ExitStatus::from_raw(1 << 8),
                stdout: Vec::new(),
                stderr: b"container abc at /private/host/workspace".to_vec(),
            };
            let error = ensure_docker(output, "test").unwrap_err().to_string();
            assert!(!error.contains("abc"));
            assert!(!error.contains("/private/host"));
        }
    }
}
