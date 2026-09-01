use std::path::{Component, Path, PathBuf};
use std::sync::Arc;

use base64::Engine;
use chudbot_api::{
    ClientToolCall, ClientToolDefinition, ClientToolExecutor, ClientToolExecutorError,
    ClientToolOutput, ClientToolResultContent, ClientToolSpec, MediaCategory, ToolInputField,
    ToolInputSchema, ToolInputValueSchema, UrlMediaRef,
};
use serde_json::json;
use thiserror::Error;
use tokio::sync::Mutex;

use crate::CodingContainer;

const READ_TOOL: &str = "read";
const EDIT_TOOL: &str = "edit";
const SHELL_TOOL: &str = "shell";
const MAX_READ_BYTES: u64 = 2 * 1024 * 1024;

#[derive(Debug, Error)]
pub enum CodingToolError {
    #[error("invalid workspace path: {0}")]
    InvalidPath(String),
    #[error("workspace I/O failed: {0}")]
    Io(#[from] std::io::Error),
    #[error("invalid tool input: {0}")]
    Input(String),
    #[error("sandbox failed: {0}")]
    Sandbox(#[from] crate::VibeError),
}

#[derive(Debug, Clone)]
pub struct VibeCodingExecutor {
    workspace: PathBuf,
    container: Option<Arc<Mutex<CodingContainer>>>,
}

impl VibeCodingExecutor {
    pub fn new(workspace: PathBuf, container: CodingContainer) -> Self {
        Self {
            workspace,
            container: Some(Arc::new(Mutex::new(container))),
        }
    }

    pub fn tool_names() -> [&'static str; 3] {
        [READ_TOOL, EDIT_TOOL, SHELL_TOOL]
    }

    async fn read(&self, call: ClientToolCall) -> Result<ClientToolOutput, CodingToolError> {
        let relative = input_string(&call, "path")?;
        let path = workspace_path(&self.workspace, relative)?;
        let metadata = tokio::fs::metadata(&path).await?;
        if metadata.is_dir() {
            let mut entries = tokio::fs::read_dir(&path).await?;
            let mut names = Vec::new();
            while let Some(entry) = entries.next_entry().await? {
                let name = entry.file_name().to_string_lossy().into_owned();
                if !matches!(name.as_str(), ".git" | "node_modules" | "dist") {
                    names.push(name);
                }
            }
            names.sort();
            return Ok(output(json!({"entries":names}), false));
        }
        if metadata.len() > MAX_READ_BYTES {
            return Err(CodingToolError::Input(
                "file is too large for read; use shell with a bounded command".into(),
            ));
        }
        if let Some(mime) = image_mime(&path) {
            let bytes = tokio::fs::read(&path).await?;
            let data = format!(
                "data:{mime};base64,{}",
                base64::engine::general_purpose::STANDARD.encode(bytes)
            );
            let mut result = output(json!({"path":relative,"kind":"image"}), false);
            result
                .media
                .push(UrlMediaRef::new(MediaCategory::Image, data, mime).boxed());
            return Ok(result);
        }
        let text = tokio::fs::read_to_string(&path)
            .await
            .map_err(|_| CodingToolError::Input("file is not UTF-8 text".into()))?;
        let start = call
            .input
            .get("startLine")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(1)
            .max(1) as usize;
        let end = call
            .input
            .get("endLine")
            .and_then(serde_json::Value::as_u64)
            .map(|value| value as usize)
            .unwrap_or_else(|| start.saturating_add(399));
        if end < start {
            return Err(CodingToolError::Input(
                "endLine must be at least startLine".into(),
            ));
        }
        let numbered = text
            .lines()
            .enumerate()
            .filter(|(index, _)| (*index + 1) >= start && (*index + 1) <= end)
            .map(|(index, line)| format!("{:>6} | {line}", index + 1))
            .collect::<Vec<_>>()
            .join("\n");
        Ok(output(
            json!({"path":relative,"startLine":start,"endLine":end,"text":numbered}),
            false,
        ))
    }

    async fn edit(&self, call: ClientToolCall) -> Result<ClientToolOutput, CodingToolError> {
        let relative = input_string(&call, "path")?;
        let path = workspace_path(&self.workspace, relative)?;
        match input_string(&call, "operation")? {
            "create" => {
                if tokio::fs::try_exists(&path).await? {
                    return Err(CodingToolError::Input(
                        "file already exists; reread it and use replace".into(),
                    ));
                }
                let content = input_string(&call, "content")?;
                if let Some(parent) = path.parent() {
                    tokio::fs::create_dir_all(parent).await?;
                }
                tokio::fs::write(&path, content).await?;
            }
            "delete" => {
                if !tokio::fs::try_exists(&path).await? {
                    return Err(CodingToolError::Input(
                        "file does not exist; reread the directory".into(),
                    ));
                }
                tokio::fs::remove_file(&path).await?;
            }
            "replace" => {
                let old = input_string(&call, "old")?;
                let new = input_string(&call, "new")?;
                if old.is_empty() {
                    return Err(CodingToolError::Input("old must not be empty".into()));
                }
                let text = tokio::fs::read_to_string(&path).await?;
                let count = text.match_indices(old).count();
                if count != 1 {
                    return Err(CodingToolError::Input(format!(
                        "replacement matched {count} times; reread the file and provide one exact unique string"
                    )));
                }
                tokio::fs::write(&path, text.replacen(old, new, 1)).await?;
            }
            other => {
                return Err(CodingToolError::Input(format!(
                    "unknown edit operation `{other}`"
                )));
            }
        }
        Ok(output(json!({"ok":true,"path":relative}), false))
    }

    async fn shell(&self, call: ClientToolCall) -> Result<ClientToolOutput, CodingToolError> {
        let command = input_string(&call, "command")?;
        let Some(container) = self.container.as_ref() else {
            return Err(CodingToolError::Input(
                "shell is unavailable in this test executor".into(),
            ));
        };
        let result = container.lock().await.shell(command).await?;
        let is_error = result.timed_out || result.exit_code != Some(0);
        Ok(output(
            json!({"stdout":result.stdout,"stderr":result.stderr,"exitCode":result.exit_code,"timedOut":result.timed_out,"truncated":result.truncated}),
            is_error,
        ))
    }
}

impl ClientToolExecutor for VibeCodingExecutor {
    type Error = CodingToolError;

    async fn execute(
        &self,
        call: ClientToolCall,
    ) -> Result<ClientToolOutput, ClientToolExecutorError<Self::Error>> {
        let result = match call.name.as_str() {
            READ_TOOL => self.read(call).await,
            EDIT_TOOL => self.edit(call).await,
            SHELL_TOOL => self.shell(call).await,
            _ => return Err(ClientToolExecutorError::unknown(call.name)),
        };
        result.map_err(ClientToolExecutorError::execution)
    }

    fn tools(&self) -> Vec<ClientToolDefinition> {
        vec![
            ClientToolDefinition::new(READ_TOOL, read_spec()),
            ClientToolDefinition::new(EDIT_TOOL, edit_spec()),
            ClientToolDefinition::new(SHELL_TOOL, shell_spec()),
        ]
    }
}

fn read_spec() -> ClientToolSpec {
    ClientToolSpec {
        description: "Read a UTF-8 file with numbered lines, inspect an image, or list a directory inside /workspace.".into(),
        input_schema: ToolInputSchema::object([
            ToolInputField::required("path", ToolInputValueSchema::string()),
            ToolInputField::optional("startLine", ToolInputValueSchema::integer().minimum(1)),
            ToolInputField::optional("endLine", ToolInputValueSchema::integer().minimum(1)),
        ]),
    }
}
fn edit_spec() -> ClientToolSpec {
    ClientToolSpec {
        description: "Create or delete a file, or replace one exact unique string. Reread after a missing or ambiguous match.".into(),
        input_schema: ToolInputSchema::object([
            ToolInputField::required("path", ToolInputValueSchema::string()),
            ToolInputField::required(
                "operation",
                ToolInputValueSchema::string().enum_values(["create", "delete", "replace"]),
            ),
            ToolInputField::optional("content", ToolInputValueSchema::string()),
            ToolInputField::optional("old", ToolInputValueSchema::string()),
            ToolInputField::optional("new", ToolInputValueSchema::string()),
        ]),
    }
}
fn shell_spec() -> ClientToolSpec {
    ClientToolSpec {
        description: "Run one non-interactive /bin/bash -lc command inside the coding container. Output and runtime are bounded.".into(),
        input_schema: ToolInputSchema::object([ToolInputField::required(
            "command",
            ToolInputValueSchema::string(),
        )]),
    }
}
fn output(value: serde_json::Value, is_error: bool) -> ClientToolOutput {
    ClientToolOutput {
        result: ClientToolResultContent::Json {
            value: value.clone(),
        },
        media: Vec::new(),
        is_error,
        trace_response: value,
        usage: Vec::new(),
    }
}
fn input_string<'a>(call: &'a ClientToolCall, name: &str) -> Result<&'a str, CodingToolError> {
    call.input
        .get(name)
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| CodingToolError::Input(format!("`{name}` must be a string")))
}
fn workspace_path(root: &Path, value: &str) -> Result<PathBuf, CodingToolError> {
    if value.is_empty() || value.contains('\\') || value.contains('\0') {
        return Err(CodingToolError::InvalidPath(value.into()));
    }
    let mut path = root.to_path_buf();
    for component in Path::new(value.trim_start_matches('/')).components() {
        match component {
            Component::Normal(part)
                if !matches!(part.to_str(), Some(".git" | "node_modules" | "dist")) =>
            {
                path.push(part);
            }
            _ => return Err(CodingToolError::InvalidPath(value.into())),
        }
    }
    Ok(path)
}
fn image_mime(path: &Path) -> Option<&'static str> {
    match path.extension()?.to_str()?.to_ascii_lowercase().as_str() {
        "png" => Some("image/png"),
        "jpg" | "jpeg" => Some("image/jpeg"),
        "gif" => Some("image/gif"),
        "webp" => Some("image/webp"),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chudbot_api::{ToolName, ToolUseId};
    use uuid::Uuid;

    fn executor(root: PathBuf) -> VibeCodingExecutor {
        VibeCodingExecutor {
            workspace: root,
            container: None,
        }
    }
    fn call(name: &str, input: serde_json::Value) -> ClientToolCall {
        ClientToolCall {
            id: ToolUseId::new("test"),
            name: ToolName::new(name),
            input,
        }
    }

    #[test]
    fn coding_tool_surface_is_exact() {
        assert_eq!(VibeCodingExecutor::tool_names(), ["read", "edit", "shell"]);
    }

    #[test]
    fn workspace_paths_reject_escape_and_reserved() {
        let root = Path::new("/workspace");
        assert!(workspace_path(root, "src/App.tsx").is_ok());
        for path in ["../x", "node_modules/x", "dist/x", ".git/config", "a\\b"] {
            assert!(workspace_path(root, path).is_err(), "{path}");
        }
    }

    #[tokio::test]
    async fn read_lists_directories_and_returns_numbered_ranges() {
        let root = std::env::temp_dir().join(format!("vibe-coding-read-{}", Uuid::new_v4()));
        tokio::fs::create_dir_all(root.join("src")).await.unwrap();
        tokio::fs::write(root.join("src/App.tsx"), "one\ntwo\nthree\n")
            .await
            .unwrap();
        let executor = executor(root.clone());
        let listing = executor
            .read(call("read", json!({"path":"src"})))
            .await
            .unwrap();
        assert!(
            listing.trace_response["entries"]
                .as_array()
                .unwrap()
                .iter()
                .any(|value| value == "App.tsx")
        );
        let range = executor
            .read(call(
                "read",
                json!({"path":"src/App.tsx","startLine":2,"endLine":3}),
            ))
            .await
            .unwrap();
        assert_eq!(range.trace_response["text"], "     2 | two\n     3 | three");
        let _ = tokio::fs::remove_dir_all(root).await;
    }

    #[tokio::test]
    async fn edit_creates_replaces_deletes_and_reports_ambiguous_matches() {
        let root = std::env::temp_dir().join(format!("vibe-coding-edit-{}", Uuid::new_v4()));
        tokio::fs::create_dir_all(&root).await.unwrap();
        let executor = executor(root.clone());
        executor
            .edit(call(
                "edit",
                json!({"path":"src/a.txt","operation":"create","content":"old old"}),
            ))
            .await
            .unwrap();
        let error = executor
            .edit(call(
                "edit",
                json!({"path":"src/a.txt","operation":"replace","old":"old","new":"new"}),
            ))
            .await
            .unwrap_err();
        assert!(error.to_string().contains("matched 2 times"));
        executor
            .edit(call(
                "edit",
                json!({"path":"src/a.txt","operation":"replace","old":"old old","new":"new"}),
            ))
            .await
            .unwrap();
        assert_eq!(
            tokio::fs::read_to_string(root.join("src/a.txt"))
                .await
                .unwrap(),
            "new"
        );
        executor
            .edit(call(
                "edit",
                json!({"path":"src/a.txt","operation":"delete"}),
            ))
            .await
            .unwrap();
        assert!(!tokio::fs::try_exists(root.join("src/a.txt")).await.unwrap());
        let _ = tokio::fs::remove_dir_all(root).await;
    }
}
