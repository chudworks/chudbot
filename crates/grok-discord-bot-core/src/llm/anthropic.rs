//! Anthropic Claude provider. Talks to `https://api.anthropic.com/v1/messages`,
//! with the server-side `web_search_20250305` tool enabled when
//! [`CompletionRequest::enable_web_search`] is set.
//!
//! Anthropic's API differs from OpenAI's in two relevant ways:
//! - The system prompt is a top-level `system` field, not a message with
//!   `role: system`. We lift system messages out of the chat history.
//! - The response `content` is an array of blocks
//!   (`text` / `server_tool_use` / `web_search_tool_result` / …) rather
//!   than a single string. We concatenate text blocks and pair server
//!   tool use blocks with their results into [`ToolCallRecord`]s.

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::HashMap;

use crate::config::AnthropicConfig;
use crate::llm::{
    CompletionRequest, CompletionResponse, LlmError, LlmProvider, MessageRole, ToolCallRecord,
};

const DEFAULT_BASE_URL: &str = "https://api.anthropic.com/v1";
const API_VERSION: &str = "2023-06-01";
const WEB_SEARCH_TOOL_TYPE: &str = "web_search_20250305";
const WEB_SEARCH_TOOL_NAME: &str = "web_search";

/// Anthropic Claude provider.
#[derive(Debug, Clone)]
pub struct AnthropicProvider {
    http: reqwest::Client,
    api_key: String,
    model: String,
    base_url: String,
    name: String,
}

impl AnthropicProvider {
    /// Construct from a config block. Uses the default
    /// `api.anthropic.com/v1` base URL.
    pub fn new(config: AnthropicConfig) -> Self {
        let name = format!("anthropic/{}", config.model);
        Self {
            http: reqwest::Client::new(),
            api_key: config.api_key,
            model: config.model,
            base_url: DEFAULT_BASE_URL.to_string(),
            name,
        }
    }

    /// Override the base URL. Used by tests.
    pub fn with_base_url(mut self, base_url: String) -> Self {
        self.base_url = base_url;
        self
    }
}

impl LlmProvider for AnthropicProvider {
    fn name(&self) -> &str {
        &self.name
    }

    async fn complete(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse, LlmError> {
        // Separate system messages from chat messages.
        let mut system_parts: Vec<&str> = Vec::new();
        let mut chat: Vec<AnthropicChatMessage<'_>> = Vec::new();
        for msg in &request.messages {
            match msg.role {
                MessageRole::System => system_parts.push(&msg.content),
                MessageRole::User => chat.push(AnthropicChatMessage {
                    role: "user",
                    content: &msg.content,
                }),
                MessageRole::Assistant => chat.push(AnthropicChatMessage {
                    role: "assistant",
                    content: &msg.content,
                }),
            }
        }
        let system = (!system_parts.is_empty()).then(|| system_parts.join("\n\n"));

        let tools: Vec<Value> = if request.enable_web_search {
            vec![json!({
                "type": WEB_SEARCH_TOOL_TYPE,
                "name": WEB_SEARCH_TOOL_NAME,
                "max_uses": 5,
            })]
        } else {
            Vec::new()
        };

        let body = AnthropicRequest {
            model: &self.model,
            max_tokens: request.max_tokens,
            messages: &chat,
            system: system.as_deref(),
            tools: &tools,
        };

        let resp = self
            .http
            .post(format!("{}/messages", self.base_url))
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", API_VERSION)
            .json(&body)
            .send()
            .await
            .map_err(|e| LlmError::Transport(e.to_string()))?;

        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(LlmError::Api {
                status: status.as_u16(),
                body,
            });
        }

        let parsed: AnthropicResponse = resp
            .json()
            .await
            .map_err(|e| LlmError::Decode(e.to_string()))?;

        let (content, tool_calls) = extract_content_and_tool_calls(&parsed.content);

        Ok(CompletionResponse {
            content,
            tool_calls,
            model_id: parsed.model.unwrap_or_else(|| self.model.clone()),
        })
    }
}

/// Walk the `content` blocks: concatenate text into the final answer and
/// pair each `server_tool_use` block with its matching
/// `web_search_tool_result` (by `tool_use_id`) into a [`ToolCallRecord`].
fn extract_content_and_tool_calls(blocks: &[Value]) -> (String, Vec<ToolCallRecord>) {
    let mut text = String::new();
    let mut pending_tool_uses: HashMap<String, (String, Value)> = HashMap::new();
    let mut tool_calls: Vec<ToolCallRecord> = Vec::new();

    for block in blocks {
        let block_type = block.get("type").and_then(Value::as_str).unwrap_or("");
        match block_type {
            "text" => {
                if let Some(t) = block.get("text").and_then(Value::as_str) {
                    text.push_str(t);
                }
            }
            "server_tool_use" => {
                let id = block
                    .get("id")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string();
                let name = block
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or("server_tool")
                    .to_string();
                let input = block.get("input").cloned().unwrap_or(Value::Null);
                pending_tool_uses.insert(id, (name, input));
            }
            "web_search_tool_result" => {
                let id = block
                    .get("tool_use_id")
                    .and_then(Value::as_str)
                    .unwrap_or("");
                let response = block.get("content").cloned().unwrap_or(Value::Null);
                if let Some((name, request)) = pending_tool_uses.remove(id) {
                    tool_calls.push(ToolCallRecord {
                        tool_name: name,
                        request,
                        response,
                    });
                } else {
                    tool_calls.push(ToolCallRecord {
                        tool_name: "web_search".to_string(),
                        request: json!({ "tool_use_id": id }),
                        response,
                    });
                }
            }
            _ => {}
        }
    }

    // Any server_tool_use without a matching result (rare, e.g. errors)
    // is still recorded so the trace is complete.
    for (_, (name, request)) in pending_tool_uses {
        tool_calls.push(ToolCallRecord {
            tool_name: name,
            request,
            response: Value::Null,
        });
    }

    (text, tool_calls)
}

#[derive(Serialize)]
struct AnthropicRequest<'a> {
    model: &'a str,
    max_tokens: u32,
    messages: &'a [AnthropicChatMessage<'a>],
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<&'a str>,
    #[serde(skip_serializing_if = "<[Value]>::is_empty")]
    tools: &'a [Value],
}

#[derive(Serialize)]
struct AnthropicChatMessage<'a> {
    role: &'a str,
    content: &'a str,
}

#[derive(Deserialize)]
struct AnthropicResponse {
    content: Vec<Value>,
    #[serde(default)]
    model: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_text_and_pairs_tool_uses_with_results() {
        let blocks = vec![
            json!({"type": "text", "text": "I'll search for that. "}),
            json!({
                "type": "server_tool_use",
                "id": "srvtoolu_1",
                "name": "web_search",
                "input": {"query": "rust 2024 edition"},
            }),
            json!({
                "type": "web_search_tool_result",
                "tool_use_id": "srvtoolu_1",
                "content": [{"type": "web_search_result", "url": "https://blog.rust-lang.org", "title": "Rust 2024"}],
            }),
            json!({"type": "text", "text": "Rust 2024 was announced in late 2024."}),
        ];

        let (text, tool_calls) = extract_content_and_tool_calls(&blocks);
        assert_eq!(text, "I'll search for that. Rust 2024 was announced in late 2024.");
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].tool_name, "web_search");
        assert_eq!(tool_calls[0].request["query"], "rust 2024 edition");
    }
}
