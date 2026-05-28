//! Anthropic Claude provider. Talks to `https://api.anthropic.com/v1/messages`.
//! Supports:
//!   - server-side web search via the built-in `web_search_20250305` tool;
//!   - client-side tools declared in the request `tools` array, with
//!     `tool_use` / `tool_result` blocks for round-trips.

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::HashMap;

use crate::config::AnthropicConfig;
use crate::llm::{
    ChatTurn, LlmError, LlmProvider, MessageRole, StepRequest, StepResponse, ToolCallRecord,
    ToolUseRequest, TurnBlock,
};

const DEFAULT_BASE_URL: &str = "https://api.anthropic.com/v1";
const API_VERSION: &str = "2023-06-01";
const WEB_SEARCH_TOOL_TYPE: &str = "web_search_20250305";
const WEB_SEARCH_TOOL_NAME: &str = "web_search";

/// Anthropic Claude provider. Model-agnostic — the specific model id
/// is supplied per request via [`StepRequest::model`].
#[derive(Debug, Clone)]
pub struct AnthropicProvider {
    http: reqwest::Client,
    api_key: String,
    base_url: String,
}

impl AnthropicProvider {
    /// Construct from a config block.
    pub fn new(config: AnthropicConfig) -> Self {
        Self {
            http: reqwest::Client::new(),
            api_key: config.api_key,
            base_url: DEFAULT_BASE_URL.to_string(),
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
        "anthropic"
    }

    async fn step(&self, request: StepRequest) -> Result<StepResponse, LlmError> {
        let (system, mut anthropic_messages) = to_anthropic_messages(&request.messages);

        let mut tools: Vec<Value> = request
            .tools
            .iter()
            .map(|t| {
                json!({
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.input_schema,
                })
            })
            .collect();
        if request.enable_web_search {
            tools.push(json!({
                "type": WEB_SEARCH_TOOL_TYPE,
                "name": WEB_SEARCH_TOOL_NAME,
                "max_uses": 5,
            }));
        }

        // Prompt caching. Two ephemeral (5-minute) breakpoints over the
        // request prefix, which Anthropic hashes in the order
        // tools → system → messages:
        //   1. on the system prompt, anchoring the most stable prefix
        //      (tools + system) — it never changes within a conversation;
        //   2. on the final message block, extending the cache to cover
        //      the whole conversation history (text AND image blocks).
        // Our agent loop re-sends the entire prefix on every tool-use
        // iteration, and every later turn re-sends all prior turns +
        // their replayed images. With these breakpoints the second and
        // subsequent sends bill the matched prefix at 0.1x instead of
        // full input price — the single biggest lever on image cost.
        // Below the model's minimum cacheable size a breakpoint is
        // silently ignored, so this is safe even for tiny prompts.
        let system_field = system.map(|s| {
            json!([{
                "type": "text",
                "text": s,
                "cache_control": { "type": "ephemeral" },
            }])
        });
        mark_last_block_ephemeral(&mut anthropic_messages);

        let body = AnthropicRequest {
            model: &request.model,
            max_tokens: request.max_tokens,
            messages: &anthropic_messages,
            system: system_field,
            tools: &tools,
            temperature: request.temperature,
            top_p: request.top_p,
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

        let model_id = parsed.model.unwrap_or_else(|| request.model.clone());
        let stop = parsed.stop_reason.unwrap_or_default();
        let (text, client_uses, server_tool_calls) = walk_blocks(&parsed.content);

        if stop == "tool_use" && !client_uses.is_empty() {
            Ok(StepResponse::UseTools {
                partial_text: if text.is_empty() { None } else { Some(text) },
                tool_uses: client_uses,
                server_tool_calls,
                model_id,
                // Anthropic continuation (extended-thinking blocks) is not
                // captured yet; its caching uses cache_control breakpoints.
                provider_state: None,
            })
        } else {
            Ok(StepResponse::Final {
                content: text,
                server_tool_calls,
                model_id,
                provider_state: None,
            })
        }
    }
}

/// Walk Anthropic content blocks. Returns (concatenated text, client
/// tool_use requests, server-side tool call records).
///
/// Server-side calls (`server_tool_use` + `web_search_tool_result`) are
/// paired by `tool_use_id` and collapsed into one [`ToolCallRecord`] each.
/// Client-side `tool_use` blocks are surfaced as [`ToolUseRequest`]s
/// that the agent loop will dispatch.
fn walk_blocks(blocks: &[Value]) -> (String, Vec<ToolUseRequest>, Vec<ToolCallRecord>) {
    let mut text = String::new();
    let mut pending_server_uses: HashMap<String, (String, Value)> = HashMap::new();
    let mut server_calls: Vec<ToolCallRecord> = Vec::new();
    let mut client_uses: Vec<ToolUseRequest> = Vec::new();

    for block in blocks {
        let kind = block.get("type").and_then(Value::as_str).unwrap_or("");
        match kind {
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
                pending_server_uses.insert(id, (name, input));
            }
            "web_search_tool_result" => {
                let id = block
                    .get("tool_use_id")
                    .and_then(Value::as_str)
                    .unwrap_or("");
                let response = block.get("content").cloned().unwrap_or(Value::Null);
                if let Some((name, request)) = pending_server_uses.remove(id) {
                    server_calls.push(ToolCallRecord {
                        tool_name: name,
                        request,
                        response,
                    });
                } else {
                    server_calls.push(ToolCallRecord {
                        tool_name: "web_search".to_string(),
                        request: json!({ "tool_use_id": id }),
                        response,
                    });
                }
            }
            "tool_use" => {
                let id = block
                    .get("id")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string();
                let name = block
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string();
                let input = block.get("input").cloned().unwrap_or(Value::Null);
                client_uses.push(ToolUseRequest { id, name, input });
            }
            _ => {}
        }
    }

    // Server tool uses without a matching result are still recorded so
    // the trace is complete.
    for (_, (name, request)) in pending_server_uses {
        server_calls.push(ToolCallRecord {
            tool_name: name,
            request,
            response: Value::Null,
        });
    }

    (text, client_uses, server_calls)
}

/// Convert our [`ChatTurn`]s into Anthropic's (system, messages) pair.
/// System messages are lifted out into the top-level `system` field.
fn to_anthropic_messages(turns: &[ChatTurn]) -> (Option<String>, Vec<Value>) {
    let mut system_parts: Vec<String> = Vec::new();
    let mut messages: Vec<Value> = Vec::new();

    for turn in turns {
        // System turns get concatenated and lifted out.
        if turn.role == MessageRole::System {
            for block in &turn.blocks {
                if let TurnBlock::Text(t) = block {
                    system_parts.push(t.clone());
                }
            }
            continue;
        }

        let role = if turn.role == MessageRole::Assistant {
            "assistant"
        } else {
            "user"
        };

        let mut content_blocks: Vec<Value> = Vec::new();
        for block in &turn.blocks {
            match block {
                TurnBlock::Text(t) if !t.is_empty() => {
                    content_blocks.push(json!({ "type": "text", "text": t }));
                }
                TurnBlock::Text(_) => {}
                TurnBlock::Image { url, .. } => {
                    content_blocks.push(json!({
                        "type": "image",
                        "source": { "type": "url", "url": url },
                    }));
                }
                TurnBlock::ToolUse { id, name, input } => {
                    content_blocks.push(json!({
                        "type": "tool_use",
                        "id": id,
                        "name": name,
                        "input": input,
                    }));
                }
                TurnBlock::ToolResult {
                    tool_use_id,
                    content,
                    is_error,
                } => {
                    let mut obj = serde_json::Map::new();
                    obj.insert("type".into(), Value::String("tool_result".into()));
                    obj.insert("tool_use_id".into(), Value::String(tool_use_id.clone()));
                    obj.insert("content".into(), Value::String(content.clone()));
                    if *is_error {
                        obj.insert("is_error".into(), Value::Bool(true));
                    }
                    content_blocks.push(Value::Object(obj));
                }
                // Another provider's opaque reasoning (xAI's encrypted
                // items, replayed across a persona switch). Not Anthropic's
                // format, so drop it — it isn't ours to send.
                TurnBlock::Reasoning { .. } => {}
            }
        }

        if content_blocks.is_empty() {
            continue;
        }
        messages.push(json!({ "role": role, "content": content_blocks }));
    }

    let system = if system_parts.is_empty() {
        None
    } else {
        Some(system_parts.join("\n\n"))
    };
    (system, messages)
}

#[derive(Serialize)]
struct AnthropicRequest<'a> {
    model: &'a str,
    max_tokens: u32,
    messages: &'a [Value],
    // A content-block array (`[{type:text,…,cache_control:…}]`) rather
    // than a bare string, so the system prompt can carry a cache
    // breakpoint. Anthropic accepts either form.
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<Value>,
    #[serde(skip_serializing_if = "<[Value]>::is_empty")]
    tools: &'a [Value],
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
}

/// Tag the last content block of the last message with an ephemeral
/// cache breakpoint, so the entire conversation prefix up to it becomes
/// cacheable. No-op if there are no messages or the last message has no
/// block array (e.g. a string-content message we never emit). Any block
/// type — text, image, or tool_result — is a valid breakpoint anchor.
fn mark_last_block_ephemeral(messages: &mut [Value]) {
    if let Some(last) = messages.last_mut()
        && let Some(content) = last.get_mut("content").and_then(Value::as_array_mut)
        && let Some(block) = content.last_mut()
        && let Some(obj) = block.as_object_mut()
    {
        obj.insert("cache_control".into(), json!({ "type": "ephemeral" }));
    }
}

#[derive(Deserialize)]
struct AnthropicResponse {
    content: Vec<Value>,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    stop_reason: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pairs_server_tool_use_with_web_search_result() {
        let blocks = vec![
            json!({"type": "text", "text": "Looking that up. "}),
            json!({
                "type": "server_tool_use",
                "id": "srvtoolu_1",
                "name": "web_search",
                "input": {"query": "rust 2024 edition"},
            }),
            json!({
                "type": "web_search_tool_result",
                "tool_use_id": "srvtoolu_1",
                "content": [{"type": "web_search_result", "url": "https://x", "title": "y"}],
            }),
            json!({"type": "text", "text": "Done."}),
        ];

        let (text, client_uses, server_calls) = walk_blocks(&blocks);
        assert_eq!(text, "Looking that up. Done.");
        assert!(client_uses.is_empty());
        assert_eq!(server_calls.len(), 1);
        assert_eq!(server_calls[0].tool_name, "web_search");
    }

    #[test]
    fn surfaces_client_tool_use_for_agent_loop() {
        let blocks = vec![
            json!({"type": "text", "text": "Let me fetch recent messages."}),
            json!({
                "type": "tool_use",
                "id": "toolu_1",
                "name": "fetch_messages",
                "input": {"limit": 30},
            }),
        ];

        let (text, client_uses, server_calls) = walk_blocks(&blocks);
        assert_eq!(text, "Let me fetch recent messages.");
        assert!(server_calls.is_empty());
        assert_eq!(client_uses.len(), 1);
        assert_eq!(client_uses[0].name, "fetch_messages");
        assert_eq!(client_uses[0].input["limit"], 30);
    }

    #[test]
    fn cache_breakpoint_lands_on_final_block() {
        let mut messages = vec![
            json!({"role": "user", "content": [{"type": "text", "text": "hi"}]}),
            json!({"role": "assistant", "content": [{"type": "text", "text": "hello"}]}),
            json!({"role": "user", "content": [
                {"type": "text", "text": "look"},
                {"type": "image", "source": {"type": "url", "url": "https://x/y.png"}},
            ]}),
        ];
        mark_last_block_ephemeral(&mut messages);

        // Breakpoint sits on the very last block (the image) of the last
        // message, and nowhere else.
        let last = messages.last().unwrap()["content"].as_array().unwrap();
        assert_eq!(last[1]["cache_control"], json!({ "type": "ephemeral" }));
        assert!(last[0].get("cache_control").is_none());
        assert!(messages[0]["content"][0].get("cache_control").is_none());
    }

    #[test]
    fn cache_breakpoint_no_op_on_empty() {
        let mut messages: Vec<Value> = vec![];
        mark_last_block_ephemeral(&mut messages); // must not panic
        assert!(messages.is_empty());
    }
}
