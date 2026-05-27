//! xAI Grok provider, talking to the **Agent Tools / Responses API** at
//! `POST https://api.x.ai/v1/responses`.
//!
//! This is the modern xAI endpoint, replacing the older
//! `/v1/chat/completions` + `search_parameters` path which now returns
//! `410 Live search is deprecated`. The Responses API uses an `input`
//! array of items (instead of `messages`), an `output` array of typed
//! blocks (instead of `choices[0].message`), and represents both
//! server-side and client-side tool calls as top-level output items.
//!
//! Server-side tools we enable on `enable_web_search`:
//!   - `web_search` — general web search with citations.
//!   - `x_search`   — X / Twitter search; Grok's distinctive grounding
//!                    surface, included for free alongside web_search.

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::config::XaiConfig;
use crate::llm::{
    ChatTurn, LlmError, LlmProvider, MessageRole, StepRequest, StepResponse, ToolCallRecord,
    ToolDefinition, ToolUseRequest, TurnBlock,
};

const DEFAULT_BASE_URL: &str = "https://api.x.ai/v1";

/// xAI Grok provider.
#[derive(Debug, Clone)]
pub struct XaiProvider {
    http: reqwest::Client,
    api_key: String,
    model: String,
    base_url: String,
    name: String,
}

impl XaiProvider {
    /// Construct from a config block.
    pub fn new(config: XaiConfig) -> Self {
        let name = format!("xai/{}", config.model);
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

impl LlmProvider for XaiProvider {
    fn name(&self) -> &str {
        &self.name
    }

    async fn step(&self, request: StepRequest) -> Result<StepResponse, LlmError> {
        let (instructions, input_items) = to_responses_input(&request.messages);
        let tools = build_tools(&request.tools, request.enable_web_search);

        let body = ResponsesRequest {
            model: &self.model,
            input: &input_items,
            instructions: instructions.as_deref(),
            tools: if tools.is_empty() { None } else { Some(&tools) },
            max_output_tokens: Some(request.max_tokens),
            temperature: request.temperature,
            top_p: request.top_p,
        };

        let resp = self
            .http
            .post(format!("{}/responses", self.base_url))
            .bearer_auth(&self.api_key)
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

        let parsed: ResponsesResponse = resp
            .json()
            .await
            .map_err(|e| LlmError::Decode(e.to_string()))?;

        let model_id = parsed.model.unwrap_or_else(|| self.model.clone());
        let (text, tool_uses, server_tool_calls) =
            walk_output(&parsed.output, parsed.citations.as_ref());

        if !tool_uses.is_empty() {
            Ok(StepResponse::UseTools {
                partial_text: if text.is_empty() { None } else { Some(text) },
                tool_uses,
                server_tool_calls,
                model_id,
            })
        } else {
            Ok(StepResponse::Final {
                content: text,
                server_tool_calls,
                model_id,
            })
        }
    }
}

/// Convert our [`ChatTurn`] history into the Responses API's
/// `(instructions, input)` pair. System turns are lifted out of the
/// input list and concatenated into the top-level `instructions` field.
fn to_responses_input(turns: &[ChatTurn]) -> (Option<String>, Vec<Value>) {
    let mut instructions: Vec<String> = Vec::new();
    let mut input: Vec<Value> = Vec::new();

    for turn in turns {
        if turn.role == MessageRole::System {
            for block in &turn.blocks {
                if let TurnBlock::Text(t) = block {
                    instructions.push(t.clone());
                }
            }
            continue;
        }

        let role_str = match turn.role {
            MessageRole::Assistant => "assistant",
            _ => "user",
        };

        let mut text_buf = String::new();
        let mut image_urls: Vec<String> = Vec::new();
        let mut deferred: Vec<Value> = Vec::new();

        for block in &turn.blocks {
            match block {
                TurnBlock::Text(t) => text_buf.push_str(t),
                TurnBlock::Image { url, .. } => image_urls.push(url.clone()),
                TurnBlock::ToolUse { id, name, input: tool_input } => {
                    // Echo the assistant's prior tool call back as its own
                    // input item; the Responses API tracks call_id for
                    // matching results.
                    let args = serde_json::to_string(tool_input).unwrap_or_else(|_| "{}".into());
                    deferred.push(json!({
                        "type": "function_call",
                        "call_id": id,
                        "name": name,
                        "arguments": args,
                    }));
                }
                TurnBlock::ToolResult { tool_use_id, content, .. } => {
                    deferred.push(json!({
                        "type": "function_call_output",
                        "call_id": tool_use_id,
                        "output": content,
                    }));
                }
            }
        }

        // Pick the content shape: plain string when text-only, content
        // array when any image is attached. Both forms are valid input
        // for the Responses API.
        if image_urls.is_empty() {
            if !text_buf.is_empty() {
                input.push(json!({
                    "role": role_str,
                    "content": text_buf,
                }));
            }
        } else {
            let mut parts: Vec<Value> = Vec::with_capacity(image_urls.len() + 1);
            if !text_buf.is_empty() {
                parts.push(json!({ "type": "input_text", "text": text_buf }));
            }
            for url in image_urls {
                parts.push(json!({ "type": "input_image", "image_url": url }));
            }
            input.push(json!({
                "role": role_str,
                "content": parts,
            }));
        }
        input.extend(deferred);
    }

    let instructions = if instructions.is_empty() {
        None
    } else {
        Some(instructions.join("\n\n"))
    };
    (instructions, input)
}

/// Build the `tools` array — client-side function definitions plus
/// xAI's server-side `web_search` + `x_search` when enabled.
fn build_tools(defs: &[ToolDefinition], enable_web_search: bool) -> Vec<Value> {
    let mut tools: Vec<Value> = Vec::with_capacity(defs.len() + 2);
    for t in defs {
        tools.push(json!({
            "type": "function",
            "name": t.name,
            "description": t.description,
            "parameters": t.input_schema,
        }));
    }
    if enable_web_search {
        tools.push(json!({ "type": "web_search" }));
        tools.push(json!({ "type": "x_search" }));
    }
    tools
}

/// Walk the `output` array. Returns:
///   - concatenated assistant text;
///   - client-side `function_call` items as [`ToolUseRequest`]s;
///   - server-side tool uses (`web_search_call`, `x_search_call`, etc.)
///     as [`ToolCallRecord`]s, attaching the top-level `citations`
///     field to whichever server tool emitted them when we can't tell
///     them apart (citations are response-wide, not per-block).
fn walk_output(
    output: &[Value],
    citations: Option<&Value>,
) -> (String, Vec<ToolUseRequest>, Vec<ToolCallRecord>) {
    let mut text = String::new();
    let mut tool_uses: Vec<ToolUseRequest> = Vec::new();
    let mut server_calls: Vec<ToolCallRecord> = Vec::new();

    for item in output {
        let kind = item.get("type").and_then(Value::as_str).unwrap_or("");
        match kind {
            "message" => {
                if let Some(content) = item.get("content").and_then(Value::as_array) {
                    for block in content {
                        let block_kind = block.get("type").and_then(Value::as_str).unwrap_or("");
                        if block_kind == "output_text" || block_kind == "text" {
                            if let Some(t) = block.get("text").and_then(Value::as_str) {
                                text.push_str(t);
                            }
                        }
                    }
                } else if let Some(t) = item.get("content").and_then(Value::as_str) {
                    text.push_str(t);
                }
            }
            "function_call" => {
                let call_id = item
                    .get("call_id")
                    .and_then(Value::as_str)
                    .or_else(|| item.get("id").and_then(Value::as_str))
                    .unwrap_or("")
                    .to_string();
                let name = item
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string();
                let args_str = item
                    .get("arguments")
                    .and_then(Value::as_str)
                    .unwrap_or("{}");
                let input: Value = serde_json::from_str(args_str).unwrap_or(Value::Null);
                tool_uses.push(ToolUseRequest {
                    id: call_id,
                    name,
                    input,
                });
            }
            // Server-side tool calls. xAI emits these as top-level items
            // (web_search_call, x_search_call, code_interpreter_call, …).
            // Match anything ending in `_call` and not function_call.
            other if other.ends_with("_call") => {
                let tool_name = other.trim_end_matches("_call").to_string();
                server_calls.push(ToolCallRecord {
                    tool_name,
                    request: item.clone(),
                    response: Value::Null,
                });
            }
            _ => {}
        }
    }

    // Attach response-wide citations to whichever server call could
    // plausibly have produced them (prefer web_search; fall back to the
    // first server call; failing that, record a freestanding entry).
    if let Some(c) = citations {
        if !server_calls.is_empty() {
            if let Some(slot) = server_calls.iter_mut().find(|r| r.tool_name == "web_search") {
                slot.response = c.clone();
            } else {
                server_calls[0].response = c.clone();
            }
        } else {
            server_calls.push(ToolCallRecord {
                tool_name: "web_search".to_string(),
                request: json!({ "implicit": true }),
                response: c.clone(),
            });
        }
    }

    (text, tool_uses, server_calls)
}

#[derive(Serialize)]
struct ResponsesRequest<'a> {
    model: &'a str,
    input: &'a [Value],
    #[serde(skip_serializing_if = "Option::is_none")]
    instructions: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<&'a [Value]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
}

#[derive(Deserialize)]
struct ResponsesResponse {
    #[serde(default)]
    output: Vec<Value>,
    #[serde(default)]
    citations: Option<Value>,
    #[serde(default)]
    model: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lifts_system_into_instructions() {
        let turns = vec![
            ChatTurn::text(MessageRole::System, "be helpful"),
            ChatTurn::text(MessageRole::User, "hi"),
        ];
        let (instructions, input) = to_responses_input(&turns);
        assert_eq!(instructions.as_deref(), Some("be helpful"));
        assert_eq!(input.len(), 1);
        assert_eq!(input[0]["role"], "user");
        assert_eq!(input[0]["content"], "hi");
    }

    #[test]
    fn encodes_function_call_round_trip() {
        let turns = vec![
            ChatTurn::text(MessageRole::User, "fetch please"),
            ChatTurn {
                role: MessageRole::Assistant,
                blocks: vec![
                    TurnBlock::Text("on it. ".into()),
                    TurnBlock::ToolUse {
                        id: "call_1".into(),
                        name: "fetch_messages".into(),
                        input: json!({ "limit": 10 }),
                    },
                ],
            },
            ChatTurn {
                role: MessageRole::User,
                blocks: vec![TurnBlock::ToolResult {
                    tool_use_id: "call_1".into(),
                    content: "[]".into(),
                    is_error: false,
                }],
            },
        ];
        let (_, input) = to_responses_input(&turns);
        // user text, assistant text, function_call, function_call_output
        assert_eq!(input.len(), 4);
        assert_eq!(input[2]["type"], "function_call");
        assert_eq!(input[2]["call_id"], "call_1");
        assert_eq!(input[2]["name"], "fetch_messages");
        assert_eq!(input[3]["type"], "function_call_output");
        assert_eq!(input[3]["call_id"], "call_1");
        assert_eq!(input[3]["output"], "[]");
    }

    #[test]
    fn parses_message_and_function_call_output() {
        let output = vec![
            json!({
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Let me check. "}],
            }),
            json!({
                "type": "function_call",
                "call_id": "call_42",
                "name": "fetch_messages",
                "arguments": "{\"limit\":30}",
            }),
        ];
        let (text, uses, server) = walk_output(&output, None);
        assert_eq!(text, "Let me check. ");
        assert_eq!(uses.len(), 1);
        assert_eq!(uses[0].id, "call_42");
        assert_eq!(uses[0].name, "fetch_messages");
        assert_eq!(uses[0].input["limit"], 30);
        assert!(server.is_empty());
    }

    #[test]
    fn attaches_citations_to_web_search_call() {
        let output = vec![
            json!({"type": "web_search_call", "id": "ws_1", "status": "completed"}),
            json!({
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Found it."}],
            }),
        ];
        let citations = json!([{"url": "https://example.com", "title": "x"}]);
        let (text, uses, server) = walk_output(&output, Some(&citations));
        assert_eq!(text, "Found it.");
        assert!(uses.is_empty());
        assert_eq!(server.len(), 1);
        assert_eq!(server[0].tool_name, "web_search");
        assert_eq!(server[0].response, citations);
    }
}
