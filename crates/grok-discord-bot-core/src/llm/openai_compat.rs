//! OpenAI-compatible provider, talking to the **Chat Completions API**
//! at `POST <base_url>/chat/completions`.
//!
//! This is the path for self-hosted models. It targets **vLLM** (the
//! best-supported OpenAI-compatible server) but works with any host that
//! speaks Chat Completions — Ollama, LM Studio, llama.cpp's server, etc.
//! Chat Completions (not the Responses API) is the lingua franca every
//! local server implements, so it's what we use here.
//!
//! Differences from the first-class [`super::openai`] (Responses API)
//! provider:
//!   - **No web search.** Regular Chat Completions models can't search,
//!     so `enable_web_search` is ignored. The model still drives the
//!     client-side tools (function calling) the bot hands it.
//!   - **No reasoning continuity / `provider_state`.** Chat Completions
//!     has no replayable opaque reasoning blob, so `provider_state` is
//!     always `None` and each request rebuilds the `messages` array from
//!     history (like the Anthropic provider).
//!   - **Caching is server-side.** vLLM's automatic prefix caching keys
//!     off a stable prompt prefix; we don't send a cache key. We just
//!     keep the system prompt first and history in order so the prefix
//!     stays stable and the server's cache hits.
//!
//! For tool calling to work, the host must be started with auto tool
//! choice enabled (vLLM: `--enable-auto-tool-choice --tool-call-parser
//! <parser>`); otherwise the model never emits `tool_calls`.

use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::config::OpenAiCompatConfig;
use crate::llm::{
    ChatTurn, LlmError, LlmProvider, MessageRole, StepRequest, StepResponse, ToolDefinition,
    ToolUseRequest, TurnBlock,
};

const PROVIDER_NAME: &str = "openai_compat";

/// OpenAI-compatible Chat Completions provider. Model-agnostic — the
/// model id is supplied per request via [`StepRequest::model`].
#[derive(Debug, Clone)]
pub struct OpenAiCompatProvider {
    http: reqwest::Client,
    /// Optional bearer token; many local servers accept any/none.
    api_key: Option<String>,
    base_url: String,
}

impl OpenAiCompatProvider {
    /// Construct from a config block.
    pub fn new(config: OpenAiCompatConfig) -> Self {
        Self {
            http: reqwest::Client::new(),
            api_key: config.api_key,
            base_url: config.base_url,
        }
    }

    /// Override the base URL. Used by tests.
    pub fn with_base_url(mut self, base_url: String) -> Self {
        self.base_url = base_url;
        self
    }
}

impl LlmProvider for OpenAiCompatProvider {
    fn name(&self) -> &str {
        PROVIDER_NAME
    }

    #[tracing::instrument(name = "step", skip_all, fields(provider = "openai_compat", model = %request.model))]
    async fn step(&self, request: StepRequest) -> Result<StepResponse, LlmError> {
        let messages = to_chat_messages(&request.messages);
        let tools = build_tools(&request.tools);

        let body = ChatRequest {
            model: &request.model,
            messages: &messages,
            tools: if tools.is_empty() { None } else { Some(&tools) },
            // OpenAI deprecated `max_tokens` for `max_completion_tokens`;
            // vLLM and most compat hosts accept the newer field too.
            max_completion_tokens: Some(request.max_tokens),
            temperature: request.temperature,
            top_p: request.top_p,
        };

        if tracing::enabled!(tracing::Level::DEBUG) {
            match serde_json::to_string(&body) {
                Ok(json) => {
                    tracing::debug!(target: "openai_compat_request", model = %request.model, body = %json, "openai_compat: sending request")
                }
                Err(e) => {
                    tracing::debug!(target: "openai_compat_request", model = %request.model, error = %e, "openai_compat: failed to serialize request for logging")
                }
            }
        }

        let url = format!("{}/chat/completions", self.base_url);
        let started = Instant::now();
        let resp = crate::retry::with_retry(
            crate::retry::RetryPolicy::default(),
            "llm[openai_compat]",
            || {
                let mut req = self.http.post(&url).json(&body);
                if let Some(key) = &self.api_key {
                    req = req.bearer_auth(key);
                }
                async move {
                    let resp = req
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
                    Ok(resp)
                }
            },
        )
        .await?;

        let parsed: ChatResponse = resp
            .json()
            .await
            .map_err(|e| LlmError::Decode(e.to_string()))?;
        let elapsed = started.elapsed();

        let model_id = parsed.model.unwrap_or_else(|| request.model.clone());
        log_usage(&model_id, parsed.usage.as_ref(), elapsed);

        let choice = parsed
            .choices
            .into_iter()
            .next()
            .ok_or_else(|| LlmError::Decode("response had no choices".into()))?;

        let text = choice.message.content.unwrap_or_default();
        let tool_uses = parse_tool_calls(&choice.message.tool_calls);

        if !tool_uses.is_empty() {
            Ok(StepResponse::UseTools {
                partial_text: if text.is_empty() { None } else { Some(text) },
                tool_uses,
                // No server-side tools on a compat host.
                server_tool_calls: Vec::new(),
                model_id,
                // Chat Completions carries no replayable reasoning blob.
                provider_state: None,
            })
        } else {
            Ok(StepResponse::Final {
                content: text,
                server_tool_calls: Vec::new(),
                model_id,
                provider_state: None,
            })
        }
    }
}

/// Convert our [`ChatTurn`] history into the Chat Completions `messages`
/// array. System/user/assistant turns map directly; a user turn's
/// `ToolResult` blocks each become a separate `{role:"tool"}` message
/// (Chat Completions answers tool calls with dedicated tool messages,
/// not content blocks).
fn to_chat_messages(turns: &[ChatTurn]) -> Vec<Value> {
    let mut messages: Vec<Value> = Vec::new();

    for turn in turns {
        match turn.role {
            MessageRole::System => {
                let mut text = String::new();
                for block in &turn.blocks {
                    if let TurnBlock::Text(t) = block {
                        text.push_str(t);
                    }
                }
                if !text.is_empty() {
                    messages.push(json!({ "role": "system", "content": text }));
                }
            }
            MessageRole::Assistant => {
                let mut text = String::new();
                let mut tool_calls: Vec<Value> = Vec::new();
                for block in &turn.blocks {
                    match block {
                        TurnBlock::Text(t) => text.push_str(t),
                        TurnBlock::ToolUse { id, name, input } => {
                            let args =
                                serde_json::to_string(input).unwrap_or_else(|_| "{}".into());
                            tool_calls.push(json!({
                                "id": id,
                                "type": "function",
                                "function": { "name": name, "arguments": args },
                            }));
                        }
                        // No reasoning replay on Chat Completions; images
                        // never appear on assistant turns.
                        _ => {}
                    }
                }
                let mut msg = serde_json::Map::new();
                msg.insert("role".into(), Value::String("assistant".into()));
                // With tool_calls, content may be null; otherwise send the text.
                if tool_calls.is_empty() {
                    msg.insert("content".into(), Value::String(text));
                } else {
                    msg.insert(
                        "content".into(),
                        if text.is_empty() {
                            Value::Null
                        } else {
                            Value::String(text)
                        },
                    );
                    msg.insert("tool_calls".into(), Value::Array(tool_calls));
                }
                messages.push(Value::Object(msg));
            }
            MessageRole::User => {
                // Tool results become their own `tool` messages; plain
                // text / images become one `user` message.
                let mut text = String::new();
                let mut image_urls: Vec<String> = Vec::new();
                let mut tool_results: Vec<Value> = Vec::new();
                for block in &turn.blocks {
                    match block {
                        TurnBlock::Text(t) => text.push_str(t),
                        TurnBlock::Image { url, .. } => image_urls.push(url.clone()),
                        TurnBlock::ToolResult {
                            tool_use_id,
                            content,
                            ..
                        } => {
                            tool_results.push(json!({
                                "role": "tool",
                                "tool_call_id": tool_use_id,
                                "content": content,
                            }));
                        }
                        TurnBlock::ToolUse { .. } | TurnBlock::Reasoning { .. } => {}
                    }
                }

                if !text.is_empty() || !image_urls.is_empty() {
                    if image_urls.is_empty() {
                        messages.push(json!({ "role": "user", "content": text }));
                    } else {
                        let mut parts: Vec<Value> = Vec::with_capacity(image_urls.len() + 1);
                        if !text.is_empty() {
                            parts.push(json!({ "type": "text", "text": text }));
                        }
                        for url in image_urls {
                            parts.push(json!({
                                "type": "image_url",
                                "image_url": { "url": url },
                            }));
                        }
                        messages.push(json!({ "role": "user", "content": parts }));
                    }
                }
                messages.extend(tool_results);
            }
        }
    }

    messages
}

/// Build the Chat Completions `tools` array. Client-side functions only —
/// compat hosts have no server-side web search.
fn build_tools(defs: &[ToolDefinition]) -> Vec<Value> {
    defs.iter()
        .map(|t| {
            json!({
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.input_schema,
                },
            })
        })
        .collect()
}

/// Parse the assistant message's `tool_calls` into [`ToolUseRequest`]s.
/// Each call's `arguments` is a JSON *string* that we parse into a Value.
fn parse_tool_calls(calls: &[ToolCall]) -> Vec<ToolUseRequest> {
    calls
        .iter()
        .map(|c| {
            let input = serde_json::from_str(&c.function.arguments).unwrap_or(Value::Null);
            ToolUseRequest {
                id: c.id.clone(),
                name: c.function.name.clone(),
                input,
            }
        })
        .collect()
}

/// Emit one INFO-level usage + timing event. `cached_tokens` (under
/// `prompt_tokens_details`) reflects the host's prefix-cache hit when it
/// reports one; vLLM's automatic prefix caching populates it.
fn log_usage(model: &str, usage: Option<&Usage>, elapsed: Duration) {
    let duration_ms = elapsed.as_millis() as u64;
    match usage {
        Some(u) => tracing::info!(
            target: "openai_compat_usage",
            model = %model,
            prompt_tokens = u.prompt_tokens,
            cached_tokens = u.prompt_tokens_details.cached_tokens,
            completion_tokens = u.completion_tokens,
            total_tokens = u.total_tokens,
            duration_ms,
            "openai_compat: chat completion complete",
        ),
        None => tracing::info!(
            target: "openai_compat_usage",
            model = %model,
            duration_ms,
            "openai_compat: chat completion complete; no usage block reported",
        ),
    }
}

#[derive(Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: &'a [Value],
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<&'a [Value]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_completion_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
}

#[derive(Deserialize)]
struct ChatResponse {
    #[serde(default)]
    choices: Vec<Choice>,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    usage: Option<Usage>,
}

#[derive(Deserialize)]
struct Choice {
    message: ResponseMessage,
}

#[derive(Deserialize)]
struct ResponseMessage {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    tool_calls: Vec<ToolCall>,
}

#[derive(Deserialize)]
struct ToolCall {
    #[serde(default)]
    id: String,
    function: ToolCallFunction,
}

#[derive(Deserialize)]
struct ToolCallFunction {
    #[serde(default)]
    name: String,
    #[serde(default)]
    arguments: String,
}

#[derive(Deserialize, Debug, Default)]
struct Usage {
    #[serde(default)]
    prompt_tokens: u64,
    #[serde(default)]
    prompt_tokens_details: TokenDetails,
    #[serde(default)]
    completion_tokens: u64,
    #[serde(default)]
    total_tokens: u64,
}

#[derive(Deserialize, Debug, Default)]
struct TokenDetails {
    #[serde(default)]
    cached_tokens: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn system_and_user_map_directly() {
        let turns = vec![
            ChatTurn::text(MessageRole::System, "be helpful"),
            ChatTurn::text(MessageRole::User, "hi"),
        ];
        let m = to_chat_messages(&turns);
        assert_eq!(m.len(), 2);
        assert_eq!(m[0], json!({"role": "system", "content": "be helpful"}));
        assert_eq!(m[1], json!({"role": "user", "content": "hi"}));
    }

    #[test]
    fn assistant_tool_call_then_tool_result_round_trip() {
        let turns = vec![
            ChatTurn {
                role: MessageRole::Assistant,
                blocks: vec![
                    TurnBlock::Text("on it".into()),
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
        let m = to_chat_messages(&turns);
        assert_eq!(m.len(), 2);
        assert_eq!(m[0]["role"], "assistant");
        assert_eq!(m[0]["content"], "on it");
        assert_eq!(m[0]["tool_calls"][0]["id"], "call_1");
        assert_eq!(m[0]["tool_calls"][0]["type"], "function");
        assert_eq!(m[0]["tool_calls"][0]["function"]["name"], "fetch_messages");
        // arguments is a JSON *string*.
        assert_eq!(
            m[0]["tool_calls"][0]["function"]["arguments"],
            "{\"limit\":10}"
        );
        // The tool result is its own `tool` message.
        assert_eq!(m[1]["role"], "tool");
        assert_eq!(m[1]["tool_call_id"], "call_1");
        assert_eq!(m[1]["content"], "[]");
    }

    #[test]
    fn assistant_with_only_tool_calls_has_null_content() {
        let turns = vec![ChatTurn {
            role: MessageRole::Assistant,
            blocks: vec![TurnBlock::ToolUse {
                id: "c1".into(),
                name: "f".into(),
                input: json!({}),
            }],
        }];
        let m = to_chat_messages(&turns);
        assert_eq!(m[0]["content"], Value::Null);
        assert!(m[0]["tool_calls"].is_array());
    }

    #[test]
    fn user_image_becomes_content_parts() {
        let turns = vec![ChatTurn {
            role: MessageRole::User,
            blocks: vec![
                TurnBlock::Text("look".into()),
                TurnBlock::Image {
                    url: "https://x/y.png".into(),
                    mime_type: None,
                },
            ],
        }];
        let m = to_chat_messages(&turns);
        assert_eq!(m.len(), 1);
        assert_eq!(m[0]["content"][0], json!({"type": "text", "text": "look"}));
        assert_eq!(
            m[0]["content"][1],
            json!({"type": "image_url", "image_url": {"url": "https://x/y.png"}})
        );
    }

    #[test]
    fn reasoning_blocks_are_dropped() {
        let turns = vec![ChatTurn {
            role: MessageRole::Assistant,
            blocks: vec![
                TurnBlock::Reasoning {
                    provider_name: "openai".into(),
                    data: json!([{ "type": "reasoning" }]),
                },
                TurnBlock::Text("hi".into()),
            ],
        }];
        let m = to_chat_messages(&turns);
        assert_eq!(m.len(), 1);
        assert_eq!(m[0]["content"], "hi");
        assert!(m[0].get("tool_calls").is_none());
    }

    #[test]
    fn tool_definitions_use_nested_function_shape() {
        let defs = vec![ToolDefinition {
            name: "fetch_messages".into(),
            description: "fetch".into(),
            input_schema: json!({ "type": "object" }),
        }];
        let tools = build_tools(&defs);
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["type"], "function");
        assert_eq!(tools[0]["function"]["name"], "fetch_messages");
        assert_eq!(tools[0]["function"]["parameters"]["type"], "object");
    }

    #[test]
    fn parses_final_message() {
        let body = json!({
            "model": "local-model",
            "choices": [{
                "message": { "role": "assistant", "content": "the answer" },
                "finish_reason": "stop",
            }],
            "usage": { "prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15 },
        });
        let parsed: ChatResponse = serde_json::from_value(body).unwrap();
        let choice = parsed.choices.into_iter().next().unwrap();
        assert_eq!(choice.message.content.as_deref(), Some("the answer"));
        assert!(choice.message.tool_calls.is_empty());
    }

    #[test]
    fn parses_tool_call_response() {
        let body = json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_9",
                        "type": "function",
                        "function": { "name": "fetch_messages", "arguments": "{\"limit\":30}" },
                    }],
                },
                "finish_reason": "tool_calls",
            }],
        });
        let parsed: ChatResponse = serde_json::from_value(body).unwrap();
        let choice = parsed.choices.into_iter().next().unwrap();
        assert!(choice.message.content.is_none());
        let uses = parse_tool_calls(&choice.message.tool_calls);
        assert_eq!(uses.len(), 1);
        assert_eq!(uses[0].id, "call_9");
        assert_eq!(uses[0].name, "fetch_messages");
        assert_eq!(uses[0].input["limit"], 30);
    }

    #[test]
    fn parses_usage_with_cached_tokens() {
        let body = json!({
            "prompt_tokens": 100,
            "prompt_tokens_details": { "cached_tokens": 64 },
            "completion_tokens": 20,
            "total_tokens": 120,
        });
        let u: Usage = serde_json::from_value(body).unwrap();
        assert_eq!(u.prompt_tokens, 100);
        assert_eq!(u.prompt_tokens_details.cached_tokens, 64);
        assert_eq!(u.completion_tokens, 20);
    }
}
