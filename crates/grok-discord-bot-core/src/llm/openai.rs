//! OpenAI provider, talking to the **Responses API** at
//! `POST https://api.openai.com/v1/responses`.
//!
//! This is OpenAI's modern agentic endpoint — the same family xAI's
//! Responses API is modeled on — so the wire shape closely matches
//! [`super::xai`]: an `input` array of items (instead of `messages`), an
//! `output` array of typed blocks (instead of `choices[0].message`), and
//! both server-side and client-side tool calls represented as top-level
//! output items. We keep it as a parallel module rather than sharing
//! helpers with `xai` so the two can diverge as each platform's API does.
//!
//! Server-side tool enabled on `enable_web_search`:
//! - `web_search` — OpenAI's built-in web search; citations come back as
//!   `url_citation` annotations on the assistant message (not a separate
//!   top-level field), which we collect onto the `web_search` trace row.
//!
//! Caching & reasoning continuity mirror xAI exactly: we send
//! `store: false` + `include: ["reasoning.encrypted_content"]`, set
//! `prompt_cache_key` to the conversation id for prefix-cache routing,
//! and capture the model's entire `output` array verbatim into
//! `provider_state` so later turns/iterations replay it byte-for-byte.

use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::config::OpenAiConfig;
use crate::llm::{
    ChatTurn, LlmError, LlmProvider, MessageRole, StepRequest, StepResponse, ToolCallRecord,
    ToolDefinition, ToolUseRequest, TurnBlock,
};

const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";

/// This provider's [`LlmProvider::name`]. Tags the `provider_state` it
/// produces so only its own continuation state is replayed back to it.
const PROVIDER_NAME: &str = "openai";

/// OpenAI provider. Model-agnostic — the specific model id is supplied
/// per request via [`StepRequest::model`].
#[derive(Debug, Clone)]
pub struct OpenAiProvider {
    http: reqwest::Client,
    api_key: String,
    base_url: String,
}

impl OpenAiProvider {
    /// Construct from a config block. Falls back to the public OpenAI
    /// base URL when the config doesn't override it.
    pub fn new(config: OpenAiConfig) -> Self {
        Self {
            http: reqwest::Client::new(),
            api_key: config.api_key,
            base_url: config
                .base_url
                .unwrap_or_else(|| DEFAULT_BASE_URL.to_string()),
        }
    }

    /// Override the base URL. Used by tests.
    pub fn with_base_url(mut self, base_url: String) -> Self {
        self.base_url = base_url;
        self
    }
}

impl LlmProvider for OpenAiProvider {
    fn name(&self) -> &str {
        PROVIDER_NAME
    }

    #[tracing::instrument(name = "step", skip_all, fields(provider = "openai", model = %request.model))]
    async fn step(&self, request: StepRequest) -> Result<StepResponse, LlmError> {
        let input_items = to_responses_input(&request.messages);
        let tools = build_tools(&request.tools, request.enable_web_search);

        // Per-persona OpenAI knobs. Today: `reasoning_effort` mapped to
        // the Responses API's `reasoning: { effort: ... }` block. The
        // field is silently ignored by non-reasoning models, so we pass
        // it through without sniffing the model id.
        let reasoning = request
            .provider_options
            .openai
            .as_ref()
            .and_then(|o| o.reasoning_effort.as_ref())
            .map(|effort| json!({ "effort": effort }));

        let body = ResponsesRequest {
            model: &request.model,
            input: &input_items,
            tools: if tools.is_empty() { None } else { Some(&tools) },
            max_output_tokens: Some(request.max_tokens),
            temperature: request.temperature,
            top_p: request.top_p,
            reasoning: reasoning.as_ref(),
            // OpenAI auto-caches prompt prefixes ≥1024 tokens for free;
            // `prompt_cache_key` steers prefix-sharing requests to the
            // same machine so the cache stays warm. Stable conversation
            // id keeps every iteration + later turn hitting.
            prompt_cache_key: request.cache_key.as_deref(),
            // Ask for the model's reasoning as an encrypted, opaque blob
            // so we can replay it verbatim on later requests without
            // storing plaintext chain-of-thought. Harmless for
            // non-reasoning models (they emit none).
            include: REASONING_INCLUDE,
            // We persist the full trace ourselves and replay history
            // explicitly each request, so server-side storage buys
            // nothing — and `store: false` is what makes the encrypted
            // reasoning replay path valid.
            store: false,
        };

        if tracing::enabled!(tracing::Level::DEBUG) {
            match serde_json::to_string(&body) {
                Ok(json) => {
                    tracing::debug!(target: "openai_request", model = %request.model, body = %json, "openai: sending request")
                }
                Err(e) => {
                    tracing::debug!(target: "openai_request", model = %request.model, error = %e, "openai: failed to serialize request for logging")
                }
            }
        }

        // Retry transient 5xx/429/transport blips with backoff. The
        // request is rebuilt each attempt (`.json` serializes eagerly, so
        // the future owns its body and borrows nothing across awaits).
        let url = format!("{}/responses", self.base_url);
        let started = Instant::now();
        let resp =
            crate::retry::with_retry(crate::retry::RetryPolicy::default(), "llm[openai]", || {
                let req = self.http.post(&url).bearer_auth(&self.api_key).json(&body);
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
            })
            .await?;

        let parsed: ResponsesResponse = resp
            .json()
            .await
            .map_err(|e| LlmError::Decode(e.to_string()))?;
        let elapsed = started.elapsed();

        let model_id = parsed.model.unwrap_or_else(|| request.model.clone());
        log_usage(&model_id, parsed.usage.as_ref(), elapsed);

        let (text, tool_uses, server_tool_calls) = walk_output(&parsed.output);
        // Carry the model's ENTIRE output array forward verbatim — the
        // encrypted `reasoning` item(s), the assistant `message`, any
        // `function_call`, and any server `web_search_call` items — in
        // emission order. Replaying it byte-for-byte (rather than
        // re-synthesizing a message) keeps OpenAI's prompt cache warm
        // across iterations and turns and keeps each encrypted reasoning
        // item attached to the message it precedes. `None` only when the
        // model produced no output at all.
        let provider_state = (!parsed.output.is_empty()).then_some(Value::Array(parsed.output));

        if !tool_uses.is_empty() {
            Ok(StepResponse::UseTools {
                partial_text: if text.is_empty() { None } else { Some(text) },
                tool_uses,
                server_tool_calls,
                model_id,
                provider_state,
            })
        } else {
            Ok(StepResponse::Final {
                content: text,
                server_tool_calls,
                model_id,
                provider_state,
            })
        }
    }
}

/// Convert our [`ChatTurn`] history into the Responses API's `input`
/// array. Identical strategy to the xAI provider: system turns become
/// inline `role: "system"` messages; an assistant turn carrying captured
/// `output` items (a [`TurnBlock::Reasoning`] tagged for us) replays them
/// verbatim; everything else is synthesized from text / image / tool
/// blocks.
fn to_responses_input(turns: &[ChatTurn]) -> Vec<Value> {
    let mut input: Vec<Value> = Vec::new();

    for turn in turns {
        if turn.role == MessageRole::System {
            let mut text = String::new();
            for block in &turn.blocks {
                if let TurnBlock::Text(t) = block {
                    text.push_str(t);
                }
            }
            if !text.is_empty() {
                input.push(json!({ "role": "system", "content": text }));
            }
            continue;
        }

        let role_str = match turn.role {
            MessageRole::Assistant => "assistant",
            _ => "user",
        };

        // Gather this turn's verbatim continuation items. Only our own
        // (`openai`) state round-trips; a block tagged for another
        // provider (a mid-conversation persona switch) is not ours.
        let mut echo: Vec<Value> = Vec::new();
        for block in &turn.blocks {
            if let TurnBlock::Reasoning {
                provider_name,
                data,
            } = block
                && provider_name == PROVIDER_NAME
            {
                match data {
                    Value::Array(items) => echo.extend(items.iter().cloned()),
                    other => echo.push(other.clone()),
                }
            }
        }
        // A turn carrying any non-`reasoning` item was captured under the
        // full-output format: replay it byte-for-byte and skip
        // re-synthesis. A turn whose only items are `reasoning` is a
        // legacy reasoning-only blob (or there are none): fall through to
        // the synthesis path, which replays the reasoning (if any) ahead
        // of the rebuilt message.
        let full_echo = echo
            .iter()
            .any(|it| it.get("type").and_then(Value::as_str) != Some("reasoning"));
        if full_echo {
            input.append(&mut echo);
            continue;
        }

        let mut text_buf = String::new();
        let mut image_urls: Vec<String> = Vec::new();
        let mut deferred: Vec<Value> = Vec::new();

        for block in &turn.blocks {
            match block {
                TurnBlock::Text(t) => text_buf.push_str(t),
                TurnBlock::Image { url, .. } => image_urls.push(url.clone()),
                TurnBlock::Reasoning { .. } => {}
                TurnBlock::ToolUse {
                    id,
                    name,
                    input: tool_input,
                } => {
                    let args = serde_json::to_string(tool_input).unwrap_or_else(|_| "{}".into());
                    deferred.push(json!({
                        "type": "function_call",
                        "call_id": id,
                        "name": name,
                        "arguments": args,
                    }));
                }
                TurnBlock::ToolResult {
                    tool_use_id,
                    content,
                    ..
                } => {
                    deferred.push(json!({
                        "type": "function_call_output",
                        "call_id": tool_use_id,
                        "output": content,
                    }));
                }
            }
        }

        // Legacy reasoning-only items lead the turn — they precede the
        // rebuilt assistant message and must be replayed in that position
        // for the cache prefix to match.
        input.append(&mut echo);

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

    input
}

/// Build the `tools` array — client-side function definitions plus
/// OpenAI's built-in `web_search` tool when enabled.
fn build_tools(defs: &[ToolDefinition], enable_web_search: bool) -> Vec<Value> {
    let mut tools: Vec<Value> = Vec::with_capacity(defs.len() + 1);
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
    }
    tools
}

/// Walk the `output` array. Returns concatenated assistant text,
/// client-side `function_call`s as [`ToolUseRequest`]s, and server-side
/// tool calls (`web_search_call`, …) as [`ToolCallRecord`]s.
///
/// Unlike xAI, OpenAI delivers web-search citations as `url_citation`
/// annotations embedded in the assistant message's `output_text` blocks
/// (there is no top-level `citations` field). We collect those and attach
/// them to the `web_search` server-call row so the trace still shows what
/// the search produced. `reasoning` items are ignored here — the caller
/// captures the WHOLE `output` array verbatim into `provider_state`.
fn walk_output(output: &[Value]) -> (String, Vec<ToolUseRequest>, Vec<ToolCallRecord>) {
    let mut text = String::new();
    let mut tool_uses: Vec<ToolUseRequest> = Vec::new();
    let mut server_calls: Vec<ToolCallRecord> = Vec::new();
    let mut citations: Vec<Value> = Vec::new();

    for item in output {
        let kind = item.get("type").and_then(Value::as_str).unwrap_or("");
        match kind {
            "message" => {
                if let Some(content) = item.get("content").and_then(Value::as_array) {
                    for block in content {
                        let block_kind = block.get("type").and_then(Value::as_str).unwrap_or("");
                        if (block_kind == "output_text" || block_kind == "text")
                            && let Some(t) = block.get("text").and_then(Value::as_str)
                        {
                            text.push_str(t);
                        }
                        // Collect url_citation annotations regardless of
                        // block type so the web_search trace is complete.
                        if let Some(anns) = block.get("annotations").and_then(Value::as_array) {
                            for ann in anns {
                                if ann.get("type").and_then(Value::as_str) == Some("url_citation") {
                                    citations.push(ann.clone());
                                }
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
            // Server-side tool calls (web_search_call, …) arrive as
            // top-level items ending in `_call`.
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

    // Attach collected citations to whichever server call could have
    // produced them (prefer web_search; fall back to the first server
    // call; failing that, record a freestanding entry).
    if !citations.is_empty() {
        let citations = Value::Array(citations);
        if let Some(slot) = server_calls.iter_mut().find(|r| r.tool_name == "web_search") {
            slot.response = citations;
        } else if let Some(first) = server_calls.first_mut() {
            first.response = citations;
        } else {
            server_calls.push(ToolCallRecord {
                tool_name: "web_search".to_string(),
                request: json!({ "implicit": true }),
                response: citations,
            });
        }
    }

    (text, tool_uses, server_calls)
}

/// Emit a single INFO-level usage + timing event for one Responses API
/// request. Token counts come from the response's `usage` block;
/// `cached_tokens` (under `input_tokens_details`) is the cache hit signal.
///
/// `usage` is parsed leniently (from a raw [`Value`]) so a schema change
/// in the usage object degrades to a `warn` here instead of failing the
/// whole response decode and dropping the model's answer.
fn log_usage(model: &str, usage: Option<&Value>, elapsed: Duration) {
    let duration_ms = elapsed.as_millis() as u64;
    match usage.map(|u| serde_json::from_value::<Usage>(u.clone())) {
        Some(Ok(u)) => tracing::info!(
            target: "openai_usage",
            model = %model,
            input_tokens = u.input_tokens,
            cached_tokens = u.input_tokens_details.cached_tokens,
            output_tokens = u.output_tokens,
            reasoning_tokens = u.output_tokens_details.reasoning_tokens,
            total_tokens = u.total_tokens,
            duration_ms,
            "openai: responses request complete",
        ),
        Some(Err(e)) => tracing::warn!(
            target: "openai_usage",
            model = %model,
            duration_ms,
            error = %e,
            "openai: responses request complete; could not parse usage block",
        ),
        None => tracing::info!(
            target: "openai_usage",
            model = %model,
            duration_ms,
            "openai: responses request complete; no usage block reported",
        ),
    }
}

/// Responses-API `include` value requesting the model's reasoning be
/// returned as an encrypted, opaque blob (so it can be replayed verbatim
/// without us storing plaintext chain-of-thought).
const REASONING_INCLUDE: &[&str] = &["reasoning.encrypted_content"];

#[derive(Serialize)]
struct ResponsesRequest<'a> {
    model: &'a str,
    input: &'a [Value],
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<&'a [Value]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning: Option<&'a Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prompt_cache_key: Option<&'a str>,
    #[serde(skip_serializing_if = "<[_]>::is_empty")]
    include: &'a [&'a str],
    store: bool,
}

#[derive(Deserialize)]
struct ResponsesResponse {
    #[serde(default)]
    output: Vec<Value>,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    usage: Option<Value>,
}

/// Token usage from a Responses API reply; field names mirror OpenAI's
/// `usage` object. Every field defaults, so a partial or older usage
/// block still parses cleanly.
#[derive(Deserialize, Debug, Default)]
struct Usage {
    #[serde(default)]
    input_tokens: u64,
    #[serde(default)]
    input_tokens_details: TokenDetails,
    #[serde(default)]
    output_tokens: u64,
    #[serde(default)]
    output_tokens_details: TokenDetails,
    #[serde(default)]
    total_tokens: u64,
}

/// The `*_tokens_details` sub-objects. `cached_tokens` lives under
/// `input_tokens_details` (the cache-hit signal); `reasoning_tokens`
/// under `output_tokens_details`. One struct covers both.
#[derive(Deserialize, Debug, Default)]
struct TokenDetails {
    #[serde(default)]
    cached_tokens: u64,
    #[serde(default)]
    reasoning_tokens: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn system_becomes_inline_role_message() {
        let turns = vec![
            ChatTurn::text(MessageRole::System, "be helpful"),
            ChatTurn::text(MessageRole::User, "hi"),
        ];
        let input = to_responses_input(&turns);
        assert_eq!(input.len(), 2);
        assert_eq!(input[0]["role"], "system");
        assert_eq!(input[0]["content"], "be helpful");
        assert_eq!(input[1]["role"], "user");
        assert_eq!(input[1]["content"], "hi");
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
        let input = to_responses_input(&turns);
        assert_eq!(input.len(), 4);
        assert_eq!(input[2]["type"], "function_call");
        assert_eq!(input[2]["call_id"], "call_1");
        assert_eq!(input[3]["type"], "function_call_output");
        assert_eq!(input[3]["output"], "[]");
    }

    #[test]
    fn parses_message_and_function_call() {
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
        let (text, uses, server) = walk_output(&output);
        assert_eq!(text, "Let me check. ");
        assert_eq!(uses.len(), 1);
        assert_eq!(uses[0].id, "call_42");
        assert_eq!(uses[0].input["limit"], 30);
        assert!(server.is_empty());
    }

    #[test]
    fn web_search_tool_present_only_when_enabled() {
        let with = build_tools(&[], true);
        assert_eq!(with.len(), 1);
        assert_eq!(with[0]["type"], "web_search");
        let without = build_tools(&[], false);
        assert!(without.is_empty());
    }

    #[test]
    fn collects_url_citations_onto_web_search_call() {
        // OpenAI returns citations as url_citation annotations on the
        // message, not a separate field; we hang them on the web_search row.
        let output = vec![
            json!({"type": "web_search_call", "id": "ws_1", "status": "completed"}),
            json!({
                "type": "message",
                "role": "assistant",
                "content": [{
                    "type": "output_text",
                    "text": "Found it.",
                    "annotations": [
                        {"type": "url_citation", "url": "https://example.com", "title": "x"}
                    ],
                }],
            }),
        ];
        let (text, uses, server) = walk_output(&output);
        assert_eq!(text, "Found it.");
        assert!(uses.is_empty());
        assert_eq!(server.len(), 1);
        assert_eq!(server[0].tool_name, "web_search");
        assert_eq!(server[0].response[0]["url"], "https://example.com");
    }

    #[test]
    fn reasoning_block_serializes_when_effort_set() {
        let reasoning = json!({ "effort": "high" });
        let body = ResponsesRequest {
            model: "gpt-5.5",
            input: &[],
            tools: None,
            max_output_tokens: Some(4096),
            temperature: None,
            top_p: None,
            reasoning: Some(&reasoning),
            prompt_cache_key: None,
            include: &[],
            store: false,
        };
        let v = serde_json::to_value(&body).unwrap();
        assert_eq!(v["reasoning"]["effort"], "high");
    }

    #[test]
    fn prompt_cache_key_and_include_serialize() {
        let body = ResponsesRequest {
            model: "gpt-5.5",
            input: &[],
            tools: None,
            max_output_tokens: None,
            temperature: None,
            top_p: None,
            reasoning: None,
            prompt_cache_key: Some("conv-123"),
            include: REASONING_INCLUDE,
            store: false,
        };
        let v = serde_json::to_value(&body).unwrap();
        assert_eq!(v["prompt_cache_key"], "conv-123");
        assert_eq!(v["include"][0], "reasoning.encrypted_content");
        assert_eq!(v["store"], false);
    }

    #[test]
    fn replays_full_output_verbatim_when_present() {
        let full_output = json!([
            { "type": "reasoning", "id": "rs_1", "encrypted_content": "BLOB" },
            {
                "type": "message",
                "role": "assistant",
                "id": "msg_1",
                "content": [{ "type": "output_text", "text": "the answer" }],
            },
        ]);
        let turns = vec![
            ChatTurn::text(MessageRole::User, "hi"),
            ChatTurn {
                role: MessageRole::Assistant,
                blocks: vec![
                    TurnBlock::Reasoning {
                        provider_name: PROVIDER_NAME.into(),
                        data: full_output,
                    },
                    TurnBlock::Text("the answer".into()),
                ],
            },
        ];
        let input = to_responses_input(&turns);
        assert_eq!(input.len(), 3);
        assert_eq!(input[1]["type"], "reasoning");
        assert_eq!(input[1]["encrypted_content"], "BLOB");
        assert_eq!(input[2]["type"], "message");
        assert_eq!(input[2]["id"], "msg_1");
    }

    #[test]
    fn drops_reasoning_tagged_for_another_provider() {
        let turns = vec![ChatTurn {
            role: MessageRole::Assistant,
            blocks: vec![
                TurnBlock::Reasoning {
                    provider_name: "xai".into(),
                    data: json!([{ "type": "reasoning", "encrypted_content": "BLOB" }]),
                },
                TurnBlock::Text("hi".into()),
            ],
        }];
        let input = to_responses_input(&turns);
        // Foreign reasoning is dropped; only the synthesized message survives.
        assert_eq!(input.len(), 1);
        assert_eq!(input[0]["role"], "assistant");
        assert_eq!(input[0]["content"], "hi");
    }

    #[test]
    fn parses_usage_block() {
        let usage = json!({
            "input_tokens": 153,
            "input_tokens_details": { "cached_tokens": 128 },
            "output_tokens": 602,
            "output_tokens_details": { "reasoning_tokens": 303 },
            "total_tokens": 755,
        });
        let u: Usage = serde_json::from_value(usage).unwrap();
        assert_eq!(u.input_tokens, 153);
        assert_eq!(u.input_tokens_details.cached_tokens, 128);
        assert_eq!(u.output_tokens_details.reasoning_tokens, 303);
        assert_eq!(u.total_tokens, 755);
    }
}
