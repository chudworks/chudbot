//! xAI Responses API language-model implementation.
//!
//! This module is the boundary between Chudbot's provider-neutral
//! [`ModelStepRequest`] contract and xAI's `/responses` JSON shape. It is
//! responsible for translating transcripts into replayable Responses input,
//! preserving xAI continuation items for future turns, decoding model output
//! into text/tool/server-use blocks, and normalizing usage into Chudbot records.

use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use chudbot_api::{
    AssistantStep, ClientToolCall, ClientToolSpec, ContentBlock, CostAmount, GroundingMetadata,
    LlmBackend, ModelId, ModelInfo, ModelInfoRequest, ModelStep, ModelStepRequest,
    ProviderContinuation, ProviderName, ServerToolSet, ServerToolUse, ToolName, ToolUseId,
    Transcript, TurnRole, UsageRecord, UsageSubject,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::imagine::media_provider_url;
use crate::{XaiClient, XaiError, json_strip_nulls};

/// Request encrypted reasoning blobs so later turns can replay provider state.
const REASONING_INCLUDE: &[&str] = &["reasoning.encrypted_content"];

impl LlmBackend for XaiClient {
    type Error = XaiError;

    fn backend_name(&self) -> &ProviderName {
        self.provider_name()
    }

    #[tracing::instrument(name = "xai.step", skip_all, fields(model = %request.model))]
    async fn step(&self, request: ModelStepRequest) -> Result<ModelStep, Self::Error> {
        // Build the full Responses payload up front: transcript replay, tool
        // declarations, sampling knobs, and provider-specific options all have
        // to agree before xAI can continue the turn.
        let input = to_responses_input(&request.transcript, self.provider_name()).await?;
        let tools = build_responses_tools(&request.client_tools, &request.server_tools);
        let options = XaiOptions::from_request(&request);
        let reasoning = options
            .reasoning_effort
            .as_ref()
            .map(|effort| json!({ "effort": effort }));
        let has_tools = !tools.is_empty();

        let body = json_strip_nulls(json!({
            "model": request.model.as_str(),
            "input": input,
            "tools": has_tools.then_some(tools),
            "parallel_tool_calls": has_tools.then_some(true),
            "max_output_tokens": request.sampling.max_output_tokens,
            "temperature": request.sampling.temperature,
            "top_p": request.sampling.top_p,
            "reasoning": reasoning,
            "prompt_cache_key": request.transcript.id,
            "include": REASONING_INCLUDE,
            "store": false,
        }));

        let started = Instant::now();
        let parsed: ResponsesResponse = self.post_json("/responses", &body, "llm[xai]").await?;
        // Prefer xAI's echoed model id when present; aliases can resolve to a
        // concrete serving model and downstream usage records should reflect it.
        let model_id = parsed
            .model
            .as_deref()
            .map(ModelId::new)
            .unwrap_or_else(|| request.model.clone());
        let usage = usage_from_xai(
            self.provider_name(),
            Some(model_id.clone()),
            UsageSubject::ModelStep,
            parsed.usage.as_ref(),
        );
        log_usage(model_id.as_str(), usage.as_ref(), started.elapsed());

        let continuation = continuation_from_output(self.provider_name(), &parsed.output);

        // Split the mixed Responses output stream into Chudbot's assistant
        // step surface while keeping the raw server/citation data available to
        // the trace viewer.
        let (text, client_tool_calls, server_tool_uses) =
            walk_output(&parsed.output, self.provider_name());
        let grounding = parsed
            .citations
            .map(|raw| {
                vec![GroundingMetadata {
                    provider: self.provider_name().clone(),
                    raw,
                }]
            })
            .unwrap_or_default();

        let mut content = Vec::new();
        if !text.is_empty() {
            content.push(ContentBlock::Text { text });
        }

        let step = AssistantStep {
            content,
            client_tool_calls,
            server_tool_uses,
            grounding,
            model_id,
            continuation,
            usage: usage.into_iter().collect(),
        };

        // Client tool calls keep the conversation loop alive; text with no
        // client calls is final, and empty content means the model produced
        // replayable/provider state but still needs another turn.
        if !step.client_tool_calls.is_empty() {
            Ok(ModelStep::UseClientTools { step })
        } else if step.content.is_empty() {
            Ok(ModelStep::Continue { step })
        } else {
            Ok(ModelStep::Final { step })
        }
    }

    #[tracing::instrument(name = "xai.model_info", skip_all, fields(model = %request.model))]
    async fn fetch_model_info(
        &self,
        request: ModelInfoRequest,
    ) -> Result<Option<ModelInfo>, Self::Error> {
        // xAI exposes model metadata as a list, so select the requested model
        // locally instead of relying on a per-model endpoint.
        let raw: Value = self.get_json("/models", "model_info[xai]").await?;
        Ok(model_info_from_models_response(request.model, raw))
    }
}

/// Extract token limits from either a single model object or an OpenAI-style list.
fn model_info_from_models_response(requested_model: ModelId, raw: Value) -> Option<ModelInfo> {
    let entry = model_entry_from_models_response(&requested_model, &raw)?;
    model_info_from_api_model(requested_model, entry)
}

fn model_entry_from_models_response(requested_model: &ModelId, raw: &Value) -> Option<Value> {
    if model_id_matches(raw, requested_model) {
        return Some(raw.clone());
    }

    // Some compatible endpoints return a single-entry `data` list without an
    // exact id match; treating that as authoritative keeps model-info discovery
    // useful for self-hosted or aliasing deployments.
    let data = raw.get("data").and_then(Value::as_array)?;
    data.iter()
        .find(|entry| model_id_matches(entry, requested_model))
        .or_else(|| (data.len() == 1).then(|| &data[0]))
        .cloned()
}

fn model_id_matches(value: &Value, requested_model: &ModelId) -> bool {
    value
        .get("id")
        .and_then(Value::as_str)
        .is_some_and(|id| id == requested_model.as_str())
}

fn model_info_from_api_model(requested_model: ModelId, raw: Value) -> Option<ModelInfo> {
    // xAI-compatible model metadata has not used one stable field name for
    // token limits, so accept the common names seen across Responses-like APIs.
    const CONTEXT_FIELDS: &[&str] = &[
        "context_window_tokens",
        "context_window",
        "context_length",
        "max_context_length",
        "max_context_len",
        "max_model_len",
        "max_sequence_length",
        "max_position_embeddings",
    ];
    const OUTPUT_FIELDS: &[&str] = &[
        "max_output_tokens",
        "max_completion_tokens",
        "output_token_limit",
        "max_tokens",
    ];

    let context_window_tokens = find_u64_field(&raw, CONTEXT_FIELDS);
    let max_output_tokens = find_u64_field(&raw, OUTPUT_FIELDS);
    if context_window_tokens.is_none() && max_output_tokens.is_none() {
        return None;
    }

    let id = raw
        .get("id")
        .and_then(Value::as_str)
        .filter(|id| !id.is_empty())
        .map(ModelId::new)
        .unwrap_or(requested_model);
    Some(ModelInfo {
        id,
        context_window_tokens,
        max_output_tokens,
        raw: Some(raw),
    })
}

fn find_u64_field(value: &Value, fields: &[&str]) -> Option<u64> {
    let object = value.as_object()?;
    if let Some(found) = fields
        .iter()
        .find_map(|field| object.get(*field).and_then(value_as_u64))
    {
        return Some(found);
    }

    // Providers often nest limits under `capabilities`, `limits`, or similar
    // objects; recurse through object children but avoid arrays to keep this
    // predictable.
    object
        .values()
        .filter(|value| value.is_object())
        .find_map(|value| find_u64_field(value, fields))
}

fn value_as_u64(value: &Value) -> Option<u64> {
    match value {
        Value::Number(number) => number.as_u64(),
        Value::String(text) => text.parse().ok(),
        _ => None,
    }
}

async fn to_responses_input(
    transcript: &Transcript,
    provider: &ProviderName,
) -> Result<Vec<Value>, XaiError> {
    let mut input = Vec::new();
    if let Some(instructions) = &transcript.instructions
        && !instructions.is_empty()
    {
        let id = transcript.id.as_deref().map(system_message_id);
        input.push(json_strip_nulls(json!({
            "id": id,
            "role": "system",
            "content": instructions,
        })));
    }

    for message in &transcript.turns {
        let role = match message.role {
            TurnRole::Assistant => "assistant",
            TurnRole::User => "user",
        };

        let mut echo = Vec::new();
        // Continuations are provider-specific raw Responses output. When a
        // previous xAI turn can be replayed, prefer it over synthesizing a lossy
        // assistant message from stored text.
        for block in &message.blocks {
            if let ContentBlock::Continuation(continuation) = block
                && &continuation.provider == provider
            {
                match &continuation.data {
                    Value::Array(items) => {
                        echo.extend(replayable_continuation_items(items.iter().cloned()))
                    }
                    other => {
                        if let Some(item) = replayable_continuation_item(other.clone()) {
                            echo.push(item);
                        }
                    }
                }
            }
        }
        let full_echo = echo
            .iter()
            .any(|item| item.get("type").and_then(Value::as_str) != Some("reasoning"));
        if full_echo {
            input.extend(echo);
            continue;
        }

        input.extend(echo);
        let id = transcript_turn_message_id(message);
        let mut text = String::new();
        let mut media_urls = Vec::new();
        // Tool calls and tool results are standalone Responses items, so flush
        // any accumulated user/assistant message before appending them.
        for block in &message.blocks {
            match block {
                ContentBlock::Text { text: t } => text.push_str(t),
                ContentBlock::Media { media } => {
                    media_urls.push(media_provider_url(media.as_ref()).await?)
                }
                ContentBlock::Continuation(_) => {}
                ContentBlock::ClientToolCall(call) => {
                    push_responses_message(&mut input, id, role, &mut text, &mut media_urls);
                    let args = serde_json::to_string(&call.input).unwrap_or_else(|_| "{}".into());
                    input.push(json!({
                        "type": "function_call",
                        "call_id": call.id.as_str(),
                        "name": call.name.as_str(),
                        "arguments": args,
                    }));
                }
                ContentBlock::ClientToolResult(result) => {
                    push_responses_message(&mut input, id, role, &mut text, &mut media_urls);
                    input.push(json!({
                        "type": "function_call_output",
                        "call_id": result.tool_use_id.as_str(),
                        "output": client_tool_result_as_string(result),
                    }));
                }
            }
        }

        push_responses_message(&mut input, id, role, &mut text, &mut media_urls);
    }
    Ok(input)
}

fn push_responses_message(
    input: &mut Vec<Value>,
    id: Option<&str>,
    role: &str,
    text: &mut String,
    media_urls: &mut Vec<String>,
) {
    if media_urls.is_empty() {
        // Text-only turns use the compact Responses message shape.
        if !text.is_empty() {
            input.push(json_strip_nulls(json!({
                "id": id,
                "role": role,
                "content": text.as_str(),
            })));
            text.clear();
        }
        return;
    }

    // Any media forces the content-array shape so text and images stay ordered
    // within one logical transcript message.
    let mut content = Vec::with_capacity(media_urls.len() + 1);
    if !text.is_empty() {
        content.push(json!({ "type": "input_text", "text": text.as_str() }));
    }
    for url in media_urls.iter() {
        content.push(json!({ "type": "input_image", "image_url": url }));
    }
    input.push(json_strip_nulls(json!({
        "id": id,
        "role": role,
        "content": content,
    })));
    text.clear();
    media_urls.clear();
}

fn continuation_from_output(
    provider: &ProviderName,
    output: &[Value],
) -> Option<ProviderContinuation> {
    // Persist the replayable subset of raw output so a later step can resume
    // xAI-side reasoning/tool state instead of reconstructing it from text.
    let items = replayable_continuation_items(output.iter().cloned());
    (!items.is_empty()).then_some(ProviderContinuation {
        provider: provider.clone(),
        data: Value::Array(items),
    })
}

fn replayable_continuation_items(items: impl IntoIterator<Item = Value>) -> Vec<Value> {
    items
        .into_iter()
        .filter_map(replayable_continuation_item)
        .collect()
}

fn replayable_continuation_item(item: Value) -> Option<Value> {
    let Some(encrypted_content) = item.get("encrypted_content") else {
        return Some(item);
    };
    let valid = encrypted_content
        .as_str()
        .is_some_and(is_replayable_encrypted_content);
    if valid {
        return Some(item);
    }

    // xAI rejects malformed encrypted reasoning on replay. Dropping only the
    // bad item preserves the rest of the continuation and lets the transcript
    // fall back to stored assistant text when no replayable state remains.
    tracing::warn!(
        item_id = item
            .get("id")
            .and_then(|value| value.as_str())
            .unwrap_or("<missing>"),
        encrypted_chars = encrypted_content.as_str().map(|text| text.chars().count()),
        "skipping malformed xAI encrypted reasoning continuation"
    );
    None
}

fn is_replayable_encrypted_content(text: &str) -> bool {
    // The encrypted field is expected to be base64/base64url-like text; commas
    // and other punctuation are a strong signal that a streamed blob was
    // corrupted before storage.
    !text.is_empty()
        && text.bytes().all(|b| {
            b.is_ascii_alphanumeric()
                || matches!(b, b'+' | b'/' | b'=' | b'-' | b'_' | b'\r' | b'\n')
        })
}

fn system_message_id(transcript_id: &str) -> String {
    format!("chudbot_conversation_{transcript_id}_system")
}

fn transcript_turn_message_id(message: &chudbot_api::TranscriptTurn) -> Option<&str> {
    message.metadata.get("id").and_then(Value::as_str)
}

fn client_tool_result_as_string(result: &chudbot_api::ClientToolResult) -> String {
    // Responses function outputs are strings even when Chudbot's tool result is
    // structured JSON.
    match &result.content {
        chudbot_api::ClientToolResultContent::Json { value } => {
            serde_json::to_string(value).unwrap_or_else(|_| value.to_string())
        }
        chudbot_api::ClientToolResultContent::Text { text } => text.clone(),
    }
}

fn build_responses_tools(
    client_tools: &BTreeMap<ToolName, ClientToolSpec>,
    server_tools: &ServerToolSet,
) -> Vec<Value> {
    let mut tools = Vec::with_capacity(client_tools.len() + 2);
    for (name, tool) in client_tools {
        // Client tools become Responses function tools whose results are routed
        // back through Chudbot before the model continues.
        tools.push(json!({
            "type": "function",
            "name": name.as_str(),
            "description": tool.description,
            "parameters": tool.input_schema.as_json_schema(),
        }));
    }
    // Server tools are executed by xAI and reported back as raw *_call output
    // items for trace visibility.
    if server_tools.contains("web_search") {
        tools.push(json!({ "type": "web_search" }));
    }
    if server_tools.contains("x_search") {
        tools.push(json!({ "type": "x_search" }));
    }
    tools
}

fn walk_output(
    output: &[Value],
    provider: &ProviderName,
) -> (String, Vec<ClientToolCall>, Vec<ServerToolUse>) {
    let mut text = String::new();
    let mut client_calls = Vec::new();
    let mut server_uses = Vec::new();

    for item in output {
        let kind = item.get("type").and_then(Value::as_str).unwrap_or("");
        match kind {
            "message" => {
                // xAI can return message content either as a Responses block
                // array or as a compact string, depending on model/API path.
                if let Some(content) = item.get("content").and_then(Value::as_array) {
                    for block in content {
                        let block_kind = block.get("type").and_then(Value::as_str).unwrap_or("");
                        if (block_kind == "output_text" || block_kind == "text")
                            && let Some(t) = block.get("text").and_then(Value::as_str)
                        {
                            text.push_str(t);
                        }
                    }
                } else if let Some(t) = item.get("content").and_then(Value::as_str) {
                    text.push_str(t);
                }
            }
            "function_call" => {
                // Chudbot-owned function calls are the only output items that
                // turn into client tool work for the next loop iteration.
                let id = item
                    .get("call_id")
                    .and_then(Value::as_str)
                    .or_else(|| item.get("id").and_then(Value::as_str))
                    .unwrap_or("");
                let name = item.get("name").and_then(Value::as_str).unwrap_or("");
                let args = item
                    .get("arguments")
                    .and_then(Value::as_str)
                    .unwrap_or("{}");
                let input = serde_json::from_str(args).unwrap_or(Value::Null);
                client_calls.push(ClientToolCall {
                    id: ToolUseId::new(id),
                    name: ToolName::new(name),
                    input,
                });
            }
            other if other.ends_with("_call") => {
                // All non-function *_call items are provider-executed server
                // tools. Preserve the raw item because each tool has its own
                // evolving response shape.
                server_uses.push(ServerToolUse {
                    provider: provider.clone(),
                    name: ToolName::new(other.trim_end_matches("_call")),
                    id: item
                        .get("id")
                        .and_then(Value::as_str)
                        .map(str::to_string)
                        .or_else(|| {
                            item.get("call_id")
                                .and_then(Value::as_str)
                                .map(str::to_string)
                        }),
                    status: item
                        .get("status")
                        .and_then(Value::as_str)
                        .map(str::to_string),
                    raw: item.clone(),
                    usage: Vec::new(),
                });
            }
            _ => {}
        }
    }

    (text, client_calls, server_uses)
}

fn log_usage(model: &str, usage: Option<&UsageRecord>, elapsed: Duration) {
    let duration_ms = elapsed.as_millis() as u64;
    match usage {
        Some(u) => tracing::info!(
            target: "xai_usage",
            model = %model,
            input_tokens = u.input_tokens.unwrap_or(0),
            cached_tokens = u.cached_input_tokens.unwrap_or(0),
            output_tokens = u.output_tokens.unwrap_or(0),
            reasoning_tokens = u.reasoning_tokens.unwrap_or(0),
            total_tokens = u.total_tokens.unwrap_or(0),
            cost = ?u.cost,
            duration_ms,
            "xai responses request complete",
        ),
        None => tracing::info!(
            target: "xai_usage",
            model = %model,
            duration_ms,
            "xai responses request complete; no usage reported",
        ),
    }
}

fn usage_from_xai(
    provider: &ProviderName,
    model: Option<ModelId>,
    subject: UsageSubject,
    usage: Option<&Value>,
) -> Option<UsageRecord> {
    let raw = usage?.clone();
    let parsed = serde_json::from_value::<Usage>(raw.clone()).ok()?;
    // xAI reports authoritative micro-dollar ticks directly, so mark them as
    // non-estimated instead of applying local pricing tables.
    let cost = (parsed.cost_in_usd_ticks > 0).then(|| CostAmount {
        amount: parsed.cost_in_usd_ticks.to_string(),
        unit: "usd_ticks".to_string(),
        estimated: false,
    });
    Some(UsageRecord {
        provider: provider.clone(),
        model,
        subject,
        input_tokens: Some(parsed.input_tokens),
        cached_input_tokens: Some(parsed.input_tokens_details.cached_tokens),
        output_tokens: Some(parsed.output_tokens),
        reasoning_tokens: Some(parsed.output_tokens_details.reasoning_tokens),
        total_tokens: Some(parsed.total_tokens),
        cost,
        raw: Some(raw),
    })
}

/// xAI-specific model request options supplied through `provider_options.value`.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct XaiOptions {
    /// Reasoning effort: `low`, `medium`, or `high`.
    #[serde(default)]
    pub reasoning_effort: Option<String>,
}

impl XaiOptions {
    /// Decode routed provider options, treating absent or malformed values as defaults.
    fn from_request(request: &ModelStepRequest) -> Self {
        request
            .provider_options
            .as_ref()
            .and_then(|opts| serde_json::from_value(opts.value.clone()).ok())
            .unwrap_or_default()
    }
}

/// Minimal `/responses` payload fields consumed by the LLM backend.
#[derive(Deserialize)]
struct ResponsesResponse {
    #[serde(default)]
    output: Vec<Value>,
    #[serde(default)]
    citations: Option<Value>,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    usage: Option<Value>,
}

/// xAI usage payload normalized into Chudbot usage records.
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
    #[serde(default)]
    cost_in_usd_ticks: u64,
}

/// Nested token details reused by xAI input and output usage sections.
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
    use chudbot_api::{ProviderOptions, TranscriptTurn};

    #[test]
    fn synthesized_role_messages_include_stable_ids() {
        let provider = ProviderName::new("xai");
        let mut transcript = Transcript::new();
        transcript.id = Some("conv-123".to_string());
        transcript.instructions = Some("be helpful".to_string());
        transcript.push(TranscriptTurn {
            role: TurnRole::User,
            blocks: vec![ContentBlock::Text {
                text: "hi".to_string(),
            }],
            metadata: json!({ "id": "chudbot_turn_user_1" }),
        });

        let input =
            futures::executor::block_on(to_responses_input(&transcript, &provider)).unwrap();

        assert_eq!(input.len(), 2);
        assert_eq!(input[0]["id"], "chudbot_conversation_conv-123_system");
        assert_eq!(input[0]["role"], "system");
        assert_eq!(input[1]["id"], "chudbot_turn_user_1");
        assert_eq!(input[1]["role"], "user");
    }

    #[test]
    fn replays_full_output_ids_verbatim_when_present() {
        let provider = ProviderName::new("xai");
        let mut transcript = Transcript::new();
        transcript.push(TranscriptTurn {
            role: TurnRole::Assistant,
            blocks: vec![
                ContentBlock::Continuation(ProviderContinuation {
                    provider: provider.clone(),
                    data: json!([
                        { "type": "reasoning", "id": "rs_1", "encrypted_content": "BLOB" },
                        {
                            "type": "message",
                            "role": "assistant",
                            "id": "msg_1",
                            "content": [{ "type": "output_text", "text": "the answer" }],
                        },
                        {
                            "type": "function_call",
                            "id": "fc_1",
                            "call_id": "call_1",
                            "name": "fetch_messages",
                            "arguments": "{}",
                        },
                        { "type": "web_search_call", "id": "ws_1", "status": "completed" },
                    ]),
                }),
                ContentBlock::Text {
                    text: "the answer".to_string(),
                },
            ],
            metadata: json!({ "id": "synthetic_assistant_id" }),
        });

        let input =
            futures::executor::block_on(to_responses_input(&transcript, &provider)).unwrap();

        assert_eq!(input.len(), 4);
        assert_eq!(input[0]["type"], "reasoning");
        assert_eq!(input[0]["id"], "rs_1");
        assert_eq!(input[1]["type"], "message");
        assert_eq!(input[1]["id"], "msg_1");
        assert_eq!(input[2]["type"], "function_call");
        assert_eq!(input[2]["id"], "fc_1");
        assert_eq!(input[2]["call_id"], "call_1");
        assert_eq!(input[3]["type"], "web_search_call");
        assert_eq!(input[3]["id"], "ws_1");
    }

    #[test]
    fn skips_malformed_encrypted_content_when_replaying_full_output() {
        let provider = ProviderName::new("xai");
        let mut transcript = Transcript::new();
        transcript.push(TranscriptTurn {
            role: TurnRole::Assistant,
            blocks: vec![
                ContentBlock::Continuation(ProviderContinuation {
                    provider: provider.clone(),
                    data: json!([
                        { "type": "reasoning", "id": "rs_good", "encrypted_content": "BLOB_1+/=" },
                        { "type": "web_search_call", "id": "ws_1", "status": "completed" },
                        { "type": "reasoning", "id": "rs_bad", "encrypted_content": "bad,blob" },
                        {
                            "type": "message",
                            "role": "assistant",
                            "id": "msg_1",
                            "content": [{ "type": "output_text", "text": "the answer" }],
                        },
                    ]),
                }),
                ContentBlock::Text {
                    text: "the answer".to_string(),
                },
            ],
            metadata: Value::Null,
        });

        let input =
            futures::executor::block_on(to_responses_input(&transcript, &provider)).unwrap();

        assert_eq!(input.len(), 3);
        assert_eq!(input[0]["id"], "rs_good");
        assert_eq!(input[1]["id"], "ws_1");
        assert_eq!(input[2]["id"], "msg_1");
        assert!(
            input
                .iter()
                .all(|item| item.get("id").and_then(Value::as_str) != Some("rs_bad"))
        );
    }

    #[test]
    fn falls_back_to_text_when_continuation_items_are_malformed() {
        let provider = ProviderName::new("xai");
        let mut transcript = Transcript::new();
        transcript.push(TranscriptTurn {
            role: TurnRole::Assistant,
            blocks: vec![
                ContentBlock::Continuation(ProviderContinuation {
                    provider: provider.clone(),
                    data: json!([
                        { "type": "reasoning", "id": "rs_bad", "encrypted_content": "bad,blob" },
                    ]),
                }),
                ContentBlock::Text {
                    text: "the answer".to_string(),
                },
            ],
            metadata: json!({ "id": "synthetic_assistant_id" }),
        });

        let input =
            futures::executor::block_on(to_responses_input(&transcript, &provider)).unwrap();

        assert_eq!(input.len(), 1);
        assert_eq!(input[0]["id"], "synthetic_assistant_id");
        assert_eq!(input[0]["role"], "assistant");
        assert_eq!(input[0]["content"], "the answer");
    }

    #[test]
    fn sanitizes_new_continuations_before_persisting() {
        let provider = ProviderName::new("xai");
        let output = vec![
            json!({ "type": "reasoning", "id": "rs_bad", "encrypted_content": "bad,blob" }),
            json!({
                "type": "message",
                "role": "assistant",
                "id": "msg_1",
                "content": [{ "type": "output_text", "text": "the answer" }],
            }),
        ];

        let continuation = continuation_from_output(&provider, &output).unwrap();
        let items = continuation.data.as_array().unwrap();

        assert_eq!(items.len(), 1);
        assert_eq!(items[0]["id"], "msg_1");
    }

    #[test]
    fn xai_options_parse_routed_provider_value() {
        let request = ModelStepRequest {
            model: ModelId::new("grok-4.3"),
            transcript: Transcript::from_user_text("hi"),
            client_tools: BTreeMap::new(),
            server_tools: ServerToolSet::new(),
            sampling: chudbot_api::SamplingOptions::default(),
            provider_options: Some(ProviderOptions {
                value: json!({ "reasoning_effort": "high" }),
            }),
        };

        assert_eq!(
            XaiOptions::from_request(&request)
                .reasoning_effort
                .as_deref(),
            Some("high")
        );
    }
}
