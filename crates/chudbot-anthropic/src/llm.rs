//! Anthropic Messages API adapter for Chudbot's provider-neutral LLM contract.
//!
//! This module owns the translation between `Transcript` turns and Anthropic
//! Messages request blocks, then maps Anthropic response blocks back into
//! ordered model-step output containing content, client tool calls, server tool
//! usage, grounding, continuation state, and token usage. Provider-specific
//! wire shapes stay here so the rest of the bot can operate on `chudbot-api`
//! types.

use std::collections::{BTreeMap, HashMap};
use std::time::{Duration, Instant};

use base64::Engine;
use base64::engine::general_purpose::STANDARD as B64;
use chudbot_api::reasoning::TurnReasoning;
use chudbot_api::sse::{ServerSentEvent, SseDecoder};
use chudbot_api::{
    ClientToolCall, ClientToolResult, ClientToolResultContent, ClientToolSpec, ContentBlock,
    GroundingMetadata, LlmBackend, MediaRef, ModelId, ModelInfo, ModelInfoRequest, ModelStepDelta,
    ModelStepEvent, ModelStepKind, ModelStepRequest, ProviderContinuation, ProviderName,
    ServerToolSet, ServerToolUse, ToolInputSchema, ToolName, ToolUseId, Transcript, TurnRole,
    UsageRecord, UsageSubject, reasoning_items_to_delta_events,
};
use futures::{Stream, StreamExt};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::pricing::{AnthropicPricing, AnthropicTokenUsage};
use crate::{AnthropicClient, AnthropicError};

const DEFAULT_MAX_OUTPUT_TOKENS: u32 = 4096;
const WEB_SEARCH_TOOL_TYPE: &str = "web_search_20250305";
const WEB_SEARCH_TOOL_NAME: &str = "web_search";

impl AnthropicClient {
    async fn build_step_body(&self, request: &ModelStepRequest) -> Result<Value, AnthropicError> {
        let (system, mut messages) =
            to_anthropic_messages(&request.transcript, self.provider_name()).await?;
        mark_last_block_ephemeral(&mut messages);

        let tools = build_messages_tools(&request.client_tools, &request.server_tools);
        let options = AnthropicOptions::from_request(request);
        serde_json::to_value(AnthropicRequest {
            model: request.model.as_str(),
            max_tokens: request
                .sampling
                .max_output_tokens
                .unwrap_or(DEFAULT_MAX_OUTPUT_TOKENS),
            messages: &messages,
            system,
            tools: &tools,
            temperature: request.sampling.temperature,
            top_p: request.sampling.top_p,
            thinking: options.thinking.as_ref(),
            effort: options.effort.as_deref(),
            stream: true,
        })
        .map_err(|e| AnthropicError::Decode(e.to_string()))
    }
}

impl LlmBackend for AnthropicClient {
    type Error = AnthropicError;

    fn backend_name(&self) -> &ProviderName {
        self.provider_name()
    }

    #[tracing::instrument(name = "anthropic.step", skip_all, fields(model = %request.model))]
    fn step(
        &self,
        request: ModelStepRequest,
    ) -> impl Stream<Item = Result<ModelStepEvent, Self::Error>> + Send + '_ {
        async_stream::try_stream! {
            let started = Instant::now();
            let requested_model = request.model.clone();
            let body = self.build_step_body(&request).await?;
            let resp = self
                .post_json_stream("/messages", &body, "llm[anthropic.stream]")
                .await?;
            let chunks = resp.bytes_stream();
            futures::pin_mut!(chunks);
            let mut decoder = SseDecoder::new();
            let mut state = AnthropicStreamState::default();

            while let Some(chunk) = chunks.next().await {
                let chunk = chunk.map_err(|error| AnthropicError::Transport(error.to_string()))?;
                for event in decoder
                    .push(&chunk)
                    .map_err(|error| AnthropicError::Decode(error.to_string()))?
                {
                    let outcome = anthropic_stream_event(
                        event,
                        self.provider_name(),
                        &requested_model,
                        self.pricing(),
                        started,
                        &mut state,
                    )?;
                    for event in outcome.events {
                        yield event;
                    }
                    if outcome.finished {
                        return;
                    }
                }
            }

            if let Some(event) = decoder
                .finish()
                .map_err(|error| AnthropicError::Decode(error.to_string()))?
            {
                let outcome = anthropic_stream_event(
                    event,
                    self.provider_name(),
                    &requested_model,
                    self.pricing(),
                    started,
                    &mut state,
                )?;
                for event in outcome.events {
                    yield event;
                }
                if outcome.finished {
                    return;
                }
            }

            Err(AnthropicError::Decode(
                "Anthropic stream ended without message_stop".to_string(),
            ))?;
        }
    }

    #[tracing::instrument(name = "anthropic.model_info", skip_all, fields(model = %request.model))]
    async fn fetch_model_info(
        &self,
        request: ModelInfoRequest,
    ) -> Result<Option<ModelInfo>, Self::Error> {
        let endpoint = format!("/models/{}", request.model.as_str());
        let raw: Value = self.get_json(&endpoint, "model_info[anthropic]").await?;
        Ok(Some(model_info_from_anthropic_model(request.model, raw)))
    }
}

#[derive(Debug, Default)]
struct AnthropicStreamState {
    model_id: Option<ModelId>,
    stop_reason: Option<String>,
    start_usage: Option<Value>,
    latest_usage: Option<Value>,
    blocks: BTreeMap<usize, AnthropicStreamBlock>,
    emitted_text_delta: bool,
    emitted_reasoning_delta: bool,
}

#[derive(Debug, Default)]
struct AnthropicStreamBlock {
    raw: Value,
    input_json: String,
    emitted_client_delta: bool,
}

#[derive(Debug, Default)]
struct StreamEventOutcome {
    events: Vec<ModelStepEvent>,
    finished: bool,
}

fn anthropic_stream_event(
    event: ServerSentEvent,
    provider: &ProviderName,
    requested_model: &ModelId,
    pricing: &AnthropicPricing,
    started: Instant,
    state: &mut AnthropicStreamState,
) -> Result<StreamEventOutcome, AnthropicError> {
    let value = serde_json::from_str::<Value>(&event.data).map_err(|error| {
        AnthropicError::Decode(format!("failed to decode Anthropic SSE event: {error}"))
    })?;
    let event_type = value
        .get("type")
        .and_then(Value::as_str)
        .or(event.event.as_deref())
        .unwrap_or("");

    match event_type {
        "message_start" => {
            if let Some(message) = value.get("message") {
                if let Some(model) = message.get("model").and_then(Value::as_str) {
                    state.model_id = Some(ModelId::new(model));
                }
                if let Some(usage) = message.get("usage")
                    && !usage.is_null()
                {
                    state.start_usage = Some(usage.clone());
                }
            }
            Ok(StreamEventOutcome::default())
        }
        "content_block_start" => {
            let Some(index) = stream_index(&value)? else {
                return Ok(StreamEventOutcome::default());
            };
            let raw = value
                .get("content_block")
                .cloned()
                .unwrap_or_else(|| json!({ "type": "unknown" }));
            let mut events = Vec::new();
            if raw.get("type").and_then(Value::as_str) == Some("text")
                && let Some(text) = raw.get("text").and_then(Value::as_str)
                && !text.is_empty()
            {
                state.emitted_text_delta = true;
                events.push(ModelStepEvent::Delta(ModelStepDelta::Text {
                    item_id: format!("anthropic_text:{index}"),
                    delta: text.to_string(),
                }));
            }
            state.blocks.insert(
                index,
                AnthropicStreamBlock {
                    raw,
                    input_json: String::new(),
                    emitted_client_delta: false,
                },
            );
            Ok(StreamEventOutcome {
                events,
                finished: false,
            })
        }
        "content_block_delta" => {
            let Some(index) = stream_index(&value)? else {
                return Ok(StreamEventOutcome::default());
            };
            let Some(delta) = value.get("delta") else {
                return Ok(StreamEventOutcome::default());
            };
            let mut events = Vec::new();
            let block = state
                .blocks
                .entry(index)
                .or_insert_with(|| AnthropicStreamBlock {
                    raw: json!({ "type": "unknown" }),
                    input_json: String::new(),
                    emitted_client_delta: false,
                });
            match delta.get("type").and_then(Value::as_str).unwrap_or("") {
                "text_delta" => {
                    if let Some(text) = delta.get("text").and_then(Value::as_str)
                        && !text.is_empty()
                    {
                        state.emitted_text_delta = true;
                        append_string_field(&mut block.raw, "text", text);
                        events.push(ModelStepEvent::Delta(ModelStepDelta::Text {
                            item_id: format!("anthropic_text:{index}"),
                            delta: text.to_string(),
                        }));
                    }
                }
                "input_json_delta" => {
                    let partial = delta
                        .get("partial_json")
                        .and_then(Value::as_str)
                        .unwrap_or("");
                    block.input_json.push_str(partial);
                    if block.raw.get("type").and_then(Value::as_str) == Some("tool_use") {
                        block.emitted_client_delta = true;
                        events.push(ModelStepEvent::Delta(ModelStepDelta::ClientToolCall {
                            item_id: format!("anthropic_tool:{index}"),
                            id: ToolUseId::new(
                                block.raw.get("id").and_then(Value::as_str).unwrap_or(""),
                            ),
                            name: block
                                .raw
                                .get("name")
                                .and_then(Value::as_str)
                                .map(ToolName::new),
                            arguments_delta: partial.to_string(),
                        }));
                    }
                }
                "thinking_delta" => {
                    if let Some(thinking) = delta.get("thinking").and_then(Value::as_str)
                        && !thinking.is_empty()
                    {
                        state.emitted_reasoning_delta = true;
                        append_string_field(&mut block.raw, "thinking", thinking);
                        events.push(ModelStepEvent::Delta(ModelStepDelta::ReasoningSummary {
                            item_id: format!("anthropic_reasoning:{index}"),
                            provider: provider.clone(),
                            kind: Some("thinking".to_string()),
                            delta: thinking.to_string(),
                        }));
                    }
                }
                "signature_delta" => {
                    if let Some(signature) = delta.get("signature").and_then(Value::as_str) {
                        set_field(
                            &mut block.raw,
                            "signature",
                            Value::String(signature.to_string()),
                        );
                    }
                }
                _ => {}
            }
            Ok(StreamEventOutcome {
                events,
                finished: false,
            })
        }
        "content_block_stop" => {
            let Some(index) = stream_index(&value)? else {
                return Ok(StreamEventOutcome::default());
            };
            let events = state
                .blocks
                .get_mut(&index)
                .map(|block| finish_anthropic_block(index, block))
                .transpose()?
                .unwrap_or_default();
            Ok(StreamEventOutcome {
                events,
                finished: false,
            })
        }
        "message_delta" => {
            if let Some(delta) = value.get("delta")
                && let Some(stop_reason) = delta.get("stop_reason").and_then(Value::as_str)
            {
                state.stop_reason = Some(stop_reason.to_string());
            }
            if let Some(usage) = value.get("usage")
                && !usage.is_null()
            {
                state.latest_usage = Some(usage.clone());
            }
            Ok(StreamEventOutcome::default())
        }
        "message_stop" => {
            let events =
                anthropic_terminal_events(provider, requested_model, pricing, started, state);
            Ok(StreamEventOutcome {
                events,
                finished: true,
            })
        }
        "error" => Err(AnthropicError::Decode(provider_error_message(
            &value,
            "Anthropic stream returned an error event",
        ))),
        "ping" => Ok(StreamEventOutcome::default()),
        _ => Ok(StreamEventOutcome::default()),
    }
}

fn stream_index(value: &Value) -> Result<Option<usize>, AnthropicError> {
    value
        .get("index")
        .and_then(Value::as_u64)
        .map(|index| {
            usize::try_from(index).map_err(|_| {
                AnthropicError::Decode(format!("Anthropic content index `{index}` is too large"))
            })
        })
        .transpose()
}

fn finish_anthropic_block(
    index: usize,
    block: &mut AnthropicStreamBlock,
) -> Result<Vec<ModelStepEvent>, AnthropicError> {
    let mut events = Vec::new();
    if !block.input_json.trim().is_empty() {
        let input = serde_json::from_str::<Value>(&block.input_json).map_err(|error| {
            AnthropicError::Decode(format!(
                "failed to decode Anthropic input JSON delta for block {index}: {error}"
            ))
        })?;
        set_field(&mut block.raw, "input", input);
    }
    if block.raw.get("type").and_then(Value::as_str) == Some("tool_use")
        && !block.emitted_client_delta
    {
        let input = block.raw.get("input").cloned().unwrap_or_else(|| json!({}));
        let arguments = serde_json::to_string(&input).unwrap_or_else(|_| "{}".to_string());
        events.push(ModelStepEvent::Delta(ModelStepDelta::ClientToolCall {
            item_id: format!("anthropic_tool:{index}"),
            id: ToolUseId::new(block.raw.get("id").and_then(Value::as_str).unwrap_or("")),
            name: block
                .raw
                .get("name")
                .and_then(Value::as_str)
                .map(ToolName::new),
            arguments_delta: arguments,
        }));
    }
    Ok(events)
}

fn anthropic_terminal_events(
    provider: &ProviderName,
    requested_model: &ModelId,
    pricing: &AnthropicPricing,
    started: Instant,
    state: &AnthropicStreamState,
) -> Vec<ModelStepEvent> {
    let model_id = state
        .model_id
        .clone()
        .unwrap_or_else(|| requested_model.clone());
    let usage_raw = merged_anthropic_usage(state);
    let usage = usage_from_anthropic(
        provider,
        Some(model_id.clone()),
        UsageSubject::ModelStep,
        usage_raw.as_ref(),
        pricing,
    );
    log_usage(model_id.as_str(), usage.as_ref(), started.elapsed());

    let content = state
        .blocks
        .values()
        .map(|block| block.raw.clone())
        .collect::<Vec<_>>();
    let (text, client_tool_calls, server_tool_uses, grounding) = walk_blocks(&content, provider);
    let kind = if !client_tool_calls.is_empty() {
        ModelStepKind::ClientTools
    } else if state.stop_reason.as_deref() == Some("pause_turn") || text.is_empty() {
        ModelStepKind::Continue
    } else {
        ModelStepKind::Final
    };
    let continuation = continuation_from_content(provider, &content);

    let mut events = Vec::new();
    if !state.emitted_text_delta && !text.is_empty() {
        events.push(ModelStepEvent::Delta(ModelStepDelta::Text {
            item_id: "anthropic_text:terminal".to_string(),
            delta: text,
        }));
    }
    if !state.emitted_reasoning_delta
        && let Some(continuation) = continuation.as_ref()
    {
        events.extend(reasoning_items_to_delta_events(
            TurnReasoning::from_continuation_and_usage(Some(continuation), Some(&model_id), &[])
                .items,
            "anthropic_reasoning",
        ));
    }
    if let Some(continuation) = continuation {
        events.push(ModelStepEvent::Continuation(continuation));
    }
    events.extend(
        server_tool_uses
            .into_iter()
            .map(ModelStepEvent::ServerToolUse),
    );
    events.extend(grounding.into_iter().map(ModelStepEvent::Grounding));
    events.extend(usage.into_iter().map(ModelStepEvent::Usage));
    events.push(ModelStepEvent::Finished { kind, model_id });
    events
}

fn merged_anthropic_usage(state: &AnthropicStreamState) -> Option<Value> {
    let mut usage = state.start_usage.clone().unwrap_or_else(|| json!({}));
    if let Some(latest) = &state.latest_usage
        && let (Some(target), Some(source)) = (usage.as_object_mut(), latest.as_object())
    {
        for (key, value) in source {
            target.insert(key.clone(), value.clone());
        }
    }
    usage
        .as_object()
        .is_some_and(|map| !map.is_empty())
        .then_some(usage)
}

fn append_string_field(value: &mut Value, field: &str, delta: &str) {
    if let Value::Object(map) = value {
        let entry = map
            .entry(field.to_string())
            .or_insert_with(|| Value::String(String::new()));
        if let Value::String(text) = entry {
            text.push_str(delta);
        }
    }
}

fn set_field(value: &mut Value, field: &str, next: Value) {
    if let Value::Object(map) = value {
        map.insert(field.to_string(), next);
    }
}

fn provider_error_message(value: &Value, fallback: &str) -> String {
    value
        .get("error")
        .and_then(|error| error.get("message").or_else(|| error.get("type")))
        .and_then(Value::as_str)
        .or_else(|| value.get("message").and_then(Value::as_str))
        .unwrap_or(fallback)
        .to_string()
}

/// Convert an Anthropic model document into Chudbot's optional model metadata.
fn model_info_from_anthropic_model(requested_model: ModelId, raw: Value) -> ModelInfo {
    let id = raw
        .get("id")
        .and_then(Value::as_str)
        .filter(|id| !id.is_empty())
        .map(ModelId::new)
        .unwrap_or(requested_model);
    ModelInfo {
        id,
        context_window_tokens: raw.get("max_input_tokens").and_then(value_as_u64),
        max_output_tokens: raw.get("max_tokens").and_then(value_as_u64),
        raw: Some(raw),
    }
}

/// Accept numeric limits whether Anthropic returns JSON numbers or strings.
fn value_as_u64(value: &Value) -> Option<u64> {
    match value {
        Value::Number(number) => number.as_u64(),
        Value::String(text) => text.parse().ok(),
        _ => None,
    }
}

/// Render the provider-neutral transcript into Anthropic Messages inputs.
///
/// Instructions become a cached `system` block, prior Anthropic continuations
/// are replayed verbatim, and fresh Chudbot content blocks are converted into
/// the closest Anthropic block type.
async fn to_anthropic_messages(
    transcript: &Transcript,
    provider: &ProviderName,
) -> Result<(Option<Value>, Vec<Value>), AnthropicError> {
    let system = transcript
        .instructions
        .as_ref()
        .filter(|instructions| !instructions.is_empty())
        .map(|instructions| {
            json!([{
                "type": "text",
                "text": instructions,
                "cache_control": { "type": "ephemeral" },
            }])
        });

    let mut messages = Vec::new();
    for turn in &transcript.turns {
        let role = match turn.role {
            TurnRole::Assistant => "assistant",
            TurnRole::User => "user",
        };

        // A stored continuation supersedes reconstructed blocks for this turn:
        // Anthropic needs its original response block sequence for pause_turn,
        // server-tool, and encrypted/thinking-compatible continuations.
        let mut content = provider_continuation_content(turn, provider);

        if !content.is_empty() {
            messages.push(json!({ "role": role, "content": content }));
            continue;
        }

        for block in &turn.blocks {
            match block {
                ContentBlock::Text { text } if !text.is_empty() => {
                    content.push(json!({ "type": "text", "text": text }));
                }
                ContentBlock::Text { .. } => {}
                ContentBlock::Media { media } => {
                    content.push(json!({
                        "type": "image",
                        "source": media_source(media.as_ref()).await?,
                    }));
                }
                ContentBlock::ClientToolCall(call) => {
                    content.push(json!({
                        "type": "tool_use",
                        "id": call.id.as_str(),
                        "name": call.name.as_str(),
                        "input": call.input.clone(),
                    }));
                }
                ContentBlock::ClientToolResult(result) => {
                    content.push(tool_result_block(result));
                }
                ContentBlock::Continuation(continuation) => {
                    if &continuation.provider == provider {
                        tracing::debug!(
                            provider = %provider,
                            "skipping empty Anthropic provider continuation",
                        );
                    }
                }
            }
        }

        if !content.is_empty() {
            messages.push(json!({ "role": role, "content": content }));
        }
    }

    Ok((system, messages))
}

/// Pull Anthropic continuation blocks out of a transcript turn without editing them.
fn provider_continuation_content(
    turn: &chudbot_api::TranscriptTurn,
    provider: &ProviderName,
) -> Vec<Value> {
    let mut content = Vec::new();
    for block in &turn.blocks {
        if let ContentBlock::Continuation(continuation) = block
            && &continuation.provider == provider
        {
            match &continuation.data {
                Value::Array(items) => content.extend(items.iter().cloned()),
                other => content.push(other.clone()),
            }
        }
    }
    content
}

/// Convert a Chudbot media reference into one of Anthropic's accepted image sources.
async fn media_source(media: &dyn MediaRef) -> Result<Value, AnthropicError> {
    let mime_type = media.mime_type();
    if !mime_type.starts_with("image/") {
        return Err(AnthropicError::Reference(format!(
            "media `{}` has MIME type `{mime_type}`, but Anthropic Messages accepts image media here",
            media.uri()
        )));
    }

    match media.load().await {
        // Prefer inline bytes when the media store can provide them; this keeps
        // private local assets usable without requiring an externally reachable
        // media URL.
        Ok(loaded) => Ok(json!({
            "type": "base64",
            "media_type": loaded.media.mime_type(),
            "data": B64.encode(&loaded.bytes),
        })),
        Err(load_error) => match media.public_url().await {
            Ok(url) => Ok(json!({
                "type": "url",
                "url": url.as_str(),
            })),
            Err(public_error) => Err(AnthropicError::Reference(format!(
                "media `{}` could not be loaded ({load_error}) and has no public URL ({public_error})",
                media.uri()
            ))),
        },
    }
}

/// Serialize a client tool result in Anthropic's `tool_result` block shape.
fn tool_result_block(result: &ClientToolResult) -> Value {
    let mut obj = serde_json::Map::new();
    obj.insert("type".into(), Value::String("tool_result".into()));
    obj.insert(
        "tool_use_id".into(),
        Value::String(result.tool_use_id.as_str().to_string()),
    );
    obj.insert(
        "content".into(),
        Value::String(client_tool_result_as_string(result)),
    );
    if result.is_error {
        obj.insert("is_error".into(), Value::Bool(true));
    }
    Value::Object(obj)
}

/// Anthropic accepts tool results as text, so JSON results are compacted first.
fn client_tool_result_as_string(result: &ClientToolResult) -> String {
    match &result.content {
        ClientToolResultContent::Json { value } => {
            serde_json::to_string(value).unwrap_or_else(|_| value.to_string())
        }
        ClientToolResultContent::Text { text } => text.clone(),
    }
}

/// Build the mixed client-tool and Anthropic-hosted server-tool declaration list.
fn build_messages_tools(
    client_tools: &BTreeMap<ToolName, ClientToolSpec>,
    server_tools: &ServerToolSet,
) -> Vec<Value> {
    let mut tools = Vec::with_capacity(client_tools.len() + 1);
    for (name, tool) in client_tools {
        tools.push(json!({
            "name": name.as_str(),
            "description": tool.description.as_str(),
            "input_schema": anthropic_tool_input_schema(&tool.input_schema),
        }));
    }
    if server_tools.contains("web_search") {
        // Chudbot's server-tool set is provider-neutral; Anthropic exposes web
        // search as a hosted Messages tool with a dated tool type.
        tools.push(json!({
            "type": WEB_SEARCH_TOOL_TYPE,
            "name": WEB_SEARCH_TOOL_NAME,
            "max_uses": 5,
        }));
    }
    tools
}

fn anthropic_tool_input_schema(input_schema: &ToolInputSchema) -> Value {
    serde_json::to_value(input_schema).expect("tool input schema serializes")
}

/// Split Anthropic response content into the pieces the agent loop consumes.
fn walk_blocks(
    blocks: &[Value],
    provider: &ProviderName,
) -> (
    String,
    Vec<ClientToolCall>,
    Vec<ServerToolUse>,
    Vec<GroundingMetadata>,
) {
    let mut text = String::new();
    let mut pending_server_uses: HashMap<String, Value> = HashMap::new();
    let mut server_uses = Vec::new();
    let mut client_uses = Vec::new();
    let mut grounding = Vec::new();

    for block in blocks {
        let kind = block.get("type").and_then(Value::as_str).unwrap_or("");
        match kind {
            "text" => {
                if let Some(t) = block.get("text").and_then(Value::as_str) {
                    text.push_str(t);
                }
                if let Some(citations) = block.get("citations")
                    && !citations.is_null()
                {
                    grounding.push(GroundingMetadata {
                        provider: provider.clone(),
                        raw: citations.clone(),
                    });
                }
            }
            "server_tool_use" => {
                // Anthropic sends hosted-tool request and result blocks
                // separately. Hold the request until its result arrives so the
                // trace can show one complete server-tool use.
                let id = block
                    .get("id")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string();
                pending_server_uses.insert(id, block.clone());
            }
            "web_search_tool_result" => {
                let id = block
                    .get("tool_use_id")
                    .and_then(Value::as_str)
                    .unwrap_or("");
                if let Some(request) = pending_server_uses.remove(id) {
                    server_uses.push(server_tool_use_from_pair(provider, request, block.clone()));
                } else {
                    // Preserve orphaned results instead of dropping trace data;
                    // some partial/provider-error responses may omit the request
                    // block even though the result carries useful status/raw data.
                    server_uses.push(ServerToolUse {
                        provider: provider.clone(),
                        name: ToolName::new(WEB_SEARCH_TOOL_NAME),
                        id: (!id.is_empty()).then(|| id.to_string()),
                        status: block
                            .get("status")
                            .and_then(Value::as_str)
                            .map(str::to_string),
                        raw: block.clone(),
                        usage: Vec::new(),
                    });
                }
            }
            "tool_use" => {
                let id = block.get("id").and_then(Value::as_str).unwrap_or("");
                let name = block.get("name").and_then(Value::as_str).unwrap_or("");
                let input = block.get("input").cloned().unwrap_or(Value::Null);
                client_uses.push(ClientToolCall {
                    id: ToolUseId::new(id),
                    name: ToolName::new(name),
                    input,
                });
            }
            _ => {}
        }
    }

    for (_, request) in pending_server_uses {
        // Surface unmatched requests as trace events so hosted-tool starts are
        // still visible if Anthropic stops before producing a result block.
        server_uses.push(server_tool_use_from_request(provider, request));
    }

    (text, client_uses, server_uses, grounding)
}

/// Store Anthropic's raw content array for exact replay on the next request.
fn continuation_from_content(
    provider: &ProviderName,
    content: &[Value],
) -> Option<ProviderContinuation> {
    (!content.is_empty()).then_some(ProviderContinuation {
        provider: provider.clone(),
        data: Value::Array(content.to_vec()),
    })
}

/// Classify the collected output for the provider-neutral agent loop.
#[cfg(test)]
fn model_step_from_output(
    output: chudbot_api::ModelStepOutput,
    stop_reason: Option<&str>,
) -> chudbot_api::ModelStep {
    if output.client_tool_calls().next().is_some() {
        chudbot_api::ModelStep::new(ModelStepKind::ClientTools, output)
    } else if stop_reason == Some("pause_turn") || output.answer_blocks().is_empty() {
        chudbot_api::ModelStep::new(ModelStepKind::Continue, output)
    } else {
        chudbot_api::ModelStep::new(ModelStepKind::Final, output)
    }
}

/// Combine an Anthropic hosted-tool request and result into one trace record.
fn server_tool_use_from_pair(
    provider: &ProviderName,
    request: Value,
    response: Value,
) -> ServerToolUse {
    let name = request
        .get("name")
        .and_then(Value::as_str)
        .unwrap_or(WEB_SEARCH_TOOL_NAME);
    let id = request
        .get("id")
        .and_then(Value::as_str)
        .map(str::to_string)
        .or_else(|| {
            response
                .get("tool_use_id")
                .and_then(Value::as_str)
                .map(str::to_string)
        });
    let status = response
        .get("status")
        .and_then(Value::as_str)
        .map(str::to_string);
    ServerToolUse {
        provider: provider.clone(),
        name: ToolName::new(name),
        id,
        status,
        raw: json!({
            "request": request,
            "response": response,
        }),
        usage: Vec::new(),
    }
}

/// Convert an unmatched Anthropic hosted-tool request into a trace record.
fn server_tool_use_from_request(provider: &ProviderName, request: Value) -> ServerToolUse {
    let name = request
        .get("name")
        .and_then(Value::as_str)
        .unwrap_or("server_tool");
    ServerToolUse {
        provider: provider.clone(),
        name: ToolName::new(name),
        id: request
            .get("id")
            .and_then(Value::as_str)
            .map(str::to_string),
        status: request
            .get("status")
            .and_then(Value::as_str)
            .map(str::to_string),
        raw: request,
        usage: Vec::new(),
    }
}

/// Emit a compact usage log line for Anthropic requests.
fn log_usage(model: &str, usage: Option<&UsageRecord>, elapsed: Duration) {
    let duration_ms = elapsed.as_millis() as u64;
    match usage {
        Some(u) => tracing::info!(
            target: "anthropic_usage",
            model = %model,
            input_tokens = u.input_tokens.unwrap_or(0),
            cached_tokens = u.cached_input_tokens.unwrap_or(0),
            output_tokens = u.output_tokens.unwrap_or(0),
            total_tokens = u.total_tokens.unwrap_or(0),
            duration_ms,
            "anthropic messages request complete",
        ),
        None => tracing::info!(
            target: "anthropic_usage",
            model = %model,
            duration_ms,
            "anthropic messages request complete; no usage reported",
        ),
    }
}

/// Convert Anthropic's token accounting into Chudbot usage and local cost estimates.
fn usage_from_anthropic(
    provider: &ProviderName,
    model: Option<ModelId>,
    subject: UsageSubject,
    usage: Option<&Value>,
    pricing: &AnthropicPricing,
) -> Option<UsageRecord> {
    let raw = usage?.clone();
    let parsed = serde_json::from_value::<Usage>(raw.clone()).ok()?;
    let cache_creation_5m_input_tokens = parsed.cache_creation_5m_input_tokens();
    let cache_creation_1h_input_tokens = parsed.cache_creation_1h_input_tokens();
    let cache_creation_input_tokens = parsed.cache_creation_input_tokens();
    // Anthropic reports cache writes and reads separately from uncached input.
    // Chudbot's aggregate input count includes all prompt-side token classes.
    let input_tokens = parsed
        .input_tokens
        .saturating_add(cache_creation_input_tokens)
        .saturating_add(parsed.cache_read_input_tokens);
    let cost = pricing.estimate_token_cost(
        model.as_ref(),
        AnthropicTokenUsage {
            input_tokens: parsed.input_tokens,
            cache_creation_5m_input_tokens,
            cache_creation_1h_input_tokens,
            cache_read_input_tokens: parsed.cache_read_input_tokens,
            output_tokens: parsed.output_tokens,
            inference_geo: parsed.inference_geo.as_deref(),
        },
    );
    Some(UsageRecord {
        provider: provider.clone(),
        model,
        subject,
        input_tokens: Some(input_tokens),
        cached_input_tokens: Some(parsed.cache_read_input_tokens),
        output_tokens: Some(parsed.output_tokens),
        reasoning_tokens: None,
        total_tokens: Some(input_tokens + parsed.output_tokens),
        cost,
        raw: Some(raw),
    })
}

/// Anthropic-specific per-request knobs.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct AnthropicOptions {
    /// Anthropic Messages `thinking` block passed through as request JSON.
    ///
    /// Example provider options can include `{ "thinking": { "type": "adaptive" } }`.
    #[serde(default)]
    pub thinking: Option<Value>,
    /// Anthropic effort level passed through to models that support it.
    ///
    /// Common values are `low`, `medium`, `high`, and `max`, though Anthropic
    /// may add model-specific values.
    #[serde(default)]
    pub effort: Option<String>,
}

impl AnthropicOptions {
    /// Decode provider-specific request options, ignoring malformed extras.
    fn from_request(request: &ModelStepRequest) -> Self {
        request
            .provider_options
            .as_ref()
            .and_then(|options| serde_json::from_value(options.value.clone()).ok())
            .unwrap_or_default()
    }
}

/// Borrowed request body for the Anthropic Messages API.
#[derive(Serialize)]
struct AnthropicRequest<'a> {
    model: &'a str,
    max_tokens: u32,
    messages: &'a [Value],
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<Value>,
    #[serde(skip_serializing_if = "<[Value]>::is_empty")]
    tools: &'a [Value],
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking: Option<&'a Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    effort: Option<&'a str>,
    stream: bool,
}

/// Mark the newest content block as Anthropic's prompt-cache breakpoint.
fn mark_last_block_ephemeral(messages: &mut [Value]) {
    if let Some(last) = messages.last_mut()
        && let Some(content) = last.get_mut("content").and_then(Value::as_array_mut)
        && let Some(block) = content.last_mut()
        && let Some(obj) = block.as_object_mut()
    {
        obj.insert("cache_control".into(), json!({ "type": "ephemeral" }));
    }
}

/// Anthropic usage payload with both legacy and detailed cache-write fields.
#[derive(Deserialize, Debug, Default)]
struct Usage {
    #[serde(default)]
    input_tokens: u64,
    #[serde(default)]
    cache_creation_input_tokens: u64,
    #[serde(default)]
    cache_creation: CacheCreationUsage,
    #[serde(default)]
    cache_read_input_tokens: u64,
    #[serde(default)]
    output_tokens: u64,
    #[serde(default)]
    inference_geo: Option<String>,
}

impl Usage {
    /// Return the total cache-write tokens regardless of which field shape is present.
    fn cache_creation_input_tokens(&self) -> u64 {
        self.cache_creation_input_tokens
            .max(self.cache_creation.total_input_tokens())
    }

    /// Treat the old flat cache-write field as 5-minute cache usage.
    fn cache_creation_5m_input_tokens(&self) -> u64 {
        if self.cache_creation.total_input_tokens() == 0 {
            self.cache_creation_input_tokens
        } else {
            self.cache_creation.ephemeral_5m_input_tokens
        }
    }

    /// Return detailed 1-hour cache-write usage when Anthropic reports it.
    fn cache_creation_1h_input_tokens(&self) -> u64 {
        self.cache_creation.ephemeral_1h_input_tokens
    }
}

/// Detailed prompt-cache write counters from newer Anthropic usage payloads.
#[derive(Deserialize, Debug, Default)]
struct CacheCreationUsage {
    #[serde(default)]
    ephemeral_5m_input_tokens: u64,
    #[serde(default)]
    ephemeral_1h_input_tokens: u64,
}

impl CacheCreationUsage {
    /// Sum all detailed cache-write buckets.
    fn total_input_tokens(&self) -> u64 {
        self.ephemeral_5m_input_tokens
            .saturating_add(self.ephemeral_1h_input_tokens)
    }
}

#[cfg(test)]
mod tests {
    use chudbot_api::{
        ClientToolResult, ClientToolResultContent, LoadedMedia, MediaCategory, MediaMetadata,
        MediaRef, MediaUri, ModelOutputBlock, ModelStepItem, ModelStepOutput, ProviderName,
        PublicMediaUrl, ToolInputField, ToolInputSchema, ToolInputValueSchema, ToolUseId,
        TranscriptTurn, TurnRole, UrlMediaRef, collect_model_step,
    };
    use serde_json::json;

    use super::*;

    #[derive(Debug, Clone)]
    struct LoadablePublicMediaRef {
        metadata: MediaMetadata,
        public_url: PublicMediaUrl,
        bytes: Vec<u8>,
    }

    impl LoadablePublicMediaRef {
        fn new(uri: &str, public_url: &str, mime_type: &str, bytes: Vec<u8>) -> Self {
            Self {
                metadata: MediaMetadata {
                    category: MediaCategory::Image,
                    name: "stored-image.png".to_string(),
                    uri: MediaUri::new(uri),
                    mime_type: mime_type.to_string(),
                    size_bytes: u64::try_from(bytes.len()).unwrap_or(u64::MAX),
                },
                public_url: PublicMediaUrl::new(public_url),
                bytes,
            }
        }
    }

    #[async_trait::async_trait]
    impl MediaRef for LoadablePublicMediaRef {
        fn metadata(&self) -> &MediaMetadata {
            &self.metadata
        }

        fn clone_box(&self) -> chudbot_api::BoxedMediaRef {
            Box::new(self.clone())
        }

        async fn public_url(&self) -> Result<PublicMediaUrl, chudbot_api::MediaError> {
            Ok(self.public_url.clone())
        }

        async fn load(&self) -> Result<LoadedMedia, chudbot_api::MediaError> {
            Ok(LoadedMedia {
                media: self.clone_box(),
                bytes: self.bytes.clone(),
            })
        }
    }

    #[test]
    fn builds_anthropic_web_search_tool_only() {
        let mut server_tools = ServerToolSet::new();
        server_tools.insert("web_search".to_string());
        server_tools.insert("x_search".to_string());

        let tools = build_messages_tools(&BTreeMap::new(), &server_tools);

        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["type"], WEB_SEARCH_TOOL_TYPE);
        assert_eq!(tools[0]["name"], WEB_SEARCH_TOOL_NAME);
        assert_eq!(tools[0]["max_uses"], 5);
    }

    #[test]
    fn builds_anthropic_client_tool_schema() {
        let mut client_tools = BTreeMap::new();
        client_tools.insert(
            ToolName::new("fetch_messages"),
            ClientToolSpec {
                description: "Fetch context.".to_string(),
                input_schema: ToolInputSchema::object([ToolInputField::required(
                    "query",
                    ToolInputValueSchema::string().description("Search query."),
                )]),
            },
        );

        let tools = build_messages_tools(&client_tools, &ServerToolSet::new());

        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["name"], "fetch_messages");
        assert_eq!(
            tools[0]["input_schema"],
            json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query."
                    }
                },
                "required": ["query"],
                "additionalProperties": false
            })
        );
    }

    #[test]
    fn replays_anthropic_continuation_content_as_is() {
        let provider = ProviderName::new("anthropic");
        let raw_content = json!([
            {"type": "text", "text": "Searching."},
            {
                "type": "server_tool_use",
                "id": "srvtoolu_1",
                "name": "web_search",
                "input": {"query": "latest rust release"}
            },
            {
                "type": "web_search_tool_result",
                "tool_use_id": "srvtoolu_1",
                "content": [{"type": "web_search_result", "url": "https://example.com"}]
            }
        ]);
        let turn = TranscriptTurn {
            role: TurnRole::Assistant,
            blocks: vec![
                ContentBlock::Text {
                    text: "Searching.".to_string(),
                },
                ContentBlock::Continuation(ProviderContinuation {
                    provider: provider.clone(),
                    data: raw_content.clone(),
                }),
            ],
            metadata: Value::Null,
        };

        assert_eq!(
            provider_continuation_content(&turn, &provider),
            raw_content.as_array().unwrap().clone()
        );
    }

    #[test]
    fn pause_turn_continues_even_with_text_content() {
        let output = ModelStepOutput {
            model_id: ModelId::new("claude-sonnet-4-6"),
            items: vec![
                ModelStepItem::OutputBlock(ModelOutputBlock::Text {
                    text: "I'll search for that.".to_string(),
                }),
                ModelStepItem::OutputBlock(ModelOutputBlock::Continuation(
                    continuation_from_content(
                        &ProviderName::new("anthropic"),
                        &[json!({"type": "text", "text": "I'll search for that."})],
                    )
                    .expect("continuation"),
                )),
            ],
            usage: Vec::new(),
        };

        assert_eq!(
            model_step_from_output(output, Some("pause_turn")).kind,
            ModelStepKind::Continue
        );
    }

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

        let (text, client_uses, server_uses, grounding) =
            walk_blocks(&blocks, &ProviderName::new("anthropic"));
        assert_eq!(text, "Looking that up. Done.");
        assert!(client_uses.is_empty());
        assert!(grounding.is_empty());
        assert_eq!(server_uses.len(), 1);
        assert_eq!(server_uses[0].name.as_str(), "web_search");
        assert_eq!(server_uses[0].id.as_deref(), Some("srvtoolu_1"));
        assert_eq!(
            server_uses[0].raw["request"]["input"]["query"],
            "rust 2024 edition"
        );
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

        let (text, client_uses, server_uses, grounding) =
            walk_blocks(&blocks, &ProviderName::new("anthropic"));
        assert_eq!(text, "Let me fetch recent messages.");
        assert!(server_uses.is_empty());
        assert!(grounding.is_empty());
        assert_eq!(client_uses.len(), 1);
        assert_eq!(client_uses[0].name.as_str(), "fetch_messages");
        assert_eq!(client_uses[0].input["limit"], 30);
    }

    #[test]
    fn streams_anthropic_text_tool_and_usage_events() {
        let provider = ProviderName::new("anthropic");
        let requested_model = ModelId::new("claude-sonnet-4-6");
        let mut state = AnthropicStreamState::default();
        let mut events = Vec::new();
        for data in [
            json!({
                "type": "message_start",
                "message": {
                    "model": "claude-sonnet-4-6",
                    "usage": { "input_tokens": 20, "output_tokens": 1 }
                }
            }),
            json!({
                "type": "content_block_start",
                "index": 0,
                "content_block": { "type": "text", "text": "" }
            }),
            json!({
                "type": "content_block_delta",
                "index": 0,
                "delta": { "type": "text_delta", "text": "Hello" }
            }),
            json!({ "type": "content_block_stop", "index": 0 }),
            json!({
                "type": "content_block_start",
                "index": 1,
                "content_block": {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "fetch_messages",
                    "input": {}
                }
            }),
            json!({
                "type": "content_block_delta",
                "index": 1,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": "{\"limit\":30}"
                }
            }),
            json!({ "type": "content_block_stop", "index": 1 }),
            json!({
                "type": "message_delta",
                "delta": { "stop_reason": "tool_use", "stop_sequence": null },
                "usage": { "output_tokens": 12 }
            }),
            json!({ "type": "message_stop" }),
        ] {
            let outcome = anthropic_stream_event(
                ServerSentEvent {
                    event: None,
                    data: data.to_string(),
                },
                &provider,
                &requested_model,
                &AnthropicPricing::default(),
                Instant::now(),
                &mut state,
            )
            .expect("stream event");
            events.extend(outcome.events);
            if outcome.finished {
                break;
            }
        }

        let step = futures::executor::block_on(collect_model_step(futures::stream::iter(
            events.into_iter().map(Ok::<_, AnthropicError>),
        )))
        .expect("finished step");
        assert!(matches!(step.kind(), ModelStepKind::ClientTools));
        let output = step.output();
        assert_eq!(output.answer_text(), "Hello");
        let calls = output.client_tool_calls().collect::<Vec<_>>();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name.as_str(), "fetch_messages");
        assert_eq!(calls[0].input["limit"], 30);
        assert_eq!(output.usage[0].input_tokens, Some(20));
        assert_eq!(output.usage[0].output_tokens, Some(12));
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

        let last = messages.last().unwrap()["content"].as_array().unwrap();
        assert_eq!(last[1]["cache_control"], json!({ "type": "ephemeral" }));
        assert!(last[0].get("cache_control").is_none());
        assert!(messages[0]["content"][0].get("cache_control").is_none());
    }

    #[tokio::test]
    async fn loadable_media_is_inlined_instead_of_sent_as_url() {
        let media = LoadablePublicMediaRef::new(
            "file://images/stored.png",
            "https://chud.example/media/images/stored.png",
            "image/png",
            b"image bytes".to_vec(),
        );

        let source = media_source(&media).await.expect("media source");

        assert_eq!(source["type"], "base64");
        assert_eq!(source["media_type"], "image/png");
        assert_eq!(source["data"], "aW1hZ2UgYnl0ZXM=");
        assert!(source.get("url").is_none());
    }

    #[tokio::test]
    async fn url_only_media_still_uses_url_source() {
        let media = UrlMediaRef::new(
            MediaCategory::Image,
            "https://example.com/image.png",
            "image/png",
        );

        let source = media_source(&media).await.expect("media source");

        assert_eq!(source["type"], "url");
        assert_eq!(source["url"], "https://example.com/image.png");
    }

    #[test]
    fn json_tool_result_is_sent_as_string_content() {
        let result = ClientToolResult {
            tool_use_id: ToolUseId::new("toolu_1"),
            content: ClientToolResultContent::Json {
                value: json!({ "ok": true }),
            },
            is_error: false,
        };

        let block = tool_result_block(&result);
        assert_eq!(block["type"], "tool_result");
        assert_eq!(block["tool_use_id"], "toolu_1");
        assert_eq!(block["content"], "{\"ok\":true}");
        assert!(block.get("is_error").is_none());
    }

    #[test]
    fn request_body_includes_anthropic_thinking_and_effort_options() {
        let messages = vec![json!({"role": "user", "content": [{"type": "text", "text": "hi"}]})];
        let tools = Vec::new();
        let thinking = json!({"type": "adaptive", "display": "summarized"});

        let body = serde_json::to_value(AnthropicRequest {
            model: "claude-sonnet-4-6",
            max_tokens: 4096,
            messages: &messages,
            system: None,
            tools: &tools,
            temperature: None,
            top_p: None,
            thinking: Some(&thinking),
            effort: Some("medium"),
            stream: true,
        })
        .expect("serialize request");

        assert_eq!(body["thinking"], thinking);
        assert_eq!(body["effort"], "medium");
    }

    #[test]
    fn anthropic_model_info_preserves_token_limits() {
        let info = model_info_from_anthropic_model(
            ModelId::new("claude-haiku-4-5-20251001"),
            json!({
                "id": "claude-haiku-4-5-20251001",
                "type": "model",
                "max_input_tokens": 200000,
                "max_tokens": 64000
            }),
        );

        assert_eq!(info.id, ModelId::new("claude-haiku-4-5-20251001"));
        assert_eq!(info.context_window_tokens, Some(200_000));
        assert_eq!(info.max_output_tokens, Some(64_000));
        assert!(info.raw.is_some());
    }

    #[test]
    fn usage_estimates_cost_with_prompt_cache_details() {
        let usage = json!({
            "input_tokens": 100,
            "cache_creation_input_tokens": 20,
            "cache_creation": {
                "ephemeral_5m_input_tokens": 12,
                "ephemeral_1h_input_tokens": 8
            },
            "cache_read_input_tokens": 40,
            "output_tokens": 10,
            "inference_geo": "global"
        });
        let provider = ProviderName::new("anthropic");

        let record = usage_from_anthropic(
            &provider,
            Some(ModelId::new("claude-sonnet-4-6")),
            UsageSubject::ModelStep,
            Some(&usage),
            &AnthropicPricing::default(),
        )
        .expect("usage record");

        assert_eq!(record.input_tokens, Some(160));
        assert_eq!(record.cached_input_tokens, Some(40));
        assert_eq!(record.output_tokens, Some(10));
        assert_eq!(record.total_tokens, Some(170));
        let cost = record.cost.expect("estimated cost");
        assert_eq!(cost.unit, "usd_ticks");
        assert!(cost.estimated);
        assert_eq!(cost.amount, "5550001");
    }

    #[test]
    fn usage_treats_flat_cache_creation_as_5m_when_details_are_absent() {
        let usage = json!({
            "input_tokens": 100,
            "cache_creation_input_tokens": 20,
            "cache_read_input_tokens": 40,
            "output_tokens": 10,
        });
        let provider = ProviderName::new("anthropic");

        let record = usage_from_anthropic(
            &provider,
            Some(ModelId::new("claude-sonnet-4-6")),
            UsageSubject::ModelStep,
            Some(&usage),
            &AnthropicPricing::default(),
        )
        .expect("usage record");

        assert_eq!(record.input_tokens, Some(160));
        assert_eq!(record.cached_input_tokens, Some(40));
        assert_eq!(record.total_tokens, Some(170));
        assert_eq!(record.cost.expect("estimated cost").amount, "5370001");
    }
}
