//! OpenAI Responses API language-model implementation.
//!
//! This module is the boundary between Chudbot's provider-neutral transcript
//! model and OpenAI's Responses API. It serializes turns into `input` items,
//! advertises Chudbot client tools as OpenAI function tools, preserves OpenAI
//! continuation items for later turns, and folds Responses output back into
//! text, tool calls, server-tool usage, grounding metadata, and token usage.

use std::collections::{BTreeMap, BTreeSet};
use std::time::{Duration, Instant};

use chudbot_api::reasoning::TurnReasoning;
use chudbot_api::retry::{RetryPolicy, retry_after_error};
use chudbot_api::sse::{ServerSentEvent, SseDecoder};
use chudbot_api::{
    ClientToolCall, ClientToolResult, ClientToolResultContent, ClientToolSpec, ContentBlock,
    GroundingMetadata, LlmBackend, ModelId, ModelInfo, ModelInfoRequest, ModelStepDelta,
    ModelStepEvent, ModelStepKind, ModelStepRequest, ProviderContinuation, ProviderName,
    ServerToolSet, ServerToolUse, ToolInputSchema, ToolName, ToolUseId, Transcript, TurnRole,
    UsageRecord, UsageSubject, reasoning_items_to_delta_events,
};
use futures::{Stream, StreamExt};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::image::media_bytes_or_url;
use crate::pricing::OpenAiPricing;
use crate::{OpenAiClient, OpenAiError, json_strip_nulls};

/// Response fields needed to replay OpenAI reasoning continuations on later turns.
const REASONING_INCLUDE: &[&str] = &["reasoning.encrypted_content"];

impl OpenAiClient {
    async fn build_step_body(&self, request: &ModelStepRequest) -> Result<Value, OpenAiError> {
        let input = to_responses_input(&request.transcript, self).await?;
        Ok(build_step_body_from_input(request, input))
    }
}

impl LlmBackend for OpenAiClient {
    type Error = OpenAiError;

    fn backend_name(&self) -> &ProviderName {
        self.provider_name()
    }

    #[tracing::instrument(name = "openai.step", skip_all, fields(model = %request.model))]
    fn step(
        &self,
        request: ModelStepRequest,
    ) -> impl Stream<Item = Result<ModelStepEvent, Self::Error>> + Send + '_ {
        async_stream::try_stream! {
            let requested_model = request.model.clone();
            let body = self.build_step_body(&request).await?;
            let policy = RetryPolicy::default();
            let label = "llm[openai.stream]";
            let mut attempt = 1;

            'attempts: loop {
                let started = Instant::now();
                let resp = self
                    .post_json_stream("/responses", &body, label)
                    .await?;
                let chunks = resp.bytes_stream();
                futures::pin_mut!(chunks);
                let mut decoder = SseDecoder::new();
                let mut state = OpenAiStreamState::default();
                let mut emitted = false;

                while let Some(chunk) = chunks.next().await {
                    let chunk = match chunk {
                        Ok(chunk) => chunk,
                        Err(error) => {
                            let error = OpenAiError::Transport(error.to_string());
                            if !emitted && retry_after_error(policy, label, &mut attempt, &error).await {
                                continue 'attempts;
                            }
                            Err(error)?
                        }
                    };
                    let events = match decoder.push(&chunk) {
                        Ok(events) => events,
                        Err(error) => {
                            let error = OpenAiError::Decode(error.to_string());
                            if !emitted && retry_after_error(policy, label, &mut attempt, &error).await {
                                continue 'attempts;
                            }
                            Err(error)?
                        }
                    };
                    for event in events {
                        let outcome = match openai_stream_event(
                            event,
                            self.provider_name(),
                            &requested_model,
                            self.pricing(),
                            started,
                            &mut state,
                        ) {
                            Ok(outcome) => outcome,
                            Err(error) => {
                                if !emitted && retry_after_error(policy, label, &mut attempt, &error).await {
                                    continue 'attempts;
                                }
                                Err(error)?
                            }
                        };
                        if !outcome.events.is_empty() {
                            emitted = true;
                        }
                        for event in outcome.events {
                            yield event;
                        }
                        if outcome.finished {
                            return;
                        }
                    }
                }

                let final_event = match decoder.finish() {
                    Ok(event) => event,
                    Err(error) => {
                        let error = OpenAiError::Decode(error.to_string());
                        if !emitted && retry_after_error(policy, label, &mut attempt, &error).await {
                            continue 'attempts;
                        }
                        Err(error)?
                    }
                };
                if let Some(event) = final_event {
                    let outcome = match openai_stream_event(
                        event,
                        self.provider_name(),
                        &requested_model,
                        self.pricing(),
                        started,
                        &mut state,
                    ) {
                        Ok(outcome) => outcome,
                        Err(error) => {
                            if !emitted && retry_after_error(policy, label, &mut attempt, &error).await {
                                continue 'attempts;
                            }
                            Err(error)?
                        }
                    };
                    if !outcome.events.is_empty() {
                        emitted = true;
                    }
                    for event in outcome.events {
                        yield event;
                    }
                    if outcome.finished {
                        return;
                    }
                }

                let error = OpenAiError::Decode(
                    "OpenAI stream ended without a terminal response event".to_string(),
                );
                if !emitted && retry_after_error(policy, label, &mut attempt, &error).await {
                    continue 'attempts;
                }
                Err(error)?
            }
        }
    }

    #[tracing::instrument(name = "openai.model_info", skip_all, fields(model = %request.model))]
    async fn fetch_model_info(
        &self,
        request: ModelInfoRequest,
    ) -> Result<Option<ModelInfo>, Self::Error> {
        let endpoint = format!("/models/{}", request.model.as_str());
        let raw: Value = self.get_json(&endpoint, "model_info[openai]").await?;
        Ok(model_info_from_openai_model(request.model, raw))
    }
}

fn build_step_body_from_input(request: &ModelStepRequest, input: Vec<Value>) -> Value {
    let tools = build_responses_tools(&request.client_tools, &request.server_tools);
    let options = OpenAiOptions::from_request(request);
    let reasoning = build_reasoning_options(&options);
    let text = build_text_options(&options);
    let sampling = model_supports_sampling(request.model.as_str());
    let has_tools = !tools.is_empty();

    json_strip_nulls(json!({
        "model": request.model.as_str(),
        "input": input,
        "tools": has_tools.then_some(tools),
        "parallel_tool_calls": has_tools.then_some(true),
        "max_output_tokens": request.sampling.max_output_tokens,
        "temperature": sampling.then_some(request.sampling.temperature).flatten(),
        "top_p": sampling.then_some(request.sampling.top_p).flatten(),
        "reasoning": reasoning,
        "text": text,
        "prompt_cache_key": request.transcript.id,
        "include": REASONING_INCLUDE,
        "store": false,
        "stream": true,
    }))
}

#[derive(Debug, Default)]
struct OpenAiStreamState {
    emitted_text_delta: bool,
    emitted_reasoning_delta: bool,
    function_calls: BTreeMap<String, StreamingFunctionCall>,
    streamed_tool_ids: BTreeSet<ToolUseId>,
}

#[derive(Debug, Clone)]
struct StreamingFunctionCall {
    id: ToolUseId,
    name: Option<ToolName>,
}

#[derive(Debug, Default)]
struct StreamEventOutcome {
    events: Vec<ModelStepEvent>,
    finished: bool,
}

fn openai_stream_event(
    event: ServerSentEvent,
    provider: &ProviderName,
    requested_model: &ModelId,
    pricing: &OpenAiPricing,
    started: Instant,
    state: &mut OpenAiStreamState,
) -> Result<StreamEventOutcome, OpenAiError> {
    if event.data.trim() == "[DONE]" {
        return Ok(StreamEventOutcome::default());
    }

    let value = serde_json::from_str::<Value>(&event.data).map_err(|error| {
        OpenAiError::Decode(format!("failed to decode OpenAI SSE event: {error}"))
    })?;
    let event_type = value
        .get("type")
        .and_then(Value::as_str)
        .or(event.event.as_deref())
        .unwrap_or("");

    match event_type {
        "response.output_text.delta" => {
            let Some(delta) = value.get("delta").and_then(Value::as_str) else {
                return Ok(StreamEventOutcome::default());
            };
            state.emitted_text_delta = true;
            Ok(StreamEventOutcome {
                events: vec![ModelStepEvent::Delta(ModelStepDelta::Text {
                    item_id: stream_item_id(&value, "openai_text"),
                    delta: delta.to_string(),
                })],
                finished: false,
            })
        }
        "response.reasoning_summary_text.delta" | "response.reasoning_text.delta" => {
            let Some(delta) = value.get("delta").and_then(Value::as_str) else {
                return Ok(StreamEventOutcome::default());
            };
            state.emitted_reasoning_delta = true;
            Ok(StreamEventOutcome {
                events: vec![ModelStepEvent::Delta(ModelStepDelta::ReasoningSummary {
                    item_id: stream_item_id(&value, "openai_reasoning"),
                    provider: provider.clone(),
                    kind: Some(event_type.trim_start_matches("response.").to_string()),
                    delta: delta.to_string(),
                })],
                finished: false,
            })
        }
        "response.output_item.added" | "response.output_item.done" => {
            if let Some((key, call)) = streaming_function_call(&value) {
                state.function_calls.insert(key, call);
            }
            Ok(StreamEventOutcome::default())
        }
        "response.function_call_arguments.delta" => {
            let Some(delta) = value.get("delta").and_then(Value::as_str) else {
                return Ok(StreamEventOutcome::default());
            };
            let key = stream_item_id(&value, "openai_function_call");
            let Some(call) = state.function_calls.get(&key) else {
                return Ok(StreamEventOutcome::default());
            };
            state.streamed_tool_ids.insert(call.id.clone());
            Ok(StreamEventOutcome {
                events: vec![ModelStepEvent::Delta(ModelStepDelta::ClientToolCall {
                    item_id: key,
                    id: call.id.clone(),
                    name: call.name.clone(),
                    arguments_delta: delta.to_string(),
                })],
                finished: false,
            })
        }
        "response.completed" | "response.incomplete" => {
            let response = value.get("response").cloned().unwrap_or(value);
            let events = openai_terminal_events(
                response,
                provider,
                requested_model,
                pricing,
                started,
                state,
            )?;
            Ok(StreamEventOutcome {
                events,
                finished: true,
            })
        }
        "response.failed" => {
            let response = value.get("response").unwrap_or(&value);
            Err(provider_stream_error(response, "OpenAI stream failed"))
        }
        "error" => Err(provider_stream_error(
            &value,
            "OpenAI stream returned an error event",
        )),
        _ => Ok(StreamEventOutcome::default()),
    }
}

fn streaming_function_call(value: &Value) -> Option<(String, StreamingFunctionCall)> {
    let item = value.get("item").unwrap_or(value);
    if item.get("type").and_then(Value::as_str) != Some("function_call") {
        return None;
    }
    let key = stream_item_id(value, "openai_function_call");
    let id = item
        .get("call_id")
        .and_then(Value::as_str)
        .or_else(|| item.get("id").and_then(Value::as_str))
        .filter(|id| !id.is_empty())?;
    let name = item
        .get("name")
        .and_then(Value::as_str)
        .filter(|name| !name.is_empty())
        .map(ToolName::new);
    Some((
        key,
        StreamingFunctionCall {
            id: ToolUseId::new(id),
            name,
        },
    ))
}

fn stream_item_id(value: &Value, fallback: &str) -> String {
    value
        .get("item_id")
        .and_then(Value::as_str)
        .or_else(|| {
            value
                .get("item")
                .and_then(|item| item.get("id"))
                .and_then(Value::as_str)
        })
        .filter(|id| !id.is_empty())
        .map(str::to_string)
        .or_else(|| {
            value
                .get("output_index")
                .and_then(Value::as_u64)
                .map(|index| format!("{fallback}:{index}"))
        })
        .unwrap_or_else(|| fallback.to_string())
}

fn openai_terminal_events(
    response: Value,
    provider: &ProviderName,
    requested_model: &ModelId,
    pricing: &OpenAiPricing,
    started: Instant,
    state: &OpenAiStreamState,
) -> Result<Vec<ModelStepEvent>, OpenAiError> {
    let parsed: ResponsesResponse = serde_json::from_value(response.clone()).map_err(|error| {
        OpenAiError::Decode(format!(
            "failed to decode terminal OpenAI response: {error}"
        ))
    })?;
    let model_id = parsed
        .model
        .as_deref()
        .map(ModelId::new)
        .unwrap_or_else(|| requested_model.clone());
    let usage = usage_from_openai(
        provider,
        Some(model_id.clone()),
        UsageSubject::ModelStep,
        parsed.usage.as_ref(),
        pricing,
    );
    log_usage(model_id.as_str(), usage.as_ref(), started.elapsed());

    let (text, client_tool_calls, server_tool_uses, grounding) =
        walk_output(&parsed.output, provider);
    let kind = if !client_tool_calls.is_empty() {
        ModelStepKind::ClientTools
    } else if text.is_empty() {
        ModelStepKind::Continue
    } else {
        ModelStepKind::Final
    };

    let continuation = (!parsed.output.is_empty()).then_some(ProviderContinuation {
        provider: provider.clone(),
        data: Value::Array(parsed.output),
    });

    let mut events = Vec::new();
    if !state.emitted_text_delta && !text.is_empty() {
        events.push(ModelStepEvent::Delta(ModelStepDelta::Text {
            item_id: "openai_text:terminal".to_string(),
            delta: text,
        }));
    }
    events.extend(
        client_tool_calls
            .into_iter()
            .filter(|call| !state.streamed_tool_ids.contains(&call.id))
            .enumerate()
            .map(|(index, call)| {
                ModelStepEvent::Delta(ModelStepDelta::ClientToolCall {
                    item_id: format!("openai_tool:{index}"),
                    id: call.id,
                    name: Some(call.name),
                    arguments_delta: call.input.to_string(),
                })
            }),
    );
    if !state.emitted_reasoning_delta
        && let Some(continuation) = continuation.as_ref()
    {
        events.extend(reasoning_items_to_delta_events(
            TurnReasoning::from_continuation_and_usage(Some(continuation), Some(&model_id), &[])
                .items,
            "openai_reasoning",
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
    Ok(events)
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

fn provider_stream_error(value: &Value, fallback: &str) -> OpenAiError {
    let message = provider_error_message(value, fallback);
    match provider_error_status(value, &message) {
        Some(status) => OpenAiError::Api {
            status,
            body: message,
        },
        None => OpenAiError::Decode(message),
    }
}

fn provider_error_status(value: &Value, message: &str) -> Option<u16> {
    let error = value.get("error");
    for field in ["status", "status_code", "http_status"] {
        if let Some(status) = value
            .get(field)
            .or_else(|| error.and_then(|error| error.get(field)))
            .and_then(Value::as_u64)
            .and_then(|status| u16::try_from(status).ok())
        {
            return Some(status);
        }
    }

    let code = error
        .and_then(|error| error.get("code").or_else(|| error.get("type")))
        .or_else(|| value.get("code").or_else(|| value.get("type")))
        .and_then(Value::as_str)
        .unwrap_or("");
    let marker = format!("{code} {message}").to_ascii_lowercase();
    if marker.contains("rate_limit") || marker.contains("rate limit") {
        Some(429)
    } else if marker.contains("temporarily unavailable")
        || marker.contains("temporary unavailable")
        || marker.contains("service_unavailable")
        || marker.contains("service unavailable")
        || marker.contains("overloaded")
    {
        Some(503)
    } else {
        None
    }
}

/// Extract known model limits from OpenAI's model metadata response.
///
/// OpenAI-compatible gateways do not agree on field names, so this accepts the
/// common spellings used by OpenAI, vLLM-style servers, and model catalogs.
fn model_info_from_openai_model(requested_model: ModelId, raw: Value) -> Option<ModelInfo> {
    const CONTEXT_FIELDS: &[&str] = &[
        "context_window_tokens",
        "context_window",
        "context_length",
        "max_context_length",
        "max_context_len",
        "max_input_tokens",
        "input_token_limit",
        "max_model_len",
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

/// Find the first numeric metadata field, descending into nested objects.
fn find_u64_field(value: &Value, fields: &[&str]) -> Option<u64> {
    let object = value.as_object()?;
    if let Some(found) = fields
        .iter()
        .find_map(|field| object.get(*field).and_then(value_as_u64))
    {
        return Some(found);
    }

    object
        .values()
        .filter(|value| value.is_object())
        .find_map(|value| find_u64_field(value, fields))
}

/// Parse numeric fields returned either as JSON numbers or decimal strings.
fn value_as_u64(value: &Value) -> Option<u64> {
    match value {
        Value::Number(number) => number.as_u64(),
        Value::String(text) => text.parse().ok(),
        _ => None,
    }
}

/// Convert a Chudbot transcript into OpenAI Responses `input` items.
///
/// The important details are preserving provider continuations, flushing mixed
/// text/media messages before tool events, and sending application instructions
/// as `developer` messages rather than user-visible content.
async fn to_responses_input(
    transcript: &Transcript,
    client: &OpenAiClient,
) -> Result<Vec<Value>, OpenAiError> {
    let mut input = Vec::new();
    if let Some(instructions) = &transcript.instructions
        && !instructions.is_empty()
    {
        input.push(json!({ "role": "developer", "content": instructions }));
    }

    for message in &transcript.turns {
        let role = match message.role {
            TurnRole::Assistant => "assistant",
            TurnRole::User => "user",
        };

        let mut echo = Vec::new();
        for block in &message.blocks {
            if let ContentBlock::Continuation(continuation) = block
                && &continuation.provider == client.provider_name()
            {
                // Only this provider can safely replay its opaque continuation payload.
                match &continuation.data {
                    Value::Array(items) => echo.extend(items.iter().cloned()),
                    other => echo.push(other.clone()),
                }
            }
        }
        let full_echo = echo
            .iter()
            .any(|item| item.get("type").and_then(Value::as_str) != Some("reasoning"));
        if full_echo {
            // A full Responses output item already carries the assistant message or
            // tool call, so replay it verbatim instead of rebuilding that turn.
            input.extend(echo);
            continue;
        }

        input.extend(echo);
        let mut text = String::new();
        let mut media_urls = Vec::new();
        for block in &message.blocks {
            match block {
                ContentBlock::Text { text: t } => text.push_str(t),
                ContentBlock::Media { media } => {
                    media_urls.push(media_bytes_or_url(media.as_ref()).await?)
                }
                ContentBlock::Continuation(_) => {}
                ContentBlock::ClientToolCall(call) => {
                    // Tool calls are standalone Responses items; flush pending
                    // message content first so transcript ordering survives replay.
                    push_responses_message(&mut input, role, &mut text, &mut media_urls);
                    let args = serde_json::to_string(&call.input).unwrap_or_else(|_| "{}".into());
                    input.push(json!({
                        "type": "function_call",
                        "call_id": call.id.as_str(),
                        "name": call.name.as_str(),
                        "arguments": args,
                    }));
                }
                ContentBlock::ClientToolResult(result) => {
                    // Responses expects function outputs at top level, not inside a
                    // message content array.
                    push_responses_message(&mut input, role, &mut text, &mut media_urls);
                    input.push(json!({
                        "type": "function_call_output",
                        "call_id": result.tool_use_id.as_str(),
                        "output": client_tool_result_as_string(result),
                    }));
                }
            }
        }

        push_responses_message(&mut input, role, &mut text, &mut media_urls);
    }
    Ok(input)
}

/// Flush accumulated text and media into one Responses message item.
fn push_responses_message(
    input: &mut Vec<Value>,
    role: &str,
    text: &mut String,
    media_urls: &mut Vec<String>,
) {
    if media_urls.is_empty() {
        if !text.is_empty() {
            input.push(json!({ "role": role, "content": text.as_str() }));
            text.clear();
        }
        return;
    }

    let mut content = Vec::with_capacity(media_urls.len() + 1);
    if !text.is_empty() {
        content.push(json!({ "type": "input_text", "text": text.as_str() }));
    }
    for url in media_urls.iter() {
        content.push(json!({ "type": "input_image", "image_url": url }));
    }
    input.push(json!({ "role": role, "content": content }));
    text.clear();
    media_urls.clear();
}

/// Flatten Chudbot tool-result content into OpenAI's string output field.
fn client_tool_result_as_string(result: &ClientToolResult) -> String {
    match &result.content {
        ClientToolResultContent::Json { value } => {
            serde_json::to_string(value).unwrap_or_else(|_| value.to_string())
        }
        ClientToolResultContent::Text { text } => text.clone(),
    }
}

/// Return whether the model accepts `temperature` and `top_p`.
///
/// Reasoning-family models reject sampling knobs, so request shaping omits them
/// for those model ids instead of letting the API fail the turn.
fn model_supports_sampling(model: &str) -> bool {
    let model = model.to_ascii_lowercase();
    let reasoning = model.starts_with("o1")
        || model.starts_with("o3")
        || model.starts_with("o4")
        || (model.starts_with("gpt-5") && !model.contains("chat"));
    !reasoning
}

/// Advertise Chudbot client tools and supported OpenAI-hosted tools.
fn build_responses_tools(
    client_tools: &BTreeMap<ToolName, ClientToolSpec>,
    server_tools: &ServerToolSet,
) -> Vec<Value> {
    let mut tools = Vec::with_capacity(client_tools.len() + 1);
    for (name, tool) in client_tools {
        tools.push(json!({
            "type": "function",
            "name": name.as_str(),
            "description": tool.description,
            "parameters": openai_tool_parameters(&tool.input_schema),
        }));
    }
    if server_tools.contains("web_search") {
        tools.push(json!({ "type": "web_search" }));
    }
    tools
}

fn openai_tool_parameters(input_schema: &ToolInputSchema) -> Value {
    serde_json::to_value(input_schema).expect("tool input schema serializes")
}

/// Build the OpenAI `reasoning` option object, omitting it when unset.
fn build_reasoning_options(options: &OpenAiOptions) -> Option<Value> {
    let value = json_strip_nulls(json!({
        "effort": options.reasoning_effort.as_deref(),
        "summary": options.reasoning_summary.as_deref(),
    }));
    match &value {
        Value::Object(map) if map.is_empty() => None,
        _ => Some(value),
    }
}

/// Build the OpenAI `text` option object, omitting it when unset.
fn build_text_options(options: &OpenAiOptions) -> Option<Value> {
    let value = json_strip_nulls(json!({
        "verbosity": options.text_verbosity.as_deref(),
    }));
    match &value {
        Value::Object(map) if map.is_empty() => None,
        _ => Some(value),
    }
}

/// Decode Responses output items into Chudbot's provider-neutral assistant step.
///
/// OpenAI may interleave messages, function calls, and hosted-tool calls.
/// Chudbot keeps raw hosted-tool items for trace display and stores citation
/// annotations as grounding metadata.
fn walk_output(
    output: &[Value],
    provider: &ProviderName,
) -> (
    String,
    Vec<ClientToolCall>,
    Vec<ServerToolUse>,
    Vec<GroundingMetadata>,
) {
    let mut text = String::new();
    let mut client_calls = Vec::new();
    let mut server_uses = Vec::new();
    let mut citations = Vec::new();

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
                        if let Some(annotations) =
                            block.get("annotations").and_then(Value::as_array)
                        {
                            // Preserve URL citations without normalizing OpenAI's
                            // annotation shape into a provider-independent schema.
                            for annotation in annotations {
                                if annotation.get("type").and_then(Value::as_str)
                                    == Some("url_citation")
                                {
                                    citations.push(annotation.clone());
                                }
                            }
                        }
                    }
                } else if let Some(t) = item.get("content").and_then(Value::as_str) {
                    text.push_str(t);
                }
            }
            "function_call" => {
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
                // Hosted tools such as `web_search_call` are not Chudbot client tools;
                // record them as provider server-tool uses for trace visibility.
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

    let grounding = if citations.is_empty() {
        Vec::new()
    } else {
        vec![GroundingMetadata {
            provider: provider.clone(),
            raw: json!({ "annotations": citations }),
        }]
    };

    (text, client_calls, server_uses, grounding)
}

/// Log token usage from a completed Responses request.
fn log_usage(model: &str, usage: Option<&UsageRecord>, elapsed: Duration) {
    let duration_ms = elapsed.as_millis() as u64;
    match usage {
        Some(u) => tracing::info!(
            target: "openai_usage",
            model = %model,
            input_tokens = u.input_tokens.unwrap_or(0),
            cached_tokens = u.cached_input_tokens.unwrap_or(0),
            output_tokens = u.output_tokens.unwrap_or(0),
            reasoning_tokens = u.reasoning_tokens.unwrap_or(0),
            total_tokens = u.total_tokens.unwrap_or(0),
            duration_ms,
            "openai responses request complete",
        ),
        None => tracing::info!(
            target: "openai_usage",
            model = %model,
            duration_ms,
            "openai responses request complete; no usage reported",
        ),
    }
}

/// Convert OpenAI usage JSON into Chudbot's usage record and cost estimate.
fn usage_from_openai(
    provider: &ProviderName,
    model: Option<ModelId>,
    subject: UsageSubject,
    usage: Option<&Value>,
    pricing: &OpenAiPricing,
) -> Option<UsageRecord> {
    let raw = usage?.clone();
    let parsed = serde_json::from_value::<Usage>(raw.clone()).ok()?;
    let cost = pricing.estimate_token_cost(
        model.as_ref(),
        parsed.input_tokens,
        parsed.input_tokens_details.cached_tokens,
        parsed.output_tokens,
    );
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

/// OpenAI-specific per-agent options routed through `ProviderOptions`.
///
/// These values map directly to Responses request fields and are intentionally
/// separate from Chudbot's provider-neutral sampling options.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct OpenAiOptions {
    /// Reasoning effort: `none`, `minimal`, `low`, `medium`, `high`, or `xhigh`.
    #[serde(default)]
    pub reasoning_effort: Option<String>,
    /// Reasoning summary detail: `auto`, `concise`, or `detailed`.
    #[serde(default)]
    pub reasoning_summary: Option<String>,
    /// Text verbosity: `low`, `medium`, or `high`.
    #[serde(default)]
    pub text_verbosity: Option<String>,
}

impl OpenAiOptions {
    /// Decode OpenAI provider options from a model-step request.
    fn from_request(request: &ModelStepRequest) -> Self {
        request
            .provider_options
            .as_ref()
            .and_then(|opts| serde_json::from_value(opts.value.clone()).ok())
            .unwrap_or_default()
    }
}

/// Minimal shape Chudbot needs from a Responses API response.
#[derive(Deserialize)]
struct ResponsesResponse {
    #[serde(default)]
    output: Vec<Value>,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    usage: Option<Value>,
}

/// Token usage shape returned by OpenAI Responses.
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

/// Nested token counters used for cache and reasoning details.
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
    use chudbot_api::{
        ProviderOptions, ToolInputField, ToolInputSchema, ToolInputValueSchema, TranscriptTurn,
        collect_model_step,
    };

    #[test]
    fn reasoning_models_reject_sampling_knobs() {
        assert!(!model_supports_sampling("gpt-5"));
        assert!(!model_supports_sampling("gpt-5-mini"));
        assert!(!model_supports_sampling("o3"));
        assert!(!model_supports_sampling("o4-mini"));
        assert!(model_supports_sampling("gpt-4o"));
        assert!(model_supports_sampling("gpt-4.1"));
        assert!(model_supports_sampling("gpt-5-chat-latest"));
    }

    #[test]
    fn builds_openai_web_search_tool_only() {
        let mut server_tools = ServerToolSet::new();
        server_tools.insert("web_search".to_string());
        server_tools.insert("x_search".to_string());
        let tools = build_responses_tools(&BTreeMap::new(), &server_tools);
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["type"], "web_search");
    }

    #[test]
    fn builds_client_tool_schema() {
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
        let tools = build_responses_tools(&client_tools, &ServerToolSet::new());
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["type"], "function");
        assert_eq!(tools[0]["name"], "fetch_messages");
        assert_eq!(
            tools[0]["parameters"],
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
    fn parses_message_function_call_server_call_and_citations() {
        let provider = ProviderName::new("openai");
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
            json!({
                "type": "function_call",
                "call_id": "call_42",
                "name": "fetch_messages",
                "arguments": "{\"limit\":30}",
            }),
        ];
        let (text, calls, server, grounding) = walk_output(&output, &provider);
        assert_eq!(text, "Found it.");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id.as_str(), "call_42");
        assert_eq!(calls[0].input["limit"], 30);
        assert_eq!(server.len(), 1);
        assert_eq!(server[0].name.as_str(), "web_search");
        assert_eq!(grounding.len(), 1);
        assert_eq!(
            grounding[0].raw["annotations"][0]["url"],
            "https://example.com"
        );
    }

    #[test]
    fn streams_responses_text_reasoning_and_tool_arguments() {
        let provider = ProviderName::new("openai");
        let requested_model = ModelId::new("gpt-5");
        let mut state = OpenAiStreamState::default();
        let mut events = Vec::new();
        for data in [
            json!({
                "type": "response.output_text.delta",
                "item_id": "msg_1",
                "delta": "Hi",
            }),
            json!({
                "type": "response.reasoning_summary_text.delta",
                "item_id": "rs_1",
                "delta": "Plan",
            }),
            json!({
                "type": "response.output_item.added",
                "item": {
                    "type": "function_call",
                    "id": "fc_1",
                    "call_id": "call_1",
                    "name": "fetch_messages",
                },
            }),
            json!({
                "type": "response.function_call_arguments.delta",
                "item_id": "fc_1",
                "delta": "{\"limit\":30}",
            }),
            json!({
                "type": "response.completed",
                "response": {
                    "model": "gpt-5",
                    "output": [
                        {
                            "type": "message",
                            "id": "msg_1",
                            "role": "assistant",
                            "content": [{ "type": "output_text", "text": "Hi" }]
                        },
                        {
                            "type": "function_call",
                            "id": "fc_1",
                            "call_id": "call_1",
                            "name": "fetch_messages",
                            "arguments": "{\"limit\":30}"
                        }
                    ],
                    "usage": {
                        "input_tokens": 10,
                        "output_tokens": 2,
                        "total_tokens": 12
                    }
                }
            }),
        ] {
            let outcome = openai_stream_event(
                ServerSentEvent {
                    event: None,
                    data: data.to_string(),
                },
                &provider,
                &requested_model,
                &OpenAiPricing::default(),
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
            events.into_iter().map(Ok::<_, OpenAiError>),
        )))
        .expect("finished step");
        assert!(matches!(step.kind(), ModelStepKind::ClientTools));
        let output = step.output();
        assert_eq!(output.answer_text(), "Hi");
        assert_eq!(output.reasoning().count(), 1);
        let calls = output.client_tool_calls().collect::<Vec<_>>();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id.as_str(), "call_1");
        assert_eq!(calls[0].input["limit"], 30);
        assert_eq!(output.usage[0].input_tokens, Some(10));
    }

    #[test]
    fn stream_service_unavailable_error_is_retryable() {
        let provider = ProviderName::new("openai");
        let requested_model = ModelId::new("gpt-5");
        let error = openai_stream_event(
            ServerSentEvent {
                event: Some("error".to_string()),
                data: json!({
                    "type": "error",
                    "error": {
                        "code": "service_unavailable",
                        "message": "Service temporarily unavailable."
                    }
                })
                .to_string(),
            },
            &provider,
            &requested_model,
            &OpenAiPricing::default(),
            Instant::now(),
            &mut OpenAiStreamState::default(),
        )
        .expect_err("stream error");

        assert!(matches!(error, OpenAiError::Api { status: 503, .. }));
        assert_eq!(
            chudbot_api::retry::ClassifyError::error_class(&error),
            chudbot_api::retry::ErrorClass::ServerTransient
        );
    }

    #[test]
    fn parses_openai_options_from_routed_provider_value() {
        let request = ModelStepRequest {
            model: ModelId::new("gpt-5"),
            transcript: Transcript::from_user_text("hi"),
            client_tools: BTreeMap::new(),
            server_tools: ServerToolSet::new(),
            sampling: chudbot_api::SamplingOptions::default(),
            provider_options: Some(ProviderOptions {
                value: json!({
                    "reasoning_effort": "high",
                    "reasoning_summary": "auto",
                    "text_verbosity": "low",
                }),
            }),
        };
        let options = OpenAiOptions::from_request(&request);
        assert_eq!(options.reasoning_effort.as_deref(), Some("high"));
        assert_eq!(options.reasoning_summary.as_deref(), Some("auto"));
        assert_eq!(options.text_verbosity.as_deref(), Some("low"));
    }

    #[test]
    fn builds_reasoning_options_with_summary() {
        let options = OpenAiOptions {
            reasoning_effort: Some("medium".to_string()),
            reasoning_summary: Some("auto".to_string()),
            text_verbosity: None,
        };
        let reasoning = build_reasoning_options(&options).unwrap();
        assert_eq!(reasoning, json!({ "effort": "medium", "summary": "auto" }));
    }

    #[test]
    fn builds_text_options_with_verbosity() {
        let options = OpenAiOptions {
            text_verbosity: Some("low".to_string()),
            ..OpenAiOptions::default()
        };
        let text = build_text_options(&options).unwrap();
        assert_eq!(text, json!({ "verbosity": "low" }));
    }

    #[test]
    fn omits_text_options_when_empty() {
        let options = OpenAiOptions::default();
        assert!(build_text_options(&options).is_none());
    }

    #[test]
    fn omits_reasoning_options_when_empty() {
        let options = OpenAiOptions::default();
        assert!(build_reasoning_options(&options).is_none());
    }

    #[test]
    fn replays_full_output_verbatim_when_present() {
        let client = OpenAiClient::new(ProviderName::new("openai"), "key");
        let mut transcript = Transcript::new();
        transcript.push(TranscriptTurn::text(TurnRole::User, "hi"));
        transcript.push(TranscriptTurn {
            role: TurnRole::Assistant,
            blocks: vec![
                ContentBlock::Continuation(ProviderContinuation {
                    provider: ProviderName::new("openai"),
                    data: json!([
                        { "type": "reasoning", "id": "rs_1", "encrypted_content": "BLOB" },
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
        let input = futures::executor::block_on(to_responses_input(&transcript, &client)).unwrap();
        assert_eq!(input.len(), 3);
        assert_eq!(input[1]["type"], "reasoning");
        assert_eq!(input[1]["encrypted_content"], "BLOB");
        assert_eq!(input[2]["type"], "message");
        assert_eq!(input[2]["id"], "msg_1");
    }

    #[test]
    fn sends_transcript_instructions_as_developer_message() {
        let client = OpenAiClient::new(ProviderName::new("openai"), "key");
        let mut transcript = Transcript::new();
        transcript.instructions = Some("Follow the application rules.".to_string());
        transcript.push(TranscriptTurn::text(TurnRole::User, "hi"));

        let input = futures::executor::block_on(to_responses_input(&transcript, &client)).unwrap();
        assert_eq!(input[0]["role"], "developer");
        assert_eq!(input[0]["content"], "Follow the application rules.");
        assert_eq!(input[1]["role"], "user");
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
        let provider = ProviderName::new("openai");
        let record = usage_from_openai(
            &provider,
            Some(ModelId::new("gpt-5")),
            UsageSubject::ModelStep,
            Some(&usage),
            &OpenAiPricing::default(),
        )
        .unwrap();
        assert_eq!(record.input_tokens, Some(153));
        assert_eq!(record.cached_input_tokens, Some(128));
        assert_eq!(record.reasoning_tokens, Some(303));
        assert_eq!(record.total_tokens, Some(755));
        assert!(record.cost.is_none());
    }

    #[test]
    fn openai_model_info_uses_provider_limits_when_present() {
        let info = model_info_from_openai_model(
            ModelId::new("gpt-test"),
            json!({
                "id": "gpt-test",
                "max_input_tokens": 1048576,
                "max_output_tokens": "32768"
            }),
        )
        .expect("model metadata");

        assert_eq!(info.id, ModelId::new("gpt-test"));
        assert_eq!(info.context_window_tokens, Some(1_048_576));
        assert_eq!(info.max_output_tokens, Some(32_768));
        assert!(info.raw.is_some());
    }

    #[test]
    fn estimates_usage_cost_for_known_openai_model() {
        let usage = json!({
            "input_tokens": 100,
            "input_tokens_details": { "cached_tokens": 40 },
            "output_tokens": 20,
            "total_tokens": 120,
        });
        let provider = ProviderName::new("openai");
        let record = usage_from_openai(
            &provider,
            Some(ModelId::new("gpt-5.5")),
            UsageSubject::ModelStep,
            Some(&usage),
            &OpenAiPricing::default(),
        )
        .unwrap();

        let cost = record.cost.expect("estimated cost");
        assert_eq!(cost.unit, "usd_ticks");
        assert!(cost.estimated);
        assert_eq!(cost.amount, "9200000");
    }
}
