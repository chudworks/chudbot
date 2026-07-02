//! xAI Responses API language-model implementation.
//!
//! This module is the boundary between Chudbot's provider-neutral
//! [`ModelStepRequest`] contract and xAI's `/responses` JSON shape. It is
//! responsible for translating transcripts into replayable Responses input,
//! preserving xAI continuation items for future turns, decoding model output
//! into text/tool/server-use blocks, and normalizing usage into Chudbot records.

use std::collections::{BTreeMap, BTreeSet};
use std::io;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use chudbot_api::reasoning::TurnReasoning;
use chudbot_api::retry::{RetryPolicy, retry_after_error};
use chudbot_api::sse::{ServerSentEvent, SseDecoder};
use chudbot_api::{
    ClientToolCall, ClientToolSpec, ContentBlock, CostAmount, GroundingMetadata, LlmBackend,
    ModelId, ModelInfo, ModelInfoRequest, ModelStepDelta, ModelStepEvent, ModelStepKind,
    ModelStepRequest, ProviderContinuation, ProviderName, ServerToolSet, ServerToolUse,
    ToolInputSchema, ToolName, ToolUseId, Transcript, TurnRole, UsageRecord, UsageSubject,
    reasoning_items_to_delta_events,
};
use futures::{Stream, StreamExt};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::fs::{self, File, OpenOptions};
use tokio::io::AsyncWriteExt;

use crate::imagine::media_provider_url;
use crate::{XaiClient, XaiError, json_strip_nulls};

/// Request encrypted reasoning blobs so later turns can replay provider state.
const REASONING_INCLUDE: &[&str] = &["reasoning.encrypted_content"];

impl XaiClient {
    async fn build_step_body(&self, request: &ModelStepRequest) -> Result<Value, XaiError> {
        let input = to_responses_input(&request.transcript, self.provider_name()).await?;
        Ok(build_step_body_from_input(request, input))
    }
}

impl LlmBackend for XaiClient {
    type Error = XaiError;

    fn backend_name(&self) -> &ProviderName {
        self.provider_name()
    }

    #[tracing::instrument(name = "xai.step", skip_all, fields(model = %request.model))]
    fn step(
        &self,
        request: ModelStepRequest,
    ) -> impl Stream<Item = Result<ModelStepEvent, Self::Error>> + Send + '_ {
        async_stream::try_stream! {
            let requested_model = request.model.clone();
            let body = self.build_step_body(&request).await?;
            let policy = RetryPolicy::default();
            let label = "llm[xai.stream]";
            let mut attempt = 1;

            'attempts: loop {
                let started = Instant::now();
                let mut dump = self.start_step_dump(&body).await;
                let resp = self
                    .post_json_stream("/responses", &body, label)
                    .await?;
                let chunks = resp.bytes_stream();
                futures::pin_mut!(chunks);
                let mut decoder = SseDecoder::new();
                let mut state = XaiStreamState::default();
                let mut emitted = false;

                while let Some(chunk) = chunks.next().await {
                    let chunk = match chunk {
                        Ok(chunk) => chunk,
                        Err(error) => {
                            let error = XaiError::Transport(error.to_string());
                            if !emitted && retry_after_error(policy, label, &mut attempt, &error).await {
                                continue 'attempts;
                            }
                            Err(error)?
                        }
                    };
                    let events = match decoder.push(&chunk) {
                        Ok(events) => events,
                        Err(error) => {
                            let error = XaiError::Decode(error.to_string());
                            if !emitted && retry_after_error(policy, label, &mut attempt, &error).await {
                                continue 'attempts;
                            }
                            Err(error)?
                        }
                    };
                    for event in events {
                        dump_xai_stream_event(&mut dump, &event).await;
                        let outcome = match xai_stream_event(
                            event,
                            self.provider_name(),
                            &requested_model,
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
                        let error = XaiError::Decode(error.to_string());
                        if !emitted && retry_after_error(policy, label, &mut attempt, &error).await {
                            continue 'attempts;
                        }
                        Err(error)?
                    }
                };
                if let Some(event) = final_event {
                    dump_xai_stream_event(&mut dump, &event).await;
                    let outcome = match xai_stream_event(
                        event,
                        self.provider_name(),
                        &requested_model,
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

                let error = XaiError::Decode(
                    "xAI stream ended without a terminal response event".to_string(),
                );
                if !emitted && retry_after_error(policy, label, &mut attempt, &error).await {
                    continue 'attempts;
                }
                Err(error)?
            }
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

fn build_step_body_from_input(request: &ModelStepRequest, input: Vec<Value>) -> Value {
    let tools = build_responses_tools(&request.client_tools, &request.server_tools);
    let options = XaiOptions::from_request(request);
    let reasoning = options
        .reasoning_effort
        .as_ref()
        .map(|effort| json!({ "effort": effort }));
    let has_tools = !tools.is_empty();

    json_strip_nulls(json!({
        "model": request.model.as_str(),
        "input": input,
        "tools": has_tools.then_some(tools),
        "parallel_tool_calls": has_tools.then_some(true),
        "max_output_tokens": request.sampling.max_output_tokens,
        "temperature": request.sampling.temperature.as_ref(),
        "top_p": request.sampling.top_p.as_ref(),
        "reasoning": reasoning,
        "prompt_cache_key": request.transcript.id,
        "include": REASONING_INCLUDE,
        "store": false,
        "stream": true,
    }))
}

impl XaiClient {
    async fn start_step_dump(&self, body: &Value) -> Option<XaiStepDump> {
        let root = self.dump_dir()?;
        match XaiStepDump::create(root, body).await {
            Ok(dump) => dump,
            Err(error) => {
                tracing::warn!(
                    provider = %self.provider_name(),
                    dump_dir = %root.display(),
                    error = %error,
                    "failed to create xAI request dump"
                );
                None
            }
        }
    }
}

#[derive(Debug)]
struct XaiStepDump {
    dir: PathBuf,
    prefix: String,
}

impl XaiStepDump {
    async fn create(root: &Path, body: &Value) -> io::Result<Option<Self>> {
        let Some(conversation) = dump_conversation_id(body) else {
            return Ok(None);
        };
        let dir = root.join(sanitize_dump_path_segment(conversation));
        fs::create_dir_all(&dir).await?;
        let start = next_dump_ordinal(&dir).await?;
        for ordinal in start.. {
            let dump = Self {
                dir: dir.clone(),
                prefix: format!("{ordinal:04}"),
            };
            match dump.write_json_new("request", body).await {
                Ok(()) => {
                    tracing::debug!(
                        dump_dir = %dump.dir.display(),
                        dump_prefix = %dump.prefix,
                        "created xAI request dump"
                    );
                    return Ok(Some(dump));
                }
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
                Err(error) => return Err(error),
            }
        }
        unreachable!("unbounded dump ordinal iterator should return or fail");
    }

    async fn write_stream_event(&mut self, event: &ServerSentEvent) -> io::Result<()> {
        if event.data.trim() == "[DONE]" {
            return Ok(());
        }

        let value: Value = serde_json::from_str(&event.data).map_err(io::Error::other)?;
        self.write_terminal_payloads(&value).await
    }

    async fn write_terminal_payloads(&self, event: &Value) -> io::Result<()> {
        let event_type = event
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or_default();
        match event_type {
            "response.completed" | "response.incomplete" => {
                let response = event.get("response").unwrap_or(event);
                self.write_json("response", response).await?;
                if let Some(output) = response.get("output").and_then(Value::as_array) {
                    let continuation = replayable_continuation_items(output.iter().cloned());
                    if !continuation.is_empty() {
                        self.write_json("continuation", &Value::Array(continuation))
                            .await?;
                    }
                }
            }
            "response.failed" | "error" => {
                self.write_json("error", event).await?;
            }
            _ => {}
        }
        Ok(())
    }

    async fn write_json(&self, suffix: &str, value: &Value) -> io::Result<()> {
        let path = self.path_for(suffix);
        write_pretty_json_file(File::create(path).await?, value).await
    }

    async fn write_json_new(&self, suffix: &str, value: &Value) -> io::Result<()> {
        let path = self.path_for(suffix);
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .await?;
        write_pretty_json_file(file, value).await
    }

    fn path_for(&self, suffix: &str) -> PathBuf {
        self.dir.join(format!("{}_{}.json", self.prefix, suffix))
    }
}

async fn dump_xai_stream_event(dump: &mut Option<XaiStepDump>, event: &ServerSentEvent) {
    let Some(writer) = dump.as_mut() else {
        return;
    };
    if let Err(error) = writer.write_stream_event(event).await {
        tracing::warn!(
            dump_dir = %writer.dir.display(),
            dump_prefix = %writer.prefix,
            error = %error,
            "failed to write xAI stream dump"
        );
        *dump = None;
    }
}

async fn write_pretty_json_file(mut file: File, value: &Value) -> io::Result<()> {
    let mut bytes = serde_json::to_vec_pretty(value).map_err(io::Error::other)?;
    bytes.push(b'\n');
    file.write_all(&bytes).await
}

async fn next_dump_ordinal(dir: &Path) -> io::Result<u32> {
    let mut max = 0;
    let mut entries = fs::read_dir(dir).await?;
    while let Some(entry) = entries.next_entry().await? {
        let Some(name) = entry.file_name().to_str().map(str::to_string) else {
            continue;
        };
        let Some((prefix, suffix)) = name.split_once('_') else {
            continue;
        };
        if suffix == "request.json"
            && prefix.len() == 4
            && prefix.chars().all(|ch| ch.is_ascii_digit())
            && let Ok(ordinal) = prefix.parse::<u32>()
        {
            max = max.max(ordinal);
        }
    }
    Ok(max.saturating_add(1).max(1))
}

fn dump_conversation_id(body: &Value) -> Option<&str> {
    body.get("prompt_cache_key")
        .and_then(Value::as_str)
        .filter(|id| !id.is_empty())
}

fn sanitize_dump_path_segment(value: &str) -> String {
    let mut segment = String::with_capacity(value.len());
    for ch in value.chars() {
        if ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '.') {
            segment.push(ch);
        } else {
            segment.push('_');
        }
    }
    if segment.is_empty() {
        "unnamed".to_string()
    } else {
        segment
    }
}

#[derive(Debug, Default)]
struct XaiStreamState {
    emitted_text_delta: bool,
    emitted_reasoning_delta: bool,
    function_calls: BTreeMap<String, StreamingFunctionCall>,
    streamed_tool_ids: BTreeSet<ToolUseId>,
    chat_seen: bool,
    chat_model: Option<ModelId>,
    chat_usage: Option<Value>,
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

fn xai_stream_event(
    event: ServerSentEvent,
    provider: &ProviderName,
    requested_model: &ModelId,
    started: Instant,
    state: &mut XaiStreamState,
) -> Result<StreamEventOutcome, XaiError> {
    if event.data.trim() == "[DONE]" {
        if state.chat_seen {
            return Ok(StreamEventOutcome {
                events: xai_chat_terminal_events(provider, requested_model, started, state),
                finished: true,
            });
        }
        return Ok(StreamEventOutcome::default());
    }

    let value = serde_json::from_str::<Value>(&event.data)
        .map_err(|error| XaiError::Decode(format!("failed to decode xAI SSE event: {error}")))?;
    if value.get("object").and_then(Value::as_str) == Some("chat.completion.chunk") {
        return xai_chat_stream_event(value, provider, requested_model, started, state);
    }

    let event_type = value
        .get("type")
        .and_then(Value::as_str)
        .or(event.event.as_deref())
        .unwrap_or("");

    match event_type {
        "response.output_text.delta" => {
            let Some(delta) = value
                .get("delta")
                .and_then(Value::as_str)
                .filter(|delta| !delta.is_empty())
            else {
                return Ok(StreamEventOutcome::default());
            };
            state.emitted_text_delta = true;
            Ok(StreamEventOutcome {
                events: vec![ModelStepEvent::Delta(ModelStepDelta::Text {
                    item_id: stream_item_id(&value, "xai_text"),
                    delta: delta.to_string(),
                })],
                finished: false,
            })
        }
        "response.reasoning_summary_text.delta" | "response.reasoning_text.delta" => {
            let Some(delta) = value
                .get("delta")
                .and_then(Value::as_str)
                .filter(|delta| !delta.is_empty())
            else {
                return Ok(StreamEventOutcome::default());
            };
            state.emitted_reasoning_delta = true;
            Ok(StreamEventOutcome {
                events: vec![ModelStepEvent::Delta(ModelStepDelta::ReasoningSummary {
                    item_id: stream_item_id(&value, "xai_reasoning"),
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
            let Some(delta) = value
                .get("delta")
                .and_then(Value::as_str)
                .filter(|delta| !delta.is_empty())
            else {
                return Ok(StreamEventOutcome::default());
            };
            let key = stream_item_id(&value, "xai_function_call");
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
            let events = xai_terminal_events(response, provider, requested_model, started, state)?;
            Ok(StreamEventOutcome {
                events,
                finished: true,
            })
        }
        "response.failed" => {
            let response = value.get("response").unwrap_or(&value);
            Err(provider_stream_error(response, "xAI stream failed"))
        }
        "error" => Err(provider_stream_error(
            &value,
            "xAI stream returned an error event",
        )),
        _ => Ok(StreamEventOutcome::default()),
    }
}

fn streaming_function_call(value: &Value) -> Option<(String, StreamingFunctionCall)> {
    let item = value.get("item").unwrap_or(value);
    if item.get("type").and_then(Value::as_str) != Some("function_call") {
        return None;
    }
    let key = stream_item_id(value, "xai_function_call");
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

fn xai_chat_stream_event(
    value: Value,
    provider: &ProviderName,
    requested_model: &ModelId,
    started: Instant,
    state: &mut XaiStreamState,
) -> Result<StreamEventOutcome, XaiError> {
    state.chat_seen = true;
    if let Some(model) = value.get("model").and_then(Value::as_str) {
        state.chat_model = Some(ModelId::new(model));
    }
    if let Some(usage) = value.get("usage")
        && !usage.is_null()
    {
        state.chat_usage = Some(usage.clone());
    }

    let mut events = Vec::new();
    let mut finished = false;
    if let Some(choices) = value.get("choices").and_then(Value::as_array) {
        for choice in choices {
            if let Some(delta) = choice.get("delta") {
                if let Some(text) = delta.get("content").and_then(Value::as_str)
                    && !text.is_empty()
                {
                    state.emitted_text_delta = true;
                    events.push(ModelStepEvent::Delta(ModelStepDelta::Text {
                        item_id: "xai_chat_text".to_string(),
                        delta: text.to_string(),
                    }));
                }
                if let Some(reasoning) = delta
                    .get("reasoning_content")
                    .or_else(|| delta.get("reasoning"))
                    .and_then(Value::as_str)
                    && !reasoning.is_empty()
                {
                    state.emitted_reasoning_delta = true;
                    events.push(ModelStepEvent::Delta(ModelStepDelta::ReasoningSummary {
                        item_id: "xai_chat_reasoning".to_string(),
                        provider: provider.clone(),
                        kind: Some("reasoning_content".to_string()),
                        delta: reasoning.to_string(),
                    }));
                }
            }
            if choice
                .get("finish_reason")
                .is_some_and(|reason| !reason.is_null())
            {
                finished = true;
            }
        }
    }

    if finished {
        events.extend(xai_chat_terminal_events(
            provider,
            requested_model,
            started,
            state,
        ));
    }
    Ok(StreamEventOutcome { events, finished })
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

fn xai_terminal_events(
    response: Value,
    provider: &ProviderName,
    requested_model: &ModelId,
    started: Instant,
    state: &XaiStreamState,
) -> Result<Vec<ModelStepEvent>, XaiError> {
    let parsed: ResponsesResponse = serde_json::from_value(response.clone()).map_err(|error| {
        XaiError::Decode(format!("failed to decode terminal xAI response: {error}"))
    })?;
    let model_id = parsed
        .model
        .as_deref()
        .map(ModelId::new)
        .unwrap_or_else(|| requested_model.clone());
    let usage = usage_from_xai(
        provider,
        Some(model_id.clone()),
        UsageSubject::ModelStep,
        parsed.usage.as_ref(),
    );
    log_usage(model_id.as_str(), usage.as_ref(), started.elapsed());

    let (text, client_tool_calls, server_tool_uses) = walk_output(&parsed.output, provider);
    let kind = if !client_tool_calls.is_empty() {
        ModelStepKind::ClientTools
    } else if text.is_empty() {
        ModelStepKind::Continue
    } else {
        ModelStepKind::Final
    };

    let grounding = parsed
        .citations
        .map(|raw| {
            vec![GroundingMetadata {
                provider: provider.clone(),
                raw,
            }]
        })
        .unwrap_or_default();
    let continuation = continuation_from_output(provider, parsed.output);

    let mut events = Vec::new();
    if !state.emitted_text_delta && !text.is_empty() {
        events.push(ModelStepEvent::Delta(ModelStepDelta::Text {
            item_id: "xai_text:terminal".to_string(),
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
                    item_id: format!("xai_tool:{index}"),
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
            "xai_reasoning",
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

fn xai_chat_terminal_events(
    provider: &ProviderName,
    requested_model: &ModelId,
    started: Instant,
    state: &XaiStreamState,
) -> Vec<ModelStepEvent> {
    let model_id = state
        .chat_model
        .clone()
        .unwrap_or_else(|| requested_model.clone());
    let usage = usage_from_xai(
        provider,
        Some(model_id.clone()),
        UsageSubject::ModelStep,
        state.chat_usage.as_ref(),
    );
    log_usage(model_id.as_str(), usage.as_ref(), started.elapsed());
    let kind = if state.emitted_text_delta {
        ModelStepKind::Final
    } else {
        ModelStepKind::Continue
    };
    let mut events = usage
        .into_iter()
        .map(ModelStepEvent::Usage)
        .collect::<Vec<_>>();
    events.push(ModelStepEvent::Finished { kind, model_id });
    events
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

fn provider_stream_error(value: &Value, fallback: &str) -> XaiError {
    let message = provider_error_message(value, fallback);
    match provider_error_status(value, &message) {
        Some(status) => XaiError::Api {
            status,
            body: message,
        },
        None => XaiError::Decode(message),
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
    output: Vec<Value>,
) -> Option<ProviderContinuation> {
    // Persist the replayable subset of raw output so a later step can resume
    // xAI-side reasoning/tool state instead of reconstructing it from text.
    let items = replayable_continuation_items(output);
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
            "parameters": xai_tool_parameters(&tool.input_schema),
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

fn xai_tool_parameters(input_schema: &ToolInputSchema) -> Value {
    serde_json::to_value(input_schema).expect("tool input schema serializes")
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
    #[serde(alias = "prompt_tokens")]
    input_tokens: u64,
    #[serde(default)]
    #[serde(alias = "prompt_tokens_details")]
    input_tokens_details: TokenDetails,
    #[serde(default)]
    #[serde(alias = "completion_tokens")]
    output_tokens: u64,
    #[serde(default)]
    #[serde(alias = "completion_tokens_details")]
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
    use chudbot_api::{
        ProviderOptions, SamplingNumber, SamplingOptions, ToolInputField, ToolInputSchema,
        ToolInputValueSchema, TranscriptTurn, collect_model_step,
    };
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_dump_root(name: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock should be after epoch")
            .as_nanos();
        std::env::temp_dir().join(format!("chudbot-xai-{name}-{}-{nanos}", std::process::id()))
    }

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

    #[tokio::test]
    async fn step_dump_writes_request_response_and_continuation_json() {
        let root = temp_dump_root("stream-dump");
        let request = json!({
            "prompt_cache_key": "conv/123",
            "input": [{ "role": "user", "content": "hi" }],
        });
        let mut dump = XaiStepDump::create(&root, &request).await.unwrap().unwrap();
        let event = ServerSentEvent {
            event: Some("response.completed".to_string()),
            data: json!({
                "type": "response.completed",
                "response": {
                    "id": "resp_1",
                    "model": "grok-4.3",
                    "output": [
                        { "type": "reasoning", "id": "rs_1", "encrypted_content": "BLOB" },
                        {
                            "type": "message",
                            "role": "assistant",
                            "id": "msg_1",
                            "content": [{ "type": "output_text", "text": "hello" }],
                        }
                    ],
                    "usage": { "input_tokens": 10, "output_tokens": 2, "total_tokens": 12 }
                }
            })
            .to_string(),
        };

        dump.write_stream_event(&event).await.unwrap();

        let dir = root.join("conv_123");
        let request_file = fs::read_to_string(dir.join("0001_request.json"))
            .await
            .unwrap();
        let response_file = fs::read_to_string(dir.join("0001_response.json"))
            .await
            .unwrap();
        let continuation_file = fs::read_to_string(dir.join("0001_continuation.json"))
            .await
            .unwrap();

        assert!(request_file.contains("\"prompt_cache_key\": \"conv/123\""));
        assert!(response_file.contains("\"id\": \"resp_1\""));
        assert!(continuation_file.contains("\"encrypted_content\": \"BLOB\""));
        assert!(!dir.join("0001_event_0001_response.completed.json").exists());

        fs::remove_dir_all(root).await.ok();
    }

    #[tokio::test]
    async fn step_dump_skips_requests_without_conversation_id() {
        let root = temp_dump_root("stream-dump-no-conversation");
        let request = json!({ "input": [{ "role": "user", "content": "hi" }] });

        let dump = XaiStepDump::create(&root, &request).await.unwrap();

        assert!(dump.is_none());
        assert!(!root.exists());
    }

    #[tokio::test]
    async fn step_dump_increments_request_ordinal_per_conversation() {
        let root = temp_dump_root("stream-dump-ordinal");
        let request = json!({ "prompt_cache_key": "conv-123" });

        XaiStepDump::create(&root, &request).await.unwrap().unwrap();
        XaiStepDump::create(&root, &request).await.unwrap().unwrap();

        let dir = root.join("conv-123");
        assert!(dir.join("0001_request.json").exists());
        assert!(dir.join("0002_request.json").exists());

        fs::remove_dir_all(root).await.ok();
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

        let continuation = continuation_from_output(&provider, output).unwrap();
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

    #[test]
    fn request_body_preserves_sampling_number_literals() {
        let request = ModelStepRequest {
            model: ModelId::new("grok-4.3"),
            transcript: Transcript::from_user_text("hi"),
            client_tools: BTreeMap::new(),
            server_tools: ServerToolSet::new(),
            sampling: SamplingOptions {
                max_output_tokens: None,
                temperature: Some(SamplingNumber::from_json_number_literal("1.30").unwrap()),
                top_p: Some(SamplingNumber::from_json_number_literal("0.950").unwrap()),
            },
            provider_options: None,
        };

        let body = build_step_body_from_input(&request, vec![json!({"role": "user"})]);
        let serialized = serde_json::to_string(&body).unwrap();

        assert!(serialized.contains(r#""temperature":1.30"#));
        assert!(serialized.contains(r#""top_p":0.950"#));
    }

    #[test]
    fn streams_documented_xai_chat_completion_chunks() {
        let provider = ProviderName::new("xai");
        let requested_model = ModelId::new("grok-4.3");
        let mut state = XaiStreamState::default();
        let mut events = Vec::new();
        for data in [
            json!({
                "id": "chunk_1",
                "object": "chat.completion.chunk",
                "model": "grok-4.3",
                "choices": [{
                    "index": 0,
                    "delta": { "role": "assistant", "content": "Ah" }
                }],
                "usage": {
                    "prompt_tokens": 41,
                    "completion_tokens": 1,
                    "total_tokens": 42,
                    "prompt_tokens_details": { "cached_tokens": 3 }
                }
            })
            .to_string(),
            "[DONE]".to_string(),
        ] {
            let outcome = xai_stream_event(
                ServerSentEvent { event: None, data },
                &provider,
                &requested_model,
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
            events.into_iter().map(Ok::<_, XaiError>),
        )))
        .expect("finished step");
        assert!(matches!(step.kind(), ModelStepKind::Final));
        let output = step.output();
        assert_eq!(output.answer_text(), "Ah");
        assert_eq!(output.usage[0].input_tokens, Some(41));
        assert_eq!(output.usage[0].cached_input_tokens, Some(3));
        assert_eq!(output.usage[0].output_tokens, Some(1));
        assert_eq!(output.usage[0].total_tokens, Some(42));
    }

    #[test]
    fn empty_responses_deltas_do_not_mark_stream_emitted() {
        let provider = ProviderName::new("xai");
        let requested_model = ModelId::new("grok-4.3");
        let mut state = XaiStreamState::default();
        for data in [
            json!({
                "type": "response.output_text.delta",
                "item_id": "msg_1",
                "delta": "",
            }),
            json!({
                "type": "response.reasoning_summary_text.delta",
                "item_id": "rs_1",
                "delta": "",
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
                "delta": "",
            }),
        ] {
            let outcome = xai_stream_event(
                ServerSentEvent {
                    event: None,
                    data: data.to_string(),
                },
                &provider,
                &requested_model,
                Instant::now(),
                &mut state,
            )
            .expect("stream event");
            assert!(outcome.events.is_empty());
            assert!(!outcome.finished);
        }

        assert!(!state.emitted_text_delta);
        assert!(!state.emitted_reasoning_delta);
        assert!(state.streamed_tool_ids.is_empty());
    }

    #[test]
    fn stream_service_unavailable_error_is_retryable() {
        let provider = ProviderName::new("xai");
        let requested_model = ModelId::new("grok-4.3");
        let error = xai_stream_event(
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
            Instant::now(),
            &mut XaiStreamState::default(),
        )
        .expect_err("stream error");

        assert!(matches!(error, XaiError::Api { status: 503, .. }));
        assert_eq!(
            chudbot_api::retry::ClassifyError::error_class(&error),
            chudbot_api::retry::ErrorClass::ServerTransient
        );
    }

    #[test]
    fn builds_xai_client_tool_schema() {
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
}
