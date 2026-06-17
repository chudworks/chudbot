//! OpenAI-compatible Chat Completions language-model implementation.

use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use base64::Engine;
use base64::engine::general_purpose::STANDARD as B64;
use chudbot_api::{
    AssistantStep, ClientToolCall, ClientToolResult, ClientToolResultContent, ClientToolSpec,
    ContentBlock, LlmBackend, MediaRef, ModelId, ModelInfo, ModelInfoRequest, ModelStep,
    ModelStepRequest, ProviderContinuation, ProviderName, ToolName, ToolUseId, Transcript,
    TurnRole, UsageRecord, UsageSubject,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::{OpenAiCompatClient, OpenAiCompatError};

impl LlmBackend for OpenAiCompatClient {
    type Error = OpenAiCompatError;

    fn backend_name(&self) -> &ProviderName {
        self.provider_name()
    }

    #[tracing::instrument(
        name = "openai_compat.step",
        skip_all,
        fields(model = %request.model)
    )]
    async fn step(&self, request: ModelStepRequest) -> Result<ModelStep, Self::Error> {
        let messages = to_chat_messages(&request.transcript, self.provider_name()).await?;
        let tools = build_chat_tools(&request.client_tools);
        if !request.server_tools.is_empty() {
            tracing::debug!(
                provider = %self.provider_name(),
                server_tools = ?request.server_tools,
                "OpenAI-compatible Chat Completions does not expose configured server tools"
            );
        }

        let options = OpenAiCompatOptions::from_request(&request);
        let tool_choice = if tools.is_empty() {
            None
        } else {
            Some(options.tool_choice.unwrap_or_else(|| json!("auto")))
        };
        let mut body = json_strip_top_level_nulls(json!({
            "model": request.model.as_str(),
            "messages": messages,
            "tools": (!tools.is_empty()).then_some(tools),
            "tool_choice": tool_choice,
            "parallel_tool_calls": options.parallel_tool_calls,
            "max_tokens": request.sampling.max_output_tokens,
            "temperature": request.sampling.temperature,
            "top_p": request.sampling.top_p,
        }));
        merge_extra_body(&mut body, options.extra_body);

        let started = Instant::now();
        let parsed: ChatResponse = self
            .post_json("/chat/completions", &body, "llm[openai_compat]")
            .await?;
        let model_id = parsed
            .model
            .as_deref()
            .map(ModelId::new)
            .unwrap_or_else(|| request.model.clone());
        let usage = usage_from_compat(
            self.provider_name(),
            Some(model_id.clone()),
            UsageSubject::ModelStep,
            parsed.usage.as_ref(),
        );
        log_usage(model_id.as_str(), usage.as_ref(), started.elapsed());

        let choice = parsed
            .choices
            .into_iter()
            .next()
            .ok_or_else(|| OpenAiCompatError::Decode("response had no choices".to_string()))?;
        let text = content_to_text(choice.message.content.as_ref());
        let client_tool_calls = parse_tool_calls(
            &choice.message.tool_calls,
            choice.message.function_call.as_ref(),
        );
        let continuation = continuation_from_reasoning_content(
            self.provider_name(),
            choice.message.reasoning_content.as_deref(),
        );

        let mut content = Vec::new();
        if !text.is_empty() {
            content.push(ContentBlock::Text { text });
        }

        let step = AssistantStep {
            content,
            client_tool_calls,
            server_tool_uses: Vec::new(),
            grounding: Vec::new(),
            model_id,
            continuation,
            usage: usage.into_iter().collect(),
        };

        if !step.client_tool_calls.is_empty() {
            Ok(ModelStep::UseClientTools { step })
        } else {
            Ok(ModelStep::Final { step })
        }
    }

    #[tracing::instrument(
        name = "openai_compat.model_info",
        skip_all,
        fields(model = %request.model)
    )]
    async fn fetch_model_info(
        &self,
        request: ModelInfoRequest,
    ) -> Result<Option<ModelInfo>, Self::Error> {
        let raw: Value = self
            .get_json("/models", "model_info[openai_compat]")
            .await?;
        Ok(model_info_from_models_response(request.model, raw))
    }
}

fn model_info_from_models_response(requested_model: ModelId, raw: Value) -> Option<ModelInfo> {
    let entry = model_entry_from_models_response(&requested_model, &raw)?;
    model_info_from_compat_model(requested_model, entry)
}

fn model_entry_from_models_response(requested_model: &ModelId, raw: &Value) -> Option<Value> {
    if model_id_matches(raw, requested_model) {
        return Some(raw.clone());
    }

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

fn model_info_from_compat_model(requested_model: ModelId, raw: Value) -> Option<ModelInfo> {
    const CONTEXT_FIELDS: &[&str] = &[
        "context_window_tokens",
        "context_window",
        "context_length",
        "max_context_length",
        "max_context_len",
        "max_model_len",
        "max_sequence_length",
        "max_position_embeddings",
        "n_ctx",
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

async fn to_chat_messages(
    transcript: &Transcript,
    provider: &ProviderName,
) -> Result<Vec<Value>, OpenAiCompatError> {
    let mut messages = Vec::new();
    if let Some(instructions) = &transcript.instructions
        && !instructions.is_empty()
    {
        messages.push(json!({ "role": "system", "content": instructions }));
    }

    for turn in &transcript.turns {
        let role = match turn.role {
            TurnRole::Assistant => "assistant",
            TurnRole::User => "user",
        };

        if turn.role == TurnRole::Assistant {
            let mut text = String::new();
            let mut tool_calls = Vec::new();
            for block in &turn.blocks {
                match block {
                    ContentBlock::Text { text: t } => text.push_str(t),
                    ContentBlock::Media { media } => {
                        tracing::debug!(
                            uri = %media.uri(),
                            "skipping assistant media for OpenAI-compatible chat history"
                        );
                    }
                    ContentBlock::ClientToolCall(call) => {
                        tool_calls.push(chat_tool_call(call));
                    }
                    ContentBlock::ClientToolResult(_) => {}
                    ContentBlock::Continuation(continuation) => {
                        if &continuation.provider == provider {
                            tracing::debug!(
                                provider = %provider,
                                "OpenAI-compatible provider continuations are not replayed"
                            );
                        }
                    }
                }
            }
            if !text.is_empty() || !tool_calls.is_empty() {
                let content = if tool_calls.is_empty() || !text.is_empty() {
                    Value::String(text)
                } else {
                    Value::Null
                };
                let mut message = serde_json::Map::new();
                message.insert("role".to_string(), Value::String(role.to_string()));
                message.insert("content".to_string(), content);
                if !tool_calls.is_empty() {
                    message.insert("tool_calls".to_string(), Value::Array(tool_calls));
                }
                messages.push(Value::Object(message));
            }
        } else {
            let mut text = String::new();
            let mut media_urls = Vec::new();
            for block in &turn.blocks {
                match block {
                    ContentBlock::Text { text: t } => text.push_str(t),
                    ContentBlock::Media { media } => {
                        media_urls.push(media_url_or_data(media.as_ref()).await?);
                    }
                    ContentBlock::ClientToolCall(_) => {}
                    ContentBlock::ClientToolResult(result) => {
                        push_chat_user_message(&mut messages, &mut text, &mut media_urls);
                        messages.push(chat_tool_result(result));
                    }
                    ContentBlock::Continuation(continuation) => {
                        if &continuation.provider == provider {
                            tracing::debug!(
                                provider = %provider,
                                "OpenAI-compatible provider continuations are not replayed"
                            );
                        }
                    }
                }
            }
            push_chat_user_message(&mut messages, &mut text, &mut media_urls);
        }
    }

    Ok(messages)
}

fn push_chat_user_message(
    messages: &mut Vec<Value>,
    text: &mut String,
    media_urls: &mut Vec<String>,
) {
    if text.is_empty() && media_urls.is_empty() {
        return;
    }
    if media_urls.is_empty() {
        messages.push(json!({ "role": "user", "content": text.as_str() }));
        text.clear();
        return;
    }

    let mut parts = Vec::with_capacity(media_urls.len() + 1);
    if !text.is_empty() {
        parts.push(json!({ "type": "text", "text": text.as_str() }));
    }
    for url in media_urls.iter() {
        parts.push(json!({
            "type": "image_url",
            "image_url": { "url": url },
        }));
    }
    messages.push(json!({ "role": "user", "content": parts }));
    text.clear();
    media_urls.clear();
}

async fn media_url_or_data(media: &dyn MediaRef) -> Result<String, OpenAiCompatError> {
    match media.public_url().await {
        Ok(url) => {
            tracing::debug!(
                uri = %media.uri(),
                category = ?media.category(),
                "resolved media public URL for OpenAI-compatible chat"
            );
            Ok(url.to_string())
        }
        Err(public_error) => match media.load().await {
            Ok(loaded) => {
                tracing::debug!(
                    uri = %media.uri(),
                    category = ?media.category(),
                    bytes = loaded.bytes.len(),
                    mime_type = loaded.media.mime_type(),
                    "inlined media bytes for OpenAI-compatible chat"
                );
                Ok(data_uri(loaded.media.mime_type(), &loaded.bytes))
            }
            Err(load_error) => {
                tracing::warn!(
                    uri = %media.uri(),
                    category = ?media.category(),
                    public_error = %public_error,
                    load_error = %load_error,
                    "failed to resolve media for OpenAI-compatible chat"
                );
                Err(OpenAiCompatError::Reference(format!(
                    "media `{}` has no public URL ({public_error}) and could not be loaded ({load_error})",
                    media.uri()
                )))
            }
        },
    }
}

fn data_uri(mime_type: &str, bytes: &[u8]) -> String {
    format!("data:{mime_type};base64,{}", B64.encode(bytes))
}

fn chat_tool_call(call: &ClientToolCall) -> Value {
    let arguments = serde_json::to_string(&call.input).unwrap_or_else(|_| "{}".to_string());
    json!({
        "id": call.id.as_str(),
        "type": "function",
        "function": {
            "name": call.name.as_str(),
            "arguments": arguments,
        },
    })
}

fn chat_tool_result(result: &ClientToolResult) -> Value {
    json!({
        "role": "tool",
        "tool_call_id": result.tool_use_id.as_str(),
        "content": client_tool_result_as_string(result),
    })
}

fn client_tool_result_as_string(result: &ClientToolResult) -> String {
    match &result.content {
        ClientToolResultContent::Json { value } => {
            serde_json::to_string(value).unwrap_or_else(|_| value.to_string())
        }
        ClientToolResultContent::Text { text } => text.clone(),
    }
}

fn build_chat_tools(client_tools: &BTreeMap<ToolName, ClientToolSpec>) -> Vec<Value> {
    client_tools
        .iter()
        .map(|(name, tool)| {
            json!({
                "type": "function",
                "function": {
                    "name": name.as_str(),
                    "description": tool.description,
                    "parameters": tool.input_schema.as_json_schema(),
                },
            })
        })
        .collect()
}

fn parse_tool_calls(
    calls: &[ToolCall],
    deprecated_function_call: Option<&ToolCallFunction>,
) -> Vec<ClientToolCall> {
    if !calls.is_empty() {
        return calls
            .iter()
            .enumerate()
            .map(|(idx, call)| {
                let id = if call.id.is_empty() {
                    format!("tool_call_{idx}")
                } else {
                    call.id.clone()
                };
                ClientToolCall {
                    id: ToolUseId::new(id),
                    name: ToolName::new(call.function.name.clone()),
                    input: parse_arguments(&call.function.arguments),
                }
            })
            .collect();
    }

    deprecated_function_call
        .map(|call| {
            vec![ClientToolCall {
                id: ToolUseId::new(format!("function_call_{}", call.name)),
                name: ToolName::new(call.name.clone()),
                input: parse_arguments(&call.arguments),
            }]
        })
        .unwrap_or_default()
}

fn parse_arguments(arguments: &str) -> Value {
    serde_json::from_str(arguments).unwrap_or(Value::Null)
}

fn content_to_text(content: Option<&Value>) -> String {
    match content {
        Some(Value::String(text)) => text.clone(),
        Some(Value::Array(parts)) => parts.iter().fold(String::new(), |mut text, part| {
            if part.get("type").and_then(Value::as_str) == Some("text")
                && let Some(part_text) = part.get("text").and_then(Value::as_str)
            {
                text.push_str(part_text);
            }
            if let Some(refusal) = part.get("refusal").and_then(Value::as_str) {
                text.push_str(refusal);
            }
            text
        }),
        _ => String::new(),
    }
}

fn log_usage(model: &str, usage: Option<&UsageRecord>, elapsed: Duration) {
    let duration_ms = elapsed.as_millis() as u64;
    match usage {
        Some(u) => tracing::info!(
            target: "openai_compat_usage",
            model = %model,
            input_tokens = u.input_tokens.unwrap_or(0),
            cached_tokens = u.cached_input_tokens.unwrap_or(0),
            output_tokens = u.output_tokens.unwrap_or(0),
            reasoning_tokens = u.reasoning_tokens.unwrap_or(0),
            total_tokens = u.total_tokens.unwrap_or(0),
            duration_ms,
            "OpenAI-compatible chat completion complete",
        ),
        None => tracing::info!(
            target: "openai_compat_usage",
            model = %model,
            duration_ms,
            "OpenAI-compatible chat completion complete; no usage reported",
        ),
    }
}

fn usage_from_compat(
    provider: &ProviderName,
    model: Option<ModelId>,
    subject: UsageSubject,
    usage: Option<&Value>,
) -> Option<UsageRecord> {
    let raw = usage?.clone();
    let parsed = serde_json::from_value::<Usage>(raw.clone()).ok()?;
    Some(UsageRecord {
        provider: provider.clone(),
        model,
        subject,
        input_tokens: Some(parsed.prompt_tokens),
        cached_input_tokens: Some(parsed.prompt_tokens_details.cached_tokens),
        output_tokens: Some(parsed.completion_tokens),
        reasoning_tokens: Some(parsed.completion_tokens_details.reasoning_tokens),
        total_tokens: Some(parsed.total_tokens),
        cost: None,
        raw: Some(raw),
    })
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct OpenAiCompatOptions {
    /// Tool choice value sent when tools are present. Defaults to `"auto"`.
    #[serde(default)]
    pub tool_choice: Option<Value>,
    /// Optional Chat Completions `parallel_tool_calls` override.
    #[serde(default)]
    pub parallel_tool_calls: Option<bool>,
    /// Extra top-level request fields for vLLM or other local-server
    /// extensions, such as `top_k`, `min_p`, or `repetition_penalty`.
    #[serde(default)]
    pub extra_body: BTreeMap<String, Value>,
}

impl OpenAiCompatOptions {
    fn from_request(request: &ModelStepRequest) -> Self {
        request
            .provider_options
            .as_ref()
            .and_then(|opts| serde_json::from_value(opts.value.clone()).ok())
            .unwrap_or_default()
    }
}

fn json_strip_top_level_nulls(mut value: Value) -> Value {
    if let Value::Object(map) = &mut value {
        map.retain(|_, value| !value.is_null());
    }
    value
}

fn merge_extra_body(body: &mut Value, extra_body: BTreeMap<String, Value>) {
    let Value::Object(map) = body else {
        return;
    };
    map.extend(extra_body);
}

fn continuation_from_reasoning_content(
    provider: &ProviderName,
    reasoning_content: Option<&str>,
) -> Option<ProviderContinuation> {
    let text = reasoning_content?.trim();
    if text.is_empty() {
        return None;
    }
    Some(ProviderContinuation {
        provider: provider.clone(),
        data: json!({
            "type": "reasoning",
            "summary": [{
                "type": "reasoning_content",
                "text": text,
            }],
        }),
    })
}

#[derive(Deserialize)]
struct ChatResponse {
    #[serde(default)]
    choices: Vec<Choice>,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    usage: Option<Value>,
}

#[derive(Deserialize)]
struct Choice {
    message: ResponseMessage,
}

#[derive(Deserialize)]
struct ResponseMessage {
    #[serde(default)]
    content: Option<Value>,
    #[serde(default)]
    reasoning_content: Option<String>,
    #[serde(default)]
    tool_calls: Vec<ToolCall>,
    #[serde(default)]
    function_call: Option<ToolCallFunction>,
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
    completion_tokens_details: TokenDetails,
    #[serde(default)]
    total_tokens: u64,
}

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
        MediaCategory, ProviderOptions, ServerToolSet, ToolInputSchema, TranscriptTurn, UrlMediaRef,
    };

    #[test]
    fn system_and_user_map_to_chat_messages() {
        let mut transcript = Transcript::new();
        transcript.instructions = Some("be helpful".to_string());
        transcript.push(TranscriptTurn::text(TurnRole::User, "hi"));

        let messages =
            futures::executor::block_on(to_chat_messages(&transcript, &ProviderName::new("x")))
                .unwrap();

        assert_eq!(messages.len(), 2);
        assert_eq!(
            messages[0],
            json!({"role": "system", "content": "be helpful"})
        );
        assert_eq!(messages[1], json!({"role": "user", "content": "hi"}));
    }

    #[test]
    fn assistant_tool_call_then_tool_result_round_trips() {
        let mut transcript = Transcript::new();
        transcript.push(TranscriptTurn {
            role: TurnRole::Assistant,
            blocks: vec![
                ContentBlock::Text {
                    text: "on it".to_string(),
                },
                ContentBlock::ClientToolCall(ClientToolCall {
                    id: ToolUseId::new("call_1"),
                    name: ToolName::new("fetch_messages"),
                    input: json!({ "limit": 10 }),
                }),
            ],
            metadata: Value::Null,
        });
        transcript.push(TranscriptTurn {
            role: TurnRole::User,
            blocks: vec![ContentBlock::ClientToolResult(ClientToolResult {
                tool_use_id: ToolUseId::new("call_1"),
                content: ClientToolResultContent::Json { value: json!([]) },
                is_error: false,
            })],
            metadata: Value::Null,
        });

        let messages =
            futures::executor::block_on(to_chat_messages(&transcript, &ProviderName::new("x")))
                .unwrap();

        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0]["role"], "assistant");
        assert_eq!(messages[0]["content"], "on it");
        assert_eq!(messages[0]["tool_calls"][0]["id"], "call_1");
        assert_eq!(messages[0]["tool_calls"][0]["type"], "function");
        assert_eq!(
            messages[0]["tool_calls"][0]["function"]["name"],
            "fetch_messages"
        );
        assert_eq!(
            messages[0]["tool_calls"][0]["function"]["arguments"],
            "{\"limit\":10}"
        );
        assert_eq!(messages[1]["role"], "tool");
        assert_eq!(messages[1]["tool_call_id"], "call_1");
        assert_eq!(messages[1]["content"], "[]");
    }

    #[test]
    fn assistant_with_only_tool_calls_has_null_content() {
        let mut transcript = Transcript::new();
        transcript.push(TranscriptTurn {
            role: TurnRole::Assistant,
            blocks: vec![ContentBlock::ClientToolCall(ClientToolCall {
                id: ToolUseId::new("call_1"),
                name: ToolName::new("fetch_messages"),
                input: json!({}),
            })],
            metadata: Value::Null,
        });

        let messages =
            futures::executor::block_on(to_chat_messages(&transcript, &ProviderName::new("x")))
                .unwrap();

        assert_eq!(messages[0]["role"], "assistant");
        assert_eq!(messages[0]["content"], Value::Null);
        assert!(messages[0]["tool_calls"].is_array());
    }

    #[test]
    fn user_image_becomes_openai_content_part() {
        let mut transcript = Transcript::new();
        transcript.push(TranscriptTurn {
            role: TurnRole::User,
            blocks: vec![
                ContentBlock::Text {
                    text: "look".to_string(),
                },
                ContentBlock::Media {
                    media: UrlMediaRef::new(
                        MediaCategory::Image,
                        "https://example.com/image.png",
                        "image/png",
                    )
                    .boxed(),
                },
            ],
            metadata: Value::Null,
        });

        let messages =
            futures::executor::block_on(to_chat_messages(&transcript, &ProviderName::new("x")))
                .unwrap();

        assert_eq!(messages.len(), 1);
        assert_eq!(
            messages[0]["content"][0],
            json!({"type": "text", "text": "look"})
        );
        assert_eq!(
            messages[0]["content"][1],
            json!({
                "type": "image_url",
                "image_url": { "url": "https://example.com/image.png" },
            })
        );
    }

    #[test]
    fn builds_nested_chat_completion_tools() {
        let mut client_tools = BTreeMap::new();
        client_tools.insert(
            ToolName::new("fetch_messages"),
            ClientToolSpec {
                description: "Fetch context.".to_string(),
                input_schema: ToolInputSchema::empty_object(),
            },
        );

        let tools = build_chat_tools(&client_tools);

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
                "message": {
                    "role": "assistant",
                    "content": "the answer",
                    "reasoning_content": "thought through it",
                },
                "finish_reason": "stop",
            }],
            "usage": { "prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15 },
        });

        let parsed: ChatResponse = serde_json::from_value(body).unwrap();
        let choice = parsed.choices.into_iter().next().unwrap();

        assert_eq!(
            content_to_text(choice.message.content.as_ref()).as_str(),
            "the answer"
        );
        assert_eq!(
            choice.message.reasoning_content.as_deref(),
            Some("thought through it")
        );
        assert!(choice.message.tool_calls.is_empty());
    }

    #[test]
    fn reasoning_content_becomes_viewer_continuation() {
        let provider = ProviderName::new("openai_compat");
        let continuation =
            continuation_from_reasoning_content(&provider, Some("  considered the prompt\n  "))
                .expect("reasoning continuation");

        assert_eq!(continuation.provider, provider);
        assert_eq!(continuation.data["type"], "reasoning");
        assert_eq!(
            continuation.data["summary"][0],
            json!({
                "type": "reasoning_content",
                "text": "considered the prompt"
            })
        );
        assert!(continuation_from_reasoning_content(&provider, Some("  ")).is_none());
        assert!(continuation_from_reasoning_content(&provider, None).is_none());
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
                        "function": {
                            "name": "fetch_messages",
                            "arguments": "{\"limit\":30}"
                        },
                    }],
                },
                "finish_reason": "tool_calls",
            }],
        });

        let parsed: ChatResponse = serde_json::from_value(body).unwrap();
        let choice = parsed.choices.into_iter().next().unwrap();
        let calls = parse_tool_calls(&choice.message.tool_calls, None);

        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id.as_str(), "call_9");
        assert_eq!(calls[0].name.as_str(), "fetch_messages");
        assert_eq!(calls[0].input["limit"], 30);
    }

    #[test]
    fn parses_deprecated_function_call_response() {
        let body = json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": null,
                    "function_call": {
                        "name": "fetch_messages",
                        "arguments": "{\"limit\":30}"
                    },
                },
            }],
        });

        let parsed: ChatResponse = serde_json::from_value(body).unwrap();
        let choice = parsed.choices.into_iter().next().unwrap();
        let calls = parse_tool_calls(
            &choice.message.tool_calls,
            choice.message.function_call.as_ref(),
        );

        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id.as_str(), "function_call_fetch_messages");
        assert_eq!(calls[0].name.as_str(), "fetch_messages");
        assert_eq!(calls[0].input["limit"], 30);
    }

    #[test]
    fn parses_usage_with_cached_and_reasoning_tokens() {
        let usage = json!({
            "prompt_tokens": 100,
            "prompt_tokens_details": { "cached_tokens": 64 },
            "completion_tokens": 20,
            "completion_tokens_details": { "reasoning_tokens": 7 },
            "total_tokens": 120,
        });
        let provider = ProviderName::new("openai_compat");
        let record = usage_from_compat(
            &provider,
            Some(ModelId::new("local-model")),
            UsageSubject::ModelStep,
            Some(&usage),
        )
        .unwrap();

        assert_eq!(record.input_tokens, Some(100));
        assert_eq!(record.cached_input_tokens, Some(64));
        assert_eq!(record.output_tokens, Some(20));
        assert_eq!(record.reasoning_tokens, Some(7));
        assert_eq!(record.total_tokens, Some(120));
    }

    #[test]
    fn parses_model_info_from_models_response() {
        let info = model_info_from_models_response(
            ModelId::new("local-model"),
            json!({
                "object": "list",
                "data": [{
                    "id": "local-model",
                    "metadata": {
                        "max_model_len": "131072",
                        "max_output_tokens": 8192
                    }
                }]
            }),
        )
        .expect("model metadata");

        assert_eq!(info.id, ModelId::new("local-model"));
        assert_eq!(info.context_window_tokens, Some(131_072));
        assert_eq!(info.max_output_tokens, Some(8_192));
        assert!(info.raw.is_some());
    }

    #[test]
    fn parses_options_and_merges_extra_body() {
        let request = ModelStepRequest {
            model: ModelId::new("qwen3"),
            transcript: Transcript::from_user_text("hi"),
            client_tools: BTreeMap::new(),
            server_tools: ServerToolSet::new(),
            sampling: chudbot_api::SamplingOptions::default(),
            provider_options: Some(ProviderOptions {
                value: json!({
                    "tool_choice": "required",
                    "parallel_tool_calls": false,
                    "extra_body": {
                        "top_k": 40,
                        "min_p": 0.05
                    }
                }),
            }),
        };

        let options = OpenAiCompatOptions::from_request(&request);
        let mut body = json!({ "model": "qwen3" });
        merge_extra_body(&mut body, options.extra_body);

        assert_eq!(options.tool_choice.unwrap(), "required");
        assert_eq!(options.parallel_tool_calls, Some(false));
        assert_eq!(body["top_k"], 40);
        assert_eq!(body["min_p"], 0.05);
    }
}
