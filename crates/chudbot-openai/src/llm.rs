//! OpenAI Responses API language-model implementation.

use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use chudbot_api::{
    AssistantStep, ClientToolCall, ClientToolResult, ClientToolResultContent, ClientToolSpec,
    ContentBlock, GroundingMetadata, LlmBackend, ModelId, ModelStep, ModelStepRequest,
    ProviderContinuation, ProviderName, ServerToolSet, ServerToolUse, ToolName, ToolUseId,
    Transcript, TurnRole, UsageRecord, UsageSubject,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::image::media_bytes_or_url;
use crate::{OpenAiClient, OpenAiError, json_strip_nulls};

const REASONING_INCLUDE: &[&str] = &["reasoning.encrypted_content"];

impl LlmBackend for OpenAiClient {
    type Error = OpenAiError;

    fn backend_name(&self) -> &ProviderName {
        self.provider_name()
    }

    #[tracing::instrument(name = "openai.step", skip_all, fields(model = %request.model))]
    async fn step(&self, request: ModelStepRequest) -> Result<ModelStep, Self::Error> {
        let input = to_responses_input(&request.transcript, self).await?;
        let tools = build_responses_tools(&request.client_tools, &request.server_tools);
        let options = OpenAiOptions::from_request(&request);
        let reasoning = build_reasoning_options(&options);
        let text = build_text_options(&options);
        let sampling = model_supports_sampling(request.model.as_str());
        let has_tools = !tools.is_empty();

        let body = json_strip_nulls(json!({
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
        }));

        let started = Instant::now();
        let parsed: ResponsesResponse = self.post_json("/responses", &body, "llm[openai]").await?;
        let model_id = parsed
            .model
            .as_deref()
            .map(ModelId::new)
            .unwrap_or_else(|| request.model.clone());
        let usage = usage_from_openai(
            self.provider_name(),
            Some(model_id.clone()),
            UsageSubject::ModelStep,
            parsed.usage.as_ref(),
        );
        log_usage(model_id.as_str(), usage.as_ref(), started.elapsed());

        let output = Value::Array(parsed.output.clone());
        let continuation = (!parsed.output.is_empty()).then_some(ProviderContinuation {
            provider: self.provider_name().clone(),
            data: output,
        });

        let (text, client_tool_calls, server_tool_uses, grounding) =
            walk_output(&parsed.output, self.provider_name());

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

        if !step.client_tool_calls.is_empty() {
            Ok(ModelStep::UseClientTools { step })
        } else if step.content.is_empty() {
            Ok(ModelStep::Continue { step })
        } else {
            Ok(ModelStep::Final { step })
        }
    }
}

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
            input.extend(echo);
            continue;
        }

        let mut text = String::new();
        let mut media_urls = Vec::new();
        let mut deferred = Vec::new();
        for block in &message.blocks {
            match block {
                ContentBlock::Text { text: t } => text.push_str(t),
                ContentBlock::Media { media } => {
                    media_urls.push(media_bytes_or_url(media.as_ref()).await?)
                }
                ContentBlock::Continuation(_) => {}
                ContentBlock::ClientToolCall(call) => {
                    let args = serde_json::to_string(&call.input).unwrap_or_else(|_| "{}".into());
                    deferred.push(json!({
                        "type": "function_call",
                        "call_id": call.id.as_str(),
                        "name": call.name.as_str(),
                        "arguments": args,
                    }));
                }
                ContentBlock::ClientToolResult(result) => {
                    deferred.push(json!({
                        "type": "function_call_output",
                        "call_id": result.tool_use_id.as_str(),
                        "output": client_tool_result_as_string(result),
                    }));
                }
            }
        }

        input.extend(echo);
        if media_urls.is_empty() {
            if !text.is_empty() {
                input.push(json!({ "role": role, "content": text }));
            }
        } else {
            let mut content = Vec::with_capacity(media_urls.len() + 1);
            if !text.is_empty() {
                content.push(json!({ "type": "input_text", "text": text }));
            }
            for url in media_urls {
                content.push(json!({ "type": "input_image", "image_url": url }));
            }
            input.push(json!({ "role": role, "content": content }));
        }
        input.extend(deferred);
    }
    Ok(input)
}

fn client_tool_result_as_string(result: &ClientToolResult) -> String {
    match &result.content {
        ClientToolResultContent::Json { value } => {
            serde_json::to_string(value).unwrap_or_else(|_| value.to_string())
        }
        ClientToolResultContent::Text { text } => text.clone(),
    }
}

fn model_supports_sampling(model: &str) -> bool {
    let model = model.to_ascii_lowercase();
    let reasoning = model.starts_with("o1")
        || model.starts_with("o3")
        || model.starts_with("o4")
        || (model.starts_with("gpt-5") && !model.contains("chat"));
    !reasoning
}

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
            "parameters": tool.input_schema.as_json_schema(),
        }));
    }
    if server_tools.contains("web_search") {
        tools.push(json!({ "type": "web_search" }));
    }
    tools
}

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

fn build_text_options(options: &OpenAiOptions) -> Option<Value> {
    let value = json_strip_nulls(json!({
        "verbosity": options.text_verbosity.as_deref(),
    }));
    match &value {
        Value::Object(map) if map.is_empty() => None,
        _ => Some(value),
    }
}

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

fn usage_from_openai(
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
        input_tokens: Some(parsed.input_tokens),
        cached_input_tokens: Some(parsed.input_tokens_details.cached_tokens),
        output_tokens: Some(parsed.output_tokens),
        reasoning_tokens: Some(parsed.output_tokens_details.reasoning_tokens),
        total_tokens: Some(parsed.total_tokens),
        cost: None,
        raw: Some(raw),
    })
}

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
    fn from_request(request: &ModelStepRequest) -> Self {
        request
            .provider_options
            .as_ref()
            .and_then(|opts| serde_json::from_value(opts.value.clone()).ok())
            .unwrap_or_default()
    }
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
    use chudbot_api::{ProviderOptions, ToolInputSchema, TranscriptTurn};

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
                input_schema: ToolInputSchema::empty_object(),
            },
        );
        let tools = build_responses_tools(&client_tools, &ServerToolSet::new());
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["type"], "function");
        assert_eq!(tools[0]["name"], "fetch_messages");
        assert_eq!(tools[0]["parameters"]["type"], "object");
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
        let client = OpenAiClient::new("key");
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
        let client = OpenAiClient::new("key");
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
        )
        .unwrap();
        assert_eq!(record.input_tokens, Some(153));
        assert_eq!(record.cached_input_tokens, Some(128));
        assert_eq!(record.reasoning_tokens, Some(303));
        assert_eq!(record.total_tokens, Some(755));
    }
}
