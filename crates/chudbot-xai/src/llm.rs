//! xAI Responses API language-model implementation.

use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use chudbot_api::{
    AssistantStep, ClientToolCall, ClientToolSpec, ContentBlock, CostAmount, GroundingMetadata,
    LlmBackend, ModelId, ModelStep, ModelStepRequest, ProviderContinuation, ProviderName,
    ServerToolSet, ServerToolUse, ToolName, ToolUseId, Transcript, TurnRole, UsageRecord,
    UsageSubject,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::imagine::media_provider_url;
use crate::{XaiClient, XaiError, json_strip_nulls};

const REASONING_INCLUDE: &[&str] = &["reasoning.encrypted_content"];

impl LlmBackend for XaiClient {
    type Error = XaiError;

    fn backend_name(&self) -> &ProviderName {
        self.provider_name()
    }

    #[tracing::instrument(name = "xai.step", skip_all, fields(model = %request.model))]
    async fn step(&self, request: ModelStepRequest) -> Result<ModelStep, Self::Error> {
        let input = to_responses_input(&request.transcript, self.provider_name()).await?;
        let tools = build_responses_tools(&request.client_tools, &request.server_tools);
        let options = XaiOptions::from_request(&request);
        let reasoning = options
            .reasoning_effort
            .as_ref()
            .map(|effort| json!({ "effort": effort }));

        let body = json_strip_nulls(json!({
            "model": request.model.as_str(),
            "input": input,
            "tools": (!tools.is_empty()).then_some(tools),
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

        let output = Value::Array(parsed.output.clone());
        let continuation = (!parsed.output.is_empty()).then_some(ProviderContinuation {
            provider: self.provider_name().clone(),
            data: output,
        });

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
        for block in &message.blocks {
            if let ContentBlock::Continuation(continuation) = block
                && &continuation.provider == provider
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
                    media_urls.push(media_provider_url(media.as_ref()).await?)
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
        let id = transcript_turn_message_id(message);
        if media_urls.is_empty() {
            if !text.is_empty() {
                input.push(json_strip_nulls(json!({
                    "id": id,
                    "role": role,
                    "content": text,
                })));
            }
        } else {
            let mut content = Vec::with_capacity(media_urls.len() + 1);
            if !text.is_empty() {
                content.push(json!({ "type": "input_text", "text": text }));
            }
            for url in media_urls {
                content.push(json!({ "type": "input_image", "image_url": url }));
            }
            input.push(json_strip_nulls(json!({
                "id": id,
                "role": role,
                "content": content,
            })));
        }
        input.extend(deferred);
    }
    Ok(input)
}

fn system_message_id(transcript_id: &str) -> String {
    format!("chudbot_conversation_{transcript_id}_system")
}

fn transcript_turn_message_id(message: &chudbot_api::TranscriptTurn) -> Option<&str> {
    message.metadata.get("id").and_then(Value::as_str)
}

fn client_tool_result_as_string(result: &chudbot_api::ClientToolResult) -> String {
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

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct XaiOptions {
    /// Reasoning effort: `low`, `medium`, or `high`.
    #[serde(default)]
    pub reasoning_effort: Option<String>,
}

impl XaiOptions {
    fn from_request(request: &ModelStepRequest) -> Self {
        request
            .provider_options
            .as_ref()
            .filter(|opts| opts.provider.as_str() == "xai")
            .and_then(|opts| serde_json::from_value(opts.value.clone()).ok())
            .unwrap_or_default()
    }
}

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
    fn xai_options_only_parse_for_matching_provider() {
        let request = ModelStepRequest {
            model: ModelId::new("grok-4.3"),
            transcript: Transcript::from_user_text("hi"),
            client_tools: BTreeMap::new(),
            server_tools: ServerToolSet::new(),
            sampling: chudbot_api::SamplingOptions::default(),
            provider_options: Some(ProviderOptions {
                provider: ProviderName::new("xai"),
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
