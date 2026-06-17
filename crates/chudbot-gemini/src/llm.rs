//! Google Gemini API language-model implementation.

use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use chudbot_api::{
    AssistantStep, ClientToolCall, ClientToolResult, ClientToolResultContent, ClientToolSpec,
    ContentBlock, GroundingMetadata, LlmBackend, ModelId, ModelInfo, ModelInfoRequest, ModelStep,
    ModelStepRequest, ProviderContinuation, ProviderName, ServerToolSet, ServerToolUse, ToolName,
    ToolUseId, Transcript, TurnRole, UsageRecord, UsageSubject,
};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

use crate::{GeminiClient, GeminiError, get_field, inline_media, json_strip_nulls};

impl LlmBackend for GeminiClient {
    type Error = GeminiError;

    fn backend_name(&self) -> &ProviderName {
        self.provider_name()
    }

    #[tracing::instrument(name = "gemini.step", skip_all, fields(model = %request.model))]
    async fn step(&self, request: ModelStepRequest) -> Result<ModelStep, Self::Error> {
        let contents = to_gemini_contents(&request.transcript, self.provider_name()).await?;
        let tools = build_tools(&request.client_tools, &request.server_tools);
        let tool_config = build_tool_config(&request.server_tools);
        let options = GeminiOptions::from_request(&request);
        let generation_config = build_generation_config(&request, &options);

        let mut body = json_strip_nulls(json!({
            "contents": contents,
            "systemInstruction": system_instruction(&request.transcript),
            "tools": (!tools.is_empty()).then_some(tools),
            "toolConfig": tool_config,
            "generationConfig": generation_config,
        }));
        merge_extra_body(&mut body, options.extra_body);

        let endpoint = format!("/models/{}:generateContent", request.model.as_str());
        let started = Instant::now();
        let parsed: Value = self.post_json(&endpoint, &body, "llm[gemini]").await?;
        let candidate = parsed
            .get("candidates")
            .and_then(Value::as_array)
            .and_then(|items| items.first())
            .ok_or_else(|| GeminiError::Decode("response had no candidates".to_string()))?;
        let content = candidate
            .get("content")
            .ok_or_else(|| GeminiError::Decode("candidate had no content".to_string()))?;

        let model_id = parsed
            .get("modelVersion")
            .or_else(|| parsed.get("model_version"))
            .and_then(Value::as_str)
            .map(ModelId::new)
            .unwrap_or_else(|| request.model.clone());
        let usage = usage_from_gemini(
            self.provider_name(),
            Some(model_id.clone()),
            UsageSubject::ModelStep,
            get_field(&parsed, "usageMetadata", "usage_metadata"),
        );
        log_usage(model_id.as_str(), usage.as_ref(), started.elapsed());

        let (text, client_tool_calls, mut server_tool_uses, mut grounding) =
            walk_content(content, self.provider_name());
        if let Some(metadata) = get_field(candidate, "groundingMetadata", "grounding_metadata") {
            grounding.push(GroundingMetadata {
                provider: self.provider_name().clone(),
                raw: metadata.clone(),
            });
        }
        if let Some(invocations) = get_field(
            candidate,
            "serverSideToolInvocations",
            "server_side_tool_invocations",
        )
        .and_then(Value::as_array)
        {
            server_tool_uses.extend(invocations.iter().cloned().map(|raw| {
                ServerToolUse {
                    provider: self.provider_name().clone(),
                    name: ToolName::new("web_search"),
                    id: raw.get("id").and_then(Value::as_str).map(str::to_string),
                    status: raw
                        .get("status")
                        .and_then(Value::as_str)
                        .map(str::to_string),
                    raw,
                    usage: Vec::new(),
                }
            }));
        }

        let mut assistant_content = Vec::new();
        if !text.is_empty() {
            assistant_content.push(ContentBlock::Text { text });
        }
        let step = AssistantStep {
            content: assistant_content,
            client_tool_calls,
            server_tool_uses,
            grounding,
            model_id,
            continuation: continuation_from_content(self.provider_name(), content),
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

    #[tracing::instrument(name = "gemini.model_info", skip_all, fields(model = %request.model))]
    async fn fetch_model_info(
        &self,
        request: ModelInfoRequest,
    ) -> Result<Option<ModelInfo>, Self::Error> {
        let endpoint = gemini_model_endpoint(&request.model);
        let raw: Value = self.get_json(&endpoint, "model_info[gemini]").await?;
        Ok(Some(model_info_from_gemini(request.model, raw)))
    }
}

fn gemini_model_endpoint(model: &ModelId) -> String {
    let model = model.as_str().trim_start_matches("models/");
    format!("/models/{model}")
}

fn model_info_from_gemini(requested_model: ModelId, raw: Value) -> ModelInfo {
    let id = get_field(&raw, "baseModelId", "base_model_id")
        .and_then(Value::as_str)
        .or_else(|| raw.get("name").and_then(Value::as_str))
        .map(|value| value.trim_start_matches("models/"))
        .filter(|value| !value.is_empty())
        .map(ModelId::new)
        .unwrap_or(requested_model);
    ModelInfo {
        id,
        context_window_tokens: u64_field(&raw, &["inputTokenLimit", "input_token_limit"]),
        max_output_tokens: u64_field(&raw, &["outputTokenLimit", "output_token_limit"]),
        raw: Some(raw),
    }
}

fn u64_field(value: &Value, names: &[&str]) -> Option<u64> {
    names
        .iter()
        .find_map(|name| value.get(*name).and_then(value_as_u64))
}

fn value_as_u64(value: &Value) -> Option<u64> {
    match value {
        Value::Number(number) => number.as_u64(),
        Value::String(text) => text.parse().ok(),
        _ => None,
    }
}

async fn to_gemini_contents(
    transcript: &Transcript,
    provider: &ProviderName,
) -> Result<Vec<Value>, GeminiError> {
    let mut contents = Vec::new();
    let mut call_names = BTreeMap::new();

    for turn in &transcript.turns {
        if let Some(continuation) = provider_continuation_content(turn, provider) {
            index_function_calls(&continuation, &mut call_names);
            contents.push(continuation);
            continue;
        }

        let role = match turn.role {
            TurnRole::Assistant => "model",
            TurnRole::User => "user",
        };
        let mut parts = Vec::new();
        for block in &turn.blocks {
            match block {
                ContentBlock::Text { text } if !text.is_empty() => {
                    parts.push(json!({ "text": text }));
                }
                ContentBlock::Text { .. } => {}
                ContentBlock::Media { media } => {
                    parts.push(inline_media(media.as_ref()).await?);
                }
                ContentBlock::ClientToolCall(call) => {
                    call_names.insert(call.id.as_str().to_string(), call.name.as_str().to_string());
                    parts.push(function_call_part(call));
                }
                ContentBlock::ClientToolResult(result) => {
                    let name = call_names
                        .get(result.tool_use_id.as_str())
                        .map(String::as_str)
                        .unwrap_or("tool_result");
                    parts.push(function_response_part(result, name));
                }
                ContentBlock::Continuation(_) => {}
            }
        }

        if !parts.is_empty() {
            contents.push(json!({ "role": role, "parts": parts }));
        }
    }

    Ok(contents)
}

fn provider_continuation_content(
    turn: &chudbot_api::TranscriptTurn,
    provider: &ProviderName,
) -> Option<Value> {
    for block in &turn.blocks {
        if let ContentBlock::Continuation(continuation) = block
            && &continuation.provider == provider
        {
            let mut data = continuation.data.clone();
            if data.get("role").is_none()
                && let Some(obj) = data.as_object_mut()
            {
                obj.insert("role".to_string(), Value::String("model".to_string()));
            }
            return Some(data);
        }
    }
    None
}

fn index_function_calls(content: &Value, call_names: &mut BTreeMap<String, String>) {
    let Some(parts) = content.get("parts").and_then(Value::as_array) else {
        return;
    };
    for part in parts {
        if let Some(function_call) = get_field(part, "functionCall", "function_call") {
            let id = get_field(function_call, "id", "id").and_then(Value::as_str);
            let name = get_field(function_call, "name", "name").and_then(Value::as_str);
            if let (Some(id), Some(name)) = (id, name) {
                call_names.insert(id.to_string(), name.to_string());
            }
        }
    }
}

fn system_instruction(transcript: &Transcript) -> Option<Value> {
    transcript
        .instructions
        .as_ref()
        .filter(|instructions| !instructions.is_empty())
        .map(|instructions| json!({ "parts": [{ "text": instructions }] }))
}

fn function_call_part(call: &ClientToolCall) -> Value {
    json!({
        "functionCall": {
            "id": call.id.as_str(),
            "name": call.name.as_str(),
            "args": call.input,
        }
    })
}

fn function_response_part(result: &ClientToolResult, name: &str) -> Value {
    json!({
        "functionResponse": {
            "id": result.tool_use_id.as_str(),
            "name": name,
            "response": tool_result_response(result),
        }
    })
}

fn tool_result_response(result: &ClientToolResult) -> Value {
    let mut response = Map::new();
    match &result.content {
        ClientToolResultContent::Json { value } if !result.is_error => {
            if let Some(obj) = value.as_object() {
                return Value::Object(obj.clone());
            }
            response.insert("result".to_string(), value.clone());
        }
        ClientToolResultContent::Json { value } => {
            response.insert("error".to_string(), value.clone());
        }
        ClientToolResultContent::Text { text } if result.is_error => {
            response.insert("error".to_string(), Value::String(text.clone()));
        }
        ClientToolResultContent::Text { text } => {
            response.insert("result".to_string(), Value::String(text.clone()));
        }
    }
    Value::Object(response)
}

fn build_tools(
    client_tools: &BTreeMap<ToolName, ClientToolSpec>,
    server_tools: &ServerToolSet,
) -> Vec<Value> {
    let mut tools = Vec::with_capacity(client_tools.len().min(1) + 1);
    if !client_tools.is_empty() {
        tools.push(json!({
            "functionDeclarations": client_tools.iter().map(|(name, tool)| {
                json!({
                    "name": name.as_str(),
                    "description": tool.description,
                    "parameters": tool.input_schema.as_json_schema(),
                })
            }).collect::<Vec<_>>(),
        }));
    }
    if server_tools.contains("web_search") {
        tools.push(json!({ "googleSearch": {} }));
    }
    tools
}

fn build_tool_config(server_tools: &ServerToolSet) -> Option<Value> {
    server_tools
        .contains("web_search")
        .then(|| json!({ "includeServerSideToolInvocations": true }))
}

fn build_generation_config(request: &ModelStepRequest, options: &GeminiOptions) -> Option<Value> {
    let thinking_config = build_thinking_config(options);
    let value = json_strip_nulls(json!({
        "maxOutputTokens": request.sampling.max_output_tokens,
        "temperature": request.sampling.temperature,
        "topP": request.sampling.top_p,
        "thinkingConfig": thinking_config,
    }));
    match &value {
        Value::Object(map) if map.is_empty() => None,
        _ => Some(value),
    }
}

fn build_thinking_config(options: &GeminiOptions) -> Option<Value> {
    let value = json_strip_nulls(json!({
        "thinkingLevel": options.thinking_level.as_deref(),
        "thinkingBudget": options.thinking_budget,
        "includeThoughts": options.include_thoughts,
    }));
    match &value {
        Value::Object(map) if map.is_empty() => None,
        _ => Some(value),
    }
}

fn walk_content(
    content: &Value,
    provider: &ProviderName,
) -> (
    String,
    Vec<ClientToolCall>,
    Vec<ServerToolUse>,
    Vec<GroundingMetadata>,
) {
    let mut text = String::new();
    let mut client_tool_calls = Vec::new();
    let mut server_tool_uses = Vec::new();
    let mut grounding = Vec::new();

    let Some(parts) = content.get("parts").and_then(Value::as_array) else {
        return (text, client_tool_calls, server_tool_uses, grounding);
    };
    for part in parts {
        let thought = get_field(part, "thought", "thought")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        if !thought && let Some(part_text) = part.get("text").and_then(Value::as_str) {
            text.push_str(part_text);
        }
        if let Some(function_call) = get_field(part, "functionCall", "function_call") {
            client_tool_calls.push(client_tool_call_from_part(function_call));
        }
        if let Some(executable_code) = get_field(part, "executableCode", "executable_code") {
            server_tool_uses.push(ServerToolUse {
                provider: provider.clone(),
                name: ToolName::new("code_execution"),
                id: None,
                status: None,
                raw: executable_code.clone(),
                usage: Vec::new(),
            });
        }
        if let Some(code_result) = get_field(part, "codeExecutionResult", "code_execution_result") {
            server_tool_uses.push(ServerToolUse {
                provider: provider.clone(),
                name: ToolName::new("code_execution"),
                id: None,
                status: get_field(code_result, "outcome", "outcome")
                    .and_then(Value::as_str)
                    .map(str::to_string),
                raw: code_result.clone(),
                usage: Vec::new(),
            });
        }
        if let Some(metadata) = get_field(part, "groundingMetadata", "grounding_metadata") {
            grounding.push(GroundingMetadata {
                provider: provider.clone(),
                raw: metadata.clone(),
            });
        }
    }

    (text, client_tool_calls, server_tool_uses, grounding)
}

fn client_tool_call_from_part(function_call: &Value) -> ClientToolCall {
    let id = function_call
        .get("id")
        .and_then(Value::as_str)
        .or_else(|| function_call.get("name").and_then(Value::as_str))
        .unwrap_or("");
    let name = function_call
        .get("name")
        .and_then(Value::as_str)
        .unwrap_or("");
    let input = function_call
        .get("args")
        .cloned()
        .unwrap_or_else(|| Value::Object(Map::new()));
    ClientToolCall {
        id: ToolUseId::new(id),
        name: ToolName::new(name),
        input,
    }
}

fn continuation_from_content(
    provider: &ProviderName,
    content: &Value,
) -> Option<ProviderContinuation> {
    let has_parts = content
        .get("parts")
        .and_then(Value::as_array)
        .is_some_and(|parts| !parts.is_empty());
    has_parts.then_some(ProviderContinuation {
        provider: provider.clone(),
        data: content.clone(),
    })
}

fn merge_extra_body(body: &mut Value, extra_body: Option<Value>) {
    let Some(Value::Object(extra)) = extra_body else {
        return;
    };
    let Some(base) = body.as_object_mut() else {
        return;
    };
    for (key, value) in extra {
        base.insert(key, value);
    }
}

fn log_usage(model: &str, usage: Option<&UsageRecord>, elapsed: Duration) {
    let duration_ms = elapsed.as_millis() as u64;
    match usage {
        Some(u) => tracing::info!(
            target: "gemini_usage",
            model = %model,
            input_tokens = u.input_tokens.unwrap_or(0),
            cached_tokens = u.cached_input_tokens.unwrap_or(0),
            output_tokens = u.output_tokens.unwrap_or(0),
            reasoning_tokens = u.reasoning_tokens.unwrap_or(0),
            total_tokens = u.total_tokens.unwrap_or(0),
            duration_ms,
            "Gemini generateContent request complete",
        ),
        None => tracing::info!(
            target: "gemini_usage",
            model = %model,
            duration_ms,
            "Gemini generateContent request complete; no usage reported",
        ),
    }
}

pub(crate) fn usage_from_gemini(
    provider: &ProviderName,
    model: Option<ModelId>,
    subject: UsageSubject,
    usage: Option<&Value>,
) -> Option<UsageRecord> {
    let raw = usage?.clone();
    let parsed = serde_json::from_value::<UsageMetadata>(raw.clone()).ok()?;
    Some(UsageRecord {
        provider: provider.clone(),
        model,
        subject,
        input_tokens: Some(parsed.prompt_token_count),
        cached_input_tokens: parsed.cached_content_token_count,
        output_tokens: Some(parsed.candidates_token_count),
        reasoning_tokens: parsed.thoughts_token_count,
        total_tokens: Some(parsed.total_token_count),
        cost: None,
        raw: Some(raw),
    })
}

/// Gemini-specific per-request knobs.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct GeminiOptions {
    /// Gemini thinking level, e.g. `minimal`, `low`, `medium`, or `high`.
    #[serde(default)]
    pub thinking_level: Option<String>,
    /// Gemini thinking budget when using models that accept token budgets.
    #[serde(default)]
    pub thinking_budget: Option<i32>,
    /// Whether Gemini should include thought parts in the raw response.
    #[serde(default)]
    pub include_thoughts: Option<bool>,
    /// Extra top-level request fields merged after Chudbot's normal body.
    #[serde(default)]
    pub extra_body: Option<Value>,
}

impl GeminiOptions {
    fn from_request(request: &ModelStepRequest) -> Self {
        request
            .provider_options
            .as_ref()
            .and_then(|options| serde_json::from_value(options.value.clone()).ok())
            .unwrap_or_default()
    }
}

#[derive(Deserialize, Debug, Default)]
struct UsageMetadata {
    #[serde(default, alias = "promptTokenCount")]
    prompt_token_count: u64,
    #[serde(default, alias = "cachedContentTokenCount")]
    cached_content_token_count: Option<u64>,
    #[serde(default, alias = "candidatesTokenCount")]
    candidates_token_count: u64,
    #[serde(default, alias = "thoughtsTokenCount")]
    thoughts_token_count: Option<u64>,
    #[serde(default, alias = "totalTokenCount")]
    total_token_count: u64,
}

#[cfg(test)]
mod tests {
    use chudbot_api::{ProviderName, ServerToolSet};
    use serde_json::json;

    use super::*;

    #[test]
    fn builds_gemini_tools_for_functions_and_search() {
        let mut client_tools = BTreeMap::new();
        client_tools.insert(
            ToolName::new("fetch_messages"),
            ClientToolSpec {
                description: "Fetch recent messages".to_string(),
                input_schema: chudbot_api::ToolInputSchema::empty_object(),
            },
        );
        let mut server_tools = ServerToolSet::new();
        server_tools.insert("web_search".to_string());

        let tools = build_tools(&client_tools, &server_tools);

        assert_eq!(tools.len(), 2);
        assert_eq!(
            tools[0]["functionDeclarations"][0]["name"],
            "fetch_messages"
        );
        assert_eq!(tools[1], json!({ "googleSearch": {} }));
        assert_eq!(
            build_tool_config(&server_tools),
            Some(json!({ "includeServerSideToolInvocations": true }))
        );
    }

    #[test]
    fn gemini_usage_preserves_token_counts() {
        let provider = ProviderName::new("gemini");
        let usage = usage_from_gemini(
            &provider,
            Some(ModelId::new("gemini-3.5-flash")),
            UsageSubject::ModelStep,
            Some(&json!({
                "promptTokenCount": 12,
                "cachedContentTokenCount": 3,
                "candidatesTokenCount": 8,
                "thoughtsTokenCount": 5,
                "totalTokenCount": 25
            })),
        )
        .unwrap();

        assert_eq!(usage.provider, provider);
        assert_eq!(usage.input_tokens, Some(12));
        assert_eq!(usage.cached_input_tokens, Some(3));
        assert_eq!(usage.output_tokens, Some(8));
        assert_eq!(usage.reasoning_tokens, Some(5));
        assert_eq!(usage.total_tokens, Some(25));
        assert!(usage.raw.is_some());
    }

    #[test]
    fn gemini_model_info_preserves_context_limits() {
        let info = model_info_from_gemini(
            ModelId::new("gemini-3.5-flash"),
            json!({
                "name": "models/gemini-3.5-flash",
                "baseModelId": "gemini-3.5-flash",
                "inputTokenLimit": 1048576,
                "outputTokenLimit": 65536
            }),
        );

        assert_eq!(info.id, ModelId::new("gemini-3.5-flash"));
        assert_eq!(info.context_window_tokens, Some(1_048_576));
        assert_eq!(info.max_output_tokens, Some(65_536));
        assert!(info.raw.is_some());
    }

    #[test]
    fn tool_result_wraps_non_object_json() {
        let result = ClientToolResult {
            tool_use_id: ToolUseId::new("call_1"),
            content: ClientToolResultContent::Json {
                value: json!(["a", "b"]),
            },
            is_error: false,
        };

        let part = function_response_part(&result, "lookup");

        assert_eq!(part["functionResponse"]["id"], "call_1");
        assert_eq!(part["functionResponse"]["name"], "lookup");
        assert_eq!(
            part["functionResponse"]["response"]["result"],
            json!(["a", "b"])
        );
    }
}
