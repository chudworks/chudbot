//! Anthropic Messages API adapter for Chudbot's provider-neutral LLM contract.
//!
//! This module owns the translation between `Transcript` turns and Anthropic
//! Messages request blocks, then maps Anthropic response blocks back into
//! `AssistantStep` content, client tool calls, server tool usage, grounding,
//! continuation state, and token usage. Provider-specific wire shapes stay here
//! so the rest of the bot can operate on `chudbot-api` types.

use std::collections::{BTreeMap, HashMap};
use std::time::{Duration, Instant};

use base64::Engine;
use base64::engine::general_purpose::STANDARD as B64;
use chudbot_api::{
    AssistantStep, ClientToolCall, ClientToolResult, ClientToolResultContent, ClientToolSpec,
    ContentBlock, GroundingMetadata, LlmBackend, MediaRef, ModelId, ModelInfo, ModelInfoRequest,
    ModelStep, ModelStepRequest, ProviderContinuation, ProviderName, ServerToolSet, ServerToolUse,
    ToolName, ToolUseId, Transcript, TurnRole, UsageRecord, UsageSubject,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::pricing::{AnthropicPricing, AnthropicTokenUsage};
use crate::{AnthropicClient, AnthropicError};

const DEFAULT_MAX_OUTPUT_TOKENS: u32 = 4096;
const WEB_SEARCH_TOOL_TYPE: &str = "web_search_20250305";
const WEB_SEARCH_TOOL_NAME: &str = "web_search";

impl LlmBackend for AnthropicClient {
    type Error = AnthropicError;

    fn backend_name(&self) -> &ProviderName {
        self.provider_name()
    }

    #[tracing::instrument(name = "anthropic.step", skip_all, fields(model = %request.model))]
    async fn step(&self, request: ModelStepRequest) -> Result<ModelStep, Self::Error> {
        // Build Anthropic's request shape first; the response parser below
        // mirrors this split so transport errors stay separate from mapping
        // provider blocks back into the bot's agent-loop model.
        let (system, mut messages) =
            to_anthropic_messages(&request.transcript, self.provider_name()).await?;
        mark_last_block_ephemeral(&mut messages);

        let tools = build_messages_tools(&request.client_tools, &request.server_tools);
        let options = AnthropicOptions::from_request(&request);
        let body = serde_json::to_value(AnthropicRequest {
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
        })
        .map_err(|e| AnthropicError::Decode(e.to_string()))?;

        let started = Instant::now();
        let parsed: AnthropicResponse =
            self.post_json("/messages", &body, "llm[anthropic]").await?;
        let model_id = parsed
            .model
            .as_deref()
            .map(ModelId::new)
            .unwrap_or_else(|| request.model.clone());
        let usage = usage_from_anthropic(
            self.provider_name(),
            Some(model_id.clone()),
            UsageSubject::ModelStep,
            parsed.usage.as_ref(),
            self.pricing(),
        );
        log_usage(model_id.as_str(), usage.as_ref(), started.elapsed());

        // Anthropic interleaves text, client tools, server tools, and citations
        // in one content array. Keep the raw array as the provider continuation
        // so pause_turn and server-tool flows can be replayed exactly.
        let (text, client_tool_calls, server_tool_uses, grounding) =
            walk_blocks(&parsed.content, self.provider_name());
        let continuation = continuation_from_content(self.provider_name(), &parsed.content);

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

        Ok(model_step_from_assistant_step(
            step,
            parsed.stop_reason.as_deref(),
        ))
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
            "input_schema": tool.input_schema.as_json_schema(),
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

/// Classify the assistant step for the provider-neutral agent loop.
fn model_step_from_assistant_step(step: AssistantStep, stop_reason: Option<&str>) -> ModelStep {
    if !step.client_tool_calls.is_empty() {
        ModelStep::UseClientTools { step }
    } else if stop_reason == Some("pause_turn") || step.content.is_empty() {
        ModelStep::Continue { step }
    } else {
        ModelStep::Final { step }
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

/// Minimal response shape needed from the Anthropic Messages API.
#[derive(Deserialize)]
struct AnthropicResponse {
    #[serde(default)]
    content: Vec<Value>,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    stop_reason: Option<String>,
    #[serde(default)]
    usage: Option<Value>,
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
        MediaRef, MediaUri, ProviderName, PublicMediaUrl, ToolUseId, TranscriptTurn, TurnRole,
        UrlMediaRef,
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
        let step = AssistantStep {
            content: vec![ContentBlock::Text {
                text: "I'll search for that.".to_string(),
            }],
            client_tool_calls: Vec::new(),
            server_tool_uses: Vec::new(),
            grounding: Vec::new(),
            model_id: ModelId::new("claude-sonnet-4-6"),
            continuation: continuation_from_content(
                &ProviderName::new("anthropic"),
                &[json!({"type": "text", "text": "I'll search for that."})],
            ),
            usage: Vec::new(),
        };

        assert!(matches!(
            model_step_from_assistant_step(step, Some("pause_turn")),
            ModelStep::Continue { .. }
        ));
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
