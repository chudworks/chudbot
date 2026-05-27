//! xAI Grok provider. Talks to the OpenAI-compatible
//! `/v1/chat/completions` endpoint at `api.x.ai`. Supports:
//!   - server-side web search via `search_parameters` when
//!     [`StepRequest::enable_web_search`] is set;
//!   - client-side tools via OpenAI-style `tools` + `tool_calls` rounds.

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::config::XaiConfig;
use crate::llm::{
    ChatTurn, LlmError, LlmProvider, MessageRole, StepRequest, StepResponse, ToolCallRecord,
    ToolDefinition, ToolUseRequest, TurnBlock,
};

const DEFAULT_BASE_URL: &str = "https://api.x.ai/v1";

/// xAI Grok provider.
#[derive(Debug, Clone)]
pub struct XaiProvider {
    http: reqwest::Client,
    api_key: String,
    model: String,
    base_url: String,
    name: String,
}

impl XaiProvider {
    /// Construct from a config block. Uses the default `api.x.ai/v1`.
    pub fn new(config: XaiConfig) -> Self {
        let name = format!("xai/{}", config.model);
        Self {
            http: reqwest::Client::new(),
            api_key: config.api_key,
            model: config.model,
            base_url: DEFAULT_BASE_URL.to_string(),
            name,
        }
    }

    /// Override the base URL. Used by tests.
    pub fn with_base_url(mut self, base_url: String) -> Self {
        self.base_url = base_url;
        self
    }
}

impl LlmProvider for XaiProvider {
    fn name(&self) -> &str {
        &self.name
    }

    async fn step(&self, request: StepRequest) -> Result<StepResponse, LlmError> {
        let openai_messages = to_openai_messages(&request.messages);
        let openai_tools = to_openai_tools(&request.tools);

        let body = XaiRequest {
            model: &self.model,
            messages: &openai_messages,
            tools: if openai_tools.is_empty() {
                None
            } else {
                Some(&openai_tools)
            },
            search_parameters: request.enable_web_search.then(|| XaiSearchParameters {
                mode: "on",
                return_citations: true,
            }),
        };

        let resp = self
            .http
            .post(format!("{}/chat/completions", self.base_url))
            .bearer_auth(&self.api_key)
            .json(&body)
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

        let parsed: XaiResponse = resp
            .json()
            .await
            .map_err(|e| LlmError::Decode(e.to_string()))?;

        let model_id = parsed.model.unwrap_or_else(|| self.model.clone());

        // Server-side web search citations attach to the response as a
        // top-level `citations` array. Record as a single ToolCallRecord.
        let mut server_tool_calls = Vec::new();
        if let Some(citations) = parsed.citations {
            server_tool_calls.push(ToolCallRecord {
                tool_name: "web_search".to_string(),
                request: json!({ "mode": "on", "via": "search_parameters" }),
                response: citations,
            });
        }

        let choice = parsed
            .choices
            .into_iter()
            .next()
            .ok_or_else(|| LlmError::Decode("no choices in response".into()))?;

        let message = choice.message;
        let finish = choice.finish_reason.unwrap_or_default();

        if finish == "tool_calls" && !message.tool_calls.is_empty() {
            let mut tool_uses = Vec::with_capacity(message.tool_calls.len());
            for tc in message.tool_calls {
                let input: Value = serde_json::from_str(&tc.function.arguments).map_err(|e| {
                    LlmError::MalformedToolCall(format!(
                        "could not parse tool arguments json for `{}`: {e}",
                        tc.function.name
                    ))
                })?;
                tool_uses.push(ToolUseRequest {
                    id: tc.id,
                    name: tc.function.name,
                    input,
                });
            }
            Ok(StepResponse::UseTools {
                partial_text: if message.content.is_empty() {
                    None
                } else {
                    Some(message.content)
                },
                tool_uses,
                server_tool_calls,
                model_id,
            })
        } else {
            Ok(StepResponse::Final {
                content: message.content,
                server_tool_calls,
                model_id,
            })
        }
    }
}

/// Convert our [`ChatTurn`]s into OpenAI's flat message list. Tool-result
/// blocks expand to multiple `role: "tool"` messages (one per result).
fn to_openai_messages(turns: &[ChatTurn]) -> Vec<Value> {
    let mut out = Vec::with_capacity(turns.len());
    for turn in turns {
        let role = turn.role.as_str();
        // Split blocks: text/tool_uses on the speaker turn, tool_results
        // emitted as separate "tool" messages.
        let mut text = String::new();
        let mut tool_calls: Vec<Value> = Vec::new();
        for block in &turn.blocks {
            match block {
                TurnBlock::Text(t) => text.push_str(t),
                TurnBlock::ToolUse { id, name, input } => {
                    // OpenAI expects arguments as a STRING (json-encoded).
                    let args = serde_json::to_string(input).unwrap_or_else(|_| "{}".into());
                    tool_calls.push(json!({
                        "id": id,
                        "type": "function",
                        "function": { "name": name, "arguments": args },
                    }));
                }
                TurnBlock::ToolResult {
                    tool_use_id,
                    content,
                    ..
                } => {
                    // Emitted as its own "tool" message below. Skip here.
                    let _ = (tool_use_id, content);
                }
            }
        }
        // Speaker turn (only if there's text or tool_calls to emit).
        let has_speaker_content = !text.is_empty() || !tool_calls.is_empty();
        if has_speaker_content && turn.role != MessageRole::User {
            // Assistant or system turn with text/tool_calls.
            let mut msg = serde_json::Map::new();
            msg.insert("role".into(), Value::String(role.into()));
            if !text.is_empty() {
                msg.insert("content".into(), Value::String(text));
            } else {
                msg.insert("content".into(), Value::Null);
            }
            if !tool_calls.is_empty() {
                msg.insert("tool_calls".into(), Value::Array(tool_calls));
            }
            out.push(Value::Object(msg));
        } else if turn.role == MessageRole::User && !text.is_empty() {
            // Plain user message.
            out.push(json!({ "role": "user", "content": text }));
        }
        // Tool results — each as its own "tool" message.
        for block in &turn.blocks {
            if let TurnBlock::ToolResult {
                tool_use_id,
                content,
                ..
            } = block
            {
                out.push(json!({
                    "role": "tool",
                    "tool_call_id": tool_use_id,
                    "content": content,
                }));
            }
        }
    }
    out
}

fn to_openai_tools(tools: &[ToolDefinition]) -> Vec<Value> {
    tools
        .iter()
        .map(|t| {
            json!({
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.input_schema,
                },
            })
        })
        .collect()
}

#[derive(Serialize)]
struct XaiRequest<'a> {
    model: &'a str,
    messages: &'a [Value],
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<&'a [Value]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    search_parameters: Option<XaiSearchParameters>,
}

#[derive(Serialize)]
struct XaiSearchParameters {
    mode: &'static str,
    return_citations: bool,
}

#[derive(Deserialize)]
struct XaiResponse {
    choices: Vec<XaiChoice>,
    #[serde(default)]
    citations: Option<Value>,
    #[serde(default)]
    model: Option<String>,
}

#[derive(Deserialize)]
struct XaiChoice {
    message: XaiResponseMessage,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct XaiResponseMessage {
    #[serde(default)]
    content: String,
    #[serde(default)]
    tool_calls: Vec<XaiToolCall>,
}

#[derive(Deserialize)]
struct XaiToolCall {
    id: String,
    function: XaiFunctionCall,
}

#[derive(Deserialize)]
struct XaiFunctionCall {
    name: String,
    arguments: String,
}
