//! Agentic harness: drives a [`LlmProvider`] through a tool-use loop.
//!
//! Each iteration:
//!   1. Send the current message history + tool definitions to the
//!      provider via [`LlmProvider::step`].
//!   2. If the model returned a final answer, stop and return it.
//!   3. Otherwise (the model emitted `tool_use` blocks), execute each
//!      via the caller-supplied [`ToolExecutor`], append both the
//!      assistant turn (with the tool_use blocks) and a user turn (with
//!      the tool_result blocks) to history, and loop.
//!
//! Every tool call — server-side (web search) and client-side (whatever
//! the executor handles) — is collected in declared execution order and
//! returned in [`AgentRun::tool_calls`], so the caller can persist them
//! into the `tool_calls` table for the web viewer.

use crate::llm::{
    ChatTurn, LlmError, LlmProvider, MessageRole, StepRequest, StepResponse, ToolCallRecord,
    ToolDefinition, ToolExecutor, ToolUseRequest, TurnBlock,
};

/// Result of [`run`].
#[derive(Debug, Clone)]
pub struct AgentRun {
    /// Final answer text from the model.
    pub content: String,
    /// All tool calls (server + client) performed during the run, in
    /// execution order.
    pub tool_calls: Vec<ToolCallRecord>,
    /// Model id reported by the last step.
    pub model_id: String,
}

/// Drive the model through a tool-use loop until it produces a final
/// answer, or `max_iterations` is hit.
#[allow(clippy::too_many_arguments)]
pub async fn run<P, T>(
    provider: &P,
    initial_messages: Vec<ChatTurn>,
    tools: Vec<ToolDefinition>,
    executor: &T,
    enable_web_search: bool,
    max_tokens: u32,
    temperature: Option<f32>,
    top_p: Option<f32>,
    max_iterations: u32,
) -> Result<AgentRun, LlmError>
where
    P: LlmProvider,
    T: ToolExecutor,
{
    let mut messages = initial_messages;
    let mut all_tool_calls: Vec<ToolCallRecord> = Vec::new();
    let mut last_model_id = String::new();

    tracing::info!(
        provider = provider.name(),
        messages = messages.len(),
        client_tools = tools.len(),
        web_search = enable_web_search,
        "agent: starting loop"
    );

    for iteration in 0..max_iterations {
        let response = provider
            .step(StepRequest {
                messages: messages.clone(),
                tools: tools.clone(),
                enable_web_search,
                max_tokens,
                temperature,
                top_p,
            })
            .await?;

        match response {
            StepResponse::Final {
                content,
                server_tool_calls,
                model_id,
            } => {
                let server_calls = server_tool_calls.len();
                all_tool_calls.extend(server_tool_calls);
                tracing::info!(
                    iteration,
                    model = %model_id,
                    text_chars = content.len(),
                    server_tool_calls = server_calls,
                    total_tool_calls = all_tool_calls.len(),
                    "agent: final answer received"
                );
                return Ok(AgentRun {
                    content,
                    tool_calls: all_tool_calls,
                    model_id,
                });
            }
            StepResponse::UseTools {
                partial_text,
                tool_uses,
                server_tool_calls,
                model_id,
            } => {
                let server_calls = server_tool_calls.len();
                all_tool_calls.extend(server_tool_calls);
                last_model_id = model_id;
                let tool_names: Vec<&str> =
                    tool_uses.iter().map(|t| t.name.as_str()).collect();
                tracing::info!(
                    iteration,
                    model = %last_model_id,
                    client_tool_uses = tool_uses.len(),
                    server_tool_calls = server_calls,
                    tools = ?tool_names,
                    "agent: model requested tools"
                );

                // Reconstruct the assistant turn so the next step can see it.
                let mut assistant_blocks: Vec<TurnBlock> = Vec::new();
                if let Some(text) = partial_text {
                    if !text.is_empty() {
                        assistant_blocks.push(TurnBlock::Text(text));
                    }
                }
                for u in &tool_uses {
                    assistant_blocks.push(TurnBlock::ToolUse {
                        id: u.id.clone(),
                        name: u.name.clone(),
                        input: u.input.clone(),
                    });
                }
                messages.push(ChatTurn {
                    role: MessageRole::Assistant,
                    blocks: assistant_blocks,
                });

                // Execute each tool and build the user-side result turn.
                let mut result_blocks: Vec<TurnBlock> = Vec::with_capacity(tool_uses.len());
                for use_req in &tool_uses {
                    tracing::info!(
                        tool = %use_req.name,
                        input = %use_req.input,
                        "agent: invoking tool"
                    );
                    let (content_str, is_error, response_json) =
                        execute_one(executor, use_req).await;
                    tracing::info!(
                        tool = %use_req.name,
                        is_error,
                        response_chars = content_str.len(),
                        "agent: tool returned"
                    );
                    all_tool_calls.push(ToolCallRecord {
                        tool_name: use_req.name.clone(),
                        request: use_req.input.clone(),
                        response: response_json,
                    });
                    result_blocks.push(TurnBlock::ToolResult {
                        tool_use_id: use_req.id.clone(),
                        content: content_str,
                        is_error,
                    });
                }
                messages.push(ChatTurn {
                    role: MessageRole::User,
                    blocks: result_blocks,
                });
            }
        }
    }

    // Out of iterations — log the partial trace via the error.
    tracing::warn!(
        iterations = max_iterations,
        model = %last_model_id,
        tool_calls = all_tool_calls.len(),
        "agent loop hit iteration cap"
    );
    Err(LlmError::TooManyIterations(max_iterations))
}

/// Run one tool and turn its outcome into (model-facing string, is_error,
/// trace-side JSON).
async fn execute_one<T: ToolExecutor>(
    executor: &T,
    req: &ToolUseRequest,
) -> (String, bool, serde_json::Value) {
    match executor.execute(&req.name, req.input.clone()).await {
        Ok(value) => {
            let as_string =
                serde_json::to_string(&value).unwrap_or_else(|_| value.to_string());
            (as_string, false, value)
        }
        Err(err) => {
            let msg = err.to_string();
            let response = serde_json::json!({ "error": msg });
            (msg, true, response)
        }
    }
}
