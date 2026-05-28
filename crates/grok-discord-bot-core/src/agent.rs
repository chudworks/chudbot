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

use futures::stream::{FuturesUnordered, StreamExt};

use crate::llm::{
    ChatTurn, LlmProvider, MessageRole, ProviderOptions, StepRequest, StepResponse, ToolCallRecord,
    ToolDefinition, ToolExecutor, ToolUseRequest, TurnBlock,
};

/// Observes events during an agent run. Currently just intermediate
/// text the model emits alongside its tool_uses — the bot uses this
/// to post natural status messages as the model narrates, no
/// dedicated `post_status_message` tool required.
///
/// NOT called for the final assistant answer; that's returned in
/// [`AgentRun::content`] for the caller to dispatch as it sees fit.
pub trait AgentObserver: Send + Sync {
    /// Fired once per agent step that returned non-empty
    /// `partial_text` alongside one or more tool_uses. Implementations
    /// should be best-effort (log errors, don't propagate) — failing
    /// to post a status shouldn't abort the whole turn.
    fn on_partial_text(&self, text: &str) -> impl std::future::Future<Output = ()> + Send;
}

/// No-op observer for tests and callers that don't care about
/// intermediate model narration.
pub struct NoopObserver;

impl AgentObserver for NoopObserver {
    async fn on_partial_text(&self, _text: &str) {}
}

/// Result of [`run`]. The run always returns a snapshot of whatever
/// it accomplished — `tool_calls` may be non-empty even when `error`
/// is set, so callers can salvage successful media generation or
/// other side effects that happened before a transient step failure.
#[derive(Debug, Clone)]
pub struct AgentRun {
    /// Final answer text from the model. Empty when the run errored
    /// before the model produced a final response.
    pub content: String,
    /// All tool calls (server + client) performed during the run, in
    /// execution order. Populated even on error.
    pub tool_calls: Vec<ToolCallRecord>,
    /// Model id reported by the last step.
    pub model_id: String,
    /// Set when the loop terminated abnormally (provider step failure,
    /// iteration cap hit). Callers should check this and decide
    /// whether the partial trace is enough to act on.
    pub error: Option<String>,
    /// Opaque, provider-tagged continuation state for the FINAL assistant
    /// response — `{"provider": <name>, "data": <items>}` — for the
    /// caller to persist so later turns can replay it and keep the prompt
    /// cache warm. `None` when the provider produced no such state or the
    /// run errored before a final answer. See [`crate::llm::TurnBlock::Reasoning`].
    pub provider_state: Option<serde_json::Value>,
}

/// Drive the model through a tool-use loop until it produces a final
/// answer, or `max_iterations` is hit, or a transient provider error
/// aborts it. ALWAYS returns an [`AgentRun`] — `error` is populated
/// when the loop didn't reach a clean final answer. This lets callers
/// salvage successful tool calls that happened earlier in the loop
/// (e.g. media that already generated) even when a later LLM step
/// failed with a 5xx.
#[allow(clippy::too_many_arguments)]
pub async fn run<P, T, O>(
    provider: &P,
    model: String,
    initial_messages: Vec<ChatTurn>,
    tools: Vec<ToolDefinition>,
    executor: &T,
    observer: &O,
    enable_web_search: bool,
    max_tokens: u32,
    temperature: Option<f32>,
    top_p: Option<f32>,
    provider_options: ProviderOptions,
    cache_key: Option<String>,
    max_iterations: u32,
) -> AgentRun
where
    P: LlmProvider,
    T: ToolExecutor,
    O: AgentObserver,
{
    let mut messages = initial_messages;
    let mut all_tool_calls: Vec<ToolCallRecord> = Vec::new();
    let mut last_model_id = String::new();

    tracing::info!(
        provider = provider.name(),
        model = %model,
        messages = messages.len(),
        client_tools = tools.len(),
        web_search = enable_web_search,
        "agent: starting loop"
    );

    for iteration in 0..max_iterations {
        let response = match provider
            .step(StepRequest {
                model: model.clone(),
                messages: messages.clone(),
                tools: tools.clone(),
                enable_web_search,
                max_tokens,
                temperature,
                top_p,
                provider_options: provider_options.clone(),
                cache_key: cache_key.clone(),
            })
            .await
        {
            Ok(r) => r,
            Err(err) => {
                tracing::warn!(
                    iteration,
                    error = %err,
                    tool_calls_so_far = all_tool_calls.len(),
                    "agent: step failed; returning partial run"
                );
                return AgentRun {
                    content: String::new(),
                    tool_calls: all_tool_calls,
                    model_id: last_model_id,
                    error: Some(err.to_string()),
                    provider_state: None,
                };
            }
        };

        match response {
            StepResponse::Final {
                content,
                server_tool_calls,
                model_id,
                provider_state,
            } => {
                log_server_tool_calls(iteration, &server_tool_calls);
                let server_calls = server_tool_calls.len();
                all_tool_calls.extend(server_tool_calls);
                tracing::info!(
                    iteration,
                    model = %model_id,
                    text_chars = content.len(),
                    server_tool_calls = server_calls,
                    total_tool_calls = all_tool_calls.len(),
                    has_reasoning = provider_state.is_some(),
                    "agent: final answer received"
                );
                return AgentRun {
                    content,
                    tool_calls: all_tool_calls,
                    model_id,
                    error: None,
                    // Tag with the producing provider so cross-turn replay
                    // never feeds this back into a different provider.
                    provider_state: provider_state.map(
                        |data| serde_json::json!({ "provider": provider.name(), "data": data }),
                    ),
                };
            }
            StepResponse::UseTools {
                partial_text,
                tool_uses,
                server_tool_calls,
                model_id,
                provider_state,
            } => {
                log_server_tool_calls(iteration, &server_tool_calls);
                let server_calls = server_tool_calls.len();
                all_tool_calls.extend(server_tool_calls);
                last_model_id = model_id;
                let tool_names: Vec<&str> = tool_uses.iter().map(|t| t.name.as_str()).collect();
                tracing::info!(
                    iteration,
                    model = %last_model_id,
                    client_tool_uses = tool_uses.len(),
                    server_tool_calls = server_calls,
                    tools = ?tool_names,
                    has_partial_text = partial_text.is_some(),
                    "agent: model requested tools"
                );

                // Surface the model's intermediate narration before we
                // execute its tool calls. This is the natural,
                // post_status_message-free path.
                if let Some(text) = partial_text.as_ref()
                    && !text.trim().is_empty()
                {
                    observer.on_partial_text(text).await;
                }

                // Reconstruct the assistant turn so the next step can see it.
                // Reasoning leads the turn (it precedes the model's
                // text/tool_use in the response): replaying it verbatim is
                // what keeps reasoning models hitting the prompt cache and,
                // for some models, is required to continue a tool loop.
                let mut assistant_blocks: Vec<TurnBlock> = Vec::new();
                if let Some(data) = provider_state {
                    assistant_blocks.push(TurnBlock::Reasoning {
                        provider_name: provider.name().to_string(),
                        data,
                    });
                }
                if let Some(text) = partial_text
                    && !text.is_empty()
                {
                    assistant_blocks.push(TurnBlock::Text(text));
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

                // Execute every tool the model asked for CONCURRENTLY.
                // The tool-use protocol (both Anthropic and xAI) requires
                // all `tool_result` blocks answering this assistant turn to
                // come back together in the NEXT user message — there's no
                // way to drip them in one at a time — so we fan the calls
                // out with unordered parallelism, await every result, then
                // assemble a single result turn and make one follow-up
                // request. Each future logs and is supervised independently
                // (`execute_one` turns a tool failure into an `is_error`
                // result rather than aborting its siblings), so one slow or
                // failing tool can't block or sink the others.
                //
                // Completions land in arbitrary order, but we slot each into
                // its declared position so the persisted trace and the
                // `tool_result` ordering stay deterministic regardless of
                // which tool finished first.
                let mut slots: Vec<Option<(ToolCallRecord, TurnBlock)>> =
                    (0..tool_uses.len()).map(|_| None).collect();
                let mut pending: FuturesUnordered<_> = tool_uses
                    .iter()
                    .enumerate()
                    .map(|(idx, use_req)| async move {
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
                            response = %truncate_for_log(&response_json, 600),
                            "agent: tool returned"
                        );
                        (idx, use_req, content_str, is_error, response_json)
                    })
                    .collect();
                while let Some((idx, use_req, content_str, is_error, response_json)) =
                    pending.next().await
                {
                    slots[idx] = Some((
                        ToolCallRecord {
                            tool_name: use_req.name.clone(),
                            request: use_req.input.clone(),
                            response: response_json,
                        },
                        TurnBlock::ToolResult {
                            tool_use_id: use_req.id.clone(),
                            content: content_str,
                            is_error,
                        },
                    ));
                }

                let mut result_blocks: Vec<TurnBlock> = Vec::with_capacity(tool_uses.len());
                for (record, block) in slots.into_iter().flatten() {
                    all_tool_calls.push(record);
                    result_blocks.push(block);
                }
                messages.push(ChatTurn {
                    role: MessageRole::User,
                    blocks: result_blocks,
                });
            }
        }
    }

    // Out of iterations — return what we accumulated.
    tracing::warn!(
        iterations = max_iterations,
        model = %last_model_id,
        tool_calls = all_tool_calls.len(),
        "agent loop hit iteration cap"
    );
    AgentRun {
        content: String::new(),
        tool_calls: all_tool_calls,
        model_id: last_model_id,
        error: Some(format!("hit iteration cap ({max_iterations})")),
        provider_state: None,
    }
}

/// Emit one info line per server-side tool call (web_search, x_search,
/// code_interpreter, …). The provider hands these back fully resolved
/// after each step; surfacing them individually makes it easy to see
/// what grounding actually fired from `tail -f` without opening the
/// viewer.
fn log_server_tool_calls(iteration: u32, calls: &[ToolCallRecord]) {
    for call in calls {
        tracing::info!(
            iteration,
            tool = %call.tool_name,
            request = %truncate_for_log(&call.request, 400),
            response = %truncate_for_log(&call.response, 600),
            "agent: server tool call"
        );
    }
}

/// Compact a JSON value for log display: serialize then trim with an
/// ellipsis at the byte boundary. Not character-aware — log fields
/// don't need to be UTF-8-perfect.
fn truncate_for_log(value: &serde_json::Value, max: usize) -> String {
    let s = serde_json::to_string(value).unwrap_or_else(|_| value.to_string());
    if s.len() <= max {
        s
    } else {
        let mut cutoff = max.saturating_sub(1);
        while cutoff > 0 && !s.is_char_boundary(cutoff) {
            cutoff -= 1;
        }
        format!("{}…", &s[..cutoff])
    }
}

/// Run one tool and turn its outcome into (model-facing string, is_error,
/// trace-side JSON).
async fn execute_one<T: ToolExecutor>(
    executor: &T,
    req: &ToolUseRequest,
) -> (String, bool, serde_json::Value) {
    match executor.execute(&req.name, req.input.clone()).await {
        Ok(value) => {
            let as_string = serde_json::to_string(&value).unwrap_or_else(|_| value.to_string());
            (as_string, false, value)
        }
        Err(err) => {
            let msg = err.to_string();
            let response = serde_json::json!({ "error": msg });
            (msg, true, response)
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::Mutex;
    use std::time::Duration;

    use serde_json::json;
    use tokio::sync::Barrier;

    use super::*;
    use crate::llm::ToolError;
    use crate::llm::mock::MockProvider;

    fn tool_use(id: &str, name: &str) -> ToolUseRequest {
        ToolUseRequest {
            id: id.to_string(),
            name: name.to_string(),
            input: json!({}),
        }
    }

    /// Drive `run` over a single `UseTools` step that names the given
    /// tools, followed by a final answer. Returns the resulting run.
    async fn run_with_tools<T: ToolExecutor>(tool_uses: Vec<ToolUseRequest>, executor: &T) -> AgentRun {
        let provider = MockProvider {
            name: "mock".to_string(),
            answer: "done".to_string(),
            server_tool_calls: Vec::new(),
            script: Mutex::new(vec![StepResponse::UseTools {
                partial_text: None,
                tool_uses,
                server_tool_calls: Vec::new(),
                model_id: "mock".to_string(),
                provider_state: None,
            }]),
        };
        run(
            &provider,
            "mock".to_string(),
            vec![ChatTurn::text(MessageRole::User, "hi")],
            Vec::new(),
            executor,
            &NoopObserver,
            false,
            1024,
            None,
            None,
            ProviderOptions::default(),
            None,
            6,
        )
        .await
    }

    /// Executor that blocks every call on a shared 2-party barrier, so
    /// the run can only finish if the two tools are in flight at the same
    /// time — sequential execution would wait on the barrier forever. To
    /// prove the trace stays in declared order regardless of completion
    /// order, the *first*-declared tool sleeps after the barrier so it
    /// finishes *last*.
    struct BarrierExecutor {
        barrier: Arc<Barrier>,
    }

    impl ToolExecutor for BarrierExecutor {
        async fn execute(&self, name: &str, _input: serde_json::Value) -> Result<serde_json::Value, ToolError> {
            self.barrier.wait().await;
            if name == "tool_a" {
                tokio::time::sleep(Duration::from_millis(40)).await;
            }
            Ok(json!({ "ran": name }))
        }
    }

    #[tokio::test]
    async fn executes_tool_calls_concurrently_and_preserves_declared_order() {
        let executor = BarrierExecutor {
            barrier: Arc::new(Barrier::new(2)),
        };
        // If the loop ran the tools sequentially the barrier would never
        // release; the timeout converts that hang into a clear failure.
        let run = tokio::time::timeout(
            Duration::from_secs(5),
            run_with_tools(vec![tool_use("1", "tool_a"), tool_use("2", "tool_b")], &executor),
        )
        .await
        .expect("agent run deadlocked — tools were not executed concurrently");

        assert!(run.error.is_none());
        assert_eq!(run.content, "done");
        let names: Vec<&str> = run.tool_calls.iter().map(|c| c.tool_name.as_str()).collect();
        // tool_b completes first (tool_a sleeps post-barrier) yet the
        // declared order is preserved in the persisted trace.
        assert_eq!(names, ["tool_a", "tool_b"]);
    }

    /// Executor that fails one named tool and succeeds the rest.
    struct FlakyExecutor;

    impl ToolExecutor for FlakyExecutor {
        async fn execute(&self, name: &str, _input: serde_json::Value) -> Result<serde_json::Value, ToolError> {
            if name == "bad" {
                Err(ToolError::Execution("boom".to_string()))
            } else {
                Ok(json!({ "ran": name }))
            }
        }
    }

    #[tokio::test]
    async fn one_tool_failure_does_not_sink_its_siblings() {
        let run = run_with_tools(
            vec![tool_use("1", "good"), tool_use("2", "bad"), tool_use("3", "good2")],
            &FlakyExecutor,
        )
        .await;

        // The whole run still reaches a clean final answer — a single tool
        // failure is reported back to the model, not propagated.
        assert!(run.error.is_none());
        assert_eq!(run.content, "done");
        let names: Vec<&str> = run.tool_calls.iter().map(|c| c.tool_name.as_str()).collect();
        assert_eq!(names, ["good", "bad", "good2"]);
        // The failed tool's record carries the error payload; the others
        // carry their results.
        assert_eq!(run.tool_calls[1].response, json!({ "error": "execution failed: boom" }));
        assert_eq!(run.tool_calls[0].response, json!({ "ran": "good" }));
        assert_eq!(run.tool_calls[2].response, json!({ "ran": "good2" }));
    }
}
