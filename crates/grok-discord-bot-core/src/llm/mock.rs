//! Test-only mock provider. Returns a canned final response (or a
//! scripted sequence of step responses) without making network calls.

use std::sync::Mutex;

use crate::llm::{LlmError, LlmProvider, StepRequest, StepResponse, ToolCallRecord};

/// Mock that returns a fixed final answer on every step. Sufficient
/// for testing the bot's plumbing without exercising the agent loop.
#[derive(Debug, Default)]
pub struct MockProvider {
    /// Name reported by [`LlmProvider::name`].
    pub name: String,
    /// Answer text returned for every step.
    pub answer: String,
    /// Server-side tool call trace returned alongside the answer.
    pub server_tool_calls: Vec<ToolCallRecord>,
    /// Optional scripted sequence of step responses. When set, the
    /// next step pops the front; when the script is empty, falls back
    /// to a `Final` response with `answer`.
    pub script: Mutex<Vec<StepResponse>>,
}

impl MockProvider {
    /// Build a mock that answers every prompt with the given string.
    pub fn with_answer(answer: impl Into<String>) -> Self {
        Self {
            name: "mock".to_string(),
            answer: answer.into(),
            server_tool_calls: Vec::new(),
            script: Mutex::new(Vec::new()),
        }
    }
}

impl LlmProvider for MockProvider {
    fn name(&self) -> &str {
        &self.name
    }

    async fn step(&self, _request: StepRequest) -> Result<StepResponse, LlmError> {
        if let Ok(mut script) = self.script.lock() {
            if !script.is_empty() {
                return Ok(script.remove(0));
            }
        }
        Ok(StepResponse::Final {
            content: self.answer.clone(),
            server_tool_calls: self.server_tool_calls.clone(),
            model_id: "mock".to_string(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::{ChatTurn, MessageRole};

    #[tokio::test]
    async fn mock_returns_canned_final() {
        let p = MockProvider::with_answer("42");
        let resp = p
            .step(StepRequest {
                messages: vec![ChatTurn::text(MessageRole::User, "hi")],
                tools: Vec::new(),
                enable_web_search: false,
                max_tokens: 1024,
                temperature: None,
                top_p: None,
            })
            .await
            .unwrap();
        match resp {
            StepResponse::Final { content, .. } => assert_eq!(content, "42"),
            _ => panic!("expected Final"),
        }
    }
}
