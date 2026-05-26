//! Test-only mock provider. Returns canned responses without making any
//! network calls. Used by tests of the bot/web layers.

use crate::llm::{CompletionRequest, CompletionResponse, LlmError, LlmProvider, ToolCallRecord};

/// Returns a fixed answer plus a configurable list of tool calls.
#[derive(Debug, Clone, Default)]
pub struct MockProvider {
    /// Name reported by [`LlmProvider::name`].
    pub name: String,
    /// Answer text returned for every request.
    pub answer: String,
    /// Tool call trace returned alongside the answer.
    pub tool_calls: Vec<ToolCallRecord>,
}

impl MockProvider {
    /// Build a mock that answers every prompt with the given string.
    pub fn with_answer(answer: impl Into<String>) -> Self {
        Self {
            name: "mock".to_string(),
            answer: answer.into(),
            tool_calls: Vec::new(),
        }
    }
}

impl LlmProvider for MockProvider {
    fn name(&self) -> &str {
        &self.name
    }

    async fn complete(
        &self,
        _request: CompletionRequest,
    ) -> Result<CompletionResponse, LlmError> {
        Ok(CompletionResponse {
            content: self.answer.clone(),
            tool_calls: self.tool_calls.clone(),
            model_id: "mock".to_string(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::ChatMessage;

    #[tokio::test]
    async fn mock_returns_canned_answer() {
        let p = MockProvider::with_answer("42");
        let resp = p
            .complete(CompletionRequest {
                messages: vec![ChatMessage {
                    role: crate::llm::MessageRole::User,
                    content: "what is the meaning of life?".into(),
                }],
                enable_web_search: false,
                max_tokens: 1024,
            })
            .await
            .unwrap();
        assert_eq!(resp.content, "42");
    }
}
