//! Grok API client.
//!
//! Defines a trait so call sites can be tested against a mock. The real
//! implementation talks to xAI's OpenAI-compatible chat completions
//! endpoint at `api.x.ai/v1`, with the built-in `web_search` tool enabled
//! for grounding "is this true" / "look up X" style queries.

use thiserror::Error;

/// Errors returned by `GrokClient` implementations.
#[derive(Debug, Error)]
pub enum GrokError {
    /// HTTP transport-level failure.
    #[error("transport error: {0}")]
    Transport(String),
    /// The API returned a non-success status.
    #[error("api error {status}: {body}")]
    Api {
        /// HTTP status code.
        status: u16,
        /// Response body (truncated by the caller as needed).
        body: String,
    },
    /// Response body could not be decoded.
    #[error("decode error: {0}")]
    Decode(String),
}

/// Abstracts away the xAI HTTP client so callers can be tested with a
/// mock implementation that returns canned responses.
pub trait GrokClient: Send + Sync {
    /// Send a single user prompt and get back an answer string.
    fn ask(
        &self,
        prompt: &str,
    ) -> impl std::future::Future<Output = Result<String, GrokError>> + Send;
}

/// Test-only mock that returns a fixed answer.
#[cfg(test)]
#[derive(Debug, Clone)]
pub struct MockGrok {
    /// Answer to return for every `ask` call.
    pub answer: String,
}

#[cfg(test)]
impl GrokClient for MockGrok {
    async fn ask(&self, _prompt: &str) -> Result<String, GrokError> {
        Ok(self.answer.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn mock_returns_canned_answer() {
        let mock = MockGrok {
            answer: "42".to_string(),
        };
        let answer = mock.ask("what is the meaning of life?").await.unwrap();
        assert_eq!(answer, "42");
    }
}
