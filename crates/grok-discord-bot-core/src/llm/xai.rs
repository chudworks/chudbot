//! xAI Grok provider. Talks to the OpenAI-compatible `/v1/chat/completions`
//! endpoint at `api.x.ai`, with the server-side web search tool enabled
//! via `search_parameters` when [`CompletionRequest::enable_web_search`]
//! is set.

use serde::{Deserialize, Serialize};

use crate::config::XaiConfig;
use crate::llm::{
    CompletionRequest, CompletionResponse, LlmError, LlmProvider, ToolCallRecord,
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
    /// Construct from a config block. Uses the default `api.x.ai/v1`
    /// base URL.
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

    /// Override the base URL. Used by tests to point at a wiremock server.
    pub fn with_base_url(mut self, base_url: String) -> Self {
        self.base_url = base_url;
        self
    }
}

impl LlmProvider for XaiProvider {
    fn name(&self) -> &str {
        &self.name
    }

    async fn complete(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse, LlmError> {
        let messages: Vec<XaiMessage<'_>> = request
            .messages
            .iter()
            .map(|m| XaiMessage {
                role: m.role.as_str(),
                content: m.content.as_str(),
            })
            .collect();

        let body = XaiRequest {
            model: &self.model,
            messages: &messages,
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

        let content = parsed
            .choices
            .into_iter()
            .next()
            .map(|c| c.message.content)
            .unwrap_or_default();

        let mut tool_calls = Vec::new();
        if let Some(citations) = parsed.citations {
            tool_calls.push(ToolCallRecord {
                tool_name: "web_search".to_string(),
                request: serde_json::json!({ "mode": "on", "model_query": "(server-side)" }),
                response: citations,
            });
        }

        Ok(CompletionResponse {
            content,
            tool_calls,
            model_id: parsed.model.unwrap_or_else(|| self.model.clone()),
        })
    }
}

#[derive(Serialize)]
struct XaiRequest<'a> {
    model: &'a str,
    messages: &'a [XaiMessage<'a>],
    #[serde(skip_serializing_if = "Option::is_none")]
    search_parameters: Option<XaiSearchParameters>,
}

#[derive(Serialize)]
struct XaiMessage<'a> {
    role: &'a str,
    content: &'a str,
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
    citations: Option<serde_json::Value>,
    #[serde(default)]
    model: Option<String>,
}

#[derive(Deserialize)]
struct XaiChoice {
    message: XaiResponseMessage,
}

#[derive(Deserialize)]
struct XaiResponseMessage {
    #[serde(default)]
    content: String,
}
