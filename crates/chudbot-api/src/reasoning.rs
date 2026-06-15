//! Provider-neutral reasoning metadata extracted from opaque continuations.
//!
//! Continuations may contain replay-only provider state such as encrypted
//! reasoning. The types here expose only normalized summary text and token
//! counts that are safe for trace viewers.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::ids::{ModelId, ProviderName};
use crate::storage::ModelStepTrace;
use crate::transcript::ProviderContinuation;
use crate::usage::{UsageRecord, UsageSubject};

/// Viewer-safe reasoning metadata for one turn.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TurnReasoning {
    /// Reasoning summary items extracted from provider continuation payloads.
    pub items: Vec<ReasoningItem>,
    /// Aggregated reasoning token usage by provider/model.
    pub usage: Vec<ReasoningUsage>,
}

impl TurnReasoning {
    /// Extract reasoning metadata from ordered provider model steps and usage.
    pub fn from_model_steps_and_usage(
        model_steps: &[ModelStepTrace],
        usage: &[UsageRecord],
    ) -> Self {
        let items = model_steps
            .iter()
            .filter_map(|step| {
                step.continuation.as_ref().map(|continuation| {
                    reasoning_items_from_continuation(continuation, Some(&step.model))
                })
            })
            .flatten()
            .collect();
        let usage = reasoning_usage_from_records(usage);
        Self { items, usage }
    }

    /// Extract reasoning metadata from the stored continuation and usage.
    pub fn from_continuation_and_usage(
        continuation: Option<&ProviderContinuation>,
        model: Option<&ModelId>,
        usage: &[UsageRecord],
    ) -> Self {
        let items = continuation
            .map(|continuation| reasoning_items_from_continuation(continuation, model))
            .unwrap_or_default();
        let usage = reasoning_usage_from_records(usage);
        Self { items, usage }
    }

    /// Whether this turn has no viewer-safe reasoning metadata.
    pub fn is_empty(&self) -> bool {
        self.items.is_empty() && self.usage.is_empty()
    }
}

/// One provider reasoning item.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReasoningItem {
    /// Provider that emitted the reasoning item.
    pub provider: ProviderName,
    /// Model active for the turn, when known.
    pub model: Option<ModelId>,
    /// Provider item id, when present.
    pub id: Option<String>,
    /// Provider item status, when present.
    pub status: Option<String>,
    /// Summary text blocks extracted from the item.
    pub summary: Vec<ReasoningSummary>,
}

/// One normalized reasoning summary text block.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReasoningSummary {
    /// Provider-specific summary block kind, e.g. `summary_text`.
    pub kind: Option<String>,
    /// Summary text.
    pub text: String,
}

/// Aggregated reasoning token usage for a provider/model pair.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReasoningUsage {
    /// Provider that reported the token usage.
    pub provider: ProviderName,
    /// Model that reported the token usage, when known.
    pub model: Option<ModelId>,
    /// Sum of reported reasoning tokens for model steps in this turn.
    pub reasoning_tokens: u64,
}

fn reasoning_items_from_continuation(
    continuation: &ProviderContinuation,
    model: Option<&ModelId>,
) -> Vec<ReasoningItem> {
    match &continuation.data {
        Value::Array(items) => items
            .iter()
            .filter_map(|item| reasoning_item_from_value(&continuation.provider, model, item))
            .collect(),
        item => reasoning_item_from_value(&continuation.provider, model, item)
            .into_iter()
            .collect(),
    }
}

fn reasoning_item_from_value(
    provider: &ProviderName,
    model: Option<&ModelId>,
    item: &Value,
) -> Option<ReasoningItem> {
    let object = item.as_object()?;
    let item_type = object.get("type").and_then(Value::as_str);
    let summary = match item_type {
        Some("reasoning") => reasoning_summaries_from_value(object.get("summary")),
        Some("thinking") => anthropic_thinking_summaries(object),
        Some("redacted_thinking") => {
            vec![summary_text(
                Some("redacted_thinking"),
                "Thinking content redacted by provider.",
            )]
        }
        _ => Vec::new(),
    };
    let summary: Vec<_> = summary
        .into_iter()
        .filter(|summary| !summary.text.trim().is_empty())
        .collect();
    if summary.is_empty() {
        return None;
    }

    Some(ReasoningItem {
        provider: provider.clone(),
        model: model.cloned(),
        id: object.get("id").and_then(Value::as_str).map(str::to_string),
        status: object
            .get("status")
            .and_then(Value::as_str)
            .map(str::to_string),
        summary,
    })
}

fn anthropic_thinking_summaries(object: &serde_json::Map<String, Value>) -> Vec<ReasoningSummary> {
    if let Some(text) = object.get("thinking").and_then(Value::as_str)
        && !text.trim().is_empty()
    {
        return vec![summary_text(Some("thinking"), text)];
    }

    if object.contains_key("signature") {
        return vec![summary_text(
            Some("thinking_omitted"),
            "Thinking content omitted by provider.",
        )];
    }

    Vec::new()
}

fn reasoning_summaries_from_value(value: Option<&Value>) -> Vec<ReasoningSummary> {
    match value {
        Some(Value::Array(entries)) => entries
            .iter()
            .filter_map(reasoning_summary_from_entry)
            .collect(),
        Some(value) => reasoning_summary_from_entry(value).into_iter().collect(),
        None => Vec::new(),
    }
}

fn reasoning_summary_from_entry(entry: &Value) -> Option<ReasoningSummary> {
    match entry {
        Value::String(text) => Some(summary_text(None, text)),
        Value::Object(object) => object.get("text").and_then(Value::as_str).map(|text| {
            let kind = object.get("type").and_then(Value::as_str);
            summary_text(kind, text)
        }),
        _ => None,
    }
}

fn summary_text(kind: Option<&str>, text: &str) -> ReasoningSummary {
    ReasoningSummary {
        kind: kind.map(str::to_string),
        text: text.to_string(),
    }
}

fn reasoning_usage_from_records(records: &[UsageRecord]) -> Vec<ReasoningUsage> {
    let mut by_provider_model = BTreeMap::<(ProviderName, Option<ModelId>), u64>::new();
    for record in records {
        if !matches!(record.subject, UsageSubject::ModelStep) {
            continue;
        }
        let Some(tokens) = record.reasoning_tokens.filter(|tokens| *tokens > 0) else {
            continue;
        };
        let key = (record.provider.clone(), record.model.clone());
        *by_provider_model.entry(key).or_default() += tokens;
    }
    by_provider_model
        .into_iter()
        .map(|((provider, model), reasoning_tokens)| ReasoningUsage {
            provider,
            model,
            reasoning_tokens,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::usage::UsageRecord;

    #[test]
    fn extracts_reasoning_summary_without_encrypted_content() {
        let provider = ProviderName::new("xai");
        let model = ModelId::new("grok-4");
        let continuation = ProviderContinuation {
            provider: provider.clone(),
            data: json!([
                {
                    "id": "rs_123",
                    "type": "reasoning",
                    "status": "completed",
                    "summary": [
                        {
                            "type": "summary_text",
                            "text": "Checked the image result.\n"
                        }
                    ],
                    "encrypted_content": "SECRET_BLOB"
                },
                {
                    "id": "msg_123",
                    "type": "message",
                    "content": [{ "type": "output_text", "text": "done" }]
                }
            ]),
        };
        let reasoning =
            TurnReasoning::from_continuation_and_usage(Some(&continuation), Some(&model), &[]);

        assert_eq!(reasoning.items.len(), 1);
        let item = &reasoning.items[0];
        assert_eq!(item.provider, provider);
        assert_eq!(item.model.as_ref(), Some(&model));
        assert_eq!(item.id.as_deref(), Some("rs_123"));
        assert_eq!(item.status.as_deref(), Some("completed"));
        assert_eq!(item.summary[0].kind.as_deref(), Some("summary_text"));
        assert_eq!(item.summary[0].text, "Checked the image result.\n");

        let value = serde_json::to_value(&reasoning).expect("serialize reasoning");
        let serialized = value.to_string();
        assert!(!serialized.contains("encrypted_content"));
        assert!(!serialized.contains("SECRET_BLOB"));
    }

    #[test]
    fn aggregates_model_step_reasoning_usage_by_provider_and_model() {
        let provider = ProviderName::new("openai");
        let model = ModelId::new("o3");
        let mut first = UsageRecord::new(provider.clone(), UsageSubject::ModelStep);
        first.model = Some(model.clone());
        first.reasoning_tokens = Some(7);
        let mut second = UsageRecord::new(provider.clone(), UsageSubject::ModelStep);
        second.model = Some(model.clone());
        second.reasoning_tokens = Some(11);
        let mut media = UsageRecord::new(provider.clone(), UsageSubject::ImageGeneration);
        media.model = Some(model.clone());
        media.reasoning_tokens = Some(99);

        let reasoning =
            TurnReasoning::from_continuation_and_usage(None, Some(&model), &[first, second, media]);

        assert!(reasoning.items.is_empty());
        assert_eq!(
            reasoning.usage,
            vec![ReasoningUsage {
                provider,
                model: Some(model),
                reasoning_tokens: 18,
            }]
        );
    }

    #[test]
    fn extracts_anthropic_thinking_if_a_provider_continuation_supplies_it() {
        let provider = ProviderName::new("anthropic");
        let continuation = ProviderContinuation {
            provider: provider.clone(),
            data: json!({
                "type": "thinking",
                "thinking": "Considered the tradeoffs."
            }),
        };

        let reasoning = TurnReasoning::from_continuation_and_usage(Some(&continuation), None, &[]);

        assert_eq!(reasoning.items.len(), 1);
        assert_eq!(reasoning.items[0].provider, provider);
        assert_eq!(
            reasoning.items[0].summary[0].kind.as_deref(),
            Some("thinking")
        );
        assert_eq!(
            reasoning.items[0].summary[0].text,
            "Considered the tradeoffs."
        );
    }

    #[test]
    fn extracts_anthropic_omitted_and_redacted_thinking_markers() {
        let provider = ProviderName::new("anthropic");
        let continuation = ProviderContinuation {
            provider: provider.clone(),
            data: json!([
                {
                    "type": "thinking",
                    "thinking": "",
                    "signature": "OPAQUE"
                },
                {
                    "type": "redacted_thinking",
                    "data": "OPAQUE"
                }
            ]),
        };

        let reasoning = TurnReasoning::from_continuation_and_usage(Some(&continuation), None, &[]);

        assert_eq!(reasoning.items.len(), 2);
        assert_eq!(reasoning.items[0].provider, provider);
        assert_eq!(
            reasoning.items[0].summary[0].kind.as_deref(),
            Some("thinking_omitted")
        );
        assert_eq!(
            reasoning.items[0].summary[0].text,
            "Thinking content omitted by provider."
        );
        assert_eq!(
            reasoning.items[1].summary[0].kind.as_deref(),
            Some("redacted_thinking")
        );
        assert_eq!(
            reasoning.items[1].summary[0].text,
            "Thinking content redacted by provider."
        );
    }
}
