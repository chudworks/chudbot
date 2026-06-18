//! Provider-neutral reasoning metadata for trace rendering.
//!
//! Provider continuations are stored as opaque JSON because each provider has a
//! different replay contract. Some of that JSON may include reasoning metadata,
//! replay-only signatures, or encrypted state. This module is the narrow
//! extraction boundary that converts the provider-owned payload into the small,
//! viewer-safe shape exposed by [`TurnReasoning`].
//!
//! The high-level flow is:
//!
//! 1. Read ordered [`ModelStepTrace`] records or a single
//!    [`ProviderContinuation`].
//! 2. Keep only recognized reasoning item shapes, such as Responses-style
//!    `reasoning` items or Anthropic `thinking` markers.
//! 3. Drop raw provider state and empty summaries before anything reaches the
//!    web trace API.
//! 4. Aggregate reasoning-token counts from model-step [`UsageRecord`] values.
//!
//! The resulting values are display metadata only. They are not fed back to
//! providers; replay still uses the original opaque continuation stored on the
//! transcript or model-step trace.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::ids::{ModelId, ProviderName};
use crate::storage::ModelStepTrace;
use crate::transcript::ProviderContinuation;
use crate::usage::{UsageRecord, UsageSubject};

/// Viewer-safe reasoning metadata for one completed turn.
///
/// A turn can have both summary items and token usage. Providers do not expose
/// those consistently: one backend may return summary text without token
/// accounting, while another may report reasoning tokens without any visible
/// summary. Callers should treat the two collections as independent display
/// channels.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TurnReasoning {
    /// Normalized summary items extracted from provider continuation payloads.
    ///
    /// These items intentionally exclude encrypted content, provider replay
    /// signatures, tool payloads, and non-reasoning message content.
    pub items: Vec<ReasoningItem>,
    /// Reasoning-token usage aggregated by provider/model.
    ///
    /// Usage rows are derived only from model-step usage records, so media,
    /// client-tool, server-tool, and sub-agent usage cannot appear here even if
    /// their raw records contain a reasoning-token field.
    pub usage: Vec<ReasoningUsage>,
}

/// One normalized provider reasoning item.
///
/// This is the item-level display shape consumed by the web trace viewer. It
/// keeps provider identity and lightweight provider metadata, but its summary
/// blocks are already sanitized and should not contain replay-only fields.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReasoningItem {
    /// Provider that emitted the reasoning item.
    pub provider: ProviderName,
    /// Model active for this model step, when known.
    ///
    /// Multi-step turns can contain items from more than one model if a future
    /// runtime mixes providers/models inside one attempt.
    pub model: Option<ModelId>,
    /// Provider item id, when present and safe to display.
    pub id: Option<String>,
    /// Provider item status, when present and safe to display.
    pub status: Option<String>,
    /// Summary text blocks extracted from the provider item.
    ///
    /// Empty or whitespace-only summaries are filtered before a
    /// [`ReasoningItem`] is built.
    pub summary: Vec<ReasoningSummary>,
}

/// One normalized reasoning summary text block.
///
/// Providers use different names for summary subtypes. The optional `kind`
/// preserves that lightweight label for display while keeping the text payload
/// provider-neutral.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReasoningSummary {
    /// Provider-specific summary block kind, e.g. `summary_text`.
    pub kind: Option<String>,
    /// Viewer-safe summary text.
    pub text: String,
}

/// Aggregated reasoning token usage for a provider/model pair.
///
/// This type is independent from [`ReasoningItem`] because some providers only
/// report token counts, and because token usage can come from accounting paths
/// that are separate from continuation extraction.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReasoningUsage {
    /// Provider that reported the token usage.
    pub provider: ProviderName,
    /// Model that reported the token usage, when known.
    pub model: Option<ModelId>,
    /// Sum of reported reasoning tokens for model steps in this turn.
    pub reasoning_tokens: u64,
}

impl TurnReasoning {
    /// Extract reasoning metadata from ordered provider model steps and usage.
    ///
    /// This is the trace-viewer path for a stored turn snapshot. Each model
    /// step can carry its own continuation and model id, which lets multi-step
    /// turns preserve the provider/model label attached to each reasoning item.
    pub fn from_model_steps_and_usage(
        model_steps: &[ModelStepTrace],
        usage: &[UsageRecord],
    ) -> Self {
        // FIXME(streaming): live `ModelStepItem::Reasoning` values are not
        // persisted directly yet. Stored reasoning views are reconstructed from
        // provider continuations, so provider adapters must keep terminal
        // continuations complete until typed reasoning has a durable storage path.
        // Step 1: walk continuations in model-step order so the viewer follows
        // the same sequence the runtime observed.
        let items = model_steps
            .iter()
            .filter_map(|step| {
                step.continuation.as_ref().map(|continuation| {
                    reasoning_items_from_continuation(continuation, Some(&step.model))
                })
            })
            .flatten()
            .collect();
        // Step 2: fold token accounting separately because usage records are
        // produced by accounting code, not by provider continuation parsing.
        let usage = reasoning_usage_from_records(usage);
        Self { items, usage }
    }

    /// Extract reasoning metadata from a single stored continuation and usage.
    ///
    /// This helper supports callers that have a final continuation rather than
    /// an ordered list of [`ModelStepTrace`] records. When a model id is
    /// supplied, every extracted item is labeled with that model.
    pub fn from_continuation_and_usage(
        continuation: Option<&ProviderContinuation>,
        model: Option<&ModelId>,
        usage: &[UsageRecord],
    ) -> Self {
        // Step 1: normalize the optional provider continuation into zero or
        // more viewer-safe reasoning items.
        let items = continuation
            .map(|continuation| reasoning_items_from_continuation(continuation, model))
            .unwrap_or_default();
        // Step 2: attach model-step reasoning-token usage, if any.
        let usage = reasoning_usage_from_records(usage);
        Self { items, usage }
    }

    /// Whether this turn has no viewer-safe reasoning metadata.
    pub fn is_empty(&self) -> bool {
        self.items.is_empty() && self.usage.is_empty()
    }
}

/// Extract reasoning items from the array-or-object continuation shapes used by
/// current providers.
fn reasoning_items_from_continuation(
    continuation: &ProviderContinuation,
    model: Option<&ModelId>,
) -> Vec<ReasoningItem> {
    match &continuation.data {
        // Responses-style providers store ordered output items as arrays.
        Value::Array(items) => items
            .iter()
            .filter_map(|item| reasoning_item_from_value(&continuation.provider, model, item))
            .collect(),
        // Chat-compatible/local providers and some future backends can emit a
        // single synthetic reasoning object.
        item => reasoning_item_from_value(&continuation.provider, model, item)
            .into_iter()
            .collect(),
    }
}

/// Convert one provider JSON object into a viewer-safe reasoning item.
///
/// Unknown item types deliberately return `None`; continuation payloads also
/// carry messages, tool calls, citations, signatures, and encrypted replay
/// state that are not reasoning summaries for the trace viewer.
fn reasoning_item_from_value(
    provider: &ProviderName,
    model: Option<&ModelId>,
    item: &Value,
) -> Option<ReasoningItem> {
    let object = item.as_object()?;
    let item_type = object.get("type").and_then(Value::as_str);
    // Step 1: classify only the provider item families that expose
    // viewer-safe reasoning text or a safe placeholder.
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
    // Step 2: never emit an item if all discovered summary text is empty.
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

/// Normalize Anthropic thinking blocks without exposing replay-only signatures.
fn anthropic_thinking_summaries(object: &serde_json::Map<String, Value>) -> Vec<ReasoningSummary> {
    // Prefer explicit thinking text when Anthropic returns it.
    if let Some(text) = object.get("thinking").and_then(Value::as_str)
        && !text.trim().is_empty()
    {
        return vec![summary_text(Some("thinking"), text)];
    }

    // A signature without text means the provider gave replay state but no
    // viewer-safe thought text. Surface that as a placeholder so the trace
    // still explains why a reasoning block exists.
    if object.contains_key("signature") {
        return vec![summary_text(
            Some("thinking_omitted"),
            "Thinking content omitted by provider.",
        )];
    }

    Vec::new()
}

/// Normalize a Responses-style `summary` field into summary blocks.
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

/// Convert one provider summary entry into a text block.
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

/// Build a normalized summary block from borrowed provider fields.
fn summary_text(kind: Option<&str>, text: &str) -> ReasoningSummary {
    ReasoningSummary {
        kind: kind.map(str::to_string),
        text: text.to_string(),
    }
}

/// Aggregate positive model-step reasoning-token counts by provider/model.
fn reasoning_usage_from_records(records: &[UsageRecord]) -> Vec<ReasoningUsage> {
    let mut by_provider_model = BTreeMap::<(&ProviderName, Option<&ModelId>), u64>::new();
    for record in records {
        // Only language-model steps belong in the reasoning panel. Other
        // subjects are displayed and billed elsewhere.
        if !matches!(record.subject, UsageSubject::ModelStep) {
            continue;
        }
        // Missing and zero counts are equivalent for display.
        let Some(tokens) = record.reasoning_tokens.filter(|tokens| *tokens > 0) else {
            continue;
        };
        let key = (&record.provider, record.model.as_ref());
        *by_provider_model.entry(key).or_default() += tokens;
    }
    by_provider_model
        .into_iter()
        .map(|((provider, model), reasoning_tokens)| ReasoningUsage {
            provider: provider.clone(),
            model: model.cloned(),
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
