//! Model-step stream reducer.
//!
//! The public LLM contract lives in `llm.rs`; this module owns the mechanics of
//! folding provider events into the stable collected `ModelStep` shape.

use std::collections::BTreeMap;

use futures::{Stream, StreamExt};
use thiserror::Error;

use crate::ids::{ModelId, ProviderName, ToolName};
use crate::reasoning::{ReasoningItem, ReasoningSummary};
use crate::storage::ModelStepKind;
use crate::tool::ClientToolCall;
use crate::usage::UsageRecord;

use crate::llm::{
    ModelOutputBlock, ModelStep, ModelStepDelta, ModelStepEvent, ModelStepItem, ModelStepOutput,
};

/// Error returned by the async model-step reducer.
#[derive(Debug, Error)]
pub enum ModelStepCollectionError<E>
where
    E: std::error::Error + Send + Sync + 'static,
{
    /// Provider stream failed.
    #[error("model step stream failed: {0}")]
    Stream(#[source] E),
    /// Provider stream violated the event protocol.
    #[error("{0}")]
    Reducer(#[from] ModelStepReducerError),
}

/// Error returned when reducing model-step events into a collected step.
#[derive(Debug, Error)]
pub enum ModelStepReducerError {
    /// The stream ended before a terminal event.
    #[error("model step stream ended without a finished event")]
    MissingFinished,
    /// The stream emitted more than one terminal event.
    #[error("model step stream emitted more than one finished event")]
    DuplicateFinished,
    /// A provider emitted non-terminal output after finishing the step.
    #[error("model step stream emitted output after the finished event")]
    EventAfterFinished,
    /// A stable item id was reused for a different delta kind.
    #[error("model step stream reused item id `{id}` for inconsistent delta kinds")]
    InconsistentItemId {
        /// Reused item id.
        id: String,
    },
    /// Reasoning deltas for one item disagreed about their provider.
    #[error("reasoning summary item `{item_id}` changed provider from `{first}` to `{second}`")]
    InconsistentReasoningProvider {
        /// Item id.
        item_id: String,
        /// First provider.
        first: ProviderName,
        /// Second provider.
        second: ProviderName,
    },
    /// Client-tool deltas for one item disagreed about their tool-use id.
    #[error("client tool-call item `{item_id}` changed tool-use id from `{first}` to `{second}`")]
    InconsistentToolUseId {
        /// Item id.
        item_id: String,
        /// First tool-use id.
        first: crate::ids::ToolUseId,
        /// Second tool-use id.
        second: crate::ids::ToolUseId,
    },
    /// Client-tool deltas for one item disagreed about their tool name.
    #[error("client tool-call item `{item_id}` changed tool name from `{first}` to `{second}`")]
    InconsistentToolName {
        /// Item id.
        item_id: String,
        /// First tool name.
        first: ToolName,
        /// Second tool name.
        second: ToolName,
    },
    /// A tool call finished without a tool name.
    #[error("client tool-call item `{item_id}` finished without a tool name")]
    MissingToolName {
        /// Item id.
        item_id: String,
    },
    /// A tool call's accumulated JSON arguments did not parse.
    #[error("client tool-call `{tool_use_id}` arguments were not valid JSON: {source}")]
    InvalidToolArguments {
        /// Tool-use id.
        tool_use_id: crate::ids::ToolUseId,
        /// Raw accumulated argument string.
        arguments: String,
        /// JSON parser error.
        #[source]
        source: serde_json::Error,
    },
}

/// Collect one streamed provider step into a stable model-step output.
pub async fn collect_model_step<S, E>(events: S) -> Result<ModelStep, ModelStepCollectionError<E>>
where
    S: Stream<Item = Result<ModelStepEvent, E>> + Send,
    E: std::error::Error + Send + Sync + 'static,
{
    futures::pin_mut!(events);
    let mut collector = ModelStepCollector::default();
    while let Some(event) = events.next().await {
        collector.push(event.map_err(ModelStepCollectionError::Stream)?)?;
    }
    collector
        .finish()
        .map_err(ModelStepCollectionError::Reducer)
}

/// Incremental reducer for one streamed model step.
#[derive(Debug, Default)]
pub(crate) struct ModelStepCollector {
    slots: Vec<OutputSlot>,
    ids: BTreeMap<String, PendingKind>,
    finished: Option<(ModelStepKind, ModelId)>,
}

impl ModelStepCollector {
    /// Push one event into the reducer.
    pub(crate) fn push(&mut self, event: ModelStepEvent) -> Result<(), ModelStepReducerError> {
        if self.finished.is_some() && !matches!(event, ModelStepEvent::Finished { .. }) {
            return Err(ModelStepReducerError::EventAfterFinished);
        }

        match event {
            ModelStepEvent::Delta(delta) => self.push_delta(delta),
            ModelStepEvent::Continuation(continuation) => {
                self.slots
                    .push(OutputSlot::Collected(ModelStepItem::OutputBlock(
                        ModelOutputBlock::Continuation(continuation),
                    )));
                Ok(())
            }
            ModelStepEvent::ServerToolUse(tool) => {
                self.slots
                    .push(OutputSlot::Collected(ModelStepItem::ServerToolUse(tool)));
                Ok(())
            }
            ModelStepEvent::Grounding(metadata) => {
                self.slots
                    .push(OutputSlot::Collected(ModelStepItem::Grounding(metadata)));
                Ok(())
            }
            ModelStepEvent::Usage(usage) => {
                self.slots.push(OutputSlot::Usage(usage));
                Ok(())
            }
            ModelStepEvent::Finished { kind, model_id } => {
                if self.finished.replace((kind, model_id)).is_some() {
                    return Err(ModelStepReducerError::DuplicateFinished);
                }
                Ok(())
            }
        }
    }

    /// Finish collection and return the collected model step.
    pub(crate) fn finish(self) -> Result<ModelStep, ModelStepReducerError> {
        let Some((kind, model_id)) = self.finished else {
            return Err(ModelStepReducerError::MissingFinished);
        };
        let mut output = ModelStepOutput::new(model_id.clone());
        for slot in self.slots {
            match slot {
                OutputSlot::Collected(item) => output.items.push(item),
                OutputSlot::Usage(usage) => output.usage.push(usage),
                OutputSlot::Pending(pending) => {
                    output.items.push(pending.finish(&model_id)?);
                }
            }
        }
        Ok(ModelStep::new(kind, output))
    }

    fn push_delta(&mut self, delta: ModelStepDelta) -> Result<(), ModelStepReducerError> {
        match delta {
            ModelStepDelta::Text { item_id, delta } => {
                let slot = self.pending_slot(&item_id, PendingKind::Text)?;
                match slot {
                    PendingItem::Text { text, .. } => text.push_str(&delta),
                    PendingItem::Reasoning { .. } | PendingItem::ClientToolCall { .. } => {
                        return Err(ModelStepReducerError::InconsistentItemId { id: item_id });
                    }
                }
                Ok(())
            }
            ModelStepDelta::ReasoningSummary {
                item_id,
                provider,
                kind,
                delta,
            } => {
                let slot = self.pending_slot(&item_id, PendingKind::Reasoning)?;
                match slot {
                    PendingItem::Reasoning {
                        item_id,
                        provider: existing_provider,
                        summaries,
                    } => {
                        if existing_provider.as_str().is_empty() {
                            *existing_provider = provider.clone();
                        } else if existing_provider != &provider {
                            return Err(ModelStepReducerError::InconsistentReasoningProvider {
                                item_id: item_id.clone(),
                                first: existing_provider.clone(),
                                second: provider,
                            });
                        }
                        if let Some(summary) = summaries.last_mut()
                            && summary.kind == kind
                        {
                            summary.text.push_str(&delta);
                            return Ok(());
                        }
                        summaries.push(PendingReasoningSummary { kind, text: delta });
                    }
                    PendingItem::Text { .. } | PendingItem::ClientToolCall { .. } => {
                        return Err(ModelStepReducerError::InconsistentItemId { id: item_id });
                    }
                }
                Ok(())
            }
            ModelStepDelta::ClientToolCall {
                item_id,
                id,
                name,
                arguments_delta,
            } => {
                let slot = self.pending_slot(&item_id, PendingKind::ClientToolCall)?;
                match slot {
                    PendingItem::ClientToolCall {
                        item_id,
                        id: existing_id,
                        name: existing_name,
                        arguments,
                    } => {
                        if existing_id.as_str().is_empty() {
                            *existing_id = id.clone();
                        } else if existing_id != &id {
                            return Err(ModelStepReducerError::InconsistentToolUseId {
                                item_id: item_id.clone(),
                                first: existing_id.clone(),
                                second: id,
                            });
                        }
                        if let Some(name) = name {
                            match existing_name {
                                Some(previous) if previous != &name => {
                                    return Err(ModelStepReducerError::InconsistentToolName {
                                        item_id: item_id.clone(),
                                        first: previous.clone(),
                                        second: name,
                                    });
                                }
                                Some(_) => {}
                                None => *existing_name = Some(name),
                            }
                        }
                        arguments.push_str(&arguments_delta);
                    }
                    PendingItem::Text { .. } | PendingItem::Reasoning { .. } => {
                        return Err(ModelStepReducerError::InconsistentItemId { id: item_id });
                    }
                }
                Ok(())
            }
        }
    }

    fn pending_slot(
        &mut self,
        key: &str,
        kind: PendingKind,
    ) -> Result<&mut PendingItem, ModelStepReducerError> {
        if let Some(existing_kind) = self.ids.get(key) {
            if *existing_kind != kind {
                return Err(ModelStepReducerError::InconsistentItemId {
                    id: key.to_string(),
                });
            }
            let slot = self
                .slots
                .iter_mut()
                .find_map(|slot| match slot {
                    OutputSlot::Pending(pending) if pending.item_id() == key => Some(pending),
                    OutputSlot::Collected(_) | OutputSlot::Usage(_) | OutputSlot::Pending(_) => {
                        None
                    }
                })
                .expect("pending id map points at a pending slot");
            return Ok(slot);
        }

        self.ids.insert(key.to_string(), kind);
        self.slots
            .push(OutputSlot::Pending(PendingItem::new(key.to_string(), kind)));
        match self.slots.last_mut().expect("just pushed pending slot") {
            OutputSlot::Pending(pending) => Ok(pending),
            OutputSlot::Collected(_) | OutputSlot::Usage(_) => unreachable!("pushed pending slot"),
        }
    }
}

#[derive(Debug)]
enum OutputSlot {
    Collected(ModelStepItem),
    Usage(UsageRecord),
    Pending(PendingItem),
}

#[derive(Debug)]
enum PendingItem {
    Text {
        item_id: String,
        text: String,
    },
    Reasoning {
        item_id: String,
        provider: ProviderName,
        summaries: Vec<PendingReasoningSummary>,
    },
    ClientToolCall {
        item_id: String,
        id: crate::ids::ToolUseId,
        name: Option<ToolName>,
        arguments: String,
    },
}

impl PendingItem {
    fn new(item_id: String, kind: PendingKind) -> Self {
        match kind {
            PendingKind::Text => Self::Text {
                item_id,
                text: String::new(),
            },
            PendingKind::Reasoning => Self::Reasoning {
                item_id,
                provider: ProviderName::new(""),
                summaries: Vec::new(),
            },
            PendingKind::ClientToolCall => Self::ClientToolCall {
                item_id,
                id: crate::ids::ToolUseId::new(""),
                name: None,
                arguments: String::new(),
            },
        }
    }

    fn item_id(&self) -> &str {
        match self {
            Self::Text { item_id, .. }
            | Self::Reasoning { item_id, .. }
            | Self::ClientToolCall { item_id, .. } => item_id,
        }
    }

    fn finish(self, model_id: &ModelId) -> Result<ModelStepItem, ModelStepReducerError> {
        match self {
            Self::Text { text, .. } => {
                Ok(ModelStepItem::OutputBlock(ModelOutputBlock::Text { text }))
            }
            Self::Reasoning {
                item_id,
                provider,
                summaries,
            } => Ok(ModelStepItem::Reasoning(ReasoningItem {
                provider,
                model: Some(model_id.clone()),
                id: Some(item_id),
                status: None,
                summary: summaries
                    .into_iter()
                    .map(|summary| ReasoningSummary {
                        kind: summary.kind,
                        text: summary.text,
                    })
                    .collect(),
            })),
            Self::ClientToolCall {
                item_id,
                id,
                name,
                arguments,
            } => {
                let Some(name) = name else {
                    return Err(ModelStepReducerError::MissingToolName { item_id });
                };
                let input = if arguments.trim().is_empty() {
                    serde_json::Value::Object(Default::default())
                } else {
                    serde_json::from_str(&arguments).map_err(|source| {
                        ModelStepReducerError::InvalidToolArguments {
                            tool_use_id: id.clone(),
                            arguments: arguments.clone(),
                            source,
                        }
                    })?
                };
                Ok(ModelStepItem::OutputBlock(
                    ModelOutputBlock::ClientToolCall(ClientToolCall { id, name, input }),
                ))
            }
        }
    }
}

#[derive(Debug)]
struct PendingReasoningSummary {
    kind: Option<String>,
    text: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PendingKind {
    Text,
    Reasoning,
    ClientToolCall,
}

#[cfg(test)]
mod tests {
    use futures::stream;
    use serde_json::json;

    use super::*;
    use crate::ids::{ToolName, ToolUseId};
    use crate::storage::ModelStepKind;
    use crate::transcript::ProviderContinuation;
    use crate::usage::{UsageRecord, UsageSubject};

    #[derive(Debug)]
    struct TestError;

    impl std::fmt::Display for TestError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("test error")
        }
    }

    impl std::error::Error for TestError {}

    fn ok(event: ModelStepEvent) -> Result<ModelStepEvent, TestError> {
        Ok(event)
    }

    #[tokio::test]
    async fn collect_text_deltas_into_ordered_output() {
        let step = collect_model_step(stream::iter([
            ok(ModelStepEvent::Delta(ModelStepDelta::Text {
                item_id: "msg-1".to_string(),
                delta: "hello".to_string(),
            })),
            ok(ModelStepEvent::Delta(ModelStepDelta::Text {
                item_id: "msg-1".to_string(),
                delta: " world".to_string(),
            })),
            ok(ModelStepEvent::Finished {
                kind: ModelStepKind::Final,
                model_id: ModelId::new("test-model"),
            }),
        ]))
        .await
        .unwrap();

        assert_eq!(step.kind, ModelStepKind::Final);
        let output = step.output;
        assert_eq!(output.answer_text(), "hello world");
        assert_eq!(output.model_id.as_str(), "test-model");
    }

    #[tokio::test]
    async fn collect_client_tool_call_deltas_into_json_input() {
        let step = collect_model_step(stream::iter([
            ok(ModelStepEvent::Delta(ModelStepDelta::ClientToolCall {
                item_id: "tool-1".to_string(),
                id: ToolUseId::new("call-1"),
                name: Some(ToolName::new("lookup")),
                arguments_delta: "{\"query\":".to_string(),
            })),
            ok(ModelStepEvent::Delta(ModelStepDelta::ClientToolCall {
                item_id: "tool-1".to_string(),
                id: ToolUseId::new("call-1"),
                name: None,
                arguments_delta: "\"rust\"}".to_string(),
            })),
            ok(ModelStepEvent::Finished {
                kind: ModelStepKind::ClientTools,
                model_id: ModelId::new("test-model"),
            }),
        ]))
        .await
        .unwrap();

        assert_eq!(step.kind, ModelStepKind::ClientTools);
        let output = step.output;
        let calls: Vec<_> = output.client_tool_calls().collect();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id.as_str(), "call-1");
        assert_eq!(calls[0].name.as_str(), "lookup");
        assert_eq!(calls[0].input, json!({ "query": "rust" }));
    }

    #[tokio::test]
    async fn reject_malformed_final_tool_call_json() {
        let error = collect_model_step(stream::iter([
            ok(ModelStepEvent::Delta(ModelStepDelta::ClientToolCall {
                item_id: "tool-1".to_string(),
                id: ToolUseId::new("call-1"),
                name: Some(ToolName::new("lookup")),
                arguments_delta: "{\"query\":".to_string(),
            })),
            ok(ModelStepEvent::Finished {
                kind: ModelStepKind::ClientTools,
                model_id: ModelId::new("test-model"),
            }),
        ]))
        .await
        .unwrap_err();

        assert!(matches!(
            error,
            ModelStepCollectionError::Reducer(ModelStepReducerError::InvalidToolArguments { .. })
        ));
    }

    #[tokio::test]
    async fn metadata_usage_and_finished_events_roundtrip_collected_step() {
        let continuation = ProviderContinuation {
            provider: ProviderName::new("test-provider"),
            data: json!({ "cursor": "abc" }),
        };
        let usage = UsageRecord {
            provider: ProviderName::new("test-provider"),
            model: Some(ModelId::new("test-model")),
            subject: UsageSubject::ModelStep,
            input_tokens: Some(10),
            cached_input_tokens: None,
            output_tokens: Some(5),
            reasoning_tokens: None,
            total_tokens: Some(15),
            cost: None,
            raw: None,
        };
        let step = collect_model_step(stream::iter([
            ok(ModelStepEvent::Continuation(continuation)),
            ok(ModelStepEvent::Usage(usage)),
            ok(ModelStepEvent::Finished {
                kind: ModelStepKind::Continue,
                model_id: ModelId::new("test-model"),
            }),
        ]))
        .await
        .unwrap();

        assert_eq!(step.kind, ModelStepKind::Continue);
        let output = step.output;
        assert_eq!(output.model_id.as_str(), "test-model");
        assert_eq!(output.usage.len(), 1);
        assert_eq!(
            output.continuation().map(|value| value.provider.as_str()),
            Some("test-provider")
        );
    }
}
