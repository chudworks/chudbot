//! Model-facing transcript reconstruction for live turns, retries, and stored
//! turn replay.
//!
//! This module translates stored conversation state and freshly fetched context
//! into the ordered `Transcript` consumed by LLM providers. It keeps platform
//! and storage details at the edge: context items become text/media blocks,
//! provider continuations are replayed when available, legacy tool traces are
//! used as a fallback, and generated assistant media is reintroduced as
//! synthetic user context so later image-edit requests can reference it.

use crate::prelude::*;
use crate::*;

/// Build stable metadata for model-facing messages reconstructed from a turn.
pub(crate) fn transcript_message_metadata(id: String) -> serde_json::Value {
    serde_json::json!({ "id": id })
}

/// Return the stable synthetic message id used for a turn/role pair.
pub(crate) fn turn_transcript_message_id(turn_id: TurnId, role: &str) -> String {
    format!("chudbot_turn_{turn_id}_{role}")
}

/// Transcript assembly methods that need runtime services such as the media
/// store.
impl<R> BotRuntime<R>
where
    R: BotRuntimeTypes + 'static,
{
    /// Build the transcript for a normal live model call.
    ///
    /// Completed history up to the turn's cutoff is replayed first, then the
    /// freshly gathered current-turn context is appended as the final user
    /// message.
    pub(crate) async fn transcript_for_turn(
        &self,
        snapshot: &ConversationSnapshot,
        turn: &Turn,
        context: &[chudbot_api::ContextItem],
    ) -> Result<Transcript, BotError> {
        let mut transcript = self
            .transcript_from_snapshot(snapshot, turn.history_cutoff)
            .await?;
        transcript.push(self.transcript_turn_from_context(turn.id, context).await);
        tracing::debug!(
            turns = transcript.turns.len(),
            "assembled transcript for live turn"
        );
        Ok(transcript)
    }

    /// Build the transcript for retrying a stored turn.
    ///
    /// Retries use the same completed-history replay as live turns, then append
    /// the retried user turn. When the original stored context is unavailable,
    /// the fallback preserves the user-visible text and appends any supplied
    /// context blocks after it.
    #[tracing::instrument(
        name = "bot.transcript_for_retry",
        skip_all,
        fields(
            conversation = %snapshot.conversation.id,
            retry_turn = %retry_turn.turn.id,
            retry_turn_ordinal = retry_turn.turn.ordinal,
            history_cutoff = ?retry_turn.turn.history_cutoff,
        )
    )]
    pub(crate) async fn transcript_for_retry(
        &self,
        snapshot: &ConversationSnapshot,
        retry_turn: &TurnSnapshot,
        context: &[chudbot_api::ContextItem],
        has_stored_context: bool,
    ) -> Result<Transcript, BotError> {
        let mut transcript = self
            .transcript_from_snapshot(snapshot, retry_turn.turn.history_cutoff)
            .await?;
        // Prefer the context captured with the original attempt; older turns
        // may only have the stored display text plus newly supplied context.
        if has_stored_context {
            transcript.push(
                self.transcript_turn_from_context(retry_turn.turn.id, context)
                    .await,
            );
        } else {
            let mut turn = TranscriptTurn::text(
                TurnRole::User,
                format!(
                    "[{}]: {}",
                    retry_turn.turn.user_display_name, retry_turn.turn.user_content
                ),
            );
            turn.metadata =
                transcript_message_metadata(turn_transcript_message_id(retry_turn.turn.id, "user"));
            let mut extra_blocks = self.context_blocks_from_items(context).await;
            turn.blocks.append(&mut extra_blocks);
            transcript.push(turn);
        }
        tracing::debug!(
            turns = transcript.turns.len(),
            "assembled transcript for retry"
        );
        Ok(transcript)
    }

    /// Convert context items for one user turn into a transcript turn.
    ///
    /// Empty context still becomes an explicit text block so providers receive a
    /// well-formed user message instead of an empty block list.
    pub(crate) async fn transcript_turn_from_context(
        &self,
        turn_id: TurnId,
        context: &[chudbot_api::ContextItem],
    ) -> TranscriptTurn {
        let mut blocks = self.context_blocks_from_items(context).await;
        if blocks.is_empty() {
            blocks.push(ContentBlock::Text {
                text: "(no message content)".to_string(),
            });
        }
        TranscriptTurn {
            role: TurnRole::User,
            blocks,
            metadata: transcript_message_metadata(turn_transcript_message_id(turn_id, "user")),
        }
    }

    /// Convert context items into provider-ready content blocks.
    ///
    /// Stored `file://` handles are resolved through the media store and only
    /// model-supported media is passed through; unsupported or missing media is
    /// skipped with logging instead of failing the whole transcript assembly.
    pub(crate) async fn context_blocks_from_items(
        &self,
        context: &[chudbot_api::ContextItem],
    ) -> Vec<ContentBlock> {
        let mut blocks = Vec::new();
        for item in context {
            if item.content.starts_with("file://") {
                match self
                    .media_store
                    .media_from_uri(&MediaUri::new(item.content.clone()))
                    .await
                {
                    Ok(media) if model_transcript_supports_media(media.as_ref()) => {
                        blocks.push(ContentBlock::Media { media })
                    }
                    Ok(media) => tracing::debug!(
                        source = %item.source,
                        uri = %media.uri(),
                        category = ?media.category(),
                        mime_type = %media.mime_type(),
                        "skipping unsupported context media while assembling transcript"
                    ),
                    Err(error) => tracing::warn!(
                        error = %error,
                        source = %item.source,
                        uri = %item.content,
                        "skipping context media while assembling transcript"
                    ),
                }
                continue;
            }
            blocks.push(ContentBlock::Text {
                text: item.content.clone(),
            });
        }
        blocks
    }

    /// Rebuild model-visible history from a stored conversation snapshot.
    ///
    /// Only completed turns at or before `history_cutoff` are replayed. A
    /// missing cutoff intentionally means no prior history should be included.
    /// For each replayed turn, the function reconstructs the user message,
    /// provider/tool continuation sequence, final assistant text, and any
    /// generated media needed for future reference-image edits.
    pub(crate) async fn transcript_from_snapshot(
        &self,
        snapshot: &ConversationSnapshot,
        history_cutoff: Option<i64>,
    ) -> Result<Transcript, BotError> {
        let mut transcript = Transcript::new();
        transcript.id = Some(snapshot.conversation.id.to_string());

        // Step 1: select only completed responses the next model call is
        // allowed to see, then sort by response order with turn order as a
        // deterministic tie-breaker.
        let mut replay_turns = snapshot
            .turns
            .iter()
            .filter(|turn| matches!(turn.turn.status, chudbot_api::TurnStatus::Completed))
            .filter(|turn| {
                let Some(history_cutoff) = history_cutoff else {
                    return false;
                };
                turn.turn
                    .response_ordinal
                    .is_some_and(|ordinal| ordinal <= history_cutoff)
            })
            .collect::<Vec<_>>();
        replay_turns.sort_by_key(|turn| {
            (
                turn.turn.response_ordinal.unwrap_or(i64::MAX),
                turn.turn.ordinal,
            )
        });

        for turn in replay_turns {
            // Step 2: rebuild the prior user message. Memory context is
            // prompt scaffolding for the original call, not durable chat
            // history, so it is dropped when replaying completed turns.
            let replay_context = replayable_context_items(&turn.context);
            let mut user_turn = if replay_context.is_empty() {
                TranscriptTurn {
                    role: TurnRole::User,
                    blocks: vec![ContentBlock::Text {
                        text: format!(
                            "[{}]: {}",
                            turn.turn.user_display_name, turn.turn.user_content
                        ),
                    }],
                    metadata: transcript_message_metadata(turn_transcript_message_id(
                        turn.turn.id,
                        "user",
                    )),
                }
            } else {
                self.transcript_turn_from_context(turn.turn.id, &replay_context)
                    .await
            };
            // Track media already present in context so replay assets do not
            // duplicate the same model-visible attachment.
            let mut replayed_media = replay_context
                .iter()
                .filter(|item| item.content.starts_with("file://"))
                .map(|item| item.content.clone())
                .collect::<Vec<_>>();
            let mut generated_media_refs = Vec::new();
            let mut generated_media_blocks = Vec::new();
            for asset in &turn.replay_assets {
                if replayed_media
                    .iter()
                    .any(|uri| uri.as_str() == asset.uri.as_str())
                {
                    continue;
                }
                match self.media_store.media_from_uri(&asset.uri).await {
                    Ok(media) => {
                        if !model_transcript_supports_media(media.as_ref()) {
                            tracing::debug!(
                                source = %asset.source,
                                uri = %media.uri(),
                                category = ?media.category(),
                                mime_type = %media.mime_type(),
                                "skipping unsupported replay media while rebuilding transcript"
                            );
                            continue;
                        }
                        replayed_media.push(asset.uri.as_str().to_string());
                        if replay_asset_belongs_to_user_turn(asset) {
                            user_turn.blocks.push(ContentBlock::Media { media });
                        } else {
                            // Generated media is replayed as a follow-up user
                            // turn so later image-edit requests can reference
                            // prior assistant outputs.
                            generated_media_refs.push(asset.uri.as_str().to_string());
                            generated_media_blocks.push(ContentBlock::Media { media });
                        }
                    }
                    Err(error) => tracing::warn!(
                        error = %error,
                        uri = %asset.uri,
                        "skipping replay media while rebuilding transcript"
                    ),
                }
            }
            tracing::trace!(
                turn = %turn.turn.id,
                replay_assets = turn.replay_assets.len(),
                user_blocks = user_turn.blocks.len(),
                "added prior user turn to transcript"
            );
            transcript.push(user_turn);
            // Step 3: prefer provider model-step replay because it preserves
            // opaque continuation state such as reasoning ids and tool-call
            // ids. Legacy client-tool replay is only used when no model steps
            // were stored for the turn.
            let replayed_from_model_steps = append_model_step_replay(
                &mut transcript,
                &turn.model_steps,
                &turn.tool_trace,
                turn.turn.assistant_content.as_deref(),
            );

            if !replayed_from_model_steps {
                append_client_tool_replay(&mut transcript, &turn.tool_trace);
            }

            if !replayed_from_model_steps && let Some(answer) = &turn.turn.assistant_content {
                let blocks = vec![ContentBlock::Text {
                    text: answer.clone(),
                }];
                transcript.push(TranscriptTurn {
                    role: TurnRole::Assistant,
                    blocks,
                    metadata: transcript_message_metadata(turn_transcript_message_id(
                        turn.turn.id,
                        "assistant",
                    )),
                });
                tracing::trace!(
                    turn = %turn.turn.id,
                    "added prior assistant turn to transcript"
                );
            }
            // Step 4: attach generated assistant media after the assistant
            // answer, as synthetic user context with stable reference ids.
            append_generated_media_replay(
                &mut transcript,
                turn.turn.id,
                generated_media_refs,
                generated_media_blocks,
            );
        }

        tracing::debug!(
            transcript_turns = transcript.turns.len(),
            "rebuilt transcript from snapshot"
        );
        Ok(transcript)
    }
}

/// Append a synthetic user turn containing media produced by the previous
/// assistant response.
///
/// The text block exposes the exact stored media ids that image tools should
/// use as `reference_images`. Keeping this as a user turn makes prior generated
/// images available to providers that only accept media in user messages.
pub(crate) fn append_generated_media_replay(
    transcript: &mut Transcript,
    turn_id: TurnId,
    media_refs: Vec<String>,
    mut media_blocks: Vec<ContentBlock>,
) {
    if media_blocks.is_empty() {
        return;
    }

    let mut text = "Generated media attached to the previous assistant reply.".to_string();
    if !media_refs.is_empty() {
        text.push_str(" Image reference IDs available for tool calls: ");
        text.push_str(&media_refs.join(", "));
        text.push_str(concat!(
            ". Use these exact IDs in generate_image.reference_images when the user asks to ",
            "edit, restyle, transform, or make a variation of the images."
        ));
    }

    let mut blocks = Vec::with_capacity(media_blocks.len() + 1);
    blocks.push(ContentBlock::Text { text });
    blocks.append(&mut media_blocks);
    transcript.push(TranscriptTurn {
        role: TurnRole::User,
        blocks,
        metadata: transcript_message_metadata(turn_transcript_message_id(
            turn_id,
            "assistant_media",
        )),
    });
}

/// Replay legacy client-tool traces as assistant calls followed by user
/// results.
///
/// Newer transcripts prefer stored model-step continuations because they carry
/// provider-specific call state; this helper remains for turns captured before
/// model steps were recorded.
pub(crate) fn append_client_tool_replay(transcript: &mut Transcript, traces: &[ToolTrace]) {
    let mut call_blocks = Vec::new();
    let mut result_blocks = Vec::new();
    for trace in traces {
        let ToolTrace::Client { trace } = trace else {
            continue;
        };
        call_blocks.push(ContentBlock::ClientToolCall(trace.call.clone()));
        result_blocks.push(ContentBlock::ClientToolResult(trace.result.clone()));
    }
    if call_blocks.is_empty() {
        return;
    }
    transcript.push(TranscriptTurn {
        role: TurnRole::Assistant,
        blocks: call_blocks,
        metadata: serde_json::Value::Null,
    });
    transcript.push(TranscriptTurn {
        role: TurnRole::User,
        blocks: result_blocks,
        metadata: serde_json::Value::Null,
    });
}

/// Collect media references already visible in a transcript.
///
/// The returned values are stored URIs and public URLs that the next model call
/// may echo in its answer. Callers merge them with media references from the
/// current run before removing redundant links from user-facing reply text.
pub(crate) async fn media_reply_refs_from_transcript(transcript: &Transcript) -> Vec<String> {
    let mut out = Vec::new();
    for turn in &transcript.turns {
        for block in &turn.blocks {
            let ContentBlock::Media { media } = block else {
                if let ContentBlock::ClientToolResult(result) = block
                    && let ClientToolResultContent::Json { value } = &result.content
                {
                    collect_generated_media_reply_refs(value, &mut out);
                }
                continue;
            };
            push_unique_string(&mut out, media.uri().as_str());
            if let Ok(public_url) = media.public_url().await {
                push_unique_string(&mut out, public_url.as_str());
            }
        }
    }
    out
}

/// Append a nonempty string once while preserving first-seen order.
pub(crate) fn push_unique_string(out: &mut Vec<String>, value: &str) {
    if value.is_empty() || out.iter().any(|seen| seen == value) {
        return;
    }
    out.push(value.to_string());
}

/// Replay stored provider model steps into transcript turns.
///
/// Returns `true` when model-step data was present and consumed. Each stored
/// assistant continuation is emitted as an assistant turn, matching client-tool
/// results are emitted as the next user turn, and the final assistant text is
/// appended to the last step when available.
pub(crate) fn append_model_step_replay(
    transcript: &mut Transcript,
    model_steps: &[ModelStepTrace],
    traces: &[ToolTrace],
    assistant_content: Option<&str>,
) -> bool {
    if model_steps.is_empty() {
        return false;
    }

    let client_results = client_tool_results_by_id(traces);
    let mut consumed_results = BTreeSet::new();
    for (index, step) in model_steps.iter().enumerate() {
        // Step 1: replay assistant-side provider state. Continuations preserve
        // tool-call ids and opaque reasoning state needed by the same provider.
        let is_final_step = index + 1 == model_steps.len();
        let mut assistant_blocks = Vec::new();
        if let Some(continuation) = &step.continuation {
            assistant_blocks.push(ContentBlock::Continuation(continuation.clone()));
        }
        if is_final_step
            && let Some(answer) = assistant_content
            && !answer.is_empty()
        {
            assistant_blocks.push(ContentBlock::Text {
                text: answer.to_string(),
            });
        }
        if !assistant_blocks.is_empty() {
            transcript.push(TranscriptTurn {
                role: TurnRole::Assistant,
                blocks: assistant_blocks,
                metadata: serde_json::Value::Null,
            });
        }

        // Step 2: pair this assistant step with the stored client-tool results
        // whose ids were named by the provider continuation.
        let call_ids = step
            .continuation
            .as_ref()
            .map(provider_client_tool_call_ids)
            .unwrap_or_default();
        let mut result_blocks = Vec::new();
        for call_id in call_ids {
            if consumed_results.contains(&call_id) {
                continue;
            }
            if let Some(result) = client_results.get(&call_id) {
                consumed_results.insert(call_id);
                result_blocks.push(ContentBlock::ClientToolResult(result.clone()));
            }
        }

        // Step 3: preserve replay for older client-tool steps that lack
        // provider continuation data and therefore cannot name call ids.
        if step.kind == ModelStepKind::ClientTools
            && step.continuation.is_none()
            && result_blocks.is_empty()
        {
            // Older traces may not carry provider continuation call ids. In that
            // case, replay any unconsumed client-tool results at the tool step.
            for (call_id, result) in &client_results {
                if consumed_results.insert(call_id.clone()) {
                    result_blocks.push(ContentBlock::ClientToolResult(result.clone()));
                }
            }
        }

        if !result_blocks.is_empty() {
            transcript.push(TranscriptTurn {
                role: TurnRole::User,
                blocks: result_blocks,
                metadata: serde_json::Value::Null,
            });
        }
    }

    true
}

/// Index stored client-tool results by provider-visible tool-call id.
pub(crate) fn client_tool_results_by_id(
    traces: &[ToolTrace],
) -> BTreeMap<String, ClientToolResult> {
    let mut results = BTreeMap::new();
    for trace in traces {
        let ToolTrace::Client { trace } = trace else {
            continue;
        };
        results.insert(
            trace.result.tool_use_id.as_str().to_string(),
            trace.result.clone(),
        );
    }
    results
}

/// Extract provider-visible client-tool call ids from a continuation payload.
pub(crate) fn provider_client_tool_call_ids(
    continuation: &chudbot_api::ProviderContinuation,
) -> Vec<String> {
    let mut ids = Vec::new();
    collect_provider_client_tool_call_ids(&continuation.data, &mut ids);
    ids
}

/// Recursively collect client-tool call ids from provider continuation JSON.
///
/// The shapes covered here match the provider payloads Chudbot stores today:
/// OpenAI-compatible `function_call` objects and Anthropic `tool_use` blocks.
pub(crate) fn collect_provider_client_tool_call_ids(
    value: &serde_json::Value,
    out: &mut Vec<String>,
) {
    match value {
        serde_json::Value::Array(items) => {
            for item in items {
                collect_provider_client_tool_call_ids(item, out);
            }
        }
        serde_json::Value::Object(object) => {
            let id = match object.get("type").and_then(serde_json::Value::as_str) {
                Some("function_call") => object
                    .get("call_id")
                    .or_else(|| object.get("id"))
                    .and_then(serde_json::Value::as_str),
                Some("tool_use") => object.get("id").and_then(serde_json::Value::as_str),
                _ => None,
            };
            if let Some(id) = id
                && !out.iter().any(|seen| seen == id)
            {
                out.push(id.to_string());
            }
        }
        _ => {}
    }
}

/// Remove generated-media references from assistant text before display.
///
/// Tool outputs and replay media can make the model include raw media ids or
/// public URLs in its final answer. This strips only the known generated-media
/// references, including full Markdown links/images when the reference is the
/// link target.
pub(crate) fn strip_generated_media_refs(text: &str, refs: &[String]) -> String {
    if refs.is_empty() {
        return text.to_string();
    }

    let mut out = text.to_string();
    for reference in refs {
        if reference.is_empty() {
            continue;
        }
        while let Some(index) = out.find(reference) {
            // Prefer removing the whole Markdown link/image wrapper when the
            // generated media URL is the target; otherwise remove just the raw
            // reference text.
            let (start, end) = markdown_link_bounds(&out, index, reference.len())
                .unwrap_or((index, index + reference.len()));
            out.replace_range(start..end, "");
        }
    }
    normalize_stripped_reply(&out)
}

/// Return the bounds of a Markdown link or image whose target is a reference.
pub(crate) fn markdown_link_bounds(
    text: &str,
    reference_start: usize,
    reference_len: usize,
) -> Option<(usize, usize)> {
    // The reference must be exactly inside `(reference)`.
    let open_paren = reference_start.checked_sub(1)?;
    if text.as_bytes().get(open_paren) != Some(&b'(') {
        return None;
    }
    let close_paren = reference_start.checked_add(reference_len)?;
    if text.as_bytes().get(close_paren) != Some(&b')') {
        return None;
    }
    let before_open = &text[..open_paren];
    if !before_open.ends_with(']') {
        return None;
    }
    // Walk back to the link label and include a leading `!` for images.
    let close_bracket = before_open.len().checked_sub(1)?;
    let open_bracket = before_open[..close_bracket].rfind('[')?;
    let start = if open_bracket > 0 && text.as_bytes().get(open_bracket - 1) == Some(&b'!') {
        open_bracket - 1
    } else {
        open_bracket
    };
    Some((start, close_paren + 1))
}

/// Normalize whitespace left after media-reference stripping.
///
/// This trims trailing spaces, collapses runs of blank lines, and removes blank
/// lines at the start or end so the user-facing reply remains clean after
/// generated media links are removed.
pub(crate) fn normalize_stripped_reply(text: &str) -> String {
    let mut lines = Vec::new();
    let mut previous_blank = true;
    for line in text.lines() {
        let line = line.trim_end();
        let blank = line.trim().is_empty();
        if blank {
            if !previous_blank {
                lines.push(String::new());
            }
        } else {
            lines.push(line.to_string());
        }
        previous_blank = blank;
    }
    while lines.last().is_some_and(|line| line.is_empty()) {
        lines.pop();
    }
    lines.join("\n")
}

/// Return context items that should be carried forward when replaying history.
///
/// Memory items are omitted because they are prompt-time context, not durable
/// chat messages or attachments that should appear again in completed-history
/// replay.
pub(crate) fn replayable_context_items(
    context: &[chudbot_api::ContextItem],
) -> Vec<chudbot_api::ContextItem> {
    context
        .iter()
        .filter(|item| !is_memory_context_item(item))
        .cloned()
        .collect()
}

/// Identify context supplied by the memory system.
pub(crate) fn is_memory_context_item(item: &chudbot_api::ContextItem) -> bool {
    item.source.starts_with("memory:")
}
