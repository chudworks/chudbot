//! Transcript reconstruction for live turns, retries, and stored turn replay.

use crate::prelude::*;
use crate::*;

pub(crate) fn transcript_message_metadata(id: String) -> serde_json::Value {
    serde_json::json!({ "id": id })
}

pub(crate) fn turn_transcript_message_id(turn_id: TurnId, role: &str) -> String {
    format!("chudbot_turn_{turn_id}_{role}")
}

/// Transcript assembly methods on the runtime.
impl<R> BotRuntime<R>
where
    R: BotRuntimeTypes + 'static,
{
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

    pub(crate) async fn transcript_from_snapshot(
        &self,
        snapshot: &ConversationSnapshot,
        history_cutoff: Option<i64>,
    ) -> Result<Transcript, BotError> {
        let mut transcript = Transcript::new();
        transcript.id = Some(snapshot.conversation.id.to_string());

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
                            // Generated media is replayed as a follow-up user turn so later
                            // image-edit requests can reference prior assistant outputs.
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

pub(crate) fn push_unique_string(out: &mut Vec<String>, value: &str) {
    if value.is_empty() || out.iter().any(|seen| seen == value) {
        return;
    }
    out.push(value.to_string());
}

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
        // Provider continuations preserve the assistant-side tool call state.
        // Matching tool results are replayed as the following user turn.
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

pub(crate) fn provider_client_tool_call_ids(
    continuation: &chudbot_api::ProviderContinuation,
) -> Vec<String> {
    let mut ids = Vec::new();
    collect_provider_client_tool_call_ids(&continuation.data, &mut ids);
    ids
}

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
            let (start, end) = markdown_link_bounds(&out, index, reference.len())
                .unwrap_or((index, index + reference.len()));
            out.replace_range(start..end, "");
        }
    }
    normalize_stripped_reply(&out)
}

pub(crate) fn markdown_link_bounds(
    text: &str,
    reference_start: usize,
    reference_len: usize,
) -> Option<(usize, usize)> {
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
    let close_bracket = before_open.len().checked_sub(1)?;
    let open_bracket = before_open[..close_bracket].rfind('[')?;
    let start = if open_bracket > 0 && text.as_bytes().get(open_bracket - 1) == Some(&b'!') {
        open_bracket - 1
    } else {
        open_bracket
    };
    Some((start, close_paren + 1))
}

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

pub(crate) fn replayable_context_items(
    context: &[chudbot_api::ContextItem],
) -> Vec<chudbot_api::ContextItem> {
    context
        .iter()
        .filter(|item| !is_memory_context_item(item))
        .cloned()
        .collect()
}

pub(crate) fn is_memory_context_item(item: &chudbot_api::ContextItem) -> bool {
    item.source.starts_with("memory:")
}
