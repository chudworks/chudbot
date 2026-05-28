//! Background conversation titler.
//!
//! After the first turn of a conversation completes successfully, the
//! bot calls [`spawn_generate`]. That task drives the same LLM
//! provider used for the conversation (so titling stays consistent
//! with the persona's voice) through a one-shot call with a very
//! short meta-system-prompt, then stores the result in
//! `conversations.title` and publishes [`EventKind::TitleUpdated`].
//!
//! The task is a one-shot per conversation — once `title_generated_at`
//! is set, we never regenerate. Manual retitle from the UI would be a
//! straightforward follow-up but isn't wired today.

use std::sync::Arc;

use grok_discord_bot_core::{
    ChatTurn, LlmProvider, MessageRole, ProviderOptions, StepRequest, StepResponse,
};
use uuid::Uuid;

use crate::app::{AppState, EventKind};

/// Hard cap on the title length we accept from the model. Discord's
/// thread name limit is 100; we stay under that to leave room for
/// any UI prefixes the viewer might add.
const MAX_TITLE_CHARS: usize = 80;

/// Soft token budget for the title call. Even a chatty model rarely
/// burns more than ~30 tokens on a 5-word title; this is just a safety
/// cap so a runaway response doesn't cost real money.
const TITLE_MAX_TOKENS: u32 = 96;

/// Meta-system prompt used to extract a title from the first turn.
/// Intentionally short and bossy — we want a label, not a sentence.
const TITLE_SYSTEM_PROMPT: &str = "You write very short conversation titles. \
Output ONLY a title for the conversation below — five words or fewer, \
no quotes, no period, no leading 'Re:' or 'Conversation about'. Just the \
title text. Title case is fine. If the conversation is small talk or \
a one-liner, pick a topic from the user's question, not a generic word \
like 'Greeting'.";

/// Schedule a background title generation for `conversation_id`. The
/// task drops itself silently if the conversation already has a
/// title — that's the natural "don't redo work after restart" guard
/// since the field is only set when this task succeeds.
pub fn spawn_generate(app: Arc<AppState>, conversation_id: Uuid, persona_name: String) {
    let tracker = app.tracker.clone();
    tracker.spawn(async move {
        if let Err(err) = generate(&app, conversation_id, &persona_name).await {
            tracing::warn!(
                conversation = %conversation_id,
                persona = %persona_name,
                error = %err,
                "title generation failed"
            );
        }
    });
}

#[derive(Debug, thiserror::Error)]
enum TitleError {
    #[error(transparent)]
    Db(#[from] grok_discord_bot_core::DbError),
    #[error(transparent)]
    Llm(#[from] grok_discord_bot_core::LlmError),
    #[error("conversation not found")]
    ConversationMissing,
    #[error("persona `{0}` is not configured")]
    PersonaMissing(String),
    #[error("no provider initialized for persona `{persona}` (`{provider}`)")]
    ProviderMissing { persona: String, provider: String },
    #[error("no completed turns to derive a title from")]
    NoTurns,
}

async fn generate(
    app: &AppState,
    conversation_id: Uuid,
    persona_name: &str,
) -> Result<(), TitleError> {
    let conv = app
        .db
        .get_conversation(conversation_id)
        .await?
        .ok_or(TitleError::ConversationMissing)?;
    if conv.title.is_some() && conv.title_generated_at.is_some() {
        tracing::debug!(
            conversation = %conversation_id,
            "title already set; skipping"
        );
        return Ok(());
    }
    let turns = app.db.load_conversation_history(conversation_id).await?;
    let first = turns.into_iter().next().ok_or(TitleError::NoTurns)?;
    let assistant = first.assistant_content.as_deref().unwrap_or("").to_string();

    let persona = app
        .personas
        .get(persona_name)
        .ok_or_else(|| TitleError::PersonaMissing(persona_name.to_string()))?;
    let provider =
        app.providers
            .get(&persona.provider)
            .ok_or_else(|| TitleError::ProviderMissing {
                persona: persona_name.to_string(),
                provider: persona.provider.as_str().to_string(),
            })?;

    let user_text = format!(
        "User said:\n{}\n\nAssistant replied:\n{}",
        first.user_content, assistant
    );

    // One-shot agent step — no tools, no web search, no looping. We
    // bypass `run_agent` because it's overkill for a single forward
    // pass that doesn't care about tool execution.
    let request = StepRequest {
        model: persona.model.clone(),
        messages: vec![
            ChatTurn::text(MessageRole::System, TITLE_SYSTEM_PROMPT.to_string()),
            ChatTurn::text(MessageRole::User, user_text),
        ],
        tools: Vec::new(),
        enable_web_search: false,
        max_tokens: TITLE_MAX_TOKENS,
        temperature: Some(0.3),
        top_p: None,
        provider_options: ProviderOptions {
            xai: persona.xai.clone(),
            anthropic: persona.anthropic.clone(),
        },
    };

    let response = tokio::select! {
        biased;
        _ = app.cancel.cancelled() => {
            tracing::debug!(conversation = %conversation_id, "title gen cancelled");
            return Ok(());
        }
        result = provider.step(request) => result?,
    };

    let raw = match response {
        StepResponse::Final { content, .. } => content,
        // The titler doesn't expose tools, so UseTools shouldn't fire.
        // If it does (a poorly-behaved provider), pull whatever
        // pre-tool text we have and use that.
        StepResponse::UseTools { partial_text, .. } => partial_text.unwrap_or_default(),
    };
    let title = clean_title(&raw);
    if title.is_empty() {
        tracing::warn!(
            conversation = %conversation_id,
            raw = %raw,
            "title generator returned empty/garbage; skipping write"
        );
        return Ok(());
    }

    app.db
        .set_conversation_title(conversation_id, &title)
        .await?;
    app.publish(conversation_id, EventKind::TitleUpdated);
    tracing::info!(
        conversation = %conversation_id,
        title = %title,
        "conversation title set"
    );
    Ok(())
}

/// Strip whitespace, surrounding quotes, and stray prefixes the model
/// likes to add ("Title:", "Conversation:", etc). Then truncate to
/// [`MAX_TITLE_CHARS`] at a char boundary so we never split a UTF-8
/// codepoint mid-byte.
fn clean_title(raw: &str) -> String {
    let trimmed = raw.trim();
    let trimmed = trimmed
        .strip_prefix("Title:")
        .or_else(|| trimmed.strip_prefix("title:"))
        .or_else(|| trimmed.strip_prefix("Conversation:"))
        .unwrap_or(trimmed)
        .trim();
    // Strip matching surrounding quotes / smart quotes if present.
    let trimmed = trimmed
        .strip_prefix('"')
        .and_then(|s| s.strip_suffix('"'))
        .or_else(|| {
            trimmed
                .strip_prefix('\'')
                .and_then(|s| s.strip_suffix('\''))
        })
        .or_else(|| trimmed.strip_prefix('“').and_then(|s| s.strip_suffix('”')))
        .unwrap_or(trimmed)
        .trim();
    // Truncate at a char boundary.
    if trimmed.chars().count() <= MAX_TITLE_CHARS {
        return trimmed.to_string();
    }
    trimmed.chars().take(MAX_TITLE_CHARS).collect::<String>() + "…"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clean_strips_quotes_and_prefix() {
        assert_eq!(clean_title("\"My Title\""), "My Title");
        assert_eq!(clean_title("Title: My Topic"), "My Topic");
        assert_eq!(clean_title("  hello world  "), "hello world");
    }

    #[test]
    fn clean_truncates_long_titles() {
        let long = "a".repeat(200);
        let cleaned = clean_title(&long);
        // 80 chars + ellipsis = 81 chars
        assert_eq!(cleaned.chars().count(), 81);
    }
}
