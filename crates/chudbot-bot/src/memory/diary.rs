//! Diary job input construction for user memory.
//!
//! A diary job summarizes one scheduler-selected window of completed user turns
//! into Markdown that later feeds memory compaction. Scheduling and result
//! persistence live in the runtime: this module resolves the reserved diary
//! agent and renders the transcript sent to it.
//!
//! The model result is not parsed here. Runtime stores the returned text
//! verbatim as diary Markdown tied to the same window and source turn ids, so
//! the output contract has to stay explicit in the prompt.
//!
//! Turn windows are selected before this code runs. Storage loads completed
//! turns from a half-open `[window_start, window_end)` interval, orders them
//! chronologically, and caps the slice by `max_transcript_turns_per_diary_job`;
//! `diary_transcript` preserves that slice exactly.

use std::collections::BTreeMap;

use chudbot_api::{
    AgentLimits, ContentBlock, MediaCategory, MediaRef, MediaStore, Transcript, TranscriptTurn,
    TurnRole, UserMemoryAudioTranscription, UserMemoryDocument, UserMemoryImageContext,
    UserMemoryKey, UserMemoryTurn,
};

use crate::config::{AgentConfig, SystemAgentConfig};

use super::{EMPTY_MEMORY, resolve_memory_agent};

/// Reserved agent name for memory diary jobs.
pub const MEMORY_DIARY_AGENT: &str = "memory_diary";

// Keep the output contract self-contained because diary text is saved verbatim
// as Markdown; there is no schema parser to repair or validate it afterward.
const DIARY_PROMPT: &str = "You write concise user-memory diary entries for Chudbot. \
Read the bounded transcript slice and optional current memory profile. Extract only \
stable, useful observations about the subject user. Include uncertainty when evidence \
is weak. Prefer factual bullets over prose. Consider relationships, preferences and \
dislikes, projects, work, hobbies, recurring topics, server lore, running jokes, \
good-natured roast material, corrections, stale facts, and visually meaningful \
image evidence. Do not invent facts.";

/// MIME types that are safe to replay as model-visible diary image inputs.
const MEMORY_DIARY_IMAGE_MIME_TYPES: &[&str] = &["image/png", "image/jpeg", "image/webp"];

/// Resolve the configured reserved diary agent, falling back to the built-in prompt.
pub(in crate::memory) fn resolve_agent(
    agents: &BTreeMap<String, AgentConfig>,
    default_limits: AgentLimits,
) -> SystemAgentConfig {
    resolve_memory_agent(
        MEMORY_DIARY_AGENT,
        DIARY_PROMPT,
        default_max_output_tokens(),
        agents,
        default_limits,
    )
}

/// Default output budget for concise, per-window diary Markdown.
fn default_max_output_tokens() -> u32 {
    1024
}

/// Build the single synthetic user turn sent to the diary agent.
///
/// `turns` must already be filtered to the diary job's completed-turn window.
/// This function does not inspect job timestamps or trim the slice; it only
/// renders the subject/profile header followed by each loaded turn in order.
pub(in crate::memory) async fn diary_transcript<M>(
    key: &UserMemoryKey,
    document: Option<&UserMemoryDocument>,
    turns: &[UserMemoryTurn],
    media_store: &M,
) -> Transcript
where
    M: MediaStore,
{
    let mut blocks = Vec::new();
    blocks.push(ContentBlock::Text {
        text: diary_header_text(key, document),
    });
    for turn in turns {
        blocks.push(ContentBlock::Text {
            text: diary_turn_text(turn),
        });
        append_diary_image_blocks(&mut blocks, turn, media_store).await;
    }
    let mut transcript = Transcript::new();
    transcript.push(TranscriptTurn {
        role: TurnRole::User,
        blocks,
        metadata: serde_json::Value::Null,
    });
    transcript
}

/// Render the subject ids and current compact memory profile.
fn diary_header_text(key: &UserMemoryKey, document: Option<&UserMemoryDocument>) -> String {
    let mut out = String::new();
    out.push_str("# Subject\n");
    out.push_str(&format!(
        "platform: {}\nscope: {}\nuser: {}\n\n",
        key.platform, key.scope_key, key.user_key
    ));
    out.push_str("# Current Memory Profile\n");
    // Keep the section shape stable: an absent or blank profile is explicit,
    // rather than silently removing the context block from the prompt.
    out.push_str(
        document
            .map(|document| document.markdown.trim())
            .filter(|markdown| !markdown.is_empty())
            .unwrap_or(EMPTY_MEMORY),
    );
    out.push_str("\n\n# Completed Turns\n");
    out
}

/// Render one completed turn as plain text within the diary prompt.
fn diary_turn_text(turn: &UserMemoryTurn) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "\n## Turn {} ({})\nUser [{}]: {}\n",
        turn.turn_id, turn.completed_at, turn.user_display_name, turn.user_content
    ));
    if let Some(answer) = &turn.assistant_content {
        out.push_str("Assistant: ");
        out.push_str(answer);
        out.push('\n');
    }
    append_image_context(&mut out, &turn.image_context);
    append_audio_transcriptions(&mut out, &turn.audio_transcriptions);
    out
}

/// Render image metadata; binary image blocks are appended separately.
fn append_image_context(out: &mut String, images: &[UserMemoryImageContext]) {
    if images.is_empty() {
        return;
    }
    out.push_str("Image content blocks:\n");
    for (index, image) in images.iter().enumerate() {
        let mut metadata = vec![
            format!("source: {}", memory_image_source_label(&image.source)),
            format!("uri: {}", image.image_uri),
        ];
        if let Some(mime_type) = image
            .mime_type
            .as_deref()
            .filter(|mime_type| !mime_type.is_empty())
        {
            metadata.push(format!("mime_type: {mime_type}"));
        }
        out.push_str(&format!(
            "- Image {} ({})\n",
            index + 1,
            metadata.join(", ")
        ));
    }
}

/// Convert stored media provenance into labels that are meaningful to the agent.
fn memory_image_source_label(source: &str) -> &str {
    if source.starts_with("platform:") {
        // `platform:<kind>:<message-id>` is provenance, not useful prompt
        // content, so collapse it to the user-visible attachment category.
        "user_or_quoted_message_attachment"
    } else if source == "generate_image" {
        "generated_image"
    } else {
        source
    }
}

/// Render successful audio transcriptions as model-visible turn context.
fn append_audio_transcriptions(out: &mut String, transcriptions: &[UserMemoryAudioTranscription]) {
    let mut rendered_any = false;
    for (index, transcription) in transcriptions.iter().enumerate() {
        // Storage has already parsed the tool trace. Here we only discard blank
        // provider text so empty transcriptions do not create prompt noise.
        let text = transcription.text.trim();
        if text.is_empty() {
            continue;
        }
        if !rendered_any {
            out.push_str("Audio transcriptions:\n");
            rendered_any = true;
        }
        let mut metadata = Vec::new();
        if let Some(uri) = transcription
            .audio_uri
            .as_deref()
            .filter(|uri| !uri.is_empty())
        {
            metadata.push(format!("uri: {uri}"));
        }
        if let Some(language) = transcription
            .language
            .as_deref()
            .filter(|language| !language.is_empty())
        {
            metadata.push(format!("language: {language}"));
        }
        // Omit unknown metadata keys instead of rendering empty placeholders.
        if let Some(duration) = transcription.duration_seconds {
            metadata.push(format!("duration_seconds: {duration:.2}"));
        }
        let metadata = if metadata.is_empty() {
            String::new()
        } else {
            format!(" ({})", metadata.join(", "))
        };
        out.push_str(&format!("- Audio {}{}: {}\n", index + 1, metadata, text));
    }
}

/// Append supported image payloads after a text breadcrumb for each image.
async fn append_diary_image_blocks<M>(
    blocks: &mut Vec<ContentBlock>,
    turn: &UserMemoryTurn,
    media_store: &M,
) where
    M: MediaStore,
{
    for (index, image) in turn.image_context.iter().enumerate() {
        // The text marker remains useful even when the binary asset is missing
        // or rejected by the provider-specific media filter below.
        blocks.push(ContentBlock::Text {
            text: format!(
                "Visual content for turn {} image {} (source: {}, uri: {}).",
                turn.turn_id,
                index + 1,
                memory_image_source_label(&image.source),
                image.image_uri
            ),
        });
        match media_store.media_from_uri(&image.image_uri).await {
            Ok(media) if memory_diary_supports_media(media.as_ref()) => {
                blocks.push(ContentBlock::Media { media });
            }
            Ok(media) => tracing::debug!(
                turn = %turn.turn_id,
                source = %image.source,
                uri = %media.uri(),
                category = ?media.category(),
                mime_type = %media.mime_type(),
                "skipping unsupported diary image media"
            ),
            Err(error) => tracing::warn!(
                turn = %turn.turn_id,
                source = %image.source,
                uri = %image.image_uri,
                error = %error,
                "skipping diary image media"
            ),
        }
    }
}

/// Return whether this media asset can be replayed as a diary image block.
fn memory_diary_supports_media(media: &dyn MediaRef) -> bool {
    matches!(media.category(), MediaCategory::Image)
        && MEMORY_DIARY_IMAGE_MIME_TYPES
            .iter()
            .any(|supported| image_mime_type_eq(media.mime_type(), supported))
}

/// Compare MIME types while ignoring parameters such as `; charset=binary`.
fn image_mime_type_eq(actual: &str, expected: &str) -> bool {
    let actual = actual.split(';').next().unwrap_or("").trim();
    actual.eq_ignore_ascii_case(expected)
}

#[cfg(test)]
/// Render only text sections for unit tests that do not need media replay.
pub(super) fn diary_input(
    key: &UserMemoryKey,
    document: Option<&UserMemoryDocument>,
    turns: &[UserMemoryTurn],
) -> String {
    let mut out = diary_header_text(key, document);
    for turn in turns {
        out.push_str(&diary_turn_text(turn));
    }
    out
}

#[cfg(test)]
mod tests {
    use chudbot_api::{
        ConversationId, PlatformName, TurnId, UserMemoryAudioTranscription, UserMemoryKey,
        UserMemoryTurn,
    };
    use time::macros::datetime;

    use super::*;

    #[test]
    fn diary_input_includes_audio_transcriptions() {
        let key = UserMemoryKey {
            platform: PlatformName::new("discord"),
            scope_key: "guild:guild-1".to_string(),
            user_key: "user-1".to_string(),
        };
        let turn = UserMemoryTurn {
            conversation_id: ConversationId::new(),
            turn_id: TurnId::new(),
            completed_at: datetime!(2026-06-03 22:27:01 UTC),
            user_display_name: "Chud".to_string(),
            user_content: "@Chudbot".to_string(),
            assistant_content: Some("Noted.".to_string()),
            image_context: Vec::new(),
            audio_transcriptions: vec![UserMemoryAudioTranscription {
                tool_trace_id: 42,
                audio_uri: Some("file://audio/voice.ogg".to_string()),
                text: "I am allergic to coconut.".to_string(),
                language: Some("en".to_string()),
                duration_seconds: Some(3.25),
            }],
        };

        let input = diary_input(&key, None, &[turn]);

        assert!(input.contains("Audio transcriptions:"));
        assert!(input.contains("file://audio/voice.ogg"));
        assert!(input.contains("language: en"));
        assert!(input.contains("duration_seconds: 3.25"));
        assert!(input.contains("I am allergic to coconut."));
    }
}
