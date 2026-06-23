//! Media attachment ingestion, generated-media delivery, and audio preflight helpers.
//!
//! This module keeps media-specific turn assembly out of the main orchestration
//! code. It bridges platform attachments, stored `MediaStore` assets, model
//! transcript blocks, media-access tools, and final platform reply attachments.
//!
//! Most helpers are intentionally best-effort. A missing attachment, failed
//! download, unavailable public URL, or unsupported MIME type should be logged
//! and skipped without failing the whole assistant turn.

use crate::config::audio_transcription_default_keyterms;
use crate::prelude::*;
use crate::*;

/// Input needed to turn a platform message into model context.
pub(crate) struct MessageContextInput<'a> {
    /// Stable source label used in stored context ids, such as `message` or `quoted`.
    pub(crate) kind: &'a str,
    /// Platform message whose text and attachments are being converted.
    pub(crate) message: &'a PlatformMessage,
    /// Relationship between the source message and the current turn.
    pub(crate) relationship: PlatformMessageRelationship,
    /// Preflight audio save state.
    ///
    /// `None` means `push_message_context` should download matching audio
    /// attachments itself. `Some(vec)` means the caller already decided which
    /// saved audio refs, possibly none, belong in this context.
    pub(crate) saved_audio: Option<Vec<StoredAttachmentMedia>>,
    /// Automatic transcriptions to inject beside the platform attachment data.
    pub(crate) audio_transcriptions: &'a [IncomingAudioTranscription],
}

/// Saved audio/transcription state produced before deciding whether to run a turn.
#[derive(Debug, Default)]
pub(crate) struct IncomingAudioContext {
    /// Optional audio refs produced or intentionally suppressed during preflight.
    pub(crate) saved_audio: Option<Vec<StoredAttachmentMedia>>,
    /// Whether saved audio refs should remain visible to the model.
    ///
    /// Successful transcription replaces the original audio with text, so this
    /// is false in that path to avoid sending the same content twice.
    pub(crate) expose_audio_to_model: bool,
    /// Successful automatic transcriptions for audio-like incoming attachments.
    pub(crate) transcriptions: Vec<IncomingAudioTranscription>,
}

impl IncomingAudioContext {
    /// Return usage records charged by automatic transcription calls.
    pub(crate) fn usage_records(&self) -> Vec<UsageRecord> {
        self.transcriptions
            .iter()
            .flat_map(|transcription| transcription.usage.iter().cloned())
            .collect()
    }

    /// Convert automatic transcription calls into synthetic client-tool traces.
    ///
    /// These traces make pre-turn audio processing visible in the stored trace
    /// viewer even though no model explicitly requested the transcription.
    pub(crate) fn tool_traces(&self) -> Vec<ToolTrace> {
        self.transcriptions
            .iter()
            .map(IncomingAudioTranscription::tool_trace)
            .collect()
    }
}

/// Automatic transcription result for an incoming audio attachment.
#[derive(Debug)]
pub(crate) struct IncomingAudioTranscription {
    /// Original attachment position in the platform message.
    pub(crate) attachment_index: usize,
    /// Stored media URI when the audio was saved and exposed in context.
    pub(crate) audio_uri: Option<String>,
    /// Transcribed user-visible text.
    pub(crate) text: String,
    /// Provider-reported language, when available.
    pub(crate) language: Option<String>,
    /// Provider-reported audio duration in seconds.
    pub(crate) duration_seconds: f64,
    /// Compact result JSON shown to the model/tool-result stream.
    pub(crate) result: serde_json::Value,
    /// Full trace payload retained for the viewer.
    pub(crate) trace_response: serde_json::Value,
    /// Provider usage records returned by the transcription call.
    pub(crate) usage: Vec<UsageRecord>,
}

impl IncomingAudioTranscription {
    /// Represent an automatic transcription as a client tool call/result pair.
    pub(crate) fn tool_trace(&self) -> ToolTrace {
        let call = ClientToolCall {
            id: ToolUseId::new(format!("auto-transcribe-audio-{}", self.attachment_index)),
            name: ToolName::new(TRANSCRIBE_AUDIO_TOOL),
            input: incoming_audio_tool_input(self),
        };
        let result = ClientToolResult {
            tool_use_id: call.id.clone(),
            content: ClientToolResultContent::Json {
                value: self.result.clone(),
            },
            is_error: false,
        };
        ToolTrace::Client {
            trace: ClientToolTrace {
                call,
                result,
                trace_response: self.trace_response.clone(),
                usage: self.usage.clone(),
            },
        }
    }
}

/// Build the synthetic input payload for the automatic transcription trace.
pub(crate) fn incoming_audio_tool_input(
    transcription: &IncomingAudioTranscription,
) -> serde_json::Value {
    let mut value = serde_json::json!({
        "attachment_index": transcription.attachment_index,
    });
    if let Some(audio_uri) = &transcription.audio_uri
        && let Some(object) = value.as_object_mut()
    {
        object.insert(
            "audio_uri".to_string(),
            serde_json::Value::String(audio_uri.clone()),
        );
    }
    value
}

/// Media saved from a platform attachment and its original attachment index.
#[derive(Debug)]
pub(crate) struct StoredAttachmentMedia {
    /// Original attachment position in the platform message.
    pub(crate) attachment_index: usize,
    /// Stored media ref created from the platform attachment.
    pub(crate) media: chudbot_api::BoxedMediaRef,
}

/// Generated media ready to attach to the platform reply plus URL fallbacks.
pub(crate) struct GeneratedReplyMedia {
    /// In-memory attachments ready for the platform adapter to send.
    pub(crate) attachments: Vec<OutgoingAttachment>,
    /// Public URLs for media that could not be attached directly.
    pub(crate) public_urls: Vec<String>,
}

impl<R> BotRuntime<R>
where
    R: BotRuntimeTypes + 'static,
{
    /// Run audio preflight for no-mention messages before deciding to wake the bot.
    ///
    /// The flow is:
    /// 1. require an agent-level transcription binding;
    /// 2. transcribe every audio-like attachment using its platform URL;
    /// 3. return state that tells later context assembly whether to expose the
    ///    original audio or the transcription text.
    pub(crate) async fn prepare_incoming_audio_context(
        &self,
        message: &PlatformMessage,
        agent_config: Option<&AgentConfig>,
    ) -> Result<IncomingAudioContext, BotError> {
        // Step 1: without a binding, audio must not wake the bot, but downstream
        // context assembly still receives an initialized audio context.
        let Some(binding) = agent_config.and_then(|agent| agent.audio_transcription.as_ref())
        else {
            tracing::debug!(
                "skipping automatic audio transcription because the selected agent has no audio transcription binding"
            );
            return Ok(IncomingAudioContext {
                saved_audio: Some(Vec::new()),
                expose_audio_to_model: false,
                transcriptions: Vec::new(),
            });
        };
        let transcriber = RoutedAudioTranscriber::new(
            self.audio.clone(),
            binding.provider.clone(),
            binding.model.clone(),
        );
        let keyterms = audio_transcription_default_keyterms(binding);
        let mut transcriptions = Vec::new();

        // Step 2: transcription is per attachment and best-effort. One bad
        // download or provider response should not hide other audio parts.
        for (attachment_index, attachment) in message.attachments.iter().enumerate() {
            if !looks_like_audio_ref(attachment) {
                continue;
            }
            let audio_mime_type = attachment
                .content_type
                .clone()
                .unwrap_or_else(|| "application/octet-stream".to_string());
            let request = AudioTranscriptionRequest {
                audio: UrlMediaRef::new(
                    MediaCategory::Audio,
                    attachment.url.clone(),
                    audio_mime_type.clone(),
                )
                .boxed(),
                language: None,
                keyterms: keyterms.clone(),
                model: None,
            };
            let audio_size_bytes = attachment.size_bytes.unwrap_or(0);
            match transcriber.transcribe_audio(request).await {
                Ok(transcription) => {
                    let result = audio_transcription_model_result_json(&transcription);
                    let trace_response = serde_json::json!({
                        "audio": {
                            "attachment_index": attachment_index,
                            "filename": attachment.filename.as_str(),
                            "mime_type": audio_mime_type,
                            "size_bytes": audio_size_bytes,
                        },
                        "transcription": result,
                    });
                    tracing::info!(
                        attachment_index,
                        duration_seconds = transcription.duration_seconds,
                        text_chars = transcription.text.chars().count(),
                        usage_records = transcription.usage.len(),
                        "automatically transcribed incoming audio"
                    );
                    transcriptions.push(IncomingAudioTranscription {
                        attachment_index,
                        audio_uri: None,
                        text: transcription.text.clone(),
                        language: transcription.language.clone(),
                        duration_seconds: transcription.duration_seconds,
                        result,
                        trace_response,
                        usage: transcription.usage,
                    });
                }
                Err(error) => tracing::warn!(
                    error = %error,
                    attachment_index,
                    "automatic incoming audio transcription failed"
                ),
            }
        }
        // Step 3: a successful transcription replaces the original audio in the
        // model-visible context. If nothing transcribed, keep the audio eligible
        // for normal attachment saving and model exposure.
        Ok(IncomingAudioContext {
            saved_audio: if transcriptions.is_empty() {
                None
            } else {
                Some(Vec::new())
            },
            expose_audio_to_model: transcriptions.is_empty(),
            transcriptions,
        })
    }

    /// Download and persist attachments that match `predicate`.
    ///
    /// The returned refs keep original attachment indexes so later JSON context
    /// injection can annotate the matching platform attachment entries.
    pub(crate) async fn save_matching_attachments(
        &self,
        message: &PlatformMessage,
        category: MediaCategory,
        label: &'static str,
        predicate: fn(&AttachmentRef) -> bool,
    ) -> Vec<StoredAttachmentMedia> {
        let mut out = Vec::new();
        for (attachment_index, attachment) in message.attachments.iter().enumerate() {
            // Step 1: classify attachments by caller-supplied media policy.
            if !predicate(attachment) {
                continue;
            }

            // Step 2: fetch platform-hosted bytes without failing the whole turn
            // when one attachment is unavailable.
            let response = match self.download_http.get(&attachment.url).send().await {
                Ok(response) => response,
                Err(error) => {
                    tracing::warn!(
                        error = %error,
                        filename = %attachment.filename,
                        media_type = label,
                        "failed to download media attachment"
                    );
                    continue;
                }
            };
            let status = response.status();
            if !status.is_success() {
                tracing::warn!(
                    status = status.as_u16(),
                    filename = %attachment.filename,
                    media_type = label,
                    "media attachment download returned non-success status"
                );
                continue;
            }
            let bytes = match response.bytes().await {
                Ok(bytes) => bytes.to_vec(),
                Err(error) => {
                    tracing::warn!(
                        error = %error,
                        filename = %attachment.filename,
                        media_type = label,
                        "failed to read media attachment bytes"
                    );
                    continue;
                }
            };

            // Step 3: hand validated bytes to the configured media store.
            match self
                .media_store
                .create_media(CreateMedia {
                    category: category.clone(),
                    bytes,
                    mime_type: attachment.content_type.clone(),
                    name: None,
                    extension: extension_from_filename(&attachment.filename),
                })
                .await
            {
                Ok(media) => {
                    tracing::info!(
                        uri = %media.uri(),
                        filename = %attachment.filename,
                        media_type = label,
                        "saved media attachment"
                    );
                    out.push(StoredAttachmentMedia {
                        attachment_index,
                        media,
                    });
                }
                Err(error) => tracing::warn!(
                    error = %error,
                    filename = %attachment.filename,
                    media_type = label,
                    "failed to store media attachment"
                ),
            }
        }
        out
    }
}

/// Resolve reply-deliverable media from tool traces into outgoing attachments.
///
/// This handles both generated media and explicit `attach` tool requests. It
/// preserves trace order, deduplicates by URI before loading, and falls back to
/// public URLs for oversized assets when the media store can provide one.
pub(crate) async fn generated_reply_media<M>(
    media_store: &M,
    trace: &[ToolTrace],
) -> GeneratedReplyMedia
where
    M: MediaStore,
{
    let uris = media_uris_from_tool_traces(trace);
    let mut media = GeneratedReplyMedia {
        attachments: Vec::with_capacity(uris.len()),
        public_urls: Vec::new(),
    };

    // Step 1: walk only URIs that came from successful delivery-producing tool
    // traces. Inspection tools such as `read` or `public_url` must not enqueue
    // final reply attachments.
    for uri in uris {
        // Keep generated media delivery best-effort: missing or oversized media
        // should not fail the assistant reply.
        let media_ref = match media_store.media_from_uri(&uri).await {
            Ok(media) => media,
            Err(error) => {
                tracing::warn!(error = %error, uri = %uri, "generated media was not found");
                continue;
            }
        };

        // Step 2: if the store knows the file is too large, avoid loading bytes.
        if media_ref.size_bytes() > MAX_OUTGOING_ATTACHMENT_BYTES as u64 {
            push_oversized_generated_media_url(
                media_ref.as_ref(),
                media_ref.size_bytes(),
                &mut media.public_urls,
            )
            .await;
            continue;
        }

        // Step 3: load bytes only for assets still eligible for direct upload.
        let loaded = match media_ref.load().await {
            Ok(loaded) => loaded,
            Err(error) => {
                tracing::warn!(error = %error, uri = %uri, "failed to load generated media");
                continue;
            }
        };

        // Step 4: re-check the concrete byte length because store metadata can
        // be unavailable, stale, or rounded differently than the loaded body.
        if loaded.bytes.len() > MAX_OUTGOING_ATTACHMENT_BYTES {
            push_oversized_generated_media_url(
                loaded.media.as_ref(),
                loaded.bytes.len() as u64,
                &mut media.public_urls,
            )
            .await;
            continue;
        }
        tracing::debug!(
            uri = %uri,
            filename = loaded.media.name(),
            mime_type = loaded.media.mime_type(),
            bytes = loaded.bytes.len(),
            "prepared generated media attachment"
        );
        media.attachments.push(OutgoingAttachment {
            filename: loaded.media.name().to_string(),
            content_type: loaded.media.mime_type().to_string(),
            bytes: loaded.bytes,
        });
    }
    media
}

/// Append a public URL fallback for an oversized generated/attached asset.
///
/// Missing public URLs are logged and otherwise ignored so the text reply can
/// still be sent.
pub(crate) async fn push_oversized_generated_media_url(
    media: &dyn MediaRef,
    bytes: u64,
    public_urls: &mut Vec<String>,
) {
    match media.public_url().await {
        Ok(public_url) => {
            tracing::warn!(
                uri = %media.uri(),
                bytes,
                limit = MAX_OUTGOING_ATTACHMENT_BYTES,
                public_url = %public_url,
                "generated media exceeds outgoing attachment size limit; using public URL"
            );
            push_unique_string(public_urls, public_url.as_str());
        }
        Err(error) => {
            tracing::warn!(
                error = %error,
                uri = %media.uri(),
                bytes,
                limit = MAX_OUTGOING_ATTACHMENT_BYTES,
                "generated media exceeds outgoing attachment size limit but no public URL is available"
            );
        }
    }
}

/// Push a non-empty string if it has not already appeared.
pub(crate) fn push_unique_string(out: &mut Vec<String>, value: &str) {
    if value.is_empty() || out.iter().any(|seen| seen == value) {
        return;
    }
    out.push(value.to_string());
}

/// Append generated media fallback URLs to assistant text.
///
/// The caller has already stripped inline media references from the model text;
/// this adds only the URLs needed because direct attachment delivery was not
/// possible.
pub(crate) fn append_generated_media_public_urls(
    mut text: String,
    public_urls: &[String],
) -> String {
    if public_urls.is_empty() {
        return text;
    }

    let trimmed_len = text.trim_end().len();
    text.truncate(trimmed_len);
    if !text.is_empty() {
        text.push_str("\n\n");
    }
    if public_urls.len() == 1 {
        text.push_str("Attached media: ");
        text.push_str(&public_urls[0]);
        return text;
    }

    text.push_str("Attached media:\n");
    for public_url in public_urls {
        text.push_str("- ");
        text.push_str(public_url);
        text.push('\n');
    }
    let trimmed_len = text.trim_end().len();
    text.truncate(trimmed_len);
    text
}

/// Extract stored media URIs that should be attached to the final reply.
///
/// Only successful client traces from delivery-producing tools are considered.
/// This prevents media-inspection tools from accidentally turning a mentioned
/// URI into a user-facing attachment.
pub(crate) fn media_uris_from_tool_traces(trace: &[ToolTrace]) -> Vec<MediaUri> {
    let mut seen = Vec::<String>::new();
    let mut out = Vec::new();
    for trace in trace {
        // Step 1: ignore model/server traces and failed client tools.
        let ToolTrace::Client { trace } = trace else {
            continue;
        };
        if trace.result.is_error {
            continue;
        }
        if !tool_trace_delivers_reply_media(trace.call.name.as_str()) {
            continue;
        }

        // Step 2: trust only the tool trace envelope for delivery URIs. Result
        // JSON can mention other refs that are useful for text cleanup but not
        // meant to enqueue attachments.
        let Some(uri) = trace
            .trace_response
            .get("uri")
            .or_else(|| trace.trace_response.get("image_uri"))
            .or_else(|| trace.trace_response.get("video_uri"))
            .and_then(serde_json::Value::as_str)
            .and_then(canonical_stored_media_uri_from_str)
        else {
            continue;
        };

        // Step 3: preserve first-seen order while deduplicating generated media
        // and explicit `attach` calls that reference the same stored asset.
        if seen.iter().any(|seen| seen == uri.as_str()) {
            continue;
        }
        seen.push(uri.to_string());
        out.push(uri);
    }
    out
}

fn canonical_stored_media_uri_from_str(uri: &str) -> Option<MediaUri> {
    canonical_stored_media_uri(&MediaUri::new(uri)).ok()
}

/// Collect media references that should be removed from assistant reply text.
///
/// Unlike `media_uris_from_tool_traces`, this scans both trace responses and
/// JSON tool results because either payload can contain a reference that the
/// model later repeats in final prose.
pub(crate) fn generated_media_reply_refs(trace: &[ToolTrace]) -> Vec<String> {
    let mut out = Vec::new();
    for trace in trace {
        let ToolTrace::Client { trace } = trace else {
            continue;
        };
        if trace.result.is_error {
            continue;
        }
        if !tool_trace_delivers_reply_media(trace.call.name.as_str()) {
            continue;
        }
        collect_generated_media_reply_refs(&trace.trace_response, &mut out);
        if let ClientToolResultContent::Json { value } = &trace.result.content {
            collect_generated_media_reply_refs(value, &mut out);
        }
    }
    out
}

/// Return whether a client tool is allowed to deliver media in the final reply.
///
/// Keep this allowlist narrow: tools that only inspect or expose media to the
/// model must not queue platform attachments.
pub(crate) fn tool_trace_delivers_reply_media(name: &str) -> bool {
    matches!(
        name,
        GENERATE_IMAGE_TOOL | GENERATE_VIDEO_TOOL | ATTACH_ASSET_TOOL
    )
}

/// Add unique URI/public-URL references from a JSON media payload.
pub(crate) fn collect_generated_media_reply_refs(value: &serde_json::Value, out: &mut Vec<String>) {
    for key in ["public_url", "uri", "image_uri", "video_uri"] {
        let Some(reference) = value.get(key).and_then(serde_json::Value::as_str) else {
            continue;
        };
        if reference.is_empty() || out.iter().any(|seen| seen == reference) {
            continue;
        }
        out.push(reference.to_string());
    }
}

/// Return whether a replay asset came from the user/platform side of a turn.
///
/// Platform assets are replayed onto the historical user message. Generated
/// assistant assets are replayed separately so later image-edit requests can
/// still see prior assistant outputs as media context.
pub(crate) fn replay_asset_belongs_to_user_turn(asset: &TurnAsset) -> bool {
    asset.source.starts_with("platform:")
}

/// Return whether a stored asset can be embedded in a model transcript.
pub(crate) fn model_transcript_supports_media(media: &dyn chudbot_api::MediaRef) -> bool {
    matches!(
        media.category(),
        MediaCategory::Image | MediaCategory::Avatar | MediaCategory::GuildIcon
    ) && model_transcript_supports_image_mime_type(media.mime_type())
}

/// Return whether an image MIME type is accepted by transcript media blocks.
pub(crate) fn model_transcript_supports_image_mime_type(mime_type: &str) -> bool {
    let mime_type = mime_type.split(';').next().unwrap_or("").trim();
    MODEL_TRANSCRIPT_IMAGE_MIME_TYPES
        .iter()
        .any(|supported| mime_type.eq_ignore_ascii_case(supported))
}

/// Return whether the `public_url` tool supports a stored asset.
pub(crate) fn public_url_supports_media(media: &dyn chudbot_api::MediaRef) -> bool {
    let mime_type = media
        .mime_type()
        .split(';')
        .next()
        .unwrap_or("")
        .trim()
        .to_ascii_lowercase();
    match media.category() {
        MediaCategory::Image | MediaCategory::Avatar | MediaCategory::GuildIcon => {
            mime_type.starts_with("image/")
        }
        MediaCategory::Video => mime_type.starts_with("video/"),
        MediaCategory::Audio => mime_type.starts_with("audio/"),
        MediaCategory::Other(_) => false,
    }
}

/// Return whether the `attach` tool may queue a stored asset for reply delivery.
pub(crate) fn attach_supports_media(media: &dyn chudbot_api::MediaRef) -> bool {
    matches!(
        media.category(),
        MediaCategory::Image | MediaCategory::Avatar | MediaCategory::GuildIcon
    ) && media
        .mime_type()
        .split(';')
        .next()
        .unwrap_or("")
        .trim()
        .to_ascii_lowercase()
        .starts_with("image/")
}

/// Inject saved audio attachment refs into platform message context JSON.
///
/// The function writes both a top-level list and per-attachment `audio_uri`
/// entries when the platform adapter supplied an `attachments` array. If the
/// payload shape differs, only the top-level list is added.
pub(crate) fn inject_audio_attachment_refs(
    value: &mut serde_json::Value,
    audio_media: &[StoredAttachmentMedia],
) {
    if audio_media.is_empty() {
        return;
    }
    let Some(object) = value.as_object_mut() else {
        return;
    };
    let audio_attachments = audio_media
        .iter()
        .map(|saved| serde_json::Value::String(saved.media.uri().to_string()))
        .collect::<Vec<_>>();

    // Keep the original platform attachment array as the primary anchor when
    // it exists, because attachment indexes come from that array.
    if let Some(attachments) = object
        .get_mut("attachments")
        .and_then(serde_json::Value::as_array_mut)
    {
        for saved in audio_media {
            let Some(attachment) = attachments
                .get_mut(saved.attachment_index)
                .and_then(serde_json::Value::as_object_mut)
            else {
                continue;
            };
            attachment.insert(
                "audio_uri".to_string(),
                serde_json::Value::String(saved.media.uri().to_string()),
            );
        }
    }

    // The top-level list gives providers a stable shape even if a platform's
    // message-context JSON does not expose per-attachment objects.
    object.insert(
        "audio_attachments".to_string(),
        serde_json::Value::Array(audio_attachments),
    );
}

/// Inject automatic audio transcriptions into platform message context JSON.
///
/// Like `inject_audio_attachment_refs`, this annotates the matching attachment
/// entries when possible and always adds a top-level summary list.
pub(crate) fn inject_audio_transcriptions(
    value: &mut serde_json::Value,
    transcriptions: &[IncomingAudioTranscription],
) {
    if transcriptions.is_empty() {
        return;
    }
    let Some(object) = value.as_object_mut() else {
        return;
    };
    let transcription_values = transcriptions
        .iter()
        .map(audio_transcription_context_json)
        .collect::<Vec<_>>();

    // Per-attachment annotations preserve the relationship between each
    // platform attachment and its transcription.
    if let Some(attachments) = object
        .get_mut("attachments")
        .and_then(serde_json::Value::as_array_mut)
    {
        for transcription in transcriptions {
            let Some(attachment) = attachments
                .get_mut(transcription.attachment_index)
                .and_then(serde_json::Value::as_object_mut)
            else {
                continue;
            };
            attachment.insert(
                "audio_transcription".to_string(),
                audio_transcription_context_json(transcription),
            );
        }
    }

    // The top-level list is the stable fallback for platform JSON shapes
    // without an attachments array.
    object.insert(
        "audio_transcriptions".to_string(),
        serde_json::Value::Array(transcription_values),
    );
}

/// Build the compact transcription JSON shown inside message context.
pub(crate) fn audio_transcription_context_json(
    transcription: &IncomingAudioTranscription,
) -> serde_json::Value {
    let mut value = serde_json::json!({
        "attachment_index": transcription.attachment_index,
        "text": transcription.text,
        "language": transcription.language,
        "duration_seconds": transcription.duration_seconds,
    });
    if let Some(audio_uri) = &transcription.audio_uri
        && let Some(object) = value.as_object_mut()
    {
        object.insert(
            "audio_uri".to_string(),
            serde_json::Value::String(audio_uri.clone()),
        );
    }
    value
}

/// Return whether a platform message has any audio-like attachment.
pub(crate) fn message_has_audio_attachments(message: &PlatformMessage) -> bool {
    message.attachments.iter().any(looks_like_audio_ref)
}

/// Return whether no-mention audio preflight should run for an agent.
pub(crate) fn no_mention_audio_preflight_enabled(
    agent_config: Option<&AgentConfig>,
    audio_continues_conversation: bool,
) -> bool {
    no_mention_audio_preflight_enabled_for_binding(
        agent_config.and_then(|agent| agent.audio_transcription.as_ref()),
        audio_continues_conversation,
    )
}

/// Return whether a transcription binding can wake or continue without mention.
///
/// Continuation is allowed only when the caller already knows the platform
/// message belongs to an existing conversation. Otherwise a wake word is
/// required so arbitrary audio-only channel traffic does not start bot turns.
pub(crate) fn no_mention_audio_preflight_enabled_for_binding(
    binding: Option<&TranscriptionBinding>,
    audio_continues_conversation: bool,
) -> bool {
    let Some(binding) = binding else {
        return false;
    };
    audio_continues_conversation || binding.wake_word().is_some()
}

/// Return whether any automatic transcription contains the configured wake word.
pub(crate) fn incoming_audio_mentions_wake_word(
    audio: &IncomingAudioContext,
    wake_word: &str,
) -> bool {
    audio
        .transcriptions
        .iter()
        .any(|transcription| text_mentions_wake_word(&transcription.text, wake_word))
}

/// Match wake words after stripping punctuation, spaces, and case.
pub(crate) fn text_mentions_wake_word(text: &str, wake_word: &str) -> bool {
    let normalized_text = normalize_wake_word_match_text(text);
    let normalized_wake_word = normalize_wake_word_match_text(wake_word);
    !normalized_wake_word.is_empty() && normalized_text.contains(&normalized_wake_word)
}

/// Normalize text for forgiving wake-word matching.
pub(crate) fn normalize_wake_word_match_text(text: &str) -> String {
    text.chars()
        .filter(|ch| ch.is_ascii_alphanumeric())
        .map(|ch| ch.to_ascii_lowercase())
        .collect()
}

/// Append transcription text to the user message content sent into the turn.
///
/// This keeps voice-only messages intelligible to text-first providers and also
/// lets the normal mention/content pipeline operate on transcription output.
pub(crate) fn append_audio_transcriptions_to_message_content(
    content: &mut String,
    transcriptions: &[IncomingAudioTranscription],
) {
    let texts = transcriptions
        .iter()
        .map(|transcription| transcription.text.trim())
        .filter(|text| !text.is_empty())
        .collect::<Vec<_>>();
    if texts.is_empty() {
        return;
    }

    // Label synthetic text so the model can distinguish transcribed voice
    // content from typed user content.
    let transcription_text = if texts.len() == 1 {
        format!("Voice message transcription: {}", texts[0])
    } else {
        let mut out = String::from("Voice message transcriptions:");
        for (index, text) in texts.iter().enumerate() {
            out.push_str(&format!("\n{}. {}", index + 1, text));
        }
        out
    };
    if content.trim().is_empty() {
        *content = transcription_text;
    } else {
        content.push_str("\n\n");
        content.push_str(&transcription_text);
    }
}

/// Classify a platform attachment as image-like using MIME type or extension.
pub(crate) fn looks_like_image_ref(attachment: &AttachmentRef) -> bool {
    looks_like_image(attachment.content_type.as_deref(), &attachment.filename)
}

/// Classify a platform attachment as audio-like using MIME type, extension, or voice flag.
pub(crate) fn looks_like_audio_ref(attachment: &AttachmentRef) -> bool {
    looks_like_audio(
        attachment.content_type.as_deref(),
        &attachment.filename,
        attachment.is_voice_message,
    )
}

/// Return whether attachment metadata looks like an image.
///
/// Filename extensions are a fallback for platforms that omit MIME types.
pub(crate) fn looks_like_image(content_type: Option<&str>, filename: &str) -> bool {
    if content_type
        .map(|content_type| content_type.starts_with("image/"))
        .unwrap_or(false)
    {
        return true;
    }
    matches!(
        extension_from_filename(filename).as_deref(),
        Some("png" | "jpg" | "jpeg" | "gif" | "webp" | "heic" | "heif")
    )
}

/// Return whether attachment metadata looks like audio.
///
/// Voice-message flags win even when the platform omits or mislabels MIME type.
pub(crate) fn looks_like_audio(
    content_type: Option<&str>,
    filename: &str,
    is_voice_message: bool,
) -> bool {
    if is_voice_message {
        return true;
    }
    if content_type
        .map(|content_type| content_type.starts_with("audio/"))
        .unwrap_or(false)
    {
        return true;
    }
    matches!(
        extension_from_filename(filename).as_deref(),
        Some("mp3" | "wav" | "ogg" | "opus" | "m4a" | "aac" | "flac" | "webm")
    )
}

/// Extract a sanitized, lowercase extension from a platform filename.
pub(crate) fn extension_from_filename(filename: &str) -> Option<String> {
    filename
        .rsplit_once('.')
        .map(|(_, extension)| {
            extension
                .chars()
                .filter(|c| c.is_ascii_alphanumeric())
                .collect::<String>()
                .to_ascii_lowercase()
        })
        .filter(|extension| !extension.is_empty())
}
