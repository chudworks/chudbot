//! Media attachment ingestion, generated-media delivery, and audio preflight helpers.

use crate::prelude::*;
use crate::*;

/// Input needed to turn a platform message into model context.
pub(crate) struct MessageContextInput<'a> {
    pub(crate) kind: &'a str,
    pub(crate) message: &'a PlatformMessage,
    pub(crate) relationship: PlatformMessageRelationship,
    pub(crate) saved_audio: Option<Vec<StoredAttachmentMedia>>,
    pub(crate) audio_transcriptions: &'a [IncomingAudioTranscription],
}

/// Saved audio/transcription state produced before deciding whether to run a turn.
#[derive(Debug, Default)]
pub(crate) struct IncomingAudioContext {
    pub(crate) saved_audio: Option<Vec<StoredAttachmentMedia>>,
    pub(crate) expose_audio_to_model: bool,
    pub(crate) transcriptions: Vec<IncomingAudioTranscription>,
}

impl IncomingAudioContext {
    pub(crate) fn usage_records(&self) -> Vec<UsageRecord> {
        self.transcriptions
            .iter()
            .flat_map(|transcription| transcription.usage.iter().cloned())
            .collect()
    }

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
    pub(crate) attachment_index: usize,
    pub(crate) audio_uri: Option<String>,
    pub(crate) text: String,
    pub(crate) language: Option<String>,
    pub(crate) duration_seconds: f64,
    pub(crate) result: serde_json::Value,
    pub(crate) trace_response: serde_json::Value,
    pub(crate) usage: Vec<UsageRecord>,
}

impl IncomingAudioTranscription {
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

/// Media saved from a platform attachment and its original attachment index.
#[derive(Debug)]
pub(crate) struct StoredAttachmentMedia {
    pub(crate) attachment_index: usize,
    pub(crate) media: chudbot_api::BoxedMediaRef,
}

impl<R> BotRuntime<R>
where
    R: BotRuntimeTypes + 'static,
{
    pub(crate) async fn prepare_incoming_audio_context(
        &self,
        message: &PlatformMessage,
        agent_config: Option<&AgentConfig>,
    ) -> Result<IncomingAudioContext, BotError> {
        let Some(binding) = agent_config.and_then(|agent| agent.audio_transcription.as_ref())
        else {
            tracing::debug!(
                "skipping automatic audio transcription because the selected agent has no audio transcription binding"
            );
            // No binding means the audio should not wake the bot, but image/audio
            // attachment saving later still expects an initialized context.
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
        // If transcription succeeded, the text is appended to the user message and
        // the original audio is hidden from the model. If no transcription was
        // produced, the audio attachment can still be exposed as media context.
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

    pub(crate) async fn save_matching_attachments(
        &self,
        message: &PlatformMessage,
        category: MediaCategory,
        label: &'static str,
        predicate: fn(&AttachmentRef) -> bool,
    ) -> Vec<StoredAttachmentMedia> {
        let mut out = Vec::new();
        for (attachment_index, attachment) in message.attachments.iter().enumerate() {
            if !predicate(attachment) {
                continue;
            }
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

/// Generated media ready to attach to the platform reply plus URL fallbacks.
pub(crate) struct GeneratedReplyMedia {
    pub(crate) attachments: Vec<OutgoingAttachment>,
    pub(crate) public_urls: Vec<String>,
}

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
        if media_ref.size_bytes() > MAX_OUTGOING_ATTACHMENT_BYTES as u64 {
            push_oversized_generated_media_url(
                media_ref.as_ref(),
                media_ref.size_bytes(),
                &mut media.public_urls,
            )
            .await;
            continue;
        }
        let loaded = match media_ref.load().await {
            Ok(loaded) => loaded,
            Err(error) => {
                tracing::warn!(error = %error, uri = %uri, "failed to load generated media");
                continue;
            }
        };
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

pub(crate) fn media_uris_from_tool_traces(trace: &[ToolTrace]) -> Vec<MediaUri> {
    let mut seen = Vec::<String>::new();
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
        let Some(uri) = trace
            .trace_response
            .get("uri")
            .or_else(|| trace.trace_response.get("image_uri"))
            .or_else(|| trace.trace_response.get("video_uri"))
            .and_then(serde_json::Value::as_str)
            .filter(|uri| uri.starts_with("file://"))
        else {
            continue;
        };
        if seen.iter().any(|seen| seen == uri) {
            continue;
        }
        seen.push(uri.to_string());
        out.push(MediaUri::new(uri));
    }
    out
}

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

pub(crate) fn tool_trace_delivers_reply_media(name: &str) -> bool {
    matches!(
        name,
        GENERATE_IMAGE_TOOL | GENERATE_VIDEO_TOOL | ATTACH_ASSET_TOOL
    )
}

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

pub(crate) fn replay_asset_belongs_to_user_turn(asset: &TurnAsset) -> bool {
    asset.source.starts_with("platform:")
}

pub(crate) fn push_unique_string(out: &mut Vec<String>, value: &str) {
    if value.is_empty() || out.iter().any(|seen| seen == value) {
        return;
    }
    out.push(value.to_string());
}

pub(crate) fn model_transcript_supports_media(media: &dyn chudbot_api::MediaRef) -> bool {
    matches!(media.category(), MediaCategory::Image)
        && model_transcript_supports_image_mime_type(media.mime_type())
}

pub(crate) fn model_transcript_supports_image_mime_type(mime_type: &str) -> bool {
    let mime_type = mime_type.split(';').next().unwrap_or("").trim();
    MODEL_TRANSCRIPT_IMAGE_MIME_TYPES
        .iter()
        .any(|supported| mime_type.eq_ignore_ascii_case(supported))
}

pub(crate) fn public_url_supports_media(media: &dyn chudbot_api::MediaRef) -> bool {
    let mime_type = media
        .mime_type()
        .split(';')
        .next()
        .unwrap_or("")
        .trim()
        .to_ascii_lowercase();
    match media.category() {
        MediaCategory::Image | MediaCategory::Avatar => mime_type.starts_with("image/"),
        MediaCategory::Video => mime_type.starts_with("video/"),
        MediaCategory::Audio => mime_type.starts_with("audio/"),
        MediaCategory::Other(_) => false,
    }
}

pub(crate) fn attach_supports_media(media: &dyn chudbot_api::MediaRef) -> bool {
    matches!(media.category(), MediaCategory::Image)
        && media
            .mime_type()
            .split(';')
            .next()
            .unwrap_or("")
            .trim()
            .to_ascii_lowercase()
            .starts_with("image/")
}

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
    object.insert(
        "audio_attachments".to_string(),
        serde_json::Value::Array(audio_attachments),
    );
}

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
    object.insert(
        "audio_transcriptions".to_string(),
        serde_json::Value::Array(transcription_values),
    );
}

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

pub(crate) fn message_has_audio_attachments(message: &PlatformMessage) -> bool {
    message.attachments.iter().any(looks_like_audio_ref)
}

pub(crate) fn no_mention_audio_preflight_enabled(
    agent_config: Option<&AgentConfig>,
    audio_continues_conversation: bool,
) -> bool {
    no_mention_audio_preflight_enabled_for_binding(
        agent_config.and_then(|agent| agent.audio_transcription.as_ref()),
        audio_continues_conversation,
    )
}

pub(crate) fn no_mention_audio_preflight_enabled_for_binding(
    binding: Option<&TranscriptionBinding>,
    audio_continues_conversation: bool,
) -> bool {
    let Some(binding) = binding else {
        return false;
    };
    audio_continues_conversation || binding.wake_word().is_some()
}

pub(crate) fn incoming_audio_mentions_wake_word(
    audio: &IncomingAudioContext,
    wake_word: &str,
) -> bool {
    audio
        .transcriptions
        .iter()
        .any(|transcription| text_mentions_wake_word(&transcription.text, wake_word))
}

pub(crate) fn text_mentions_wake_word(text: &str, wake_word: &str) -> bool {
    let normalized_text = normalize_wake_word_match_text(text);
    let normalized_wake_word = normalize_wake_word_match_text(wake_word);
    !normalized_wake_word.is_empty() && normalized_text.contains(&normalized_wake_word)
}

pub(crate) fn normalize_wake_word_match_text(text: &str) -> String {
    text.chars()
        .filter(|ch| ch.is_ascii_alphanumeric())
        .map(|ch| ch.to_ascii_lowercase())
        .collect()
}

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

pub(crate) fn looks_like_image_ref(attachment: &AttachmentRef) -> bool {
    looks_like_image(attachment.content_type.as_deref(), &attachment.filename)
}

pub(crate) fn looks_like_audio_ref(attachment: &AttachmentRef) -> bool {
    looks_like_audio(
        attachment.content_type.as_deref(),
        &attachment.filename,
        attachment.is_voice_message,
    )
}

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

pub(crate) fn avatar_media_name(user: &UserProfile, url: &str) -> String {
    let tail = url
        .split('?')
        .next()
        .and_then(|url| url.rsplit('/').next())
        .unwrap_or("avatar.png");
    let stem = tail.strip_suffix(".png").unwrap_or(tail);
    let stem = if url.contains("/embed/avatars/") {
        format!("default{stem}")
    } else {
        stem.to_string()
    };
    format!(
        "{}_{}.png",
        user.id.user_id.as_str(),
        safe_media_name_part(&stem)
    )
}

pub(crate) fn safe_media_name_part(input: &str) -> String {
    let out = input
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_'))
        .collect::<String>();
    if out.is_empty() {
        "avatar".to_string()
    } else {
        out
    }
}
