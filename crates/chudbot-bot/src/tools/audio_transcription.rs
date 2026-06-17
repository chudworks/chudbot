//! `transcribe_audio` client tool.

use super::*;

/// Tool wrapper for transcribing stored audio media through the configured provider.
pub(crate) struct AudioTranscriptionTool<T, M> {
    pub(crate) transcriber: T,
    pub(crate) media_store: M,
    pub(crate) default_keyterms: Vec<String>,
    pub(crate) description: String,
}

impl<T, M> AudioTranscriptionTool<T, M> {
    pub(crate) fn new(transcriber: T, media_store: M) -> Self {
        Self {
            transcriber,
            media_store,
            default_keyterms: Vec::new(),
            description: "Transcribe a stored audio attachment and return its speech as text."
                .to_string(),
        }
    }

    pub(crate) fn with_default_keyterms(mut self, keyterms: Vec<String>) -> Self {
        self.default_keyterms = keyterms;
        self
    }

    pub(crate) fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = description.into();
        self
    }
}

impl<T, M> AudioTranscriptionTool<T, M>
where
    T: AudioTranscriber,
    M: MediaStore,
{
    pub(crate) fn spec(&self) -> ClientToolSpec {
        ClientToolSpec {
            description: self.description.clone(),
            input_schema: audio_transcription_tool_schema(),
        }
    }

    #[tracing::instrument(
        name = "tool.transcribe_audio",
        skip_all,
        fields(tool_call = %call.id)
    )]
    pub(crate) async fn call(
        &self,
        call: ClientToolCall,
    ) -> Result<ClientToolOutput, BotToolError> {
        let mut request =
            audio_transcription_request_from_tool_input(&self.media_store, call.input).await?;
        append_default_audio_keyterms(&mut request.keyterms, &self.default_keyterms);
        let audio_uri = request.audio.uri().to_string();
        let audio_mime_type = request.audio.mime_type().to_string();
        let audio_size_bytes = request.audio.size_bytes();
        tracing::debug!(
            audio_uri = %audio_uri,
            audio_mime_type = %audio_mime_type,
            audio_size_bytes,
            language = ?request.language.as_deref(),
            keyterms = request.keyterms.len(),
            model = ?request.model.as_ref().map(ModelId::as_str),
            "parsed audio transcription request"
        );
        let transcription = self
            .transcriber
            .transcribe_audio(request)
            .await
            .map_err(|error| {
                tracing::warn!(error = %error, "audio transcription failed");
                BotToolError::Generator(error.to_string())
            })?;
        let result = audio_transcription_model_result_json(&transcription);
        let trace_response = serde_json::json!({
            "audio": {
                "uri": audio_uri,
                "mime_type": audio_mime_type,
                "size_bytes": audio_size_bytes,
            },
            "transcription": result,
        });
        tracing::info!(
            duration_seconds = transcription.duration_seconds,
            text_chars = transcription.text.chars().count(),
            usage_records = transcription.usage.len(),
            "audio transcription tool completed"
        );

        Ok(ClientToolOutput {
            result: ClientToolResultContent::Json {
                value: result.clone(),
            },
            media: Vec::new(),
            is_error: false,
            trace_response,
            usage: transcription.usage,
        })
    }
}

pub(crate) async fn audio_transcription_request_from_tool_input<M>(
    media_store: &M,
    input: serde_json::Value,
) -> Result<AudioTranscriptionRequest, BotToolError>
where
    M: MediaStore,
{
    let audio_value = input
        .get("audio_uri")
        .or_else(|| input.get("audio"))
        .ok_or_else(|| BotToolError::InvalidInput("`audio_uri` is required".to_string()))?;
    let audio = resolve_tool_media_arg(media_store, MediaCategory::Audio, audio_value).await?;
    let keyterms = match tool_optional_string_list(&input, "keyterm")? {
        Some(keyterms) => keyterms,
        None => tool_optional_string_list(&input, "keyterms")?.unwrap_or_default(),
    };
    Ok(AudioTranscriptionRequest {
        audio,
        language: tool_optional_string(&input, "language")?,
        keyterms,
        model: tool_optional_string(&input, "model")?.map(ModelId::new),
    })
}

pub(crate) fn audio_transcription_model_result_json(
    transcription: &AudioTranscription,
) -> serde_json::Value {
    serde_json::json!({
        "text": transcription.text,
        "language": transcription.language,
        "duration_seconds": transcription.duration_seconds,
        "words": transcription.words,
        "channels": transcription.channels,
        "model": transcription.model.as_ref().map(ModelId::as_str),
    })
}

pub(crate) fn audio_transcription_tool_schema() -> ToolInputSchema {
    ToolInputSchema::new(serde_json::json!({
        "type": "object",
        "required": ["audio_uri"],
        "properties": {
            "audio_uri": {
                "type": "string",
                "description": "A file://audio/... URI from the message JSON audio_attachments or attachment audio_uri field."
            },
            "audio": {
                "type": "string",
                "description": "Alias for audio_uri."
            },
            "language": {
                "type": "string",
                "description": "Optional language code such as en, fr, de, or ja for text formatting."
            },
            "keyterm": {
                "oneOf": [
                    { "type": "string" },
                    { "type": "array", "items": { "type": "string" } }
                ],
                "description": "Optional key term or terms to bias transcription toward."
            },
            "keyterms": {
                "type": "array",
                "items": { "type": "string" },
                "description": "Alias for keyterm when passing multiple terms."
            },
            "model": {
                "type": "string",
                "description": "Optional provider-specific transcription model id."
            }
        },
        "additionalProperties": false
    }))
}
