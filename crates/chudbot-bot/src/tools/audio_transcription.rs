//! `transcribe_audio` client tool.
//!
//! The tool accepts a stored audio media URI, resolves it through the configured
//! media store, and passes the resulting media reference to an audio
//! transcriber. The model sees transcript metadata in the tool result while
//! provider usage records stay on the `ClientToolOutput` for turn accounting.

use super::*;

/// Tool wrapper for transcribing stored audio media through the configured provider.
///
/// The generic parameters keep the executor on static dispatch while still
/// allowing tests to supply fake transcribers and media stores.
pub(crate) struct AudioTranscriptionTool<T, M> {
    /// Provider implementation that performs speech-to-text.
    pub(crate) transcriber: T,
    /// Store used to resolve the model-supplied media URI into verified audio.
    pub(crate) media_store: M,
    /// Deployment-level keyterms appended to every request after input parsing.
    pub(crate) default_keyterms: Vec<String>,
    /// Client-visible tool description.
    pub(crate) description: String,
}

impl<T, M> AudioTranscriptionTool<T, M> {
    /// Creates a transcription tool with no default keyterms.
    pub(crate) fn new(transcriber: T, media_store: M) -> Self {
        Self {
            transcriber,
            media_store,
            default_keyterms: Vec::new(),
            description: "Transcribe a stored audio attachment and return its speech as text."
                .to_string(),
        }
    }

    /// Adds provider-level keyterms that should bias every transcription.
    pub(crate) fn with_default_keyterms(mut self, keyterms: Vec<String>) -> Self {
        self.default_keyterms = keyterms;
        self
    }

    /// Overrides the default client-visible tool description.
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
    /// Builds the tool spec sent to model providers.
    pub(crate) fn spec(&self) -> ClientToolSpec {
        ClientToolSpec {
            description: self.description.clone(),
            input_schema: audio_transcription_tool_schema(),
        }
    }

    /// Executes one `transcribe_audio` call from validation through provider output.
    ///
    /// The tool result is JSON-only; it does not attach media to the next model
    /// step or the final platform reply. Usage records reported by the provider
    /// are forwarded unchanged on the output.
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
        // Deployment defaults supplement user-supplied keyterms without
        // changing the input validation rules below.
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
        // Keep the model-facing result focused on transcription data. The trace
        // also records the resolved source audio for viewer/debugging context.
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
            // Provider usage is not included in `result`; the turn runner reads
            // this field for accounting and storage.
            usage: transcription.usage,
        })
    }
}

/// Converts raw tool input into a provider transcription request.
///
/// `audio_uri` is the primary input field and `audio` is an alias. The selected
/// value is resolved as stored audio through `MediaStore`; this is the trust
/// boundary that rejects unsupported categories and invalid stored-media
/// references before any provider request is built. Optional language,
/// keyterms, and model fields are validated with the shared tool helpers.
pub(crate) async fn audio_transcription_request_from_tool_input<M>(
    media_store: &M,
    input: serde_json::Value,
) -> Result<AudioTranscriptionRequest, BotToolError>
where
    M: MediaStore,
{
    // Require a stored-media reference, accepting the short alias for provider
    // compatibility while keeping the error message on the canonical field.
    let audio_value = input
        .get("audio_uri")
        .or_else(|| input.get("audio"))
        .ok_or_else(|| BotToolError::InvalidInput("`audio_uri` is required".to_string()))?;
    // Resolution loads the stored media metadata/reference and enforces the
    // audio category before the transcriber receives the request.
    let audio = resolve_tool_media_arg(media_store, MediaCategory::Audio, audio_value).await?;
    // `keyterm` accepts a single string or list. `keyterms` is only consulted
    // when the singular field is absent so existing prompts keep precedence.
    let keyterms = match tool_optional_string_list(&input, "keyterm")? {
        Some(keyterms) => keyterms,
        None => tool_optional_string_list(&input, "keyterms")?.unwrap_or_default(),
    };
    // At this point the request is provider-neutral: it contains resolved media
    // plus optional hints, leaving provider-specific serialization downstream.
    Ok(AudioTranscriptionRequest {
        audio,
        language: tool_optional_string(&input, "language")?,
        keyterms,
        model: tool_optional_string(&input, "model")?.map(ModelId::new),
    })
}

/// Shapes the successful transcription payload returned to the model.
///
/// Usage and source-audio metadata are intentionally omitted here. Usage travels
/// on `ClientToolOutput::usage`, and the source audio is recorded in the trace
/// response built by `AudioTranscriptionTool::call`.
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

/// Returns the JSON schema used to validate model-supplied transcription input.
///
/// The schema rejects unknown fields and documents the aliases accepted by
/// `audio_transcription_request_from_tool_input`; detailed type coercion and
/// media validation still happen in the request parser.
pub(crate) fn audio_transcription_tool_schema() -> ToolInputSchema {
    ToolInputSchema::object([
        ToolInputField::required(
            "audio_uri",
            ToolInputValueSchema::string().description(
                "A file://audio/... URI from the message JSON audio_attachments or attachment audio_uri field.",
            ),
        ),
        ToolInputField::optional(
            "language",
            ToolInputValueSchema::string()
                .description("Optional language code such as en, fr, de, or ja for text formatting."),
        ),
        ToolInputField::optional(
            "keyterms",
            ToolInputValueSchema::array(ToolInputValueSchema::string())
                .description("Optional key terms to bias transcription toward."),
        ),
        ToolInputField::optional(
            "model",
            ToolInputValueSchema::string()
                .description("Optional provider-specific transcription model id."),
        ),
    ])
}
