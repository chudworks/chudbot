//! Client-tool adapters for media generators.

use std::time::Duration;

use thiserror::Error;

use crate::ids::{ModelId, VideoJobId};
use crate::tool::{
    ClientToolCall, ClientToolOutput, ClientToolResultContent, ClientToolSpec, ToolInputSchema,
};

use super::{
    BoxedMediaRef, CreateMedia, GeneratedImage, ImageGenerator, ImageRequest, MediaCategory,
    MediaError, MediaRef, MediaStore, MediaUri, PublicMediaUrl, UrlMediaRef, VideoGenerator,
    VideoJobStatus, VideoRequest,
};

const MAX_REFERENCE_IMAGES: usize = 3;
const MAX_VIDEO_DURATION_SECONDS: u8 = 15;

/// Adapter exposing any [`ImageGenerator`] as a client-side tool.
#[derive(Debug)]
pub struct ImageGeneratorTool<G, S> {
    generator: G,
    media_store: S,
    description: String,
}

impl<G, S> ImageGeneratorTool<G, S> {
    /// Build an image-generation client tool.
    pub fn new(generator: G, media_store: S) -> Self {
        Self {
            generator,
            media_store,
            description: concat!(
                "Generate an image, edit/restyle/combine existing images when reference URIs ",
                "are available, save it to media storage, and return its media URI."
            )
            .to_string(),
        }
    }

    /// Override the tool description shown to the model.
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = description.into();
        self
    }
}

impl<G, S> ImageGeneratorTool<G, S>
where
    G: ImageGenerator,
    S: MediaStore,
{
    /// Tool specification shown to the model.
    pub fn spec(&self) -> ClientToolSpec {
        ClientToolSpec {
            description: self.description.clone(),
            input_schema: image_tool_schema(),
        }
    }

    /// Execute one image-generation call.
    #[tracing::instrument(
        name = "media_tool.generate_image",
        skip_all,
        fields(tool = %call.name, tool_use_id = %call.id)
    )]
    pub async fn call(
        &self,
        call: ClientToolCall,
    ) -> Result<ClientToolOutput, MediaToolError<G::Error>> {
        let request = image_request_from_tool_input(&self.media_store, call.input).await?;
        tracing::debug!(
            prompt_chars = request.prompt.chars().count(),
            references = request.references.len(),
            aspect_ratio = ?request.aspect_ratio.as_deref(),
            model = ?request.model.as_ref().map(|model| model.as_str()),
            "parsed image tool request"
        );
        let generated = match self.generator.generate_image(request).await {
            Ok(generated) => generated,
            Err(error) => {
                tracing::warn!(error = %error, "image generator failed");
                return Err(MediaToolError::Generator(error));
            }
        };
        let GeneratedImage {
            bytes,
            mime_type,
            model,
            revised_prompt,
            usage,
        } = generated;
        tracing::info!(
            model = %model,
            mime_type = %mime_type,
            bytes = bytes.len(),
            revised_prompt = revised_prompt.is_some(),
            usage_records = usage.len(),
            "image generated"
        );
        let media = match self
            .media_store
            .create_media(CreateMedia {
                category: MediaCategory::Image,
                bytes,
                mime_type: Some(mime_type),
                name: None,
                extension: None,
            })
            .await
        {
            Ok(media) => media,
            Err(error) => {
                tracing::warn!(error = %error, "failed to store generated image");
                return Err(MediaToolError::Media(error));
            }
        };
        let public_url = media.public_url().await.ok();
        let trace_response = media_result_json(
            media.as_ref(),
            public_url.as_ref(),
            serde_json::json!({
                "model": model.as_str(),
                "revised_prompt": revised_prompt,
            }),
        );
        let result = model_media_result_json(
            media.as_ref(),
            serde_json::json!({
                "model": model.as_str(),
                "revised_prompt": revised_prompt,
            }),
        );
        tracing::info!(
            uri = %media.uri(),
            name = media.name(),
            bytes = media.size_bytes(),
            has_public_url = public_url.is_some(),
            "image tool completed"
        );

        Ok(ClientToolOutput {
            result: ClientToolResultContent::Json {
                value: result.clone(),
            },
            media: Vec::new(),
            is_error: false,
            trace_response,
            usage,
        })
    }
}

/// Extension methods for turning image generators into client tools.
pub trait ImageGeneratorToolExt: ImageGenerator + Sized {
    /// Wrap this generator as an image-generation client tool.
    fn into_image_tool<S>(self, media_store: S) -> ImageGeneratorTool<Self, S>
    where
        S: MediaStore,
    {
        ImageGeneratorTool::new(self, media_store)
    }
}

impl<T> ImageGeneratorToolExt for T where T: ImageGenerator + Sized {}

/// Adapter exposing any [`VideoGenerator`] as a client-side tool.
#[derive(Debug)]
pub struct VideoGeneratorTool<G, S> {
    generator: G,
    media_store: S,
    description: String,
    poll_interval: Duration,
    max_polls: u32,
}

impl<G, S> VideoGeneratorTool<G, S> {
    /// Build a video-generation client tool.
    pub fn new(generator: G, media_store: S) -> Self {
        Self {
            generator,
            media_store,
            description: "Generate a video, save it to media storage, and return its media URI."
                .to_string(),
            poll_interval: Duration::from_secs(2),
            max_polls: 600,
        }
    }

    /// Override the tool description shown to the model.
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = description.into();
        self
    }

    /// Configure video polling.
    pub fn with_polling(mut self, poll_interval: Duration, max_polls: u32) -> Self {
        self.poll_interval = poll_interval;
        self.max_polls = max_polls;
        self
    }
}

impl<G, S> VideoGeneratorTool<G, S>
where
    G: VideoGenerator,
    S: MediaStore,
{
    /// Tool specification shown to the model.
    pub fn spec(&self) -> ClientToolSpec {
        ClientToolSpec {
            description: self.description.clone(),
            input_schema: video_tool_schema(),
        }
    }

    /// Execute one video-generation call.
    #[tracing::instrument(
        name = "media_tool.generate_video",
        skip_all,
        fields(tool = %call.name, tool_use_id = %call.id, max_polls = self.max_polls)
    )]
    pub async fn call(
        &self,
        call: ClientToolCall,
    ) -> Result<ClientToolOutput, MediaToolError<G::Error>> {
        let request = video_request_from_tool_input(&self.media_store, call.input).await?;
        tracing::debug!(
            prompt_chars = request.prompt.chars().count(),
            has_image = request.image.is_some(),
            duration_seconds = ?request.duration_seconds,
            aspect_ratio = ?request.aspect_ratio.as_deref(),
            resolution = ?request.resolution.as_deref(),
            model = ?request.model.as_ref().map(|model| model.as_str()),
            "parsed video tool request"
        );
        let job_id = match self.generator.submit_video(request).await {
            Ok(job_id) => job_id,
            Err(error) => {
                tracing::warn!(error = %error, "video submit failed");
                return Err(MediaToolError::Generator(error));
            }
        };
        tracing::info!(job_id = %job_id, "video submitted");

        for poll in 0..self.max_polls {
            match self
                .generator
                .check_video(job_id.clone())
                .await
                .map_err(|error| {
                    tracing::warn!(
                        job_id = %job_id,
                        poll = poll + 1,
                        error = %error,
                        "video status check failed"
                    );
                    MediaToolError::Generator(error)
                })? {
                VideoJobStatus::Pending => {
                    tracing::debug!(
                        job_id = %job_id,
                        poll = poll + 1,
                        max_polls = self.max_polls,
                        "video still pending"
                    );
                    if poll + 1 < self.max_polls {
                        tokio::time::sleep(self.poll_interval).await;
                    }
                }
                VideoJobStatus::Done { meta } => {
                    tracing::info!(
                        job_id = %job_id,
                        poll = poll + 1,
                        duration_seconds = meta.duration_seconds,
                        usage_records = meta.usage.len(),
                        "video generation completed upstream"
                    );
                    let bytes = self
                        .generator
                        .download_video(meta.url.clone())
                        .await
                        .map_err(|error| {
                            tracing::warn!(
                                job_id = %job_id,
                                error = %error,
                                "video download failed"
                            );
                            MediaToolError::Generator(error)
                        })?;
                    tracing::debug!(
                        job_id = %job_id,
                        bytes = bytes.len(),
                        "video downloaded"
                    );
                    let media = match self
                        .media_store
                        .create_media(CreateMedia {
                            category: MediaCategory::Video,
                            bytes,
                            mime_type: None,
                            name: None,
                            extension: None,
                        })
                        .await
                    {
                        Ok(media) => media,
                        Err(error) => {
                            tracing::warn!(
                                job_id = %job_id,
                                error = %error,
                                "failed to store generated video"
                            );
                            return Err(MediaToolError::Media(error));
                        }
                    };
                    let public_url = media.public_url().await.ok();
                    let trace_response = media_result_json(
                        media.as_ref(),
                        public_url.as_ref(),
                        serde_json::json!({
                            "provider_job_id": job_id.as_str(),
                            "download_url": meta.url,
                            "duration_seconds": meta.duration_seconds,
                        }),
                    );
                    let result = model_media_result_json(
                        media.as_ref(),
                        serde_json::json!({
                            "provider_job_id": job_id.as_str(),
                            "duration_seconds": meta.duration_seconds,
                        }),
                    );
                    tracing::info!(
                        job_id = %job_id,
                        uri = %media.uri(),
                        name = media.name(),
                        bytes = media.size_bytes(),
                        has_public_url = public_url.is_some(),
                        "video tool completed"
                    );

                    return Ok(ClientToolOutput {
                        result: ClientToolResultContent::Json {
                            value: result.clone(),
                        },
                        media: Vec::new(),
                        is_error: false,
                        trace_response,
                        usage: meta.usage,
                    });
                }
                VideoJobStatus::Failed { message } => {
                    tracing::warn!(
                        job_id = %job_id,
                        poll = poll + 1,
                        message = %message,
                        "video generation failed upstream"
                    );
                    return Err(MediaToolError::VideoFailed(message));
                }
                VideoJobStatus::Expired => {
                    tracing::warn!(
                        job_id = %job_id,
                        poll = poll + 1,
                        "video generation expired upstream"
                    );
                    return Err(MediaToolError::VideoExpired);
                }
            }
        }

        tracing::warn!(
            job_id = %job_id,
            polls = self.max_polls,
            "video remained pending after polling budget"
        );
        Err(MediaToolError::VideoPending {
            job_id,
            polls: self.max_polls,
        })
    }
}

/// Extension methods for turning video generators into client tools.
pub trait VideoGeneratorToolExt: VideoGenerator + Sized {
    /// Wrap this generator as a video-generation client tool.
    fn into_video_tool<S>(self, media_store: S) -> VideoGeneratorTool<Self, S>
    where
        S: MediaStore,
    {
        VideoGeneratorTool::new(self, media_store)
    }
}

impl<T> VideoGeneratorToolExt for T where T: VideoGenerator + Sized {}

/// Errors from image/video client-tool adapters.
#[derive(Debug, Error)]
pub enum MediaToolError<GE>
where
    GE: std::error::Error + Send + Sync + 'static,
{
    /// Invalid model-supplied tool input.
    #[error("invalid tool input: {0}")]
    InvalidInput(String),
    /// Media generation failed.
    #[error("generator error: {0}")]
    Generator(#[source] GE),
    /// Media storage failed.
    #[error("media storage error: {0}")]
    Media(#[source] MediaError),
    /// Video generation failed upstream.
    #[error("video generation failed: {0}")]
    VideoFailed(String),
    /// Video generation expired upstream.
    #[error("video generation expired")]
    VideoExpired,
    /// Video generation did not finish within the tool polling budget.
    #[error("video generation still pending after {polls} polls: {job_id}")]
    VideoPending {
        /// Provider job id.
        job_id: VideoJobId,
        /// Poll attempts.
        polls: u32,
    },
}

async fn image_request_from_tool_input<S, GE>(
    media_store: &S,
    input: serde_json::Value,
) -> Result<ImageRequest, MediaToolError<GE>>
where
    S: MediaStore,
    GE: std::error::Error + Send + Sync + 'static,
{
    let prompt = required_string(&input, "prompt")?;
    let mut references = Vec::new();
    if let Some(value) = input
        .get("reference_images")
        .or_else(|| input.get("references"))
    {
        let values = value.as_array().ok_or_else(|| {
            MediaToolError::InvalidInput("`reference_images` must be an array".to_string())
        })?;
        if values.len() > MAX_REFERENCE_IMAGES {
            return Err(MediaToolError::InvalidInput(format!(
                "`reference_images` must contain at most {MAX_REFERENCE_IMAGES} items"
            )));
        }
        for value in values {
            references.push(resolve_media_arg(media_store, MediaCategory::Image, value).await?);
        }
    }

    Ok(ImageRequest {
        prompt,
        references,
        aspect_ratio: optional_string(&input, "aspect_ratio")?,
        model: optional_model(&input, "model")?,
    })
}

async fn video_request_from_tool_input<S, GE>(
    media_store: &S,
    input: serde_json::Value,
) -> Result<VideoRequest, MediaToolError<GE>>
where
    S: MediaStore,
    GE: std::error::Error + Send + Sync + 'static,
{
    let prompt = required_string(&input, "prompt")?;
    let image = match input.get("image").or_else(|| input.get("image_url")) {
        Some(value) => Some(resolve_media_arg(media_store, MediaCategory::Image, value).await?),
        None => None,
    };

    Ok(VideoRequest {
        prompt,
        image,
        duration_seconds: optional_bounded_u8(
            &input,
            "duration_seconds",
            MAX_VIDEO_DURATION_SECONDS,
        )?,
        aspect_ratio: optional_string(&input, "aspect_ratio")?,
        resolution: optional_string(&input, "resolution")?,
        model: optional_model(&input, "model")?,
    })
}

async fn resolve_media_arg<S, GE>(
    media_store: &S,
    category: MediaCategory,
    value: &serde_json::Value,
) -> Result<BoxedMediaRef, MediaToolError<GE>>
where
    S: MediaStore,
    GE: std::error::Error + Send + Sync + 'static,
{
    let text = value.as_str().ok_or_else(|| {
        MediaToolError::InvalidInput("media references must be strings".to_string())
    })?;
    if text.starts_with("http://") || text.starts_with("https://") {
        tracing::debug!(
            category = ?category,
            reference_kind = "public_url",
            "resolved media argument"
        );
        return Ok(UrlMediaRef::new(category, text, "application/octet-stream").boxed());
    }

    tracing::debug!(
        category = ?category,
        reference_kind = "media_uri",
        uri = %text,
        "resolving media argument"
    );
    let media = media_store
        .media_from_uri(&MediaUri::new(text))
        .await
        .map_err(MediaToolError::Media)?;
    tracing::debug!(
        category = ?media.category(),
        uri = %media.uri(),
        name = media.name(),
        "resolved media argument"
    );
    Ok(media)
}

fn required_string<GE>(input: &serde_json::Value, field: &str) -> Result<String, MediaToolError<GE>>
where
    GE: std::error::Error + Send + Sync + 'static,
{
    input
        .get(field)
        .and_then(serde_json::Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .map(str::to_string)
        .ok_or_else(|| MediaToolError::InvalidInput(format!("`{field}` is required")))
}

fn optional_string<GE>(
    input: &serde_json::Value,
    field: &str,
) -> Result<Option<String>, MediaToolError<GE>>
where
    GE: std::error::Error + Send + Sync + 'static,
{
    let Some(value) = input.get(field) else {
        return Ok(None);
    };
    value
        .as_str()
        .map(str::to_string)
        .map(Some)
        .ok_or_else(|| MediaToolError::InvalidInput(format!("`{field}` must be a string")))
}

fn optional_model<GE>(
    input: &serde_json::Value,
    field: &str,
) -> Result<Option<ModelId>, MediaToolError<GE>>
where
    GE: std::error::Error + Send + Sync + 'static,
{
    optional_string(input, field).map(|value| value.map(ModelId::new))
}

fn optional_u8<GE>(input: &serde_json::Value, field: &str) -> Result<Option<u8>, MediaToolError<GE>>
where
    GE: std::error::Error + Send + Sync + 'static,
{
    let Some(value) = input.get(field) else {
        return Ok(None);
    };
    let Some(value) = value.as_u64() else {
        return Err(MediaToolError::InvalidInput(format!(
            "`{field}` must be an integer"
        )));
    };
    u8::try_from(value)
        .map(Some)
        .map_err(|_| MediaToolError::InvalidInput(format!("`{field}` is too large")))
}

fn optional_bounded_u8<GE>(
    input: &serde_json::Value,
    field: &str,
    max: u8,
) -> Result<Option<u8>, MediaToolError<GE>>
where
    GE: std::error::Error + Send + Sync + 'static,
{
    let value = optional_u8(input, field)?;
    if let Some(value) = value
        && value > max
    {
        return Err(MediaToolError::InvalidInput(format!(
            "`{field}` must be at most {max}"
        )));
    }
    Ok(value)
}

fn media_result_json(
    media: &dyn MediaRef,
    public_url: Option<&PublicMediaUrl>,
    extra: serde_json::Value,
) -> serde_json::Value {
    serde_json::json!({
        "uri": media.uri().as_str(),
        "category": media.category(),
        "name": media.name(),
        "mime_type": media.mime_type(),
        "size_bytes": media.size_bytes(),
        "public_url": public_url.map(|url| url.as_str()),
        "extra": extra,
    })
}

fn model_media_result_json(media: &dyn MediaRef, extra: serde_json::Value) -> serde_json::Value {
    serde_json::json!({
        "uri": media.uri().as_str(),
        "category": media.category(),
        "mime_type": media.mime_type(),
        "size_bytes": media.size_bytes(),
        "delivery": {
            "platform_reply": "The generated media will be attached to the final platform reply automatically. Do not paste media URIs, filenames, public URLs, or markdown image/video links in user-facing text."
        },
        "extra": extra,
    })
}

fn image_tool_schema() -> ToolInputSchema {
    let prompt_description = concat!(
        "Detailed description of the image to generate or the full desired result when editing ",
        "a reference image."
    );
    let reference_images_description = concat!(
        "Optional list of 1-3 existing images to edit, restyle, transform, vary, or combine. ",
        "Prefer this when the user refers to an image already visible in the conversation, ",
        "such as \"this image\", \"the image above\", or \"make it...\". Use exact ",
        "file://images/... URIs from prior tool results, generated-media notes, or image ",
        "attachment reference notes; public https URLs also work. Never invent paths. For ",
        "2-3 references, refer to them in the prompt as <IMAGE_0>, <IMAGE_1>, etc. in this ",
        "array's order."
    );
    ToolInputSchema::new(serde_json::json!({
        "type": "object",
        "required": ["prompt"],
        "properties": {
            "prompt": {
                "type": "string",
                "description": prompt_description
            },
            "reference_images": {
                "type": "array",
                "description": reference_images_description,
                "maxItems": MAX_REFERENCE_IMAGES,
                "items": { "type": "string" }
            },
            "aspect_ratio": {
                "type": "string",
                "description": "Optional provider-specific aspect ratio."
            },
            "model": {
                "type": "string",
                "description": "Optional provider-specific model id or quality tier."
            }
        },
        "additionalProperties": false
    }))
}

fn video_tool_schema() -> ToolInputSchema {
    ToolInputSchema::new(serde_json::json!({
        "type": "object",
        "required": ["prompt"],
        "properties": {
            "prompt": {
                "type": "string",
                "description": "The video prompt."
            },
            "image": {
                "type": "string",
                "description": "Optional media URI or public URL for an image to animate. Use file:// media URIs from prior tool results; do not invent local filesystem paths."
            },
            "duration_seconds": {
                "type": "integer",
                "minimum": 1,
                "maximum": MAX_VIDEO_DURATION_SECONDS
            },
            "aspect_ratio": {
                "type": "string",
                "description": "Optional provider-specific aspect ratio."
            },
            "resolution": {
                "type": "string",
                "description": "Optional provider-specific resolution or quality tier."
            },
            "model": {
                "type": "string",
                "description": "Optional provider-specific model id."
            }
        },
        "additionalProperties": false
    }))
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use serde_json::json;

    use super::*;
    use crate::ids::{ProviderName, ToolUseId};
    use crate::media::{LoadedMedia, MediaFuture, MediaMetadata, VideoMeta};
    use crate::tool::{ClientToolCall, ClientToolResultContent};
    use crate::usage::UsageRecord;

    #[derive(Debug)]
    struct TestError;

    impl std::fmt::Display for TestError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("test error")
        }
    }

    #[tokio::test]
    async fn image_generator_tool_rejects_more_than_three_reference_images() {
        let store = MemoryMediaStore::default();
        let error = image_request_from_tool_input::<_, TestError>(
            &store,
            json!({
                "prompt": "draw this",
                "reference_images": [
                    "https://example.com/1.png",
                    "https://example.com/2.png",
                    "https://example.com/3.png",
                    "https://example.com/4.png"
                ]
            }),
        )
        .await
        .unwrap_err();

        assert!(
            matches!(error, MediaToolError::InvalidInput(message) if message.contains("at most 3"))
        );
    }

    impl std::error::Error for TestError {}

    #[derive(Debug, Clone, Default)]
    struct MemoryMediaStore {
        created: Arc<Mutex<Vec<CreateMedia>>>,
    }

    impl MediaStore for MemoryMediaStore {
        async fn create_media(&self, input: CreateMedia) -> Result<BoxedMediaRef, MediaError> {
            self.created.lock().unwrap().push(input.clone());
            let category = input.category;
            let name = input
                .name
                .unwrap_or_else(|| format!("generated.{}", category.prefix()));
            let mime_type = input
                .mime_type
                .unwrap_or_else(|| "application/octet-stream".to_string());
            let uri = MediaUri::new(format!("memory://{}/{name}", category.prefix()));
            Ok(Box::new(MemoryMediaRef {
                metadata: MediaMetadata {
                    category,
                    name,
                    uri,
                    mime_type,
                    size_bytes: input.bytes.len() as u64,
                },
                bytes: input.bytes,
                public_url: Some(PublicMediaUrl::new("https://media.example/generated")),
            }))
        }

        async fn media_from_uri(&self, uri: &MediaUri) -> Result<BoxedMediaRef, MediaError> {
            Ok(Box::new(MemoryMediaRef {
                metadata: MediaMetadata {
                    category: MediaCategory::Image,
                    name: "reference.png".to_string(),
                    uri: uri.clone(),
                    mime_type: "image/png".to_string(),
                    size_bytes: 10,
                },
                bytes: b"reference".to_vec(),
                public_url: None,
            }))
        }

        async fn media_from_name(
            &self,
            category: MediaCategory,
            name: &str,
        ) -> Result<BoxedMediaRef, MediaError> {
            self.media_from_uri(&MediaUri::new(format!(
                "memory://{}/{name}",
                category.prefix()
            )))
            .await
        }
    }

    #[derive(Debug, Clone)]
    struct MemoryMediaRef {
        metadata: MediaMetadata,
        bytes: Vec<u8>,
        public_url: Option<PublicMediaUrl>,
    }

    impl MediaRef for MemoryMediaRef {
        fn metadata(&self) -> &MediaMetadata {
            &self.metadata
        }

        fn clone_box(&self) -> BoxedMediaRef {
            Box::new(self.clone())
        }

        fn public_url(&self) -> MediaFuture<'_, PublicMediaUrl> {
            Box::pin(async move {
                self.public_url
                    .clone()
                    .ok_or_else(|| MediaError::NoPublicUrl {
                        uri: self.uri().clone(),
                    })
            })
        }

        fn load(&self) -> MediaFuture<'_, LoadedMedia> {
            Box::pin(async move {
                Ok(LoadedMedia {
                    media: self.clone_box(),
                    bytes: self.bytes.clone(),
                })
            })
        }
    }

    #[derive(Debug, Clone)]
    struct MockImageGenerator {
        seen: Arc<Mutex<Option<(String, usize)>>>,
    }

    impl ImageGenerator for MockImageGenerator {
        type Error = TestError;

        fn backend_name(&self) -> &ProviderName {
            static NAME: std::sync::OnceLock<ProviderName> = std::sync::OnceLock::new();
            NAME.get_or_init(|| ProviderName::new("test"))
        }

        async fn generate_image(
            &self,
            request: ImageRequest,
        ) -> Result<GeneratedImage, Self::Error> {
            *self.seen.lock().unwrap() = Some((request.prompt.clone(), request.references.len()));
            Ok(GeneratedImage {
                bytes: b"generated image".to_vec(),
                mime_type: "image/png".to_string(),
                model: ModelId::new("image-model"),
                revised_prompt: Some(format!("revised {}", request.prompt)),
                usage: vec![UsageRecord::new(
                    ProviderName::new("test"),
                    crate::usage::UsageSubject::ImageGeneration,
                )],
            })
        }
    }

    #[tokio::test]
    async fn image_generator_tool_saves_media_and_returns_uri() {
        let seen = Arc::new(Mutex::new(None));
        let store = MemoryMediaStore::default();
        let tool = MockImageGenerator { seen: seen.clone() }.into_image_tool(store.clone());

        let output = tool
            .call(ClientToolCall {
                id: ToolUseId::new("call-1"),
                name: crate::ids::ToolName::new("generate_image"),
                input: json!({
                    "prompt": "draw a diagram",
                    "reference_images": ["https://example.com/reference.png"],
                    "aspect_ratio": "16:9"
                }),
            })
            .await
            .unwrap();

        assert!(!output.is_error);
        assert_eq!(
            *seen.lock().unwrap(),
            Some(("draw a diagram".to_string(), 1))
        );
        assert_eq!(store.created.lock().unwrap().len(), 1);
        assert_eq!(
            store.created.lock().unwrap()[0].mime_type.as_deref(),
            Some("image/png")
        );

        let ClientToolResultContent::Json { value } = output.result else {
            panic!("expected json tool result");
        };
        assert_eq!(value["uri"], "memory://images/generated.images");
        assert_eq!(value["mime_type"], "image/png");
        assert!(value.get("public_url").is_none());
        assert!(
            value["delivery"]["platform_reply"]
                .as_str()
                .unwrap()
                .contains("attached")
        );
        assert_eq!(
            output.trace_response["public_url"],
            "https://media.example/generated"
        );
        assert_eq!(output.usage.len(), 1);
    }

    #[tokio::test]
    async fn video_generator_tool_rejects_duration_above_xai_limit() {
        let store = MemoryMediaStore::default();
        let error = video_request_from_tool_input::<_, TestError>(
            &store,
            json!({
                "prompt": "animate this",
                "duration_seconds": 16
            }),
        )
        .await
        .unwrap_err();

        assert!(
            matches!(error, MediaToolError::InvalidInput(message) if message.contains("at most 15"))
        );
    }

    #[derive(Debug, Clone)]
    struct MockVideoGenerator;

    impl VideoGenerator for MockVideoGenerator {
        type Error = TestError;

        fn backend_name(&self) -> &ProviderName {
            static NAME: std::sync::OnceLock<ProviderName> = std::sync::OnceLock::new();
            NAME.get_or_init(|| ProviderName::new("test"))
        }

        async fn submit_video(&self, request: VideoRequest) -> Result<VideoJobId, Self::Error> {
            assert_eq!(request.prompt, "animate this");
            assert!(request.image.is_some());
            Ok(VideoJobId::new("video-job-1"))
        }

        async fn check_video(&self, job: VideoJobId) -> Result<VideoJobStatus, Self::Error> {
            assert_eq!(job.as_str(), "video-job-1");
            Ok(VideoJobStatus::Done {
                meta: VideoMeta {
                    url: "https://media.example/render.mp4".to_string(),
                    duration_seconds: Some(4.0),
                    usage: vec![UsageRecord::new(
                        ProviderName::new("test"),
                        crate::usage::UsageSubject::VideoGeneration,
                    )],
                },
            })
        }

        async fn download_video(&self, url: String) -> Result<Vec<u8>, Self::Error> {
            assert_eq!(url, "https://media.example/render.mp4");
            Ok(b"video bytes".to_vec())
        }
    }

    #[tokio::test]
    async fn video_generator_tool_polls_downloads_and_saves_media() {
        let store = MemoryMediaStore::default();
        let tool = MockVideoGenerator
            .into_video_tool(store.clone())
            .with_polling(Duration::from_millis(0), 2);

        let output = tool
            .call(ClientToolCall {
                id: ToolUseId::new("call-1"),
                name: crate::ids::ToolName::new("generate_video"),
                input: json!({
                    "prompt": "animate this",
                    "image": "memory://images/reference.png",
                    "duration_seconds": 4
                }),
            })
            .await
            .unwrap();

        assert!(!output.is_error);
        assert_eq!(store.created.lock().unwrap().len(), 1);
        assert_eq!(
            store.created.lock().unwrap()[0].category,
            MediaCategory::Video
        );
        assert_eq!(store.created.lock().unwrap()[0].mime_type, None);

        let ClientToolResultContent::Json { value } = output.result else {
            panic!("expected json tool result");
        };
        assert_eq!(value["uri"], "memory://videos/generated.videos");
        assert_eq!(value["extra"]["provider_job_id"], "video-job-1");
        assert_eq!(output.usage.len(), 1);
    }
}
