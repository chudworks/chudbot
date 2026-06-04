//! Image, video, and audio media provider contracts.

use std::future::Future;

use serde::{Deserialize, Serialize};

use crate::ids::{ModelId, ProviderName, VideoJobId};
use crate::usage::UsageRecord;

use super::BoxedMediaRef;

/// Audio transcription request.
#[derive(Debug, Clone)]
pub struct AudioTranscriptionRequest {
    /// Audio file to transcribe.
    pub audio: BoxedMediaRef,
    /// Optional language code for provider-side text formatting.
    pub language: Option<String>,
    /// Optional key terms to bias transcription toward domain vocabulary.
    pub keyterms: Vec<String>,
    /// Provider-specific model id when applicable.
    pub model: Option<ModelId>,
}

/// One transcribed word with timing metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioTranscriptWord {
    /// Word text.
    pub text: String,
    /// Start timestamp in seconds.
    #[serde(rename = "start")]
    pub start_seconds: f64,
    /// End timestamp in seconds.
    #[serde(rename = "end")]
    pub end_seconds: f64,
    /// Provider confidence when reported.
    pub confidence: Option<f64>,
    /// Speaker index when diarization is enabled and reported.
    pub speaker: Option<u32>,
}

/// Per-channel transcript when multichannel transcription is enabled.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioTranscriptChannel {
    /// Channel index.
    pub index: u32,
    /// Channel transcript text.
    pub text: String,
    /// Word-level timing for this channel.
    pub words: Vec<AudioTranscriptWord>,
}

/// Audio transcription result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioTranscription {
    /// Full transcript text.
    pub text: String,
    /// Detected or requested language when reported.
    pub language: Option<String>,
    /// Audio duration in seconds.
    pub duration_seconds: f64,
    /// Word-level timing.
    pub words: Vec<AudioTranscriptWord>,
    /// Per-channel transcripts.
    pub channels: Vec<AudioTranscriptChannel>,
    /// Actual model used when applicable.
    pub model: Option<ModelId>,
    /// Usage/cost reported or estimated for this transcription.
    pub usage: Vec<UsageRecord>,
}

/// Image generation request.
#[derive(Debug, Clone)]
pub struct ImageRequest {
    /// Text prompt.
    pub prompt: String,
    /// Optional reference images for editing/restyling.
    pub references: Vec<BoxedMediaRef>,
    /// Optional aspect ratio.
    pub aspect_ratio: Option<String>,
    /// Provider-specific model or quality tier.
    pub model: Option<ModelId>,
}

/// Generated image bytes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratedImage {
    /// Raw image bytes.
    pub bytes: Vec<u8>,
    /// MIME type, e.g. `image/png`.
    pub mime_type: String,
    /// Actual model used.
    pub model: ModelId,
    /// Optional provider-revised prompt.
    pub revised_prompt: Option<String>,
    /// Usage/cost reported for this image generation.
    pub usage: Vec<UsageRecord>,
}

/// Video generation request.
#[derive(Debug, Clone)]
pub struct VideoRequest {
    /// Text prompt.
    pub prompt: String,
    /// Optional image to animate.
    pub image: Option<BoxedMediaRef>,
    /// Optional duration in seconds.
    pub duration_seconds: Option<u8>,
    /// Optional aspect ratio.
    pub aspect_ratio: Option<String>,
    /// Optional resolution or quality tier.
    pub resolution: Option<String>,
    /// Provider-specific model id.
    pub model: Option<ModelId>,
}

/// Generated video bytes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratedVideo {
    /// Raw video bytes.
    pub bytes: Vec<u8>,
    /// MIME type, e.g. `video/mp4`.
    pub mime_type: String,
    /// Actual duration in seconds.
    pub duration_seconds: f32,
    /// Provider job id.
    pub job_id: VideoJobId,
    /// Usage/cost reported for this video generation.
    pub usage: Vec<UsageRecord>,
}

/// Video metadata returned when a generation job completes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoMeta {
    /// URL to download the render.
    pub url: String,
    /// Actual duration in seconds when known.
    pub duration_seconds: Option<f32>,
    /// Usage/cost reported for this video generation.
    pub usage: Vec<UsageRecord>,
}

/// Status of an async video generation job.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum VideoJobStatus {
    /// Still running.
    Pending,
    /// Completed successfully.
    Done {
        /// Render metadata.
        meta: VideoMeta,
    },
    /// Failed upstream.
    Failed {
        /// Failure message.
        message: String,
    },
    /// Expired upstream before completion.
    Expired,
}

/// Image generation provider.
pub trait ImageGenerator: Send + Sync {
    /// Provider error type.
    type Error: std::error::Error + Send + Sync + 'static;

    /// Short backend name.
    fn backend_name(&self) -> &ProviderName;

    /// Generate one image.
    fn generate_image(
        &self,
        request: ImageRequest,
    ) -> impl Future<Output = Result<GeneratedImage, Self::Error>> + Send;
}

/// Video generation provider.
pub trait VideoGenerator: Send + Sync {
    /// Provider error type.
    type Error: std::error::Error + Send + Sync + 'static;

    /// Short backend name.
    fn backend_name(&self) -> &ProviderName;

    /// Submit a video generation job.
    fn submit_video(
        &self,
        request: VideoRequest,
    ) -> impl Future<Output = Result<VideoJobId, Self::Error>> + Send;

    /// Poll a video generation job once.
    fn check_video(
        &self,
        job: VideoJobId,
    ) -> impl Future<Output = Result<VideoJobStatus, Self::Error>> + Send;

    /// Download finished video bytes.
    fn download_video(
        &self,
        url: String,
    ) -> impl Future<Output = Result<Vec<u8>, Self::Error>> + Send;
}

/// Audio transcription provider.
pub trait AudioTranscriber: Send + Sync {
    /// Provider error type.
    type Error: std::error::Error + Send + Sync + 'static;

    /// Short backend name.
    fn backend_name(&self) -> &ProviderName;

    /// Transcribe one audio file.
    fn transcribe_audio(
        &self,
        request: AudioTranscriptionRequest,
    ) -> impl Future<Output = Result<AudioTranscription, Self::Error>> + Send;
}
