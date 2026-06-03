//! Image and video generation provider contracts.

use std::future::Future;

use serde::{Deserialize, Serialize};

use crate::ids::{ModelId, ProviderName, VideoJobId};
use crate::usage::UsageRecord;

use super::BoxedMediaRef;

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
