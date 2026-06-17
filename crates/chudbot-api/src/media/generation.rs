//! Provider-neutral image, video, and audio generation contracts.
//!
//! This module is the shared boundary between bot orchestration and concrete
//! media provider crates. It describes the request/result shapes that provider
//! implementations understand, but it does not choose configured provider
//! names, persist returned bytes, or expose any vendor transport details.
//!
//! # Lifecycle
//!
//! 1. Bot/tool code resolves user input into request structs such as
//!    [`ImageRequest`], [`VideoRequest`], or [`AudioTranscriptionRequest`].
//! 2. Named registries route those requests to a configured implementation.
//! 3. Image generation and audio transcription return final artifacts in one
//!    provider call.
//! 4. Video generation is a long-running job: submit with
//!    [`VideoGenerator::submit_video`], poll with [`VideoGenerator::check_video`],
//!    then fetch bytes with [`VideoGenerator::download_video`] after the job is
//!    [`VideoJobStatus::Done`].
//! 5. Callers decide how to store returned bytes through
//!    [`crate::media::MediaStore`] and how to record jobs through
//!    [`crate::storage::CreateVideoJob`]/[`crate::storage::UpdateVideoJob`].
//!
//! # Boundary Rules
//!
//! [`BoxedMediaRef`] is the handoff type for existing media. A provider may ask
//! the handle for a public URL or load bytes, but this API crate does not
//! require a particular storage backend, HTTP client, or provider upload flow.
//! Returned `bytes` are raw provider output, not stored media; persistence is a
//! caller responsibility. [`UsageRecord`] travels with media results so storage
//! and reporting can account for provider-reported or locally estimated cost
//! without coupling this crate to any one billing format.

use std::future::Future;

use serde::{Deserialize, Serialize};

use crate::ids::{ModelId, ProviderName, VideoJobId};
use crate::usage::UsageRecord;

use super::BoxedMediaRef;

// Request types are provider-facing and intentionally contain only normalized
// options plus runtime media handles. Configured provider names live in the
// registry layer, not in these structs.

/// Provider-facing request to transcribe one audio input.
///
/// The audio handle may represent stored media or a public URL. The transcriber
/// chooses whether to call [`crate::media::MediaRef::public_url`] or
/// [`crate::media::MediaRef::load`] based on what the upstream API accepts.
#[derive(Debug, Clone)]
pub struct AudioTranscriptionRequest {
    /// Audio file to transcribe.
    pub audio: BoxedMediaRef,
    /// Optional BCP-47-ish language hint for provider-side recognition or text
    /// formatting.
    pub language: Option<String>,
    /// Optional key terms to bias transcription toward domain vocabulary.
    pub keyterms: Vec<String>,
    /// Provider-specific model id when applicable. `None` means use the
    /// configured provider default.
    pub model: Option<ModelId>,
}

// Result types are serializable because they can be stored with traces or usage
// records after the provider call has completed.

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

/// Transcript for one channel when multichannel transcription is enabled.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioTranscriptChannel {
    /// Channel index.
    pub index: u32,
    /// Channel transcript text.
    pub text: String,
    /// Word-level timing for this channel.
    pub words: Vec<AudioTranscriptWord>,
}

/// Completed audio transcription result.
///
/// Providers should preserve the most precise timing and channel information
/// they receive. Empty `words` or `channels` vectors mean the provider did not
/// report that level of structure, not that transcription failed.
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

/// Provider-facing request to create or edit an image.
///
/// `references` are optional source images for editing, restyling, or other
/// image-to-image workflows. Providers that do not support references should
/// reject the request in their own error type rather than silently ignoring
/// inputs that change user intent.
#[derive(Debug, Clone)]
pub struct ImageRequest {
    /// Text prompt.
    pub prompt: String,
    /// Optional reference images for editing/restyling.
    pub references: Vec<BoxedMediaRef>,
    /// Optional aspect ratio or provider-normalized ratio string.
    pub aspect_ratio: Option<String>,
    /// Provider-specific model or quality tier. `None` means use the configured
    /// provider default.
    pub model: Option<ModelId>,
}

/// Complete image artifact returned by a provider.
///
/// The API layer keeps this as bytes plus metadata. Bot orchestration can then
/// decide whether to attach it directly, persist it through
/// [`crate::media::MediaStore`], or both.
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

/// Provider-facing request to submit a video generation job.
///
/// Video is modeled as an async job because most upstream providers queue and
/// render outside the initial request. The neutral contract supports prompt-only
/// text-to-video and a single optional image input for image-to-video.
#[derive(Debug, Clone)]
pub struct VideoRequest {
    /// Text prompt.
    pub prompt: String,
    /// Optional image to animate or use as a visual reference.
    pub image: Option<BoxedMediaRef>,
    /// Optional duration in seconds.
    pub duration_seconds: Option<u8>,
    /// Optional aspect ratio or provider-normalized ratio string.
    pub aspect_ratio: Option<String>,
    /// Optional resolution or quality tier.
    pub resolution: Option<String>,
    /// Provider-specific model id. `None` means use the configured provider
    /// default.
    pub model: Option<ModelId>,
}

/// Complete video artifact when a caller has already downloaded final bytes.
///
/// Most orchestration code observes [`VideoJobStatus::Done`], downloads bytes
/// with [`VideoGenerator::download_video`], then persists those bytes through a
/// media store. This struct is the in-memory representation after that flow has
/// produced bytes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratedVideo {
    /// Raw video bytes.
    pub bytes: Vec<u8>,
    /// MIME type, e.g. `video/mp4`.
    pub mime_type: String,
    /// Actual duration in seconds.
    pub duration_seconds: f32,
    /// Provider job id that produced these bytes.
    pub job_id: VideoJobId,
    /// Usage/cost reported for this video generation.
    pub usage: Vec<UsageRecord>,
}

/// Video metadata returned when a generation job completes.
///
/// `url` is intentionally provider-scoped: callers should pass it back to the
/// same provider's [`VideoGenerator::download_video`] implementation instead of
/// assuming it is browser-safe or permanently valid.
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
///
/// This is the polling surface between provider crates and orchestration. Only
/// [`VideoJobStatus::Done`] carries a download location; callers should treat
/// all other variants as non-downloadable terminal or in-progress states.
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

// Provider traits use native async traits/RPITIT so concrete crates can expose
// static dispatch without boxing futures at this boundary.

/// Provider implementation for image generation.
///
/// This trait owns vendor-specific request translation, media upload/fetch
/// mechanics, and usage extraction. It does not persist generated bytes or know
/// which configured name routed the call; that is handled by
/// [`crate::registries::ImageGeneratorRegistry`].
pub trait ImageGenerator: Send + Sync {
    /// Provider error type.
    type Error: std::error::Error + Send + Sync + 'static;

    /// Stable backend kind, such as a vendor or local adapter family.
    fn backend_name(&self) -> &ProviderName;

    /// Generate one image and return the final bytes.
    ///
    /// Implementations should translate [`ImageRequest`] into the upstream API,
    /// resolve any reference media as needed, and attach usage/cost records when
    /// the provider reports enough information.
    fn generate_image(
        &self,
        request: ImageRequest,
    ) -> impl Future<Output = Result<GeneratedImage, Self::Error>> + Send;
}

/// Provider implementation for async video generation.
///
/// The three-step API mirrors queue-based provider contracts. Orchestration
/// stores the submitted job id, polls it later, and downloads final bytes only
/// after [`VideoJobStatus::Done`] returns a provider download URL.
pub trait VideoGenerator: Send + Sync {
    /// Provider error type.
    type Error: std::error::Error + Send + Sync + 'static;

    /// Stable backend kind, such as a vendor or local adapter family.
    fn backend_name(&self) -> &ProviderName;

    /// Submit a video generation job and return the provider job id.
    ///
    /// Implementations should not block until rendering completes. They should
    /// perform only the initial upstream request and return the id needed by
    /// [`VideoGenerator::check_video`].
    fn submit_video(
        &self,
        request: VideoRequest,
    ) -> impl Future<Output = Result<VideoJobId, Self::Error>> + Send;

    /// Poll a video generation job once.
    ///
    /// This method should be side-effect light: inspect the upstream job and map
    /// it to [`VideoJobStatus`]. Storage updates, quota accounting, and retry
    /// scheduling belong to callers.
    fn check_video(
        &self,
        job: VideoJobId,
    ) -> impl Future<Output = Result<VideoJobStatus, Self::Error>> + Send;

    /// Download finished video bytes from a provider-scoped URL.
    ///
    /// The `url` should usually be one previously returned in [`VideoMeta`].
    /// Implementations may sign requests, use provider auth, or rewrite URLs as
    /// needed; callers should treat the returned bytes as the only portable
    /// artifact.
    fn download_video(
        &self,
        url: String,
    ) -> impl Future<Output = Result<Vec<u8>, Self::Error>> + Send;
}

/// Provider implementation for audio transcription.
///
/// Transcription is single-call at this boundary: provider crates should return
/// transcript text plus any timing, speaker, channel, model, and usage metadata
/// they can recover from the upstream response.
pub trait AudioTranscriber: Send + Sync {
    /// Provider error type.
    type Error: std::error::Error + Send + Sync + 'static;

    /// Stable backend kind, such as a vendor or local adapter family.
    fn backend_name(&self) -> &ProviderName;

    /// Transcribe one audio file and return the completed transcript.
    fn transcribe_audio(
        &self,
        request: AudioTranscriptionRequest,
    ) -> impl Future<Output = Result<AudioTranscription, Self::Error>> + Send;
}
