//! Video generation provider abstraction.
//!
//! Same shape as [`crate::imagegen`]: a trait, one impl per backend, a
//! static-dispatch [`AnyVideoProvider`] enum. Today only xAI is wired
//! in — Runway, Pika, Sora, etc. drop in by implementing the trait and
//! adding a variant.
//!
//! Video generation is asynchronous: provider impls expose `submit` /
//! `check_once` / `download_bytes` as low-level primitives so the bot
//! can interleave status messages between polls, and the higher-level
//! `generate` convenience for direct callers.

use thiserror::Error;

use crate::config::VideoProviderKind;

pub mod xai;

pub use xai::XaiVideoProvider;

/// Errors returned by any [`VideoProvider`].
#[derive(Debug, Error)]
pub enum VideoGenError {
    /// Network/transport failure.
    #[error("transport: {0}")]
    Transport(String),
    /// Backend returned a non-success status.
    #[error("api {status}: {body}")]
    Api {
        /// HTTP status code.
        status: u16,
        /// Truncated response body.
        body: String,
    },
    /// Response couldn't be decoded to the expected shape.
    #[error("decode: {0}")]
    Decode(String),
    /// Status came back `failed` or `expired` from the backend.
    #[error("upstream {status}: {message}")]
    Upstream {
        /// e.g. `failed`, `expired`.
        status: String,
        /// Error message from the backend.
        message: String,
    },
    /// Hit the polling timeout without a terminal status.
    #[error("polling timed out after {0:?}")]
    Timeout(std::time::Duration),
}

/// Input to [`VideoProvider::submit`] (and the convenience `generate`).
#[derive(Debug, Clone)]
pub struct VideoGenRequest {
    /// Text prompt describing the desired video.
    pub prompt: String,
    /// Optional image URL/URI to animate from. Backends that don't
    /// support image-to-video should error if this is set.
    pub image_url: Option<String>,
    /// Length in seconds (1-15). `None` lets the backend pick a default.
    pub duration_seconds: Option<u8>,
    /// Aspect ratio (e.g. `"16:9"`). `None` lets the backend pick.
    pub aspect_ratio: Option<String>,
    /// Free-form resolution/quality string — e.g. `"480p"`, `"720p"`,
    /// or any other tier the backend exposes. `None` picks the
    /// cheapest reasonable default.
    pub resolution: Option<String>,
    /// Free-form model id for backends that expose multiple models.
    /// `None` lets the backend pick its default video model.
    pub model: Option<String>,
}

/// Output of [`VideoProvider::generate`] and (after downloading) the
/// `download_bytes` step.
#[derive(Debug, Clone)]
pub struct GeneratedVideo {
    /// Raw video bytes (typically MP4).
    pub bytes: Vec<u8>,
    /// MIME type, e.g. `video/mp4`.
    pub mime_type: String,
    /// Actual generated duration in seconds.
    pub duration_seconds: f32,
    /// Echoed back so the caller can correlate with logs / pricing.
    pub request_id: String,
}

/// Polling outcome for [`VideoProvider::check_once`].
#[derive(Debug, Clone)]
pub enum JobStatus {
    /// Not done yet; caller should sleep and try again.
    Pending,
    /// Generation complete; the contained [`VideoMeta`] carries the URL
    /// + duration.
    Done(VideoMeta),
    /// Backend's classifiers refused or the generation otherwise failed.
    /// The string is the upstream error message.
    Failed(String),
    /// Job expired before completion.
    Expired,
}

/// Metadata returned when polling reports `status=done`.
#[derive(Debug, Clone)]
pub struct VideoMeta {
    /// Direct download URL.
    pub url: String,
    /// Duration in seconds (may be slightly off the requested duration).
    pub duration: Option<f32>,
}

/// Shared interface for one video generation backend. Split into
/// submit / poll / download primitives so the bot can interleave
/// status messages between polls and persist progress to
/// `video_jobs`.
pub trait VideoProvider: Send + Sync {
    /// Short, stable identifier (e.g. `"xai"`).
    fn name(&self) -> &str;

    /// Submit a generation request. Returns the backend's job id.
    /// Fast (~hundreds of ms) — meant to be awaited inline before any
    /// status updates are posted.
    fn submit(
        &self,
        request: &VideoGenRequest,
    ) -> impl std::future::Future<Output = Result<String, VideoGenError>> + Send;

    /// Poll the job once. Returns the current status without sleeping.
    /// Caller is responsible for spacing repeated calls.
    fn check_once(
        &self,
        request_id: &str,
    ) -> impl std::future::Future<Output = Result<JobStatus, VideoGenError>> + Send;

    /// Download the bytes at `url` (typically the URL returned by
    /// [`JobStatus::Done`]).
    fn download_bytes(
        &self,
        url: &str,
    ) -> impl std::future::Future<Output = Result<Vec<u8>, VideoGenError>> + Send;
}

/// Static-dispatch union of every supported video provider.
#[derive(Debug, Clone)]
pub enum AnyVideoProvider {
    /// xAI Grok Imagine Video.
    Xai(XaiVideoProvider),
}

impl AnyVideoProvider {
    /// Provider kind discriminator.
    pub fn kind(&self) -> VideoProviderKind {
        match self {
            Self::Xai(_) => VideoProviderKind::Xai,
        }
    }
}

impl VideoProvider for AnyVideoProvider {
    fn name(&self) -> &str {
        match self {
            Self::Xai(p) => p.name(),
        }
    }

    async fn submit(&self, request: &VideoGenRequest) -> Result<String, VideoGenError> {
        match self {
            Self::Xai(p) => p.submit(request).await,
        }
    }

    async fn check_once(&self, request_id: &str) -> Result<JobStatus, VideoGenError> {
        match self {
            Self::Xai(p) => p.check_once(request_id).await,
        }
    }

    async fn download_bytes(&self, url: &str) -> Result<Vec<u8>, VideoGenError> {
        match self {
            Self::Xai(p) => p.download_bytes(url).await,
        }
    }
}

impl From<crate::config::XaiVideoConfig> for AnyVideoProvider {
    fn from(c: crate::config::XaiVideoConfig) -> Self {
        Self::Xai(XaiVideoProvider::new(c.api_key))
    }
}
