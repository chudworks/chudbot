//! Image generation provider abstraction.
//!
//! Mirrors the LLM provider design in [`crate::llm`]: a single trait
//! ([`ImageProvider`]) with native async fn, one implementation per
//! backend, and a static-dispatch [`AnyImageProvider`] enum so the bot
//! can hold a map of provider kinds without involving `Box<dyn …>` or
//! `async-trait`.
//!
//! Today's implementations:
//!   - [`xai::XaiImageProvider`] — xAI Grok Imagine (`grok-imagine-image`
//!     family).
//!
//! Per-request model selection is free-form ([`ImageGenRequest::model`]):
//! providers map the string to whatever model id their API expects.

use std::path::PathBuf;

use thiserror::Error;

use crate::config::ImageProviderKind;

pub mod xai;

pub use xai::XaiImageProvider;

/// Errors returned by any [`ImageProvider`].
#[derive(Debug, Error)]
pub enum ImageGenError {
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
    /// A reference image URI couldn't be resolved (file missing, etc.).
    #[error("reference image: {0}")]
    Reference(String),
}

/// Input to [`ImageProvider::generate`].
#[derive(Debug, Clone)]
pub struct ImageGenRequest {
    /// Text prompt describing the desired image.
    pub prompt: String,
    /// 0-3 reference images (URLs or `file://` URIs). Backends that
    /// don't support reference images should error if this is non-empty.
    pub references: Vec<String>,
    /// Aspect ratio (e.g. `"16:9"`). `None` lets the backend pick its
    /// default.
    pub aspect_ratio: Option<String>,
    /// Free-form model / quality tier. Each provider interprets this
    /// against its own model catalog (e.g. xAI maps `"standard"` →
    /// `grok-imagine-image`, `"quality"` → `grok-imagine-image-quality`;
    /// Midjourney might map `"v6"` → its v6 endpoint). `None` lets the
    /// provider pick a sensible default.
    pub model: Option<String>,
    /// Directory to resolve `file://` references against. Required even
    /// when `references` is empty because providers may need it to
    /// stage local data.
    pub images_dir: PathBuf,
}

/// Output of [`ImageProvider::generate`].
#[derive(Debug, Clone)]
pub struct GeneratedImage {
    /// Raw bytes of the generated image (PNG/JPEG/WebP/etc.).
    pub bytes: Vec<u8>,
    /// MIME type reported by the backend, e.g. `image/jpeg`.
    pub mime_type: String,
    /// Model id the backend actually used.
    pub model: String,
    /// Optional revised prompt the model used internally.
    pub revised_prompt: Option<String>,
}

/// Shared interface for one image generation backend.
pub trait ImageProvider: Send + Sync {
    /// Short, stable identifier (e.g. `"xai"`, `"midjourney"`).
    fn name(&self) -> &str;

    /// Generate a single image. Reference images and aspect ratio are
    /// passed through to the backend if it supports them.
    fn generate(
        &self,
        request: ImageGenRequest,
    ) -> impl std::future::Future<Output = Result<GeneratedImage, ImageGenError>> + Send;
}

/// Static-dispatch union of every supported image provider. Same shape
/// as [`crate::llm::AnyProvider`] — the bot stores a map of these by
/// [`ImageProviderKind`] and clones the appropriate one per turn.
#[derive(Debug, Clone)]
pub enum AnyImageProvider {
    /// xAI Grok Imagine.
    Xai(XaiImageProvider),
}

impl AnyImageProvider {
    /// Provider kind discriminator, useful for looking up which
    /// backend a persona maps to.
    pub fn kind(&self) -> ImageProviderKind {
        match self {
            Self::Xai(_) => ImageProviderKind::Xai,
        }
    }
}

impl ImageProvider for AnyImageProvider {
    fn name(&self) -> &str {
        match self {
            Self::Xai(p) => p.name(),
        }
    }

    async fn generate(&self, request: ImageGenRequest) -> Result<GeneratedImage, ImageGenError> {
        match self {
            Self::Xai(p) => p.generate(request).await,
        }
    }
}

impl From<crate::config::XaiImageConfig> for AnyImageProvider {
    fn from(c: crate::config::XaiImageConfig) -> Self {
        Self::Xai(XaiImageProvider::new(c.api_key))
    }
}
