//! Runtime media references and media access errors.
//!
//! Media flows through the API layer in three distinct forms:
//!
//! 1. [`MediaUri`] is the stable, model-facing identifier that can be stored in
//!    transcripts, tool results, and config-like data.
//! 2. [`MediaMetadata`] is the serializable snapshot attached to that URI.
//! 3. [`MediaRef`] is the runtime handle provider code uses to either mint a
//!    provider-fetchable [`PublicMediaUrl`] or load bytes.
//!
//! The API crate deliberately stops at that handle boundary. A concrete
//! [`super::MediaStore`] owns persistence and URI resolution, while provider
//! crates consume only [`BoxedMediaRef`] values.

use std::future::Future;
use std::pin::Pin;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::LoadedMedia;

/// Boxed media operation future.
///
/// [`MediaRef`] methods use this alias because trait-object methods cannot
/// return `impl Future` directly. All media access failures are normalized to
/// [`MediaError`] at this contract boundary.
pub type MediaFuture<'a, T> = Pin<Box<dyn Future<Output = Result<T, MediaError>> + Send + 'a>>;

/// Runtime media reference handle.
///
/// This is the type stored in requests, transcripts, and tool payloads when a
/// caller needs a cloneable runtime handle without knowing which backend owns
/// the bytes.
pub type BoxedMediaRef = Box<dyn MediaRef>;

/// Stable model-facing URI for a media item.
///
/// A media URI is identity, not necessarily a fetch URL. Stored media commonly
/// uses backend-owned forms such as `file://images/abc123.png`; URL-backed
/// media may use the original public URL as its URI.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct MediaUri(String);

impl MediaUri {
    /// Construct from any string-like value.
    ///
    /// This constructor performs no validation because URI ownership belongs to
    /// the backend that later resolves the value.
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    /// Borrow the URI string.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for MediaUri {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Public URL providers can fetch.
///
/// Unlike [`MediaUri`], this value is meant for immediate provider-side access.
/// Backends may return signed or otherwise temporary URLs, so callers should
/// not treat it as the durable media identity.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct PublicMediaUrl(String);

impl PublicMediaUrl {
    /// Construct from any string-like value.
    ///
    /// Validation is intentionally left to the backend or provider adapter that
    /// is about to use the URL.
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    /// Borrow the URL string.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for PublicMediaUrl {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Media class used for storage layout and provider routing.
///
/// Storage backends can map categories to prefixes, buckets, or directories,
/// while providers can use them to decide which content blocks are acceptable
/// for a request.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MediaCategory {
    /// Image media.
    Image,
    /// Video media.
    Video,
    /// Audio media.
    Audio,
    /// Avatar/user media.
    Avatar,
    /// Any other caller-defined media class.
    ///
    /// Backends that use this as a path or object-store prefix should still
    /// validate it before trusting it.
    Other(String),
}

impl MediaCategory {
    /// Stable path-ish prefix for storage-oriented category routing.
    ///
    /// Built-in variants return fixed prefixes; [`MediaCategory::Other`]
    /// returns its caller-provided value unchanged.
    pub fn prefix(&self) -> &str {
        match self {
            Self::Image => "images",
            Self::Video => "videos",
            Self::Audio => "audio",
            Self::Avatar => "avatars",
            Self::Other(prefix) => prefix.as_str(),
        }
    }
}

/// Serializable metadata for a media reference.
///
/// This is not the media handle. Persistent storage can keep the URI plus any
/// useful display or filtering fields, then resolve a fresh [`BoxedMediaRef`]
/// from a [`super::MediaStore`] when provider code needs access.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MediaMetadata {
    /// Media category for routing, storage prefixes, and provider validation.
    pub category: MediaCategory,
    /// Storage-local name, usually a filename or object key segment.
    pub name: String,
    /// Stable URI shown to models and persisted in traces.
    pub uri: MediaUri,
    /// MIME type reported by the backend or upstream source.
    pub mime_type: String,
    /// Size in bytes when known.
    ///
    /// URL-backed references may use `0` when the API layer has not fetched the
    /// resource.
    pub size_bytes: u64,
}

/// Runtime media handle.
///
/// A `MediaRef` owns the access path needed to load bytes or mint a public URL.
/// Provider crates should depend only on this trait, not on a separate media
/// storage service. This keeps media storage decisions out of provider
/// contracts while still allowing providers to choose the access mode they need.
pub trait MediaRef: std::fmt::Debug + Send + Sync {
    /// Static metadata for this media item.
    ///
    /// Implementations should return cached or already-known metadata here.
    /// Expensive access belongs in [`MediaRef::public_url`] or
    /// [`MediaRef::load`].
    fn metadata(&self) -> &MediaMetadata;

    /// Clone this trait object.
    ///
    /// This method backs the [`Clone`] implementation for [`BoxedMediaRef`].
    fn clone_box(&self) -> BoxedMediaRef;

    /// Public URL for provider-side fetching.
    ///
    /// Implementations may mint signed or temporary URLs. Returning
    /// [`MediaError::NoPublicUrl`] is valid when the backend only supports byte
    /// loading.
    fn public_url(&self) -> MediaFuture<'_, PublicMediaUrl>;

    /// Load raw bytes.
    ///
    /// Returning [`MediaError::BytesUnavailable`] is valid for handles that are
    /// already public URLs or otherwise cannot load bytes inside this process.
    fn load(&self) -> MediaFuture<'_, LoadedMedia>;

    /// Media category.
    fn category(&self) -> &MediaCategory {
        &self.metadata().category
    }

    /// Storage-local name.
    fn name(&self) -> &str {
        &self.metadata().name
    }

    /// Stable model-facing URI.
    fn uri(&self) -> &MediaUri {
        &self.metadata().uri
    }

    /// MIME type.
    fn mime_type(&self) -> &str {
        &self.metadata().mime_type
    }

    /// Size in bytes.
    fn size_bytes(&self) -> u64 {
        self.metadata().size_bytes
    }
}

impl Clone for Box<dyn MediaRef> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

/// Public URL-backed media reference.
///
/// This covers the "the caller already has an `https://...` URL" case. It can
/// be used directly by providers that accept public URLs; byte loading is not
/// available from the API crate because that would bake an HTTP client into the
/// contract layer.
#[derive(Debug, Clone)]
pub struct UrlMediaRef {
    metadata: MediaMetadata,
    url: PublicMediaUrl,
}

impl UrlMediaRef {
    /// Build a URL-backed media handle.
    ///
    /// The URL is used as the handle's durable URI and display name because no
    /// storage backend has assigned a separate local identity.
    pub fn new(
        category: MediaCategory,
        url: impl Into<String>,
        mime_type: impl Into<String>,
    ) -> Self {
        // Step 1: capture the caller-supplied URL once so metadata and the
        // public fetch URL stay in sync.
        let url = url.into();

        // Step 2: expose URL-backed media through the same metadata shape as
        // stored media. Size is unknown because the API layer does not fetch.
        Self {
            metadata: MediaMetadata {
                category,
                name: url.clone(),
                uri: MediaUri::new(url.clone()),
                mime_type: mime_type.into(),
                size_bytes: 0,
            },
            url: PublicMediaUrl::new(url),
        }
    }

    /// Return this handle boxed for APIs that traffic in [`BoxedMediaRef`].
    pub fn boxed(self) -> BoxedMediaRef {
        Box::new(self)
    }
}

impl MediaRef for UrlMediaRef {
    fn metadata(&self) -> &MediaMetadata {
        &self.metadata
    }

    fn clone_box(&self) -> BoxedMediaRef {
        Box::new(self.clone())
    }

    fn public_url(&self) -> MediaFuture<'_, PublicMediaUrl> {
        Box::pin(async move { Ok(self.url.clone()) })
    }

    fn load(&self) -> MediaFuture<'_, LoadedMedia> {
        Box::pin(async move {
            // URL-backed refs intentionally do not fetch bytes in chudbot-api;
            // provider or storage crates decide whether HTTP access is needed.
            Err(MediaError::BytesUnavailable {
                uri: self.uri().clone(),
            })
        })
    }
}

/// Media storage/access error.
///
/// These variants describe failures at the media contract boundary. Concrete
/// backends can preserve detailed diagnostics in logs while returning a stable
/// API error to callers.
#[derive(Debug, Error)]
pub enum MediaError {
    /// Media category is not supported by a backend.
    #[error("unsupported media category: {0}")]
    UnsupportedCategory(String),
    /// URI scheme or prefix is not owned by a backend.
    #[error("unsupported media uri: {0}")]
    UnsupportedUri(String),
    /// Unsafe storage-local name.
    #[error("unsafe media name: {0}")]
    UnsafeName(String),
    /// No public URL is currently available.
    #[error("media has no public URL: {uri}")]
    NoPublicUrl {
        /// Stable media URI.
        uri: MediaUri,
    },
    /// Bytes cannot be loaded by this handle.
    #[error("media bytes are unavailable: {uri}")]
    BytesUnavailable {
        /// Stable media URI.
        uri: MediaUri,
    },
    /// Filesystem or object-store I/O failed.
    #[error("io: {0}")]
    Io(String),
}

impl From<std::io::Error> for MediaError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error.to_string())
    }
}
