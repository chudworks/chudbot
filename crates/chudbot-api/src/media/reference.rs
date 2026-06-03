//! Runtime media references and media access errors.

use std::future::Future;
use std::pin::Pin;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::LoadedMedia;

/// Boxed media operation future.
pub type MediaFuture<'a, T> = Pin<Box<dyn Future<Output = Result<T, MediaError>> + Send + 'a>>;

/// Runtime media reference handle.
pub type BoxedMediaRef = Box<dyn MediaRef>;

/// Stable model-facing URI for a stored media item.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct MediaUri(String);

impl MediaUri {
    /// Construct from any string-like value.
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
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct PublicMediaUrl(String);

impl PublicMediaUrl {
    /// Construct from any string-like value.
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

/// Media class. Storage backends can map this to prefixes, buckets, or
/// directories.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MediaCategory {
    /// Image media.
    Image,
    /// Video media.
    Video,
    /// Avatar/user media.
    Avatar,
    /// Any other caller-defined media class.
    Other(String),
}

impl MediaCategory {
    /// Stable path-ish prefix for the built-in categories.
    pub fn prefix(&self) -> &str {
        match self {
            Self::Image => "images",
            Self::Video => "videos",
            Self::Avatar => "avatars",
            Self::Other(prefix) => prefix.as_str(),
        }
    }
}

/// Serializable metadata for a media reference.
///
/// This is not the media handle. The database can store the URI and any useful
/// metadata, then resolve a fresh [`BoxedMediaRef`] from a [`MediaStore`] when
/// provider code needs access.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MediaMetadata {
    /// Media category.
    pub category: MediaCategory,
    /// Storage-local name, usually a filename.
    pub name: String,
    /// Stable URI shown to models and persisted.
    pub uri: MediaUri,
    /// MIME type.
    pub mime_type: String,
    /// Size in bytes.
    pub size_bytes: u64,
}

/// Runtime media handle.
///
/// A `MediaRef` owns the access path needed to load bytes or mint a public URL.
/// Provider crates should depend only on this trait, not on a separate media
/// storage service.
pub trait MediaRef: std::fmt::Debug + Send + Sync {
    /// Static metadata for this media item.
    fn metadata(&self) -> &MediaMetadata;

    /// Clone this trait object.
    fn clone_box(&self) -> BoxedMediaRef;

    /// Public URL for provider-side fetching.
    fn public_url(&self) -> MediaFuture<'_, PublicMediaUrl>;

    /// Load raw bytes.
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
/// This covers the "the model supplied an `https://...` URL" case. It can be
/// used directly by providers that accept public URLs; byte loading is not
/// available from the API crate because that would bake an HTTP client into the
/// contract layer.
#[derive(Debug, Clone)]
pub struct UrlMediaRef {
    metadata: MediaMetadata,
    url: PublicMediaUrl,
}

impl UrlMediaRef {
    /// Build a URL-backed media handle.
    pub fn new(
        category: MediaCategory,
        url: impl Into<String>,
        mime_type: impl Into<String>,
    ) -> Self {
        let url = url.into();
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

    /// Return this handle boxed.
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
            Err(MediaError::BytesUnavailable {
                uri: self.uri().clone(),
            })
        })
    }
}

/// Media storage/access error.
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
