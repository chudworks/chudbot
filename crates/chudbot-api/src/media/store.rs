//! Media storage backend contract.

use std::future::Future;

use serde::{Deserialize, Serialize};

use super::{BoxedMediaRef, MediaCategory, MediaError, MediaUri};

/// Media bytes to persist.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateMedia {
    /// Media category.
    pub category: MediaCategory,
    /// Raw bytes.
    pub bytes: Vec<u8>,
    /// Optional MIME type override or upstream hint. Stores should detect the
    /// MIME type when this is absent.
    #[serde(default)]
    pub mime_type: Option<String>,
    /// Optional storage-local name. Backends may reject unsafe names.
    pub name: Option<String>,
    /// Optional file extension override, without the dot.
    pub extension: Option<String>,
}

/// Loaded media bytes.
#[derive(Debug, Clone)]
pub struct LoadedMedia {
    /// Media handle.
    pub media: BoxedMediaRef,
    /// Raw bytes.
    pub bytes: Vec<u8>,
}

/// Media storage backend.
pub trait MediaStore: Send + Sync {
    /// Persist bytes and return a runtime media handle.
    fn create_media(
        &self,
        input: CreateMedia,
    ) -> impl Future<Output = Result<BoxedMediaRef, MediaError>> + Send;

    /// Resolve a model-facing media URI such as `file://images/abc123.png` or
    /// `file://audio/abc123.ogg`.
    fn media_from_uri(
        &self,
        uri: &MediaUri,
    ) -> impl Future<Output = Result<BoxedMediaRef, MediaError>> + Send;

    /// Resolve media by category and storage-local name.
    fn media_from_name(
        &self,
        category: MediaCategory,
        name: &str,
    ) -> impl Future<Output = Result<BoxedMediaRef, MediaError>> + Send;
}
