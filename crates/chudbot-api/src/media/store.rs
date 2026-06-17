//! Media storage backend contract.
//!
//! The storage flow is:
//!
//! 1. Runtime code submits provider outputs or uploaded bytes as [`CreateMedia`].
//! 2. A backend persists those bytes and returns a [`BoxedMediaRef`] with stable
//!    metadata.
//! 3. Durable records store the model-facing [`MediaUri`] from that metadata.
//! 4. Later turns resolve that URI back into a fresh [`BoxedMediaRef`] through
//!    [`MediaStore::media_from_uri`].
//!
//! This crate owns only the contract. Concrete backends decide whether media
//! lives on disk, in an object store, or behind signed/public URLs.

use std::future::Future;

use serde::{Deserialize, Serialize};

use super::{BoxedMediaRef, MediaCategory, MediaError, MediaUri};

/// Request to persist a new media object.
///
/// This is the write-side DTO for media storage. Callers provide the bytes plus
/// any upstream hints they already have; backends validate those hints, choose a
/// stable name/URI, and return a runtime [`BoxedMediaRef`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateMedia {
    /// Media category used for backend routing.
    ///
    /// File-backed stores typically map this to a path prefix, while object
    /// stores may map it to a bucket prefix or metadata field.
    pub category: MediaCategory,
    /// Raw bytes to persist.
    pub bytes: Vec<u8>,
    /// Optional MIME type override or upstream hint.
    ///
    /// Backends should detect the MIME type when this is absent and may still
    /// reject or normalize caller-provided values when they conflict with the
    /// stored bytes.
    #[serde(default)]
    pub mime_type: Option<String>,
    /// Optional storage-local name requested by the caller.
    ///
    /// This is not a path. Backends remain responsible for rejecting unsafe or
    /// unsupported names before they construct a durable URI.
    pub name: Option<String>,
    /// Optional file extension override, without the dot.
    ///
    /// The extension is a naming hint only; it should not be treated as proof of
    /// the media type.
    pub extension: Option<String>,
}

/// Media bytes loaded from a runtime handle.
///
/// [`MediaRef::load`](super::MediaRef::load) returns this shape so callers get
/// both the bytes and the handle metadata that identifies exactly what was
/// loaded.
#[derive(Debug, Clone)]
pub struct LoadedMedia {
    /// Media handle used to load these bytes.
    pub media: BoxedMediaRef,
    /// Raw bytes returned by the backend.
    pub bytes: Vec<u8>,
}

/// Backend that persists media and resolves stored identifiers.
///
/// `MediaStore` is the boundary between durable, model-facing media references
/// and backend-specific access paths. Provider code should generally receive
/// [`BoxedMediaRef`] values and use [`super::MediaRef`] methods; orchestration
/// code uses this trait when it needs to create media or rehydrate a persisted
/// reference.
pub trait MediaStore: Send + Sync {
    /// Persist bytes and return a runtime media handle.
    ///
    /// Implementations should validate caller hints, write the bytes, derive
    /// stable metadata, and return a handle whose URI can be stored in
    /// transcripts or tool output for future resolution.
    fn create_media(
        &self,
        input: CreateMedia,
    ) -> impl Future<Output = Result<BoxedMediaRef, MediaError>> + Send;

    /// Resolve a model-facing media URI into a runtime handle.
    ///
    /// This is the trust boundary for URIs that may have crossed model,
    /// transcript, or tool-call surfaces. Implementations should accept only
    /// schemes and prefixes they own, reject unsafe names, and avoid loading
    /// bytes until the returned [`super::MediaRef`] is asked to do so.
    ///
    /// Examples of local-store URIs are `file://images/abc123.png` and
    /// `file://audio/abc123.ogg`.
    fn media_from_uri(
        &self,
        uri: &MediaUri,
    ) -> impl Future<Output = Result<BoxedMediaRef, MediaError>> + Send;

    /// Resolve media by category and storage-local name.
    ///
    /// This lookup is for code that already has structured storage metadata.
    /// Use [`Self::media_from_uri`] for strings that came from model-visible or
    /// serialized URI fields.
    fn media_from_name(
        &self,
        category: MediaCategory,
        name: &str,
    ) -> impl Future<Output = Result<BoxedMediaRef, MediaError>> + Send;
}
