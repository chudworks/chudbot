//! Provider-neutral media contracts.
//!
//! This module is the shared boundary between platform ingestion, media stores,
//! model/tool code, and concrete provider crates. `chudbot-api` does not know
//! how to download from Discord, write to a filesystem, serve public URLs, or
//! call an image/video/audio provider; it only defines the handles and request
//! shapes those crates exchange.
//!
//! # Flow
//!
//! 1. Platform or tool code persists bytes through [`MediaStore::create_media`]
//!    using [`CreateMedia`] and receives a [`BoxedMediaRef`].
//! 2. Stored media is carried through transcripts and tool results as
//!    [`MediaUri`] plus [`MediaMetadata`], so traces can be serialized without
//!    embedding bytes.
//! 3. Runtime code resolves those URIs through [`MediaStore::media_from_uri`]
//!    when a provider needs an actual [`MediaRef`].
//! 4. Provider crates consume request structs such as [`ImageRequest`],
//!    [`VideoRequest`], and [`AudioTranscriptionRequest`], then return generated
//!    bytes, job status, transcript text, and usage metadata.
//!
//! The re-exports below are grouped by responsibility rather than media type:
//! provider request/result traits, runtime references/errors, and store
//! persistence. That keeps `chudbot_api::media::*` imports stable while the
//! implementation details stay split across focused submodules.

// Keep the implementation modules private; this facade is the public media API.
mod generation;
mod reference;
mod store;

// Provider-facing generation and transcription contracts. These are the shapes
// provider crates implement or return, independent of where media is stored.
pub use generation::{
    AudioTranscriber, AudioTranscriptChannel, AudioTranscriptWord, AudioTranscription,
    AudioTranscriptionRequest, GeneratedImage, GeneratedVideo, ImageGenerator, ImageRequest,
    VideoGenerator, VideoJobStatus, VideoMeta, VideoRequest,
};

// Runtime references, serializable metadata, URL handles, and shared access
// errors. These types let traces store stable references instead of raw bytes.
pub use reference::{
    BoxedMediaRef, LEGACY_FILE_MEDIA_SCHEME, MediaCategory, MediaError, MediaMetadata, MediaRef,
    MediaUri, PublicMediaUrl, STORED_MEDIA_SCHEME, StoredMediaUri, UrlMediaRef,
    canonical_stored_media_uri, is_stored_media_uri, parse_stored_media_uri,
    stored_media_served_path, stored_media_uri,
};

// Store persistence contracts. These turn newly created bytes or stable URIs
// into `MediaRef` handles that provider/runtime code can use later.
pub use store::{CreateMedia, LoadedMedia, MediaStore};
