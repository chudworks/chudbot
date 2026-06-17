//! Media storage, media generation, and media reference contracts.

mod generation;
mod reference;
mod store;

pub use generation::{
    AudioTranscriber, AudioTranscriptChannel, AudioTranscriptWord, AudioTranscription,
    AudioTranscriptionRequest, GeneratedImage, GeneratedVideo, ImageGenerator, ImageRequest,
    VideoGenerator, VideoJobStatus, VideoMeta, VideoRequest,
};
pub use reference::{
    BoxedMediaRef, MediaCategory, MediaError, MediaFuture, MediaMetadata, MediaRef, MediaUri,
    PublicMediaUrl, UrlMediaRef,
};
pub use store::{CreateMedia, LoadedMedia, MediaStore};
