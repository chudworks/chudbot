//! Media storage, media generation, and media reference contracts.

mod generation;
mod reference;
mod store;
mod tool;

pub use generation::{
    GeneratedImage, GeneratedVideo, ImageGenerator, ImageRequest, VideoGenerator, VideoJobStatus,
    VideoMeta, VideoRequest,
};
pub use reference::{
    BoxedMediaRef, MediaCategory, MediaError, MediaFuture, MediaMetadata, MediaRef, MediaUri,
    PublicMediaUrl, UrlMediaRef,
};
pub use store::{CreateMedia, LoadedMedia, MediaStore};
pub use tool::{
    ImageGeneratorTool, ImageGeneratorToolExt, MediaToolError, VideoGeneratorTool,
    VideoGeneratorToolExt,
};
