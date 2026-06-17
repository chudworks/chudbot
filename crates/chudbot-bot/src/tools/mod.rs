//! Runtime client tools exposed to model agents.
//!
//! This module is the crate-private facade for Chudbot's built-in client-tool
//! surface. Individual submodules own each tool contract, input validation, and
//! call implementation; [`RuntimeToolExecutor`] combines those tools with
//! per-turn context, configured provider bindings, and feature flags.
//!
//! Keeping tool specs and name-based dispatch behind the same wrapper lets
//! agent construction advertise exactly the tools the runtime can execute.

use crate::config::{
    GenerationBinding, TranscriptionBinding, VideoGenerationRateLimit,
    append_default_audio_keyterms, audio_transcription_default_keyterms,
    image_generation_tool_description, video_generation_tool_description,
};
use crate::constants::*;
use crate::media::{
    attach_supports_media, model_transcript_supports_media, public_url_supports_media,
};
use crate::platform::{privacy_mode_kind, requested_channel};
use crate::prelude::*;
use crate::registries::{
    RoutedAudioTranscriber, RoutedImageGenerator, RoutedLlmBackend, RoutedVideoGenerator,
};
use crate::runtime::BotRuntimeTypes;

mod audio_transcription;
mod executor;
mod fetch_messages;
mod image_generation;
mod media_access;
mod memory;
mod reaction;
mod shared;
mod status;
mod usage_report;
mod video_generation;

pub(crate) use audio_transcription::*;
pub(crate) use executor::*;
pub(crate) use fetch_messages::*;
pub(crate) use image_generation::*;
pub(crate) use media_access::*;
pub(crate) use memory::*;
pub(crate) use reaction::*;
pub(crate) use shared::*;
pub(crate) use status::*;
pub(crate) use usage_report::*;
pub(crate) use video_generation::*;
