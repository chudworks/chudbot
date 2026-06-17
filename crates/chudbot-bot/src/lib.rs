//! Platform-neutral bot orchestration.
//!
//! This crate owns the chudbot turn lifecycle without knowing about Discord,
//! Postgres, Axum, or concrete model-provider HTTP clients. It consumes the
//! contracts from `chudbot-api` and routes work through named service
//! registries supplied by the binary crate.

#![allow(async_fn_in_trait)]

mod action;
mod agents;
mod commands;
mod config;
mod constants;
mod error;
mod media;
pub mod memory;
mod platform;
mod prelude;
mod registries;
mod runtime;
mod tools;
mod transcript;
mod turns;

#[cfg(test)]
mod tests;

pub use action::BotAction;
pub use config::{
    AgentConfig, BotConfig, BotRunOptions, GenerationBinding, PlatformBinding, PlatformScopeBypass,
    SubagentBinding, TranscriptionBinding, VideoGenerationRateLimit,
};
pub use error::BotError;
pub use memory::MemoryConfig;
pub use registries::{
    RoutedAudioTranscriber, RoutedImageGenerator, RoutedLlmBackend, RoutedVideoGenerator,
};
pub use runtime::{BotRuntime, BotRuntimeParts, BotRuntimeTypes};

pub(crate) use action::*;
pub(crate) use agents::*;
pub(crate) use commands::*;
pub(crate) use config::{
    SystemAgentConfig, append_default_audio_keyterms, audio_transcription_default_keyterms,
    image_generation_tool_description, video_generation_tool_description,
};
pub(crate) use constants::*;
pub(crate) use error::*;
pub(crate) use media::*;
pub(crate) use platform::*;
pub(crate) use runtime::*;
pub(crate) use tools::*;
pub(crate) use transcript::*;
pub(crate) use turns::*;
