//! Platform-neutral bot orchestration.
//!
//! This crate owns the chudbot turn lifecycle without knowing about Discord,
//! Postgres, Axum, or concrete model-provider HTTP clients. It consumes the
//! contracts from `chudbot-api` and routes work through named service
//! registries supplied by the binary crate.
//!
//! Callers assemble a [`BotRuntime`] from [`BotRuntimeParts`], feed it
//! platform-neutral events, and execute the returned [`BotAction`]s through a
//! platform adapter.
//!
//! ## Major boundaries
//!
//! - `runtime` owns the process-level runtime and delegates individual turn
//!   handling to `turns` and transcript assembly to `transcript`.
//! - `agents` and `tools` build model agents and expose tool capabilities, while
//!   `media` prepares generated or uploaded assets for providers and replies.
//! - `config` and `registries` connect agent bindings to named runtime services,
//!   keeping concrete provider clients outside this crate.
//! - `platform`, `commands`, and `action` translate between platform events,
//!   slash-style commands, and platform-neutral side effects.
//! - [`memory`] contains long-running memory job support used by the runtime.

#![allow(async_fn_in_trait)]

mod action;
mod agents;
mod avatars;
mod commands;
mod config;
mod constants;
mod error;
mod media;
/// Memory configuration and background job support used by the bot runtime.
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

/// Platform-neutral side effects that adapters execute against their own APIs.
pub use action::BotAction;
/// Configuration types used to bind agents, platforms, and media generation.
pub use config::{
    AgentConfig, BotConfig, BotRunOptions, GenerationBinding, PlatformBinding, PlatformScopeBypass,
    SubagentBinding, TranscriptionBinding, VideoGenerationRateLimit,
};
/// Error type returned by bot runtime operations.
pub use error::BotError;
/// Memory runtime configuration exposed to the process config loader.
pub use memory::MemoryConfig;
/// Service routers that resolve named agent bindings to concrete providers.
pub use registries::{
    RoutedAudioTranscriber, RoutedImageGenerator, RoutedLlmBackend, RoutedVideoGenerator,
};
/// Runtime entry points and construction state for the process launcher.
pub use runtime::{BotRuntime, BotRuntimeParts, BotRuntimeTypes};

pub(crate) use action::*;
pub(crate) use agents::*;
pub(crate) use commands::*;
pub(crate) use config::SystemAgentConfig;
pub(crate) use constants::*;
pub(crate) use error::*;
pub(crate) use media::*;
pub(crate) use platform::*;
pub(crate) use runtime::*;
pub(crate) use tools::*;
pub(crate) use transcript::*;
pub(crate) use turns::*;
