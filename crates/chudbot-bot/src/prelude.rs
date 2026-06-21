//! Crate-private import prelude for the bot orchestration modules.
//!
//! The split modules in `chudbot-bot` all work against the same platform-neutral
//! contracts, runtime helpers, and small utility types. Keeping those imports
//! here lets each module use `crate::prelude::*` and keep its local source
//! focused on bot behavior instead of repeating the same long dependency list.
//!
//! This module is not part of the crate's public API. Public exports remain in
//! `lib.rs`; these re-exports are only a maintenance aid for sibling modules.

/// Standard-library building blocks used throughout runtime state and helper
/// signatures.
pub(crate) use std::collections::{BTreeMap, BTreeSet};
pub(crate) use std::future::Future;
pub(crate) use std::sync::{Arc, Mutex};
pub(crate) use std::time::Duration;

/// Provider-neutral Chudbot contracts shared by the orchestration modules.
///
/// Grouping these `chudbot-api` imports keeps the split modules readable while
/// making new cross-module contract dependencies visible in one place.
pub(crate) use chudbot_api::{
    Agent, AgentLimits, AgentOutcome, AgentRun, AgentRunError, AgentSelection, AttachmentRef,
    AudioTranscriber, AudioTranscriberRegistry, AudioTranscription, AudioTranscriptionRequest,
    BeginTurn, BotStorage, ChannelLink, ChannelRef, ClientToolCall, ClientToolDefinition,
    ClientToolExecutor, ClientToolExecutorError, ClientToolOutput, ClientToolResult,
    ClientToolResultContent, ClientToolSpec, ClientToolTrace, ContentBlock, Conversation,
    ConversationEventKind, ConversationId, ConversationLookup, ConversationSnapshot,
    ConversationStop, CountActiveVideoGenerations, CreateMedia, CreateVideoJob, EventSink,
    ExternalId, FetchMessages, FinishTurn, GeneratedImage, GuildProfile, ImageGenerator,
    ImageGeneratorRegistry, ImageRequest, LiveEvent, LlmProviderRegistry, MediaCategory, MediaRef,
    MediaStore, MediaUri, MessageLink, MessagePlatformRegistry, MessageRef, Model, ModelId,
    ModelSpec, ModelStepEvent, ModelStepKind, ModelStepTrace, NoClientTools, OpenConversation,
    OutgoingAttachment, PlatformCommand, PlatformCommandDefinition, PlatformCommandInput,
    PlatformCommandOption, PlatformCommandOptionChoice, PlatformCommandOptionKind,
    PlatformCommandResponse, PlatformCommandValue, PlatformEvent, PlatformMessage,
    PlatformMessageReference, PlatformMessageRelationship, PlatformName, PlatformReaction,
    PrivacyMode, ProviderName, ReactionKind, ResolveAgent, RuntimeSettings, SamplingNumber,
    SamplingOptions, SaveTurnInput, SendMessage, StoredVideoJob, ThreadRequest, ToolInputField,
    ToolInputSchema, ToolInputValueSchema, ToolName, ToolTrace, ToolUseId, Transcript,
    TranscriptTurn, Turn, TurnAsset, TurnId, TurnRole, TurnSnapshot, UpdateVideoJob, UrlMediaRef,
    UsageCostGrouping, UsageCostQuery, UsageCostRow, UsageCostScope, UsageRecord, UserProfile,
    UserRef, VideoGenerator, VideoGeneratorRegistry, VideoJobId, VideoJobStatus, VideoRequest,
    collect_agent_run,
};

// Small helper crates used broadly enough that local imports would add noise.
/// Serialization derives and bounds used by trace, tool, and command payloads.
pub(crate) use serde::Serialize;
/// Error derive used by crate-local error types.
pub(crate) use thiserror::Error;
/// Timestamp type used in stored conversations, turns, jobs, and usage rows.
pub(crate) use time::OffsetDateTime;

// Async primitives shared by the runtime, turn execution, and background jobs.
/// Tokio mutex type used when a lock must be held across async work.
pub(crate) use tokio::sync::{Mutex as AsyncMutex, OwnedMutexGuard};
/// Tokio task handles used by spawned runtime and provider work.
pub(crate) use tokio::task::{JoinError, JoinHandle, JoinSet};
/// Cooperative cancellation handle for in-flight turns and shutdown paths.
pub(crate) use tokio_util::sync::CancellationToken;
/// Background task tracker used to drain owned runtime work on shutdown.
pub(crate) use tokio_util::task::TaskTracker;

// Unicode emoji helpers are kept together because text handling usually needs
// the marker trait and the low-level classifier functions side by side.
/// Trait extension used for Unicode emoji classification on characters.
pub(crate) use unicode_properties::UnicodeEmoji;
/// Low-level emoji classifier helpers used by message text handling.
pub(crate) use unicode_properties::emoji::{
    is_emoji_presentation_selector, is_regional_indicator, is_tag_character,
    is_text_presentation_selector, is_zwj,
};
