//! Provider-neutral contracts for Chudbot.
//!
//! This crate is the boundary between the bot runtime and concrete integrations.
//! It defines the data shapes and traits that platform adapters, model/media
//! providers, storage backends, and the web viewer agree on. Concrete crates
//! such as `chudbot-discord`, `chudbot-storage-sqlx`, `chudbot-openai`, and
//! `chudbot-xai` implement these contracts without making this crate depend on
//! Twilight, SQLx, Reqwest, Axum, or provider SDKs.
//!
//! The high-level runtime flow is:
//!
//! 1. A [`platform::MessagePlatform`] yields a [`platform::PlatformEvent`].
//! 2. Bot orchestration uses [`storage::BotStorage`] to resolve or open a
//!    conversation and build a [`transcript::Transcript`].
//! 3. An [`agent::Agent`] drives one or more [`llm::LlmBackend`] model steps,
//!    interleaving [`tool::ClientToolExecutor`] calls when the model requests
//!    client-side tools.
//! 4. Model continuations, tool traces, media references, usage, and live
//!    updates are persisted or published through the storage, media, usage, and
//!    event contracts below.
//!
//! Most downstream crates import from this crate root as a small prelude. The
//! public modules remain available for focused docs and for code that wants a
//! clearer namespace, while the re-exports keep service-wiring code readable
//! when it touches several contract families at once.
//!
//! ## Contract families
//!
//! - [`ids`] and [`transcript`] define the portable identity and model-context
//!   vocabulary shared by every boundary.
//! - [`platform`] and [`storage`] connect external messaging events to persisted
//!   conversations, turns, context, and trace data.
//! - [`llm`], [`agent`], and [`tool`] define the model loop: provider requests,
//!   assistant steps, client tool execution, server tool traces, and outcomes.
//! - [`media`], [`usage`], [`reasoning`], and [`events`] carry side-channel data
//!   that the bot and trace viewer need without coupling them to one provider.
//! - [`registries`] and [`retry`] provide shared runtime plumbing for named
//!   services and transient upstream failures.

// Public traits in this crate use native async trait methods/RPITIT. Concrete
// implementations stay statically dispatched, and callers add `Send` bounds at
// spawn or service boundaries where they need them.
#![allow(async_fn_in_trait)]

// Topic modules. These are public so rustdoc has stable pages for each contract
// family; the grouped re-exports below are the crate-root convenience surface.
pub mod agent;
pub mod events;
pub mod ids;
pub mod llm;
pub mod media;
pub mod platform;
pub mod reasoning;
pub mod registries;
pub mod retry;
pub mod storage;
pub mod tool;
pub mod transcript;
pub mod usage;

// Shared id newtypes keep platform/provider ids explicit without making the API
// crate depend on any concrete platform SDK.
pub use ids::{
    ChannelRef, ConversationId, ExternalId, MessageRef, ModelId, PlatformName, ProviderName,
    ToolName, ToolUseId, TurnId, UserRef, VideoJobId,
};

// Transcript types are the provider-neutral model input/output stream, including
// media blocks, tool call/result blocks, and opaque provider continuations.
pub use transcript::{ContentBlock, ProviderContinuation, Transcript, TranscriptTurn, TurnRole};

// Platform adapters normalize external messaging systems into bot events,
// messages, commands, replies, reactions, and history fetches.
pub use platform::{
    AttachmentRef, FetchMessages, MessagePlatform, OutgoingAttachment, PlatformCommand,
    PlatformCommandDefinition, PlatformCommandInput, PlatformCommandOption,
    PlatformCommandOptionChoice, PlatformCommandOptionKind, PlatformCommandResponse,
    PlatformCommandResponseTarget, PlatformCommandValue, PlatformEvent, PlatformMessage,
    PlatformMessageReference, PlatformMessageRelationship, PlatformReaction, PlatformReady,
    PostedMessage, ReactionKind, SendMessage, ThreadRequest, UserProfile,
};

// Persistence contracts are intentionally workflow-shaped rather than
// table-shaped: storage backends answer bot/runtime questions, not SQL row APIs.
pub use storage::{
    AgentSelection, BeginTurn, BotStorage, ChannelLink, ContextItem, Conversation,
    ConversationLookup, ConversationSnapshot, ConversationStop, CountActiveVideoGenerations,
    CreateVideoJob, FinishTurn, MemoryJobCompletion, MemoryJobKind, MemoryJobSchedule,
    MemoryTurnWindow, MessageLink, ModelStepKind, ModelStepTrace, NewUserMemoryDiaryEntry,
    NewUserMemoryDocumentRevision, NewUserMemoryEvent, OpenConversation, PrivacyMode, ResolveAgent,
    RetryTurn, RuntimeSettings, SaveTurnInput, StoredUserProfile, StoredVideoJob, Turn, TurnAsset,
    TurnSnapshot, TurnStatus, UpdateVideoJob, UserMemoryAudioTranscription, UserMemoryDiaryEntry,
    UserMemoryDocument, UserMemoryEvent, UserMemoryEventKind, UserMemoryImageContext,
    UserMemoryJob, UserMemoryKey, UserMemoryTurn,
};

// Model contracts cover both static model configuration and one provider
// round-trip; provider crates translate these shapes to their native APIs.
pub use llm::{
    AssistantStep, LlmBackend, Model, ModelInfo, ModelInfoRequest, ModelSpec, ModelStep,
    ModelStepRequest, ProviderOptions, SamplingOptions, ServerToolSet,
};

// Agent loop contracts: static agent config, runtime agent execution, outcomes,
// subagent adapters, and final assistant answers.
pub use agent::{
    Agent, AgentError, AgentLimits, AgentOutcome, AgentRun, AgentRunError, AgentSpec,
    AssistantAnswer, Subagent,
};

// Tool protocol shapes cover model-visible client tools, provider-run server
// tools, grounding metadata, persisted traces, and tool usage.
pub use tool::{
    ClientToolCall, ClientToolDefinition, ClientToolExecutor, ClientToolExecutorError,
    ClientToolOutput, ClientToolResult, ClientToolResultContent, ClientToolSpec, ClientToolTrace,
    GroundingMetadata, NoClientTools, ServerToolUse, ToolInputSchema, ToolTrace,
};

// Media combines three related boundaries: stored media references, media store
// IO, and provider-side image/video/audio generation.
pub use media::{
    AudioTranscriber, AudioTranscriptChannel, AudioTranscriptWord, AudioTranscription,
    AudioTranscriptionRequest, BoxedMediaRef, CreateMedia, GeneratedImage, GeneratedVideo,
    ImageGenerator, ImageRequest, LoadedMedia, MediaCategory, MediaError, MediaFuture,
    MediaMetadata, MediaRef, MediaStore, MediaUri, PublicMediaUrl, UrlMediaRef, VideoGenerator,
    VideoJobStatus, VideoMeta, VideoRequest,
};

// Usage records and aggregate query shapes let model, tool, media, and nested
// agent work report cost without forcing every provider into one billing model.
pub use usage::{
    CostAmount, UsageCostGrouping, UsageCostQuery, UsageCostRow, UsageCostScope, UsageRecord,
    UsageSubject,
};

// Viewer-safe reasoning summaries extracted from provider continuations and
// associated usage, without exposing replay-only opaque provider state.
pub use reasoning::{ReasoningItem, ReasoningSummary, ReasoningUsage, TurnReasoning};

// Live event contracts shared by the bot runtime and the web trace viewer.
pub use events::{ConversationEventKind, EventSink, LiveEvent, NoopEventSink};

// Named service registries are the thin dispatch layer used by config-driven
// runtime wiring; they route requests to already-constructed providers.
pub use registries::{
    AudioTranscriberRegistry, ImageGeneratorRegistry, LlmProviderRegistry, MessagePlatformRegistry,
    VideoGeneratorRegistry,
};
