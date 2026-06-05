//! Contract crate for the chudbot 2.0 shape.
//!
//! This crate intentionally contains only provider/platform/storage-neutral
//! types and traits. Concrete crates such as Discord, SQLx storage, and xAI
//! should implement these contracts without making this crate depend on their
//! transport libraries.

#![allow(async_fn_in_trait)]

pub mod agent;
pub mod events;
pub mod ids;
pub mod llm;
pub mod media;
pub mod platform;
pub mod retry;
pub mod storage;
pub mod tool;
pub mod transcript;
pub mod usage;

pub use agent::{
    Agent, AgentBuilder, AgentError, AgentLimits, AgentOutcome, AgentRun, AgentRunError, AgentSpec,
    AssistantAnswer, Subagent,
};
pub use events::{ConversationEventKind, EventSink, LiveEvent, NoopEventSink};
pub use ids::{
    ChannelRef, ConversationId, ExternalId, MessageRef, ModelId, PlatformName, ProviderName,
    ToolName, ToolUseId, TurnId, UserRef, VideoJobId,
};
pub use llm::{
    AssistantStep, LlmBackend, Model, ModelSpec, ModelStep, ModelStepRequest, ProviderOptions,
    SamplingOptions, ServerToolSet,
};
pub use media::{
    AudioTranscriber, AudioTranscriptChannel, AudioTranscriptWord, AudioTranscription,
    AudioTranscriptionRequest, BoxedMediaRef, CreateMedia, GeneratedImage, GeneratedVideo,
    ImageGenerator, ImageGeneratorTool, ImageGeneratorToolExt, ImageRequest, LoadedMedia,
    MediaCategory, MediaError, MediaFuture, MediaMetadata, MediaRef, MediaStore, MediaToolError,
    MediaUri, PublicMediaUrl, UrlMediaRef, VideoGenerator, VideoGeneratorTool,
    VideoGeneratorToolExt, VideoJobStatus, VideoMeta, VideoRequest,
};
pub use platform::{
    AttachmentRef, FetchMessages, MessagePlatform, OutgoingAttachment, PlatformCommand,
    PlatformCommandDefinition, PlatformCommandInput, PlatformCommandOption,
    PlatformCommandOptionChoice, PlatformCommandOptionKind, PlatformCommandResponse,
    PlatformCommandResponseTarget, PlatformCommandValue, PlatformEvent, PlatformMessage,
    PlatformMessageReference, PlatformMessageRelationship, PlatformReaction, PlatformReady,
    PostedMessage, ReactionKind, SendMessage, ThreadRequest, UserProfile,
};
pub use storage::{
    AgentSelection, BeginTurn, BotStorage, ChannelLink, ContextItem, Conversation,
    ConversationLookup, ConversationSnapshot, ConversationStop, CountActiveVideoGenerations,
    CreateVideoJob, FinishTurn, MemoryJobCompletion, MemoryJobKind, MemoryJobSchedule,
    MemoryTurnWindow, MessageLink, NewUserMemoryDiaryEntry, NewUserMemoryDocumentRevision,
    NewUserMemoryEvent, OpenConversation, PrivacyMode, ResolveAgent, RetryTurn, RuntimeSettings,
    SaveTurnInput, StoredUserProfile, StoredVideoJob, Turn, TurnAsset, TurnSnapshot, TurnStatus,
    UpdateVideoJob, UserMemoryAudioTranscription, UserMemoryDiaryEntry, UserMemoryDocument,
    UserMemoryEvent, UserMemoryEventKind, UserMemoryImageContext, UserMemoryJob, UserMemoryKey,
    UserMemoryTurn,
};
pub use tool::{
    ClientTool, ClientToolCall, ClientToolOutput, ClientToolResult, ClientToolResultContent,
    ClientToolSpec, ClientToolTrace, GroundingMetadata, ServerToolUse, ToolInputSchema, ToolTrace,
};
pub use transcript::{ContentBlock, ProviderContinuation, Transcript, TranscriptTurn, TurnRole};
pub use usage::{CostAmount, UsageRecord, UsageSubject};
