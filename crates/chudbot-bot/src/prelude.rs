//! Internal import bundle shared by the split bot modules.

pub(crate) use std::collections::{BTreeMap, BTreeSet};
pub(crate) use std::future::Future;
pub(crate) use std::sync::{Arc, Mutex};
pub(crate) use std::time::Duration;

pub(crate) use chudbot_api::{
    Agent, AgentLimits, AgentOutcome, AgentSelection, AttachmentRef, AudioTranscriber,
    AudioTranscriberRegistry, AudioTranscription, AudioTranscriptionRequest, BeginTurn, BotStorage,
    ChannelLink, ChannelRef, ClientToolCall, ClientToolDefinition, ClientToolExecutor,
    ClientToolExecutorError, ClientToolOutput, ClientToolResult, ClientToolResultContent,
    ClientToolSpec, ClientToolTrace, ContentBlock, Conversation, ConversationEventKind,
    ConversationId, ConversationLookup, ConversationSnapshot, ConversationStop,
    CountActiveVideoGenerations, CreateMedia, CreateVideoJob, EventSink, ExternalId, FetchMessages,
    FinishTurn, GeneratedImage, ImageGenerator, ImageGeneratorRegistry, ImageRequest, LiveEvent,
    LlmProviderRegistry, MediaCategory, MediaRef, MediaStore, MediaUri, MessageLink,
    MessagePlatformRegistry, MessageRef, Model, ModelId, ModelSpec, ModelStep, ModelStepKind,
    ModelStepTrace, NoClientTools, OpenConversation, OutgoingAttachment, PlatformCommand,
    PlatformCommandDefinition, PlatformCommandInput, PlatformCommandOption,
    PlatformCommandOptionChoice, PlatformCommandOptionKind, PlatformCommandResponse,
    PlatformCommandValue, PlatformEvent, PlatformMessage, PlatformMessageReference,
    PlatformMessageRelationship, PlatformName, PlatformReaction, PrivacyMode, ProviderName,
    ReactionKind, ResolveAgent, RuntimeSettings, SamplingOptions, SaveTurnInput, SendMessage,
    StoredVideoJob, Subagent, ThreadRequest, ToolInputSchema, ToolName, ToolTrace, ToolUseId,
    Transcript, TranscriptTurn, Turn, TurnAsset, TurnId, TurnRole, TurnSnapshot, UpdateVideoJob,
    UrlMediaRef, UsageCostGrouping, UsageCostQuery, UsageCostRow, UsageCostScope, UsageRecord,
    UserProfile, UserRef, VideoGenerator, VideoGeneratorRegistry, VideoJobId, VideoJobStatus,
    VideoRequest,
};
pub(crate) use serde::Serialize;
pub(crate) use thiserror::Error;
pub(crate) use time::OffsetDateTime;
pub(crate) use tokio::sync::{Mutex as AsyncMutex, OwnedMutexGuard};
pub(crate) use tokio::task::{JoinError, JoinHandle, JoinSet};
pub(crate) use tokio_util::sync::CancellationToken;
pub(crate) use tokio_util::task::TaskTracker;
pub(crate) use unicode_properties::UnicodeEmoji;
pub(crate) use unicode_properties::emoji::{
    is_emoji_presentation_selector, is_regional_indicator, is_tag_character,
    is_text_presentation_selector, is_zwj,
};
