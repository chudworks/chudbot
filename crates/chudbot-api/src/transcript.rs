//! Provider-neutral transcript model used for language-model requests.
//!
//! A [`Transcript`] is the model-facing view of a conversation at one backend
//! step. The bot layer gathers platform messages, stored media, client tool
//! results, and provider continuation state into this shape before building a
//! [`crate::llm::ModelStepRequest`].
//!
//! The usual flow is:
//!
//! 1. Create or load a [`Transcript`] with stable prompt instructions.
//! 2. Append [`TranscriptTurn`] values in the exact order the model should see.
//! 3. Put text, media, client tool calls/results, and provider continuations in
//!    each turn's ordered [`ContentBlock`] list.
//! 4. Hand the transcript to a provider crate, which translates it into that
//!    provider's native request format.
//!
//! This module intentionally does not define persistence, Discord-specific
//! messages, HTTP request shapes, or provider-side/server-side tool traces.
//! Those contracts live in the storage, platform, provider, and tool modules.

use serde::{Deserialize, Serialize};

use crate::ids::ProviderName;
use crate::media::BoxedMediaRef;
use crate::tool::{ClientToolCall, ClientToolResult};

/// Full model-facing conversation state for one language-model step.
///
/// The transcript is a neutral interchange format between the bot runtime and
/// provider crates. It is ordered, cloneable request input rather than a
/// complete persisted turn trace: usage, model-step traces, server tool use,
/// and viewer-only metadata are carried by neighboring API types.
#[derive(Debug, Clone, Default)]
pub struct Transcript {
    /// Stable transcript id.
    ///
    /// Providers may use this for prompt-cache routing or request correlation,
    /// but it is not a provider response id and should not be parsed for
    /// semantics.
    pub id: Option<String>,
    /// App-owned system/developer instructions for this transcript.
    ///
    /// Provider crates decide how to map these instructions into their native
    /// roles. For example, one provider may send them as a system prompt while
    /// another maps them to a developer instruction field.
    pub instructions: Option<String>,
    /// Ordered model-visible turns.
    ///
    /// The order of this vector is the order the backend should receive.
    pub turns: Vec<TranscriptTurn>,
}

impl Transcript {
    /// Build an empty transcript with no id, instructions, or turns.
    pub fn new() -> Self {
        Self::default()
    }

    /// Build a transcript containing a single user text turn.
    ///
    /// This is a narrow test/helper constructor. Production callers usually set
    /// instructions and append a history assembled from storage and platform
    /// context.
    pub fn from_user_text(text: impl Into<String>) -> Self {
        let mut transcript = Self::new();
        // Keep the helper behavior identical to hand-building a transcript and
        // pushing a single user message.
        transcript.push(TranscriptTurn::text(TurnRole::User, text.into()));
        transcript
    }

    /// Append one model-visible turn at the end of the transcript.
    ///
    /// This method does not coalesce adjacent roles or inspect block contents;
    /// callers own the final ordering and provider crates own any required
    /// native normalization.
    pub fn push(&mut self, turn: TranscriptTurn) {
        self.turns.push(turn);
    }
}

/// One model-facing turn in a transcript.
///
/// A turn is the unit that carries a role and an ordered list of content
/// blocks. The provider translation layer may need to split or merge blocks to
/// fit a native API, but it should preserve the logical order represented here.
#[derive(Debug, Clone)]
pub struct TranscriptTurn {
    /// Model-visible role for this turn.
    pub role: TurnRole,
    /// Ordered content blocks within the turn.
    ///
    /// Text, media, client tool calls/results, and continuation blocks are kept
    /// together because different providers support different native nesting.
    pub blocks: Vec<ContentBlock>,
    /// Opaque metadata owned by the caller or provider adapter.
    ///
    /// This is deliberately not interpreted by `chudbot-api`. Use it only for
    /// data that is meaningful to the component that created it; cross-crate
    /// contracts should be modeled as typed fields instead.
    pub metadata: serde_json::Value,
}

impl TranscriptTurn {
    /// Build a turn with one text block and no metadata.
    pub fn text(role: TurnRole, text: impl Into<String>) -> Self {
        Self {
            role,
            // Preserve text as a content block so callers can later compose it
            // with media, tool results, or continuations in the same turn.
            blocks: vec![ContentBlock::Text { text: text.into() }],
            metadata: serde_json::Value::Null,
        }
    }
}

/// Model-facing speaker role.
///
/// Keep this enum intentionally small. Platform authors may know about users,
/// bots, channels, or command invocations, but provider-facing chat APIs only
/// need the roles that distinguish user-supplied context from assistant/model
/// output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TurnRole {
    /// User-supplied context, including platform history and client tool
    /// results that are being returned to the model.
    User,
    /// Assistant/model output, including requested client tool calls and
    /// provider continuation blocks.
    Assistant,
}

/// One ordered piece of content inside a [`TranscriptTurn`].
///
/// Blocks are provider-neutral and intentionally mixed. Provider crates are
/// responsible for translating each variant into their native request shape,
/// including providers that represent tool calls outside normal message
/// content.
#[derive(Debug, Clone)]
pub enum ContentBlock {
    /// Plain UTF-8 text visible to the model.
    Text {
        /// Text content exactly as the model should receive it.
        text: String,
    },
    /// Runtime media reference visible to the model.
    ///
    /// This is a [`BoxedMediaRef`] handle, not a persisted media record.
    /// Providers can ask the handle for bytes or a public URL depending on
    /// their native upload/fetch requirements.
    Media {
        /// Media reference handle.
        media: BoxedMediaRef,
    },
    /// Assistant-requested client tool invocation.
    ///
    /// The bot runtime executes this through a [`crate::tool::ClientToolExecutor`]
    /// and later feeds a matching [`ContentBlock::ClientToolResult`] back to the
    /// model. Provider-side/server-side tools are traced separately and do not
    /// appear as transcript blocks.
    ClientToolCall(ClientToolCall),
    /// User-provided result for a prior assistant client tool invocation.
    ///
    /// The `tool_use_id` inside [`ClientToolResult`] ties this block back to the
    /// matching [`ContentBlock::ClientToolCall`].
    ClientToolResult(ClientToolResult),
    /// Opaque provider continuation state, replayed only to the provider that
    /// emitted it.
    ///
    /// Continuations are for provider-native state that must survive between
    /// model steps, such as encrypted reasoning or other resumable response
    /// handles. They are not user-visible content and should be ignored by
    /// other provider backends.
    Continuation(ProviderContinuation),
}

/// Opaque provider-specific state to replay on a later model step.
///
/// Providers return this through [`crate::llm::AssistantStep::continuation`].
/// The agent loop can append it to the transcript so the same provider can
/// continue from private state without forcing `chudbot-api` to understand the
/// provider's payload.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderContinuation {
    /// Provider that emitted this continuation.
    ///
    /// Provider adapters should only consume continuations matching their own
    /// backend name.
    pub provider: ProviderName,
    /// Provider-specific serialized payload.
    ///
    /// The shape is owned by the provider crate and may change with that
    /// provider's API. Shared behavior should graduate to a typed API contract
    /// instead of being inferred from this value.
    pub data: serde_json::Value,
}
