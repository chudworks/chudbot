//! Shared application state and live-update event bus.
//!
//! `AppState` holds everything both halves of the binary — the Discord
//! gateway loop in [`crate::bot`] and the Axum API server in
//! [`crate::web`] — need access to. It's wrapped in `Arc` and passed by
//! clone into spawn points.
//!
//! The event bus is a [`tokio::sync::broadcast`] channel typed on
//! [`ConversationEvent`]. The bot publishes on every persisted change
//! (turn started, completed, tool call recorded, title generated,
//! avatar updated, etc); the web layer subscribes per-SSE-connection
//! and filters by `conversation_id`. Receivers that lag past the
//! channel capacity simply miss events — the client refetches the
//! whole conversation on each event anyway, so a missed notification
//! at worst delays a refresh until the next event arrives.
//!
//! Lifecycle hooks ([`CancellationToken`] + [`TaskTracker`]) live here
//! so background work spawned in either half participates in graceful
//! shutdown without each spawner needing to plumb the handles
//! separately.

use std::collections::HashMap;
use std::path::PathBuf;

use tokio::sync::broadcast;
use tokio_util::sync::CancellationToken;
use tokio_util::task::TaskTracker;
use uuid::Uuid;

use grok_discord_bot_core::{
    AnyImageProvider, AnyProvider, AnyVideoProvider, Db, ImageProviderKind, LlmProviderKind,
    Persona, PrivacyMode, StorageConfig, VideoProviderKind,
};

/// Capacity of the broadcast channel that fans out [`ConversationEvent`]s
/// to SSE consumers. Sized generously — events are cheap (a UUID + a
/// small enum) and most consumers will keep up. A consumer that lags
/// past this many events misses them, which is fine because the
/// frontend refetches the whole conversation on each event it sees.
const EVENT_CHANNEL_CAPACITY: usize = 256;

/// A change to a conversation that the web viewer should know about.
/// Sent over [`AppState::events`] after the writer has committed the
/// underlying DB change. The kind is a hint to consumers; in practice
/// the frontend re-fetches `/api/conversations/<id>` on any event for
/// the conversation it's viewing, so a missed kind discriminator just
/// means an unnecessary refetch.
#[derive(Debug, Clone, Copy)]
pub struct ConversationEvent {
    /// Which conversation changed.
    pub conversation_id: Uuid,
    /// What kind of change.
    pub kind: EventKind,
}

/// Hint about what kind of change a [`ConversationEvent`] represents.
/// Consumers can use it to skip refetches they don't care about, but
/// the contract is "any event means *something* changed under
/// `/api/conversations/<id>`".
#[derive(Debug, Clone, Copy)]
pub enum EventKind {
    /// A new conversation row was inserted.
    Created,
    /// A turn was inserted (status = pending).
    TurnStarted,
    /// A turn row was updated (status, persona_name, assistant_content, …).
    TurnUpdated,
    /// A row was appended to `tool_calls`.
    ToolCallRecorded,
    /// A row was appended to `context_items`.
    ContextItemAdded,
    /// `conversations.title` was set by the background titler.
    TitleUpdated,
    /// An avatar was downloaded and pinned for `user_id`. Any rendered
    /// reference to that user should refresh its image. Carries the
    /// user id so consumers can scope without a full conversation
    /// refetch when they care.
    UserAvatarUpdated {
        /// Discord user id whose avatar changed on disk.
        user_id: i64,
    },
}

/// Application-wide state shared between bot and web halves.
///
/// Cheap to clone: every field is either `Arc`-shared internally
/// (Db, broadcast::Sender, CancellationToken, TaskTracker, reqwest::Client)
/// or `Clone`-by-value. The struct is moved into an `Arc` once at
/// startup and shared by clone.
#[derive(Debug)]
pub struct AppState {
    /// Postgres pool wrapper.
    pub db: Db,
    /// One LLM provider per configured kind. Personas key into this map
    /// at turn time.
    pub providers: HashMap<LlmProviderKind, AnyProvider>,
    /// Image generation backends keyed by kind. Empty if no
    /// `[image.<kind>]` block was configured.
    pub image_providers: HashMap<ImageProviderKind, AnyImageProvider>,
    /// Video generation backends keyed by kind. Same shape as `image_providers`.
    pub video_providers: HashMap<VideoProviderKind, AnyVideoProvider>,
    /// Named personas. Looked up by name on every turn.
    pub personas: HashMap<String, Persona>,
    /// Floor fallback persona name.
    pub default_persona: String,
    /// Per-guild bootstrap default privacy mode.
    pub default_privacy: PrivacyMode,
    /// Public base URL of the viewer; used to build the link the bot
    /// posts into Discord on new conversations.
    pub web_base_url: String,
    /// Filesystem directory the Axum server serves the React bundle
    /// from. Vite's `dist/` is copied here by `serve.sh deploy`.
    pub web_frontend_dir: PathBuf,
    /// Prefix the viewer prepends to every browser tab title. Surfaced
    /// to the frontend via `/api/config`.
    pub web_title_prefix: String,
    /// Optional on-disk favicon served at `/favicon.ico`. `None` means
    /// the route 404s and browsers show their default icon.
    pub web_favicon_path: Option<PathBuf>,
    /// Media storage settings (images, videos, avatars dirs).
    pub storage: StorageConfig,
    /// Operator-supplied global system-prompt addendum (e.g. Discord
    /// ToS), appended to every persona's composed system prompt. `None`
    /// when the operator didn't configure one.
    pub extra_system_prompt: Option<String>,
    /// HTTP client used by background tasks (avatar fetcher, image
    /// downloads, etc). Separate from twilight's Discord client.
    pub download_http: reqwest::Client,
    /// Live-update event sender. Clone via `subscribe()` to consume.
    pub events: broadcast::Sender<ConversationEvent>,
    /// Top-level cancellation token. Background tasks should select on
    /// `cancel.cancelled()` to exit promptly on Ctrl+C.
    pub cancel: CancellationToken,
    /// Tracks spawned background tasks so the shutdown handler can
    /// wait for them to drain.
    pub tracker: TaskTracker,
}

impl AppState {
    /// Publish a [`ConversationEvent`] to all active SSE subscribers.
    /// Returns silently when there are no subscribers (which is the
    /// common case — most events fire while nobody's looking).
    pub fn publish(&self, conversation_id: Uuid, kind: EventKind) {
        // broadcast::Sender::send returns Err only when there are zero
        // receivers, which is the normal "nobody's watching" case. We
        // also don't care if some subscribers lag and miss this — the
        // frontend refetches on each event it does see.
        let _ = self.events.send(ConversationEvent {
            conversation_id,
            kind,
        });
    }
}

impl ConversationEvent {
    /// `true` for events that aren't scoped to a single conversation
    /// (e.g. an avatar update affects every viewer that renders that
    /// user). SSE handlers forward globals to every subscriber
    /// regardless of conversation filter.
    pub fn is_global(&self) -> bool {
        matches!(self.kind, EventKind::UserAvatarUpdated { .. })
    }
}

/// Build a fresh broadcast channel sized for our event volume.
pub fn new_event_channel() -> broadcast::Sender<ConversationEvent> {
    let (tx, _rx) = broadcast::channel(EVENT_CHANNEL_CAPACITY);
    tx
}
