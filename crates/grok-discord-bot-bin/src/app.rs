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

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

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
    /// A conversation-level field other than the title changed — today
    /// the admin 🛑 stop/resume flag (`stopped_at`). The viewer refetches
    /// and re-renders the stopped banner.
    ConversationUpdated,
    /// An avatar was downloaded and pinned for `user_id`. Any rendered
    /// reference to that user should refresh its image. Carries the
    /// user id so consumers can scope without a full conversation
    /// refetch when they care.
    UserAvatarUpdated {
        /// Discord user id whose avatar changed on disk.
        user_id: i64,
    },
}

/// Registry of cancellation tokens for in-flight turns, so an admin 🛑
/// reaction can abort a turn that's mid-LLM-call or mid-tool-execution —
/// not just block the *next* mention. Keyed by turn id; each entry also
/// records its conversation so a single stop cancels every turn running
/// in that conversation at once (rare, but possible if a user fires two
/// mentions back-to-back).
///
/// Cancelling a token drops the awaited agent-loop future, which in turn
/// drops the in-flight `reqwest` request (aborting the HTTP call) and any
/// running tool futures — Rust async cancellation is just future-drop.
/// Tokens are deliberately NOT children of [`AppState::cancel`]: Ctrl+C
/// should let in-flight turns *drain* (the existing behavior), whereas a
/// 🛑 should *abort* — different intents, separate tokens.
#[derive(Debug, Default)]
pub struct TurnCancellations {
    inner: Mutex<HashMap<Uuid, (Uuid, CancellationToken)>>,
}

impl TurnCancellations {
    /// Register a fresh token for a running turn and return a guard. Hold
    /// the guard for the cancellable region (the agent loop); dropping it
    /// deregisters, so a completed or panicking turn never leaves a stale
    /// entry. Call [`TurnCancelGuard::token`] to get the handle to select
    /// on.
    pub fn register(self: &Arc<Self>, conversation_id: Uuid, turn_id: Uuid) -> TurnCancelGuard {
        let token = CancellationToken::new();
        self.inner
            .lock()
            .expect("turn-cancellation registry mutex poisoned")
            .insert(turn_id, (conversation_id, token.clone()));
        TurnCancelGuard {
            registry: Arc::clone(self),
            turn_id,
            token,
        }
    }

    /// Cancel every in-flight turn belonging to `conversation_id`.
    /// Returns how many tokens were fired (0 when nothing is running).
    pub fn cancel_conversation(&self, conversation_id: Uuid) -> usize {
        let guard = self
            .inner
            .lock()
            .expect("turn-cancellation registry mutex poisoned");
        let mut n = 0;
        for (conv, token) in guard.values() {
            if *conv == conversation_id {
                token.cancel();
                n += 1;
            }
        }
        n
    }

    fn deregister(&self, turn_id: Uuid) {
        self.inner
            .lock()
            .expect("turn-cancellation registry mutex poisoned")
            .remove(&turn_id);
    }
}

/// RAII handle returned by [`TurnCancellations::register`]. Deregisters
/// its turn on drop. Exposes the [`CancellationToken`] to select on while
/// the turn runs.
#[derive(Debug)]
pub struct TurnCancelGuard {
    registry: Arc<TurnCancellations>,
    turn_id: Uuid,
    token: CancellationToken,
}

impl TurnCancelGuard {
    /// The token to `select!` on; fires when an admin stops the
    /// conversation this turn belongs to.
    pub fn token(&self) -> CancellationToken {
        self.token.clone()
    }
}

impl Drop for TurnCancelGuard {
    fn drop(&mut self) {
        self.registry.deregister(self.turn_id);
    }
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
    /// Hard-coded operator admin Discord user ids (from config's
    /// top-level `admins`). These users can pause/resume the bot in a
    /// conversation via the 🛑 reaction. See [`Self::is_admin`].
    pub admins: HashSet<u64>,
    /// Ordered "vN" version number for the running build, resolved once
    /// at startup from `app_versions` (see `Db::register_app_version`).
    /// Stamped onto every turn and surfaced in the operational block of
    /// the system prompt.
    pub app_version: i32,
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
    /// Whether the web access logger trusts `X-Forwarded-For` for the
    /// client IP. `true` behind a trusted proxy (Cloudflare tunnel);
    /// `false` falls back to the TCP peer address. See
    /// [`grok_discord_bot_core::WebConfig::trust_forwarded_for`].
    pub web_trust_forwarded_for: bool,
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
    /// In-flight turn cancellation tokens, so an admin 🛑 reaction aborts
    /// a turn that's mid-flight (LLM call / tool execution) instead of
    /// only blocking the next mention. See [`TurnCancellations`].
    pub turn_cancellations: Arc<TurnCancellations>,
}

impl AppState {
    /// Whether `user_id` is a configured operator admin (the 🛑
    /// stop-sign kill-switch). Returns `false` when no admins are
    /// configured, so the feature is inert until the operator opts in.
    pub fn is_admin(&self, user_id: u64) -> bool {
        self.admins.contains(&user_id)
    }

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

#[cfg(test)]
mod tests {
    use super::*;

    // Cancelling a conversation fires only the tokens for turns in THAT
    // conversation, leaving turns in other conversations untouched.
    #[test]
    fn cancel_conversation_is_scoped() {
        let reg: Arc<TurnCancellations> = Arc::default();
        let conv_a = Uuid::new_v4();
        let conv_b = Uuid::new_v4();

        let a1 = reg.register(conv_a, Uuid::new_v4());
        let a2 = reg.register(conv_a, Uuid::new_v4());
        let b1 = reg.register(conv_b, Uuid::new_v4());

        let fired = reg.cancel_conversation(conv_a);
        assert_eq!(fired, 2, "both turns in conv_a should be cancelled");
        assert!(a1.token().is_cancelled());
        assert!(a2.token().is_cancelled());
        assert!(!b1.token().is_cancelled(), "conv_b must be untouched");
    }

    // The guard deregisters on drop, so a completed turn leaves no stale
    // entry that a later stop would try to cancel.
    #[test]
    fn guard_deregisters_on_drop() {
        let reg: Arc<TurnCancellations> = Arc::default();
        let conv = Uuid::new_v4();
        {
            let _guard = reg.register(conv, Uuid::new_v4());
            assert_eq!(reg.inner.lock().unwrap().len(), 1);
        }
        assert_eq!(reg.inner.lock().unwrap().len(), 0, "drop must deregister");
        assert_eq!(reg.cancel_conversation(conv), 0);
    }

    // Cancelling a conversation with nothing in flight is a no-op.
    #[test]
    fn cancel_with_nothing_in_flight() {
        let reg: Arc<TurnCancellations> = Arc::default();
        assert_eq!(reg.cancel_conversation(Uuid::new_v4()), 0);
    }
}
