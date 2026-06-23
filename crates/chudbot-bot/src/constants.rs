//! Canonical names, limits, and system prompts shared by bot orchestration.
//!
//! Most string values here are runtime contracts rather than local labels:
//! providers see tool names, operators may configure reserved agent names, and
//! tests assert user-facing reaction behavior. Change them only when the
//! corresponding executor, config, and stored-trace expectations change too.

use std::time::Duration;

/// Model-visible client tool name for conversation history retrieval.
pub(crate) const FETCH_MESSAGES_TOOL: &str = "fetch_messages";

/// Model-visible client tool name for configured image generation.
pub(crate) const GENERATE_IMAGE_TOOL: &str = "generate_image";

/// Model-visible client tool name for configured video generation.
pub(crate) const GENERATE_VIDEO_TOOL: &str = "generate_video";

/// Model-visible client tool name for configured audio transcription.
pub(crate) const TRANSCRIBE_AUDIO_TOOL: &str = "transcribe_audio";

/// Model-visible media-store tool name for reading supported stored assets.
pub(crate) const READ_ASSET_TOOL: &str = "read";

/// Model-visible media-store tool name for inspecting stored asset metadata.
pub(crate) const STAT_ASSET_TOOL: &str = "stat";

/// Model-visible media-store tool name for resolving a stored asset URL.
pub(crate) const PUBLIC_URL_ASSET_TOOL: &str = "public_url";

/// Model-visible media-store tool name for attaching a stored asset to the reply.
pub(crate) const ATTACH_ASSET_TOOL: &str = "attach";

/// Model-visible client tool name for posting short turn progress messages.
pub(crate) const POST_STATUS_TOOL: &str = "post_status_message";

/// Model-visible client tool name for reacting to the current user message.
pub(crate) const ADD_REACTION_TOOL: &str = "add_reaction";

/// Model-visible client tool name for generating usage and cost summaries.
pub(crate) const USAGE_REPORT_TOOL: &str = "usage_report";

/// Reaction used while a turn or retry is actively running.
pub(crate) const WORKING_REACTION: &str = "👀";

/// Reaction used when a turn completes successfully.
pub(crate) const SUCCESS_REACTION: &str = "✅";

/// Reaction used when a turn fails.
pub(crate) const ERROR_REACTION: &str = "❌";

/// User-facing reaction that requests a retry from the reacted message.
pub(crate) const RETRY_REACTION: &str = "🔄";

/// User-facing reaction that requests cancellation of an active turn.
pub(crate) const STOP_REACTION: &str = "🛑";

/// Reaction used when the moderation preflight refuses a turn.
pub(crate) const REFUSED_REACTION: &str = "❓";

/// Reactions that the `add_reaction` tool is not allowed to emit.
///
/// The bot owns these glyphs as status or control affordances. Blocking them
/// keeps model-selected reactions from looking like system state transitions.
pub(crate) const RESERVED_TOOL_REACTIONS: &[&str] = &[
    WORKING_REACTION,
    SUCCESS_REACTION,
    ERROR_REACTION,
    RETRY_REACTION,
    STOP_REACTION,
    REFUSED_REACTION,
];

/// Default time allowed for in-flight work to drain during shutdown.
pub(crate) const DEFAULT_SHUTDOWN_DRAIN_TIMEOUT: Duration = Duration::from_secs(30);

/// Interval used to refresh typing indicators during long model turns.
pub(crate) const TYPING_REFRESH_INTERVAL: Duration = Duration::from_secs(8);

/// Maximum attachment size uploaded directly to Discord.
///
// Discord's default per-file upload limit is 10 MiB; larger generated media is
// linked by public URL instead of uploaded.
pub(crate) const MAX_OUTGOING_ATTACHMENT_BYTES: usize = 10 * 1024 * 1024;

/// Maximum stored/displayed conversation title length after title generation.
pub(crate) const TITLE_MAX_CHARS: usize = 80;

/// Output-token cap for the title-generation system agent.
pub(crate) const TITLE_MAX_TOKENS: u32 = 96;

/// Reserved agent config key for the moderation preflight system agent.
pub(crate) const TOS_PREFLIGHT_AGENT: &str = "tos_preflight";

/// Reserved agent config key for the conversation-title system agent.
pub(crate) const CONVERSATION_TITLE_AGENT: &str = "conversation_title";

/// Default character threshold for moving a new Discord reply into a thread.
pub(crate) const DEFAULT_THREAD_THRESHOLD_CHARS: usize = 1500;

/// Default rendered-line threshold for moving a new Discord reply into a thread.
pub(crate) const DEFAULT_THREAD_THRESHOLD_LINES: usize = 20;

/// Width used to estimate rendered line count for thread-threshold decisions.
pub(crate) const THREAD_REPLY_WRAP_WIDTH: usize = 80;

/// Scalar-value cap for a single standard Unicode emoji reaction sequence.
pub(crate) const MAX_REACTION_EMOJI_SCALARS: usize = 32;

/// Image MIME types that can be replayed into provider model transcripts.
///
/// Public URLs may support broader media classes; this list is only for media
/// that is safe to present back to the model as image content.
pub(crate) const MODEL_TRANSCRIPT_IMAGE_MIME_TYPES: &[&str] = &[
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
    "image/x-icon",
    "image/vnd.microsoft.icon",
];

/// System prompt for the background agent that names conversations.
///
/// The title code separately trims and truncates output; the prompt keeps the
/// model contract simple enough that common providers return plain title text.
pub(crate) const TITLE_SYSTEM_PROMPT: &str = "You write very short conversation titles. \
Output ONLY a title for the conversation below: five words or fewer, no quotes, \
no period, no leading 'Re:' or 'Conversation about'. Just the title text.";

/// System prompt for the background moderation preflight.
///
/// The classifier intentionally defaults to `ALLOW` and only returns one token,
/// which keeps borderline private-server chatter from becoming a hard failure
/// unless it falls into one of the narrow refusal classes listed below.
pub(crate) const MODERATION_PROMPT: &str = "You are a TOS compliance classifier for a \
private friends-only Discord server. Each message you classify is prefixed \
with the sender's display name as `[name]: `. The DEFAULT IS ALLOW. Only \
REFUSE the narrowly listed categories below.

REFUSE these:
- CSAM or any sexualization of minors
- Doxxing: sharing someone's non-public personal info with apparent intent to harm
- Credible, specific threats of violence against a real, identifiable person
- Coordinated incitement to suicide or self-harm directed at a specific person
- Illegal arrangements: drug or weapon sales, human trafficking, exploitation rings
- Malware, phishing payloads, or coordinated large-scale spam campaigns

ALLOW EVERYTHING ELSE, including profanity, insults, dark humor, political \
opinions, criticism of public figures, news/current-events questions, and \
edgy art requests that do not involve minors.

When in any doubt, ALLOW.

Respond with EXACTLY one token: ALLOW or REFUSE. No punctuation. No explanation.";
