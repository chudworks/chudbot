//! Runtime constants shared across the bot orchestration modules.

use std::time::Duration;

pub(crate) const FETCH_MESSAGES_TOOL: &str = "fetch_messages";
pub(crate) const GENERATE_IMAGE_TOOL: &str = "generate_image";
pub(crate) const GENERATE_VIDEO_TOOL: &str = "generate_video";
pub(crate) const TRANSCRIBE_AUDIO_TOOL: &str = "transcribe_audio";
pub(crate) const READ_ASSET_TOOL: &str = "read";
pub(crate) const STAT_ASSET_TOOL: &str = "stat";
pub(crate) const PUBLIC_URL_ASSET_TOOL: &str = "public_url";
pub(crate) const ATTACH_ASSET_TOOL: &str = "attach";
pub(crate) const POST_STATUS_TOOL: &str = "post_status_message";
pub(crate) const ADD_REACTION_TOOL: &str = "add_reaction";
pub(crate) const USAGE_REPORT_TOOL: &str = "usage_report";
pub(crate) const WORKING_REACTION: &str = "👀";
pub(crate) const SUCCESS_REACTION: &str = "✅";
pub(crate) const ERROR_REACTION: &str = "❌";
pub(crate) const RETRY_REACTION: &str = "🔄";
pub(crate) const STOP_REACTION: &str = "🛑";
pub(crate) const REFUSED_REACTION: &str = "❓";
pub(crate) const RESERVED_TOOL_REACTIONS: &[&str] = &[
    WORKING_REACTION,
    SUCCESS_REACTION,
    ERROR_REACTION,
    RETRY_REACTION,
    STOP_REACTION,
    REFUSED_REACTION,
];
pub(crate) const DEFAULT_SHUTDOWN_DRAIN_TIMEOUT: Duration = Duration::from_secs(30);
pub(crate) const TYPING_REFRESH_INTERVAL: Duration = Duration::from_secs(8);
// Discord's default per-file upload limit is 10 MiB; larger generated media is
// linked by public URL instead of uploaded.
pub(crate) const MAX_OUTGOING_ATTACHMENT_BYTES: usize = 10 * 1024 * 1024;
pub(crate) const HISTORY_SIZE_MIN: i64 = 1;
pub(crate) const HISTORY_SIZE_MAX: i64 = 100;
pub(crate) const TITLE_MAX_CHARS: usize = 80;
pub(crate) const TITLE_MAX_TOKENS: u32 = 96;
pub(crate) const TOS_PREFLIGHT_AGENT: &str = "tos_preflight";
pub(crate) const CONVERSATION_TITLE_AGENT: &str = "conversation_title";
pub(crate) const DEFAULT_THREAD_THRESHOLD_CHARS: usize = 1500;
pub(crate) const DEFAULT_THREAD_THRESHOLD_LINES: usize = 20;
pub(crate) const THREAD_REPLY_WRAP_WIDTH: usize = 80;
pub(crate) const MAX_REACTION_EMOJI_SCALARS: usize = 32;
pub(crate) const MODEL_TRANSCRIPT_IMAGE_MIME_TYPES: &[&str] = &[
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
    "image/x-icon",
    "image/vnd.microsoft.icon",
];

pub(crate) const TITLE_SYSTEM_PROMPT: &str = "You write very short conversation titles. \
Output ONLY a title for the conversation below: five words or fewer, no quotes, \
no period, no leading 'Re:' or 'Conversation about'. Just the title text.";

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
