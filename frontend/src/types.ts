// Mirrors the Rust `grok_discord_bot_core::domain` types serialized
// over the JSON API. Keep field names in sync with the Rust structs
// (serde uses Rust field names by default).
//
// Every Discord snowflake ID is a `string`, not a `number`. Snowflakes
// are 64-bit integers (~10^18) that exceed JS's Number.MAX_SAFE_INTEGER
// (2^53), so the backend serializes them as JSON strings — a number
// would be silently rounded by `JSON.parse` and never match the
// string-keyed `users` map. Treat IDs as opaque strings; never do
// arithmetic on them.

export interface ConversationView {
  conversation: Conversation;
  turns: TurnView[];
  /** Map of discord_user_id -> DiscordUser. Keys are snowflake strings
   * (JSON object keys are always strings, and the backend emits the
   * matching ID fields as strings too, so lookups line up exactly). */
  users: Record<string, DiscordUser>;
}

export interface Conversation {
  id: string;
  created_at: string;
  discord_guild_id: string;
  discord_channel_id: string;
  created_by_user_id: string;
  root_discord_message_id: string;
  title: string | null;
  title_generated_at: string | null;
  model: string;
}

export interface TurnView {
  turn: Turn;
  context: ContextItem[];
  tool_calls: ToolCallRecord[];
}

export interface Turn {
  id: string;
  conversation_id: string;
  turn_index: number;
  created_at: string;
  completed_at: string | null;
  user_discord_message_id: string;
  user_content: string;
  assistant_discord_message_id: string | null;
  assistant_content: string | null;
  status: 'pending' | 'completed' | 'failed' | string;
  error: string | null;
  persona_name: string | null;
  discord_user_id: string | null;
  discord_user_name: string | null;
}

export interface ContextItem {
  position: number;
  source: string;
  role: string;
  content: string;
  discord_message_id: string | null;
}

export interface ToolCallRecord {
  tool_name: string;
  request: unknown;
  response: unknown;
}

export interface DiscordUser {
  id: string;
  username: string;
  display_name: string | null;
  avatar_hash: string | null;
  avatar_local_path: string | null;
  last_avatar_fetched_at: string | null;
  last_seen_at: string;
}

/** Names of SSE events emitted by the backend. Kept in sync with the
 *  match arm in `crates/grok-discord-bot-bin/src/web.rs::event_payload`. */
export type ServerEventName =
  | 'created'
  | 'turn_started'
  | 'turn_updated'
  | 'tool_call_recorded'
  | 'context_item_added'
  | 'title_updated'
  | 'user_avatar_updated'
  | 'lag';
