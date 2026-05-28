// Mirrors the Rust `grok_discord_bot_core::domain` types serialized
// over the JSON API. Keep field names in sync with the Rust structs
// (serde uses Rust field names by default).

export interface ConversationView {
  conversation: Conversation;
  turns: TurnView[];
  /** Map of discord_user_id -> DiscordUser (string keys because JSON
   * object keys are always strings, even when the value semantically
   * represents an integer). */
  users: Record<string, DiscordUser>;
}

export interface Conversation {
  id: string;
  created_at: string;
  discord_guild_id: number;
  discord_channel_id: number;
  created_by_user_id: number;
  root_discord_message_id: number;
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
  user_discord_message_id: number;
  user_content: string;
  assistant_discord_message_id: number | null;
  assistant_content: string | null;
  status: 'pending' | 'completed' | 'failed' | string;
  error: string | null;
  persona_name: string | null;
  discord_user_id: number | null;
  discord_user_name: string | null;
}

export interface ContextItem {
  position: number;
  source: string;
  role: string;
  content: string;
  discord_message_id: number | null;
}

export interface ToolCallRecord {
  tool_name: string;
  request: unknown;
  response: unknown;
}

export interface DiscordUser {
  id: number;
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
