// Mirrors the v2 `chudbot_api` DTOs serialized by `chudbot-web`.
// Provider/platform/model ids are opaque strings. Do not parse or do
// arithmetic on Discord snowflakes or platform ids.

export interface ConversationSnapshot {
  conversation: Conversation;
  turns: TurnSnapshot[];
  users: Record<string, UserMetadata>;
}

export interface Conversation {
  id: string;
  created_at: string;
  channel: ChannelRef;
  created_by: UserRef;
  root_message: MessageRef;
  initial_model: string;
  agent_name: string;
  provider: string;
  system_instructions: string;
  title: string | null;
  stopped_at: string | null;
  stopped_by: UserRef | null;
}

export interface TurnSnapshot {
  turn: Turn;
  system_instructions: string | null;
  context: ContextItem[];
  tool_trace: ToolTrace[];
  replay_assets: TurnAsset[];
  usage: UsageRecord[];
}

export interface Turn {
  id: string;
  ordinal: number;
  history_cutoff: number | null;
  response_ordinal: number | null;
  created_at: string;
  user_message_created_at: string;
  completed_at: string | null;
  user_message: MessageRef;
  user: UserRef;
  user_display_name: string;
  user_content: string;
  assistant_message: MessageRef | null;
  assistant_content: string | null;
  status: 'pending' | 'completed' | 'failed' | 'cancelled' | string;
  error: string | null;
  agent_name: string | null;
  provider: string | null;
  model: string | null;
  app_version_id: number | null;
}

export interface ContextItem {
  position: number;
  source: string;
  role: string;
  content: string;
  message: MessageRef | null;
}

export interface TurnAsset {
  uri: string;
  turn_id: string;
  source: string;
  mime_type: string | null;
}

export interface UserRef {
  platform: string;
  guild_id: string | null;
  user_id: string;
}

export interface UserMetadata {
  id: UserRef;
  username: string;
  display_name: string | null;
  label: string;
  avatar_url: string | null;
  avatar_media_uri: string | null;
  is_bot: boolean;
}

export interface ChannelRef {
  platform: string;
  guild_id: string | null;
  channel_id: string;
}

export interface MessageRef {
  platform: string;
  guild_id: string | null;
  channel_id: string;
  message_id: string;
}

export type ToolTrace =
  | { kind: 'client'; trace: ClientToolTrace }
  | { kind: 'server'; tool: ServerToolUse }
  | { kind: 'grounding'; metadata: GroundingMetadata };

export interface ClientToolTrace {
  call: ClientToolCall;
  result: ClientToolResult;
  trace_response: unknown;
  usage: UsageRecord[];
}

export interface ClientToolCall {
  id: string;
  name: string;
  input: unknown;
}

export interface ClientToolResult {
  tool_use_id: string;
  content: ClientToolResultContent;
  is_error: boolean;
}

export type ClientToolResultContent =
  | { kind: 'json'; value: unknown }
  | { kind: 'text'; text: string };

export interface ServerToolUse {
  provider: string;
  name: string;
  id: string | null;
  status: string | null;
  raw: unknown;
  usage: UsageRecord[];
}

export interface GroundingMetadata {
  provider: string;
  raw: unknown;
}

export interface UsageRecord {
  kind: string;
  provider: string;
  model?: string | null;
  subject?: string | null;
  input_tokens?: number | null;
  output_tokens?: number | null;
  total_tokens?: number | null;
  reasoning_tokens?: number | null;
  cached_input_tokens?: number | null;
  cost?: unknown;
  raw?: unknown;
}

export interface SiteConfig {
  title_prefix: string;
  version: string;
}

export type ServerEventName =
  | 'created'
  | 'turn_started'
  | 'turn_updated'
  | 'tool_trace_recorded'
  | 'context_recorded'
  | 'title_updated'
  | 'conversation_updated'
  | 'user_profile_updated'
  | 'lag';
