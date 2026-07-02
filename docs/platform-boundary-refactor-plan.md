# Platform Boundary Refactor Plan

Updated: 2026-07-02.

This document captures an implementation plan for moving Discord-specific
formatting, delivery policy, and interaction behavior out of `chudbot-bot` and
into the Discord platform adapter. It is a plan only. Do not treat it as
already implemented.

## Goal

Make `chudbot-bot` genuinely reusable across message platforms:

- `chudbot-bot` owns conversation orchestration, agent selection, transcript
  assembly, model/tool execution, storage lifecycle, and abstract retry/stop
  semantics.
- Platform crates own how a user addresses the bot, how mentions render, how a
  reply is formatted, how status/control affordances appear, and how messages
  are delivered to that platform.
- Discord keeps the current user-visible behavior after the refactor.
- Future platforms, including a debug TUI, can stream and render turn activity
  without pretending to be Discord.

## Non-Goals

- Do not implement the debug TUI in this refactor.
- Do not redesign the web trace viewer.
- Do not change provider APIs or agent behavior beyond forwarding existing
  streaming events to a platform-facing sink.
- Do not persist per-token deltas in the first pass.
- Do not rename every durable `guild` column or route. Storage compatibility is
  more important than cosmetic naming in this step.

## Current Boundary Leaks

The current platform boundary is neutral at the type level, but several
Discord assumptions still live above it.

- Inbound mention parsing and prompt text normalization live in
  `chudbot-bot/src/platform.rs`.
  - Bot mention stripping knows `<@id>` and `<@!id>`.
  - Non-bot mentions are rewritten as `Name (<@id>)`.
- Outgoing mention repair is Discord-specific.
  - `fix_bare_mentions` wraps snowflake-length `@123...` text in `<@...>`.
- Trace-link rendering is Discord-specific.
  - The first reply uses the `-#` Discord subtext prefix.
  - Prompt guidance says to provide a "Discord-friendly" line.
- Typing lifecycle is owned by `chudbot-bot`.
  - `spawn_typing_indicator` refreshes every eight seconds, matching Discord's
    transient typing indicator model.
- Thread behavior is decided by `chudbot-bot`.
  - Character/line thresholds, rendered-line estimation, thread title timing,
    and `ThreadRequest` are Discord-shaped presentation policy.
- System/status/control reactions are bot constants.
  - Working, success, error, retry, stop, and refused are all emoji glyphs.
  - This is a good Discord UI, but it is not the right abstraction for a TUI.
- Delivery limits are Discord-shaped.
  - `MAX_OUTGOING_ATTACHMENT_BYTES` uses Discord's default upload cap.
  - `suppress_embeds` is sent from bot code on every reply/status message.
- Tool descriptions and system prompt fragments mention Discord/server/guild
  directly.
  - Media access tools mention "current Discord guild icon".
  - Usage report mentions "server" and emits Discord user/channel mention
    markup.
  - Memory tools parse Discord mention strings for target users.
- The moderation fallback prompt says "private friends-only Discord server".

## Target Architecture

Keep the current low-level `MessagePlatform` concepts, but add a higher-level
platform presentation/delivery contract. The bot should no longer assemble a
Discord-ready final string and call `send_message` directly for turn output.

The platform layer should expose these responsibilities:

- Decide whether an inbound message addresses the bot.
- Normalize inbound message text for model-visible transcripts.
- Render user and channel references for platform-visible replies.
- Render the trace link line for the platform.
- Decide whether and how to show typing/activity.
- Decide how to expose turn controls such as retry and stop.
- Decide direct upload versus public URL fallback for outgoing media.
- Decide whether a new reply should move to a thread or equivalent surface.
- Optionally stream provisional output events to the user.

### Proposed API Shape

The exact names can change during implementation, but the shape should be close
to this.

```rust
pub struct PlatformMessageAdmission {
    pub should_handle: bool,
    pub normalized_content: String,
    pub addressed_bot: bool,
}

pub struct PlatformReplyContext {
    pub conversation_id: ConversationId,
    pub turn_id: TurnId,
    pub channel: ChannelRef,
    pub reply_to: MessageRef,
    pub is_new_conversation: bool,
    pub trace_url: String,
}

pub enum PlatformTurnEvent {
    Started,
    TextDelta {
        step: u32,
        item_id: String,
        delta: String,
    },
    ReasoningDelta {
        step: u32,
        item_id: String,
        delta: String,
    },
    ToolStarted {
        name: ToolName,
    },
    ToolFinished {
        name: ToolName,
        is_error: bool,
    },
    StatusText {
        content: String,
    },
    RetryAvailable,
    Stopped,
    Resumed,
}

pub struct PlatformFinalReply {
    pub text: String,
    pub media: Vec<MediaUri>,
    pub title_hint: Option<String>,
}
```

Platform registry methods should then support a turn delivery lifecycle:

```rust
fn admit_message(
    &self,
    message: &PlatformMessage,
    bot: &UserProfile,
) -> impl Future<Output = Result<PlatformMessageAdmission, Self::Error>> + Send;

fn begin_turn(
    &self,
    context: PlatformReplyContext,
) -> impl Future<Output = Result<PlatformTurnHandle, Self::Error>> + Send;

fn emit_turn_event(
    &self,
    handle: &PlatformTurnHandle,
    event: PlatformTurnEvent,
) -> impl Future<Output = Result<(), Self::Error>> + Send;

fn finish_turn(
    &self,
    handle: PlatformTurnHandle,
    reply: PlatformFinalReply,
) -> impl Future<Output = Result<PostedMessage, Self::Error>> + Send;
```

This can be introduced alongside the existing `send_message` path, then the
normal turn path can move over once Discord preserves existing behavior.

## Discord Behavior After Refactor

The Discord adapter should own the details currently spread across bot code:

- `admit_message`
  - Use Discord mention syntax to decide whether the bot was addressed.
  - Strip the bot mention from model-visible content.
  - Expand other mentioned users with Discord mention markup and display names.
- `begin_turn`
  - Add the working reaction.
  - Start typing refresh internally.
- `emit_turn_event`
  - Ignore token deltas for now.
  - Keep `post_status_message` visible as Discord replies.
  - Optionally map tool or reasoning events to logs only.
- `finish_turn`
  - Stop typing.
  - Apply Discord mention repair.
  - Append the Discord-formatted trace footer for new conversations.
  - Choose whether to open a thread.
  - Split long messages.
  - Suppress embeds.
  - Upload eligible attachments and use public URL fallback for oversized media.
  - Add success, error, refused, and retry reactions as appropriate.

`chudbot-bot` should only report semantic outcomes: completed, failed, refused,
cancelled, retry available, stopped, resumed.

## Implementation Phases

### Phase 1: Introduce Platform Rendering Helpers

- Add platform-registry methods for:
  - inbound message admission and normalized content;
  - trace-link rendering;
  - user/channel reference rendering.
- Implement them in Discord using the existing helper logic.
- Move `normalize_mention_content`, `strip_user_mention`,
  `fix_bare_mentions`, `full_trace_link_markdown`, and
  `trace_link_prompt_guidance` behind the platform registry.
- Keep existing `send_message` behavior for final replies.
- Add tests that prove Discord-rendered strings do not change.

### Phase 2: Move Turn Activity and Controls

- Replace direct bot calls to `add_unicode_reaction`,
  `remove_own_unicode_reaction`, and `spawn_typing_indicator` with semantic
  platform turn events.
- Let Discord map those events to typing and reactions.
- Let tests assert semantic calls at the bot boundary and concrete Discord
  behavior in the Discord adapter.

### Phase 3: Move Reply Delivery Policy

- Introduce a platform final-reply delivery method.
- Move Discord-specific thread threshold logic, rendered-line estimation,
  message splitting, embed suppression, and direct attachment limits into
  `chudbot-discord`.
- Keep title generation in `chudbot-bot`; the platform can request or accept a
  title hint, but it must not call LLM providers.
- Move `thread_threshold_chars` and `thread_threshold_lines` out of global
  `BotConfig` and into Discord platform config.
- Preserve backward compatibility in config parsing with diagnostics for old
  keys during one transition window.

### Phase 4: Stream Agent Events to Platform Delivery

- Stop using `collect_agent_run` directly in `execute_turn`.
- Consume `AgentRunEvent` in a local reducer that:
  - forwards `ModelStepEvent::Delta` as provisional platform turn events;
  - still produces the same final `AgentRun`;
  - persists model/tool traces at the same durable boundaries as today.
- Label streamed text as provisional. A provider step that streams text can
  still finish as a tool-call step, so consumers must reconcile against the
  final turn outcome.
- Discord should buffer or ignore token deltas in this phase.
- A debug TUI can render token deltas immediately.

### Phase 5: Platform-Specific Tool and Prompt Vocabulary

- Move platform-specific tool descriptions into a platform vocabulary surface.
- Replace hard-coded "Discord", "server", "guild", and mention-markup text in
  runtime prompts with platform-provided terms.
- Let usage report rows ask the platform to render user/channel references.
- Keep memory storage keys neutral; only user-facing descriptions should vary.

### Phase 6: Naming Cleanup

- Keep durable `guild_id` fields for compatibility unless there is a concrete
  migration reason to rename them.
- In new public API docs and comments, prefer "workspace" or
  "workspace/server" over "guild" when the concept is not Discord-specific.
- Introduce helper names such as `workspace_key` at new call sites while leaving
  existing storage columns alone.

## Testing Plan

- Unit tests for Discord admission:
  - bot mention stripping;
  - non-bot mention annotation;
  - messages without mentions ignored unless audio wake rules apply.
- Unit tests for Discord reply rendering:
  - trace footer unchanged;
  - bare mention repair unchanged;
  - public URL fallback unchanged for oversized media.
- Bot-runtime tests with a fake semantic platform:
  - completed turn emits started, deltas, finish;
  - failed turn emits retry available;
  - cancelled turn emits stopped/cancelled state;
  - bot does not call Discord-specific helpers.
- Discord adapter tests for:
  - thread decision;
  - reaction mapping;
  - attachment limit behavior;
  - command response behavior remains unchanged.
- Existing storage and trace-viewer tests should pass without schema changes.

## Migration Notes

- Keep `MessagePlatform::send_message` available for command responses and
  narrow platform tools until turn delivery fully moves to the new lifecycle.
- Do not make platform adapters depend on `chudbot-bot`. New contracts belong
  in `chudbot-api`.
- Avoid broad trait-object service bags. Continue using statically dispatched
  registries and native async trait methods.
- Any config movement must preserve rich `check-config` diagnostics.

## Acceptance Criteria

- `chudbot-bot` no longer knows Discord mention syntax.
- `chudbot-bot` no longer formats Discord trace-link footers.
- `chudbot-bot` no longer owns typing refresh or Discord status reactions.
- `chudbot-bot` no longer decides Discord thread creation or Discord upload
  size limits.
- Discord user-visible behavior is preserved.
- A non-Discord platform can receive semantic turn events and produce a usable
  chat experience without emulating Discord message syntax.

## Open Questions

- Should the abstract platform control surface include retry/stop as required
  capabilities, or should platforms opt into them individually?
- Should `post_status_message` remain a model tool, or become an ordinary
  semantic turn event produced by bot/tool execution?
- Should thread-title generation happen before final model output for platforms
  that can open a reply surface early?
- How much platform vocabulary belongs in prompts versus structured
  `message_context` JSON?
