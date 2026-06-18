# Streaming Agents Refactor Plan

Updated: 2026-06-18.

This document captures the agreed implementation direction for making Chudbot's
agent framework stream provider output and runtime activity while preserving the
current Discord and storage behavior through collection/reduction.

This is a plan only. Do not treat it as already implemented.

## Goal

Make streaming the primary execution contract:

- `LlmBackend::step` returns a stream of model-step events.
- `Agent::run` returns a stream of agent-run events.
- Discord, storage, background jobs, and other non-streaming consumers collect
  those streams into the existing stable results at their call sites.
- Future UIs can consume the same `Agent::run` stream directly to render
  token-by-token assistant output, reasoning summaries, tool progress, and final
  reconciliation.

The refactor should support extracting the agent runtime into a reusable library
for non-Discord projects, such as a platform that drives long-running agentic
movie re-encoding workflows.

## Non-Goals

- Do not add a separate non-streaming `step` or `run` method with a new primary
  name.
- Do not persist per-token deltas in the first pass.
- Do not put reasoning summaries, grounding, or provider/server-tool trace data
  into the model-facing `Transcript`.
- Do not make provider crates execute Chudbot client tools.
- Do not redesign the frontend or Discord reply behavior as part of the core
  API refactor.

## Current Shape

The current shared LLM contract is in `crates/chudbot-api/src/llm.rs`.

Today:

```rust
pub trait LlmBackend: Send + Sync {
    type Error: std::error::Error + Send + Sync + 'static;

    fn backend_name(&self) -> &ProviderName;

    fn step(
        &self,
        request: ModelStepRequest,
    ) -> impl Future<Output = Result<ModelStep, Self::Error>> + Send;
}
```

The current agent loop is in `crates/chudbot-api/src/agent.rs`.

Today:

```rust
pub async fn run(&self, transcript: Transcript) -> Result<AgentRun, AgentRunError<B::Error>>
```

The current transcript model is in `crates/chudbot-api/src/transcript.rs`.
Keep its role narrow: `Transcript` is the ordered model-facing replay/input
shape, not a complete event log.

## Design Principles

- Streaming events are the live observation channel.
- Collected results are the durable and compatibility channel.
- `Transcript` remains model-facing replay state.
- Provider output order should not be flattened into unrelated buckets.
- Client tool execution lifecycle belongs to the agent runtime, not providers.
- Provider-owned/server-side tool use and grounding belong to model-step events.
- Reasoning summaries should be typed trace data, not hidden inside opaque
  provider continuations when the provider exposes viewer-safe summaries.

## Target Data Model

### Ordered Collected Model Step

Refactor the collected model-step payload away from the current bucketed
`AssistantStep` shape.

Recommended replacement:

```rust
#[derive(Debug, Clone)]
pub struct ModelStepOutput {
    pub model_id: ModelId,
    pub items: Vec<ModelStepItem>,
    pub usage: Vec<UsageRecord>,
}

#[derive(Debug, Clone)]
pub enum ModelStepItem {
    /// Model-facing output produced by a provider step.
    OutputBlock(ModelOutputBlock),
    /// Viewer-safe reasoning summary metadata.
    Reasoning(ReasoningItem),
    /// Provider-owned hosted/server-side tool activity.
    ServerToolUse(ServerToolUse),
    /// Provider-owned citation or grounding metadata.
    Grounding(GroundingMetadata),
}
```

Then keep the existing control-flow enum shape, but replace the payload:

```rust
#[derive(Debug, Clone)]
pub enum ModelStep {
    Final {
        output: ModelStepOutput,
    },
    UseClientTools {
        output: ModelStepOutput,
    },
    Continue {
        output: ModelStepOutput,
    },
}
```

This preserves the most important semantic split:

- `Final`: return an assistant answer and stop the agent loop.
- `UseClientTools`: append assistant output, execute Chudbot client tools, append
  tool results, then call the model again.
- `Continue`: append provider continuation/replay state, then call the model
  again.

Add helper methods on `ModelStepOutput` and/or `ModelStep` so current call sites
do not need to repeatedly scan `items` by hand:

```rust
impl ModelStepOutput {
    pub fn transcript_blocks(&self) -> impl Iterator<Item = &ContentBlock>;
    pub fn client_tool_calls(&self) -> impl Iterator<Item = &ClientToolCall>;
    pub fn reasoning(&self) -> impl Iterator<Item = &ReasoningItem>;
    pub fn server_tool_uses(&self) -> impl Iterator<Item = &ServerToolUse>;
    pub fn grounding(&self) -> impl Iterator<Item = &GroundingMetadata>;
    pub fn continuation(&self) -> Option<&ProviderContinuation>;
    pub fn answer_text(&self) -> String;
}
```

The exact helper signatures can be adjusted for ownership, but keep call sites
simple and avoid reintroducing bucket fields.

### Model Step Events

Change `LlmBackend::step` to return streaming events for one provider/model
round trip:

```rust
use futures::Stream;

pub trait LlmBackend: Send + Sync {
    type Error: std::error::Error + Send + Sync + 'static;

    fn backend_name(&self) -> &ProviderName;

    fn step(
        &self,
        request: ModelStepRequest,
    ) -> impl Stream<Item = Result<ModelStepEvent, Self::Error>> + Send;

    fn fetch_model_info(
        &self,
        request: ModelInfoRequest,
    ) -> impl Future<Output = Result<Option<ModelInfo>, Self::Error>> + Send;
}
```

Recommended event shape:

```rust
#[derive(Debug, Clone)]
pub enum ModelStepEvent {
    /// Delta for one ordered output item.
    Delta(ModelStepDelta),
    /// Opaque provider continuation state.
    Continuation(ProviderContinuation),
    /// Provider-owned hosted/server-side tool activity.
    ServerToolUse(ServerToolUse),
    /// Provider-owned citation or grounding metadata.
    Grounding(GroundingMetadata),
    /// Usage/cost reported by the provider for this step.
    Usage(UsageRecord),
    /// Terminal control-flow classification for this provider step.
    Finished {
        kind: ModelStepKind,
        model_id: ModelId,
    },
}

#[derive(Debug, Clone)]
pub enum ModelStepDelta {
    Text {
        item_id: String,
        delta: String,
    },
    ReasoningSummary {
        item_id: String,
        provider: ProviderName,
        kind: Option<String>,
        delta: String,
    },
    ClientToolCall {
        item_id: String,
        id: ToolUseId,
        name: Option<ToolName>,
        arguments_delta: String,
    },
}
```

Notes:

- `ModelStepDelta` is the only event representation for model-produced text,
  reasoning summaries, and client tool-call intent.
- True streaming providers emit real provider deltas. Single-shot providers, or
  terminal fallback paths that did not observe provider deltas, emit one
  complete chunk as a delta and let the shared collector build the collected
  output.
- The collector owns assembling deltas into ordered `ModelStepItem` values.
- Provider continuations should be emitted as `ModelStepEvent::Continuation`.
- Provider-owned server tools and grounding should be emitted as
  `ModelStepEvent::ServerToolUse` and `ModelStepEvent::Grounding`.
- Client tool results are not model-step events. They are emitted by the agent
  runtime after it executes Chudbot client tools.

If a provider exposes stable output indexes or item ids, preserve them in
`item_id` or provider-owned metadata so the collector can update items in
first-seen order. If a provider does not expose ids, the adapter can generate
local step-scoped ids.

### Model Step Collection

Add a reducer in `chudbot-api`, implemented as the private `collector.rs`
module with `collect_model_step` as the public API:

```rust
pub async fn collect_model_step<S, E>(events: S) -> Result<ModelStep, E>
where
    S: Stream<Item = Result<ModelStepEvent, E>> + Send,
    E: std::error::Error + Send + Sync + 'static,
{
    // ...
}
```

Responsibilities:

- Maintain ordered pending output items in first-seen order.
- Accumulate `Text` deltas into `ModelOutputBlock::Text`.
- Accumulate `ReasoningSummary` deltas into `ReasoningItem` /
  `ReasoningSummary`.
- Accumulate `ClientToolCall` deltas into a complete `ClientToolCall`.
- Append `Continuation`, `ServerToolUse`, and `Grounding` events to the ordered
  collected output without pretending they are text/content deltas.
- Append `Usage` events to `ModelStepOutput::usage`.
- Require exactly one `Finished` event.
- Return `ModelStep::{Final, UseClientTools, Continue}` according to the
  `Finished.kind`.
- Return an error if the stream ends without `Finished`, if tool-call arguments
  never parse as JSON, or if a provider emits inconsistent duplicate ids.

Keep collection strict enough that provider bugs are visible in tests. Providers
that cannot stream should emit one complete delta chunk for each content item
instead of a second collected-item representation.

## Agent Run Events

Change `Agent::run` to return a stream:

```rust
pub fn run(
    &self,
    transcript: Transcript,
) -> impl Stream<Item = Result<AgentRunEvent, AgentRunError<B::Error>>> + Send + '_;
```

Recommended event shape:

```rust
#[derive(Debug, Clone)]
pub enum AgentRunEvent {
    RunStarted,
    ModelStepStarted {
        ordinal: u32,
    },
    ModelEvent {
        ordinal: u32,
        event: ModelStepEvent,
    },
    ClientToolStarted {
        call: ClientToolCall,
    },
    ClientToolFinished {
        result: ClientToolResult,
        trace: ClientToolTrace,
        media: Vec<BoxedMediaRef>,
    },
    RunFinished {
        run: AgentRun,
    },
}
```

Notes:

- The LLM provider emits `ClientToolCall` intent only.
- The agent runtime emits `ClientToolStarted` and `ClientToolFinished`.
- Provider-owned hosted/server tools remain model events through
  `ModelStepItem::ServerToolUse`.
- Provider grounding/citations remain model events through
  `ModelStepItem::Grounding`.
- `RunFinished` carries the collected stable `AgentRun` used by Discord,
  memory, title generation, TOS preflight, and storage-facing paths.

### Agent Run Collection

Add a reducer:

```rust
pub async fn collect_agent_run<S, E>(events: S) -> Result<AgentRun, E>
where
    S: Stream<Item = Result<AgentRunEvent, E>> + Send,
{
    // ...
}
```

This reducer should return the `AgentRun` from the `RunFinished` event and
surface stream errors. It should validate that the stream does not finish before
`RunFinished`.

It is acceptable to add an inherent convenience method on `Agent` if useful for
tests or internal call sites, but keep `run` itself streaming.

## Transcript Semantics

Keep `Transcript`, `TranscriptTurn`, and `ContentBlock` conceptually unchanged.

Model-step collection produces ordered `ModelStepItem` values. The agent runtime
appends only `ModelStepItem::OutputBlock` values that convert to transcript
blocks to the model-facing transcript:

```rust
TranscriptTurn {
    role: TurnRole::Assistant,
    blocks: vec![
        ContentBlock::Text { text },
        ContentBlock::ClientToolCall(call),
        ContentBlock::Continuation(continuation),
    ],
    metadata: serde_json::Value::Null,
}
```

Client tool results are appended by the agent runtime as user transcript blocks
after `ClientToolFinished`:

```rust
TranscriptTurn {
    role: TurnRole::User,
    blocks: vec![
        ContentBlock::ClientToolResult(result),
        ContentBlock::Media { media },
    ],
    metadata: serde_json::Value::Null,
}
```

Do not add reasoning summaries or grounding as `ContentBlock` variants. They are
trace/viewer metadata, not normal model-facing conversation text.

## Storage Semantics

First pass:

- Do not persist individual token deltas.
- Continue saving the collected turn input transcript with
  `save_turn_input(...)`.
- Continue persisting collected model-step traces and tool traces after
  collection.
- Preserve current Discord behavior by reducing `Agent::run` to `AgentRun` in
  `chudbot-bot`.

Reasoning persistence needs one explicit decision during implementation:

1. Minimal path: keep deriving reasoning from `ProviderContinuation` where
   current providers already include it, and add typed reasoning only to live
   stream events.
2. Better path: add a typed `reasoning: Vec<ReasoningItem>` or
   `items: Vec<ModelStepItem>` field to `ModelStepTrace` and persist it in
   `chudbot-storage-sqlx`.

Recommendation: implement the better path if the schema change is not too large.
The point of this refactor is to surface reasoning summaries as first-class
agent framework output. Keeping typed reasoning only in live events would make
completed stored turns less useful.

Avoid persisting `ModelStepEvent` deltas unless a later feature explicitly needs
replayable token-by-token animation.

## Provider Boundary

Provider crates should emit `ModelStepEvent` streams.

Initial compatibility path for each provider:

1. Move full-response parsing into provider-local state or output helpers.
2. Emit content as `ModelStepDelta` events, using one full chunk for
   non-streaming responses.
3. Emit continuation/server-tool/grounding/usage/finished as their explicit
   `ModelStepEvent` variants.
4. Keep tests passing before adding true streaming network calls.

Then add true streaming where supported:

- `chudbot-openai`
  - Stream Responses API events.
  - Emit text deltas, reasoning-summary deltas, tool-call deltas, hosted-tool
    events, grounding, usage, continuation, and final step kind.
- `chudbot-anthropic`
  - Stream Messages API events.
  - Emit text/thinking deltas, tool-use items, server-tool events, grounding,
    usage, continuation, and final step kind.
- `chudbot-openai-compat`
  - Stream Chat Completions chunks where the backend supports `stream = true`.
  - Map `delta.content` to text deltas, `delta.reasoning_content` to reasoning
    deltas, and `delta.tool_calls` to client-tool call deltas.
- `chudbot-xai`
  - Stream Responses API events.
  - Emit text deltas, reasoning-summary deltas, tool-call deltas, hosted-tool
    events, grounding, usage, continuation, and final step kind.
  - Add true streaming only after confirming the current xAI endpoint and retry
    helpers support the required event stream cleanly.
- `chudbot-gemini`
  - Stream `streamGenerateContent` SSE chunks.
  - Emit text/tool-call deltas, server-tool events, grounding, usage,
    continuation, and final step kind.

Provider adapters must not execute Chudbot client tools. They only emit model
intent via `ContentBlock::ClientToolCall`.

## Registry Boundary

Update `LlmProviderRegistry::step` in `crates/chudbot-api/src/registries.rs` to
return a stream of `ModelStepEvent`.

This boundary routes across multiple concrete provider stream types. It is
reasonable to use `futures::stream::BoxStream` at registry/routing boundaries
even if individual provider crates return concrete `impl Stream` values.

Expected touch points:

- `crates/chudbot-api/src/registries.rs`
- `crates/chudbot-bot/src/registries.rs`
- `crates/chudbot-bin/src/services.rs`

Keep provider lookup errors as stream errors. A missing provider can return
`stream::once(async { Err(ConfiguredLlmError::Missing(...)) })`.

## Agent Loop Implementation Notes

The existing `Agent::run` body has real control flow: loop limits, provider
steps, tool dispatch, transcript mutation, trace accumulation, and terminal
outcomes. Returning a stream from it needs an implementation strategy.

Recommended first pass:

- Keep the loop logic in one place in `crates/chudbot-api/src/agent.rs`.
- Use an implementation that can yield events from async control flow without
  spawning detached work.
- If adding a tiny dependency is acceptable, `async-stream` is the simplest way
  to express this with readable code.
- If avoiding a new dependency is preferred, implement a small custom stream
  state machine or use `futures::stream::try_unfold`. Avoid an unbounded mpsc
  task unless cancellation and backpressure are handled deliberately.

The stream should preserve backpressure: if the UI or collector stops polling,
provider reads and tool execution should stop as naturally as possible.

## Chudbot Call Site Changes

Update every existing `agent.run(...).await` call to collect the stream.

Important call sites:

- `crates/chudbot-bot/src/turns.rs`
  - Main Discord turn execution.
  - Keep cancellation around collection:

    ```rust
    let events = agent.run(transcript);
    let run = tokio::select! {
        biased;
        () = cancel_token.cancelled() => None,
        run = collect_agent_run(events) => Some(run),
    };
    ```

- `crates/chudbot-bot/src/tools/executor.rs`
  - Subagent tool execution.

- `crates/chudbot-bot/src/agents/tos.rs`
  - TOS preflight agent.

- `crates/chudbot-bot/src/agents/title.rs`
  - Conversation title generation.

- `crates/chudbot-bot/src/memory/runtime.rs`
  - Memory diary/compact agent paths.

- Tests in `crates/chudbot-api` and `crates/chudbot-bot`.

The web viewer can continue using coarse SSE invalidation events initially.
Token streaming to the frontend should be a follow-up that consumes
`AgentRunEvent` directly rather than refetching snapshots for each token.

## Suggested Phases

### Phase 1: Shared Types And Reducers

- Add `ModelStepEvent`, `ModelStepDelta`, `ModelStepOutput`, and
  `ModelStepItem`.
- Refactor `ModelStep` to use `ModelStepOutput`.
- Add helper methods for transcript blocks, tool calls, reasoning, grounding,
  continuation, and answer text.
- Add `collect_model_step`.
- Add `AgentRunEvent`.
- Add `collect_agent_run`.
- Update unit tests for collectors.

### Phase 2: Compatibility Streams

- Change `LlmBackend::step` to return a stream.
- Change `LlmProviderRegistry::step` and `RoutedLlmBackend::step` to return
  streams.
- Update providers that do not have true streaming yet to emit one full delta
  chunk per content item plus explicit metadata events.
- Update `Agent::run` to consume provider streams and emit `AgentRunEvent`.
- Update all call sites to collect.
- Confirm Discord behavior is unchanged.

### Phase 3: Ordered Step Output Cleanup

- Replace uses of bucketed `AssistantStep` fields with `ModelStepOutput` helper
  methods.
- Update model-step trace construction.
- Decide and implement typed reasoning persistence.
- Keep replay logic grounded in transcript blocks and continuations.

### Phase 4: True Provider Streaming

- Implement true streaming for `chudbot-openai`.
- Implement true streaming for `chudbot-anthropic`.
- Implement true streaming for `chudbot-openai-compat`.
- Keep single-shot/full-response fallbacks as one complete delta chunk only for
  providers or code paths that cannot stream.

### Phase 5: UI Streaming

- Add a live in-process event path from top-level `AgentRunEvent` to web SSE.
- Render text deltas, reasoning deltas, and client tool lifecycle in the trace
  viewer.
- Reconcile live model events with the stored snapshot carried by `RunFinished`.
- Keep Discord on collected `AgentRun`.

## Test Plan

Focused checks during implementation:

```sh
cargo test -p chudbot-api
cargo test -p chudbot-bot
cargo test -p chudbot-openai
cargo test -p chudbot-anthropic
cargo test -p chudbot-openai-compat
cargo check -p chudbot-bin
cargo run -p chudbot-bin -- check-config
git diff --check
```

Useful unit tests to add:

- Collect text deltas into one ordered `ContentBlock::Text`.
- Collect multiple text items without reordering them.
- Collect reasoning-summary deltas into `ModelStepItem::Reasoning`.
- Collect client-tool-call argument deltas into a parsed `ClientToolCall`.
- Reject malformed final tool-call JSON.
- Preserve interleaving of text, reasoning, server-tool, grounding, and
  continuation items.
- Convert single-shot provider output into events and back without losing
  control-flow kind, transcript blocks, usage, or continuation.
- Collect an `AgentRunEvent` stream into the same `AgentRun` shape the current
  Discord code expects.

## Open Questions

- Should typed reasoning be stored as a new `ModelStepTrace.reasoning` JSON field
  or as a more general ordered `ModelStepTrace.items` field?
- Should `ModelStepEvent::Finished` carry the provider-reported model id, or
  should model id be emitted as a separate event before finish?
- Should generated local item ids be exposed to UI clients, or only provider
  item ids when available?
- Should the first true streaming provider be OpenAI Responses or
  OpenAI-compatible Chat Completions against the local LM Studio path?

## Recommended First Implementation Target

Start with `chudbot-api` and `chudbot-bot` only:

1. Add the new event/output types and reducers.
2. Refactor the agent loop to stream while still collecting at current call
   sites.
3. Emit one complete delta chunk for any provider path that still receives a
   single completed response.
4. Prove the collected Discord path remains behaviorally unchanged.

Only after that should provider-specific true streaming be added. This keeps the
core API refactor separate from the complexity of each provider's event stream
format.
