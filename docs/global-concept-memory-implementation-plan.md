# Global Concept Memory Implementation Plan

Updated: 2026-06-05.

This document is the implementation plan for Chudbot global concept memory:
memory that is not scoped to a single user, but can be recalled by semantic
relationship to an incoming turn. Example: if a user says "Walt Disney hated
Jews", a later question about "Tigger" or "the CEO of Disney" may retrieve the
stored claim as relevant context.

This is a plan only. Do not treat it as already implemented.

## Goals

- Extract global concept claims from completed transcript windows in the
  background.
- Give the extractor agent exactly one global-memory write tool:
  `store_global_fact(fact: String)`.
- Compute embeddings inside Chudbot provider/runtime code, never by asking the
  LLM to produce vectors.
- Store raw claims, embeddings, provenance, extraction metadata, and usage in
  Postgres.
- Retrieve global concept memory automatically before the main agent run.
- Inject retrieved context as `memory:global:*` turn context items.
- Keep replay behavior stable: `memory:` context items should remain
  non-replayable and should be recomputed for retries.
- Prevent cross-guild leakage unless a deployment-wide sharing policy is
  explicitly enabled.

## Non-Goals

- Do not replace existing user memory.
- Do not auto-inject compact user memory profiles.
- Do not add public web APIs for browsing or listing memory.
- Do not make the main agent decide when to recall global memory through a
  recall tool.
- Do not treat extracted claims as authoritative facts.

## Recommended Architecture

Add global concept memory as a sibling to the existing user-memory pipeline.
Keep user memory and global concept memory operationally separate:

- User memory remains tool-driven through `lookup_user_memory`,
  `remember_user_memory`, and `forget_user_memory`.
- Global concept memory is extracted in the background and recalled
  automatically during turn preparation.

The main components should be:

- `chudbot-api`
  - Add provider-neutral embedding contracts.
  - Add global-memory storage structs and `BotStorage` methods.
- `chudbot-openai`
  - Implement OpenAI embeddings through `reqwest`.
- `chudbot-bin`
  - Add named `[embedding]` provider config and an embedding registry.
- `chudbot-bot`
  - Add `global_memory` module.
  - Add `GlobalMemoryConfig`.
  - Add `GlobalMemoryRuntime`.
  - Add extractor write tool.
  - Add pre-agent recall and context injection.
- `chudbot-storage-sqlx`
  - Add pgvector migration and storage methods.

This follows the current split: provider crates own HTTP calls, `chudbot-bin`
builds named provider registries, `chudbot-bot` orchestrates, and
`chudbot-api` stays free of SQLx, Reqwest, Axum, Twilight, and concrete
provider config.

## Migration Plan

The production DGX Spark has the PGDG `postgresql-18-pgvector` package
installed. The migration should still create the database extension because
extension activation is per database.

Create a new migration after the current latest migration:

```sql
-- Global concept memory with pgvector-backed semantic recall.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE global_memory_jobs (
    id                  UUID PRIMARY KEY,
    kind                TEXT NOT NULL CHECK (kind IN ('extract')),
    message_provider    TEXT NOT NULL,
    scope_key           TEXT NOT NULL,
    channel_key         TEXT NOT NULL,
    window_start        TIMESTAMPTZ NOT NULL,
    window_end          TIMESTAMPTZ NOT NULL,
    source_turn_ids     UUID[] NOT NULL DEFAULT '{}',
    status              TEXT NOT NULL
        CHECK (status IN ('pending', 'running', 'completed', 'failed')),
    attempts            INTEGER NOT NULL DEFAULT 0,
    next_run_at         TIMESTAMPTZ NOT NULL,
    leased_by           TEXT,
    leased_until        TIMESTAMPTZ,
    dedupe_key          TEXT NOT NULL,
    started_at          TIMESTAMPTZ,
    completed_at        TIMESTAMPTZ,
    error               TEXT,
    agent_name          TEXT,
    llm_provider        TEXT,
    llm_model           TEXT,
    usage               JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE global_memory_facts (
    id                         UUID PRIMARY KEY,
    message_provider           TEXT NOT NULL,
    scope_key                  TEXT NOT NULL,
    visibility                 TEXT NOT NULL DEFAULT 'scope'
        CHECK (visibility IN ('scope', 'deployment')),
    fact                       TEXT NOT NULL,
    fact_hash                  TEXT NOT NULL,
    dedupe_key                 TEXT NOT NULL,
    embedding                  vector(1536) NOT NULL,
    embedding_provider         TEXT NOT NULL,
    embedding_model            TEXT NOT NULL,
    embedding_dimensions       INTEGER NOT NULL CHECK (embedding_dimensions = 1536),
    claim_status               TEXT NOT NULL DEFAULT 'unverified_user_assertion'
        CHECK (claim_status IN (
            'unverified_user_assertion',
            'observed_conversation_context',
            'operator_verified',
            'corrected',
            'retracted'
        )),
    confidence                 REAL CHECK (confidence IS NULL OR confidence BETWEEN 0 AND 1),
    source_count               INTEGER NOT NULL DEFAULT 1,
    first_seen_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_seen_at               TIMESTAMPTZ NOT NULL DEFAULT now(),
    first_source_conversation_id UUID REFERENCES conversations(id) ON DELETE SET NULL,
    first_source_turn_id       UUID REFERENCES turns(id) ON DELETE SET NULL,
    first_job_id               UUID REFERENCES global_memory_jobs(id) ON DELETE SET NULL,
    superseded_by_fact_id      UUID REFERENCES global_memory_facts(id) ON DELETE SET NULL,
    deleted_at                 TIMESTAMPTZ,
    delete_reason              TEXT,
    created_at                 TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at                 TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE global_memory_fact_sources (
    id                         BIGSERIAL PRIMARY KEY,
    fact_id                    UUID NOT NULL REFERENCES global_memory_facts(id) ON DELETE CASCADE,
    extraction_job_id          UUID REFERENCES global_memory_jobs(id) ON DELETE SET NULL,
    source_conversation_id     UUID REFERENCES conversations(id) ON DELETE SET NULL,
    source_turn_ids            UUID[] NOT NULL DEFAULT '{}',
    source_metadata            JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at                 TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX global_memory_jobs_active_dedupe_idx
    ON global_memory_jobs (dedupe_key)
    WHERE status IN ('pending', 'running');

CREATE INDEX global_memory_jobs_due_idx
    ON global_memory_jobs (status, next_run_at, leased_until);

CREATE INDEX global_memory_jobs_scope_window_idx
    ON global_memory_jobs (message_provider, scope_key, channel_key, window_end);

CREATE UNIQUE INDEX global_memory_facts_active_dedupe_idx
    ON global_memory_facts (dedupe_key)
    WHERE deleted_at IS NULL;

CREATE INDEX global_memory_facts_scope_model_idx
    ON global_memory_facts (message_provider, scope_key, embedding_model, created_at)
    WHERE deleted_at IS NULL;

CREATE INDEX global_memory_facts_last_seen_idx
    ON global_memory_facts (message_provider, scope_key, last_seen_at DESC)
    WHERE deleted_at IS NULL;

CREATE INDEX global_memory_facts_embedding_hnsw_idx
    ON global_memory_facts
    USING hnsw (embedding vector_cosine_ops)
    WHERE deleted_at IS NULL;

CREATE INDEX global_memory_fact_sources_fact_idx
    ON global_memory_fact_sources (fact_id, created_at);

CREATE TRIGGER global_memory_jobs_touch_updated_at
    BEFORE UPDATE ON global_memory_jobs
    FOR EACH ROW EXECUTE FUNCTION chudbot_touch_updated_at();

CREATE TRIGGER global_memory_facts_touch_updated_at
    BEFORE UPDATE ON global_memory_facts
    FOR EACH ROW EXECUTE FUNCTION chudbot_touch_updated_at();
```

The default embedding dimension should be 1536 so the schema can use
`vector(1536)` and HNSW without half-precision workarounds. That matches
`text-embedding-3-small`. If a future implementation needs 3072-dimensional
embeddings, either make a second migration or use a reduced `dimensions` value
from the embedding provider.

## SQL Query Shape

Keep hot-path search simple. First query candidates with pgvector, then apply
thresholding, dedupe, and scoring in Rust.

```sql
SELECT id, fact, claim_status, confidence, source_count, last_seen_at,
       embedding <=> $4 AS distance
  FROM global_memory_facts
 WHERE message_provider = $1
   AND scope_key = $2
   AND deleted_at IS NULL
   AND embedding_model = $3
 ORDER BY embedding <=> $4
 LIMIT $5;
```

If deployment-wide sharing is enabled, run a second query for
`visibility = 'deployment'`, or use a simple `OR` guarded by config. Keep the
default scoped query separate for clarity and to avoid accidental leakage.

## Rust pgvector Integration

Prefer the `pgvector` crate if it resolves cleanly with SQLx 0.9:

```toml
pgvector = { version = "0.4", features = ["sqlx"] }
```

In `chudbot-storage-sqlx`, use `pgvector::Vector` for binds and row decoding.
If dependency resolution conflicts with SQLx 0.9, avoid changing SQLx versions.
Use a validated vector literal fallback and bind it as `$N::vector`:

```rust
fn vector_literal(values: &[f32]) -> String {
    let mut out = String::from("[");
    for (index, value) in values.iter().enumerate() {
        if index > 0 {
            out.push(',');
        }
        out.push_str(&value.to_string());
    }
    out.push(']');
    out
}
```

Validate that every embedding is finite and has the configured dimension before
building the literal.

## API Additions

Add a new `embedding` module in `chudbot-api`:

```rust
pub trait EmbeddingGenerator: Send + Sync {
    type Error: std::error::Error + Send + Sync + 'static;

    fn backend_name(&self) -> &ProviderName;

    fn embed(
        &self,
        request: EmbeddingRequest,
    ) -> impl Future<Output = Result<EmbeddingResponse, Self::Error>> + Send;
}

pub struct EmbeddingRequest {
    pub input: Vec<String>,
    pub model: ModelId,
    pub dimensions: Option<u32>,
}

pub struct EmbeddingResponse {
    pub model: ModelId,
    pub embeddings: Vec<EmbeddingVector>,
    pub usage: Vec<UsageRecord>,
}

pub struct EmbeddingVector {
    pub values: Vec<f32>,
}
```

Add `UsageSubject::Embedding`.

Add storage structs near the existing user-memory storage contracts:

```rust
pub struct GlobalMemoryScope {
    pub platform: PlatformName,
    pub scope_key: String,
    pub channel_key: String,
}

pub struct GlobalMemoryJobSchedule {
    pub now: OffsetDateTime,
    pub extract_cutoff: OffsetDateTime,
    pub extract_due_before: OffsetDateTime,
    pub extract_window_seconds: u64,
}

pub struct GlobalMemoryJob {
    pub id: Uuid,
    pub scope: GlobalMemoryScope,
    pub window_start: OffsetDateTime,
    pub window_end: OffsetDateTime,
    pub source_turn_ids: Vec<TurnId>,
    pub attempts: i32,
    pub dedupe_key: String,
}

pub struct GlobalMemoryTurn {
    pub conversation_id: ConversationId,
    pub turn_id: TurnId,
    pub completed_at: OffsetDateTime,
    pub user_display_name: String,
    pub user_content: String,
    pub assistant_content: Option<String>,
    pub audio_transcriptions: Vec<UserMemoryAudioTranscription>,
}

pub struct NewGlobalMemoryFact {
    pub scope: GlobalMemoryScope,
    pub fact: String,
    pub embedding: Vec<f32>,
    pub embedding_provider: ProviderName,
    pub embedding_model: ModelId,
    pub embedding_dimensions: u32,
    pub claim_status: GlobalMemoryClaimStatus,
    pub confidence: Option<f32>,
    pub source_conversation_id: Option<ConversationId>,
    pub source_turn_id: Option<TurnId>,
    pub extraction_job_id: Option<Uuid>,
    pub source_turn_ids: Vec<TurnId>,
    pub source_metadata: serde_json::Value,
}

pub struct GlobalMemorySearch {
    pub platform: PlatformName,
    pub scope_key: String,
    pub embedding: Vec<f32>,
    pub embedding_model: ModelId,
    pub embedding_dimensions: u32,
    pub candidate_limit: u32,
    pub top_k: u32,
    pub include_deployment: bool,
}

pub struct GlobalMemorySearchResult {
    pub id: Uuid,
    pub fact: String,
    pub claim_status: GlobalMemoryClaimStatus,
    pub confidence: Option<f32>,
    pub source_count: i32,
    pub last_seen_at: OffsetDateTime,
    pub distance: f32,
    pub similarity: f32,
}
```

Add `BotStorage` methods:

```rust
fn enqueue_due_global_memory_jobs(...);
fn claim_global_memory_jobs(...);
fn finish_global_memory_job(...);
fn load_global_memory_turn_window(...);
fn upsert_global_memory_fact(...);
fn search_global_memory(...);
fn tombstone_global_memory_fact(...);
fn correct_global_memory_fact(...);
```

Use normal names in the real code; the list above is the required behavior
rather than exact final signatures.

## Config Additions

Add top-level global-memory config:

```toml
[global_memory]
enabled = true
extractor_provider = "grok"
extractor_model = "grok-4.3"
embedding_provider = "openai_embeddings"
embedding_model = "text-embedding-3-small"
embedding_dimensions = 1536
poll_interval_seconds = 60
extract_backfill_window = "3d"
extract_interval = "6h"
lease_seconds = 300
max_jobs_per_tick = 4
max_concurrent_jobs = 2
max_turns_per_extract_job = 80
max_extract_output_tokens = 1024
retry_backoff_seconds = 300
max_job_attempts = 5
recall_top_k = 5
recall_candidate_limit = 24
recall_min_similarity = 0.78
recall_max_context_chars = 2400
share_across_guilds = false

[embedding.openai_embeddings]
kind = "openai"
api_key = "sk-..."
# base_url = "https://api.openai.com/v1"
```

Add a per-agent opt-in:

```toml
[bot.agents.default]
global_memory = true
```

Keep `memory = true` as the user-memory tool switch. Do not overload it with
global concept recall.

## Embedding Provider Implementation

In `chudbot-openai`, implement embeddings on `OpenAiClient` with `reqwest`:

- Endpoint: `POST /embeddings`.
- Request body:
  - `model`
  - `input`
  - `encoding_format = "float"`
  - `dimensions` when configured.
- Response:
  - Collect `data[].embedding` into `Vec<f32>`.
  - Preserve provider `usage` as `UsageRecord` with `UsageSubject::Embedding`.
  - Use the provider-reported model when available; otherwise use requested
    model.

Add a named embedding registry in `chudbot-bin`, matching the existing LLM,
image, video, and audio registry pattern.

## Extraction Runtime

Add `crates/chudbot-bot/src/global_memory.rs`.

The runtime should mirror the durable shape of `MemoryRuntime`:

- SQL-backed job enqueueing.
- Expiring leases.
- `FOR UPDATE SKIP LOCKED` claim behavior.
- Per-scope/channel active-key deconfliction.
- Retry and failed status handling.
- Shutdown through `CancellationToken`.

Extraction windows should be keyed by:

- `message_provider`
- `scope_key`
- `channel_key`
- time window

Do not key extractor windows by user. This memory is global concept memory,
not per-user diary memory.

The extractor transcript should include bounded completed turns in the window:

- User display name.
- User content.
- Assistant content.
- Audio transcriptions already produced by successful `transcribe_audio` tool
  traces or preflight audio handling, when available.
- Source ids in metadata.

Extractor prompt requirements:

- Extract only durable concepts, claims, lore, corrections, reusable context,
  relationships between named entities, and recurring server context.
- Store short atomic facts or claims.
- Do not store sensitive personal information.
- Do not store throwaway reactions.
- Keep hostile or discriminatory claims as claims with provenance, not truth.
- Use only `store_global_fact`.
- If nothing should be stored, make no tool calls and produce a short final
  answer such as `No global facts to store.`

## Store Tool

`store_global_fact(fact: String)` is available only to the extractor agent.

Tool behavior:

1. Trim and validate `fact`.
2. Normalize a dedupe key:
   - Lowercase.
   - Collapse whitespace.
   - Hash the normalized fact plus scope and embedding model.
3. Call the configured embedding provider.
4. Validate dimension and finite values.
5. Upsert `global_memory_facts`.
6. Insert `global_memory_fact_sources`.
7. Return a compact JSON result for traceability.

The LLM never sees or writes the embedding vector.

## Recall and Context Injection

Add recall after platform context is prepared and before
`save_turn_input`.

Recommended flow:

1. Build a recall query from:
   - Current user text.
   - Relevant preflight audio transcription text.
   - A short excerpt from quoted/current platform message context.
2. Skip recall if the query is empty or too short.
3. Embed the query.
4. Search global memory by current `message_provider` and `scope_key`.
5. Apply threshold and top-k ranking in Rust.
6. Dedupe near-identical facts.
7. Inject one or a few `ContextItem`s before the current message context:

```text
source: memory:global:<fact-id>
role: user
content:
Stored global memory claim/context, not an authoritative fact:
- claim: ...
- status: unverified_user_assertion
- confidence: ...
- similarity: ...
- source_count: ...
```

Keep injected context compact and bounded by `recall_max_context_chars`.

## Prompt Guidance

Update composed system prompt only when the selected agent has
`global_memory = true` and global memory is enabled:

```text
- Relevant global memory claims may be included in this turn as
  `memory:global:*` context. Treat them as stored claims/context, not
  authoritative facts. If current user evidence conflicts with stored memory,
  trust the current turn and note the conflict naturally when relevant.
```

Do not ask the main agent to call a recall tool, because recall is automatic.

## Forget and Correction

First pass should support operator/storage-level tombstone and correction
methods even if no public command is exposed yet:

- Tombstone:
  - Set `deleted_at`.
  - Set `delete_reason`.
  - Exclude from recall.
- Correction:
  - Tombstone or supersede the old fact.
  - Store a new corrected fact with fresh embedding.
  - Set `superseded_by_fact_id`.
  - Preserve old provenance for audit.

Do not physically delete facts in normal operation.

## Replay and Retry

Current replay drops context items whose source starts with `memory:`. Global
memory should follow that rule by using `memory:global:*` sources.

For retries:

- Do not replay the previously injected global memory.
- Recompute recall from the retry turn's saved user content and current memory
  state.
- This lets forget/correction take effect between the original attempt and the
  retry.

## Privacy and Scope

Default scope:

- Discord guild messages: `scope_key = guild:<guild_id>`.
- DMs or no guild: `scope_key = global`.

Do not share facts across guilds unless `[global_memory].share_across_guilds`
is true. Even then, only recall rows marked `visibility = 'deployment'`.

Never add route listing or memory browsing to the unauthenticated viewer.

## Operational Notes

- Production must have `postgresql-18-pgvector` installed on the server before
  running the migration.
- The migration's `CREATE EXTENSION IF NOT EXISTS vector;` activates pgvector
  in the target database.
- HNSW index builds can use meaningful memory on large tables. For first
  deploy, the table will be empty, so index creation is cheap.
- Keep `embedding_dimensions = 1536` until there is a clear reason to migrate.
- Log extraction job ids, scope keys, source turn ids, embedding provider/model,
  and usage counts.

## Test Plan

Focused commands:

```sh
cargo test -p chudbot-api storage::tests
cargo test -p chudbot-openai embeddings
cargo test -p chudbot-bot global_memory::tests
cargo test -p chudbot-storage-sqlx global_memory
cargo run -p chudbot-bin -- check-config --config config.example.toml
cargo check --all-targets --all-features
cargo test --all-features
git diff --check
```

Behavior tests to add:

- Config validation fails when global memory references missing embedding or
  extractor providers.
- Embedding response parsing rejects missing embeddings, wrong dimensions, NaN,
  and infinities.
- Store tool writes fact rows and source rows with dedupe.
- Store tool upserts repeated facts by incrementing source metadata instead of
  creating duplicate active facts.
- Search excludes tombstoned facts.
- Search filters by guild scope by default.
- Recall injects `memory:global:*` context before model execution.
- Replay drops global memory context.
- Retry recomputes global memory context rather than replaying old injected
  rows.
- Extractor windows are channel/scope/time keyed, not user keyed.

## Risks and Tradeoffs

- False memories: mitigated by storing provenance, confidence, claim status,
  and prompt wording that treats rows as stored claims.
- Privacy leakage: mitigated by guild-scoped defaults and explicit
  deployment-wide visibility.
- Latency: one embedding call per eligible incoming turn. Start with small
  top-k and simple query text before adding caches.
- Cost: extraction and recall both use provider calls. Record usage and make
  runtime switches easy to disable.
- Operational dependency: migrations now require pgvector extension files on
  the Postgres host.
- Dimension changes: fixed `vector(1536)` is simple and index-friendly, but
  changing dimensions later requires a migration.
- Query quality: semantic recall can retrieve related but unhelpful facts.
  Use thresholding, top-k, and compact prompt wording before adding complex
  rerankers.

## Implementation Order

1. Add config types and validation for `[global_memory]` and `[embedding]`.
2. Add `EmbeddingGenerator` contracts and `UsageSubject::Embedding`.
3. Add OpenAI embedding implementation and embedding registry.
4. Add pgvector migration with `CREATE EXTENSION IF NOT EXISTS vector;`.
5. Add `BotStorage` global-memory structs and SQLx methods.
6. Add `global_memory` module with store tool and extraction runtime.
7. Spawn `GlobalMemoryRuntime` beside the existing user-memory runtime.
8. Add pre-agent recall and `memory:global:*` context injection.
9. Add retry/replay tests.
10. Update `config.example.toml` and `docs/2.0-api-shapes.md` after the code
    lands.
