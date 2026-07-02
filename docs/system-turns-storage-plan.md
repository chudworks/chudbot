# System Turns Storage Status

Status: implemented (written 2026-07-02, updated 2026-07-02).

Chudbot now treats agent instructions as transcript state. The stored
per-attempt input transcript is the source of truth for instructions that were
visible to a model call, including mid-conversation instruction updates.

## Current Shape

- `TurnRole::System` is part of the provider-neutral transcript model.
- `SaveTurnInput.transcript` is required and contains the exact stored model
  input for the attempt.
- `TurnSnapshot.agent_instructions` is an explicit enum:
  `LegacyText { text }` for old monolithic history, or `Parts { parts }` for
  modern labeled instruction parts.
- Modern top-level conversation instructions are split into labeled parts and
  persisted as marked system-role transcript turns. Part metadata includes
  `agent_instructions`, `agent_instruction_part`,
  `agent_instruction_part_key`, and `agent_instruction_part_ordinal`.
- Storage reconstructs the effective instruction state by replaying marked
  transcript rows in conversation order. A legacy marker replaces the whole
  instruction state; a labeled marker updates only that part key.
- Runtime compares the currently rendered instruction parts with the latest
  stored state and inserts only changed parts before the current user turn.
- `Agent::run` preserves persisted instruction markers when they are already in
  the transcript. Standalone agents and subagents with no persisted markers get
  instructions inserted from their `AgentSpec`.
- Memory notes remain separate system-role context turns. They are not labeled
  as agent instructions and are not part of the instruction-part reducer.

## Migration History

- `0027_system_turn_roles.sql` allowed `system` input-message roles.
- `0028_system_instructions_into_transcript.sql` backfilled legacy instruction
  text into marked transcript rows, preserved relevant timestamps, shifted input
  message ordinals to make room for instruction rows, and dropped the old
  monolithic instruction columns.
- `0029_deduplicate_agent_instruction_texts.sql` removes repeated identical
  legacy instruction rows from migrated history and resequences input-message
  ordinals.

## Current Config/API Names

- Config uses `instructions` for an agent's authored instruction text.
- Config uses `extra_agent_instructions` for deployment-wide operator policy.
- Old TOML keys are still accepted as serde aliases for compatibility.
- The viewer API exposes structured `agent_instructions`, not a monolithic
  instruction string.

## Verified

- `cargo fmt`
- `cargo check --workspace`
- `cargo test -p chudbot-api -p chudbot-bot -p chudbot-bin`
- `cd frontend && bun run typecheck`
- `git diff --check`
