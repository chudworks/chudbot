//! Discord bot event loop. Wires serenity + poise to the `core` crate's
//! Grok client and conversation store.

/// Entry point for the `grok bot` subcommand. Placeholder until the
/// serenity gateway and command handlers are implemented.
pub async fn run() -> Result<(), Box<dyn std::error::Error>> {
    tracing::warn!("bot subcommand: not yet implemented");
    Ok(())
}
