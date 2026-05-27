//! Slash command registration and dispatch.
//!
//! Two top-level commands are registered globally on startup:
//!   - `/grok-privacy {in|out|status}` — per-user, anyone can run.
//!   - `/grok-mode {show|set}` — guild admins only; sets the
//!     guild-wide context-gathering policy (one of four "designs").
//!
//! Commands invoked from DMs (no `guild_id`) return a polite error
//! since both privacy preferences and mode settings are per-guild.

use std::sync::Arc;

use grok_discord_bot_core::{Db, PrivacyMode};
use twilight_http::Client as HttpClient;
use twilight_model::application::command::{Command, CommandOptionType};
use twilight_model::application::interaction::application_command::{
    CommandData, CommandDataOption, CommandOptionValue,
};
use twilight_model::application::interaction::{Interaction, InteractionData};
use twilight_model::channel::message::MessageFlags;
use twilight_model::guild::Permissions;
use twilight_model::http::interaction::{InteractionResponse, InteractionResponseType};
use twilight_model::id::Id;
use twilight_model::id::marker::ApplicationMarker;
use twilight_util::builder::InteractionResponseDataBuilder;
use twilight_util::builder::command::{
    ChannelBuilder, CommandBuilder, IntegerBuilder, StringBuilder, SubCommandBuilder,
};

const HISTORY_SIZE_MIN: i64 = 1;
const HISTORY_SIZE_MAX: i64 = 100;

/// Build the slash command definitions. Called once on startup and
/// pushed to Discord via `set_global_commands`.
pub fn definitions() -> Vec<Command> {
    let privacy = CommandBuilder::new(
        "grok-privacy",
        "Manage your personal Grok privacy preference in this server",
        twilight_model::application::command::CommandType::ChatInput,
    )
    .option(
        SubCommandBuilder::new(
            "in",
            "Allow Grok to use your messages as quoted-message context",
        )
        .build(),
    )
    .option(
        SubCommandBuilder::new(
            "out",
            "Stop letting Grok use your messages as quoted-message context (default)",
        )
        .build(),
    )
    .option(
        SubCommandBuilder::new("status", "Show your current privacy preference here").build(),
    )
    .build();

    let mode = CommandBuilder::new(
        "grok-mode",
        "Configure how the bot gathers context in this server",
        twilight_model::application::command::CommandType::ChatInput,
    )
    .default_member_permissions(Permissions::ADMINISTRATOR)
    .option(
        SubCommandBuilder::new("show", "Show the active privacy mode for this server").build(),
    )
    .option(
        SubCommandBuilder::new("set", "Change the privacy mode for this server")
            .option(
                StringBuilder::new("mode", "Which of the four designs to use")
                    .required(true)
                    .choices([
                        ("Open (Design 1) — see everything", "open"),
                        ("Channel only (Design 2)", "channel_only"),
                        ("Opt-in (Design 3) — default", "opt_in"),
                        ("Conversation only (Design 4)", "conversation_only"),
                    ]),
            )
            .option(ChannelBuilder::new(
                "channel",
                "Channel to confine the bot to (for channel_only)",
            ))
            .option(
                IntegerBuilder::new(
                    "history_size",
                    "How many recent channel messages to include (for open / channel_only)",
                )
                .min_value(HISTORY_SIZE_MIN)
                .max_value(HISTORY_SIZE_MAX),
            )
            .build(),
    )
    .build();

    vec![privacy, mode]
}

/// Push the command set to Discord globally. Idempotent — Discord
/// replaces the entire registered set with what we send.
pub async fn register(
    http: &HttpClient,
    app_id: Id<ApplicationMarker>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let commands = definitions();
    http.interaction(app_id)
        .set_global_commands(&commands)
        .await?;
    tracing::info!(count = commands.len(), "registered slash commands");
    Ok(())
}

/// Top-level interaction dispatcher. Routes to the right command handler
/// and replies with an ephemeral message (visible only to the invoker).
pub async fn handle(
    http: Arc<HttpClient>,
    db: Db,
    default_privacy: PrivacyMode,
    app_id: Id<ApplicationMarker>,
    interaction: Interaction,
) {
    let Some(InteractionData::ApplicationCommand(data)) = interaction.data.as_ref() else {
        return;
    };

    let response = match data.name.as_str() {
        "grok-privacy" => handle_privacy(&db, &interaction, data).await,
        "grok-mode" => handle_mode(&db, &default_privacy, &interaction, data).await,
        other => {
            tracing::warn!(name = other, "unknown slash command");
            ephemeral("Unknown command — try `/grok-privacy` or `/grok-mode`.")
        }
    };

    if let Err(err) = http
        .interaction(app_id)
        .create_response(interaction.id, &interaction.token, &response)
        .await
    {
        tracing::error!(error = %err, "failed to respond to interaction");
    }
}

async fn handle_privacy(
    db: &Db,
    interaction: &Interaction,
    data: &CommandData,
) -> InteractionResponse {
    let Some(guild_id) = interaction.guild_id else {
        return ephemeral(
            "Privacy preferences are per-server. Run this command from inside a server channel.",
        );
    };
    let Some(user) = interaction.author() else {
        return ephemeral("Could not determine your user id — try again.");
    };

    let guild_id_i64 = i64::try_from(guild_id.get()).unwrap_or(i64::MAX);
    let user_id_i64 = i64::try_from(user.id.get()).unwrap_or(i64::MAX);

    let Some(sub) = data.options.first() else {
        return ephemeral("Missing subcommand.");
    };

    match sub.name.as_str() {
        "in" => match db.set_user_privacy(guild_id_i64, user_id_i64, true).await {
            Ok(()) => ephemeral(
                "✅ Opted **in**. Grok may now use your quoted messages as context here.",
            ),
            Err(err) => {
                tracing::error!(error = %err, "set_user_privacy in failed");
                ephemeral("Sorry — couldn't save that. Try again.")
            }
        },
        "out" => match db.set_user_privacy(guild_id_i64, user_id_i64, false).await {
            Ok(()) => ephemeral(
                "✅ Opted **out**. Grok will not use your quoted messages as context here. \
                 (Your direct `@Grok` mentions and any messages inside a Grok thread are \
                  still visible — that's how the bot can answer at all.)",
            ),
            Err(err) => {
                tracing::error!(error = %err, "set_user_privacy out failed");
                ephemeral("Sorry — couldn't save that. Try again.")
            }
        },
        "status" => match db.get_user_privacy(guild_id_i64, user_id_i64).await {
            Ok(Some(true)) => ephemeral("You are opted **in** here."),
            Ok(Some(false)) | Ok(None) => ephemeral(
                "You are opted **out** here (the default). Use `/grok-privacy in` to opt in.",
            ),
            Err(err) => {
                tracing::error!(error = %err, "get_user_privacy failed");
                ephemeral("Sorry — couldn't read your preference. Try again.")
            }
        },
        other => ephemeral(&format!("Unknown subcommand `{other}`.")),
    }
}

async fn handle_mode(
    db: &Db,
    default_privacy: &PrivacyMode,
    interaction: &Interaction,
    data: &CommandData,
) -> InteractionResponse {
    let Some(guild_id) = interaction.guild_id else {
        return ephemeral(
            "Privacy mode is per-server. Run this command from inside a server channel.",
        );
    };
    let guild_id_i64 = i64::try_from(guild_id.get()).unwrap_or(i64::MAX);

    let Some(sub) = data.options.first() else {
        return ephemeral("Missing subcommand.");
    };

    match sub.name.as_str() {
        "show" => {
            match db.guild_privacy_mode_or(guild_id_i64, default_privacy).await {
                Ok(mode) => ephemeral(&format!(
                    "Current mode: `{}`\n\n```\n{}\n```",
                    privacy_mode_short_name(&mode),
                    pretty_mode(&mode),
                )),
                Err(err) => {
                    tracing::error!(error = %err, "get_guild_privacy_mode failed");
                    ephemeral("Sorry — couldn't read the mode. Try again.")
                }
            }
        }
        "set" => {
            let options = match &sub.value {
                CommandOptionValue::SubCommand(opts) => opts.as_slice(),
                _ => return ephemeral("Malformed subcommand."),
            };
            let mode_str = match find_string(options, "mode") {
                Some(s) => s,
                None => return ephemeral("Missing required `mode` option."),
            };
            let channel_id = find_channel(options, "channel");
            let history_size =
                find_integer(options, "history_size").map(|n| n.max(1) as u32);

            let new_mode = match build_mode(mode_str, channel_id, history_size) {
                Ok(m) => m,
                Err(msg) => return ephemeral(msg),
            };

            match db.set_guild_privacy_mode(guild_id_i64, &new_mode).await {
                Ok(()) => ephemeral(&format!(
                    "✅ Mode set to `{}`.\n```\n{}\n```",
                    privacy_mode_short_name(&new_mode),
                    pretty_mode(&new_mode),
                )),
                Err(err) => {
                    tracing::error!(error = %err, "set_guild_privacy_mode failed");
                    ephemeral("Sorry — couldn't save that. Try again.")
                }
            }
        }
        other => ephemeral(&format!("Unknown subcommand `{other}`.")),
    }
}

fn build_mode(
    mode: &str,
    channel_id: Option<u64>,
    history_size: Option<u32>,
) -> Result<PrivacyMode, &'static str> {
    match mode {
        "open" => Ok(PrivacyMode::Open {
            history_size: history_size.unwrap_or(20),
        }),
        "channel_only" => {
            let Some(cid) = channel_id else {
                return Err("`channel_only` requires the `channel` option.");
            };
            Ok(PrivacyMode::ChannelOnly {
                channel_id: cid,
                history_size: history_size.unwrap_or(20),
            })
        }
        "opt_in" => Ok(PrivacyMode::OptIn),
        "conversation_only" => Ok(PrivacyMode::ConversationOnly),
        _ => Err("Unknown mode."),
    }
}

fn privacy_mode_short_name(mode: &PrivacyMode) -> &'static str {
    match mode {
        PrivacyMode::Open { .. } => "open",
        PrivacyMode::ChannelOnly { .. } => "channel_only",
        PrivacyMode::OptIn => "opt_in",
        PrivacyMode::ConversationOnly => "conversation_only",
    }
}

fn pretty_mode(mode: &PrivacyMode) -> String {
    serde_json::to_string_pretty(mode).unwrap_or_else(|_| "<unprintable>".to_string())
}

fn find_string<'a>(options: &'a [CommandDataOption], name: &str) -> Option<&'a str> {
    options.iter().find(|o| o.name == name).and_then(|o| match &o.value {
        CommandOptionValue::String(s) => Some(s.as_str()),
        _ => None,
    })
}

fn find_integer(options: &[CommandDataOption], name: &str) -> Option<i64> {
    options.iter().find(|o| o.name == name).and_then(|o| match &o.value {
        CommandOptionValue::Integer(n) => Some(*n),
        _ => None,
    })
}

fn find_channel(options: &[CommandDataOption], name: &str) -> Option<u64> {
    options.iter().find(|o| o.name == name).and_then(|o| match &o.value {
        CommandOptionValue::Channel(id) => Some(id.get()),
        _ => None,
    })
}

fn ephemeral(content: &str) -> InteractionResponse {
    InteractionResponse {
        kind: InteractionResponseType::ChannelMessageWithSource,
        data: Some(
            InteractionResponseDataBuilder::new()
                .content(content)
                .flags(MessageFlags::EPHEMERAL)
                .build(),
        ),
    }
}

// Suppress "unused" warning until we use CommandOptionType somewhere
// (helps if the import is here for future expansion); silently keeping
// it referenced via this no-op helper.
#[allow(dead_code)]
fn _force_use(_t: CommandOptionType) {}
