//! Slash command registration and dispatch.
//!
//! Top-level commands registered globally on startup:
//!   - `/grok-privacy {in|out|status}` — per-user, anyone can run.
//!   - `/grok-mode {show|set}` — guild admins only; sets the
//!     guild-wide context-gathering policy (one of four "designs").
//!   - `/grok-persona {set|show|list|clear}` — picks which persona the
//!     bot uses, scoped per-conversation, per-user, per-channel, or
//!     per-guild. Guild and channel scopes require admin; user and
//!     conversation scopes are self-service.
//!
//! Commands invoked from DMs (no `guild_id`) return a polite error
//! when the requested scope is per-guild.

use std::collections::HashMap;
use std::sync::Arc;

use grok_discord_bot_core::{Db, Persona, PrivacyMode};
use twilight_http::Client as HttpClient;
use twilight_model::application::command::{Command, CommandOptionType};
use twilight_model::application::interaction::application_command::{
    CommandData, CommandDataOption, CommandOptionValue,
};
use twilight_model::application::interaction::{Interaction, InteractionData};
use twilight_model::channel::ChannelType;
use twilight_model::channel::message::MessageFlags;
use twilight_model::guild::Permissions;
use twilight_model::http::interaction::{InteractionResponse, InteractionResponseType};
use twilight_model::id::Id;
use twilight_model::id::marker::{ApplicationMarker, GuildMarker};
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
    .option(SubCommandBuilder::new("status", "Show your current privacy preference here").build())
    .build();

    let mode = CommandBuilder::new(
        "grok-mode",
        "Configure how the bot gathers context in this server",
        twilight_model::application::command::CommandType::ChatInput,
    )
    .default_member_permissions(Permissions::ADMINISTRATOR)
    .option(SubCommandBuilder::new("show", "Show the active privacy mode for this server").build())
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

    let persona = CommandBuilder::new(
        "grok-persona",
        "Pick which persona the bot uses; scope it per-conversation, per-user, per-channel, or per-guild",
        twilight_model::application::command::CommandType::ChatInput,
    )
    .option(
        SubCommandBuilder::new("set", "Pick a persona for a scope")
            .option(
                StringBuilder::new("name", "Persona name from config")
                    .required(true),
            )
            .option(
                StringBuilder::new("scope", "Which scope this override applies to")
                    .required(true)
                    .choices([
                        ("Just this conversation (thread)", "conversation"),
                        ("Just me (this server)", "user"),
                        ("This channel (admin)", "channel"),
                        ("Whole server (admin)", "guild"),
                    ]),
            )
            .build(),
    )
    .option(
        SubCommandBuilder::new(
            "show",
            "Show which persona is active here and where it came from",
        )
        .build(),
    )
    .option(SubCommandBuilder::new("list", "List available personas from config").build())
    .option(
        SubCommandBuilder::new("clear", "Remove a persona override")
            .option(
                StringBuilder::new("scope", "Scope whose override to clear")
                    .required(true)
                    .choices([
                        ("Conversation (thread)", "conversation"),
                        ("Me (this server)", "user"),
                        ("Channel (admin)", "channel"),
                        ("Server (admin)", "guild"),
                    ]),
            )
            .build(),
    )
    .build();

    vec![privacy, mode, persona]
}

/// Push the command set to Discord. Idempotent — Discord replaces the
/// entire registered set with what we send. When `dev_guild_id` is
/// `Some`, registers as guild commands (visible instantly in that
/// guild only); when `None`, registers globally (up to ~1 hour to
/// propagate, then visible in every guild the bot is in).
pub async fn register(
    http: &HttpClient,
    app_id: Id<ApplicationMarker>,
    dev_guild_id: Option<u64>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let commands = definitions();
    let interaction = http.interaction(app_id);
    match dev_guild_id {
        Some(gid) => {
            let guild = Id::<GuildMarker>::new(gid);
            interaction.set_guild_commands(guild, &commands).await?;
            tracing::info!(
                count = commands.len(),
                guild = %guild,
                "registered slash commands to guild (instant)"
            );
        }
        None => {
            interaction.set_global_commands(&commands).await?;
            tracing::info!(
                count = commands.len(),
                "registered slash commands globally (up to ~1h propagation)"
            );
        }
    }
    Ok(())
}

/// Top-level interaction dispatcher. Routes to the right command handler
/// and replies with an ephemeral message (visible only to the invoker).
#[allow(clippy::too_many_arguments)]
pub async fn handle(
    http: Arc<HttpClient>,
    db: Db,
    default_privacy: PrivacyMode,
    personas: HashMap<String, Persona>,
    default_persona: String,
    app_id: Id<ApplicationMarker>,
    interaction: Interaction,
) {
    let Some(InteractionData::ApplicationCommand(data)) = interaction.data.as_ref() else {
        return;
    };

    let response = match data.name.as_str() {
        "grok-privacy" => handle_privacy(&db, &interaction, data).await,
        "grok-mode" => handle_mode(&db, &default_privacy, &interaction, data).await,
        "grok-persona" => {
            handle_persona(&db, &personas, &default_persona, &interaction, data).await
        }
        other => {
            tracing::warn!(name = other, "unknown slash command");
            ephemeral("Unknown command — try `/grok-privacy`, `/grok-mode`, or `/grok-persona`.")
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
            Ok(()) => {
                ephemeral("✅ Opted **in**. Grok may now use your quoted messages as context here.")
            }
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
            match db
                .guild_privacy_mode_or(guild_id_i64, default_privacy)
                .await
            {
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
            let history_size = find_integer(options, "history_size").map(|n| n.max(1) as u32);

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

async fn handle_persona(
    db: &Db,
    personas: &HashMap<String, Persona>,
    default_persona: &str,
    interaction: &Interaction,
    data: &CommandData,
) -> InteractionResponse {
    let Some(sub) = data.options.first() else {
        return ephemeral("Missing subcommand.");
    };

    match sub.name.as_str() {
        "list" => persona_list_response(personas, default_persona),
        "show" => handle_persona_show(db, personas, default_persona, interaction).await,
        "set" => handle_persona_set(db, personas, interaction, sub).await,
        "clear" => handle_persona_clear(db, interaction, sub).await,
        other => ephemeral(&format!("Unknown subcommand `{other}`.")),
    }
}

fn persona_list_response(
    personas: &HashMap<String, Persona>,
    default_persona: &str,
) -> InteractionResponse {
    let mut names: Vec<&String> = personas.keys().collect();
    names.sort();
    let mut out = String::from("**Available personas**\n");
    for name in names {
        let p = &personas[name];
        let marker = if name == default_persona {
            " (default)"
        } else {
            ""
        };
        out.push_str(&format!(
            "• `{name}`{marker} — `{}` / `{}`\n",
            p.provider.as_str(),
            p.model,
        ));
    }
    ephemeral(&out)
}

async fn handle_persona_show(
    db: &Db,
    personas: &HashMap<String, Persona>,
    default_persona: &str,
    interaction: &Interaction,
) -> InteractionResponse {
    let guild_id = interaction
        .guild_id
        .map(|g| i64::try_from(g.get()).unwrap_or(i64::MAX));
    // The raw channel id is the thread the user is sitting in (when in
    // a thread), used to resolve which conversation they're in. The
    // parent channel id is what `channel`-scope overrides key off of.
    let raw_channel_id = interaction
        .channel
        .as_ref()
        .map(|c| i64::try_from(c.id.get()).unwrap_or(i64::MAX));
    let channel_id = interaction.channel.as_ref().map(|c| {
        let effective = match c.kind {
            ChannelType::AnnouncementThread
            | ChannelType::PublicThread
            | ChannelType::PrivateThread => c.parent_id.unwrap_or(c.id),
            _ => c.id,
        };
        i64::try_from(effective.get()).unwrap_or(i64::MAX)
    });
    let user_id = interaction
        .author()
        .map(|u| i64::try_from(u.id.get()).unwrap_or(i64::MAX));

    let conversation_id = match raw_channel_id {
        Some(cid) => db.lookup_conversation_by_message(cid).await.ok().flatten(),
        None => None,
    };

    let mut lines = vec!["**Persona resolution here**".to_string()];

    let conv_pick = match conversation_id {
        Some(cid) => db
            .get_persona_selection("conversation", &cid.to_string())
            .await
            .ok()
            .flatten(),
        None => None,
    };
    lines.push(format!(
        "• conversation: {}",
        conv_pick
            .as_deref()
            .map(|n| format!("`{n}`"))
            .unwrap_or_else(|| "—".into())
    ));

    let user_pick = match (guild_id, user_id) {
        (Some(g), Some(u)) => db
            .get_persona_selection("user", &format!("{g}:{u}"))
            .await
            .ok()
            .flatten(),
        _ => None,
    };
    lines.push(format!(
        "• user: {}",
        user_pick
            .as_deref()
            .map(|n| format!("`{n}`"))
            .unwrap_or_else(|| "—".into())
    ));

    let channel_pick = match channel_id {
        Some(c) => db
            .get_persona_selection("channel", &c.to_string())
            .await
            .ok()
            .flatten(),
        None => None,
    };
    lines.push(format!(
        "• channel: {}",
        channel_pick
            .as_deref()
            .map(|n| format!("`{n}`"))
            .unwrap_or_else(|| "—".into())
    ));

    let guild_pick = match guild_id {
        Some(g) => db
            .get_persona_selection("guild", &g.to_string())
            .await
            .ok()
            .flatten(),
        None => None,
    };
    lines.push(format!(
        "• guild: {}",
        guild_pick
            .as_deref()
            .map(|n| format!("`{n}`"))
            .unwrap_or_else(|| "—".into())
    ));

    lines.push(format!("• fallback: `{default_persona}` (from config)"));

    let active_name = conv_pick
        .or(user_pick)
        .or(channel_pick)
        .or(guild_pick)
        .unwrap_or_else(|| default_persona.to_string());
    let active = personas.get(&active_name);
    let active_line = match active {
        Some(p) => format!(
            "\n**Active**: `{active_name}` — `{}` / `{}`",
            p.provider.as_str(),
            p.model,
        ),
        None => format!(
            "\n**Active**: `{active_name}` ⚠️ (not in current config — falling back to `{default_persona}`)",
        ),
    };

    let mut out = lines.join("\n");
    out.push_str(&active_line);
    ephemeral(&out)
}

async fn handle_persona_set(
    db: &Db,
    personas: &HashMap<String, Persona>,
    interaction: &Interaction,
    sub: &CommandDataOption,
) -> InteractionResponse {
    let options = match &sub.value {
        CommandOptionValue::SubCommand(opts) => opts.as_slice(),
        _ => return ephemeral("Malformed subcommand."),
    };
    let Some(name) = find_string(options, "name") else {
        return ephemeral("Missing `name`.");
    };
    let Some(scope) = find_string(options, "scope") else {
        return ephemeral("Missing `scope`.");
    };
    if !personas.contains_key(name) {
        let mut listed: Vec<&String> = personas.keys().collect();
        listed.sort();
        let avail = listed
            .iter()
            .map(|n| format!("`{n}`"))
            .collect::<Vec<_>>()
            .join(", ");
        return ephemeral(&format!("Unknown persona `{name}`. Available: {avail}"));
    }

    let key_result = build_scope_key(db, interaction, scope, true).await;
    let key = match key_result {
        Ok(k) => k,
        Err(msg) => return ephemeral(&msg),
    };

    match db.set_persona_selection(scope, &key, name).await {
        Ok(()) => ephemeral(&format!(
            "✅ Set persona for **{}** to `{name}`.",
            scope_description(scope)
        )),
        Err(err) => {
            tracing::error!(error = %err, "set_persona_selection failed");
            ephemeral("Sorry — couldn't save that. Try again.")
        }
    }
}

async fn handle_persona_clear(
    db: &Db,
    interaction: &Interaction,
    sub: &CommandDataOption,
) -> InteractionResponse {
    let options = match &sub.value {
        CommandOptionValue::SubCommand(opts) => opts.as_slice(),
        _ => return ephemeral("Malformed subcommand."),
    };
    let Some(scope) = find_string(options, "scope") else {
        return ephemeral("Missing `scope`.");
    };
    let key = match build_scope_key(db, interaction, scope, true).await {
        Ok(k) => k,
        Err(msg) => return ephemeral(&msg),
    };
    match db.clear_persona_selection(scope, &key).await {
        Ok(true) => ephemeral(&format!(
            "✅ Cleared persona override for **{}**.",
            scope_description(scope)
        )),
        Ok(false) => ephemeral(&format!(
            "No override was set for **{}**.",
            scope_description(scope)
        )),
        Err(err) => {
            tracing::error!(error = %err, "clear_persona_selection failed");
            ephemeral("Sorry — couldn't clear that. Try again.")
        }
    }
}

/// Compute the `persona_selections.key` for a given scope from the
/// interaction context, returning a human-readable error string when
/// the scope can't be resolved (e.g. conversation scope outside a
/// Grok thread). When `enforce_admin` is true, scopes that require
/// admin privileges (`channel`, `guild`) are gated on the invoking
/// user's permissions.
async fn build_scope_key(
    db: &Db,
    interaction: &Interaction,
    scope: &str,
    enforce_admin: bool,
) -> Result<String, String> {
    match scope {
        "conversation" => {
            let Some(channel) = interaction.channel.as_ref() else {
                return Err("Couldn't determine the channel for this interaction.".into());
            };
            let channel_id = i64::try_from(channel.id.get()).unwrap_or(i64::MAX);
            match db.lookup_conversation_by_message(channel_id).await {
                Ok(Some(conv_id)) => Ok(conv_id.to_string()),
                Ok(None) => Err(
                    "No conversation is bound to this channel. Run this inside a thread the bot \
                     opened for an answer."
                        .into(),
                ),
                Err(err) => {
                    tracing::error!(error = %err, "conversation lookup failed");
                    Err("Couldn't read conversation state. Try again.".into())
                }
            }
        }
        "user" => {
            let Some(guild_id) = interaction.guild_id else {
                return Err(
                    "User-scoped persona only makes sense in a server. Run this from a channel."
                        .into(),
                );
            };
            let Some(user) = interaction.author() else {
                return Err("Couldn't determine your user id — try again.".into());
            };
            let gid = i64::try_from(guild_id.get()).unwrap_or(i64::MAX);
            let uid = i64::try_from(user.id.get()).unwrap_or(i64::MAX);
            Ok(format!("{gid}:{uid}"))
        }
        "channel" => {
            let Some(channel) = interaction.channel.as_ref() else {
                return Err("Couldn't determine the channel for this interaction.".into());
            };
            if enforce_admin && !interaction_is_admin(interaction) {
                return Err("Channel-scoped persona requires admin privileges.".into());
            }
            // Threads roll up to their parent channel — the operator
            // expects /grok-persona scope:channel to apply to the
            // channel they can see, not to one ephemeral thread.
            let effective_id = match channel.kind {
                ChannelType::AnnouncementThread
                | ChannelType::PublicThread
                | ChannelType::PrivateThread => channel.parent_id.unwrap_or(channel.id),
                _ => channel.id,
            };
            Ok(i64::try_from(effective_id.get())
                .unwrap_or(i64::MAX)
                .to_string())
        }
        "guild" => {
            let Some(guild_id) = interaction.guild_id else {
                return Err(
                    "Guild-scoped persona only makes sense in a server. Run this from a channel."
                        .into(),
                );
            };
            if enforce_admin && !interaction_is_admin(interaction) {
                return Err("Guild-scoped persona requires admin privileges.".into());
            }
            Ok(i64::try_from(guild_id.get())
                .unwrap_or(i64::MAX)
                .to_string())
        }
        other => Err(format!("Unknown scope `{other}`.")),
    }
}

fn interaction_is_admin(interaction: &Interaction) -> bool {
    interaction
        .member
        .as_ref()
        .and_then(|m| m.permissions)
        .map(|p| p.contains(Permissions::ADMINISTRATOR))
        .unwrap_or(false)
}

fn scope_description(scope: &str) -> &'static str {
    match scope {
        "conversation" => "this conversation",
        "user" => "you in this server",
        "channel" => "this channel",
        "guild" => "this server",
        _ => "this scope",
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
    options
        .iter()
        .find(|o| o.name == name)
        .and_then(|o| match &o.value {
            CommandOptionValue::String(s) => Some(s.as_str()),
            _ => None,
        })
}

fn find_integer(options: &[CommandDataOption], name: &str) -> Option<i64> {
    options
        .iter()
        .find(|o| o.name == name)
        .and_then(|o| match &o.value {
            CommandOptionValue::Integer(n) => Some(*n),
            _ => None,
        })
}

fn find_channel(options: &[CommandDataOption], name: &str) -> Option<u64> {
    options
        .iter()
        .find(|o| o.name == name)
        .and_then(|o| match &o.value {
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
