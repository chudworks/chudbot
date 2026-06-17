//! Platform-neutral slash-command definitions and handlers.
//!
//! This module owns the command surface Chudbot registers with each platform and
//! the handlers that turn normalized `PlatformCommand` inputs into ephemeral
//! responses. The code stays on `chudbot-api` types so platform adapters can
//! translate Discord or future interaction payloads before they reach bot
//! orchestration.
//!
//! Commands here mutate durable runtime state: per-user privacy opt-in records,
//! server privacy modes, and scoped agent overrides.

use crate::prelude::*;
use crate::*;

impl<R> BotRuntime<R>
where
    R: BotRuntimeTypes + 'static,
{
    /// Dispatch a normalized platform command and send the platform response.
    ///
    /// Recoverable command-input errors are rendered as the ephemeral command
    /// response body; storage and platform failures still bubble out as
    /// `BotError`s.
    pub(crate) async fn handle_command(
        &self,
        command: PlatformCommand,
    ) -> Result<BotAction, BotError> {
        // Step 1: route by the stable command names registered below.
        let handled = match command.name.as_str() {
            "chudbot-privacy" => self.handle_privacy_command(&command).await,
            "chudbot-mode" => self.handle_mode_command(&command).await,
            "chudbot-agent" => self.handle_agent_command(&command).await,
            other => {
                tracing::warn!(name = other, "unknown command");
                Ok("Unknown command. Try `/chudbot-privacy`, `/chudbot-mode`, or `/chudbot-agent`."
                    .to_string())
            }
        };

        // Step 2: command-input problems are user-facing validation messages,
        // while operational failures should be handled by the event runner.
        let content = match handled {
            Ok(content) => content,
            Err(BotError::CommandInput(message)) => message,
            Err(error) => return Err(error),
        };

        // Step 3: slash-command replies are private by default so status and
        // configuration details do not clutter the channel.
        self.platforms
            .respond_to_command(PlatformCommandResponse {
                target: command.response_target,
                content,
                ephemeral: true,
            })
            .await
            .map_err(platform_error)?;
        Ok(BotAction::HandledCommand)
    }

    /// Handle `/chudbot-privacy` for the invoking user's server opt-in state.
    ///
    /// Privacy opt-in records are scoped by platform, guild, and user. Direct
    /// messages have no guild key, so the command only reports that the action
    /// must be run inside a server channel.
    pub(crate) async fn handle_privacy_command(
        &self,
        command: &PlatformCommand,
    ) -> Result<String, BotError> {
        let Some(guild) = command.channel.guild_id.as_ref() else {
            return Ok(
                "Privacy preferences are per-server. Run this from inside a server channel."
                    .to_string(),
            );
        };
        let Some(sub) = command_subcommand(command) else {
            return Ok("Missing subcommand.".to_string());
        };

        // Store only the explicit opt-in bit. Runtime privacy mode resolution
        // combines this with the guild's active mode when a turn is built.
        match sub.name.as_str() {
            "in" => {
                self.storage
                    .set_user_privacy(
                        command.channel.platform.clone(),
                        guild.as_str().to_string(),
                        command.user.user_id.as_str().to_string(),
                        true,
                    )
                    .await
                    .map_err(storage_error)?;
                Ok("Opted in. Chudbot may use your quoted messages as context here.".to_string())
            }
            "out" => {
                self.storage
                    .set_user_privacy(
                        command.channel.platform.clone(),
                        guild.as_str().to_string(),
                        command.user.user_id.as_str().to_string(),
                        false,
                    )
                    .await
                    .map_err(storage_error)?;
                Ok(
                    "Opted out. Your direct mentions and messages inside a Chudbot thread still remain visible so the bot can answer."
                        .to_string(),
                )
            }
            "status" => {
                let opted_in = self
                    .storage
                    .user_privacy(
                        command.channel.platform.clone(),
                        guild.as_str().to_string(),
                        command.user.user_id.as_str().to_string(),
                    )
                    .await
                    .map_err(storage_error)?
                    .unwrap_or(false);
                if opted_in {
                    Ok("You are opted in here.".to_string())
                } else {
                    Ok("You are opted out here. Use `/chudbot-privacy in` to opt in.".to_string())
                }
            }
            other => Ok(format!("Unknown subcommand `{other}`.")),
        }
    }

    /// Handle `/chudbot-mode`, the administrator command for server privacy mode.
    ///
    /// This command reads or updates the guild-level runtime privacy mode that
    /// controls which surrounding messages may be considered as model context.
    pub(crate) async fn handle_mode_command(
        &self,
        command: &PlatformCommand,
    ) -> Result<String, BotError> {
        let Some(guild) = command.channel.guild_id.as_ref() else {
            return Ok(
                "Privacy mode is per-server. Run this from inside a server channel.".to_string(),
            );
        };
        if !command.is_admin {
            return Ok(
                "Changing server privacy mode requires administrator privileges.".to_string(),
            );
        }
        let Some(sub) = command_subcommand(command) else {
            return Ok("Missing subcommand.".to_string());
        };
        match sub.name.as_str() {
            "show" => {
                let settings = self
                    .storage
                    .runtime_settings(
                        command.channel.platform.clone(),
                        Some(guild.as_str().to_string()),
                        command.user.user_id.as_str().to_string(),
                    )
                    .await
                    .map_err(storage_error)?;
                Ok(format!(
                    "Current mode: `{}`\n\n```json\n{}\n```",
                    privacy_mode_kind(&settings.privacy),
                    pretty_json(&settings.privacy),
                ))
            }
            "set" => {
                let mode = sub_option_string(&sub, "mode")
                    .ok_or_else(|| BotError::CommandInput("missing `mode`".to_string()))?;
                let channel = sub_option_string(&sub, "channel");

                // Platforms should enforce the declared bounds, but clamping
                // keeps imported or replayed normalized commands safe too.
                let history_size = sub_option_integer(&sub, "history_size").map(|value| {
                    u32::try_from(value.clamp(HISTORY_SIZE_MIN, HISTORY_SIZE_MAX)).unwrap_or(20)
                });

                // Translate the flat slash-command options into the stored
                // privacy enum before persisting so invalid combinations can
                // return a command-input response.
                let privacy = command_privacy_mode(
                    command.channel.platform.clone(),
                    guild.as_str().to_string(),
                    mode,
                    channel,
                    history_size,
                )?;
                self.storage
                    .set_privacy_mode(
                        command.channel.platform.clone(),
                        guild.as_str().to_string(),
                        privacy.clone(),
                    )
                    .await
                    .map_err(storage_error)?;
                Ok(format!(
                    "Mode set to `{}`.\n```json\n{}\n```",
                    privacy_mode_kind(&privacy),
                    pretty_json(&privacy),
                ))
            }
            other => Ok(format!("Unknown subcommand `{other}`.")),
        }
    }

    /// Route `/chudbot-agent` subcommands for listing and scoped overrides.
    pub(crate) async fn handle_agent_command(
        &self,
        command: &PlatformCommand,
    ) -> Result<String, BotError> {
        let Some(sub) = command_subcommand(command) else {
            return Ok("Missing subcommand.".to_string());
        };
        match sub.name.as_str() {
            "list" => Ok(agent_list_response(&self.config)),
            "show" => self.handle_agent_show(command).await,
            "set" => self.handle_agent_set(command, &sub).await,
            "clear" => self.handle_agent_clear(command, &sub).await,
            other => Ok(format!("Unknown subcommand `{other}`.")),
        }
    }

    /// Build the `/chudbot-agent show` explanation for the current command site.
    ///
    /// The response lists every scope that can contribute an agent selection,
    /// then reports the active configured agent after applying the canonical
    /// narrow-to-broad command precedence.
    pub(crate) async fn handle_agent_show(
        &self,
        command: &PlatformCommand,
    ) -> Result<String, BotError> {
        // Step 1: derive the storage keys that depend on the current command
        // location. Channel scope intentionally uses the parent thread/channel
        // when the platform can resolve one.
        let conversation = self.command_conversation(command).await?;
        let channel = self.command_scope_channel(command).await;
        let guild = command
            .channel
            .guild_id
            .as_ref()
            .map(|id| id.as_str().to_string());

        // Step 2: load each possible override independently so the user can see
        // shadowed settings instead of only the winner.
        let conv_pick = match conversation {
            Some(conversation_id) => self
                .storage
                .load_agent_selection(AgentSelection::Conversation { conversation_id })
                .await
                .map_err(storage_error)?,
            None => None,
        };
        let user_pick = match guild.as_deref() {
            Some(guild) => self
                .storage
                .load_agent_selection(AgentSelection::User {
                    message_provider: command.channel.platform.clone(),
                    guild_key: guild.to_string(),
                    user_key: command.user.user_id.as_str().to_string(),
                })
                .await
                .map_err(storage_error)?,
            None => None,
        };
        let channel_pick = self
            .storage
            .load_agent_selection(AgentSelection::Channel {
                message_provider: command.channel.platform.clone(),
                guild_key: guild.clone(),
                channel_key: channel.channel_id.as_str().to_string(),
            })
            .await
            .map_err(storage_error)?;
        let guild_pick = match guild.as_deref() {
            Some(guild) => self
                .storage
                .load_agent_selection(AgentSelection::Guild {
                    message_provider: command.channel.platform.clone(),
                    guild_key: guild.to_string(),
                })
                .await
                .map_err(storage_error)?,
            None => None,
        };
        let platform_pick = self
            .storage
            .load_agent_selection(AgentSelection::Platform {
                message_provider: command.channel.platform.clone(),
            })
            .await
            .map_err(storage_error)?;

        // Step 3: resolve the active name from most specific to broadest scope,
        // falling back to the configured default only when storage has no pick.
        let active_name = conv_pick
            .clone()
            .or_else(|| user_pick.clone())
            .or_else(|| channel_pick.clone())
            .or_else(|| guild_pick.clone())
            .or_else(|| platform_pick.clone())
            .unwrap_or_else(|| self.config.default_agent.clone());
        let active = self.config.agents.get(&active_name);

        // Step 4: render a compact diagnostic that is useful in an ephemeral
        // command response.
        let mut out = String::from("Agent resolution here\n");
        out.push_str(&format!(
            "conversation: {}\n",
            option_tick(conv_pick.as_deref())
        ));
        out.push_str(&format!("user: {}\n", option_tick(user_pick.as_deref())));
        out.push_str(&format!(
            "channel: {}\n",
            option_tick(channel_pick.as_deref())
        ));
        out.push_str(&format!("guild: {}\n", option_tick(guild_pick.as_deref())));
        out.push_str(&format!(
            "platform: {}\n",
            option_tick(platform_pick.as_deref())
        ));
        out.push_str(&format!("fallback: `{}`\n", self.config.default_agent));
        match active {
            Some(agent) => out.push_str(&format!(
                "\nActive: `{active_name}`: `{}` / `{}`",
                agent.provider, agent.model.id
            )),
            None => out.push_str(&format!(
                "\nActive: `{active_name}` is no longer configured; falling back to `{}`",
                self.config.default_agent
            )),
        }
        Ok(out)
    }

    /// Store a configured agent override for the requested `/chudbot-agent` scope.
    ///
    /// User-scoped selections are self-service, while channel and guild scopes
    /// are protected by `command_agent_selection` when `enforce_admin` is true.
    pub(crate) async fn handle_agent_set(
        &self,
        command: &PlatformCommand,
        sub: &PlatformCommandInput,
    ) -> Result<String, BotError> {
        let Some(name) = sub_option_string(sub, "name") else {
            return Ok("Missing `name`.".to_string());
        };
        if is_system_agent_name(name) {
            return Ok(format!(
                "`{name}` is reserved for internal system use. {}",
                available_agents(&self.config)
            ));
        }
        if !self.config.agents.contains_key(name) {
            return Ok(format!(
                "Unknown agent `{name}`. {}",
                available_agents(&self.config)
            ));
        }
        let Some(scope) = sub_option_string(sub, "scope") else {
            return Ok("Missing `scope`.".to_string());
        };
        let selection = self.command_agent_selection(command, scope, true).await?;
        self.storage
            .set_agent_selection(selection, name.to_string())
            .await
            .map_err(storage_error)?;
        Ok(format!(
            "Set agent for {} to `{name}`.",
            scope_description(scope)
        ))
    }

    /// Clear a configured agent override for the requested `/chudbot-agent` scope.
    pub(crate) async fn handle_agent_clear(
        &self,
        command: &PlatformCommand,
        sub: &PlatformCommandInput,
    ) -> Result<String, BotError> {
        let Some(scope) = sub_option_string(sub, "scope") else {
            return Ok("Missing `scope`.".to_string());
        };
        let selection = self.command_agent_selection(command, scope, true).await?;
        let cleared = self
            .storage
            .clear_agent_selection(selection)
            .await
            .map_err(storage_error)?;
        if cleared {
            Ok(format!(
                "Cleared agent override for {}.",
                scope_description(scope)
            ))
        } else {
            Ok(format!(
                "No agent override was set for {}.",
                scope_description(scope)
            ))
        }
    }

    /// Convert a user-visible agent scope into the storage selection key.
    ///
    /// The `enforce_admin` flag lets callers reuse the mapper for read-only
    /// diagnostics if needed while still requiring administrators for channel
    /// and guild mutations in the command handlers.
    pub(crate) async fn command_agent_selection(
        &self,
        command: &PlatformCommand,
        scope: &str,
        enforce_admin: bool,
    ) -> Result<AgentSelection, BotError> {
        match scope {
            "conversation" => {
                // Conversation overrides require the command to run in a
                // channel already linked to a stored conversation.
                let Some(conversation_id) = self.command_conversation(command).await? else {
                    return Err(BotError::CommandInput(
                        "No conversation is bound to this channel. Run this inside a thread the bot opened for an answer."
                            .to_string(),
                    ));
                };
                Ok(AgentSelection::Conversation { conversation_id })
            }
            "user" => {
                let Some(guild) = command.channel.guild_id.as_ref() else {
                    return Err(BotError::CommandInput(
                        "User-scoped agent selection only makes sense in a server.".to_string(),
                    ));
                };
                Ok(AgentSelection::User {
                    message_provider: command.channel.platform.clone(),
                    guild_key: guild.as_str().to_string(),
                    user_key: command.user.user_id.as_str().to_string(),
                })
            }
            "channel" => {
                if enforce_admin && !command.is_admin {
                    return Err(BotError::CommandInput(
                        "Channel-scoped agent selection requires administrator privileges."
                            .to_string(),
                    ));
                }
                let channel = self.command_scope_channel(command).await;

                // Match turn-time agent resolution by keying thread commands to
                // their parent channel when a platform exposes that relationship.
                Ok(AgentSelection::Channel {
                    message_provider: command.channel.platform.clone(),
                    guild_key: command
                        .channel
                        .guild_id
                        .as_ref()
                        .map(|id| id.as_str().to_string()),
                    channel_key: channel.channel_id.as_str().to_string(),
                })
            }
            "guild" => {
                if enforce_admin && !command.is_admin {
                    return Err(BotError::CommandInput(
                        "Guild-scoped agent selection requires administrator privileges."
                            .to_string(),
                    ));
                }
                let Some(guild) = command.channel.guild_id.as_ref() else {
                    return Err(BotError::CommandInput(
                        "Guild-scoped agent selection only makes sense in a server.".to_string(),
                    ));
                };
                Ok(AgentSelection::Guild {
                    message_provider: command.channel.platform.clone(),
                    guild_key: guild.as_str().to_string(),
                })
            }
            other => Err(BotError::CommandInput(format!("Unknown scope `{other}`."))),
        }
    }

    /// Find the conversation currently bound to the command channel, if any.
    pub(crate) async fn command_conversation(
        &self,
        command: &PlatformCommand,
    ) -> Result<Option<ConversationId>, BotError> {
        let snapshot = self
            .storage
            .load_conversation(ConversationLookup::Channel {
                channel: command.channel.clone(),
            })
            .await
            .map_err(storage_error)?;
        Ok(snapshot.map(|snapshot| snapshot.conversation.id))
    }

    /// Resolve the channel key used for command-scoped settings.
    ///
    /// Platforms such as Discord may invoke commands inside threads. Agent
    /// channel overrides should usually apply to the parent channel, matching
    /// turn-time agent resolution, so failures fall back to the interaction
    /// channel with a warning.
    pub(crate) async fn command_scope_channel(&self, command: &PlatformCommand) -> ChannelRef {
        match self.platforms.parent_channel(command.channel.clone()).await {
            Ok(parent) => parent,
            Err(error) => {
                tracing::warn!(
                    error = %error,
                    channel = %command.channel.channel_id,
                    "failed to resolve command parent channel; using interaction channel"
                );
                command.channel.clone()
            }
        }
    }
}

/// Return the platform-neutral slash-command definitions registered at startup.
///
/// Platform adapters translate these definitions into native command
/// registrations, but the names and option values here are also the stable
/// strings consumed by the handlers above.
pub(crate) fn command_definitions() -> Vec<PlatformCommandDefinition> {
    vec![
        PlatformCommandDefinition {
            name: "chudbot-privacy".to_string(),
            description: "Manage your personal Chudbot privacy preference in this server"
                .to_string(),
            admin_only: false,
            options: vec![
                subcommand(
                    "in",
                    "Allow Chudbot to use your messages as quoted-message context",
                    Vec::new(),
                ),
                subcommand(
                    "out",
                    "Stop letting Chudbot use your messages as quoted-message context",
                    Vec::new(),
                ),
                subcommand(
                    "status",
                    "Show your current privacy preference here",
                    Vec::new(),
                ),
            ],
        },
        PlatformCommandDefinition {
            name: "chudbot-mode".to_string(),
            description: "Configure how Chudbot gathers context in this server".to_string(),
            admin_only: true,
            options: vec![
                subcommand(
                    "show",
                    "Show the active privacy mode for this server",
                    Vec::new(),
                ),
                subcommand(
                    "set",
                    "Change the privacy mode for this server",
                    vec![
                        string_option(
                            "mode",
                            "Which context-gathering mode to use",
                            true,
                            vec![
                                choice("Open: see recent channel history", "open"),
                                choice("Channel only", "channel_only"),
                                choice("Opt-in", "opt_in"),
                                choice("Conversation only", "conversation_only"),
                            ],
                        ),
                        option(
                            "channel",
                            "Channel for channel_only mode",
                            PlatformCommandOptionKind::Channel,
                            false,
                        ),
                        integer_option(
                            "history_size",
                            "How many recent channel messages to include",
                            false,
                            Some(HISTORY_SIZE_MIN),
                            Some(HISTORY_SIZE_MAX),
                        ),
                    ],
                ),
            ],
        },
        PlatformCommandDefinition {
            name: "chudbot-agent".to_string(),
            description: "Pick which configured agent Chudbot uses".to_string(),
            admin_only: false,
            options: vec![
                subcommand(
                    "set",
                    "Pick an agent for a scope",
                    vec![
                        option(
                            "name",
                            "Agent name from config",
                            PlatformCommandOptionKind::String,
                            true,
                        ),
                        string_option(
                            "scope",
                            "Which scope this override applies to",
                            true,
                            vec![
                                choice("This conversation", "conversation"),
                                choice("Me in this server", "user"),
                                choice("This channel", "channel"),
                                choice("This server", "guild"),
                            ],
                        ),
                    ],
                ),
                subcommand(
                    "show",
                    "Show which agent is active here and why",
                    Vec::new(),
                ),
                subcommand("list", "List available configured agents", Vec::new()),
                subcommand(
                    "clear",
                    "Remove an agent override",
                    vec![string_option(
                        "scope",
                        "Scope whose override to clear",
                        true,
                        vec![
                            choice("This conversation", "conversation"),
                            choice("Me in this server", "user"),
                            choice("This channel", "channel"),
                            choice("This server", "guild"),
                        ],
                    )],
                ),
            ],
        },
    ]
}

/// Build a bare command option definition with no choices or nested options.
pub(crate) fn option(
    name: &str,
    description: &str,
    kind: PlatformCommandOptionKind,
    required: bool,
) -> PlatformCommandOption {
    PlatformCommandOption {
        name: name.to_string(),
        description: description.to_string(),
        kind,
        required,
        choices: Vec::new(),
        options: Vec::new(),
        min_integer: None,
        max_integer: None,
    }
}

/// Build a top-level subcommand option with nested arguments.
pub(crate) fn subcommand(
    name: &str,
    description: &str,
    options: Vec<PlatformCommandOption>,
) -> PlatformCommandOption {
    PlatformCommandOption {
        options,
        ..option(
            name,
            description,
            PlatformCommandOptionKind::SubCommand,
            false,
        )
    }
}

/// Build a string option definition, optionally constrained to choices.
pub(crate) fn string_option(
    name: &str,
    description: &str,
    required: bool,
    choices: Vec<PlatformCommandOptionChoice>,
) -> PlatformCommandOption {
    PlatformCommandOption {
        choices,
        ..option(
            name,
            description,
            PlatformCommandOptionKind::String,
            required,
        )
    }
}

/// Build an integer option definition with optional platform-enforced bounds.
pub(crate) fn integer_option(
    name: &str,
    description: &str,
    required: bool,
    min_integer: Option<i64>,
    max_integer: Option<i64>,
) -> PlatformCommandOption {
    PlatformCommandOption {
        min_integer,
        max_integer,
        ..option(
            name,
            description,
            PlatformCommandOptionKind::Integer,
            required,
        )
    }
}

/// Build a user-visible choice label and its stable stored value.
pub(crate) fn choice(name: &str, value: &str) -> PlatformCommandOptionChoice {
    PlatformCommandOptionChoice {
        name: name.to_string(),
        value: value.to_string(),
    }
}

/// Return the only top-level subcommand supplied to these commands.
///
/// The current command definitions do not use subcommand groups, so the first
/// normalized option is the subcommand selected by the user.
pub(crate) fn command_subcommand(command: &PlatformCommand) -> Option<PlatformCommandInput> {
    command.options.first().cloned()
}

/// Extract a string-like nested option from a subcommand.
///
/// Channel options are returned as their channel id string so callers can pass
/// the value directly into storage-key builders.
pub(crate) fn sub_option_string<'a>(
    option: &'a PlatformCommandInput,
    name: &str,
) -> Option<&'a str> {
    option
        .options
        .iter()
        .find(|option| option.name == name)
        .and_then(|option| option.value.as_ref())
        .and_then(|value| match value {
            PlatformCommandValue::String(value) => Some(value.as_str()),
            PlatformCommandValue::Channel(channel) => Some(channel.channel_id.as_str()),
            _ => None,
        })
}

/// Extract an integer nested option from a subcommand.
pub(crate) fn sub_option_integer(option: &PlatformCommandInput, name: &str) -> Option<i64> {
    option
        .options
        .iter()
        .find(|option| option.name == name)
        .and_then(|option| option.value.as_ref())
        .and_then(|value| match value {
            PlatformCommandValue::Integer(value) => Some(*value),
            _ => None,
        })
}

/// Convert `/chudbot-mode set` options into the stored `PrivacyMode`.
///
/// `channel_only` is the only mode that needs a channel id; other modes ignore
/// the optional channel argument but still share the optional history size.
pub(crate) fn command_privacy_mode(
    platform: PlatformName,
    guild: String,
    mode: &str,
    channel: Option<&str>,
    history_size: Option<u32>,
) -> Result<PrivacyMode, BotError> {
    match mode {
        "open" => Ok(PrivacyMode::Open {
            history_size: history_size.unwrap_or(20),
        }),
        "channel_only" => {
            let Some(channel) = channel else {
                return Err(BotError::CommandInput(
                    "`channel_only` requires the `channel` option.".to_string(),
                ));
            };
            Ok(PrivacyMode::ChannelOnly {
                channel: ChannelRef {
                    platform,
                    guild_id: Some(guild.into()),
                    channel_id: channel.into(),
                },
                history_size: history_size.unwrap_or(20),
            })
        }
        "opt_in" => Ok(PrivacyMode::OptIn),
        "conversation_only" => Ok(PrivacyMode::ConversationOnly),
        other => Err(BotError::CommandInput(format!("Unknown mode `{other}`."))),
    }
}

/// Render the user-selectable configured agents for `/chudbot-agent list`.
///
/// Reserved system agents are hidden because they are implementation details
/// for memory, safety, and title-generation jobs rather than interactive chat
/// agents.
pub(crate) fn agent_list_response(config: &BotConfig) -> String {
    let mut out = String::from("Available agents\n");
    for (name, agent) in &config.agents {
        if is_system_agent_name(name) {
            continue;
        }
        let marker = if name == &config.default_agent {
            " (default)"
        } else {
            ""
        };
        out.push_str(&format!(
            "`{name}`{marker}: `{}` / `{}`\n",
            agent.provider, agent.model.id
        ));
    }
    out
}

/// Render the same user-selectable agent set as a short inline help fragment.
pub(crate) fn available_agents(config: &BotConfig) -> String {
    let names = config
        .agents
        .keys()
        .filter(|name| !is_system_agent_name(name))
        .map(|name| format!("`{name}`"))
        .collect::<Vec<_>>()
        .join(", ");
    format!("Available agents: {names}")
}

/// Return whether an agent name is reserved for internal background work.
pub(crate) fn is_system_agent_name(name: &str) -> bool {
    matches!(
        name,
        memory::MEMORY_DIARY_AGENT
            | memory::MEMORY_COMPACT_AGENT
            | TOS_PREFLIGHT_AGENT
            | CONVERSATION_TITLE_AGENT
    )
}

/// Add a required client tool to an explicit allowlist.
///
/// `None` means the agent allows all client tools, so only `Some` lists need to
/// be updated.
pub(crate) fn ensure_client_tool_enabled(tools: &mut Option<Vec<ToolName>>, name: &str) {
    let Some(tools) = tools else {
        return;
    };
    if tools.iter().any(|tool| tool.as_str() == name) {
        return;
    }
    tools.push(ToolName::new(name));
}

/// Format an optional selected value for the agent-resolution status display.
pub(crate) fn option_tick(value: Option<&str>) -> String {
    value
        .map(|value| format!("`{value}`"))
        .unwrap_or_else(|| "-".to_string())
}

/// Convert a stable scope value into user-facing response text.
pub(crate) fn scope_description(scope: &str) -> &'static str {
    match scope {
        "conversation" => "this conversation",
        "user" => "you in this server",
        "channel" => "this channel",
        "guild" => "this server",
        _ => "this scope",
    }
}

/// Pretty-print command diagnostics as JSON without failing the command path.
pub(crate) fn pretty_json<T>(value: &T) -> String
where
    T: Serialize,
{
    serde_json::to_string_pretty(value).unwrap_or_else(|_| "<unprintable>".to_string())
}
