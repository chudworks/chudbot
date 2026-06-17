//! `fetch_messages` client tool and privacy redaction.
//!
//! This is the model-facing channel-history tool. The runtime executor omits it
//! for `conversation_only` privacy; this module still owns the per-call channel
//! boundary, defensive page-size handling, and opt-in redaction before messages
//! are shaped for the model.

use super::*;

/// Tool for fetching recent platform messages subject to runtime privacy mode.
///
/// Calls are anchored to the current turn's platform and guild. Input may only
/// choose a channel id within that scope, and `channel_only` privacy rejects
/// any request outside the configured channel before a platform fetch runs.
pub(crate) struct FetchMessagesTool<P, S> {
    /// Platform registry used to fetch and render messages from the active
    /// platform implementation.
    pub(crate) platforms: P,
    /// Storage backend consulted for per-user opt-in state during redaction.
    pub(crate) storage: S,
    /// Channel associated with the current conversation turn.
    pub(crate) default_channel: ChannelRef,
    /// Effective runtime privacy mode for this turn.
    pub(crate) privacy: PrivacyMode,
}

impl<P, S> FetchMessagesTool<P, S>
where
    P: MessagePlatformRegistry + Clone,
    S: BotStorage + Clone,
{
    /// Describes the model-facing tool contract.
    ///
    /// The schema advertises the allowed page size and cursor shape to model
    /// providers. Security-sensitive checks are still repeated in [`Self::call`]
    /// because providers may be lenient about JSON Schema enforcement.
    pub(crate) fn spec(&self) -> ClientToolSpec {
        ClientToolSpec {
            description: "Fetch recent messages from the current channel for context.".to_string(),
            input_schema: ToolInputSchema::new(serde_json::json!({
                "type": "object",
                "properties": {
                    "channel_id": {
                        "type": "string",
                        "description": "Optional platform channel id. Defaults to the current channel."
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 100,
                        "default": 20
                    },
                    "before_message_id": {
                        "type": "string",
                        "description": "Optional platform message id to page before."
                    }
                },
                "additionalProperties": false
            })),
        }
    }

    /// Executes the fetch, applies privacy policy, and returns transcript-ready
    /// message context JSON.
    #[tracing::instrument(
        name = "tool.fetch_messages",
        skip_all,
        fields(
            tool_call = %call.id,
            default_platform = %self.default_channel.platform,
            default_channel = %self.default_channel.channel_id,
            privacy = privacy_mode_kind(&self.privacy),
        )
    )]
    pub(crate) async fn call(
        &self,
        call: ClientToolCall,
    ) -> Result<ClientToolOutput, BotToolError> {
        let channel = requested_channel(&self.default_channel, &call.input)?;
        // `requested_channel` keeps the platform and guild fixed; this check
        // enforces the narrower channel boundary when the deployment selected
        // channel-only history access.
        if let PrivacyMode::ChannelOnly {
            channel: allowed, ..
        } = &self.privacy
            && &channel != allowed
        {
            tracing::warn!(
                requested_channel = %channel.channel_id,
                allowed_channel = %allowed.channel_id,
                "fetch_messages rejected by channel_only privacy mode"
            );
            return Err(BotToolError::InvalidInput(
                "fetch_messages is limited to the configured channel".to_string(),
            ));
        }
        // Clamp defensively even though the schema advertises the same bounds.
        // Direct callers and some providers can still deliver out-of-range JSON.
        let limit = call
            .input
            .get("limit")
            .and_then(serde_json::Value::as_u64)
            .and_then(|value| u16::try_from(value).ok())
            .unwrap_or(20)
            .clamp(1, 100);
        // Page cursors are scoped to the resolved channel so input cannot
        // smuggle a different platform or guild through `before_message_id`.
        let before = call
            .input
            .get("before_message_id")
            .and_then(serde_json::Value::as_str)
            .map(|message_id| MessageRef {
                platform: channel.platform.clone(),
                guild_id: channel.guild_id.clone(),
                channel_id: channel.channel_id.clone(),
                message_id: message_id.into(),
            });
        // Fetch raw platform messages first; bot-specific privacy shaping is
        // applied below so platform adapters do not need storage policy logic.
        let messages = self
            .platforms
            .fetch_messages(FetchMessages {
                channel: channel.clone(),
                limit,
                before,
            })
            .await
            .map_err(|error| BotToolError::Platform(error.to_string()))?;
        let messages =
            redact_messages_for_privacy(&self.storage, &self.privacy, &channel, messages).await?;
        tracing::info!(
            messages = messages.len(),
            limit,
            "fetched platform messages"
        );
        // Render into the same context objects used in model transcripts. The
        // visible tool result and trace payload intentionally match.
        let mut rendered = Vec::with_capacity(messages.len());
        for message in &messages {
            rendered.push(
                self.platforms
                    .message_context(message, PlatformMessageRelationship::Fetched)
                    .await
                    .map_err(|error| BotToolError::Platform(error.to_string()))?,
            );
        }
        let value = serde_json::Value::Array(rendered);
        Ok(ClientToolOutput {
            result: ClientToolResultContent::Json {
                value: value.clone(),
            },
            media: Vec::new(),
            is_error: false,
            trace_response: value,
            usage: Vec::new(),
        })
    }
}

/// Applies opt-in privacy redaction to fetched platform messages.
///
/// Non-opted-in authors keep their message envelope but lose content, mentions,
/// attachments, and reply/reference data so the model can see that history
/// existed without seeing private user-authored material. Other privacy modes
/// either constrain the fetch before this point or pass through unchanged.
pub(crate) async fn redact_messages_for_privacy<S>(
    storage: &S,
    privacy: &PrivacyMode,
    channel: &ChannelRef,
    messages: Vec<PlatformMessage>,
) -> Result<Vec<PlatformMessage>, BotToolError>
where
    S: BotStorage,
{
    if !matches!(privacy, PrivacyMode::OptIn) {
        return Ok(messages);
    }
    // Opt-in settings are guild-scoped. Without a guild id there is no stored
    // opt-in state to consult, so the already-fetched messages pass through.
    let Some(guild_id) = channel.guild_id.as_ref() else {
        return Ok(messages);
    };
    let mut redacted = Vec::with_capacity(messages.len());
    for mut message in messages {
        let opted_in = storage
            .user_privacy(
                channel.platform.clone(),
                guild_id.as_str().to_string(),
                message.author.id.user_id.as_str().to_string(),
            )
            .await
            .map_err(|error| BotToolError::Storage(error.to_string()))?
            .unwrap_or(false);
        if !opted_in {
            // Preserve ordering and message metadata, but remove fields that
            // can carry private user content or private referenced context.
            message.content = "[redacted: user has not opted in]".to_string();
            message.mentions.clear();
            message.mention_profiles.clear();
            message.attachments.clear();
            message.reference = PlatformMessageReference::None;
        }
        redacted.push(message);
    }
    Ok(redacted)
}
