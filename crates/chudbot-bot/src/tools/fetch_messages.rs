//! `fetch_messages` client tool and privacy redaction.

use super::*;

/// Tool for fetching recent platform messages subject to runtime privacy mode.
pub(crate) struct FetchMessagesTool<P, S> {
    pub(crate) platforms: P,
    pub(crate) storage: S,
    pub(crate) default_channel: ChannelRef,
    pub(crate) privacy: PrivacyMode,
}

impl<P, S> FetchMessagesTool<P, S>
where
    P: MessagePlatformRegistry + Clone,
    S: BotStorage + Clone,
{
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
        let limit = call
            .input
            .get("limit")
            .and_then(serde_json::Value::as_u64)
            .and_then(|value| u16::try_from(value).ok())
            .unwrap_or(20)
            .clamp(1, 100);
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
