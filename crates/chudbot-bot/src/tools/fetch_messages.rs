//! `fetch_messages` client tool.
//!
//! This is the model-facing channel-history tool. Platform adapters decide what
//! history is visible to the bot; this module owns per-call platform/guild
//! scoping, defensive page-size handling, and transcript shaping.

use super::*;

/// Tool for fetching recent platform messages.
///
/// Calls are anchored to the current turn's platform and guild. Input may only
/// choose a channel id within that scope before a platform fetch runs.
pub(crate) struct FetchMessagesTool<P> {
    /// Platform registry used to fetch and render messages from the active
    /// platform implementation.
    pub(crate) platforms: P,
    /// Channel associated with the current conversation turn.
    pub(crate) default_channel: ChannelRef,
}

impl<P> FetchMessagesTool<P>
where
    P: MessagePlatformRegistry + Clone,
{
    /// Describes the model-facing tool contract.
    ///
    /// The schema advertises the allowed page size and cursor shape to model
    /// providers. Security-sensitive checks are still repeated in [`Self::call`]
    /// because providers may be lenient about JSON Schema enforcement.
    pub(crate) fn spec(&self) -> ClientToolSpec {
        ClientToolSpec {
            description: "Fetch recent messages from the current channel for context.".to_string(),
            input_schema: ToolInputSchema::object([
                ToolInputField::optional(
                    "channel_id",
                    ToolInputValueSchema::string().description(
                        "Optional platform channel id. Defaults to the current channel.",
                    ),
                ),
                ToolInputField::optional(
                    "limit",
                    ToolInputValueSchema::integer()
                        .minimum(1)
                        .maximum(100)
                        .default(20),
                ),
                ToolInputField::optional(
                    "before_message_id",
                    ToolInputValueSchema::string()
                        .description("Optional platform message id to page before."),
                ),
            ]),
        }
    }

    /// Executes the fetch and returns transcript-ready message context JSON.
    #[tracing::instrument(
        name = "tool.fetch_messages",
        skip_all,
        fields(
            tool_call = %call.id,
            default_platform = %self.default_channel.platform,
            default_channel = %self.default_channel.channel_id,
        )
    )]
    pub(crate) async fn call(
        &self,
        call: ClientToolCall,
    ) -> Result<ClientToolOutput, BotToolError> {
        let channel = requested_channel(&self.default_channel, &call.input)?;
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
        let messages = self
            .platforms
            .fetch_messages(FetchMessages {
                channel: channel.clone(),
                limit,
                before,
            })
            .await
            .map_err(|error| BotToolError::Platform(error.to_string()))?;
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
