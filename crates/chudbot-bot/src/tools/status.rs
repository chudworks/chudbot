//! Runtime support for the `post_status_message` client tool.
//!
//! The tool lets an agent post a real, visible interim reply during slow work.
//! It complements the platform typing indicator: typing only shows that the bot
//! is still active, while this tool records explicit progress text in the
//! conversation. Successful calls return sent-message metadata as JSON and do
//! not produce media attachments or usage records.

use super::*;

/// Tool for posting a short interim platform reply during the active turn.
///
/// Status messages are always sent to the current turn's channel, as replies to
/// the user message that started the turn. They are linked back to the same
/// conversation and turn with the `assistant_status` role so storage and the
/// trace viewer can distinguish progress updates from the final assistant
/// answer.
pub(crate) struct PostStatusTool<P, S> {
    /// Platform registry used to send the visible status message.
    pub(crate) platforms: P,
    /// Storage used to link posted platform messages into the turn trace.
    pub(crate) storage: S,
    /// Channel that receives status updates for this turn.
    pub(crate) channel: ChannelRef,
    /// User message that the status update should reply to.
    pub(crate) reply_to: MessageRef,
    /// Conversation owning the status message.
    pub(crate) conversation_id: ConversationId,
    /// Turn owning the status message.
    pub(crate) turn_id: TurnId,
}

impl<P, S> PostStatusTool<P, S>
where
    P: MessagePlatformRegistry + Clone,
    S: BotStorage + Clone,
{
    /// Build the model-visible schema for progress text.
    ///
    /// The runtime accepts one non-empty `text` field. The system prompt owns
    /// when the model should call this tool: slow, user-visible work such as
    /// generation, research, or subagent calls.
    pub(crate) fn spec(&self) -> ClientToolSpec {
        ClientToolSpec {
            description: "Post a short interim status reply before slow work.".to_string(),
            input_schema: ToolInputSchema::object([ToolInputField::required(
                "text",
                ToolInputValueSchema::string()
                    .description("Short status message to send to the user."),
            )]),
        }
    }

    /// Send one progress update and return platform message metadata.
    ///
    /// A successful output mirrors the trace response: JSON containing the main
    /// sent message, its channel, and any platform-created extra messages. It is
    /// never marked as an error and carries no media or usage entries.
    #[tracing::instrument(
        name = "tool.post_status_message",
        skip_all,
        fields(
            tool_call = %call.id,
            conversation = %self.conversation_id,
            turn = %self.turn_id,
            platform = %self.channel.platform,
            channel = %self.channel.channel_id,
            reply_to = %self.reply_to.message_id,
        )
    )]
    pub(crate) async fn call(
        &self,
        call: ClientToolCall,
    ) -> Result<ClientToolOutput, BotToolError> {
        // Empty or whitespace-only text would create noisy platform messages
        // and unhelpful trace entries, so reject it before any side effects.
        let text = call
            .input
            .get("text")
            .and_then(serde_json::Value::as_str)
            .filter(|text| !text.trim().is_empty())
            .ok_or_else(|| BotToolError::InvalidInput("`text` is required".to_string()))?;
        tracing::debug!(text_chars = text.chars().count(), "posting status message");
        // Status updates are visible replies, not typing notifications; the
        // turn runner handles typing separately for the whole agent run.
        let posted = self
            .platforms
            .send_message(SendMessage {
                channel: self.channel.clone(),
                reply_to: Some(self.reply_to.clone()),
                content: text.to_string(),
                attachments: Vec::new(),
                suppress_embeds: true,
                open_thread: None,
            })
            .await
            .map_err(|error| BotToolError::Platform(error.to_string()))?;
        tracing::info!(
            message = %posted.id.message_id,
            channel = %posted.channel.channel_id,
            "posted status message"
        );
        // Link every platform message emitted for this status update so later
        // message lookups and the trace viewer preserve progress context.
        self.storage
            .link_message(MessageLink {
                message: posted.id.clone(),
                conversation_id: self.conversation_id,
                turn_id: self.turn_id,
                role: "assistant_status".to_string(),
            })
            .await
            .map_err(|error| BotToolError::Storage(error.to_string()))?;
        for message in &posted.extra_messages {
            self.storage
                .link_message(MessageLink {
                    message: message.clone(),
                    conversation_id: self.conversation_id,
                    turn_id: self.turn_id,
                    role: "assistant_status".to_string(),
                })
                .await
                .map_err(|error| BotToolError::Storage(error.to_string()))?;
        }
        // Keep the model result and stored trace response identical; callers
        // get concrete platform ids but no generated-media delivery side effect.
        let value = serde_json::json!({
            "message": posted.id,
            "channel": posted.channel,
            "extra_messages": posted.extra_messages,
        });
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
