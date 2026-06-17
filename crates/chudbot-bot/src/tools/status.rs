//! `post_status_message` client tool.

use super::*;

/// Tool for posting a short interim platform reply during slow work.
pub(crate) struct PostStatusTool<P, S> {
    pub(crate) platforms: P,
    pub(crate) storage: S,
    pub(crate) channel: ChannelRef,
    pub(crate) reply_to: MessageRef,
    pub(crate) conversation_id: ConversationId,
    pub(crate) turn_id: TurnId,
}

impl<P, S> PostStatusTool<P, S>
where
    P: MessagePlatformRegistry + Clone,
    S: BotStorage + Clone,
{
    pub(crate) fn spec(&self) -> ClientToolSpec {
        ClientToolSpec {
            description: "Post a short interim status reply before slow work.".to_string(),
            input_schema: ToolInputSchema::new(serde_json::json!({
                "type": "object",
                "required": ["text"],
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Short status message to send to the user."
                    }
                },
                "additionalProperties": false
            })),
        }
    }

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
        let text = call
            .input
            .get("text")
            .and_then(serde_json::Value::as_str)
            .filter(|text| !text.trim().is_empty())
            .ok_or_else(|| BotToolError::InvalidInput("`text` is required".to_string()))?;
        tracing::debug!(text_chars = text.chars().count(), "posting status message");
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
