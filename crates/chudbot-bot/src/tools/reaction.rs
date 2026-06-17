//! `add_reaction` client tool.

use super::*;

/// Tool for reacting to the current user message.
pub(crate) struct AddReactionTool<P> {
    pub(crate) platforms: P,
    pub(crate) message: MessageRef,
}

impl<P> AddReactionTool<P>
where
    P: MessagePlatformRegistry + Clone,
{
    pub(crate) fn spec(&self) -> ClientToolSpec {
        ClientToolSpec {
            description: concat!(
                "Add one standard Unicode emoji reaction to the user message that started ",
                "this turn. Use this sparingly for compact nonverbal signals: quick ",
                "acknowledgement, warmth, topic-fit emphasis, or lightweight progress that ",
                "would be noisy as text. Do not use it to replace the final answer or the ",
                "automatic completion/error reaction. Pass exactly one Unicode emoji ",
                "glyph/sequence such as 👍, 🏊, 🔎, 🎉, or ❤️; never pass words, ",
                "shortcodes like :smile:, custom emoji syntax, markdown, or multiple emoji."
            )
            .to_string(),
            input_schema: ToolInputSchema::new(serde_json::json!({
                "type": "object",
                "required": ["emoji"],
                "properties": {
                    "emoji": {
                        "type": "string",
                        "description": "Exactly one standard Unicode emoji reaction. Do not include text, Discord :shortcodes:, custom emoji markup, markdown, or multiple emoji.",
                        "minLength": 1,
                        "maxLength": 64
                    }
                },
                "additionalProperties": false
            })),
        }
    }

    #[tracing::instrument(
        name = "tool.add_reaction",
        skip_all,
        fields(
            tool_call = %call.id,
            platform = %self.message.platform,
            channel = %self.message.channel_id,
            message = %self.message.message_id,
        )
    )]
    pub(crate) async fn call(
        &self,
        call: ClientToolCall,
    ) -> Result<ClientToolOutput, BotToolError> {
        let emoji = reaction_emoji_from_tool_input(&call.input)?;
        tracing::debug!(emoji = %emoji, "adding reaction to current user message");
        self.platforms
            .add_reaction(
                self.message.clone(),
                ReactionKind::Unicode {
                    name: emoji.clone(),
                },
            )
            .await
            .map_err(|error| BotToolError::Platform(error.to_string()))?;
        tracing::info!(emoji = %emoji, "added reaction to current user message");
        let value = serde_json::json!({
            "message": self.message,
            "emoji": emoji,
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

pub(crate) fn reaction_emoji_from_tool_input(
    input: &serde_json::Value,
) -> Result<String, BotToolError> {
    let emoji = tool_required_string(input, "emoji")?;
    validate_reaction_emoji(&emoji)?;
    if is_reserved_tool_reaction(&emoji) {
        return Err(reserved_reaction_emoji());
    }
    Ok(emoji)
}

pub(crate) fn validate_reaction_emoji(emoji: &str) -> Result<(), BotToolError> {
    if is_keycap_emoji(emoji) {
        return Ok(());
    }

    let mut has_emoji_char = false;
    let mut non_component_base_count = 0usize;
    let mut regional_indicator_count = 0usize;
    let mut scalar_count = 0usize;
    let mut saw_zwj = false;
    let mut previous_can_join = false;
    let mut previous_was_zwj = false;

    for ch in emoji.chars() {
        scalar_count += 1;
        if scalar_count > MAX_REACTION_EMOJI_SCALARS
            || ch.is_control()
            || ch.is_whitespace()
            || is_text_presentation_selector(ch)
            || is_keycap_base(ch)
        {
            return Err(invalid_reaction_emoji());
        }

        if is_zwj(ch) {
            if !previous_can_join {
                return Err(invalid_reaction_emoji());
            }
            saw_zwj = true;
            previous_can_join = false;
            previous_was_zwj = true;
            continue;
        }

        if is_emoji_presentation_selector(ch) || is_tag_character(ch) {
            if !has_emoji_char || previous_was_zwj {
                return Err(invalid_reaction_emoji());
            }
            previous_was_zwj = false;
            continue;
        }

        if !ch.is_emoji_char_or_emoji_component() {
            return Err(invalid_reaction_emoji());
        }

        if ch.is_emoji_char() {
            has_emoji_char = true;
            previous_can_join = true;
            if is_regional_indicator(ch) {
                regional_indicator_count += 1;
            } else if !ch.is_emoji_component() {
                non_component_base_count += 1;
            }
        }
        previous_was_zwj = false;
    }

    if scalar_count == 0 || !has_emoji_char || previous_was_zwj {
        return Err(invalid_reaction_emoji());
    }

    if regional_indicator_count > 0 {
        if regional_indicator_count == 2 && non_component_base_count == 0 && !saw_zwj {
            return Ok(());
        }
        return Err(invalid_reaction_emoji());
    }

    if non_component_base_count == 0 || (non_component_base_count > 1 && !saw_zwj) {
        return Err(invalid_reaction_emoji());
    }

    Ok(())
}

pub(crate) fn invalid_reaction_emoji() -> BotToolError {
    BotToolError::InvalidInput(
        "`emoji` must be exactly one standard Unicode emoji; text, shortcodes, custom emoji, and multiple emoji are not allowed"
            .to_string(),
    )
}

pub(crate) fn reserved_reaction_emoji() -> BotToolError {
    BotToolError::InvalidInput(
        "`emoji` is reserved for Chudbot system status/control reactions".to_string(),
    )
}

pub(crate) fn is_reserved_tool_reaction(emoji: &str) -> bool {
    RESERVED_TOOL_REACTIONS.contains(&emoji)
}

pub(crate) fn is_keycap_emoji(emoji: &str) -> bool {
    let mut chars = emoji.chars();
    let Some(base) = chars.next() else {
        return false;
    };
    if !is_keycap_base(base) {
        return false;
    }
    match chars.next() {
        Some('\u{20E3}') => chars.next().is_none(),
        Some(ch) if is_emoji_presentation_selector(ch) => {
            matches!(chars.next(), Some('\u{20E3}')) && chars.next().is_none()
        }
        _ => false,
    }
}

pub(crate) fn is_keycap_base(ch: char) -> bool {
    ch.is_ascii_digit() || matches!(ch, '#' | '*')
}
