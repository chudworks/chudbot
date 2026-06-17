//! `usage_report` client tool.

use super::*;

/// Tool for querying stored usage costs for the current platform scope.
pub(crate) struct UsageReportTool<S> {
    pub(crate) storage: S,
    pub(crate) channel: ChannelRef,
}

impl<S> UsageReportTool<S>
where
    S: BotStorage + Clone,
{
    pub(crate) fn spec(&self) -> ClientToolSpec {
        ClientToolSpec {
            description: concat!(
                "Report this bot's stored model/tool/media usage and its cost in USD. ",
                "Defaults to the current server's lifetime total; group rows per server, ",
                "channel, user, agent, provider, model, or usage kind to answer questions ",
                "like \"how much has this channel or user cost?\". Background memory-job ",
                "usage is included and grouped under the `memory` pseudo-channel."
            )
            .to_string(),
            input_schema: ToolInputSchema::new(serde_json::json!({
                "type": "object",
                "properties": {
                    "group_by": {
                        "type": "string",
                        "enum": ["total", "guild", "channel", "user", "agent", "provider", "model", "kind"],
                        "default": "total",
                        "description": "Aggregation dimension for the report rows. `guild` buckets guild-less usage under `direct`; `kind` splits by usage subject such as `model_step` or `image_generation`."
                    },
                    "scope": {
                        "type": "string",
                        "enum": ["guild", "channel", "global"],
                        "default": "guild",
                        "description": "guild = the current server (or this direct-message channel outside a server), channel = the current channel or thread only, global = every server and DM this bot serves."
                    },
                    "days": {
                        "type": "number",
                        "exclusiveMinimum": 0,
                        "description": "Look-back window in days; fractions allowed. Omit for lifetime totals."
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 50,
                        "default": 10,
                        "description": "Maximum number of report rows, costliest first."
                    }
                },
                "additionalProperties": false
            })),
        }
    }

    #[tracing::instrument(
        name = "tool.usage_report",
        skip_all,
        fields(
            tool_call = %call.id,
            platform = %self.channel.platform,
            channel = %self.channel.channel_id,
        )
    )]
    pub(crate) async fn call(
        &self,
        call: ClientToolCall,
    ) -> Result<ClientToolOutput, BotToolError> {
        let request = usage_report_request(&call.input, &self.channel, OffsetDateTime::now_utc())?;
        let mut groups = self
            .storage
            .usage_cost_report(request.query.clone())
            .await
            .map_err(|error| BotToolError::Storage(error.to_string()))?;
        let truncated = groups.len() > request.limit as usize;
        groups.truncate(request.limit as usize);
        let total = if request.query.group_by == UsageCostGrouping::Total {
            groups.first().cloned()
        } else {
            self.storage
                .usage_cost_report(UsageCostQuery {
                    group_by: UsageCostGrouping::Total,
                    limit: 1,
                    ..request.query.clone()
                })
                .await
                .map_err(|error| BotToolError::Storage(error.to_string()))?
                .into_iter()
                .next()
        };
        tracing::info!(
            group_by = ?request.query.group_by,
            groups = groups.len(),
            truncated,
            "built usage cost report"
        );
        let value = usage_report_value(&request, total.as_ref(), &groups, truncated);
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

/// Parsed usage report query plus presentation limits.
#[derive(Debug, Clone)]
pub(crate) struct UsageReportRequest {
    pub(crate) query: UsageCostQuery,
    pub(crate) days: Option<f64>,
    pub(crate) limit: u32,
}

pub(crate) fn usage_report_request(
    input: &serde_json::Value,
    channel: &ChannelRef,
    now: OffsetDateTime,
) -> Result<UsageReportRequest, BotToolError> {
    let group_by = match tool_optional_string(input, "group_by")?
        .as_deref()
        .unwrap_or("total")
    {
        "total" => UsageCostGrouping::Total,
        "guild" => UsageCostGrouping::Guild,
        "channel" => UsageCostGrouping::Channel,
        "user" => UsageCostGrouping::User,
        "agent" => UsageCostGrouping::Agent,
        "provider" => UsageCostGrouping::Provider,
        "model" => UsageCostGrouping::Model,
        "kind" => UsageCostGrouping::Kind,
        other => {
            return Err(BotToolError::InvalidInput(format!(
                "unknown `group_by` value `{other}`"
            )));
        }
    };
    let scope = match tool_optional_string(input, "scope")?
        .as_deref()
        .unwrap_or("guild")
    {
        "guild" => match &channel.guild_id {
            Some(guild_id) => UsageCostScope::Guild {
                guild_id: guild_id.as_str().to_string(),
            },
            None => current_channel_scope(channel),
        },
        "channel" => current_channel_scope(channel),
        "global" => UsageCostScope::All,
        other => {
            return Err(BotToolError::InvalidInput(format!(
                "unknown `scope` value `{other}`"
            )));
        }
    };
    let days = match input.get("days") {
        None | Some(serde_json::Value::Null) => None,
        Some(value) => {
            let days = value
                .as_f64()
                .filter(|days| days.is_finite() && *days > 0.0);
            Some(days.ok_or_else(|| {
                BotToolError::InvalidInput("`days` must be a positive number".to_string())
            })?)
        }
    };
    let since = days.map(|days| now - time::Duration::seconds_f64(days * 86_400.0));
    let limit = match input.get("limit") {
        None | Some(serde_json::Value::Null) => 10,
        Some(value) => {
            let limit = value
                .as_u64()
                .filter(|limit| (1..=50).contains(limit))
                .ok_or_else(|| {
                    BotToolError::InvalidInput(
                        "`limit` must be an integer between 1 and 50".to_string(),
                    )
                })?;
            u32::try_from(limit).expect("limit fits u32 after range check")
        }
    };
    Ok(UsageReportRequest {
        query: UsageCostQuery {
            platform: channel.platform.clone(),
            scope,
            since,
            group_by,
            // Fetch one extra row to detect truncation.
            limit: limit + 1,
        },
        days,
        limit,
    })
}

pub(crate) fn current_channel_scope(channel: &ChannelRef) -> UsageCostScope {
    UsageCostScope::Channel {
        guild_id: channel
            .guild_id
            .as_ref()
            .map(|guild_id| guild_id.as_str().to_string()),
        channel_id: channel.channel_id.as_str().to_string(),
    }
}

pub(crate) fn usage_report_value(
    request: &UsageReportRequest,
    total: Option<&UsageCostRow>,
    groups: &[UsageCostRow],
    truncated: bool,
) -> serde_json::Value {
    let mut value = serde_json::json!({
        "group_by": request.query.group_by,
        "scope": request.query.scope,
        "window_days": request.days,
        "since": request.query.since.and_then(|since| {
            since.format(&time::format_description::well_known::Rfc3339).ok()
        }),
        "total": total.map(|row| usage_cost_row_value(request.query.group_by, row)),
    });
    if request.query.group_by != UsageCostGrouping::Total {
        value["groups"] = groups
            .iter()
            .map(|row| usage_cost_row_value(request.query.group_by, row))
            .collect();
        value["truncated"] = serde_json::Value::Bool(truncated);
    }
    value
}

pub(crate) fn usage_cost_row_value(
    group_by: UsageCostGrouping,
    row: &UsageCostRow,
) -> serde_json::Value {
    let mut value = serde_json::to_value(row).unwrap_or_default();
    if let Some(mention) = usage_cost_row_mention(group_by, row) {
        value["mention"] = serde_json::Value::String(mention);
    }
    value
}

/// Platform mention markup the model can paste into a reply so ids render as
/// user/channel names.
pub(crate) fn usage_cost_row_mention(
    group_by: UsageCostGrouping,
    row: &UsageCostRow,
) -> Option<String> {
    let key = row.key.as_deref()?;
    match group_by {
        UsageCostGrouping::User => Some(format!("<@{key}>")),
        UsageCostGrouping::Channel => {
            channel_id_from_channel_key(key).map(|channel_id| format!("<#{channel_id}>"))
        }
        _ => None,
    }
}

/// Extract the platform channel id from a storage channel key
/// (`guild:<g>:channel:<c>` or `channel:<c>`).
pub(crate) fn channel_id_from_channel_key(key: &str) -> Option<&str> {
    if let Some(rest) = key.strip_prefix("guild:") {
        return rest
            .split_once(":channel:")
            .map(|(_, channel_id)| channel_id);
    }
    key.strip_prefix("channel:")
}
