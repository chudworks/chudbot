use std::collections::BTreeMap;

use axum::Json;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use chudbot_api::{
    BotStorage, ClientToolCall, ClientToolResult, ClientToolResultContent, ContextItem,
    Conversation, ConversationId, ConversationLookup, ConversationSnapshot, GroundingMetadata,
    LlmProviderRegistry, MediaUri, ModelId, ModelInfo, ModelInfoRequest, ProviderName,
    ServerToolUse, ToolTrace, Turn, TurnAsset, TurnReasoning, TurnSnapshot, UsageRecord,
    UsageSubject, UserRef,
};
use serde::Serialize;
use thiserror::Error;
use uuid::Uuid;

use crate::server::{WebRuntimeTypes, WebState};

/// Conversation read model served to the React viewer.
#[derive(Debug, Clone, Serialize)]
pub struct ConversationView {
    pub conversation: Conversation,
    /// Ordered turn snapshots, shaped for the viewer.
    pub turns: Vec<TurnView>,
    /// User metadata keyed by `platform:guild:user` string.
    pub users: BTreeMap<String, UserMetadata>,
    /// Provider-reported model metadata keyed by provider/model pairs.
    pub model_info: Vec<ModelInfoView>,
}

/// User metadata for frontend rendering.
#[derive(Debug, Clone, Serialize)]
pub struct UserMetadata {
    /// Stable platform user reference.
    pub id: UserRef,
    /// Last platform username seen.
    pub username: String,
    /// Best display name seen by the bot.
    pub display_name: Option<String>,
    /// Resolved label the UI can render directly.
    pub label: String,
    /// Platform avatar URL, usually remote/CDN-backed.
    pub avatar_url: Option<String>,
    /// Cached local avatar media URI, when available.
    pub avatar_media_uri: Option<MediaUri>,
    /// Whether the platform marked this user as a bot.
    pub is_bot: bool,
}

/// Viewer-facing provider model metadata.
#[derive(Debug, Clone, Serialize)]
pub struct ModelInfoView {
    /// Provider registry key.
    pub provider: ProviderName,
    /// Model id used to request metadata.
    pub requested_model: ModelId,
    /// Provider-reported model id.
    pub model: ModelId,
    /// Maximum input/context tokens accepted by the model.
    pub context_window_tokens: Option<u64>,
    /// Maximum output tokens the model can produce, when reported separately.
    pub max_output_tokens: Option<u64>,
}

impl ModelInfoView {
    fn new(provider: ProviderName, requested_model: ModelId, info: ModelInfo) -> Self {
        Self {
            provider,
            requested_model,
            model: info.id,
            context_window_tokens: info.context_window_tokens,
            max_output_tokens: info.max_output_tokens,
        }
    }
}

/// One turn plus viewer-safe trace data.
#[derive(Debug, Clone, Serialize)]
pub struct TurnView {
    pub turn: Turn,
    /// System/developer instructions used for this attempt/turn.
    pub system_instructions: Option<String>,
    /// Novel context items captured for this turn.
    pub context: Vec<ContextItem>,
    /// Tool/server/grounding trace events.
    pub tool_trace: Vec<ToolTraceView>,
    /// Assets that should be replayed with this turn.
    pub replay_assets: Vec<TurnAsset>,
    /// Usage/cost accumulated by this turn.
    pub usage: Vec<UsageRecord>,
    /// Viewer-safe reasoning summaries and token counts.
    pub reasoning: TurnReasoning,
}

impl From<TurnSnapshot> for TurnView {
    fn from(snapshot: TurnSnapshot) -> Self {
        let reasoning =
            TurnReasoning::from_model_steps_and_usage(&snapshot.model_steps, &snapshot.usage);
        Self {
            turn: snapshot.turn,
            system_instructions: snapshot.system_instructions,
            context: snapshot.context,
            tool_trace: snapshot
                .tool_trace
                .into_iter()
                .map(ToolTraceView::from)
                .collect(),
            replay_assets: snapshot.replay_assets,
            usage: snapshot.usage,
            reasoning,
        }
    }
}

/// Viewer-facing tool trace event.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ToolTraceView {
    /// Client-side tool call/result.
    Client { trace: ClientToolTraceView },
    /// Provider-side tool use, with no client-furnished result.
    Server { tool: ServerToolUse },
    /// Provider grounding/citation metadata.
    Grounding { metadata: GroundingMetadata },
}

impl From<ToolTrace> for ToolTraceView {
    fn from(trace: ToolTrace) -> Self {
        match trace {
            ToolTrace::Client { trace } => Self::Client {
                trace: ClientToolTraceView::from(trace),
            },
            ToolTrace::Server { tool } => Self::Server { tool },
            ToolTrace::Grounding { metadata } => Self::Grounding { metadata },
        }
    }
}

/// Viewer-facing client-side tool trace.
#[derive(Debug, Clone, Serialize)]
pub struct ClientToolTraceView {
    /// Tool call requested by the model.
    pub call: ClientToolCall,
    /// Tool result furnished back to the model.
    pub result: ClientToolResult,
    /// Extra trace/debug payload, omitted when it duplicates the result.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub trace_payload: Option<serde_json::Value>,
    /// Usage/cost incurred by this client tool.
    pub usage: Vec<UsageRecord>,
}

impl From<chudbot_api::ClientToolTrace> for ClientToolTraceView {
    fn from(trace: chudbot_api::ClientToolTrace) -> Self {
        let trace_payload = if trace_response_matches_result(&trace.trace_response, &trace.result) {
            None
        } else {
            Some(trace.trace_response)
        };
        Self {
            call: trace.call,
            result: trace.result,
            trace_payload,
            usage: trace.usage,
        }
    }
}

#[tracing::instrument(name = "web.get_config", skip_all)]
pub(crate) async fn get_config<R>(State(state): State<WebState<R>>) -> Json<serde_json::Value>
where
    R: WebRuntimeTypes,
{
    tracing::debug!("serving web config");
    Json(serde_json::json!({
        "title_prefix": state.config.title_prefix,
        "version": state.config.version,
    }))
}

#[tracing::instrument(
    name = "web.get_conversation",
    skip_all,
    fields(conversation = %id)
)]
pub(crate) async fn get_conversation<R>(
    State(state): State<WebState<R>>,
    Path(id): Path<Uuid>,
) -> Result<Json<ConversationView>, ApiError>
where
    R: WebRuntimeTypes,
{
    let snapshot = state
        .storage
        .load_conversation(ConversationLookup::Id {
            id: ConversationId(id),
        })
        .await
        .map_err(|error| ApiError::Storage(error.to_string()))?
        .ok_or(ApiError::NotFound)?;
    tracing::debug!(
        turns = snapshot.turns.len(),
        stopped = snapshot.conversation.stopped_at.is_some(),
        "loaded conversation snapshot"
    );
    let users = user_metadata(&state.storage, &snapshot).await?;
    let model_info = model_info_for_snapshot(&state.llms, &snapshot).await;
    let turns = snapshot.turns.into_iter().map(TurnView::from).collect();
    Ok(Json(ConversationView {
        conversation: snapshot.conversation,
        turns,
        users,
        model_info,
    }))
}

async fn model_info_for_snapshot<L>(llms: &L, snapshot: &ConversationSnapshot) -> Vec<ModelInfoView>
where
    L: LlmProviderRegistry,
{
    let mut out = Vec::new();
    let Some((provider, model)) = latest_context_model_target(snapshot) else {
        return out;
    };
    if !llms.contains_provider(&provider) {
        tracing::debug!(
            provider = %provider,
            model = %model,
            "skipping model metadata for provider unavailable in this runtime"
        );
        return out;
    }

    {
        let request = ModelInfoRequest {
            model: model.clone(),
            provider_options: None,
        };
        match llms.fetch_model_info(&provider, request).await {
            Ok(Some(info)) => out.push(ModelInfoView::new(provider, model, info)),
            Ok(None) => tracing::debug!(
                provider = %provider,
                model = %model,
                "model metadata unavailable"
            ),
            Err(error) => tracing::warn!(
                provider = %provider,
                model = %model,
                error = %error,
                "failed to fetch model metadata"
            ),
        }
    }
    out
}

fn latest_context_model_target(snapshot: &ConversationSnapshot) -> Option<(ProviderName, ModelId)> {
    for turn in snapshot.turns.iter().rev() {
        for usage in turn.usage.iter().rev() {
            if !matches!(&usage.subject, UsageSubject::ModelStep) || usage.input_tokens.is_none() {
                continue;
            }
            if let Some(model) = usage.model.clone().or_else(|| turn.turn.model.clone()) {
                return Some((usage.provider.clone(), model));
            }
        }
    }
    None
}

async fn user_metadata<S>(
    storage: &S,
    snapshot: &chudbot_api::ConversationSnapshot,
) -> Result<std::collections::BTreeMap<String, UserMetadata>, ApiError>
where
    S: BotStorage,
{
    let mut users = std::collections::BTreeMap::<String, UserMetadata>::new();
    insert_user_fallback(&mut users, snapshot.conversation.created_by.clone(), None);
    if let Some(user) = snapshot.conversation.stopped_by.clone() {
        insert_user_fallback(&mut users, user, None);
    }
    for turn in &snapshot.turns {
        insert_user_fallback(
            &mut users,
            turn.turn.user.clone(),
            Some(turn.turn.user_display_name.clone()),
        );
    }

    let refs = users
        .values()
        .map(|user| user.id.clone())
        .collect::<Vec<_>>();
    let stored = storage
        .load_user_profiles(refs)
        .await
        .map_err(|error| ApiError::Storage(error.to_string()))?;
    for stored in stored {
        let key = user_key(&stored.profile.id);
        users.insert(
            key,
            UserMetadata {
                label: stored
                    .profile
                    .display_name
                    .clone()
                    .unwrap_or_else(|| stored.profile.username.clone()),
                username: stored.profile.username,
                display_name: stored.profile.display_name,
                avatar_url: stored.profile.avatar_url,
                avatar_media_uri: stored.avatar,
                is_bot: stored.profile.is_bot,
                id: stored.profile.id,
            },
        );
    }
    Ok(users)
}

fn insert_user_fallback(
    users: &mut std::collections::BTreeMap<String, UserMetadata>,
    user: UserRef,
    label: Option<String>,
) {
    let key = user_key(&user);
    users.entry(key).or_insert_with(|| {
        let label = label.unwrap_or_else(|| user.user_id.as_str().to_string());
        UserMetadata {
            id: user,
            username: label.clone(),
            display_name: Some(label.clone()),
            label,
            avatar_url: None,
            avatar_media_uri: None,
            is_bot: false,
        }
    });
}

fn user_key(user: &UserRef) -> String {
    format!(
        "{}:{}:{}",
        user.platform.as_str(),
        user.guild_id
            .as_ref()
            .map(|id| id.as_str())
            .unwrap_or("global"),
        user.user_id.as_str()
    )
}

fn trace_response_matches_result(
    trace_response: &serde_json::Value,
    result: &ClientToolResult,
) -> bool {
    if let Ok(content) = serde_json::to_value(&result.content)
        && trace_response == &content
    {
        return true;
    }

    match &result.content {
        ClientToolResultContent::Json { value } => trace_response == value,
        ClientToolResultContent::Text { text } => {
            trace_response.as_str() == Some(text.as_str())
                || trace_response
                    .as_object()
                    .filter(|object| object.len() == 1)
                    .and_then(|object| object.get("text"))
                    .and_then(serde_json::Value::as_str)
                    == Some(text.as_str())
        }
    }
}

#[derive(Debug, Error)]
pub(crate) enum ApiError {
    #[error("conversation not found")]
    NotFound,
    #[error("storage error: {0}")]
    Storage(String),
    #[error("media error: {0}")]
    Media(String),
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let status = match self {
            Self::NotFound => StatusCode::NOT_FOUND,
            Self::Storage(_) => StatusCode::INTERNAL_SERVER_ERROR,
            Self::Media(_) => StatusCode::NOT_FOUND,
        };
        match status {
            StatusCode::INTERNAL_SERVER_ERROR => {
                tracing::error!(error = %self, status = status.as_u16(), "api error")
            }
            _ => tracing::warn!(error = %self, status = status.as_u16(), "api error"),
        }
        let body = serde_json::json!({ "error": self.to_string() });
        (status, Json(body)).into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chudbot_api::{ClientToolTrace, ToolName, ToolUseId};
    use serde_json::json;

    fn client_trace_view(
        content: ClientToolResultContent,
        trace_response: serde_json::Value,
    ) -> ClientToolTraceView {
        ClientToolTraceView::from(ClientToolTrace {
            call: ClientToolCall {
                id: ToolUseId::from("call-1"),
                name: ToolName::from("test_tool"),
                input: json!({ "prompt": "draw this" }),
            },
            result: ClientToolResult {
                tool_use_id: ToolUseId::from("call-1"),
                content,
                is_error: false,
            },
            trace_response,
            usage: Vec::new(),
        })
    }

    #[test]
    fn viewer_trace_omits_duplicate_json_trace_payload() {
        let view = client_trace_view(
            ClientToolResultContent::Json {
                value: json!({ "ok": true }),
            },
            json!({ "ok": true }),
        );

        assert!(view.trace_payload.is_none());
        let value = serde_json::to_value(view).expect("serialize trace view");
        assert!(value.get("trace_payload").is_none());
    }

    #[test]
    fn viewer_trace_keeps_distinct_trace_payload() {
        let trace_payload = json!({
            "uri": "media://images/generated.png",
            "public_url": "https://media.example/generated.png"
        });
        let view = client_trace_view(
            ClientToolResultContent::Json {
                value: json!({ "uri": "media://images/generated.png" }),
            },
            trace_payload.clone(),
        );

        assert_eq!(view.trace_payload, Some(trace_payload));
    }
}
