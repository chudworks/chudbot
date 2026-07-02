use std::convert::Infallible;
use std::time::Duration;

use axum::extract::{Path, State};
use axum::http::HeaderValue;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use chudbot_api::{BotStorage, ConversationId, ConversationLookup, EventSink, LiveEvent};
use futures::StreamExt;
use tokio_stream::wrappers::BroadcastStream;
use uuid::Uuid;

use crate::api::ApiError;
use crate::server::{WebRuntimeTypes, WebState};

/// Broadcast event bus shared with the bot runtime.
#[derive(Debug, Clone)]
pub struct EventBus {
    sender: tokio::sync::broadcast::Sender<LiveEvent>,
}

impl EventBus {
    /// Creates a broadcast-backed event bus with the given per-receiver
    /// buffer capacity.
    pub fn new(capacity: usize) -> Self {
        let (sender, _receiver) = tokio::sync::broadcast::channel(capacity);
        tracing::debug!(capacity, "constructed web event bus");
        Self { sender }
    }

    /// Returns a receiver for all live events; route handlers filter events
    /// for their own scope.
    pub fn subscribe(&self) -> tokio::sync::broadcast::Receiver<LiveEvent> {
        tracing::trace!(
            receivers = self.sender.receiver_count(),
            "subscribing to event bus"
        );
        self.sender.subscribe()
    }
}

impl EventSink for EventBus {
    fn publish(&self, event: LiveEvent) {
        let event_name = event.event_name();
        match &event {
            LiveEvent::Conversation {
                conversation_id,
                kind,
            } => tracing::trace!(
                event = event_name,
                conversation = %conversation_id,
                kind = ?kind,
                receivers = self.sender.receiver_count(),
                "publishing live event"
            ),
            LiveEvent::UserProfileUpdated { user } => tracing::trace!(
                event = event_name,
                platform = %user.platform,
                guild = ?user.guild_id,
                user = %user.user_id,
                receivers = self.sender.receiver_count(),
                "publishing live event"
            ),
        }
        if self.sender.send(event).is_err() {
            tracing::trace!("live event dropped because there are no subscribers");
        }
    }
}

const SSE_KEEPALIVE: Duration = Duration::from_secs(30);

#[tracing::instrument(
    name = "web.conversation_events",
    skip_all,
    fields(conversation = %id)
)]
pub(crate) async fn conversation_events<R>(
    State(state): State<WebState<R>>,
    Path(id): Path<Uuid>,
) -> Result<Response, ApiError>
where
    R: WebRuntimeTypes,
{
    let conversation_id = ConversationId(id);
    state
        .storage
        .load_conversation(ConversationLookup::Id {
            id: conversation_id,
        })
        .await
        .map_err(|error| ApiError::Storage(error.to_string()))?
        .ok_or(ApiError::NotFound)?;

    tracing::info!("opening conversation event stream");
    let stream = BroadcastStream::new(state.events.subscribe()).filter_map(move |item| {
        let event = match item {
            Ok(event) if event.applies_to_conversation(conversation_id) => event,
            Ok(_) => return futures::future::ready(None),
            Err(tokio_stream::wrappers::errors::BroadcastStreamRecvError::Lagged(n)) => {
                tracing::warn!(
                    conversation = %conversation_id,
                    skipped = n,
                    "conversation event stream lagged"
                );
                return futures::future::ready(Some(Ok::<Event, Infallible>(
                    Event::default().event("lag").data(n.to_string()),
                )));
            }
        };
        tracing::trace!(
            conversation = %conversation_id,
            event = event.event_name(),
            "forwarding live event to SSE client"
        );
        let data = serde_json::to_string(&event).unwrap_or_else(|_| "{}".to_string());
        futures::future::ready(Some(Ok::<Event, Infallible>(
            Event::default().event(event.event_name()).data(data),
        )))
    });
    let stream = stream.take_until(state.shutdown_token().cancelled_owned());
    let mut response = Sse::new(stream)
        .keep_alive(KeepAlive::new().interval(SSE_KEEPALIVE))
        .into_response();
    response
        .headers_mut()
        .insert("x-accel-buffering", HeaderValue::from_static("no"));
    Ok(response)
}
