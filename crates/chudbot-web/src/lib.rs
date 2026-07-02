//! Axum web service for the trace viewer: JSON API, SSE events, media/static
//! routes, and SPA serving.

mod events;
mod server;

mod api;
mod media;
mod middleware;
mod spa;
mod static_files;

pub use api::{
    ClientToolTraceView, ConversationView, ModelInfoView, ToolTraceView, TurnView, UserMetadata,
};
pub use events::EventBus;
pub use server::{
    WebConfig, WebRuntimeTypes, WebServerError, WebState, WebStateInner, router, run_until_shutdown,
};
