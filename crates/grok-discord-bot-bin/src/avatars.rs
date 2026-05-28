//! Background avatar fetcher.
//!
//! When the bot sees a `MessageCreate` from a user whose
//! `discord_users.avatar_hash` is unknown or has changed, the message
//! handler in [`crate::bot`] calls [`spawn_fetch`]. That schedules a
//! task on `AppState.tracker` that:
//!
//!   1. Re-reads the user row to pick up the latest hash.
//!   2. Downloads the avatar from Discord's CDN (or, for users with no
//!      custom avatar, downloads the appropriate "default" avatar
//!      bucket based on the snowflake-derived index).
//!   3. Saves the bytes to `<avatars_dir>/<user_id>_<hash>.png` and
//!      updates the row via [`Db::mark_avatar_fetched`].
//!   4. Publishes [`EventKind::UserAvatarUpdated`] so any open viewer
//!      refreshes the avatar live.
//!
//! Failures are logged and silently dropped — a missing avatar isn't
//! fatal; the frontend can fall back to initials or the default
//! Discord placeholder rendered client-side.

use std::sync::Arc;

use crate::app::{AppState, EventKind};

/// Schedule a background avatar fetch for `user_id`. Idempotent: the
/// task itself re-reads the user row at the top, so the caller doesn't
/// need to know whether a fetch is already in flight. Cancellation-
/// aware: the task exits early if `AppState.cancel` fires.
pub fn spawn_fetch(app: Arc<AppState>, user_id: i64) {
    let tracker = app.tracker.clone();
    tracker.spawn(async move {
        if let Err(err) = fetch(&app, user_id).await {
            tracing::warn!(user_id, error = %err, "avatar fetch failed");
        }
    });
}

#[derive(Debug, thiserror::Error)]
enum AvatarError {
    #[error(transparent)]
    Db(#[from] grok_discord_bot_core::DbError),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error("user not found in discord_users table")]
    UserMissing,
    #[error("http {status}: {body}")]
    Http { status: u16, body: String },
    #[error("transport: {0}")]
    Transport(String),
}

#[tracing::instrument(name = "avatar_fetch", skip_all, fields(user_id = user_id))]
async fn fetch(app: &AppState, user_id: i64) -> Result<(), AvatarError> {
    let user = app
        .db
        .get_discord_user(user_id)
        .await?
        .ok_or(AvatarError::UserMissing)?;

    // Build the CDN URL. For users with a custom avatar, the hash is
    // baked into the URL; for users without one, Discord's "default
    // avatar" endpoint uses an index derived from the snowflake.
    let (url, filename_hash) = match &user.avatar_hash {
        Some(hash) => (
            format!("https://cdn.discordapp.com/avatars/{user_id}/{hash}.png?size=128"),
            hash.clone(),
        ),
        None => {
            // New username system: (user_id >> 22) % 6 picks one of the
            // 6 default avatars. Legacy users (discriminator-based) are
            // rare enough now that the unified bucket is fine.
            let bucket = ((user_id as u64) >> 22) % 6;
            (
                format!("https://cdn.discordapp.com/embed/avatars/{bucket}.png"),
                format!("default{bucket}"),
            )
        }
    };

    tokio::select! {
        biased;
        _ = app.cancel.cancelled() => {
            tracing::debug!("avatar fetch cancelled before request");
            return Ok(());
        }
        result = download_and_save(app, user_id, &url, &filename_hash) => result?,
    }

    app.publish(uuid::Uuid::nil(), EventKind::UserAvatarUpdated { user_id });
    Ok(())
}

/// Download the bytes, write them to disk, and stamp the user row.
/// The avatar filename is deterministic — `<user_id>_<hash>.png` — so
/// a hash change naturally supersedes the previous file. The DB row
/// only ever holds the *current* path, so the previous file is left as
/// a stale orphan until the (yet-to-be-built) GC step removes it.
async fn download_and_save(
    app: &AppState,
    user_id: i64,
    url: &str,
    filename_hash: &str,
) -> Result<(), AvatarError> {
    let resp = app
        .download_http
        .get(url)
        .send()
        .await
        .map_err(|e| AvatarError::Transport(e.to_string()))?;
    let status = resp.status();
    if !status.is_success() {
        let body = resp.text().await.unwrap_or_default();
        let mut snippet = body;
        if snippet.len() > 200 {
            snippet.truncate(200);
        }
        return Err(AvatarError::Http {
            status: status.as_u16(),
            body: snippet,
        });
    }
    let bytes = resp
        .bytes()
        .await
        .map_err(|e| AvatarError::Transport(e.to_string()))?;

    tokio::fs::create_dir_all(&app.storage.avatars_dir).await?;
    let filename = format!("{user_id}_{filename_hash}.png");
    let path = app.storage.avatars_dir.join(&filename);
    tokio::fs::write(&path, &bytes).await?;
    app.db.mark_avatar_fetched(user_id, &filename).await?;
    tracing::info!(
        bytes = bytes.len(),
        path = %path.display(),
        "avatar cached"
    );
    Ok(())
}
