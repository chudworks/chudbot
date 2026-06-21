//! User avatar download and media-cache updates.
//!
//! Platform profile events carry remote avatar URLs, while the trace viewer and
//! bot runtime prefer stable `MediaUri` values. This module bridges those
//! shapes by downloading avatar images in the background, storing them as media,
//! and recording the cached URI on the user profile.

use crate::prelude::*;
use crate::*;

impl<R> BotRuntime<R>
where
    R: BotRuntimeTypes + 'static,
{
    /// Starts a best-effort background refresh for a user's avatar image.
    ///
    /// Missing or blank URLs are ignored because platform profiles may not have
    /// a custom avatar. Network, media-store, and storage errors are logged from
    /// the spawned task instead of blocking the caller that observed the user.
    pub(crate) fn spawn_avatar_download(&self, user: UserProfile) {
        // Normalize the optional platform URL before moving work to the task.
        let Some(url) = user
            .avatar_url
            .as_deref()
            .filter(|url| !url.trim().is_empty())
            .map(str::to_string)
        else {
            return;
        };
        let runtime = (*self).clone();
        // The task owns the runtime clone so user handling can continue at once.
        spawn_background_task(&self.background, "avatar download", async move {
            if let Err(error) = runtime.download_avatar(user, url).await {
                tracing::warn!(error = %error, "avatar download failed");
            }
        });
    }

    /// Downloads an avatar, writes it to the media store, and saves its user URI.
    ///
    /// The cache key is deterministic for a `(user, url)` pair. A matching
    /// stored URI means the current avatar URL has already been cached, so the
    /// function can return before doing any network work.
    pub(crate) async fn download_avatar(
        &self,
        user: UserProfile,
        url: String,
    ) -> Result<(), BotError> {
        // Step 1: derive the exact media URI storage should hold for this URL.
        let name = avatar_media_name(&user, &url);
        let expected_uri = MediaUri::new(format!("file://avatars/{name}"));
        // The local avatar media path is deterministic, so URI equality is the
        // freshness check. This avoids repeated downloads on every user event.
        if self
            .storage
            .load_user_avatar(user.id.clone())
            .await
            .map_err(storage_error)?
            .as_ref()
            .is_some_and(|uri| uri == &expected_uri)
        {
            tracing::trace!(uri = %expected_uri, "avatar already cached");
            return Ok(());
        }

        if self.media_store.media_from_uri(&expected_uri).await.is_ok() {
            self.storage
                .set_user_avatar(user.id.clone(), url, expected_uri.clone())
                .await
                .map_err(storage_error)?;
            self.publish_user(user.id);
            tracing::trace!(uri = %expected_uri, "avatar media already existed");
            return Ok(());
        }

        // Step 2: fetch the remote bytes after the storage cache check.
        let response = self
            .download_http
            .get(&url)
            .send()
            .await
            .map_err(|error| BotError::AvatarDownload(error.to_string()))?;
        let status = response.status();
        if !status.is_success() {
            return Err(BotError::AvatarDownload(format!("http {status}")));
        }
        let bytes = response
            .bytes()
            .await
            .map_err(|error| BotError::AvatarDownload(error.to_string()))?
            .to_vec();
        // Step 3: persist the bytes as avatar media using the same name used in
        // the expected URI above.
        let media = self
            .media_store
            .create_media(CreateMedia {
                category: MediaCategory::Avatar,
                bytes,
                mime_type: Some("image/png".to_string()),
                name: Some(name),
                extension: Some("png".to_string()),
            })
            .await
            .map_err(|error| BotError::AvatarDownload(error.to_string()))?;
        // Step 4: publish after storage points at the new cached media URI.
        self.storage
            .set_user_avatar(user.id.clone(), url, media.uri().clone())
            .await
            .map_err(storage_error)?;
        self.publish_user(user.id);
        tracing::info!(uri = %media.uri(), "avatar cached");
        Ok(())
    }
}

/// Builds the stable media filename used as the avatar cache key.
///
/// The user id prevents collisions between users that share the same CDN tail.
/// Discord default avatars live under `/embed/avatars/`, so their numeric tails
/// get a prefix before sanitizing to keep names recognizable.
fn avatar_media_name(user: &UserProfile, url: &str) -> String {
    let tail = url
        .split('?')
        .next()
        .and_then(|url| url.rsplit('/').next())
        .unwrap_or("avatar.png");
    let stem = tail.strip_suffix(".png").unwrap_or(tail);
    let stem = if url.contains("/embed/avatars/") {
        format!("default{stem}")
    } else {
        stem.to_string()
    };
    format!(
        "{}_{}.png",
        user.id.user_id.as_str(),
        safe_avatar_name_part(&stem)
    )
}

/// Keeps the URL-derived filename component safe for the local media store.
///
/// Only ASCII letters, numbers, `-`, and `_` survive. If the URL tail has no
/// usable characters, a generic stem keeps the final filename non-empty.
fn safe_avatar_name_part(input: &str) -> String {
    let out = input
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_'))
        .collect::<String>();
    if out.is_empty() {
        "avatar".to_string()
    } else {
        out
    }
}
