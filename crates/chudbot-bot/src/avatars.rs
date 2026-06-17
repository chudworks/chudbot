//! User avatar download and media-cache updates.

use crate::prelude::*;
use crate::*;

impl<R> BotRuntime<R>
where
    R: BotRuntimeTypes + 'static,
{
    pub(crate) fn spawn_avatar_download(&self, user: UserProfile) {
        let Some(url) = user
            .avatar_url
            .as_deref()
            .filter(|url| !url.trim().is_empty())
            .map(str::to_string)
        else {
            return;
        };
        let runtime = (*self).clone();
        spawn_background_task(&self.background, "avatar download", async move {
            if let Err(error) = runtime.download_avatar(user, url).await {
                tracing::warn!(error = %error, "avatar download failed");
            }
        });
    }

    pub(crate) async fn download_avatar(
        &self,
        user: UserProfile,
        url: String,
    ) -> Result<(), BotError> {
        let name = avatar_media_name(&user, &url);
        let expected_uri = MediaUri::new(format!("file://avatars/{name}"));
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

        let response = reqwest::Client::new()
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
        self.storage
            .set_user_avatar(user.id.clone(), media.uri().clone())
            .await
            .map_err(storage_error)?;
        self.publish_user(user.id);
        tracing::info!(uri = %media.uri(), "avatar cached");
        Ok(())
    }
}

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
