//! Guild/workspace profile and icon-cache updates.
//!
//! Platform guild events carry the remote icon hash and URL. The runtime stores
//! the metadata immediately, then downloads the current icon in the background
//! and records the stable media URI for future management views.

use crate::prelude::*;
use crate::*;

impl<R> BotRuntime<R>
where
    R: BotRuntimeTypes + 'static,
{
    /// Store the latest guild profile and refresh its icon in the background.
    pub(crate) async fn handle_guild_profile(
        &self,
        guild: GuildProfile,
    ) -> Result<BotAction, BotError> {
        self.storage
            .upsert_guild(guild.clone())
            .await
            .map_err(storage_error)?;
        self.spawn_guild_icon_download(guild);
        Ok(BotAction::Ignored)
    }

    /// Starts a best-effort background refresh for a guild/workspace icon.
    fn spawn_guild_icon_download(&self, guild: GuildProfile) {
        let Some(url) = guild
            .icon_url
            .as_deref()
            .filter(|url| !url.trim().is_empty())
            .map(str::to_string)
        else {
            return;
        };
        let Some(icon_hash) = guild
            .icon_hash
            .as_deref()
            .filter(|hash| !hash.trim().is_empty())
            .map(str::to_string)
        else {
            return;
        };
        let runtime = (*self).clone();
        spawn_background_task(&self.background, "guild icon download", async move {
            if let Err(error) = runtime.download_guild_icon(guild, icon_hash, url).await {
                tracing::warn!(error = %error, "guild icon download failed");
            }
        });
    }

    /// Downloads a guild icon, writes it to the media store, and saves its URI.
    async fn download_guild_icon(
        &self,
        guild: GuildProfile,
        icon_hash: String,
        url: String,
    ) -> Result<(), BotError> {
        let name = guild_icon_media_name(&guild, &icon_hash);
        let expected_uri = stored_media_uri(&MediaCategory::GuildIcon, &name);
        if self
            .storage
            .load_guild_icon(guild.platform.clone(), guild.guild_id.clone())
            .await
            .map_err(storage_error)?
            .as_ref()
            .is_some_and(|uri| uri == &expected_uri)
        {
            tracing::trace!(
                guild = %guild.guild_id,
                uri = %expected_uri,
                "guild icon already cached"
            );
            return Ok(());
        }

        if self.media_store.media_from_uri(&expected_uri).await.is_ok() {
            self.storage
                .set_guild_icon(
                    guild.platform.clone(),
                    guild.guild_id.clone(),
                    icon_hash,
                    expected_uri.clone(),
                )
                .await
                .map_err(storage_error)?;
            tracing::trace!(
                guild = %guild.guild_id,
                uri = %expected_uri,
                "guild icon media already existed"
            );
            return Ok(());
        }

        let response = self
            .download_http
            .get(&url)
            .send()
            .await
            .map_err(|error| BotError::GuildIconDownload(error.to_string()))?;
        let status = response.status();
        if !status.is_success() {
            return Err(BotError::GuildIconDownload(format!("http {status}")));
        }
        let bytes = response
            .bytes()
            .await
            .map_err(|error| BotError::GuildIconDownload(error.to_string()))?
            .to_vec();
        let media = self
            .media_store
            .create_media(CreateMedia {
                category: MediaCategory::GuildIcon,
                bytes,
                mime_type: Some("image/png".to_string()),
                name: Some(name),
                extension: Some("png".to_string()),
            })
            .await
            .map_err(|error| BotError::GuildIconDownload(error.to_string()))?;
        self.storage
            .set_guild_icon(
                guild.platform,
                guild.guild_id.clone(),
                icon_hash,
                media.uri().clone(),
            )
            .await
            .map_err(storage_error)?;
        tracing::info!(
            guild = %guild.guild_id,
            uri = %media.uri(),
            "guild icon cached"
        );
        Ok(())
    }
}

/// Builds the stable media filename used as the guild-icon cache key.
fn guild_icon_media_name(guild: &GuildProfile, icon_hash: &str) -> String {
    format!(
        "{}_{}.png",
        safe_guild_icon_name_part(guild.guild_id.as_str()),
        safe_guild_icon_name_part(icon_hash)
    )
}

/// Keeps the platform-derived filename components safe for media stores.
fn safe_guild_icon_name_part(input: &str) -> String {
    let out = input
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_'))
        .collect::<String>();
    if out.is_empty() {
        "guild".to_string()
    } else {
        out
    }
}
