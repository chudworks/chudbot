//! Direct-message login-link delivery for Vibe sites.

use super::*;
use chudbot_api::{NewVibeLoginLink, VibeLoginLink, VibeMembership};
use chudbot_vibe::{
    LOGIN_LINK_TTL_MINUTES, VibeAccess, VibeConfig, VibeOperation, direct_login_url, random_token,
    token_hash,
};
use time::Duration as TimeDuration;

/// Narrow persistence boundary used by the login-link tool.
pub(crate) trait VibeLoginLinkStorage: Clone + Send + Sync {
    fn create_login_link(
        &self,
        link: NewVibeLoginLink,
    ) -> impl Future<Output = Result<(), String>> + Send;

    fn consume_login_link(
        &self,
        token_hash: &[u8],
        now: OffsetDateTime,
    ) -> impl Future<Output = Result<Option<VibeLoginLink>, String>> + Send;
}

impl<T> VibeLoginLinkStorage for T
where
    T: VibeStorage + Clone,
{
    async fn create_login_link(&self, link: NewVibeLoginLink) -> Result<(), String> {
        VibeStorage::create_login_link(self, link)
            .await
            .map_err(|error| error.to_string())
    }

    async fn consume_login_link(
        &self,
        token_hash: &[u8],
        now: OffsetDateTime,
    ) -> Result<Option<VibeLoginLink>, String> {
        VibeStorage::consume_login_link(self, token_hash, now)
            .await
            .map_err(|error| error.to_string())
    }
}

/// Narrow platform boundary used to validate membership and deliver a DM.
pub(crate) trait VibeLoginLinkPlatform: Clone + Send + Sync {
    fn lookup_login_link_membership(
        &self,
        platform: &PlatformName,
        guild_id: &ExternalId,
        user_id: &ExternalId,
    ) -> impl Future<Output = Result<Option<VibeMembership>, String>> + Send;

    fn deliver_login_link(
        &self,
        recipient: UserRef,
        content: String,
    ) -> impl Future<Output = Result<(), String>> + Send;
}

impl<T> VibeLoginLinkPlatform for T
where
    T: MessagePlatformRegistry + VibeIdentityProvider + Clone,
{
    async fn lookup_login_link_membership(
        &self,
        platform: &PlatformName,
        guild_id: &ExternalId,
        user_id: &ExternalId,
    ) -> Result<Option<VibeMembership>, String> {
        VibeIdentityProvider::guild_membership(self, platform, guild_id, user_id)
            .await
            .map_err(|error| error.to_string())
    }

    async fn deliver_login_link(&self, recipient: UserRef, content: String) -> Result<(), String> {
        MessagePlatformRegistry::send_direct_message(self, recipient, content)
            .await
            .map(|_| ())
            .map_err(|error| error.to_string())
    }
}

/// Model tool that issues a target-bound link without exposing it to the requester.
pub(crate) struct VibeLoginLinkTool<P, S> {
    pub(crate) platforms: P,
    pub(crate) storage: S,
    pub(crate) context: RuntimeToolContext,
    pub(crate) config: VibeConfig,
}

impl<P, S> VibeLoginLinkTool<P, S>
where
    P: VibeLoginLinkPlatform,
    S: VibeLoginLinkStorage,
{
    pub(crate) fn spec(&self) -> ClientToolSpec {
        ClientToolSpec {
            description: "Send a single-use Vibe login, sign-in, or auth link to the current user or another current server member. The secret link is delivered only to the target user's DM and never returned in the tool result. Use a mentioned user's numeric id from trusted message context for userId; omit it to send to the requester.".to_string(),
            input_schema: ToolInputSchema::object([ToolInputField::optional(
                "userId",
                ToolInputValueSchema::string().description(
                    "Target platform user id copied from trusted message context. Omit for the current requester.",
                ),
            )]),
        }
    }

    #[tracing::instrument(
        name = "tool.send_vibe_login_link",
        skip_all,
        fields(
            tool_call = %call.id,
            platform = %self.context.turn_user.platform,
            requester = %self.context.turn_user.user_id,
        )
    )]
    pub(crate) async fn call(
        &self,
        call: ClientToolCall,
    ) -> Result<ClientToolOutput, BotToolError> {
        let guild_id = self
            .context
            .default_channel
            .guild_id
            .clone()
            .ok_or_else(|| {
                BotToolError::InvalidInput(
                    "Vibe login links can only be requested from a server".to_string(),
                )
            })?;
        let actor = chudbot_api::VibeActor {
            platform: self.context.turn_user.platform.clone(),
            guild_id: Some(guild_id.clone()),
            user_id: self.context.turn_user.user_id.clone(),
            conversation_id: self.context.conversation_id,
            turn_id: self.context.turn_id,
            is_admin: false,
        };
        VibeAccess::new(self.config.enabled, self.config.access.clone())
            .check_rollout(&actor, VibeOperation::View)
            .map_err(|error| BotToolError::InvalidInput(error.to_string()))?;

        let target_user_id = tool_optional_string(&call.input, "userId")?
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty())
            .unwrap_or_else(|| self.context.turn_user.user_id.as_str().to_string());
        if target_user_id.len() > 64 || !target_user_id.bytes().all(|byte| byte.is_ascii_digit()) {
            return Err(BotToolError::InvalidInput(
                "`userId` must be a numeric platform user id from message context".to_string(),
            ));
        }
        let target_user_id = ExternalId::new(target_user_id);
        let membership = self
            .platforms
            .lookup_login_link_membership(&actor.platform, &guild_id, &target_user_id)
            .await
            .map_err(BotToolError::Platform)?
            .ok_or_else(|| {
                BotToolError::InvalidInput(
                    "the target user is not a current member of this server".to_string(),
                )
            })?;

        let link_token = random_token();
        let now = OffsetDateTime::now_utc();
        self.storage
            .create_login_link(NewVibeLoginLink {
                token_hash: token_hash(&link_token),
                platform: actor.platform.clone(),
                guild_id: guild_id.clone(),
                user_id: target_user_id.clone(),
                requested_by_user_id: actor.user_id.clone(),
                expires_at: now + TimeDuration::minutes(LOGIN_LINK_TTL_MINUTES),
            })
            .await
            .map_err(BotToolError::Storage)?;

        let login_url = direct_login_url(&self.config.base_domain, &link_token);
        let session_days = self.config.auth.session_days;
        let content = format!(
            "Chudbot Vibe sign-in\n\n<{login_url}>\n\nThis single-use link was requested for you by <@{}>. It expires in {LOGIN_LINK_TTL_MINUTES} minutes. After you open it, your sign-in lasts for {session_days} days. Do not share this link.",
            actor.user_id.as_str(),
        );
        let recipient = UserRef {
            platform: actor.platform,
            guild_id: Some(guild_id),
            user_id: target_user_id.clone(),
        };
        if let Err(error) = self.platforms.deliver_login_link(recipient, content).await {
            if let Err(invalidate_error) = self
                .storage
                .consume_login_link(&token_hash(&link_token), now)
                .await
            {
                tracing::warn!(error=%invalidate_error, "failed to invalidate undelivered Vibe login link");
            }
            return Err(BotToolError::Platform(format!(
                "could not send the login link by DM: {error}"
            )));
        }

        let value = serde_json::json!({
            "delivered": true,
            "userId": target_user_id.as_str(),
            "displayName": membership.display_name,
            "expiresInMinutes": LOGIN_LINK_TTL_MINUTES,
            "sessionDays": session_days,
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

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use super::*;
    use chudbot_vibe::config::{VibeAccessConfig, VibeAuthConfig, VibeSandboxConfig};

    #[derive(Clone, Default)]
    struct FakeStorage {
        created: Arc<Mutex<Vec<NewVibeLoginLink>>>,
        consumed: Arc<Mutex<usize>>,
    }

    impl VibeLoginLinkStorage for FakeStorage {
        async fn create_login_link(&self, link: NewVibeLoginLink) -> Result<(), String> {
            self.created.lock().unwrap().push(link);
            Ok(())
        }

        async fn consume_login_link(
            &self,
            _token_hash: &[u8],
            _now: OffsetDateTime,
        ) -> Result<Option<VibeLoginLink>, String> {
            *self.consumed.lock().unwrap() += 1;
            Ok(None)
        }
    }

    #[derive(Clone)]
    struct FakePlatform {
        member: bool,
        fail_dm: bool,
        messages: Arc<Mutex<Vec<(UserRef, String)>>>,
    }

    impl VibeLoginLinkPlatform for FakePlatform {
        async fn lookup_login_link_membership(
            &self,
            _platform: &PlatformName,
            guild_id: &ExternalId,
            user_id: &ExternalId,
        ) -> Result<Option<VibeMembership>, String> {
            Ok(self.member.then(|| VibeMembership {
                user_id: user_id.clone(),
                username: "target".to_string(),
                display_name: "Target User".to_string(),
                avatar_url: None,
                guild_id: guild_id.clone(),
                guild_display_name: "Target User".to_string(),
            }))
        }

        async fn deliver_login_link(
            &self,
            recipient: UserRef,
            content: String,
        ) -> Result<(), String> {
            if self.fail_dm {
                return Err("DMs are closed".to_string());
            }
            self.messages.lock().unwrap().push((recipient, content));
            Ok(())
        }
    }

    fn tool(
        platform: FakePlatform,
        storage: FakeStorage,
    ) -> VibeLoginLinkTool<FakePlatform, FakeStorage> {
        let user = UserRef {
            platform: PlatformName::new("discord"),
            guild_id: Some(ExternalId::new("100")),
            user_id: ExternalId::new("200"),
        };
        VibeLoginLinkTool {
            platforms: platform,
            storage,
            context: RuntimeToolContext::new(
                MessageRef {
                    platform: PlatformName::new("discord"),
                    guild_id: Some(ExternalId::new("100")),
                    channel_id: ExternalId::new("300"),
                    message_id: ExternalId::new("400"),
                },
                ConversationId::new(),
                TurnId::new(),
                user,
            ),
            config: VibeConfig {
                enabled: true,
                base_domain: "vibe.example".to_string(),
                root_dir: std::path::PathBuf::from("/tmp/vibe"),
                reserved_names: Vec::new(),
                access: VibeAccessConfig::default(),
                auth: VibeAuthConfig {
                    client_id: "client".to_string(),
                    client_secret: "secret".to_string(),
                    session_days: 14,
                },
                sandbox: VibeSandboxConfig::default(),
                limits: chudbot_vibe::VibeLimitsConfig::default(),
            },
        }
    }

    fn call(input: serde_json::Value) -> ClientToolCall {
        ClientToolCall {
            id: ToolUseId::new("call-1"),
            name: ToolName::new(SEND_VIBE_LOGIN_LINK_TOOL),
            input,
        }
    }

    #[tokio::test]
    async fn sends_secret_only_to_target_dm_and_returns_safe_metadata() {
        let storage = FakeStorage::default();
        let messages = Arc::new(Mutex::new(Vec::new()));
        let output = tool(
            FakePlatform {
                member: true,
                fail_dm: false,
                messages: messages.clone(),
            },
            storage.clone(),
        )
        .call(call(serde_json::json!({"userId": "201"})))
        .await
        .unwrap();

        let created = storage.created.lock().unwrap();
        assert_eq!(created.len(), 1);
        assert_eq!(created[0].user_id.as_str(), "201");
        assert_eq!(created[0].requested_by_user_id.as_str(), "200");
        assert_eq!(output.trace_response["delivered"], true);
        assert_eq!(output.trace_response["sessionDays"], 14);
        assert!(!output.trace_response.to_string().contains("login/direct"));

        let messages = messages.lock().unwrap();
        assert_eq!(messages[0].0.user_id.as_str(), "201");
        assert!(
            messages[0]
                .1
                .contains("https://vibe.example/login/direct?token=")
        );
        assert!(messages[0].1.contains("lasts for 14 days"));
    }

    #[tokio::test]
    async fn rejects_nonmember_before_creating_or_sending_link() {
        let storage = FakeStorage::default();
        let messages = Arc::new(Mutex::new(Vec::new()));
        let error = tool(
            FakePlatform {
                member: false,
                fail_dm: false,
                messages: messages.clone(),
            },
            storage.clone(),
        )
        .call(call(serde_json::json!({"userId": "201"})))
        .await
        .unwrap_err();

        assert!(error.to_string().contains("not a current member"));
        assert!(storage.created.lock().unwrap().is_empty());
        assert!(messages.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn invalidates_link_when_dm_delivery_fails() {
        let storage = FakeStorage::default();
        let error = tool(
            FakePlatform {
                member: true,
                fail_dm: true,
                messages: Arc::default(),
            },
            storage.clone(),
        )
        .call(call(serde_json::json!({})))
        .await
        .unwrap_err();

        assert!(error.to_string().contains("could not send"));
        assert_eq!(*storage.consumed.lock().unwrap(), 1);
    }
}
