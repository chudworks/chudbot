use chudbot_api::{VibeActor, VibeRole, VibeSite};
use thiserror::Error;

use crate::config::VibeAccessConfig;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VibeOperation {
    View,
    Create,
    Edit,
    Manage,
    Purge,
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum VibeAccessError {
    #[error("Vibe is disabled")]
    Disabled,
    #[error("Vibe sites can only be created from a server channel")]
    DirectMessage,
    #[error("this server is not enabled for Vibe")]
    GuildNotAllowed,
    #[error("you must still be a member of this server")]
    NotMember,
    #[error("this site belongs to another server")]
    WrongGuild,
    #[error("you do not have permission to perform this action")]
    Forbidden,
}

#[derive(Debug, Clone)]
pub struct VibeAccess {
    enabled: bool,
    config: VibeAccessConfig,
}

impl VibeAccess {
    pub fn new(enabled: bool, config: VibeAccessConfig) -> Self {
        Self { enabled, config }
    }

    pub fn check_rollout(
        &self,
        actor: &VibeActor,
        operation: VibeOperation,
    ) -> Result<(), VibeAccessError> {
        if !self.enabled {
            return Err(VibeAccessError::Disabled);
        }
        let Some(guild_id) = actor.guild_id.as_ref() else {
            return Err(VibeAccessError::DirectMessage);
        };
        if !self.config.allowed_guilds.is_empty()
            && !self
                .config
                .allowed_guilds
                .iter()
                .any(|allowed| allowed.platform == actor.platform && allowed.guild_id == *guild_id)
        {
            return Err(VibeAccessError::GuildNotAllowed);
        }
        if self.config.admins_only
            && matches!(operation, VibeOperation::Create | VibeOperation::Edit)
            && !actor.is_admin
        {
            return Err(VibeAccessError::Forbidden);
        }
        Ok(())
    }

    pub fn check_site(
        &self,
        actor: &VibeActor,
        site: &VibeSite,
        role: VibeRole,
        operation: VibeOperation,
        current_member: bool,
    ) -> Result<(), VibeAccessError> {
        self.check_rollout(actor, operation)?;
        if !current_member {
            return Err(VibeAccessError::NotMember);
        }
        if actor.platform != site.platform || actor.guild_id.as_ref() != Some(&site.guild_id) {
            return Err(VibeAccessError::WrongGuild);
        }
        let allowed = match operation {
            VibeOperation::View => true,
            VibeOperation::Create => false,
            VibeOperation::Edit => {
                matches!(role, VibeRole::Editor | VibeRole::Owner | VibeRole::Admin)
            }
            VibeOperation::Manage => matches!(role, VibeRole::Owner | VibeRole::Admin),
            VibeOperation::Purge => matches!(role, VibeRole::Admin),
        };
        allowed.then_some(()).ok_or(VibeAccessError::Forbidden)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{VibeAccessConfig, VibeAllowedGuild};
    use chudbot_api::{
        ConversationId, ExternalId, PlatformName, TurnId, VibeSiteId, VibeSiteStatus,
    };
    use test_case::test_case;
    use time::OffsetDateTime;

    fn actor(guild: Option<&str>, admin: bool) -> VibeActor {
        VibeActor {
            platform: PlatformName::new("discord"),
            guild_id: guild.map(ExternalId::new),
            user_id: ExternalId::new("10"),
            conversation_id: ConversationId::new(),
            turn_id: TurnId::new(),
            is_admin: admin,
        }
    }
    fn site() -> VibeSite {
        VibeSite {
            id: VibeSiteId::new(),
            name: "site".into(),
            platform: PlatformName::new("discord"),
            guild_id: ExternalId::new("1"),
            owner_user_id: ExternalId::new("10"),
            description: String::new(),
            status: VibeSiteStatus::Active,
            active_revision_id: None,
            running_job_id: None,
            created_at: OffsetDateTime::UNIX_EPOCH,
            updated_at: OffsetDateTime::UNIX_EPOCH,
        }
    }

    #[test_case(VibeRole::Member,VibeOperation::View,true,true;"member views")]
    #[test_case(VibeRole::Member,VibeOperation::Edit,true,false;"member cannot edit")]
    #[test_case(VibeRole::Editor,VibeOperation::Edit,true,true;"editor edits")]
    #[test_case(VibeRole::Owner,VibeOperation::Manage,true,true;"owner manages")]
    #[test_case(VibeRole::Admin,VibeOperation::Purge,true,true;"admin purges")]
    #[test_case(VibeRole::Admin,VibeOperation::View,false,false;"admin must remain member")]
    fn site_roles(role: VibeRole, operation: VibeOperation, member: bool, allowed: bool) {
        let access = VibeAccess::new(true, VibeAccessConfig::default());
        assert_eq!(
            access
                .check_site(
                    &actor(Some("1"), role == VibeRole::Admin),
                    &site(),
                    role,
                    operation,
                    member
                )
                .is_ok(),
            allowed
        );
    }

    #[test]
    fn rollout_rejects_dm_allowlist_and_nonadmin_writes() {
        let config = VibeAccessConfig {
            admins_only: true,
            allowed_guilds: vec![VibeAllowedGuild {
                platform: PlatformName::new("discord"),
                guild_id: ExternalId::new("1"),
            }],
        };
        let access = VibeAccess::new(true, config);
        assert_eq!(
            access.check_rollout(&actor(None, false), VibeOperation::Create),
            Err(VibeAccessError::DirectMessage)
        );
        assert_eq!(
            access.check_rollout(&actor(Some("2"), false), VibeOperation::Create),
            Err(VibeAccessError::GuildNotAllowed)
        );
        assert_eq!(
            access.check_rollout(&actor(Some("1"), false), VibeOperation::Edit),
            Err(VibeAccessError::Forbidden)
        );
        assert!(
            access
                .check_rollout(&actor(Some("1"), true), VibeOperation::Edit)
                .is_ok()
        );
        assert!(
            access
                .check_rollout(&actor(Some("1"), false), VibeOperation::View)
                .is_ok()
        );
    }

    #[test]
    fn wrong_guild_is_rejected() {
        let access = VibeAccess::new(true, VibeAccessConfig::default());
        assert_eq!(
            access.check_site(
                &actor(Some("2"), false),
                &site(),
                VibeRole::Owner,
                VibeOperation::Edit,
                true
            ),
            Err(VibeAccessError::WrongGuild)
        );
    }
}
