//! Provider-neutral contracts for Vibe websites.

use std::future::Future;

use serde::{Deserialize, Serialize};
use time::OffsetDateTime;
use uuid::Uuid;

use crate::{ConversationId, ExternalId, PlatformName, ToolUseId, TurnId};

macro_rules! uuid_id {
    ($name:ident, $doc:literal) => {
        #[doc = $doc]
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
        #[serde(transparent)]
        pub struct $name(pub Uuid);

        impl $name {
            /// Allocate a random id.
            pub fn new() -> Self {
                Self(Uuid::new_v4())
            }
        }

        impl Default for $name {
            fn default() -> Self {
                Self::new()
            }
        }

        impl std::fmt::Display for $name {
            fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                self.0.fmt(formatter)
            }
        }
    };
}

uuid_id!(VibeSiteId, "Stable id for one Vibe site.");
uuid_id!(VibeRevisionId, "Stable id for one immutable site revision.");
uuid_id!(VibeJobId, "Stable id for one coding job.");

/// Trusted actor assembled from the incoming platform event.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VibeActor {
    pub platform: PlatformName,
    pub guild_id: Option<ExternalId>,
    pub user_id: ExternalId,
    pub conversation_id: ConversationId,
    pub turn_id: TurnId,
    pub is_admin: bool,
}

/// Site lifecycle state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VibeSiteStatus {
    Creating,
    Active,
    Archived,
}

/// Who may view a deployed site.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VibeSiteAccess {
    /// Require Discord OAuth and current membership in the owning guild.
    #[default]
    Protected,
    /// Serve the deployed site without authenticating the viewer.
    Public,
}

impl VibeSiteAccess {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Protected => "protected",
            Self::Public => "public",
        }
    }

    pub const fn label(self) -> &'static str {
        match self {
            Self::Protected => "🔒 protected",
            Self::Public => "public",
        }
    }
}

/// Coding job action.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VibeAction {
    Create,
    Edit,
}

/// Coding job lifecycle state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VibeJobState {
    Queued,
    Coding,
    Building,
    Repair,
    Committing,
    Done,
    NoChanges,
    Failed,
    Cancelled,
    TimedOut,
}

/// Actor's effective role for a site.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VibeRole {
    Member,
    Editor,
    Owner,
    Admin,
}

/// Stored site metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VibeSite {
    pub id: VibeSiteId,
    pub name: String,
    pub platform: PlatformName,
    pub guild_id: ExternalId,
    pub owner_user_id: ExternalId,
    pub description: String,
    pub status: VibeSiteStatus,
    pub access: VibeSiteAccess,
    pub active_revision_id: Option<VibeRevisionId>,
    pub running_job_id: Option<VibeJobId>,
    pub created_at: OffsetDateTime,
    pub updated_at: OffsetDateTime,
}

/// Immutable revision metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VibeRevision {
    pub id: VibeRevisionId,
    pub site_id: VibeSiteId,
    pub ordinal: i32,
    pub parent_revision_id: Option<VibeRevisionId>,
    pub commit_oid: String,
    pub image_id: String,
    pub message: String,
    pub build_log: String,
    pub actor_user_id: ExternalId,
    pub conversation_id: ConversationId,
    pub turn_id: TurnId,
    pub job_id: VibeJobId,
    pub created_at: OffsetDateTime,
}

/// Stored coding job.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VibeJob {
    pub id: VibeJobId,
    pub site_id: Option<VibeSiteId>,
    pub site_name: String,
    pub action: VibeAction,
    pub actor_user_id: ExternalId,
    pub platform: PlatformName,
    pub guild_id: ExternalId,
    pub conversation_id: ConversationId,
    pub turn_id: TurnId,
    pub tool_use_id: ToolUseId,
    pub state: VibeJobState,
    pub error: Option<String>,
    pub created_at: OffsetDateTime,
    pub updated_at: OffsetDateTime,
}

/// Atomic create/edit job request.
#[derive(Debug, Clone)]
pub struct CreateVibeJob {
    pub id: VibeJobId,
    pub site_id: VibeSiteId,
    pub site_name: String,
    pub description: String,
    pub action: VibeAction,
    pub actor: VibeActor,
    pub tool_use_id: ToolUseId,
    pub max_running_jobs_per_guild: u16,
}

/// Values written when a clean build becomes active.
#[derive(Debug, Clone)]
pub struct CompleteVibeRevision {
    pub revision: VibeRevision,
    pub expected_job_id: VibeJobId,
}

/// Hashed browser session row.
#[derive(Debug, Clone)]
pub struct VibeSession {
    pub token_hash: Vec<u8>,
    pub platform: PlatformName,
    pub user_id: ExternalId,
    pub created_at: OffsetDateTime,
    pub expires_at: OffsetDateTime,
    pub revoked_at: Option<OffsetDateTime>,
}

/// New browser session values.
#[derive(Debug, Clone)]
pub struct NewVibeSession {
    pub token_hash: Vec<u8>,
    pub platform: PlatformName,
    pub user_id: ExternalId,
    pub expires_at: OffsetDateTime,
}

/// OAuth state row.
#[derive(Debug, Clone)]
pub struct VibeOauthState {
    pub state_hash: Vec<u8>,
    pub return_url: String,
    pub expires_at: OffsetDateTime,
    pub consumed_at: Option<OffsetDateTime>,
}

/// New OAuth state values.
#[derive(Debug, Clone)]
pub struct NewVibeOauthState {
    pub state_hash: Vec<u8>,
    pub return_url: String,
    pub expires_at: OffsetDateTime,
}

/// Result of the identify-only Discord OAuth exchange.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VibeOauthLogin {
    pub platform: PlatformName,
    pub user_id: ExternalId,
}

/// Current guild membership and display metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VibeMembership {
    pub user_id: ExternalId,
    pub username: String,
    pub display_name: String,
    pub avatar_url: Option<String>,
    pub guild_id: ExternalId,
    pub guild_display_name: String,
}

/// Public identity returned to a site's browser SDK.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct VibeIdentity {
    pub id: String,
    pub username: String,
    pub display_name: String,
    pub avatar_url: Option<String>,
    pub guild: VibeIdentityGuild,
    pub site: VibeIdentitySite,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct VibeIdentityGuild {
    pub id: String,
    pub display_name: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct VibeIdentitySite {
    pub name: String,
}

/// Durable Vibe persistence operations.
pub trait VibeStorage: Send + Sync {
    type Error: std::error::Error + Send + Sync + 'static;

    fn find_site_by_name(
        &self,
        name: &str,
    ) -> impl Future<Output = Result<Option<VibeSite>, Self::Error>> + Send;
    fn find_site(
        &self,
        id: VibeSiteId,
    ) -> impl Future<Output = Result<Option<VibeSite>, Self::Error>> + Send;
    fn create_job(
        &self,
        input: CreateVibeJob,
    ) -> impl Future<Output = Result<VibeJob, Self::Error>> + Send;
    fn find_job_by_tool_use(
        &self,
        tool_use_id: &ToolUseId,
    ) -> impl Future<Output = Result<Option<VibeJob>, Self::Error>> + Send;
    fn update_job_state(
        &self,
        id: VibeJobId,
        state: VibeJobState,
        error: Option<&str>,
    ) -> impl Future<Output = Result<(), Self::Error>> + Send;
    fn complete_revision(
        &self,
        input: CompleteVibeRevision,
    ) -> impl Future<Output = Result<(), Self::Error>> + Send;
    fn active_revision(
        &self,
        site_id: VibeSiteId,
    ) -> impl Future<Output = Result<Option<VibeRevision>, Self::Error>> + Send;
    fn list_revisions(
        &self,
        site_id: VibeSiteId,
    ) -> impl Future<Output = Result<Vec<VibeRevision>, Self::Error>> + Send;
    fn list_sites(
        &self,
        actor: &VibeActor,
        filter: Option<&str>,
    ) -> impl Future<Output = Result<Vec<(VibeSite, VibeRole)>, Self::Error>> + Send;
    fn count_running_jobs(
        &self,
        platform: &PlatformName,
        guild_id: &ExternalId,
    ) -> impl Future<Output = Result<u64, Self::Error>> + Send;
    fn is_editor(
        &self,
        site_id: VibeSiteId,
        platform: &PlatformName,
        user_id: &ExternalId,
    ) -> impl Future<Output = Result<bool, Self::Error>> + Send;
    fn add_editor(
        &self,
        site_id: VibeSiteId,
        platform: &PlatformName,
        user_id: &ExternalId,
        added_by: &ExternalId,
    ) -> impl Future<Output = Result<(), Self::Error>> + Send;
    fn remove_editor(
        &self,
        site_id: VibeSiteId,
        platform: &PlatformName,
        user_id: &ExternalId,
    ) -> impl Future<Output = Result<bool, Self::Error>> + Send;
    fn set_site_status(
        &self,
        site_id: VibeSiteId,
        status: VibeSiteStatus,
    ) -> impl Future<Output = Result<(), Self::Error>> + Send;
    fn set_site_access(
        &self,
        site_id: VibeSiteId,
        access: VibeSiteAccess,
    ) -> impl Future<Output = Result<(), Self::Error>> + Send;
    fn activate_existing_revision(
        &self,
        site_id: VibeSiteId,
        revision_id: VibeRevisionId,
    ) -> impl Future<Output = Result<(), Self::Error>> + Send;
    fn purge_site(
        &self,
        site_id: VibeSiteId,
    ) -> impl Future<Output = Result<(), Self::Error>> + Send;
    fn recover_interrupted_jobs(
        &self,
    ) -> impl Future<Output = Result<Vec<VibeSite>, Self::Error>> + Send;

    fn create_oauth_state(
        &self,
        state: NewVibeOauthState,
    ) -> impl Future<Output = Result<(), Self::Error>> + Send;
    fn consume_oauth_state(
        &self,
        state_hash: &[u8],
        now: OffsetDateTime,
    ) -> impl Future<Output = Result<Option<VibeOauthState>, Self::Error>> + Send;
    fn create_session(
        &self,
        session: NewVibeSession,
    ) -> impl Future<Output = Result<(), Self::Error>> + Send;
    fn find_session(
        &self,
        token_hash: &[u8],
        now: OffsetDateTime,
    ) -> impl Future<Output = Result<Option<VibeSession>, Self::Error>> + Send;
    fn revoke_session(
        &self,
        token_hash: &[u8],
    ) -> impl Future<Output = Result<(), Self::Error>> + Send;
}

/// Discord-specific login and membership boundary.
pub trait VibeIdentityProvider: Clone + Send + Sync + 'static {
    type Error: std::error::Error + Send + Sync + 'static;

    fn authorization_url(&self, state: &str, callback_url: &str) -> Result<String, Self::Error>;
    fn exchange_code(
        &self,
        code: &str,
        callback_url: &str,
    ) -> impl Future<Output = Result<VibeOauthLogin, Self::Error>> + Send;
    fn guild_membership(
        &self,
        platform: &PlatformName,
        guild_id: &ExternalId,
        user_id: &ExternalId,
    ) -> impl Future<Output = Result<Option<VibeMembership>, Self::Error>> + Send;
}
