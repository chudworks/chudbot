use std::cmp::Ordering;

use chudbot_api::vibe::*;
use chudbot_api::{ConversationId, ExternalId, PlatformName, ToolUseId, TurnId};
use serde_json::{Map, Value};
use sqlx::Row;
use time::OffsetDateTime;
use uuid::Uuid;

use crate::{SqlxStorage, SqlxStorageError};

impl VibeStorage for SqlxStorage {
    type Error = SqlxStorageError;

    async fn find_site_by_name(&self, name: &str) -> Result<Option<VibeSite>, Self::Error> {
        site_query("WHERE name = $1")
            .bind(name)
            .fetch_optional(&self.pool)
            .await?
            .map(site_from_row)
            .transpose()
    }

    async fn find_site(&self, id: VibeSiteId) -> Result<Option<VibeSite>, Self::Error> {
        site_query("WHERE id = $1")
            .bind(id.0)
            .fetch_optional(&self.pool)
            .await?
            .map(site_from_row)
            .transpose()
    }

    async fn create_job(&self, input: CreateVibeJob) -> Result<VibeJob, Self::Error> {
        if let Some(job) = self.find_job_by_tool_use(&input.tool_use_id).await? {
            return Ok(job);
        }
        let guild =
            input.actor.guild_id.as_ref().ok_or_else(|| {
                SqlxStorageError::InvalidReference("Vibe actor has no guild".into())
            })?;
        let mut tx = self.pool.begin().await?;
        sqlx::query("SELECT pg_advisory_xact_lock(hashtextextended($1 || ':' || $2, 0))")
            .bind(input.actor.platform.as_str())
            .bind(guild.as_str())
            .execute(&mut *tx)
            .await?;
        let running = sqlx::query_scalar::<_, i64>("SELECT count(*) FROM vibe_jobs WHERE platform=$1 AND guild_id=$2 AND state IN('queued','coding','building','repair','committing')")
            .bind(input.actor.platform.as_str())
            .bind(guild.as_str())
            .fetch_one(&mut *tx)
            .await?;
        if running >= i64::from(input.max_running_jobs_per_guild) {
            return Err(SqlxStorageError::VibeConflict(
                "guild job limit reached".into(),
            ));
        }
        match input.action {
            VibeAction::Create => {
                sqlx::query("INSERT INTO vibe_sites (id,name,platform,guild_id,owner_user_id,description,status,access) VALUES ($1,$2,$3,$4,$5,$6,'creating','protected')")
                    .bind(input.site_id.0).bind(&input.site_name).bind(input.actor.platform.as_str()).bind(guild.as_str())
                    .bind(input.actor.user_id.as_str()).bind(&input.description).execute(&mut *tx).await?;
            }
            VibeAction::Edit => {
                let claim_state = sqlx::query_scalar::<_, String>(
                    "SELECT CASE WHEN platform<>$2 OR guild_id<>$3 OR NOT ($5 OR owner_user_id=$4 OR EXISTS(SELECT 1 FROM vibe_site_editors e WHERE e.site_id=vibe_sites.id AND e.platform=$2 AND e.user_id=$4)) THEN 'unauthorized' WHEN running_job_id IS NOT NULL THEN 'busy' ELSE 'ready' END FROM vibe_sites WHERE id=$1 FOR UPDATE",
                )
                .bind(input.site_id.0)
                .bind(input.actor.platform.as_str())
                .bind(guild.as_str())
                .bind(input.actor.user_id.as_str())
                .bind(input.actor.is_admin)
                .fetch_optional(&mut *tx)
                .await?;
                match claim_state.as_deref() {
                    Some("ready") => {}
                    Some("unauthorized") => {
                        return Err(SqlxStorageError::VibeConflict(
                            "site edit not authorized".into(),
                        ));
                    }
                    _ => {
                        return Err(SqlxStorageError::VibeConflict(
                            "site is missing or busy".into(),
                        ));
                    }
                }
            }
        }
        let row = sqlx::query("INSERT INTO vibe_jobs (id,site_id,site_name,action,actor_user_id,platform,guild_id,conversation_id,turn_id,tool_use_id,state) VALUES ($1,$2,$3,$4::vibe_action,$5,$6,$7,$8,$9,$10,'queued') RETURNING id,site_id,site_name,action::text AS action,actor_user_id,platform,guild_id,conversation_id,turn_id,tool_use_id,state::text AS state,error,created_at,updated_at")
            .bind(input.id.0).bind(input.site_id.0).bind(&input.site_name).bind(action_text(input.action))
            .bind(input.actor.user_id.as_str()).bind(input.actor.platform.as_str()).bind(guild.as_str())
            .bind(input.actor.conversation_id.0).bind(input.actor.turn_id.0).bind(input.tool_use_id.as_str())
            .fetch_one(&mut *tx).await?;
        sqlx::query("UPDATE vibe_sites SET running_job_id=$2,updated_at=now() WHERE id=$1")
            .bind(input.site_id.0)
            .bind(input.id.0)
            .execute(&mut *tx)
            .await?;
        tx.commit().await?;
        job_from_row(row)
    }

    async fn find_job_by_tool_use(&self, id: &ToolUseId) -> Result<Option<VibeJob>, Self::Error> {
        sqlx::query("SELECT id,site_id,site_name,action::text AS action,actor_user_id,platform,guild_id,conversation_id,turn_id,tool_use_id,state::text AS state,error,created_at,updated_at FROM vibe_jobs WHERE tool_use_id=$1")
            .bind(id.as_str()).fetch_optional(&self.pool).await?.map(job_from_row).transpose()
    }

    async fn update_job_state(
        &self,
        id: VibeJobId,
        state: VibeJobState,
        error: Option<&str>,
    ) -> Result<(), Self::Error> {
        let terminal = matches!(
            state,
            VibeJobState::Done
                | VibeJobState::NoChanges
                | VibeJobState::Failed
                | VibeJobState::Cancelled
                | VibeJobState::TimedOut
        );
        let mut tx = self.pool.begin().await?;
        sqlx::query(
            "UPDATE vibe_jobs SET state=$2::vibe_job_state,error=$3,updated_at=now() WHERE id=$1",
        )
        .bind(id.0)
        .bind(job_state_text(state))
        .bind(error.map(|e| e.chars().take(8_192).collect::<String>()))
        .execute(&mut *tx)
        .await?;
        if terminal {
            sqlx::query("UPDATE vibe_sites SET running_job_id=NULL,updated_at=now() WHERE running_job_id=$1").bind(id.0).execute(&mut *tx).await?;
            if state != VibeJobState::Done {
                sqlx::query("DELETE FROM vibe_sites WHERE status='creating' AND id=(SELECT site_id FROM vibe_jobs WHERE id=$1)").bind(id.0).execute(&mut *tx).await?;
            }
        }
        tx.commit().await?;
        Ok(())
    }

    async fn complete_revision(&self, input: CompleteVibeRevision) -> Result<(), Self::Error> {
        let r = input.revision;
        let mut tx = self.pool.begin().await?;
        let lock = sqlx::query_scalar::<_, Option<Uuid>>(
            "SELECT running_job_id FROM vibe_sites WHERE id=$1 FOR UPDATE",
        )
        .bind(r.site_id.0)
        .fetch_optional(&mut *tx)
        .await?
        .flatten();
        if lock != Some(input.expected_job_id.0) {
            return Err(SqlxStorageError::VibeConflict(
                "site job lock changed".into(),
            ));
        }
        sqlx::query("INSERT INTO vibe_revisions (id,site_id,ordinal,parent_revision_id,commit_oid,image_id,message,build_log,actor_user_id,conversation_id,turn_id,job_id,created_at) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13)")
            .bind(r.id.0).bind(r.site_id.0).bind(r.ordinal).bind(r.parent_revision_id.map(|id| id.0)).bind(&r.commit_oid)
            .bind(&r.image_id).bind(&r.message).bind(r.build_log.chars().take(32_768).collect::<String>())
            .bind(r.actor_user_id.as_str()).bind(r.conversation_id.0).bind(r.turn_id.0).bind(r.job_id.0).bind(r.created_at)
            .execute(&mut *tx).await?;
        sqlx::query("UPDATE vibe_sites SET active_revision_id=$2,running_job_id=NULL,status='active',updated_at=now() WHERE id=$1")
            .bind(r.site_id.0).bind(r.id.0).execute(&mut *tx).await?;
        sqlx::query("UPDATE vibe_jobs SET state='done',error=NULL,updated_at=now() WHERE id=$1")
            .bind(input.expected_job_id.0)
            .execute(&mut *tx)
            .await?;
        tx.commit().await?;
        Ok(())
    }

    async fn active_revision(&self, site: VibeSiteId) -> Result<Option<VibeRevision>, Self::Error> {
        revision_query("JOIN vibe_sites s ON s.active_revision_id=r.id WHERE s.id=$1")
            .bind(site.0)
            .fetch_optional(&self.pool)
            .await?
            .map(revision_from_row)
            .transpose()
    }

    async fn list_revisions(&self, site: VibeSiteId) -> Result<Vec<VibeRevision>, Self::Error> {
        revision_query("WHERE r.site_id=$1 ORDER BY r.ordinal DESC")
            .bind(site.0)
            .fetch_all(&self.pool)
            .await?
            .into_iter()
            .map(revision_from_row)
            .collect()
    }

    async fn list_sites(
        &self,
        actor: &VibeActor,
        filter: Option<&str>,
    ) -> Result<Vec<(VibeSite, VibeRole)>, Self::Error> {
        let Some(guild) = actor.guild_id.as_ref() else {
            return Ok(Vec::new());
        };
        let pattern = filter.map(|f| format!("%{}%", f.replace('%', "\\%").replace('_', "\\_")));
        let rows = sqlx::query("SELECT s.id,s.name,s.platform,s.guild_id,s.owner_user_id,s.description,s.status::text AS status,s.access::text AS access,s.active_revision_id,s.running_job_id,s.created_at,s.updated_at,EXISTS(SELECT 1 FROM vibe_site_editors e WHERE e.site_id=s.id AND e.platform=$1 AND e.user_id=$3) AS editor,EXISTS(SELECT 1 FROM vibe_jobs j WHERE j.site_id=s.id AND j.conversation_id=$4) AS current_conversation FROM vibe_sites s WHERE s.platform=$1 AND s.guild_id=$2 AND s.status='active' AND ($5::text IS NULL OR s.name ILIKE $5 ESCAPE '\\' OR s.description ILIKE $5 ESCAPE '\\') ORDER BY current_conversation DESC,(s.owner_user_id=$3) DESC,s.updated_at DESC")
            .bind(actor.platform.as_str()).bind(guild.as_str()).bind(actor.user_id.as_str()).bind(actor.conversation_id.0).bind(pattern).fetch_all(&self.pool).await?;
        rows.into_iter()
            .map(|row| {
                let role = if actor.is_admin {
                    VibeRole::Admin
                } else if row.get::<String, _>("owner_user_id") == actor.user_id.as_str() {
                    VibeRole::Owner
                } else if row.get::<bool, _>("editor") {
                    VibeRole::Editor
                } else {
                    VibeRole::Member
                };
                Ok((site_from_row(row)?, role))
            })
            .collect()
    }

    async fn is_editor(
        &self,
        site: VibeSiteId,
        platform: &PlatformName,
        user: &ExternalId,
    ) -> Result<bool, Self::Error> {
        Ok(sqlx::query_scalar("SELECT EXISTS(SELECT 1 FROM vibe_site_editors WHERE site_id=$1 AND platform=$2 AND user_id=$3)").bind(site.0).bind(platform.as_str()).bind(user.as_str()).fetch_one(&self.pool).await?)
    }
    async fn count_running_jobs(
        &self,
        platform: &PlatformName,
        guild: &ExternalId,
    ) -> Result<u64, Self::Error> {
        let count = sqlx::query_scalar::<_, i64>("SELECT count(*) FROM vibe_jobs WHERE platform=$1 AND guild_id=$2 AND state IN('queued','coding','building','repair','committing')")
            .bind(platform.as_str()).bind(guild.as_str()).fetch_one(&self.pool).await?;
        Ok(count.try_into().unwrap_or(u64::MAX))
    }
    async fn add_editor(
        &self,
        site: VibeSiteId,
        platform: &PlatformName,
        user: &ExternalId,
        by: &ExternalId,
    ) -> Result<(), Self::Error> {
        sqlx::query("INSERT INTO vibe_site_editors(site_id,platform,user_id,added_by_user_id) VALUES($1,$2,$3,$4) ON CONFLICT DO NOTHING").bind(site.0).bind(platform.as_str()).bind(user.as_str()).bind(by.as_str()).execute(&self.pool).await?;
        Ok(())
    }
    async fn remove_editor(
        &self,
        site: VibeSiteId,
        platform: &PlatformName,
        user: &ExternalId,
    ) -> Result<bool, Self::Error> {
        Ok(sqlx::query(
            "DELETE FROM vibe_site_editors WHERE site_id=$1 AND platform=$2 AND user_id=$3",
        )
        .bind(site.0)
        .bind(platform.as_str())
        .bind(user.as_str())
        .execute(&self.pool)
        .await?
        .rows_affected()
            > 0)
    }
    async fn set_site_status(
        &self,
        site: VibeSiteId,
        status: VibeSiteStatus,
    ) -> Result<(), Self::Error> {
        sqlx::query(
            "UPDATE vibe_sites SET status=$2::vibe_site_status,updated_at=now() WHERE id=$1",
        )
        .bind(site.0)
        .bind(site_status_text(status))
        .execute(&self.pool)
        .await?;
        Ok(())
    }
    async fn set_site_access(
        &self,
        site: VibeSiteId,
        access: VibeSiteAccess,
    ) -> Result<(), Self::Error> {
        sqlx::query(
            "UPDATE vibe_sites SET access=$2::vibe_site_access,updated_at=now() WHERE id=$1",
        )
        .bind(site.0)
        .bind(access.as_str())
        .execute(&self.pool)
        .await?;
        Ok(())
    }
    async fn activate_existing_revision(
        &self,
        site: VibeSiteId,
        revision: VibeRevisionId,
    ) -> Result<(), Self::Error> {
        let n=sqlx::query("UPDATE vibe_sites SET active_revision_id=$2,status='active',updated_at=now() WHERE id=$1 AND EXISTS(SELECT 1 FROM vibe_revisions WHERE id=$2 AND site_id=$1)").bind(site.0).bind(revision.0).execute(&self.pool).await?.rows_affected();
        if n == 0 {
            Err(SqlxStorageError::VibeConflict(
                "revision does not belong to site".into(),
            ))
        } else {
            Ok(())
        }
    }
    async fn purge_site(&self, site: VibeSiteId) -> Result<(), Self::Error> {
        sqlx::query("DELETE FROM vibe_sites WHERE id=$1")
            .bind(site.0)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    async fn recover_interrupted_jobs(&self) -> Result<Vec<VibeSite>, Self::Error> {
        let mut tx = self.pool.begin().await?;
        sqlx::query("UPDATE vibe_jobs SET state='failed',error='interrupted by process restart',updated_at=now() WHERE state IN('queued','coding','building','repair','committing')").execute(&mut *tx).await?;
        sqlx::query("UPDATE vibe_sites SET running_job_id=NULL,updated_at=now() WHERE running_job_id IS NOT NULL").execute(&mut *tx).await?;
        sqlx::query("DELETE FROM vibe_sites WHERE status='creating'")
            .execute(&mut *tx)
            .await?;
        let rows = site_query("WHERE active_revision_id IS NOT NULL")
            .fetch_all(&mut *tx)
            .await?;
        tx.commit().await?;
        rows.into_iter().map(site_from_row).collect()
    }

    async fn create_oauth_state(&self, state: NewVibeOauthState) -> Result<(), Self::Error> {
        sqlx::query(
            "INSERT INTO vibe_oauth_states(state_hash,return_url,expires_at) VALUES($1,$2,$3)",
        )
        .bind(state.state_hash)
        .bind(state.return_url)
        .bind(state.expires_at)
        .execute(&self.pool)
        .await?;
        Ok(())
    }
    async fn consume_oauth_state(
        &self,
        hash: &[u8],
        now: OffsetDateTime,
    ) -> Result<Option<VibeOauthState>, Self::Error> {
        Ok(sqlx::query("UPDATE vibe_oauth_states SET consumed_at=$2 WHERE state_hash=$1 AND consumed_at IS NULL AND expires_at>$2 RETURNING state_hash,return_url,expires_at,consumed_at").bind(hash).bind(now).fetch_optional(&self.pool).await?.map(|row| VibeOauthState{state_hash:row.get("state_hash"),return_url:row.get("return_url"),expires_at:row.get("expires_at"),consumed_at:row.get("consumed_at")}))
    }
    async fn create_login_link(&self, link: NewVibeLoginLink) -> Result<(), Self::Error> {
        sqlx::query("INSERT INTO vibe_login_links(token_hash,platform,guild_id,user_id,requested_by_user_id,expires_at) VALUES($1,$2,$3,$4,$5,$6)")
            .bind(link.token_hash)
            .bind(link.platform.as_str())
            .bind(link.guild_id.as_str())
            .bind(link.user_id.as_str())
            .bind(link.requested_by_user_id.as_str())
            .bind(link.expires_at)
            .execute(&self.pool)
            .await?;
        Ok(())
    }
    async fn consume_login_link(
        &self,
        hash: &[u8],
        now: OffsetDateTime,
    ) -> Result<Option<VibeLoginLink>, Self::Error> {
        Ok(sqlx::query("UPDATE vibe_login_links SET consumed_at=$2 WHERE token_hash=$1 AND consumed_at IS NULL AND expires_at>$2 RETURNING token_hash,platform,guild_id,user_id,requested_by_user_id,expires_at,consumed_at")
            .bind(hash)
            .bind(now)
            .fetch_optional(&self.pool)
            .await?
            .map(|row| VibeLoginLink {
                token_hash: row.get("token_hash"),
                platform: PlatformName::new(row.get::<String, _>("platform")),
                guild_id: ExternalId::new(row.get::<String, _>("guild_id")),
                user_id: ExternalId::new(row.get::<String, _>("user_id")),
                requested_by_user_id: ExternalId::new(
                    row.get::<String, _>("requested_by_user_id"),
                ),
                expires_at: row.get("expires_at"),
                consumed_at: row.get("consumed_at"),
            }))
    }
    async fn redeem_login_link(
        &self,
        link_hash: &[u8],
        session_hash: Vec<u8>,
        now: OffsetDateTime,
        session_expires_at: OffsetDateTime,
    ) -> Result<Option<VibeLoginLink>, Self::Error> {
        Ok(sqlx::query("WITH consumed AS (UPDATE vibe_login_links SET consumed_at=$3 WHERE token_hash=$1 AND consumed_at IS NULL AND expires_at>$3 RETURNING token_hash,platform,guild_id,user_id,requested_by_user_id,expires_at,consumed_at), inserted AS (INSERT INTO vibe_sessions(token_hash,platform,user_id,expires_at) SELECT $2,platform,user_id,$4 FROM consumed RETURNING 1) SELECT consumed.token_hash,consumed.platform,consumed.guild_id,consumed.user_id,consumed.requested_by_user_id,consumed.expires_at,consumed.consumed_at FROM consumed CROSS JOIN inserted")
            .bind(link_hash)
            .bind(session_hash)
            .bind(now)
            .bind(session_expires_at)
            .fetch_optional(&self.pool)
            .await?
            .map(|row| VibeLoginLink {
                token_hash: row.get("token_hash"),
                platform: PlatformName::new(row.get::<String, _>("platform")),
                guild_id: ExternalId::new(row.get::<String, _>("guild_id")),
                user_id: ExternalId::new(row.get::<String, _>("user_id")),
                requested_by_user_id: ExternalId::new(
                    row.get::<String, _>("requested_by_user_id"),
                ),
                expires_at: row.get("expires_at"),
                consumed_at: row.get("consumed_at"),
            }))
    }
    async fn create_session(&self, session: NewVibeSession) -> Result<(), Self::Error> {
        sqlx::query(
            "INSERT INTO vibe_sessions(token_hash,platform,user_id,expires_at) VALUES($1,$2,$3,$4)",
        )
        .bind(session.token_hash)
        .bind(session.platform.as_str())
        .bind(session.user_id.as_str())
        .bind(session.expires_at)
        .execute(&self.pool)
        .await?;
        Ok(())
    }
    async fn find_session(
        &self,
        hash: &[u8],
        now: OffsetDateTime,
    ) -> Result<Option<VibeSession>, Self::Error> {
        Ok(sqlx::query("SELECT token_hash,platform,user_id,created_at,expires_at,revoked_at FROM vibe_sessions WHERE token_hash=$1 AND revoked_at IS NULL AND expires_at>$2").bind(hash).bind(now).fetch_optional(&self.pool).await?.map(|row|VibeSession{token_hash:row.get("token_hash"),platform:PlatformName::new(row.get::<String,_>("platform")),user_id:ExternalId::new(row.get::<String,_>("user_id")),created_at:row.get("created_at"),expires_at:row.get("expires_at"),revoked_at:row.get("revoked_at")}))
    }
    async fn revoke_session(&self, hash: &[u8]) -> Result<(), Self::Error> {
        sqlx::query("UPDATE vibe_sessions SET revoked_at=now() WHERE token_hash=$1")
            .bind(hash)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    async fn execute_collection_operation(
        &self,
        site: VibeSiteId,
        collection: &str,
        actor: &ExternalId,
        operation: VibeCollectionOperation,
    ) -> Result<VibeCollectionOutcome, Self::Error> {
        let mut tx = self.pool.begin().await?;
        let access = sqlx::query_scalar::<_, String>(
            "SELECT access::text FROM vibe_sites WHERE id=$1 FOR SHARE",
        )
        .bind(site.0)
        .fetch_optional(&mut *tx)
        .await?;
        if access.as_deref() != Some("protected") {
            return Ok(VibeCollectionOutcome::Unavailable);
        }

        let outcome = match operation {
            VibeCollectionOperation::Write { mode, id, fields } => {
                let (row, before, kind) = match (mode, id) {
                    (VibeCollectionWriteMode::Put | VibeCollectionWriteMode::Insert, None) => {
                        let row = sqlx::query("INSERT INTO vibe_collection_documents(site_id,collection,id,document,inserted_by,updated_by) VALUES($1,$2,$3,$4,$5,$5) RETURNING id,document,inserted_by,inserted_at,updated_by,updated_at")
                            .bind(site.0)
                            .bind(collection)
                            .bind(Uuid::new_v4())
                            .bind(Value::Object(fields))
                            .bind(actor.as_str())
                            .fetch_optional(&mut *tx)
                            .await?;
                        (row, None, VibeCollectionChangeKind::Insert)
                    }
                    (VibeCollectionWriteMode::Put | VibeCollectionWriteMode::Update, Some(id)) => {
                        let before = sqlx::query("SELECT id,document,inserted_by,inserted_at,updated_by,updated_at FROM vibe_collection_documents WHERE site_id=$1 AND collection=$2 AND id=$3 FOR UPDATE")
                            .bind(site.0)
                            .bind(collection)
                            .bind(id)
                            .fetch_optional(&mut *tx)
                            .await?
                            .map(collection_document_from_row)
                            .transpose()?;
                        let row = sqlx::query("UPDATE vibe_collection_documents SET document=$4,updated_by=$5,updated_at=now() WHERE site_id=$1 AND collection=$2 AND id=$3 RETURNING id,document,inserted_by,inserted_at,updated_by,updated_at")
                            .bind(site.0)
                            .bind(collection)
                            .bind(id)
                            .bind(Value::Object(fields))
                            .bind(actor.as_str())
                            .fetch_optional(&mut *tx)
                            .await?;
                        (row, before, VibeCollectionChangeKind::Update)
                    }
                    _ => (None, None, VibeCollectionChangeKind::Update),
                };
                match row {
                    Some(row) => {
                        let after = collection_document_from_row(row)?;
                        let value = document_value(after.clone(), None);
                        VibeCollectionOutcome::Documents {
                            values: vec![value.clone()],
                            changes: vec![VibeCollectionChange {
                                kind,
                                before: before.map(|document| document_value(document, None)),
                                after: Some(value),
                            }],
                        }
                    }
                    None => VibeCollectionOutcome::MissingDocument,
                }
            }
            VibeCollectionOperation::Find(query) => {
                let documents = load_collection_documents(&mut tx, site, collection, false).await?;
                let selected = select_collection_documents(documents, &query);
                VibeCollectionOutcome::Documents {
                    values: project_collection_documents(selected, &query),
                    changes: Vec::new(),
                }
            }
            VibeCollectionOperation::Count(query) => {
                let documents = load_collection_documents(&mut tx, site, collection, false).await?;
                let selected = select_collection_documents(documents, &query);
                let count = project_collection_documents(selected, &query).len();
                VibeCollectionOutcome::Count(count.try_into().unwrap_or(u64::MAX))
            }
            VibeCollectionOperation::Delete { query, one } => {
                let documents = load_collection_documents(&mut tx, site, collection, true).await?;
                let selected = select_collection_documents(documents, &query);
                if one && selected.len() > 1 {
                    VibeCollectionOutcome::TooManyDocuments
                } else {
                    let ids = selected
                        .iter()
                        .map(|document| document.id)
                        .collect::<Vec<_>>();
                    if !ids.is_empty() {
                        sqlx::query("DELETE FROM vibe_collection_documents WHERE site_id=$1 AND collection=$2 AND id=ANY($3)")
                            .bind(site.0)
                            .bind(collection)
                            .bind(&ids)
                            .execute(&mut *tx)
                            .await?;
                    }
                    let changes = selected
                        .iter()
                        .cloned()
                        .map(|document| VibeCollectionChange {
                            kind: VibeCollectionChangeKind::Delete,
                            before: Some(document_value(document, None)),
                            after: None,
                        })
                        .collect();
                    VibeCollectionOutcome::Documents {
                        values: project_collection_documents(selected, &query),
                        changes,
                    }
                }
            }
        };
        tx.commit().await?;
        Ok(outcome)
    }
}

async fn load_collection_documents(
    tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
    site: VibeSiteId,
    collection: &str,
    lock: bool,
) -> Result<Vec<VibeCollectionDocument>, SqlxStorageError> {
    let sql = if lock {
        "SELECT id,document,inserted_by,inserted_at,updated_by,updated_at FROM vibe_collection_documents WHERE site_id=$1 AND collection=$2 ORDER BY id FOR UPDATE"
    } else {
        "SELECT id,document,inserted_by,inserted_at,updated_by,updated_at FROM vibe_collection_documents WHERE site_id=$1 AND collection=$2 ORDER BY id"
    };
    sqlx::query(sql)
        .bind(site.0)
        .bind(collection)
        .fetch_all(&mut **tx)
        .await?
        .into_iter()
        .map(collection_document_from_row)
        .collect()
}

fn collection_document_from_row(
    row: sqlx::postgres::PgRow,
) -> Result<VibeCollectionDocument, SqlxStorageError> {
    let document = row.get::<Value, _>("document");
    let Value::Object(fields) = document else {
        return Err(SqlxStorageError::InvalidReference(
            "Vibe collection document is not an object".into(),
        ));
    };
    Ok(VibeCollectionDocument {
        id: row.get("id"),
        fields,
        inserted_by: ExternalId::new(row.get::<String, _>("inserted_by")),
        inserted_at: row.get("inserted_at"),
        updated_by: ExternalId::new(row.get::<String, _>("updated_by")),
        updated_at: row.get("updated_at"),
    })
}

fn select_collection_documents(
    mut documents: Vec<VibeCollectionDocument>,
    query: &VibeCollectionQuery,
) -> Vec<VibeCollectionDocument> {
    documents.retain(|document| collection_document_matches(document, &query.filters));
    if !query.order_by.is_empty() {
        documents.sort_by(|left, right| compare_collection_documents(left, right, &query.order_by));
    }
    let offset = usize::try_from(query.offset).unwrap_or(usize::MAX);
    let limit = query
        .limit
        .map(|limit| usize::try_from(limit).unwrap_or(usize::MAX))
        .unwrap_or(usize::MAX);
    documents.into_iter().skip(offset).take(limit).collect()
}

fn collection_document_matches(
    document: &VibeCollectionDocument,
    filters: &std::collections::BTreeMap<String, Value>,
) -> bool {
    filters.iter().all(|(column, expected)| {
        let Some(actual) = collection_document_column(document, column) else {
            return false;
        };
        if !is_json_scalar(&actual) {
            return false;
        }
        match expected {
            Value::Array(values) => values
                .iter()
                .any(|value| is_json_scalar(value) && value == &actual),
            value => is_json_scalar(value) && value == &actual,
        }
    })
}

fn compare_collection_documents(
    left: &VibeCollectionDocument,
    right: &VibeCollectionDocument,
    order: &[VibeCollectionOrder],
) -> Ordering {
    for item in order {
        let ordering = compare_collection_values(
            collection_document_column(left, &item.column).as_ref(),
            collection_document_column(right, &item.column).as_ref(),
            item.direction,
        );
        if ordering != Ordering::Equal {
            return ordering;
        }
    }
    Ordering::Equal
}

fn compare_collection_values(
    left: Option<&Value>,
    right: Option<&Value>,
    direction: VibeCollectionOrderDirection,
) -> Ordering {
    match (
        sortable_collection_value(left),
        sortable_collection_value(right),
    ) {
        (
            Some(SortableCollectionValue::Number(left)),
            Some(SortableCollectionValue::Number(right)),
        ) => directed_order(
            left.partial_cmp(&right).unwrap_or(Ordering::Equal),
            direction,
        ),
        (Some(SortableCollectionValue::Text(left)), Some(SortableCollectionValue::Text(right))) => {
            directed_order(left.cmp(right), direction)
        }
        (Some(SortableCollectionValue::Number(_)), Some(SortableCollectionValue::Text(_))) => {
            directed_order(Ordering::Less, direction)
        }
        (Some(SortableCollectionValue::Text(_)), Some(SortableCollectionValue::Number(_))) => {
            directed_order(Ordering::Greater, direction)
        }
        (Some(_), None) => Ordering::Less,
        (None, Some(_)) => Ordering::Greater,
        (None, None) => Ordering::Equal,
    }
}

fn directed_order(ordering: Ordering, direction: VibeCollectionOrderDirection) -> Ordering {
    match direction {
        VibeCollectionOrderDirection::Asc => ordering,
        VibeCollectionOrderDirection::Desc => ordering.reverse(),
    }
}

enum SortableCollectionValue<'a> {
    Number(f64),
    Text(&'a str),
}

fn sortable_collection_value(value: Option<&Value>) -> Option<SortableCollectionValue<'_>> {
    match value? {
        Value::Number(number) => number.as_f64().map(SortableCollectionValue::Number),
        Value::String(text) => Some(SortableCollectionValue::Text(text)),
        _ => None,
    }
}

fn project_collection_documents(
    documents: Vec<VibeCollectionDocument>,
    query: &VibeCollectionQuery,
) -> Vec<Value> {
    let mut values = Vec::with_capacity(documents.len());
    for document in documents {
        let value = document_value(document, query.select.as_deref());
        if !query.distinct || !values.contains(&value) {
            values.push(value);
        }
    }
    values
}

fn document_value(document: VibeCollectionDocument, select: Option<&[String]>) -> Value {
    let all = serde_json::to_value(document)
        .expect("Vibe collection document serialization cannot fail")
        .as_object()
        .expect("Vibe collection document serializes as an object")
        .clone();
    let Some(select) = select else {
        return Value::Object(all);
    };
    Value::Object(
        select
            .iter()
            .filter_map(|column| {
                all.get(column)
                    .cloned()
                    .map(|value| (column.clone(), value))
            })
            .collect::<Map<_, _>>(),
    )
}

fn collection_document_column(document: &VibeCollectionDocument, column: &str) -> Option<Value> {
    match column {
        "id" => Some(Value::String(document.id.to_string())),
        "inserted_by" => Some(Value::String(document.inserted_by.to_string())),
        "inserted_at" => timestamp_value(document.inserted_at),
        "updated_by" => Some(Value::String(document.updated_by.to_string())),
        "updated_at" => timestamp_value(document.updated_at),
        _ => document.fields.get(column).cloned(),
    }
}

fn timestamp_value(timestamp: OffsetDateTime) -> Option<Value> {
    timestamp
        .format(&time::format_description::well_known::Rfc3339)
        .ok()
        .map(Value::String)
}

fn is_json_scalar(value: &Value) -> bool {
    !matches!(value, Value::Array(_) | Value::Object(_))
}

fn site_query(
    suffix: &str,
) -> sqlx::query::Query<'static, sqlx::Postgres, sqlx::postgres::PgArguments> {
    match suffix {
        "WHERE name = $1" => sqlx::query(
            "SELECT id,name,platform,guild_id,owner_user_id,description,status::text AS status,access::text AS access,active_revision_id,running_job_id,created_at,updated_at FROM vibe_sites WHERE name = $1",
        ),
        "WHERE id = $1" => sqlx::query(
            "SELECT id,name,platform,guild_id,owner_user_id,description,status::text AS status,access::text AS access,active_revision_id,running_job_id,created_at,updated_at FROM vibe_sites WHERE id = $1",
        ),
        "WHERE active_revision_id IS NOT NULL" => sqlx::query(
            "SELECT id,name,platform,guild_id,owner_user_id,description,status::text AS status,access::text AS access,active_revision_id,running_job_id,created_at,updated_at FROM vibe_sites WHERE active_revision_id IS NOT NULL",
        ),
        _ => unreachable!("site query suffix is internal and fixed"),
    }
}
fn revision_query(
    suffix: &str,
) -> sqlx::query::Query<'static, sqlx::Postgres, sqlx::postgres::PgArguments> {
    match suffix {
        "JOIN vibe_sites s ON s.active_revision_id=r.id WHERE s.id=$1" => sqlx::query(
            "SELECT r.id,r.site_id,r.ordinal,r.parent_revision_id,r.commit_oid,r.image_id,r.message,r.build_log,r.actor_user_id,r.conversation_id,r.turn_id,r.job_id,r.created_at FROM vibe_revisions r JOIN vibe_sites s ON s.active_revision_id=r.id WHERE s.id=$1",
        ),
        "WHERE r.site_id=$1 ORDER BY r.ordinal DESC" => sqlx::query(
            "SELECT r.id,r.site_id,r.ordinal,r.parent_revision_id,r.commit_oid,r.image_id,r.message,r.build_log,r.actor_user_id,r.conversation_id,r.turn_id,r.job_id,r.created_at FROM vibe_revisions r WHERE r.site_id=$1 ORDER BY r.ordinal DESC",
        ),
        _ => unreachable!("revision query suffix is internal and fixed"),
    }
}

fn site_from_row(row: sqlx::postgres::PgRow) -> Result<VibeSite, SqlxStorageError> {
    Ok(VibeSite {
        id: VibeSiteId(row.get("id")),
        name: row.get("name"),
        platform: PlatformName::new(row.get::<String, _>("platform")),
        guild_id: ExternalId::new(row.get::<String, _>("guild_id")),
        owner_user_id: ExternalId::new(row.get::<String, _>("owner_user_id")),
        description: row.get("description"),
        status: parse_site_status(&row.get::<String, _>("status"))?,
        access: parse_site_access(&row.get::<String, _>("access"))?,
        active_revision_id: row
            .get::<Option<Uuid>, _>("active_revision_id")
            .map(VibeRevisionId),
        running_job_id: row.get::<Option<Uuid>, _>("running_job_id").map(VibeJobId),
        created_at: row.get("created_at"),
        updated_at: row.get("updated_at"),
    })
}
fn job_from_row(row: sqlx::postgres::PgRow) -> Result<VibeJob, SqlxStorageError> {
    Ok(VibeJob {
        id: VibeJobId(row.get("id")),
        site_id: row.get::<Option<Uuid>, _>("site_id").map(VibeSiteId),
        site_name: row.get("site_name"),
        action: parse_action(&row.get::<String, _>("action"))?,
        actor_user_id: ExternalId::new(row.get::<String, _>("actor_user_id")),
        platform: PlatformName::new(row.get::<String, _>("platform")),
        guild_id: ExternalId::new(row.get::<String, _>("guild_id")),
        conversation_id: ConversationId(row.get("conversation_id")),
        turn_id: TurnId(row.get("turn_id")),
        tool_use_id: ToolUseId::new(row.get::<String, _>("tool_use_id")),
        state: parse_job_state(&row.get::<String, _>("state"))?,
        error: row.get("error"),
        created_at: row.get("created_at"),
        updated_at: row.get("updated_at"),
    })
}
fn revision_from_row(row: sqlx::postgres::PgRow) -> Result<VibeRevision, SqlxStorageError> {
    Ok(VibeRevision {
        id: VibeRevisionId(row.get("id")),
        site_id: VibeSiteId(row.get("site_id")),
        ordinal: row.get("ordinal"),
        parent_revision_id: row
            .get::<Option<Uuid>, _>("parent_revision_id")
            .map(VibeRevisionId),
        commit_oid: row.get("commit_oid"),
        image_id: row.get("image_id"),
        message: row.get("message"),
        build_log: row.get("build_log"),
        actor_user_id: ExternalId::new(row.get::<String, _>("actor_user_id")),
        conversation_id: ConversationId(row.get("conversation_id")),
        turn_id: TurnId(row.get("turn_id")),
        job_id: VibeJobId(row.get("job_id")),
        created_at: row.get("created_at"),
    })
}
fn action_text(v: VibeAction) -> &'static str {
    match v {
        VibeAction::Create => "create",
        VibeAction::Edit => "edit",
    }
}
fn site_status_text(v: VibeSiteStatus) -> &'static str {
    match v {
        VibeSiteStatus::Creating => "creating",
        VibeSiteStatus::Active => "active",
        VibeSiteStatus::Archived => "archived",
    }
}
fn job_state_text(v: VibeJobState) -> &'static str {
    match v {
        VibeJobState::Queued => "queued",
        VibeJobState::Coding => "coding",
        VibeJobState::Building => "building",
        VibeJobState::Repair => "repair",
        VibeJobState::Committing => "committing",
        VibeJobState::Done => "done",
        VibeJobState::NoChanges => "no_changes",
        VibeJobState::Failed => "failed",
        VibeJobState::Cancelled => "cancelled",
        VibeJobState::TimedOut => "timed_out",
    }
}
fn invalid(kind: &str, v: &str) -> SqlxStorageError {
    SqlxStorageError::InvalidReference(format!("invalid Vibe {kind} `{v}`"))
}
fn parse_action(v: &str) -> Result<VibeAction, SqlxStorageError> {
    match v {
        "create" => Ok(VibeAction::Create),
        "edit" => Ok(VibeAction::Edit),
        _ => Err(invalid("action", v)),
    }
}
fn parse_site_status(v: &str) -> Result<VibeSiteStatus, SqlxStorageError> {
    match v {
        "creating" => Ok(VibeSiteStatus::Creating),
        "active" => Ok(VibeSiteStatus::Active),
        "archived" => Ok(VibeSiteStatus::Archived),
        _ => Err(invalid("site status", v)),
    }
}
fn parse_site_access(v: &str) -> Result<VibeSiteAccess, SqlxStorageError> {
    match v {
        "protected" => Ok(VibeSiteAccess::Protected),
        "public" => Ok(VibeSiteAccess::Public),
        _ => Err(invalid("site access", v)),
    }
}
fn parse_job_state(v: &str) -> Result<VibeJobState, SqlxStorageError> {
    match v {
        "queued" => Ok(VibeJobState::Queued),
        "coding" => Ok(VibeJobState::Coding),
        "building" => Ok(VibeJobState::Building),
        "repair" => Ok(VibeJobState::Repair),
        "committing" => Ok(VibeJobState::Committing),
        "done" => Ok(VibeJobState::Done),
        "no_changes" => Ok(VibeJobState::NoChanges),
        "failed" => Ok(VibeJobState::Failed),
        "cancelled" => Ok(VibeJobState::Cancelled),
        "timed_out" => Ok(VibeJobState::TimedOut),
        _ => Err(invalid("job state", v)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn document(id: u128, fields: Value) -> VibeCollectionDocument {
        VibeCollectionDocument {
            id: Uuid::from_u128(id),
            fields: fields.as_object().unwrap().clone(),
            inserted_by: ExternalId::new("1"),
            inserted_at: OffsetDateTime::UNIX_EPOCH,
            updated_by: ExternalId::new("1"),
            updated_at: OffsetDateTime::UNIX_EPOCH,
        }
    }

    #[test]
    fn collection_filters_lists_and_rejects_json_columns() {
        let scalar = document(1, json!({"user":"123","score":10}));
        let json_column = document(2, json!({"user":["123"],"score":20}));
        let filters = std::collections::BTreeMap::from([("user".into(), json!(["123", "456"]))]);
        assert!(collection_document_matches(&scalar, &filters));
        assert!(!collection_document_matches(&json_column, &filters));
    }

    #[test]
    fn collection_query_orders_pages_projects_and_deduplicates() {
        let query = VibeCollectionQuery {
            limit: Some(2),
            offset: 1,
            order_by: vec![VibeCollectionOrder {
                column: "score".into(),
                direction: VibeCollectionOrderDirection::Desc,
            }],
            select: Some(vec!["team".into()]),
            distinct: true,
            ..VibeCollectionQuery::default()
        };
        let selected = select_collection_documents(
            vec![
                document(1, json!({"team":"red","score":10})),
                document(2, json!({"team":"red","score":30})),
                document(3, json!({"team":"red","score":20})),
            ],
            &query,
        );
        assert_eq!(
            project_collection_documents(selected, &query),
            vec![json!({"team":"red"})]
        );
    }

    #[test]
    fn automatic_timestamps_are_queryable_strings() {
        let value = collection_document_column(&document(1, json!({})), "inserted_at").unwrap();
        assert_eq!(value, json!("1970-01-01T00:00:00Z"));
    }
}
