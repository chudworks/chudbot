//! SQLx/Postgres storage for chudbot.
//!
//! This crate owns the database boundary for the bot runtime. It
//! intentionally uses runtime-checked SQLx queries so normal builds do not
//! require a live `DATABASE_URL`.

use chudbot_api::{ConversationId, TurnId};
use sqlx::postgres::PgPoolOptions;
use sqlx::{PgPool, Row};
use thiserror::Error;
use time::OffsetDateTime;

mod bot;
mod memory;
mod snapshots;
mod vibe;

static MIGRATOR: sqlx::migrate::Migrator = sqlx::migrate!("../../migrations");

/// Postgres-backed bot storage.
#[derive(Debug, Clone)]
pub struct SqlxStorage {
    pool: PgPool,
    app_version_id: Option<i32>,
}

/// Registered build version row.
#[derive(Debug, Clone)]
pub struct AppVersion {
    /// Human-facing ordered version number.
    pub id: i32,
    /// Full `git describe --tags --always --dirty` string.
    pub git_version: String,
    /// First time this build was seen by this database.
    pub first_seen_at: OffsetDateTime,
}

impl SqlxStorage {
    /// Connect to Postgres.
    #[tracing::instrument(name = "storage_sqlx.connect", skip_all)]
    pub async fn connect(database_url: &str) -> Result<Self, SqlxStorageError> {
        let pool = PgPoolOptions::new()
            .max_connections(10)
            .connect(database_url)
            .await?;
        tracing::info!("connected SQLx storage");
        Ok(Self {
            pool,
            app_version_id: None,
        })
    }

    /// Construct from an existing pool.
    pub fn new(pool: PgPool) -> Self {
        Self {
            pool,
            app_version_id: None,
        }
    }

    /// Stamp newly written conversations, turns, and attempts with an app
    /// version row resolved by [`Self::register_app_version`].
    pub fn with_app_version_id(mut self, app_version_id: i32) -> Self {
        self.app_version_id = Some(app_version_id);
        self
    }

    /// Borrow the underlying pool.
    pub fn pool(&self) -> &PgPool {
        &self.pool
    }

    /// Run embedded workspace-root migrations.
    #[tracing::instrument(name = "storage_sqlx.migrate", skip_all)]
    pub async fn run_migrations(&self) -> Result<(), SqlxStorageError> {
        MIGRATOR.run(&self.pool).await?;
        tracing::info!("database migrations complete");
        Ok(())
    }

    /// Resolve or insert the ordered version row for the running build.
    ///
    /// This deliberately selects before inserting instead of using an upsert:
    /// `app_versions.id` is the user-facing `vN` number, and Postgres
    /// sequences advance even when `ON CONFLICT DO NOTHING` rejects a row.
    #[tracing::instrument(
        name = "storage_sqlx.register_app_version",
        skip_all,
        fields(git_version)
    )]
    pub async fn register_app_version(
        &self,
        git_version: &str,
    ) -> Result<AppVersion, SqlxStorageError> {
        if let Some(row) = sqlx::query(
            "SELECT id, git_version, first_seen_at \
               FROM app_versions \
              WHERE git_version = $1",
        )
        .bind(git_version)
        .fetch_optional(&self.pool)
        .await?
        {
            return Ok(app_version_from_row(row));
        }

        let row = sqlx::query(
            "INSERT INTO app_versions (git_version) \
             VALUES ($1) \
             RETURNING id, git_version, first_seen_at",
        )
        .bind(git_version)
        .fetch_one(&self.pool)
        .await?;
        Ok(app_version_from_row(row))
    }
}

/// Storage errors.
#[derive(Debug, Error)]
pub enum SqlxStorageError {
    /// SQLx query failed.
    #[error("sqlx: {0}")]
    Sqlx(#[from] sqlx::Error),
    /// Migration failed.
    #[error("migration: {0}")]
    Migrate(#[from] sqlx::migrate::MigrateError),
    /// JSON encode/decode failed.
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
    /// A referenced conversation was missing.
    #[error("conversation `{conversation_id}` was not found")]
    MissingConversation {
        /// Conversation id.
        conversation_id: ConversationId,
    },
    /// A turn has no saved attempt.
    #[error("turn `{turn_id}` has no saved attempt")]
    MissingAttempt {
        /// Turn id.
        turn_id: TurnId,
    },
    /// Stored platform reference was malformed.
    #[error("invalid platform reference: {0}")]
    InvalidReference(String),
    /// Stored model step kind was malformed.
    #[error("invalid model step kind: {0}")]
    InvalidModelStepKind(String),
    /// Stored media URI was malformed.
    #[error("invalid media uri: {0}")]
    InvalidMediaUri(String),
    /// A transactional Vibe precondition failed.
    #[error("Vibe conflict: {0}")]
    VibeConflict(String),
}

fn app_version_from_row(row: sqlx::postgres::PgRow) -> AppVersion {
    AppVersion {
        id: row.get("id"),
        git_version: row.get("git_version"),
        first_seen_at: row.get("first_seen_at"),
    }
}
