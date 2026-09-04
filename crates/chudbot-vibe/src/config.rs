use std::path::PathBuf;

use chudbot_api::{ExternalId, PlatformName};
use serde::{Deserialize, Serialize};

/// One platform guild admitted by the rollout allowlist.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VibeAllowedGuild {
    pub platform: PlatformName,
    /// Kept as a string in TOML and parsed by the platform adapter at startup.
    pub guild_id: ExternalId,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct VibeAccessConfig {
    #[serde(default)]
    pub admins_only: bool,
    #[serde(default)]
    pub allowed_guilds: Vec<VibeAllowedGuild>,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct VibeAuthConfig {
    pub client_id: String,
    pub client_secret: String,
    #[serde(default = "default_session_days")]
    pub session_days: u16,
}

impl std::fmt::Debug for VibeAuthConfig {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("VibeAuthConfig")
            .field("client_id", &self.client_id)
            .field("client_secret", &"[REDACTED]")
            .field("session_days", &self.session_days)
            .finish()
    }
}

fn default_session_days() -> u16 {
    7
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VibeSandboxConfig {
    #[serde(default = "default_docker_socket")]
    pub docker_socket: PathBuf,
    #[serde(default = "default_image")]
    pub image: String,
    #[serde(default = "default_job_timeout")]
    pub job_timeout_seconds: u64,
    #[serde(default = "default_command_timeout")]
    pub command_timeout_seconds: u64,
    #[serde(default = "default_build_timeout")]
    pub build_timeout_seconds: u64,
    #[serde(default = "default_repairs")]
    pub max_repair_attempts: u8,
    #[serde(default = "default_memory")]
    pub memory_mebibytes: u32,
    #[serde(default = "default_cpus")]
    pub cpus: u16,
    #[serde(default = "default_pids")]
    pub pids: u32,
}

impl Default for VibeSandboxConfig {
    fn default() -> Self {
        Self {
            docker_socket: default_docker_socket(),
            image: default_image(),
            job_timeout_seconds: default_job_timeout(),
            command_timeout_seconds: default_command_timeout(),
            build_timeout_seconds: default_build_timeout(),
            max_repair_attempts: default_repairs(),
            memory_mebibytes: default_memory(),
            cpus: default_cpus(),
            pids: default_pids(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VibeLimitsConfig {
    #[serde(default = "default_source_files")]
    pub max_source_files: usize,
    #[serde(default = "default_file_bytes")]
    pub max_file_bytes: u64,
    #[serde(default = "default_tree_bytes")]
    pub max_source_bytes: u64,
    #[serde(default = "default_tree_bytes")]
    pub max_artifact_bytes: u64,
    #[serde(default = "default_guild_jobs")]
    pub max_running_jobs_per_guild: u16,
    #[serde(default = "default_rooms_per_site")]
    pub max_rooms_per_site: u16,
    #[serde(default = "default_room_connections")]
    pub max_room_connections: u16,
    #[serde(default = "default_site_room_connections")]
    pub max_room_connections_per_site: u16,
    #[serde(default = "default_total_room_connections")]
    pub max_room_connections_total: usize,
    #[serde(default = "default_room_user_states")]
    pub max_room_user_states: u16,
    #[serde(default = "default_room_user_state_bytes")]
    pub max_room_user_state_bytes: usize,
    #[serde(default = "default_room_message_bytes")]
    pub max_room_message_bytes: usize,
    #[serde(default = "default_room_outbound_queue")]
    pub room_outbound_queue: u16,
    #[serde(default = "default_room_commands_per_second")]
    pub max_room_commands_per_second: u16,
}

impl Default for VibeLimitsConfig {
    fn default() -> Self {
        Self {
            max_source_files: default_source_files(),
            max_file_bytes: default_file_bytes(),
            max_source_bytes: default_tree_bytes(),
            max_artifact_bytes: default_tree_bytes(),
            max_running_jobs_per_guild: default_guild_jobs(),
            max_rooms_per_site: default_rooms_per_site(),
            max_room_connections: default_room_connections(),
            max_room_connections_per_site: default_site_room_connections(),
            max_room_connections_total: default_total_room_connections(),
            max_room_user_states: default_room_user_states(),
            max_room_user_state_bytes: default_room_user_state_bytes(),
            max_room_message_bytes: default_room_message_bytes(),
            room_outbound_queue: default_room_outbound_queue(),
            max_room_commands_per_second: default_room_commands_per_second(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VibeConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_base_domain")]
    pub base_domain: String,
    #[serde(default = "default_root_dir")]
    pub root_dir: PathBuf,
    #[serde(default)]
    pub reserved_names: Vec<String>,
    #[serde(default)]
    pub access: VibeAccessConfig,
    pub auth: VibeAuthConfig,
    #[serde(default)]
    pub sandbox: VibeSandboxConfig,
    #[serde(default)]
    pub limits: VibeLimitsConfig,
}

fn default_docker_socket() -> PathBuf {
    PathBuf::from("/var/run/docker.sock")
}
fn default_image() -> String {
    "chudbot-vibe-sandbox:latest".to_string()
}
fn default_job_timeout() -> u64 {
    900
}
fn default_command_timeout() -> u64 {
    60
}
fn default_build_timeout() -> u64 {
    300
}
fn default_repairs() -> u8 {
    2
}
fn default_memory() -> u32 {
    1024
}
fn default_cpus() -> u16 {
    2
}
fn default_pids() -> u32 {
    256
}
fn default_source_files() -> usize {
    2_000
}
fn default_file_bytes() -> u64 {
    10 * 1024 * 1024
}
fn default_tree_bytes() -> u64 {
    50 * 1024 * 1024
}
fn default_guild_jobs() -> u16 {
    2
}
fn default_rooms_per_site() -> u16 {
    32
}
fn default_room_connections() -> u16 {
    64
}
fn default_site_room_connections() -> u16 {
    256
}
fn default_total_room_connections() -> usize {
    1_024
}
fn default_room_user_states() -> u16 {
    16
}
fn default_room_user_state_bytes() -> usize {
    64 * 1024
}
fn default_room_message_bytes() -> usize {
    32 * 1024
}
fn default_room_outbound_queue() -> u16 {
    64
}
fn default_room_commands_per_second() -> u16 {
    32
}
fn default_base_domain() -> String {
    "vibe.example".to_string()
}
fn default_root_dir() -> PathBuf {
    PathBuf::from("vibe")
}
