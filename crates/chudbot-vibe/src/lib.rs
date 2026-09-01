//! Vibe website names, access policy, source/artifact storage, and sandboxing.

pub mod access;
pub mod coding;
pub mod config;
pub mod export;
pub mod git;
pub mod names;
pub mod sandbox;

pub use access::{VibeAccess, VibeAccessError, VibeOperation};
pub use coding::VibeCodingExecutor;
pub use config::{VibeConfig, VibeLimitsConfig, VibeSandboxConfig};
pub use export::{ExportLimits, ExportedTree, validate_and_export};
pub use git::{GitCommit, VibeDiskStore};
pub use names::{NameAvailability, NameCheck, VibeNames};
pub use sandbox::{BuildOutput, CodingContainer, SandboxCommandOutput, VibeSandbox};

/// Cloneable production services shared by Vibe jobs.
#[derive(Debug, Clone)]
pub struct VibeRuntime {
    pub config: VibeConfig,
    pub disk: VibeDiskStore,
    pub sandbox: VibeSandbox,
}

impl VibeRuntime {
    pub fn new(config: VibeConfig, disk: VibeDiskStore) -> Self {
        let sandbox = VibeSandbox::new(config.sandbox.clone());
        Self {
            config,
            disk,
            sandbox,
        }
    }
}

use thiserror::Error;

/// Errors returned by local Vibe services.
#[derive(Debug, Error)]
pub enum VibeError {
    #[error(transparent)]
    Access(#[from] VibeAccessError),
    #[error("filesystem operation failed: {0}")]
    Io(#[from] std::io::Error),
    #[error("invalid source tree: {0}")]
    InvalidSource(String),
    #[error("git operation `{operation}` failed: {message}")]
    Git {
        operation: &'static str,
        message: String,
    },
    #[error("sandbox operation `{operation}` failed: {message}")]
    Sandbox {
        operation: &'static str,
        message: String,
    },
    #[error("sandbox operation timed out")]
    TimedOut,
}
