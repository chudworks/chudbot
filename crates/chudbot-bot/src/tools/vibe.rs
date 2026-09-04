use chudbot_api::vibe::{
    CompleteVibeRevision, CreateVibeJob, VibeAction, VibeActor, VibeJobState, VibeRevision,
    VibeRevisionId, VibeRole, VibeSiteAccess, VibeSiteId,
};
use chudbot_vibe::{
    ExportLimits, VibeAccess, VibeCodingExecutor, VibeOperation, VibeRuntime, validate_and_export,
};
use serde::Deserialize;

use super::*;

pub(crate) const VIBE_CHECK_NAMES_TOOL: &str = "vibe_check_names";
pub(crate) const VIBE_LIST_SITES_TOOL: &str = "vibe_list_sites";
pub(crate) const VIBE_MANAGE_TOOL: &str = "vibe_manage";

#[derive(Debug, Clone)]
pub(crate) struct VibeCoderConfig {
    pub(crate) provider: ProviderName,
    pub(crate) model: ModelSpec,
    pub(crate) instructions: String,
    pub(crate) limits: AgentLimits,
}

pub(crate) struct VibeSubagent<R: BotRuntimeTypes> {
    description: String,
    coder: VibeCoderConfig,
    deps: RuntimeToolDeps<R>,
    context: RuntimeToolContext,
    runtime: VibeRuntime,
    is_admin: bool,
}

impl<R: BotRuntimeTypes> std::fmt::Debug for VibeSubagent<R> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("VibeSubagent")
            .field("coder", &self.coder)
            .finish_non_exhaustive()
    }
}

impl<R: BotRuntimeTypes> VibeSubagent<R> {
    pub(crate) fn runtime(&self) -> &VibeRuntime {
        &self.runtime
    }
    pub(crate) fn is_admin(&self) -> bool {
        self.is_admin
    }
    pub(crate) fn new(
        description: String,
        coder: VibeCoderConfig,
        deps: RuntimeToolDeps<R>,
        context: RuntimeToolContext,
        runtime: VibeRuntime,
        is_admin: bool,
    ) -> Self {
        Self {
            description,
            coder,
            deps,
            context,
            runtime,
            is_admin,
        }
    }

    pub(crate) fn spec(&self) -> ClientToolSpec {
        ClientToolSpec {
            description: self.description.clone(),
            input_schema: ToolInputSchema::object([
                ToolInputField::required(
                    "action",
                    ToolInputValueSchema::string().enum_values(["create", "edit"]),
                ),
                ToolInputField::required("siteName", ToolInputValueSchema::string()),
                ToolInputField::required("task", ToolInputValueSchema::string()),
            ]),
        }
    }

    pub(crate) async fn call(&self, call: ClientToolCall) -> ClientToolOutput {
        let request = match serde_json::from_value::<VibeToolInput>(call.input.clone()) {
            Ok(request) => request,
            Err(error) => return vibe_error("invalid_input", &error.to_string()),
        };
        let action = match request.action.as_str() {
            "create" => VibeAction::Create,
            "edit" => VibeAction::Edit,
            _ => return vibe_error("invalid_action", "action must be create or edit"),
        };
        let actor = VibeActor {
            platform: self.context.turn_user.platform.clone(),
            guild_id: self.context.turn_user.guild_id.clone(),
            user_id: self.context.turn_user.user_id.clone(),
            conversation_id: self.context.conversation_id,
            turn_id: self.context.turn_id,
            is_admin: self.is_admin,
        };
        let terminal_reason = Arc::new(AtomicU8::new(0));
        let future = self.run_job(
            action,
            request.site_name,
            request.task,
            actor,
            call.id,
            terminal_reason.clone(),
        );
        tokio::pin!(future);
        let timeout = tokio::time::sleep(Duration::from_secs(
            self.runtime.config.sandbox.job_timeout_seconds,
        ));
        tokio::pin!(timeout);
        tokio::select! {
            output=&mut future=>output,
            ()=&mut timeout=>{terminal_reason.store(1,Ordering::Release);vibe_error("timed_out","the Vibe coding job exceeded its time limit")}
        }
    }

    async fn run_job(
        &self,
        action: VibeAction,
        name: String,
        task: String,
        actor: VibeActor,
        tool_use_id: ToolUseId,
        terminal_reason: Arc<AtomicU8>,
    ) -> ClientToolOutput {
        let access = VibeAccess::new(
            self.runtime.config.enabled,
            self.runtime.config.access.clone(),
        );
        if let Err(error) = access.check_rollout(
            &actor,
            if action == VibeAction::Create {
                VibeOperation::Create
            } else {
                VibeOperation::Edit
            },
        ) {
            return vibe_error("access_denied", &error.to_string());
        }
        let Some(guild_id) = actor.guild_id.as_ref() else {
            return vibe_error(
                "dm_not_allowed",
                "Vibe sites can only be created in a server",
            );
        };
        if let Err(error) =
            chudbot_vibe::VibeNames::new(self.runtime.config.reserved_names.clone()).validate(&name)
        {
            return vibe_error("invalid_name", error);
        }
        match self
            .deps
            .platforms
            .guild_membership(&actor.platform, guild_id, &actor.user_id)
            .await
        {
            Ok(Some(_)) => {}
            Ok(None) => {
                return vibe_error(
                    "not_a_member",
                    "you must be a current member of this server",
                );
            }
            Err(error) => {
                tracing::warn!(error=%error,"Discord membership unavailable before Vibe job");
                return vibe_error(
                    "discord_unavailable",
                    "Discord membership could not be checked",
                );
            }
        }
        if let Ok(Some(existing)) = self.deps.storage.find_job_by_tool_use(&tool_use_id).await {
            return self.existing_job_output(existing).await;
        }
        let running = match self
            .deps
            .storage
            .count_running_jobs(&actor.platform, guild_id)
            .await
        {
            Ok(count) => count,
            Err(error) => {
                tracing::error!(error=%error,"Vibe site lookup failed");
                return vibe_error(
                    "storage_unavailable",
                    "Vibe storage is temporarily unavailable",
                );
            }
        };
        if running >= u64::from(self.runtime.config.limits.max_running_jobs_per_guild) {
            return vibe_error(
                "guild_busy",
                "this server already has the maximum number of Vibe jobs running",
            );
        }

        let existing_site = match self.deps.storage.find_site_by_name(&name).await {
            Ok(site) => site,
            Err(error) => return vibe_error("storage_unavailable", &error.to_string()),
        };
        let site_id = match (action, existing_site.as_ref()) {
            (VibeAction::Create, Some(_)) => {
                return vibe_error("name_unavailable", "that site name is already taken");
            }
            (VibeAction::Create, None) => VibeSiteId::new(),
            (VibeAction::Edit, None) => {
                return vibe_error("site_not_found", "that site does not exist");
            }
            (VibeAction::Edit, Some(site)) => {
                let role = if actor.is_admin {
                    VibeRole::Admin
                } else if site.owner_user_id == actor.user_id {
                    VibeRole::Owner
                } else if self
                    .deps
                    .storage
                    .is_editor(site.id, &actor.platform, &actor.user_id)
                    .await
                    .unwrap_or(false)
                {
                    VibeRole::Editor
                } else {
                    VibeRole::Member
                };
                if let Err(error) = access.check_site(&actor, site, role, VibeOperation::Edit, true)
                {
                    return vibe_error("access_denied", &error.to_string());
                }
                site.id
            }
        };
        let job_id = chudbot_api::VibeJobId::new();
        let job = match self
            .deps
            .storage
            .create_job(CreateVibeJob {
                id: job_id,
                site_id,
                site_name: name.clone(),
                description: description(&task),
                action,
                actor: actor.clone(),
                tool_use_id: tool_use_id.clone(),
                max_running_jobs_per_guild: self.runtime.config.limits.max_running_jobs_per_guild,
            })
            .await
        {
            Ok(job) => job,
            Err(error) => {
                if error.to_string().contains("guild job limit") {
                    return vibe_error(
                        "guild_busy",
                        "this server already has the maximum number of Vibe jobs running",
                    );
                }
                if let Ok(Some(job)) = self.deps.storage.find_job_by_tool_use(&tool_use_id).await {
                    return self.existing_job_output(job).await;
                }
                if self
                    .deps
                    .storage
                    .find_site_by_name(&name)
                    .await
                    .ok()
                    .flatten()
                    .is_some()
                {
                    return vibe_error(
                        "name_unavailable",
                        "that site name was claimed by another request",
                    );
                }
                tracing::error!(error=%error,"Vibe job claim failed");
                return vibe_error("job_start_failed", "the Vibe job could not be started");
            }
        };
        let mut cleanup = JobCleanup::new(
            self.deps.storage.clone(),
            self.runtime.clone(),
            job.id,
            terminal_reason,
        );
        let result = self.execute_claimed_job(&job, &task, &actor).await;
        self.cleanup_job_files(job.id).await;
        cleanup.disarm();
        match result {
            Ok(output) => output,
            Err(error) => {
                tracing::error!(job=%job.id,error=%error,"Vibe job failed");
                let _ = self
                    .deps
                    .storage
                    .update_job_state(job.id, VibeJobState::Failed, Some(&error))
                    .await;
                vibe_error(
                    "job_failed",
                    "the Vibe job failed without changing the live site",
                )
            }
        }
    }

    async fn execute_claimed_job(
        &self,
        job: &chudbot_api::VibeJob,
        task: &str,
        actor: &VibeActor,
    ) -> Result<ClientToolOutput, String> {
        let site_id = job.site_id.ok_or_else(|| "job has no site".to_string())?;
        let parent = self
            .deps
            .storage
            .active_revision(site_id)
            .await
            .map_err(|e| e.to_string())?;
        let workspace = self
            .runtime
            .disk
            .create_workspace(
                site_id,
                job.id,
                parent.as_ref().map(|r| r.commit_oid.as_str()),
            )
            .await
            .map_err(|e| e.to_string())?;
        self.post_status("Vibe is starting the coding agent.").await;
        self.deps
            .storage
            .update_job_state(job.id, VibeJobState::Coding, None)
            .await
            .map_err(|e| e.to_string())?;
        let container = self
            .runtime
            .sandbox
            .start_coding(job.id, &workspace)
            .await
            .map_err(|e| e.to_string())?;
        let executor = VibeCodingExecutor::new(workspace.clone(), container);
        let mut spec = AgentSpec::from_legacy_prompt_text(self.coder.instructions.clone())
            .with_limits(self.coder.limits);
        spec.client_tools = Some(
            VibeCodingExecutor::tool_names()
                .into_iter()
                .map(ToolName::new)
                .collect(),
        );
        spec.server_tools = Some(BTreeSet::new());
        let model = Model {
            backend: RoutedLlmBackend::new(self.deps.llms.clone(), self.coder.provider.clone()),
            spec: self.coder.model.clone(),
        };
        let agent = Agent::new(model, spec, executor);
        let mut transcript = Transcript::new();
        transcript.push(TranscriptTurn::text(TurnRole::User, task));
        let mut runs = Vec::new();
        let mut summary: String;
        let export_limits = ExportLimits {
            max_files: self.runtime.config.limits.max_source_files,
            max_file_bytes: self.runtime.config.limits.max_file_bytes,
            max_tree_bytes: self.runtime.config.limits.max_source_bytes,
        };
        let mut exported;
        let build;
        let mut repairs = 0u8;
        loop {
            let run = collect_agent_run(agent.run(transcript))
                .await
                .map_err(|e| format!("coding model failed: {e}"))?;
            summary = match &run.outcome {
                AgentOutcome::Completed { answer } => answer.text.clone(),
                AgentOutcome::Cancelled { reason } => {
                    return Err(format!("coding agent cancelled: {reason}"));
                }
                AgentOutcome::Failed { error, .. } => {
                    return Err(format!("coding agent failed: {error}"));
                }
                AgentOutcome::IterationLimit { max_iterations } => {
                    return Err(format!("coding agent reached {max_iterations} iterations"));
                }
            };
            transcript = run.transcript.clone();
            runs.push(run);
            exported = validate_and_export(
                &workspace,
                self.runtime.disk.root().join("workspaces").as_path(),
                export_limits,
            )
            .await
            .map_err(|e| e.to_string())?;
            if let Some(parent) = &parent {
                let source_tree = self
                    .runtime
                    .disk
                    .source_tree_oid(site_id, &exported.root)
                    .await
                    .map_err(|e| e.to_string())?;
                let old_tree = self
                    .runtime
                    .disk
                    .commit_tree_oid(site_id, &parent.commit_oid)
                    .await
                    .map_err(|e| e.to_string())?;
                if source_tree == old_tree {
                    let _ = self
                        .deps
                        .storage
                        .update_job_state(job.id, VibeJobState::NoChanges, None)
                        .await;
                    return Ok(vibe_success(
                        &job.site_name,
                        "no_changes",
                        &summary,
                        &self.runtime.config.base_domain,
                        &runs,
                    ));
                }
            }
            self.post_status(if repairs == 0 {
                "Vibe is running the clean build."
            } else {
                "Vibe is running the clean build again."
            })
            .await;
            self.deps
                .storage
                .update_job_state(job.id, VibeJobState::Building, None)
                .await
                .map_err(|e| e.to_string())?;
            let revision_id = VibeRevisionId::new();
            let artifact = self.runtime.disk.artifact_path(site_id, revision_id);
            match self
                .runtime
                .sandbox
                .clean_build(
                    job.id,
                    &exported.root,
                    &artifact,
                    self.runtime.config.limits.max_artifact_bytes,
                )
                .await
            {
                Ok(output) => {
                    build = (revision_id, output);
                    break;
                }
                Err(error) if repairs < self.runtime.config.sandbox.max_repair_attempts => {
                    repairs += 1;
                    let log = error.to_string().chars().take(16_000).collect::<String>();
                    self.post_status(&format!(
                        "Vibe clean build failed; repair attempt {repairs}."
                    ))
                    .await;
                    self.deps
                        .storage
                        .update_job_state(job.id, VibeJobState::Repair, Some(&log))
                        .await
                        .map_err(|e| e.to_string())?;
                    transcript.push(TranscriptTurn::text(TurnRole::User,format!("The clean build failed. Fix the source and run the project build before summarizing again. Bounded build log:\n\n{log}")));
                    let _ = tokio::fs::remove_dir_all(&exported.root).await;
                }
                Err(error) => {
                    return Err(format!("clean build failed after repair budget: {error}"));
                }
            }
        }
        self.deps
            .storage
            .update_job_state(job.id, VibeJobState::Committing, None)
            .await
            .map_err(|e| e.to_string())?;
        let message = commit_message(job.action, &job.site_name, task, &summary);
        let commit = self
            .runtime
            .disk
            .commit_tree(
                site_id,
                &exported.root,
                parent.as_ref().map(|r| r.commit_oid.as_str()),
                &actor.user_id,
                &message,
            )
            .await
            .map_err(|e| e.to_string())?;
        let revision = VibeRevision {
            id: build.0,
            site_id,
            ordinal: parent.as_ref().map_or(1, |r| r.ordinal + 1),
            parent_revision_id: parent.as_ref().map(|r| r.id),
            commit_oid: commit.oid.clone(),
            image_id: build.1.image_id,
            message: message.clone(),
            build_log: build.1.log,
            actor_user_id: actor.user_id.clone(),
            conversation_id: actor.conversation_id,
            turn_id: actor.turn_id,
            job_id: job.id,
            created_at: OffsetDateTime::now_utc(),
        };
        self.deps
            .storage
            .complete_revision(CompleteVibeRevision {
                revision,
                expected_job_id: job.id,
            })
            .await
            .map_err(|e| e.to_string())?;
        if let Err(error) = self.runtime.disk.activate_main(site_id, &commit.oid).await {
            tracing::error!(site=%job.site_name,error=%error,"Vibe main ref will be repaired on restart");
        }
        let _ = tokio::fs::remove_dir_all(exported.root).await;
        Ok(vibe_success(
            &job.site_name,
            "done",
            &summary,
            &self.runtime.config.base_domain,
            &runs,
        ))
    }

    async fn post_status(&self, text: &str) {
        let tool = PostStatusTool {
            platforms: self.deps.platforms.clone(),
            storage: self.deps.storage.clone(),
            channel: self.context.default_channel.clone(),
            reply_to: self.context.reply_to.clone(),
            conversation_id: self.context.conversation_id,
            turn_id: self.context.turn_id,
        };
        let _ = tool
            .call(ClientToolCall {
                id: ToolUseId::new(format!("vibe-status-{}", uuid::Uuid::new_v4())),
                name: ToolName::new(POST_STATUS_TOOL),
                input: serde_json::json!({"text":text}),
            })
            .await;
    }
    async fn cleanup_job_files(&self, id: chudbot_api::VibeJobId) {
        let _ = self.runtime.sandbox.cleanup_job(id).await;
        let path = self.runtime.disk.workspace_path(id);
        if tokio::fs::try_exists(&path).await.unwrap_or(false) {
            let _ = tokio::fs::remove_dir_all(path).await;
        }
    }
    async fn existing_job_output(&self, job: chudbot_api::VibeJob) -> ClientToolOutput {
        match job.state {
            VibeJobState::Done | VibeJobState::NoChanges => vibe_success(
                &job.site_name,
                job_state(job.state),
                "This retried tool call returned the existing job.",
                &self.runtime.config.base_domain,
                &[],
            ),
            VibeJobState::Failed | VibeJobState::Cancelled | VibeJobState::TimedOut => {
                vibe_error(job_state(job.state), "the existing job did not complete")
            }
            _ => vibe_error(
                "job_running",
                "this tool call already has a running Vibe job",
            ),
        }
    }
}

pub(crate) fn vibe_check_names_spec() -> ClientToolSpec {
    ClientToolSpec {
        description: "Check up to eight Vibe site name candidates without reserving them.".into(),
        input_schema: ToolInputSchema::object([ToolInputField::required(
            "names",
            ToolInputValueSchema::array(ToolInputValueSchema::string())
                .min_items(1)
                .max_items(8),
        )]),
    }
}

pub(crate) fn vibe_list_sites_spec() -> ClientToolSpec {
    ClientToolSpec { description:"List active Vibe sites in the current server, ordered for resolving references such as 'that site'.".into(), input_schema:ToolInputSchema::object([ToolInputField::optional("filter",ToolInputValueSchema::string())]) }
}

pub(crate) fn vibe_manage_spec() -> ClientToolSpec {
    ClientToolSpec {
        description: "Roll back, manage editors, change access, archive, or restore a Vibe site when the current actor has permission. Sites are 🔒 protected by default.".into(),
        input_schema: ToolInputSchema::object([
            ToolInputField::required(
                "action",
                ToolInputValueSchema::string().enum_values([
                    "rollback",
                    "add_editor",
                    "remove_editor",
                    "set_access",
                    "archive",
                    "restore",
                ]),
            ),
            ToolInputField::required("siteName", ToolInputValueSchema::string()),
            ToolInputField::optional("revision", ToolInputValueSchema::integer().minimum(1)),
            ToolInputField::optional("userId", ToolInputValueSchema::string()),
            ToolInputField::optional(
                "accessLevel",
                ToolInputValueSchema::string().enum_values(["protected", "public"]),
            ),
        ]),
    }
}

impl<R: BotRuntimeTypes> RuntimeToolExecutor<R> {
    pub(crate) async fn vibe_check_names(
        &self,
        call: ClientToolCall,
    ) -> Result<ClientToolOutput, ClientToolExecutorError<RuntimeToolError>> {
        let Some(runtime) = self.vibe.as_ref() else {
            return Err(ClientToolExecutorError::unknown(call.name));
        };
        let actor = VibeActor {
            platform: self.context.turn_user.platform.clone(),
            guild_id: self.context.turn_user.guild_id.clone(),
            user_id: self.context.turn_user.user_id.clone(),
            conversation_id: self.context.conversation_id,
            turn_id: self.context.turn_id,
            is_admin: self.vibe_actor_is_admin,
        };
        if VibeAccess::new(runtime.config.enabled, runtime.config.access.clone())
            .check_rollout(&actor, VibeOperation::View)
            .is_err()
        {
            return Ok(vibe_error(
                "access_denied",
                "Vibe is unavailable in this server",
            ));
        }
        let Some(guild) = actor.guild_id.as_ref() else {
            return Ok(vibe_error("dm_not_allowed", "Vibe sites belong to servers"));
        };
        match self
            .deps
            .platforms
            .guild_membership(&actor.platform, guild, &actor.user_id)
            .await
        {
            Ok(Some(_)) => {}
            Ok(None) => return Ok(vibe_error("not_a_member", "you must be a current member")),
            Err(_) => {
                return Ok(vibe_error(
                    "discord_unavailable",
                    "Discord membership could not be checked",
                ));
            }
        }
        let Some(names) = call
            .input
            .get("names")
            .and_then(serde_json::Value::as_array)
        else {
            return Ok(vibe_error("invalid_input", "names must be an array"));
        };
        if names.is_empty() || names.len() > 8 {
            return Ok(vibe_error(
                "invalid_input",
                "provide between one and eight names",
            ));
        }
        let validator = chudbot_vibe::VibeNames::new(runtime.config.reserved_names.clone());
        let mut results = Vec::with_capacity(names.len());
        for value in names {
            let Some(name) = value.as_str() else {
                return Ok(vibe_error("invalid_input", "every name must be a string"));
            };
            let status = if validator.validate(name).is_err() {
                "invalid"
            } else {
                match self.deps.storage.find_site_by_name(name).await {
                    Ok(Some(_)) => "unavailable",
                    Ok(None) => "available",
                    Err(error) => {
                        tracing::error!(error=%error,"Vibe name lookup failed");
                        return Ok(vibe_error(
                            "storage_unavailable",
                            "Vibe storage is temporarily unavailable",
                        ));
                    }
                }
            };
            results.push(serde_json::json!({"name":name,"status":status}));
        }
        let value = serde_json::json!({"names":results});
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

    pub(crate) async fn vibe_list_sites(
        &self,
        call: ClientToolCall,
    ) -> Result<ClientToolOutput, ClientToolExecutorError<RuntimeToolError>> {
        let Some(runtime) = self.vibe.as_ref() else {
            return Err(ClientToolExecutorError::unknown(call.name));
        };
        let actor = VibeActor {
            platform: self.context.turn_user.platform.clone(),
            guild_id: self.context.turn_user.guild_id.clone(),
            user_id: self.context.turn_user.user_id.clone(),
            conversation_id: self.context.conversation_id,
            turn_id: self.context.turn_id,
            is_admin: self.vibe_actor_is_admin,
        };
        if VibeAccess::new(runtime.config.enabled, runtime.config.access.clone())
            .check_rollout(&actor, VibeOperation::View)
            .is_err()
        {
            return Ok(vibe_error(
                "access_denied",
                "Vibe is unavailable in this server",
            ));
        }
        let Some(guild) = actor.guild_id.as_ref() else {
            return Ok(vibe_error("dm_not_allowed", "Vibe sites belong to servers"));
        };
        match self
            .deps
            .platforms
            .guild_membership(&actor.platform, guild, &actor.user_id)
            .await
        {
            Ok(Some(_)) => {}
            Ok(None) => return Ok(vibe_error("not_a_member", "you must be a current member")),
            Err(_) => {
                return Ok(vibe_error(
                    "discord_unavailable",
                    "Discord membership could not be checked",
                ));
            }
        }
        let filter = call.input.get("filter").and_then(serde_json::Value::as_str);
        let sites = self
            .deps
            .storage
            .list_sites(&actor, filter)
            .await
            .map_err(|error| {
                tracing::error!(error=%error,"Vibe site list failed");
                ClientToolExecutorError::execution(RuntimeToolError(
                    "Vibe storage is temporarily unavailable".into(),
                ))
            })?;
        let sites=sites.into_iter().map(|(site,role)|serde_json::json!({"name":site.name,"description":site.description,"role":format!("{role:?}").to_ascii_lowercase(),"accessLevel":site.access.as_str(),"accessLabel":site.access.label(),"siteUrl":format!("https://{}.{}",site.name,runtime.config.base_domain),"sourceUrl":format!("https://src.{}/sites/{}",runtime.config.base_domain,site.name),"lastChanged":site.updated_at})).collect::<Vec<_>>();
        let value = serde_json::json!({"sites":sites});
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

    pub(crate) async fn vibe_manage(
        &self,
        call: ClientToolCall,
    ) -> Result<ClientToolOutput, ClientToolExecutorError<RuntimeToolError>> {
        let Some(runtime) = self.vibe.as_ref() else {
            return Err(ClientToolExecutorError::unknown(call.name));
        };
        let Some(action) = call.input.get("action").and_then(serde_json::Value::as_str) else {
            return Ok(vibe_error("invalid_input", "action is required"));
        };
        let Some(name) = call
            .input
            .get("siteName")
            .and_then(serde_json::Value::as_str)
        else {
            return Ok(vibe_error("invalid_input", "siteName is required"));
        };
        let actor = VibeActor {
            platform: self.context.turn_user.platform.clone(),
            guild_id: self.context.turn_user.guild_id.clone(),
            user_id: self.context.turn_user.user_id.clone(),
            conversation_id: self.context.conversation_id,
            turn_id: self.context.turn_id,
            is_admin: self.vibe_actor_is_admin,
        };
        let Some(guild) = actor.guild_id.as_ref() else {
            return Ok(vibe_error(
                "dm_not_allowed",
                "Vibe management requires a server",
            ));
        };
        let site = match self.deps.storage.find_site_by_name(name).await {
            Ok(Some(site)) => site,
            Ok(None) => return Ok(vibe_error("site_not_found", "that site does not exist")),
            Err(error) => {
                tracing::error!(error=%error,"Vibe management site lookup failed");
                return Ok(vibe_error(
                    "storage_unavailable",
                    "Vibe storage is temporarily unavailable",
                ));
            }
        };
        if site.running_job_id.is_some() {
            return Ok(vibe_error(
                "site_busy",
                "this site has a coding job in progress",
            ));
        }
        match self
            .deps
            .platforms
            .guild_membership(&actor.platform, guild, &actor.user_id)
            .await
        {
            Ok(Some(_)) => {}
            Ok(None) => return Ok(vibe_error("not_a_member", "you must be a current member")),
            Err(_) => {
                return Ok(vibe_error(
                    "discord_unavailable",
                    "Discord membership could not be checked",
                ));
            }
        }
        let role = if actor.is_admin {
            VibeRole::Admin
        } else if site.owner_user_id == actor.user_id {
            VibeRole::Owner
        } else if self
            .deps
            .storage
            .is_editor(site.id, &actor.platform, &actor.user_id)
            .await
            .unwrap_or(false)
        {
            VibeRole::Editor
        } else {
            VibeRole::Member
        };
        let operation = if action == "rollback" {
            VibeOperation::Edit
        } else {
            VibeOperation::Manage
        };
        if let Err(error) = VibeAccess::new(runtime.config.enabled, runtime.config.access.clone())
            .check_site(&actor, &site, role, operation, true)
        {
            return Ok(vibe_error("access_denied", &error.to_string()));
        }
        let result = match action {
            "rollback" => {
                let revisions =
                    self.deps
                        .storage
                        .list_revisions(site.id)
                        .await
                        .map_err(|error| {
                            ClientToolExecutorError::execution(RuntimeToolError(error.to_string()))
                        })?;
                let active = self
                    .deps
                    .storage
                    .active_revision(site.id)
                    .await
                    .map_err(|error| {
                        ClientToolExecutorError::execution(RuntimeToolError(error.to_string()))
                    })?;
                let wanted = call
                    .input
                    .get("revision")
                    .and_then(serde_json::Value::as_i64)
                    .and_then(|value| i32::try_from(value).ok())
                    .or_else(|| active.as_ref().map(|revision| revision.ordinal - 1));
                let Some(revision) = wanted.and_then(|ordinal| {
                    revisions
                        .iter()
                        .find(|revision| revision.ordinal == ordinal)
                }) else {
                    return Ok(vibe_error(
                        "revision_not_found",
                        "the requested rollback revision does not exist",
                    ));
                };
                self.deps
                    .storage
                    .activate_existing_revision(site.id, revision.id)
                    .await
                    .map_err(|error| {
                        ClientToolExecutorError::execution(RuntimeToolError(error.to_string()))
                    })?;
                if let Err(error) = runtime
                    .disk
                    .activate_main(site.id, &revision.commit_oid)
                    .await
                {
                    tracing::error!(site=name,error=%error,"rollback main ref will be repaired on restart");
                }
                serde_json::json!({"action":"rollback","revision":revision.ordinal})
            }
            "add_editor" => {
                let Some(user) = call.input.get("userId").and_then(serde_json::Value::as_str)
                else {
                    return Ok(vibe_error("invalid_input", "userId is required"));
                };
                let user = ExternalId::new(user);
                match self
                    .deps
                    .platforms
                    .guild_membership(&actor.platform, guild, &user)
                    .await
                {
                    Ok(Some(_)) => {}
                    Ok(None) => {
                        return Ok(vibe_error(
                            "not_a_member",
                            "the editor must be a current server member",
                        ));
                    }
                    Err(_) => {
                        return Ok(vibe_error(
                            "discord_unavailable",
                            "Discord membership could not be checked",
                        ));
                    }
                }
                self.deps
                    .storage
                    .add_editor(site.id, &actor.platform, &user, &actor.user_id)
                    .await
                    .map_err(|error| {
                        ClientToolExecutorError::execution(RuntimeToolError(error.to_string()))
                    })?;
                serde_json::json!({"action":"add_editor","userId":user})
            }
            "remove_editor" => {
                let Some(user) = call.input.get("userId").and_then(serde_json::Value::as_str)
                else {
                    return Ok(vibe_error("invalid_input", "userId is required"));
                };
                let user = ExternalId::new(user);
                let removed = self
                    .deps
                    .storage
                    .remove_editor(site.id, &actor.platform, &user)
                    .await
                    .map_err(|error| {
                        ClientToolExecutorError::execution(RuntimeToolError(error.to_string()))
                    })?;
                serde_json::json!({"action":"remove_editor","userId":user,"removed":removed})
            }
            "set_access" => {
                let access = match call
                    .input
                    .get("accessLevel")
                    .and_then(serde_json::Value::as_str)
                {
                    Some("protected") => VibeSiteAccess::Protected,
                    Some("public") => VibeSiteAccess::Public,
                    _ => {
                        return Ok(vibe_error(
                            "invalid_input",
                            "accessLevel must be protected or public",
                        ));
                    }
                };
                self.deps
                    .storage
                    .set_site_access(site.id, access)
                    .await
                    .map_err(|error| {
                        ClientToolExecutorError::execution(RuntimeToolError(error.to_string()))
                    })?;
                serde_json::json!({"action":"set_access","accessLevel":access.as_str(),"accessLabel":access.label()})
            }
            "archive" => {
                self.deps
                    .storage
                    .set_site_status(site.id, chudbot_api::VibeSiteStatus::Archived)
                    .await
                    .map_err(|error| {
                        ClientToolExecutorError::execution(RuntimeToolError(error.to_string()))
                    })?;
                serde_json::json!({"action":"archive"})
            }
            "restore" => {
                if site.active_revision_id.is_none() {
                    return Ok(vibe_error(
                        "no_revision",
                        "a site without a successful revision cannot be restored",
                    ));
                }
                self.deps
                    .storage
                    .set_site_status(site.id, chudbot_api::VibeSiteStatus::Active)
                    .await
                    .map_err(|error| {
                        ClientToolExecutorError::execution(RuntimeToolError(error.to_string()))
                    })?;
                serde_json::json!({"action":"restore"})
            }
            _ => {
                return Ok(vibe_error(
                    "invalid_action",
                    "unknown Vibe management action",
                ));
            }
        };
        let value = serde_json::json!({"ok":true,"siteName":name,"result":result,"siteUrl":format!("https://{name}.{}/",runtime.config.base_domain),"sourceUrl":format!("https://src.{}/sites/{name}",runtime.config.base_domain)});
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

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct VibeToolInput {
    action: String,
    site_name: String,
    task: String,
}
fn description(task: &str) -> String {
    task.lines()
        .next()
        .unwrap_or_default()
        .chars()
        .take(240)
        .collect()
}
fn commit_message(action: VibeAction, name: &str, task: &str, summary: &str) -> String {
    format!(
        "{} {name}\n\n{}\n\n{}",
        if action == VibeAction::Create {
            "Create"
        } else {
            "Update"
        },
        task.chars().take(4000).collect::<String>(),
        summary.chars().take(2000).collect::<String>()
    )
}
fn job_state(state: VibeJobState) -> &'static str {
    match state {
        VibeJobState::Done => "done",
        VibeJobState::NoChanges => "no_changes",
        VibeJobState::Failed => "failed",
        VibeJobState::Cancelled => "cancelled",
        VibeJobState::TimedOut => "timed_out",
        _ => "running",
    }
}
fn vibe_error(code: &str, message: &str) -> ClientToolOutput {
    let value = serde_json::json!({"error":{"code":code,"message":message}});
    ClientToolOutput {
        result: ClientToolResultContent::Json {
            value: value.clone(),
        },
        media: Vec::new(),
        is_error: true,
        trace_response: value,
        usage: Vec::new(),
    }
}
fn vibe_success(
    name: &str,
    state: &str,
    summary: &str,
    base: &str,
    runs: &[AgentRun],
) -> ClientToolOutput {
    let coding_runs = runs
        .iter()
        .map(|run| {
            serde_json::json!({
                "summary": subagent_trace_response(run),
                "toolTrace": run.trace,
                "modelSteps": run.model_steps,
            })
        })
        .collect::<Vec<_>>();
    let value = serde_json::json!({"state":state,"siteName":name,"summary":summary,"siteUrl":format!("https://{name}.{base}/"),"sourceUrl":format!("https://src.{base}/sites/{name}"),"codingRuns":coding_runs});
    ClientToolOutput {
        result: ClientToolResultContent::Json {
            value: value.clone(),
        },
        media: Vec::new(),
        is_error: false,
        trace_response: value,
        usage: runs.iter().flat_map(AgentRun::all_usage).collect(),
    }
}

struct JobCleanup<S: VibeStorage + Clone + 'static> {
    storage: S,
    runtime: VibeRuntime,
    job: Option<chudbot_api::VibeJobId>,
    terminal_reason: Arc<AtomicU8>,
}
impl<S: VibeStorage + Clone + 'static> JobCleanup<S> {
    fn new(
        storage: S,
        runtime: VibeRuntime,
        job: chudbot_api::VibeJobId,
        terminal_reason: Arc<AtomicU8>,
    ) -> Self {
        Self {
            storage,
            runtime,
            job: Some(job),
            terminal_reason,
        }
    }
    fn disarm(&mut self) {
        self.job = None;
    }
}
impl<S: VibeStorage + Clone + 'static> Drop for JobCleanup<S> {
    fn drop(&mut self) {
        let Some(job) = self.job.take() else { return };
        let storage = self.storage.clone();
        let runtime = self.runtime.clone();
        let timed_out = self.terminal_reason.load(Ordering::Acquire) == 1;
        if let Ok(handle) = tokio::runtime::Handle::try_current() {
            handle.spawn(async move {
                let _ = runtime.sandbox.cleanup_job(job).await;
                let path = runtime.disk.workspace_path(job);
                let _ = tokio::fs::remove_dir_all(path).await;
                let (state, message) = if timed_out {
                    (VibeJobState::TimedOut, "job timed out")
                } else {
                    (VibeJobState::Cancelled, "turn cancelled")
                };
                let _ = storage.update_job_state(job, state, Some(message)).await;
            });
        }
    }
}
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};
