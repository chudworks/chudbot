//! Persistent `generate_video` client tool.
//!
//! This module owns the client-tool boundary for async video generation. It
//! validates model JSON into a provider-neutral request, delegates submission,
//! polling, and download to the configured video provider route, persists
//! provider job state, stores completed bytes in the media store, and returns
//! separate JSON shapes for the model transcript and durable trace.

use super::*;

/// Runtime client tool for configured video generation.
///
/// `generator` is already routed to the agent's configured provider and default
/// model. This tool owns the rest of the lifecycle: input parsing, quota
/// enforcement, job persistence, polling, media storage, and result shaping.
#[derive(Debug, Clone)]
pub(crate) struct PersistentVideoGeneratorTool<G, M, S> {
    /// Provider route selected by the agent's video-generation binding.
    pub(crate) generator: G,
    /// Media store where completed video bytes are imported.
    pub(crate) media_store: M,
    /// Bot storage used for provider job rows and active-count queries.
    pub(crate) storage: S,
    /// Runtime-shared locks that close same-scope parallel submit races.
    pub(crate) rate_limit_locks: VideoRateLimitLocks,
    /// Turn that requested the provider job.
    pub(crate) turn_id: TurnId,
    /// User and platform scope used for quota checks.
    pub(crate) turn_user: UserRef,
    /// Provider name stored with job rows and status updates.
    pub(crate) provider: ProviderName,
    /// Optional active-job quota for this agent binding.
    pub(crate) rate_limit: Option<VideoGenerationRateLimit>,
    /// Model-facing tool description advertised in the tool spec.
    pub(crate) description: String,
    /// Delay between provider job polls.
    pub(crate) poll_interval: Duration,
    /// Maximum number of provider polls before the tool returns a timeout.
    pub(crate) max_polls: u32,
}

impl<G, M, S> PersistentVideoGeneratorTool<G, M, S> {
    /// Build a video tool from a configured provider route and turn context.
    pub(crate) fn new(
        generator: G,
        media_store: M,
        storage: S,
        rate_limit_locks: VideoRateLimitLocks,
        turn_id: TurnId,
        turn_user: UserRef,
        provider: ProviderName,
        rate_limit: Option<VideoGenerationRateLimit>,
    ) -> Self {
        Self {
            generator,
            media_store,
            storage,
            rate_limit_locks,
            turn_id,
            turn_user,
            provider,
            rate_limit,
            description: "Generate a video, save it to media storage, and return its media URI."
                .to_string(),
            poll_interval: Duration::from_secs(2),
            max_polls: 600,
        }
    }

    /// Override the model-facing tool description for a specific binding.
    pub(crate) fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = description.into();
        self
    }
}

impl<G, M, S> PersistentVideoGeneratorTool<G, M, S>
where
    G: VideoGenerator,
    M: MediaStore,
    S: PersistentVideoStorage,
{
    /// Return the model-facing tool declaration.
    ///
    /// Runtime parsing repeats the constraints that affect behavior because
    /// provider schema enforcement can vary.
    pub(crate) fn spec(&self) -> ClientToolSpec {
        ClientToolSpec {
            description: self.description.clone(),
            input_schema: video_tool_schema(),
        }
    }

    #[tracing::instrument(
        name = "tool.generate_video",
        skip_all,
        fields(
            turn = %self.turn_id,
            provider = %self.provider,
            user = %self.turn_user.user_id,
            scope = ?self.turn_user.guild_id.as_ref().map(ExternalId::as_str),
            tool_call = %call.id
        )
    )]
    pub(crate) async fn call(
        &self,
        call: ClientToolCall,
    ) -> Result<ClientToolOutput, BotToolError> {
        // Validate JSON and resolve optional input media before touching quota
        // or provider state; malformed calls should not consume capacity.
        let request = video_request_from_tool_input(&self.media_store, call.input).await?;
        let prompt = request.prompt.clone();
        let job_id = if let Some(rate_limit) = &self.rate_limit
            && !rate_limit.bypasses(&self.turn_user)
        {
            // The in-process lock keeps the active-job count and provider submit
            // atomic for this single-node runtime.
            let _guard = self.rate_limit_locks.lock(&self.turn_user).await;
            self.enforce_video_rate_limit(rate_limit).await?;
            self.submit_and_persist_video_job(request, prompt).await?
        } else {
            self.submit_and_persist_video_job(request, prompt).await?
        };

        for poll in 0..self.max_polls {
            // Providers expose video as an async job, but the model tool call
            // should return only after the output media is available or terminal.
            match self
                .generator
                .check_video(job_id.clone())
                .await
                .map_err(|error| BotToolError::Generator(error.to_string()))?
            {
                VideoJobStatus::Pending => {
                    if poll + 1 < self.max_polls {
                        tokio::time::sleep(self.poll_interval).await;
                    }
                }
                VideoJobStatus::Done { meta } => {
                    // Import the finished bytes before marking the job done so
                    // persisted output URIs always point at servable media.
                    let bytes = self
                        .generator
                        .download_video(meta.url.clone())
                        .await
                        .map_err(|error| BotToolError::Generator(error.to_string()))?;
                    let media = self
                        .media_store
                        .create_media(CreateMedia {
                            category: MediaCategory::Video,
                            bytes,
                            mime_type: None,
                            name: None,
                            extension: None,
                        })
                        .await
                        .map_err(|error| BotToolError::Media(error.to_string()))?;
                    self.storage
                        .update_video_job(UpdateVideoJob {
                            provider: self.provider.clone(),
                            provider_job_id: job_id.as_str().to_string(),
                            status: "done".to_string(),
                            output_uri: Some(media.uri().clone()),
                            error: None,
                        })
                        .await
                        .map_err(|error| BotToolError::Storage(error.to_string()))?;
                    let public_url = media.public_url().await.ok();
                    // Trace keeps provider/download details for debugging; the
                    // model result omits direct download URLs and focuses on
                    // the stored media reference and delivery instruction.
                    let trace_response = media_tool_trace_json(
                        media.as_ref(),
                        public_url.as_ref().map(|url| url.as_str()),
                        serde_json::json!({
                            "provider_job_id": job_id.as_str(),
                            "download_url": meta.url,
                            "duration_seconds": meta.duration_seconds,
                        }),
                    );
                    let result = media_tool_model_result_json(
                        media.as_ref(),
                        serde_json::json!({
                            "provider_job_id": job_id.as_str(),
                            "duration_seconds": meta.duration_seconds,
                        }),
                    );
                    tracing::info!(job = %job_id, uri = %media.uri(), "video job completed");
                    // `media` stays empty because final platform delivery
                    // reloads generated media from successful tool traces.
                    return Ok(ClientToolOutput {
                        result: ClientToolResultContent::Json {
                            value: result.clone(),
                        },
                        media: Vec::new(),
                        is_error: false,
                        trace_response,
                        usage: meta.usage,
                    });
                }
                VideoJobStatus::Failed { message } => {
                    self.storage
                        .update_video_job(UpdateVideoJob {
                            provider: self.provider.clone(),
                            provider_job_id: job_id.as_str().to_string(),
                            status: "failed".to_string(),
                            output_uri: None,
                            error: Some(message.clone()),
                        })
                        .await
                        .map_err(|error| BotToolError::Storage(error.to_string()))?;
                    return Err(BotToolError::Generator(format!(
                        "video generation failed: {message}"
                    )));
                }
                VideoJobStatus::Expired => {
                    self.storage
                        .update_video_job(UpdateVideoJob {
                            provider: self.provider.clone(),
                            provider_job_id: job_id.as_str().to_string(),
                            status: "expired".to_string(),
                            output_uri: None,
                            error: Some("expired".to_string()),
                        })
                        .await
                        .map_err(|error| BotToolError::Storage(error.to_string()))?;
                    return Err(BotToolError::Generator(
                        "video generation job expired".to_string(),
                    ));
                }
            }
        }

        // The upstream job may still finish later. Keep the status pending but
        // store the timeout message so the trace explains why this tool ended.
        let message = format!(
            "video generation still pending after {} polls: {}",
            self.max_polls, job_id
        );
        self.storage
            .update_video_job(UpdateVideoJob {
                provider: self.provider.clone(),
                provider_job_id: job_id.as_str().to_string(),
                status: "pending".to_string(),
                output_uri: None,
                error: Some(message.clone()),
            })
            .await
            .map_err(|error| BotToolError::Storage(error.to_string()))?;
        Err(BotToolError::Generator(message))
    }
}

/// Storage operations needed by the video-generation tool.
///
/// The trait mirrors the `BotStorage` subset used here so tests can exercise
/// persistence and rate-limit behavior without depending on the full storage
/// surface.
pub(crate) trait PersistentVideoStorage: Clone + Send + Sync {
    /// Storage backend error type.
    type Error: std::error::Error + Send + Sync + 'static;

    /// Insert the local row for a provider job after upstream submission.
    fn create_video_job(
        &self,
        input: CreateVideoJob,
    ) -> impl Future<Output = Result<StoredVideoJob, Self::Error>> + Send;

    /// Record a provider job status transition and optional output URI/error.
    fn update_video_job(
        &self,
        input: UpdateVideoJob,
    ) -> impl Future<Output = Result<(), Self::Error>> + Send;

    /// Count pending plus completed video jobs inside the rolling quota window.
    fn count_active_video_generations(
        &self,
        input: CountActiveVideoGenerations,
    ) -> impl Future<Output = Result<u64, Self::Error>> + Send;
}

impl<T> PersistentVideoStorage for T
where
    T: BotStorage + Clone + Send + Sync,
{
    type Error = T::Error;

    fn create_video_job(
        &self,
        input: CreateVideoJob,
    ) -> impl Future<Output = Result<StoredVideoJob, Self::Error>> + Send {
        BotStorage::create_video_job(self, input)
    }

    fn update_video_job(
        &self,
        input: UpdateVideoJob,
    ) -> impl Future<Output = Result<(), Self::Error>> + Send {
        BotStorage::update_video_job(self, input)
    }

    fn count_active_video_generations(
        &self,
        input: CountActiveVideoGenerations,
    ) -> impl Future<Output = Result<u64, Self::Error>> + Send {
        BotStorage::count_active_video_generations(self, input)
    }
}

/// Single-node rate-limit lock scope.
///
/// Video quotas are scoped to the platform workspace/server when one exists.
/// Platform-only scopes use `None`, which still serializes all calls on that
/// platform together.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct VideoRateLimitLockKey {
    /// Messaging platform that owns the video-generation scope.
    pub(crate) platform: PlatformName,
    /// Optional platform workspace/server/guild id.
    pub(crate) scope_id: Option<ExternalId>,
}

impl VideoRateLimitLockKey {
    /// Build the quota key for the user that issued the current turn.
    pub(crate) fn from_user(user: &UserRef) -> Self {
        Self {
            platform: user.platform.clone(),
            scope_id: user.guild_id.clone(),
        }
    }
}

/// Shared per-scope async locks for video quota checks and submissions.
///
/// A video tool value is created per turn, so the lock map must be supplied by
/// the runtime and cloned into each tool instance. The lock is intentionally
/// in-process because Chudbot runs as a single node.
#[derive(Debug, Clone, Default)]
pub(crate) struct VideoRateLimitLocks {
    /// Lazily-created async mutexes keyed by platform scope.
    pub(crate) inner: Arc<Mutex<BTreeMap<VideoRateLimitLockKey, Arc<AsyncMutex<()>>>>>,
}

impl VideoRateLimitLocks {
    /// Acquire the async mutex for the caller's platform scope.
    pub(crate) async fn lock(&self, user: &UserRef) -> OwnedMutexGuard<()> {
        let lock = {
            let mut locks = self
                .inner
                .lock()
                .expect("video rate limit lock map mutex poisoned");
            locks
                .entry(VideoRateLimitLockKey::from_user(user))
                .or_insert_with(|| Arc::new(AsyncMutex::new(())))
                .clone()
        };
        lock.lock_owned().await
    }
}

impl<G, M, S> PersistentVideoGeneratorTool<G, M, S> {
    /// Enforce the configured active-video quota for the current platform scope.
    ///
    /// The caller is expected to hold the matching `VideoRateLimitLocks` guard
    /// unless the user is bypassed. The storage count includes jobs still
    /// pending plus completed jobs inside the rolling interval.
    pub(crate) async fn enforce_video_rate_limit(
        &self,
        rate_limit: &VideoGenerationRateLimit,
    ) -> Result<(), BotToolError>
    where
        S: PersistentVideoStorage,
    {
        let interval_seconds = rate_limit
            .interval_seconds()
            .map_err(BotToolError::InvalidInput)?;
        let used = self
            .storage
            .count_active_video_generations(CountActiveVideoGenerations {
                platform: self.turn_user.platform.clone(),
                scope_id: self.turn_user.guild_id.clone(),
                interval_seconds,
            })
            .await
            .map_err(|error| BotToolError::Storage(error.to_string()))?;
        if used >= u64::from(rate_limit.limit) {
            tracing::warn!(
                used,
                limit = rate_limit.limit,
                interval = %rate_limit.interval,
                "video generation rate limit exceeded"
            );
            return Err(BotToolError::RateLimit(format!(
                "video generation rate limit exceeded for this platform scope: {} active video generation{} per {}",
                rate_limit.limit,
                if rate_limit.limit == 1 { "" } else { "s" },
                rate_limit.interval
            )));
        }
        Ok(())
    }

    /// Submit a provider job and create its local pending row.
    ///
    /// In the rate-limited path this runs inside the per-scope critical section
    /// so another call cannot observe the same active count before this row
    /// exists.
    pub(crate) async fn submit_and_persist_video_job(
        &self,
        request: VideoRequest,
        prompt: String,
    ) -> Result<VideoJobId, BotToolError>
    where
        G: VideoGenerator,
        S: PersistentVideoStorage,
    {
        let job_id = self
            .generator
            .submit_video(request)
            .await
            .map_err(|error| BotToolError::Generator(error.to_string()))?;
        self.storage
            .create_video_job(CreateVideoJob {
                turn_id: self.turn_id,
                provider: self.provider.clone(),
                provider_job_id: job_id.as_str().to_string(),
                prompt,
            })
            .await
            .map_err(|error| BotToolError::Storage(error.to_string()))?;
        tracing::info!(job = %job_id, "video job submitted and persisted");
        Ok(job_id)
    }
}

/// Convert model-supplied tool JSON into a provider-neutral video request.
///
/// The parser requires a non-empty `prompt`, accepts `image` or `image_url` as
/// a single optional image reference, bounds `duration_seconds` to the schema's
/// maximum, and passes provider-specific `aspect_ratio`, `resolution`, and
/// `model` strings through unchanged.
pub(crate) async fn video_request_from_tool_input<M>(
    media_store: &M,
    input: serde_json::Value,
) -> Result<VideoRequest, BotToolError>
where
    M: MediaStore,
{
    let prompt = tool_required_string(&input, "prompt")?;
    let image = match input.get("image").or_else(|| input.get("image_url")) {
        Some(value) => {
            // Stored media references and direct HTTP(S) URLs are resolved by
            // the shared helper using image semantics before provider submit.
            Some(resolve_tool_media_arg(media_store, MediaCategory::Image, value).await?)
        }
        None => None,
    };
    Ok(VideoRequest {
        prompt,
        image,
        duration_seconds: tool_optional_u8_bounded(&input, "duration_seconds", 15)?,
        aspect_ratio: tool_optional_string(&input, "aspect_ratio")?,
        resolution: tool_optional_string(&input, "resolution")?,
        model: tool_optional_string(&input, "model")?.map(ModelId::new),
    })
}

/// Build the durable trace payload for generated media.
///
/// Trace JSON includes store metadata plus optional public URL and provider
/// extras that are useful in the viewer or debugging, but not necessarily safe
/// or useful for the model to repeat.
pub(crate) fn media_tool_trace_json(
    media: &dyn chudbot_api::MediaRef,
    public_url: Option<&str>,
    extra: serde_json::Value,
) -> serde_json::Value {
    serde_json::json!({
        "uri": media.uri().as_str(),
        "category": media.category(),
        "name": media.name(),
        "mime_type": media.mime_type(),
        "size_bytes": media.size_bytes(),
        "public_url": public_url,
        "extra": extra,
    })
}

/// Build the model-facing JSON result for generated media.
///
/// The result exposes the stored media URI and a delivery note, while keeping
/// provider-only download URLs out of the model transcript.
pub(crate) fn media_tool_model_result_json(
    media: &dyn chudbot_api::MediaRef,
    extra: serde_json::Value,
) -> serde_json::Value {
    serde_json::json!({
        "uri": media.uri().as_str(),
        "category": media.category(),
        "mime_type": media.mime_type(),
        "size_bytes": media.size_bytes(),
        "delivery": {
            "platform_reply": "The generated media will be attached to the final platform reply automatically. Do not paste media URIs, filenames, public URLs, or markdown image/video links in user-facing text."
        },
        "extra": extra,
    })
}

/// Return the JSON schema advertised for the `generate_video` tool.
///
/// The schema is deliberately stricter than many provider APIs: it rejects
/// unknown fields and caps requested duration before provider routing.
pub(crate) fn video_tool_schema() -> ToolInputSchema {
    ToolInputSchema::new(serde_json::json!({
        "type": "object",
        "required": ["prompt"],
        "properties": {
            "prompt": {
                "type": "string",
                "description": "The video prompt."
            },
            "image": {
                "type": "string",
                "description": "Optional media URI or public URL for an image to animate. Use file:// media URIs from prior tool results; do not invent local filesystem paths."
            },
            "image_url": {
                "type": "string",
                "description": "Alias for image."
            },
            "duration_seconds": {
                "type": "integer",
                "minimum": 1,
                "maximum": 15
            },
            "aspect_ratio": {
                "type": "string",
                "description": "Optional provider-specific aspect ratio."
            },
            "resolution": {
                "type": "string",
                "description": "Optional provider-specific resolution or quality tier."
            },
            "model": {
                "type": "string",
                "description": "Optional provider-specific model id."
            }
        },
        "additionalProperties": false
    }))
}
