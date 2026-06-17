//! Persistent `generate_video` client tool with stored job status.

use super::*;

/// Storage operations needed to persist provider video jobs.
pub(crate) trait PersistentVideoStorage: Clone + Send + Sync {
    type Error: std::error::Error + Send + Sync + 'static;

    fn create_video_job(
        &self,
        input: CreateVideoJob,
    ) -> impl Future<Output = Result<StoredVideoJob, Self::Error>> + Send;

    fn update_video_job(
        &self,
        input: UpdateVideoJob,
    ) -> impl Future<Output = Result<(), Self::Error>> + Send;

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
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct VideoRateLimitLockKey {
    pub(crate) platform: PlatformName,
    pub(crate) scope_id: Option<ExternalId>,
}

impl VideoRateLimitLockKey {
    pub(crate) fn from_user(user: &UserRef) -> Self {
        Self {
            platform: user.platform.clone(),
            scope_id: user.guild_id.clone(),
        }
    }
}

/// Per-scope async locks that serialize video quota checks and submits.
#[derive(Debug, Clone, Default)]
pub(crate) struct VideoRateLimitLocks {
    pub(crate) inner: Arc<Mutex<BTreeMap<VideoRateLimitLockKey, Arc<AsyncMutex<()>>>>>,
}

impl VideoRateLimitLocks {
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

/// Video generation tool that submits, persists, polls, downloads, and stores output media.
#[derive(Debug, Clone)]
pub(crate) struct PersistentVideoGeneratorTool<G, M, S> {
    pub(crate) generator: G,
    pub(crate) media_store: M,
    pub(crate) storage: S,
    pub(crate) rate_limit_locks: VideoRateLimitLocks,
    pub(crate) turn_id: TurnId,
    pub(crate) turn_user: UserRef,
    pub(crate) provider: ProviderName,
    pub(crate) rate_limit: Option<VideoGenerationRateLimit>,
    pub(crate) description: String,
    pub(crate) poll_interval: Duration,
    pub(crate) max_polls: u32,
}

impl<G, M, S> PersistentVideoGeneratorTool<G, M, S> {
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

    pub(crate) fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = description.into();
        self
    }

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

impl<G, M, S> PersistentVideoGeneratorTool<G, M, S>
where
    G: VideoGenerator,
    M: MediaStore,
    S: PersistentVideoStorage,
{
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
