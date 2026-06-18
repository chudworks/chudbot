//! Unit tests for platform-neutral bot orchestration helpers.

use super::*;
use crate::config::{append_default_audio_keyterms, audio_transcription_default_keyterms};
use crate::prelude::*;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use chudbot_api::{
    AgentSpec, AssistantStep, BoxedMediaRef, CreateMedia, ExternalId, LlmBackend, LoadedMedia,
    MediaError, MediaMetadata, MediaRef, MediaStore, MediaUri, ModelStep, PlatformName,
    PostedMessage, PublicMediaUrl, ServerToolSet, UsageSubject, VideoJobId,
};
use serde_json::json;
use test_case::test_case;

// Shared constructors keep platform ids consistent across tests without pulling in
// Discord-specific types.
fn user(platform: &str, guild: Option<&str>, id: &str) -> chudbot_api::UserRef {
    chudbot_api::UserRef {
        platform: PlatformName::new(platform),
        guild_id: guild.map(ExternalId::new),
        user_id: ExternalId::new(id),
    }
}

fn message_ref(id: &str) -> MessageRef {
    MessageRef {
        platform: PlatformName::new("discord"),
        guild_id: Some(ExternalId::new("guild-1")),
        channel_id: ExternalId::new("channel-1"),
        message_id: ExternalId::new(id),
    }
}

fn channel_ref(guild: Option<&str>) -> ChannelRef {
    ChannelRef {
        platform: PlatformName::new("discord"),
        guild_id: guild.map(ExternalId::new),
        channel_id: ExternalId::new("channel-1"),
    }
}

fn usage_cost_row(key: Option<&str>) -> UsageCostRow {
    UsageCostRow {
        key: key.map(str::to_string),
        label: None,
        records: 3,
        conversations: 2,
        turns: 2,
        input_tokens: 100,
        cached_input_tokens: 10,
        output_tokens: 50,
        reasoning_tokens: 5,
        total_tokens: 165,
        cost_usd: Some("0.0123".to_string()),
        cost_estimated: false,
        unpriced_records: 0,
    }
}

// Usage report tests cover the JSON tool contract: default scoping, input
// validation, grouping labels, and response payload shape.
#[test]
fn generated_tool_schemas_advertise_canonical_input_fields() {
    let audio = audio_transcription_tool_schema().json_schema();
    assert!(audio["properties"].get("audio_uri").is_some());
    assert!(audio["properties"].get("keyterms").is_some());
    assert!(audio["properties"].get("audio").is_none());
    assert!(audio["properties"].get("keyterm").is_none());

    let image = image_tool_schema().json_schema();
    assert!(image["properties"].get("reference_images").is_some());
    assert!(image["properties"].get("references").is_none());

    let video = video_tool_schema().json_schema();
    assert!(video["properties"].get("image").is_some());
    assert!(video["properties"].get("image_url").is_none());
}

#[test]
fn usage_report_request_defaults_to_guild_lifetime_total() {
    let request = usage_report_request(
        &json!({}),
        &channel_ref(Some("guild-1")),
        OffsetDateTime::UNIX_EPOCH,
    )
    .expect("default input parses");
    assert_eq!(request.query.group_by, UsageCostGrouping::Total);
    assert_eq!(
        request.query.scope,
        UsageCostScope::Guild {
            guild_id: "guild-1".to_string()
        }
    );
    assert!(request.query.since.is_none());
    assert_eq!(request.days, None);
    assert_eq!(request.limit, 10);
    // The storage query asks for one extra row so the caller can detect truncation.
    assert_eq!(request.query.limit, 11);
    assert_eq!(request.query.platform, PlatformName::new("discord"));
}

#[test]
fn usage_report_request_guild_scope_falls_back_to_dm_channel() {
    let request = usage_report_request(&json!({}), &channel_ref(None), OffsetDateTime::UNIX_EPOCH)
        .expect("dm input parses");
    assert_eq!(
        request.query.scope,
        UsageCostScope::Channel {
            guild_id: None,
            channel_id: "channel-1".to_string()
        }
    );
}

#[test_case(json!({"scope": "channel"}), UsageCostScope::Channel {
    guild_id: Some("guild-1".to_string()),
    channel_id: "channel-1".to_string(),
} ; "channel scope keeps guild")]
#[test_case(json!({"scope": "global"}), UsageCostScope::All ; "global scope")]
fn usage_report_request_maps_scopes(input: serde_json::Value, expected: UsageCostScope) {
    let request = usage_report_request(
        &input,
        &channel_ref(Some("guild-1")),
        OffsetDateTime::UNIX_EPOCH,
    )
    .expect("scope parses");
    assert_eq!(request.query.scope, expected);
}

#[test_case("guild", UsageCostGrouping::Guild ; "guild grouping")]
#[test_case("channel", UsageCostGrouping::Channel ; "channel grouping")]
#[test_case("user", UsageCostGrouping::User ; "user grouping")]
#[test_case("agent", UsageCostGrouping::Agent ; "agent grouping")]
#[test_case("provider", UsageCostGrouping::Provider ; "provider grouping")]
#[test_case("model", UsageCostGrouping::Model ; "model grouping")]
#[test_case("kind", UsageCostGrouping::Kind ; "kind grouping")]
fn usage_report_request_maps_groupings(name: &str, expected: UsageCostGrouping) {
    let request = usage_report_request(
        &json!({ "group_by": name }),
        &channel_ref(Some("guild-1")),
        OffsetDateTime::UNIX_EPOCH,
    )
    .expect("grouping parses");
    assert_eq!(request.query.group_by, expected);
}

#[test]
fn usage_report_request_window_days_sets_since() {
    let now = OffsetDateTime::UNIX_EPOCH + time::Duration::days(400);
    let request = usage_report_request(&json!({"days": 1.5}), &channel_ref(Some("guild-1")), now)
        .expect("days parses");
    assert_eq!(request.days, Some(1.5));
    assert_eq!(
        request.query.since,
        Some(now - time::Duration::seconds_f64(1.5 * 86_400.0))
    );
}

#[test_case(json!({"group_by": "vibes"}) ; "unknown grouping")]
#[test_case(json!({"scope": "universe"}) ; "unknown scope")]
#[test_case(json!({"days": 0}) ; "zero days")]
#[test_case(json!({"days": -3}) ; "negative days")]
#[test_case(json!({"days": "week"}) ; "non numeric days")]
#[test_case(json!({"limit": 0}) ; "limit below range")]
#[test_case(json!({"limit": 51}) ; "limit above range")]
fn usage_report_request_rejects_invalid_input(input: serde_json::Value) {
    let result = usage_report_request(
        &input,
        &channel_ref(Some("guild-1")),
        OffsetDateTime::UNIX_EPOCH,
    );
    assert!(matches!(result, Err(BotToolError::InvalidInput(_))));
}

#[test_case("guild:g-1:channel:c-9", Some("c-9") ; "guild channel key")]
#[test_case("channel:c-9", Some("c-9") ; "dm channel key")]
#[test_case("memory", None ; "memory sentinel")]
fn channel_id_from_channel_key_extracts_id(key: &str, expected: Option<&str>) {
    assert_eq!(channel_id_from_channel_key(key), expected);
}

#[test_case(UsageCostGrouping::User, Some("u-1"), Some("<@u-1>") ; "user mention")]
#[test_case(
    UsageCostGrouping::Channel,
    Some("guild:g-1:channel:c-1"),
    Some("<#c-1>") ; "channel mention"
)]
#[test_case(UsageCostGrouping::Channel, Some("memory"), None ; "memory has no mention")]
#[test_case(UsageCostGrouping::Guild, Some("g-1"), None ; "guild has no mention")]
#[test_case(UsageCostGrouping::User, None, None ; "missing key has no mention")]
fn usage_cost_row_mention_decorates_rows(
    group_by: UsageCostGrouping,
    key: Option<&str>,
    expected: Option<&str>,
) {
    let row = usage_cost_row(key);
    assert_eq!(
        usage_cost_row_mention(group_by, &row),
        expected.map(str::to_string)
    );
}

#[test]
fn usage_report_value_includes_groups_only_when_grouped() {
    let grouped = usage_report_request(
        &json!({"group_by": "user", "days": 1.0}),
        &channel_ref(Some("guild-1")),
        OffsetDateTime::UNIX_EPOCH + time::Duration::days(400),
    )
    .expect("request parses");
    let total_row = usage_cost_row(None);
    let group_row = usage_cost_row(Some("u-1"));
    let value = usage_report_value(&grouped, Some(&total_row), &[group_row], true);
    assert_eq!(value["group_by"], "user");
    assert_eq!(value["scope"]["kind"], "guild");
    assert_eq!(value["window_days"], 1.0);
    assert!(value["since"].is_string());
    assert_eq!(value["total"]["cost_usd"], "0.0123");
    assert_eq!(value["groups"][0]["mention"], "<@u-1>");
    // Truncation is omitted for non-grouped totals, so this grouped case checks
    // the only payload shape where it should be present.
    assert_eq!(value["truncated"], true);

    let total = usage_report_request(
        &json!({}),
        &channel_ref(Some("guild-1")),
        OffsetDateTime::UNIX_EPOCH,
    )
    .expect("request parses");
    let value = usage_report_value(&total, Some(&total_row), &[], false);
    assert!(value["since"].is_null());
    assert!(value.get("groups").is_none());
    assert!(value.get("truncated").is_none());
}

// Media-store doubles keep asset and generator tests in memory while preserving
// the MediaStore/MediaRef contracts used by production code.
#[derive(Debug, Clone)]
struct NoopMediaStore;

impl MediaStore for NoopMediaStore {
    async fn create_media(&self, _input: CreateMedia) -> Result<BoxedMediaRef, MediaError> {
        Err(MediaError::UnsupportedCategory("test".to_string()))
    }

    async fn media_from_uri(&self, uri: &MediaUri) -> Result<BoxedMediaRef, MediaError> {
        Err(MediaError::UnsupportedUri(uri.to_string()))
    }

    async fn media_from_name(
        &self,
        category: MediaCategory,
        name: &str,
    ) -> Result<BoxedMediaRef, MediaError> {
        Err(MediaError::UnsupportedUri(format!(
            "file://{}/{name}",
            category.prefix()
        )))
    }
}

#[derive(Debug, Clone, Default)]
struct RecordingMediaStore {
    created: Arc<Mutex<Vec<CreateMedia>>>,
}

impl MediaStore for RecordingMediaStore {
    async fn create_media(&self, input: CreateMedia) -> Result<BoxedMediaRef, MediaError> {
        self.created.lock().unwrap().push(input.clone());
        let category = input.category;
        let name = input
            .name
            .unwrap_or_else(|| format!("generated.{}", category.prefix()));
        let uri = MediaUri::new(format!("memory://{}/{name}", category.prefix()));
        let mime_type = input
            .mime_type
            .unwrap_or_else(|| "application/octet-stream".to_string());
        Ok(Box::new(RecordingMediaRef {
            metadata: MediaMetadata {
                category,
                name,
                uri,
                mime_type,
                size_bytes: input.bytes.len() as u64,
            },
            bytes: input.bytes,
            public_url: Some(PublicMediaUrl::new("https://media.example/generated")),
        }))
    }

    async fn media_from_uri(&self, uri: &MediaUri) -> Result<BoxedMediaRef, MediaError> {
        Ok(Box::new(RecordingMediaRef {
            metadata: MediaMetadata {
                category: MediaCategory::Image,
                name: "reference.png".to_string(),
                uri: uri.clone(),
                mime_type: "image/png".to_string(),
                size_bytes: 10,
            },
            bytes: b"reference".to_vec(),
            public_url: None,
        }))
    }

    async fn media_from_name(
        &self,
        category: MediaCategory,
        name: &str,
    ) -> Result<BoxedMediaRef, MediaError> {
        self.media_from_uri(&MediaUri::new(format!(
            "memory://{}/{name}",
            category.prefix()
        )))
        .await
    }
}

#[derive(Debug, Clone)]
struct RecordingMediaRef {
    metadata: MediaMetadata,
    bytes: Vec<u8>,
    public_url: Option<PublicMediaUrl>,
}

#[async_trait::async_trait]
impl MediaRef for RecordingMediaRef {
    fn metadata(&self) -> &MediaMetadata {
        &self.metadata
    }

    fn clone_box(&self) -> BoxedMediaRef {
        Box::new(self.clone())
    }

    async fn public_url(&self) -> Result<PublicMediaUrl, MediaError> {
        self.public_url
            .clone()
            .ok_or_else(|| MediaError::NoPublicUrl {
                uri: self.uri().clone(),
            })
    }

    async fn load(&self) -> Result<LoadedMedia, MediaError> {
        Ok(LoadedMedia {
            media: self.clone_box(),
            bytes: self.bytes.clone(),
        })
    }
}

#[derive(Debug, thiserror::Error)]
#[error("test image error")]
struct TestImageError;

// Image generator double records the prompt/reference count and returns a saved
// image payload so the tool wrapper can be tested end to end.
#[derive(Debug, Clone)]
struct RecordingImageGenerator {
    seen: Arc<Mutex<Option<(String, usize)>>>,
}

impl ImageGenerator for RecordingImageGenerator {
    type Error = TestImageError;

    fn backend_name(&self) -> &ProviderName {
        static NAME: std::sync::OnceLock<ProviderName> = std::sync::OnceLock::new();
        NAME.get_or_init(|| ProviderName::new("test_image"))
    }

    async fn generate_image(&self, request: ImageRequest) -> Result<GeneratedImage, Self::Error> {
        *self.seen.lock().unwrap() = Some((request.prompt.clone(), request.references.len()));
        Ok(GeneratedImage {
            bytes: b"generated image".to_vec(),
            mime_type: "image/png".to_string(),
            model: ModelId::new("image-model"),
            revised_prompt: Some(format!("revised {}", request.prompt)),
            usage: vec![UsageRecord::new(
                ProviderName::new("test_image"),
                chudbot_api::UsageSubject::ImageGeneration,
            )],
        })
    }
}

#[tokio::test]
async fn image_generation_rejects_more_than_three_reference_images() {
    let error = image_request_from_tool_input(
        &NoopMediaStore,
        json!({
            "prompt": "draw this",
            "reference_images": [
                "https://example.com/1.png",
                "https://example.com/2.png",
                "https://example.com/3.png",
                "https://example.com/4.png"
            ]
        }),
    )
    .await
    .unwrap_err();

    assert!(matches!(error, BotToolError::InvalidInput(message) if message.contains("at most 3")));
}

#[tokio::test]
async fn image_generation_tool_saves_media_and_returns_uri() {
    let seen = Arc::new(Mutex::new(None));
    let store = RecordingMediaStore::default();
    let tool = ImageGeneratorTool::new(
        RecordingImageGenerator { seen: seen.clone() },
        store.clone(),
    );

    let output = tool
        .call(ClientToolCall {
            id: ToolUseId::new("call-1"),
            name: ToolName::new(GENERATE_IMAGE_TOOL),
            input: json!({
                "prompt": "draw a diagram",
                "reference_images": ["https://example.com/reference.png"],
                "aspect_ratio": "16:9"
            }),
        })
        .await
        .expect("image tool should succeed");

    assert!(!output.is_error);
    assert_eq!(
        *seen.lock().unwrap(),
        Some(("draw a diagram".to_string(), 1))
    );
    // The tool should persist generated bytes through MediaStore and return a
    // model-safe URI, not raw bytes or a direct provider artifact.
    assert_eq!(store.created.lock().unwrap().len(), 1);
    assert_eq!(
        store.created.lock().unwrap()[0].mime_type.as_deref(),
        Some("image/png")
    );

    let ClientToolResultContent::Json { value } = output.result else {
        panic!("expected json tool result");
    };
    assert_eq!(value["uri"], "memory://images/generated.images");
    assert_eq!(value["mime_type"], "image/png");
    assert!(value.get("public_url").is_none());
    assert!(
        value["delivery"]["platform_reply"]
            .as_str()
            .unwrap()
            .contains("attached")
    );
    assert_eq!(
        output.trace_response["public_url"],
        "https://media.example/generated"
    );
    assert_eq!(output.usage.len(), 1);
}

// Platform double used by reaction-tool tests. Every method except add_reaction
// fails loudly so validation tests can prove no platform call was attempted.
#[derive(Debug, Clone, Default)]
struct ReactionRecordingPlatform {
    reactions: Arc<Mutex<Vec<(MessageRef, ReactionKind)>>>,
}

impl MessagePlatformRegistry for ReactionRecordingPlatform {
    type Error = TestPlatformError;

    async fn bot_user(&self, _platform: &PlatformName) -> Result<UserProfile, Self::Error> {
        Err(TestPlatformError("unexpected bot_user".to_string()))
    }

    async fn register_commands(
        &self,
        _commands: Vec<PlatformCommandDefinition>,
    ) -> Result<(), Self::Error> {
        Err(TestPlatformError(
            "unexpected register_commands".to_string(),
        ))
    }

    async fn next_event(&self) -> Result<PlatformEvent, Self::Error> {
        Err(TestPlatformError("unexpected next_event".to_string()))
    }

    async fn respond_to_command(
        &self,
        _response: PlatformCommandResponse,
    ) -> Result<(), Self::Error> {
        Err(TestPlatformError(
            "unexpected respond_to_command".to_string(),
        ))
    }

    async fn send_message(&self, _request: SendMessage) -> Result<PostedMessage, Self::Error> {
        Err(TestPlatformError("unexpected send_message".to_string()))
    }

    async fn delete_message(&self, _message: MessageRef) -> Result<(), Self::Error> {
        Err(TestPlatformError("unexpected delete_message".to_string()))
    }

    async fn add_reaction(
        &self,
        message: MessageRef,
        reaction: ReactionKind,
    ) -> Result<(), Self::Error> {
        self.reactions.lock().unwrap().push((message, reaction));
        Ok(())
    }

    async fn remove_own_reaction(
        &self,
        _message: MessageRef,
        _reaction: ReactionKind,
    ) -> Result<(), Self::Error> {
        Err(TestPlatformError(
            "unexpected remove_own_reaction".to_string(),
        ))
    }

    async fn typing(&self, _channel: ChannelRef) -> Result<(), Self::Error> {
        Err(TestPlatformError("unexpected typing".to_string()))
    }

    async fn fetch_messages(
        &self,
        _request: FetchMessages,
    ) -> Result<Vec<PlatformMessage>, Self::Error> {
        Err(TestPlatformError("unexpected fetch_messages".to_string()))
    }

    async fn message_context(
        &self,
        _message: &PlatformMessage,
        _relationship: PlatformMessageRelationship,
    ) -> Result<serde_json::Value, Self::Error> {
        Err(TestPlatformError("unexpected message_context".to_string()))
    }

    async fn parent_channel(&self, channel: ChannelRef) -> Result<ChannelRef, Self::Error> {
        Ok(channel)
    }
}

#[derive(Debug, thiserror::Error)]
#[error("{0}")]
struct TestPlatformError(String);

// Reaction tests protect the Discord-facing emoji contract, including reserved
// bot status reactions and the guarantee that invalid input never touches I/O.
#[test_case("👍" ; "thumbs up")]
#[test_case("🏊🏾" ; "skin tone sequence")]
#[test_case("❤️" ; "emoji presentation selector")]
#[test_case("🇺🇸" ; "regional indicator flag")]
#[test_case("1️⃣" ; "keycap")]
#[test_case("👨‍👩‍👧‍👦" ; "zwj family")]
#[test_case("🏳️‍🌈" ; "zwj flag")]
fn reaction_emoji_validation_accepts_single_unicode_emoji(input: &str) {
    validate_reaction_emoji(input).expect("emoji should be accepted");
}

#[test_case("done" ; "plain text")]
#[test_case(":smile:" ; "discord shortcode")]
#[test_case("<:party:123>" ; "custom emoji markup")]
#[test_case("👍 ok" ; "emoji plus text")]
#[test_case("👍🎉" ; "multiple standalone emoji")]
#[test_case("1" ; "bare digit")]
#[test_case("🏾" ; "modifier alone")]
#[test_case("👍 " ; "trailing whitespace")]
fn reaction_emoji_validation_rejects_non_emoji_input(input: &str) {
    let error = validate_reaction_emoji(input).expect_err("non-emoji input should be rejected");

    assert!(matches!(error, BotToolError::InvalidInput(_)));
}

#[test_case("👀" ; "working")]
#[test_case("✅" ; "success")]
#[test_case("❌" ; "error")]
#[test_case("🔄" ; "retry")]
#[test_case("🛑" ; "stop")]
#[test_case("❓" ; "refused")]
fn reaction_tool_input_rejects_reserved_system_reactions(input: &str) {
    let error = reaction_emoji_from_tool_input(&json!({ "emoji": input }))
        .expect_err("reserved reaction should be rejected");

    assert!(matches!(error, BotToolError::InvalidInput(message) if message.contains("reserved")));
}

#[tokio::test]
async fn add_reaction_tool_reacts_to_current_user_message() {
    let platform = ReactionRecordingPlatform::default();
    let target = message_ref("user-message-1");
    let tool = AddReactionTool {
        platforms: platform.clone(),
        message: target.clone(),
    };

    let output = tool
        .call(ClientToolCall {
            id: ToolUseId::new("call-1"),
            name: ToolName::new(ADD_REACTION_TOOL),
            input: json!({ "emoji": "🏊" }),
        })
        .await
        .expect("reaction tool should succeed");

    assert!(!output.is_error);
    let reactions = platform.reactions.lock().unwrap();
    assert_eq!(reactions.len(), 1);
    assert_eq!(reactions[0].0, target);
    assert_eq!(
        reactions[0].1,
        ReactionKind::Unicode {
            name: "🏊".to_string()
        }
    );
}

#[tokio::test]
async fn add_reaction_tool_rejects_text_without_platform_call() {
    let platform = ReactionRecordingPlatform::default();
    let tool = AddReactionTool {
        platforms: platform.clone(),
        message: message_ref("user-message-1"),
    };

    let error = tool
        .call(ClientToolCall {
            id: ToolUseId::new("call-1"),
            name: ToolName::new(ADD_REACTION_TOOL),
            input: json!({ "emoji": "done" }),
        })
        .await
        .expect_err("text should be rejected");

    assert!(matches!(error, BotToolError::InvalidInput(_)));
    // Invalid tool input must fail before attempting any Discord mutation.
    assert!(platform.reactions.lock().unwrap().is_empty());
}

#[tokio::test]
async fn add_reaction_tool_rejects_reserved_reaction_without_platform_call() {
    let platform = ReactionRecordingPlatform::default();
    let tool = AddReactionTool {
        platforms: platform.clone(),
        message: message_ref("user-message-1"),
    };

    let error = tool
        .call(ClientToolCall {
            id: ToolUseId::new("call-1"),
            name: ToolName::new(ADD_REACTION_TOOL),
            input: json!({ "emoji": "✅" }),
        })
        .await
        .expect_err("reserved reaction should be rejected");

    assert!(matches!(error, BotToolError::InvalidInput(message) if message.contains("reserved")));
    // Reserved system reactions are filtered locally for the same reason as
    // malformed emoji: they should never trigger a platform call.
    assert!(platform.reactions.lock().unwrap().is_empty());
}

// Video generation fixtures exercise the persistent rate-limit path without
// polling or downloading real provider jobs.
#[derive(Debug, Clone)]
struct CountingVideoGenerator {
    submits: Arc<AtomicUsize>,
    submit_delay: Duration,
}

impl VideoGenerator for CountingVideoGenerator {
    type Error = TestVideoError;

    fn backend_name(&self) -> &ProviderName {
        static NAME: std::sync::OnceLock<ProviderName> = std::sync::OnceLock::new();
        NAME.get_or_init(|| ProviderName::new("test_video"))
    }

    async fn submit_video(&self, _request: VideoRequest) -> Result<VideoJobId, Self::Error> {
        let submit = self.submits.fetch_add(1, Ordering::SeqCst) + 1;
        if !self.submit_delay.is_zero() {
            tokio::time::sleep(self.submit_delay).await;
        }
        Ok(VideoJobId::new(format!("job-{submit}")))
    }

    async fn check_video(&self, _job: VideoJobId) -> Result<VideoJobStatus, Self::Error> {
        Err(TestVideoError("unexpected poll".to_string()))
    }

    async fn download_video(&self, _url: String) -> Result<Vec<u8>, Self::Error> {
        Err(TestVideoError("unexpected download".to_string()))
    }
}

#[derive(Debug, thiserror::Error)]
#[error("{0}")]
struct TestVideoError(String);

#[derive(Debug, Clone)]
struct VideoRateLimitStorage {
    count: Arc<AtomicU64>,
    count_requests: Arc<Mutex<Vec<CountActiveVideoGenerations>>>,
    creates: Arc<AtomicUsize>,
    updates: Arc<AtomicUsize>,
}

impl VideoRateLimitStorage {
    fn new(count: u64) -> Self {
        Self {
            count: Arc::new(AtomicU64::new(count)),
            count_requests: Arc::new(Mutex::new(Vec::new())),
            creates: Arc::new(AtomicUsize::new(0)),
            updates: Arc::new(AtomicUsize::new(0)),
        }
    }
}

impl PersistentVideoStorage for VideoRateLimitStorage {
    type Error = TestVideoStorageError;

    async fn create_video_job(&self, input: CreateVideoJob) -> Result<StoredVideoJob, Self::Error> {
        self.creates.fetch_add(1, Ordering::SeqCst);
        self.count.fetch_add(1, Ordering::SeqCst);
        Ok(StoredVideoJob {
            turn_id: input.turn_id,
            provider: input.provider,
            provider_job_id: input.provider_job_id,
            prompt: input.prompt,
            status: "pending".to_string(),
            output_uri: None,
            error: None,
        })
    }

    async fn update_video_job(&self, _input: UpdateVideoJob) -> Result<(), Self::Error> {
        self.updates.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    async fn count_active_video_generations(
        &self,
        input: CountActiveVideoGenerations,
    ) -> Result<u64, Self::Error> {
        self.count_requests.lock().unwrap().push(input);
        Ok(self.count.load(Ordering::SeqCst))
    }
}

#[derive(Debug, thiserror::Error)]
#[error("test storage error")]
struct TestVideoStorageError;

fn video_rate_limit_tool<G>(
    generator: G,
    storage: VideoRateLimitStorage,
    rate_limit_locks: VideoRateLimitLocks,
    turn_user: UserRef,
    rate_limit: Option<VideoGenerationRateLimit>,
) -> PersistentVideoGeneratorTool<G, NoopMediaStore, VideoRateLimitStorage> {
    PersistentVideoGeneratorTool {
        generator,
        media_store: NoopMediaStore,
        storage,
        rate_limit_locks,
        context: RuntimeToolContext::new(
            message_ref("message-1"),
            ConversationId::new(),
            TurnId::new(),
            turn_user,
            PrivacyMode::ConversationOnly,
        ),
        binding: GenerationBinding {
            provider: ProviderName::new("grok_video"),
            model: ModelId::new("grok-video-test"),
            rate_limit,
        },
        poll_interval: DEFAULT_VIDEO_POLL_INTERVAL,
        max_polls: DEFAULT_VIDEO_MAX_POLLS,
    }
}

#[tokio::test]
async fn video_rate_limit_fails_before_provider_submit() {
    let submits = Arc::new(AtomicUsize::new(0));
    let storage = VideoRateLimitStorage::new(2);
    let tool = video_rate_limit_tool(
        CountingVideoGenerator {
            submits: submits.clone(),
            submit_delay: Duration::ZERO,
        },
        storage.clone(),
        VideoRateLimitLocks::default(),
        user("discord", Some("guild-1"), "user-1"),
        Some(VideoGenerationRateLimit {
            limit: 2,
            interval: "4h".to_string(),
            bypass_scopes: Vec::new(),
        }),
    );

    let error = tool
        .call(ClientToolCall {
            id: ToolUseId::new("call-1"),
            name: ToolName::new("generate_video"),
            input: json!({ "prompt": "animate this" }),
        })
        .await
        .expect_err("rate limit should fail the tool call");

    assert!(
        matches!(error, BotToolError::RateLimit(message) if message.contains("2 active video generations per 4h"))
    );
    // A rejected request must not create provider or storage side effects.
    assert_eq!(submits.load(Ordering::SeqCst), 0);
    assert_eq!(storage.creates.load(Ordering::SeqCst), 0);
    assert_eq!(storage.updates.load(Ordering::SeqCst), 0);
    let requests = storage.count_requests.lock().unwrap();
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].platform.as_str(), "discord");
    assert_eq!(
        requests[0].scope_id.as_ref().map(ExternalId::as_str),
        Some("guild-1")
    );
    assert_eq!(requests[0].interval_seconds, 4 * 60 * 60);
}

#[tokio::test]
async fn video_rate_limit_counts_pending_jobs_between_parallel_calls() {
    let submits = Arc::new(AtomicUsize::new(0));
    let storage = VideoRateLimitStorage::new(0);
    let rate_limit_locks = VideoRateLimitLocks::default();
    let rate_limit = Some(VideoGenerationRateLimit {
        limit: 1,
        interval: "4h".to_string(),
        bypass_scopes: Vec::new(),
    });
    let generator = CountingVideoGenerator {
        submits: submits.clone(),
        submit_delay: Duration::from_millis(50),
    };
    let mut first = video_rate_limit_tool(
        generator.clone(),
        storage.clone(),
        rate_limit_locks.clone(),
        user("discord", Some("guild-1"), "user-1"),
        rate_limit.clone(),
    );
    first.max_polls = 0;
    let mut second = video_rate_limit_tool(
        generator,
        storage.clone(),
        rate_limit_locks,
        user("discord", Some("guild-1"), "user-2"),
        rate_limit,
    );
    second.max_polls = 0;

    let (first_result, second_result) = tokio::join!(
        first.call(ClientToolCall {
            id: ToolUseId::new("call-1"),
            name: ToolName::new("generate_video"),
            input: json!({ "prompt": "animate this" }),
        }),
        second.call(ClientToolCall {
            id: ToolUseId::new("call-2"),
            name: ToolName::new("generate_video"),
            input: json!({ "prompt": "animate that" }),
        })
    );

    let results = [&first_result, &second_result];
    assert_eq!(
        results
            .iter()
            .filter(|result| matches!(result, Err(BotToolError::RateLimit(_))))
            .count(),
        1
    );
    // The per-scope lock makes the pending create visible to the racing call.
    assert_eq!(submits.load(Ordering::SeqCst), 1);
    assert_eq!(storage.creates.load(Ordering::SeqCst), 1);
    assert_eq!(storage.count_requests.lock().unwrap().len(), 2);
}

// Video rate-limit config tests cover bypass scopes and validation after the
// async race checks above.
#[test]
fn video_rate_limit_bypasses_configured_platform_scope() {
    let rate_limit = VideoGenerationRateLimit {
        limit: 1,
        interval: "30m".to_string(),
        bypass_scopes: vec![PlatformScopeBypass {
            platform: PlatformName::new("discord"),
            scope_id: ExternalId::new("guild-1"),
        }],
    };

    assert_eq!(rate_limit.interval_seconds().unwrap(), 30 * 60);
    assert!(rate_limit.bypasses(&user("discord", Some("guild-1"), "user-1")));
    assert!(rate_limit.bypasses(&user("discord", Some("guild-1"), "user-2")));
    assert!(!rate_limit.bypasses(&user("discord", Some("guild-2"), "user-1")));
    assert!(!rate_limit.bypasses(&user("discord", None, "user-1")));
    assert!(!rate_limit.bypasses(&user("slack", Some("guild-1"), "user-1")));
}

#[test]
fn rejects_invalid_video_rate_limit_config() {
    let binding = GenerationBinding {
        provider: ProviderName::new("grok_video"),
        model: ModelId::new("grok-imagine-video"),
        rate_limit: Some(VideoGenerationRateLimit {
            limit: 0,
            interval: "4h".to_string(),
            bypass_scopes: Vec::new(),
        }),
    };

    let error = super::config::validate_generation_binding("default", "video_generation", &binding)
        .expect_err("zero limit should be rejected");

    assert!(error.to_string().contains("must be greater than zero"));
}

// Minimal runtime config fixtures for tests that only need agent/provider
// selection, not the full TOML loader path.
fn test_model_spec(model: &str) -> ModelSpec {
    ModelSpec {
        id: ModelId::new(model),
        server_tools: Default::default(),
        sampling: SamplingOptions::default(),
        provider_options: None,
    }
}

fn test_agent_config(provider: &str, model: &str) -> AgentConfig {
    AgentConfig {
        provider: ProviderName::new(provider),
        system_prompt: "test prompt".to_string(),
        model: test_model_spec(model),
        server_tools: None,
        client_tools: None,
        limits: None,
        image_generation: None,
        video_generation: None,
        audio_transcription: None,
        memory: false,
        subagents: BTreeMap::new(),
    }
}

fn test_bot_config() -> BotConfig {
    let mut agents = BTreeMap::new();
    agents.insert(
        "assistant".to_string(),
        test_agent_config("default_provider", "default_model"),
    );
    BotConfig {
        web_base_url: "http://localhost:3000".to_string(),
        default_agent: "assistant".to_string(),
        agents,
        admins: Vec::new(),
        platforms: BTreeMap::new(),
        extra_system_prompt: None,
        version: String::new(),
        limits: AgentLimits::default(),
        thread_threshold_chars: DEFAULT_THREAD_THRESHOLD_CHARS,
        thread_threshold_lines: DEFAULT_THREAD_THRESHOLD_LINES,
    }
}

#[derive(Debug)]
struct TestLlmError;

impl std::fmt::Display for TestLlmError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("test llm error")
    }
}

impl std::error::Error for TestLlmError {}

fn test_llm_model<B>(backend: B) -> Model<B> {
    Model {
        backend,
        spec: ModelSpec {
            id: ModelId::new("test-model"),
            server_tools: ServerToolSet::default(),
            sampling: SamplingOptions::default(),
            provider_options: None,
        },
    }
}

#[derive(Debug, Clone)]
struct RecordingLlmBackend {
    name: ProviderName,
    requests: Arc<Mutex<Vec<chudbot_api::ModelStepRequest>>>,
    step: AssistantStep,
}

impl LlmBackend for RecordingLlmBackend {
    type Error = TestLlmError;

    fn backend_name(&self) -> &ProviderName {
        &self.name
    }

    async fn step(&self, request: chudbot_api::ModelStepRequest) -> Result<ModelStep, Self::Error> {
        self.requests.lock().unwrap().push(request);
        Ok(ModelStep::Final {
            step: self.step.clone(),
        })
    }
}

#[tokio::test]
async fn subagent_exposes_spec_and_executes_nested_agent() {
    let usage = UsageRecord {
        provider: ProviderName::new("openai"),
        model: Some(ModelId::new("gpt-5")),
        subject: UsageSubject::ModelStep,
        input_tokens: Some(100),
        cached_input_tokens: Some(25),
        output_tokens: Some(40),
        reasoning_tokens: Some(10),
        total_tokens: Some(140),
        cost: None,
        raw: None,
    };
    let requests = Arc::new(Mutex::new(Vec::new()));
    let backend = RecordingLlmBackend {
        name: ProviderName::new("openai"),
        requests: requests.clone(),
        step: AssistantStep {
            content: vec![ContentBlock::Text {
                text: "use VTI".to_string(),
            }],
            client_tool_calls: Vec::new(),
            server_tool_uses: Vec::new(),
            grounding: Vec::new(),
            model_id: ModelId::new("gpt-5"),
            continuation: None,
            usage: vec![usage],
        },
    };
    let agent = Agent::new(
        test_llm_model(backend),
        AgentSpec::new("expert"),
        NoClientTools,
    );

    let subagent = Subagent::new("Ask the OpenAI expert for a second opinion.", agent);
    let spec = subagent.spec();
    assert_eq!(
        spec.description,
        "Ask the OpenAI expert for a second opinion."
    );
    assert!(spec.input_schema.json_schema().get("properties").is_some());

    let output = subagent
        .call(ClientToolCall {
            id: ToolUseId::new("call-1"),
            name: ToolName::new("ask_openai_expert"),
            input: json!({ "prompt": "Which total-market ETF is best?" }),
        })
        .await
        .unwrap();

    assert!(!output.is_error);
    assert_eq!(output.usage.len(), 1);
    match output.result {
        ClientToolResultContent::Text { text } => assert_eq!(text, "use VTI"),
        ClientToolResultContent::Json { .. } => panic!("expected text output"),
    }

    let inputs = requests.lock().unwrap();
    assert_eq!(inputs.len(), 1);
    assert_eq!(inputs[0].transcript.instructions.as_deref(), Some("expert"));
    assert_eq!(inputs[0].transcript.turns.len(), 1);
    assert_text_block(
        &inputs[0].transcript.turns[0],
        TurnRole::User,
        "Which total-market ETF is best?",
    );
}

#[tokio::test]
async fn subagent_ignores_registration_name() {
    let backend = RecordingLlmBackend {
        name: ProviderName::new("openai"),
        requests: Arc::new(Mutex::new(Vec::new())),
        step: AssistantStep {
            content: vec![ContentBlock::Text {
                text: "ok".to_string(),
            }],
            client_tool_calls: Vec::new(),
            server_tool_uses: Vec::new(),
            grounding: Vec::new(),
            model_id: ModelId::new("gpt-5"),
            continuation: None,
            usage: Vec::new(),
        },
    };
    let agent = Agent::new(
        test_llm_model(backend),
        AgentSpec::new("expert"),
        NoClientTools,
    );
    let subagent = Subagent::new("Ask the OpenAI expert.", agent);

    let output = subagent
        .call(ClientToolCall {
            id: ToolUseId::new("call-1"),
            name: ToolName::new("registered_elsewhere"),
            input: json!({ "prompt": "hello" }),
        })
        .await
        .unwrap();

    assert!(!output.is_error);
}

fn assert_text_block(message: &TranscriptTurn, role: TurnRole, text: &str) {
    assert_eq!(message.role, role);
    match message.blocks.as_slice() {
        [ContentBlock::Text { text: actual }] => assert_eq!(actual, text),
        blocks => panic!("expected one text block, got {blocks:?}"),
    }
}

// System-agent tests pin the configured-agent override and inheritance rules for
// reserved agents such as ToS preflight and conversation titles.
#[test]
fn configured_system_agents_override_inherited_defaults() {
    let mut config = test_bot_config();
    config.agents.insert(
        TOS_PREFLIGHT_AGENT.to_string(),
        test_agent_config("tos_provider", "tos_model"),
    );
    config.agents.insert(
        CONVERSATION_TITLE_AGENT.to_string(),
        test_agent_config("title_provider", "title_model"),
    );

    let system_agents = RuntimeSystemAgents::from_config(&config);
    let platform = PlatformName::new("discord");
    let tos = system_agents
        .tos_preflight
        .get(&platform, &config.default_agent)
        .expect("configured tos agent");
    let title = system_agents
        .conversation_title
        .get("assistant", &platform, &config.default_agent)
        .expect("configured title agent");

    assert_eq!(tos.provider, ProviderName::new("tos_provider"));
    assert_eq!(tos.model.id, ModelId::new("tos_model"));
    assert_eq!(title.provider, ProviderName::new("title_provider"));
    assert_eq!(title.model.id, ModelId::new("title_model"));
    assert!(system_agents.tos_preflight.platform_defaults.is_empty());
    assert!(system_agents.conversation_title.agent_defaults.is_empty());
}

#[test]
fn default_system_agents_preserve_active_agent_inheritance() {
    let mut config = test_bot_config();
    config.agents.insert(
        "research".to_string(),
        test_agent_config("research_provider", "research_model"),
    );
    config.platforms.insert(
        PlatformName::new("discord"),
        PlatformBinding {
            agent: "research".to_string(),
        },
    );

    let system_agents = RuntimeSystemAgents::from_config(&config);
    let discord = PlatformName::new("discord");
    let slack = PlatformName::new("slack");
    let tos_discord = system_agents
        .tos_preflight
        .get(&discord, &config.default_agent)
        .expect("platform inherited tos agent");
    let tos_slack = system_agents
        .tos_preflight
        .get(&slack, &config.default_agent)
        .expect("global inherited tos agent");
    let title_source_agent = system_agents
        .conversation_title
        .get("assistant", &discord, &config.default_agent)
        .expect("source inherited title agent");
    let title_platform_fallback = system_agents
        .conversation_title
        .get("missing", &discord, &config.default_agent)
        .expect("platform inherited title agent");

    assert_eq!(tos_discord.provider, ProviderName::new("research_provider"));
    assert_eq!(tos_discord.model.id, ModelId::new("research_model"));
    assert_eq!(tos_slack.provider, ProviderName::new("default_provider"));
    assert_eq!(tos_slack.model.id, ModelId::new("default_model"));
    assert_eq!(
        title_source_agent.provider,
        ProviderName::new("default_provider")
    );
    assert_eq!(
        title_platform_fallback.provider,
        ProviderName::new("research_provider")
    );
}

// Conversation-context helpers normalize platform mentions and decide what can
// be replayed into future model turns.
#[test]
fn platform_user_match_ignores_guild_scope() {
    let global_bot = user("discord", None, "123456789012345678");
    let guild_mention = user("discord", Some("guild-1"), "123456789012345678");
    let other_platform = user("slack", None, "123456789012345678");

    assert!(same_platform_user(&guild_mention, &global_bot));
    assert!(!same_platform_user(&other_platform, &global_bot));
}

#[test]
fn normalizes_bot_and_member_mentions() {
    let bot = user("discord", None, "111111111111111111");
    let mentioned = user("discord", Some("guild-1"), "222222222222222222");
    let profiles = [UserProfile {
        id: mentioned.clone(),
        username: "alice".to_string(),
        name: Some("Alice Global".to_string()),
        display_name: Some("Alice".to_string()),
        avatar_url: None,
        is_bot: false,
    }];

    let normalized = normalize_mention_content(
        "<@111111111111111111> hi <@!222222222222222222>",
        &bot,
        &[bot.clone(), mentioned],
        &profiles,
    );

    assert_eq!(normalized, "hi Alice (<@222222222222222222>)");
}

#[test]
fn profile_label_falls_back_to_platform_name_before_username() {
    let profile = UserProfile {
        id: user("discord", Some("guild-1"), "222222222222222222"),
        username: "alice".to_string(),
        name: Some("Alice Global".to_string()),
        display_name: None,
        avatar_url: None,
        is_bot: false,
    };

    assert_eq!(display_name_for_profile(&profile), "Alice Global");
}

#[test]
fn linked_assistant_message_replays_only_for_same_conversation() {
    let conversation_id = ConversationId::new();
    let other_conversation_id = ConversationId::new();
    let link = MessageLink {
        message: MessageRef {
            platform: PlatformName::new("discord"),
            guild_id: Some(ExternalId::new("guild-1")),
            channel_id: ExternalId::new("channel-1"),
            message_id: ExternalId::new("assistant-message-1"),
        },
        conversation_id,
        turn_id: TurnId::new(),
        role: "assistant".to_string(),
    };
    let user_link = MessageLink {
        role: "user".to_string(),
        ..link.clone()
    };

    assert!(message_link_replays_as_assistant(&link, conversation_id));
    assert!(!message_link_replays_as_assistant(
        &link,
        other_conversation_id
    ));
    assert!(!message_link_replays_as_assistant(
        &user_link,
        conversation_id
    ));
}

#[test]
fn replayable_context_items_drop_memory_context() {
    let platform_item = chudbot_api::ContextItem {
        position: 0,
        source: "platform:message:message-1".to_string(),
        role: "user".to_string(),
        content: "{\"content\":\"hi\"}".to_string(),
        message: None,
    };
    let memory_item = chudbot_api::ContextItem {
        position: 1,
        source: "memory:user:user-1".to_string(),
        role: "user".to_string(),
        content: "Background memory for the current user.".to_string(),
        message: None,
    };

    let replayable = replayable_context_items(&[platform_item.clone(), memory_item]);

    assert_eq!(replayable.len(), 1);
    assert_eq!(replayable[0].source, platform_item.source);
    assert_eq!(replayable[0].content, platform_item.content);
}

#[test]
fn repairs_bare_snowflake_mentions() {
    assert_eq!(
        fix_bare_mentions("talk to @123456789012345678"),
        "talk to <@123456789012345678>"
    );
    assert_eq!(
        fix_bare_mentions("already <@123456789012345678>"),
        "already <@123456789012345678>"
    );
    assert_eq!(fix_bare_mentions("short @123"), "short @123");
}

// Tool-trace fixtures model generated-media client tools closely enough for
// replay, reply-cleanup, and attachment-selection tests.
fn generated_image_trace(uri: &str, public_url: &str) -> ToolTrace {
    let tool_use_id = chudbot_api::ToolUseId::new("call-1");
    ToolTrace::Client {
        trace: chudbot_api::ClientToolTrace {
            call: ClientToolCall {
                id: tool_use_id.clone(),
                name: ToolName::new("generate_image"),
                input: json!({ "prompt": "a worm" }),
            },
            result: chudbot_api::ClientToolResult {
                tool_use_id,
                content: ClientToolResultContent::Json {
                    value: json!({
                        "uri": uri,
                        "category": "image",
                        "name": "generated.jpg",
                        "mime_type": "image/jpeg",
                        "size_bytes": 42,
                        "delivery": {
                            "platform_reply": "attached automatically"
                        },
                        "extra": {}
                    }),
                },
                is_error: false,
            },
            trace_response: json!({
                "uri": uri,
                "category": "image",
                "name": "generated.jpg",
                "mime_type": "image/jpeg",
                "size_bytes": 42,
                "public_url": public_url,
                "extra": {}
            }),
            usage: Vec::new(),
        },
    }
}

fn generated_video_trace(uri: &str, public_url: &str) -> ToolTrace {
    let tool_use_id = chudbot_api::ToolUseId::new("call-1");
    ToolTrace::Client {
        trace: chudbot_api::ClientToolTrace {
            call: ClientToolCall {
                id: tool_use_id.clone(),
                name: ToolName::new("generate_video"),
                input: json!({ "prompt": "a worm riding a bike" }),
            },
            result: chudbot_api::ClientToolResult {
                tool_use_id,
                content: ClientToolResultContent::Json {
                    value: json!({
                        "uri": uri,
                        "video_uri": uri,
                        "category": "video",
                        "name": "generated.mp4",
                        "mime_type": "video/mp4",
                        "size_bytes": MAX_OUTGOING_ATTACHMENT_BYTES + 1,
                        "delivery": {
                            "platform_reply": "attached automatically"
                        },
                        "extra": {}
                    }),
                },
                is_error: false,
            },
            trace_response: json!({
                "uri": uri,
                "video_uri": uri,
                "category": "video",
                "name": "generated.mp4",
                "mime_type": "video/mp4",
                "size_bytes": MAX_OUTGOING_ATTACHMENT_BYTES + 1,
                "public_url": public_url,
                "extra": {}
            }),
            usage: Vec::new(),
        },
    }
}

fn attach_trace(uri: &str) -> ToolTrace {
    let tool_use_id = chudbot_api::ToolUseId::new("call-attach");
    ToolTrace::Client {
        trace: chudbot_api::ClientToolTrace {
            call: ClientToolCall {
                id: tool_use_id.clone(),
                name: ToolName::new(ATTACH_ASSET_TOOL),
                input: json!({ "uri": uri }),
            },
            result: chudbot_api::ClientToolResult {
                tool_use_id,
                content: ClientToolResultContent::Json {
                    value: json!({
                        "uri": uri,
                        "category": "image",
                        "name": "generated.jpg",
                        "mime_type": "image/jpeg",
                        "size_bytes": 42,
                        "attached": true,
                    }),
                },
                is_error: false,
            },
            trace_response: json!({
                "uri": uri,
                "category": "image",
                "name": "generated.jpg",
                "mime_type": "image/jpeg",
                "size_bytes": 42,
                "attached": true,
            }),
            usage: Vec::new(),
        },
    }
}

// Prompt/reply media fixtures distinguish model-visible transcript media from
// generated media that should be attached to the final platform reply.
#[derive(Debug, Clone)]
struct PromptMediaRef {
    metadata: MediaMetadata,
    public_url: PublicMediaUrl,
}

impl PromptMediaRef {
    fn boxed(uri: &str, public_url: &str) -> chudbot_api::BoxedMediaRef {
        Box::new(Self {
            metadata: MediaMetadata {
                category: MediaCategory::Image,
                name: "generated.jpg".to_string(),
                uri: MediaUri::new(uri),
                mime_type: "image/jpeg".to_string(),
                size_bytes: 42,
            },
            public_url: PublicMediaUrl::new(public_url),
        })
    }

    fn boxed_audio(uri: &str) -> chudbot_api::BoxedMediaRef {
        Box::new(Self {
            metadata: MediaMetadata {
                category: MediaCategory::Audio,
                name: "voice.ogg".to_string(),
                uri: MediaUri::new(uri),
                mime_type: "audio/ogg".to_string(),
                size_bytes: 42,
            },
            public_url: PublicMediaUrl::new("https://chud.example/audio/voice.ogg"),
        })
    }
}

#[async_trait::async_trait]
impl MediaRef for PromptMediaRef {
    fn metadata(&self) -> &MediaMetadata {
        &self.metadata
    }

    fn clone_box(&self) -> chudbot_api::BoxedMediaRef {
        Box::new(self.clone())
    }

    async fn public_url(&self) -> Result<PublicMediaUrl, MediaError> {
        Ok(self.public_url.clone())
    }

    async fn load(&self) -> Result<chudbot_api::LoadedMedia, MediaError> {
        Err(MediaError::BytesUnavailable {
            uri: self.metadata.uri.clone(),
        })
    }
}

#[derive(Debug, Clone)]
struct ReplyMediaRef {
    metadata: MediaMetadata,
    bytes: Vec<u8>,
    public_url: Option<PublicMediaUrl>,
}

impl ReplyMediaRef {
    fn image(uri: &str, public_url: &str) -> Self {
        Self::image_with_mime(uri, "image/jpeg", public_url)
    }

    fn image_with_mime(uri: &str, mime_type: &str, public_url: &str) -> Self {
        Self {
            metadata: MediaMetadata {
                category: MediaCategory::Image,
                name: "generated.jpg".to_string(),
                uri: MediaUri::new(uri),
                mime_type: mime_type.to_string(),
                size_bytes: 42,
            },
            bytes: Vec::new(),
            public_url: Some(PublicMediaUrl::new(public_url)),
        }
    }

    fn video(uri: &str, size_bytes: u64, public_url: &str) -> Self {
        Self {
            metadata: MediaMetadata {
                category: MediaCategory::Video,
                name: "generated.mp4".to_string(),
                uri: MediaUri::new(uri),
                mime_type: "video/mp4".to_string(),
                size_bytes,
            },
            bytes: Vec::new(),
            public_url: Some(PublicMediaUrl::new(public_url)),
        }
    }
}

#[async_trait::async_trait]
impl MediaRef for ReplyMediaRef {
    fn metadata(&self) -> &MediaMetadata {
        &self.metadata
    }

    fn clone_box(&self) -> BoxedMediaRef {
        Box::new(self.clone())
    }

    async fn public_url(&self) -> Result<PublicMediaUrl, MediaError> {
        self.public_url
            .clone()
            .ok_or_else(|| MediaError::NoPublicUrl {
                uri: self.uri().clone(),
            })
    }

    async fn load(&self) -> Result<LoadedMedia, MediaError> {
        Ok(LoadedMedia {
            media: self.clone_box(),
            bytes: self.bytes.clone(),
        })
    }
}

#[derive(Debug, Clone)]
struct ReplyMediaStore {
    media: ReplyMediaRef,
}

impl ReplyMediaStore {
    fn new(media: ReplyMediaRef) -> Self {
        Self { media }
    }
}

impl MediaStore for ReplyMediaStore {
    async fn create_media(&self, _input: CreateMedia) -> Result<BoxedMediaRef, MediaError> {
        Err(MediaError::UnsupportedCategory("test".to_string()))
    }

    async fn media_from_uri(&self, uri: &MediaUri) -> Result<BoxedMediaRef, MediaError> {
        if self.media.uri() == uri {
            return Ok(Box::new(self.media.clone()));
        }
        Err(MediaError::UnsupportedUri(uri.to_string()))
    }

    async fn media_from_name(
        &self,
        category: MediaCategory,
        name: &str,
    ) -> Result<BoxedMediaRef, MediaError> {
        self.media_from_uri(&MediaUri::new(format!(
            "file://{}/{name}",
            category.prefix()
        )))
        .await
    }
}

// Generated-media reply tests keep automatic attachment handling separate from
// text cleanup and public-URL fallbacks.
#[test]
fn strips_generated_media_markdown_from_reply_text() {
    let trace = generated_image_trace(
        "file://images/generated.jpg",
        "https://chud.example/images/generated.jpg",
    );
    let refs = generated_media_reply_refs(&[trace]);

    let reply = strip_generated_media_refs(
        "Worm generated.\n\n![image](https://chud.example/images/generated.jpg)\n\nfile://images/generated.jpg",
        &refs,
    );

    assert_eq!(reply, "Worm generated.");
}

#[test]
fn generated_media_strip_preserves_unrelated_links() {
    let trace = generated_image_trace(
        "file://images/generated.jpg",
        "https://chud.example/images/generated.jpg",
    );
    let refs = generated_media_reply_refs(&[trace]);

    let reply = strip_generated_media_refs(
        "Done.\n\n-# [full trace](https://chud.example/c/abc)",
        &refs,
    );

    assert_eq!(
        reply,
        "Done.\n\n-# [full trace](https://chud.example/c/abc)"
    );
}

#[tokio::test]
async fn oversized_generated_video_uses_public_url_fallback() {
    let uri = "file://videos/generated.mp4";
    let public_url = "https://chud.example/videos/generated.mp4";
    let trace = generated_video_trace(uri, public_url);
    let store = ReplyMediaStore::new(ReplyMediaRef::video(
        uri,
        (MAX_OUTGOING_ATTACHMENT_BYTES + 1) as u64,
        public_url,
    ));

    let media = generated_reply_media(&store, &[trace]).await;

    assert!(media.attachments.is_empty());
    assert_eq!(media.public_urls, vec![public_url.to_string()]);
}

#[test]
fn appends_generated_media_public_urls_to_reply_text() {
    let reply = append_generated_media_public_urls(
        "Done.  \n".to_string(),
        &["https://chud.example/videos/generated.mp4".to_string()],
    );

    assert_eq!(
        reply,
        "Done.\n\nAttached media: https://chud.example/videos/generated.mp4"
    );
}

// Stored-asset tool tests verify which assets are exposed to the model, queued
// for reply attachment, or reported as unavailable without leaking bytes.
#[tokio::test]
async fn read_asset_exposes_supported_image_without_returning_bytes() {
    let uri = "file://images/generated.jpg";
    let store = ReplyMediaStore::new(ReplyMediaRef::image(
        uri,
        "https://chud.example/images/generated.jpg",
    ));

    let output = read_asset(
        &store,
        ClientToolCall {
            id: ToolUseId::new("call-1"),
            name: ToolName::new(READ_ASSET_TOOL),
            input: json!({ "uri": uri }),
        },
    )
    .await
    .expect("stored image should be readable");

    assert!(!output.is_error);
    assert_eq!(output.media.len(), 1);
    let ClientToolResultContent::Json { value } = &output.result else {
        panic!("expected json result");
    };
    assert_eq!(value["uri"], uri);
    assert_eq!(value["visible_to_model"], true);
    // read_asset attaches media to the model transcript; the JSON payload stays
    // metadata-only so traces and tool responses do not duplicate image bytes.
    assert_eq!(value["content"]["bytes_returned"], false);
    assert!(value.get("bytes").is_none());
    assert!(value.get("base64").is_none());
    assert!(value.get("data_url").is_none());
}

#[tokio::test]
async fn read_asset_does_not_queue_final_reply_attachment() {
    let uri = "file://images/generated.jpg";
    let store = ReplyMediaStore::new(ReplyMediaRef::image(
        uri,
        "https://chud.example/images/generated.jpg",
    ));
    let call = ClientToolCall {
        id: ToolUseId::new("call-1"),
        name: ToolName::new(READ_ASSET_TOOL),
        input: json!({ "uri": uri }),
    };
    let output = read_asset(&store, call.clone())
        .await
        .expect("stored image should be readable");
    let result = chudbot_api::ClientToolResult {
        tool_use_id: call.id.clone(),
        content: output.result,
        is_error: output.is_error,
    };
    let trace = ToolTrace::Client {
        trace: chudbot_api::ClientToolTrace {
            call,
            result,
            trace_response: output.trace_response,
            usage: output.usage,
        },
    };

    let media = generated_reply_media(&store, &[trace]).await;

    assert!(media.attachments.is_empty());
    assert!(media.public_urls.is_empty());
}

#[tokio::test]
async fn read_asset_rejects_video_without_loading_bytes() {
    let uri = "file://videos/generated.mp4";
    let store = ReplyMediaStore::new(ReplyMediaRef::video(
        uri,
        42,
        "https://chud.example/videos/generated.mp4",
    ));

    let error = read_asset(
        &store,
        ClientToolCall {
            id: ToolUseId::new("call-1"),
            name: ToolName::new(READ_ASSET_TOOL),
            input: json!({ "uri": uri }),
        },
    )
    .await
    .expect_err("videos should not be exposed through read");

    assert!(
        matches!(error, BotToolError::InvalidInput(message) if message.contains("only supports stored image assets"))
    );
}

#[tokio::test]
async fn read_asset_rejects_unsupported_image_mime_without_returning_public_url() {
    let uri = "file://images/upload.pdf";
    let store = ReplyMediaStore::new(ReplyMediaRef::image_with_mime(
        uri,
        "application/pdf",
        "https://chud.example/images/upload.pdf",
    ));

    let error = read_asset(
        &store,
        ClientToolCall {
            id: ToolUseId::new("call-1"),
            name: ToolName::new(READ_ASSET_TOOL),
            input: json!({ "uri": uri }),
        },
    )
    .await
    .expect_err("unsupported image MIME should not be exposed through read");
    assert!(
        matches!(error, BotToolError::InvalidInput(message) if message.contains("only supports stored image assets"))
    );

    let output = public_url_asset(
        &store,
        ClientToolCall {
            id: ToolUseId::new("call-2"),
            name: ToolName::new(PUBLIC_URL_ASSET_TOOL),
            input: json!({ "uri": uri }),
        },
    )
    .await
    .expect("public_url should return an availability payload");
    let ClientToolResultContent::Json { value } = &output.result else {
        panic!("expected json result");
    };
    assert_eq!(value["exists"], true);
    assert_eq!(value["available"], false);
    assert_eq!(value["public_url"], serde_json::Value::Null);
}

#[tokio::test]
async fn attach_asset_queues_supported_image_without_returning_bytes() {
    let uri = "file://images/generated.jpg";
    let store = ReplyMediaStore::new(ReplyMediaRef::image(
        uri,
        "https://chud.example/images/generated.jpg",
    ));

    let output = attach_asset(
        &store,
        ClientToolCall {
            id: ToolUseId::new("call-1"),
            name: ToolName::new(ATTACH_ASSET_TOOL),
            input: json!({ "uri": uri }),
        },
    )
    .await
    .expect("stored image should be attachable");

    assert!(!output.is_error);
    assert!(output.media.is_empty());
    let ClientToolResultContent::Json { value } = &output.result else {
        panic!("expected json result");
    };
    assert_eq!(value["uri"], uri);
    assert_eq!(value["attached"], true);
    assert!(value.get("bytes").is_none());
    assert!(value.get("base64").is_none());
    assert!(value.get("data_url").is_none());
}

#[tokio::test]
async fn attach_asset_rejects_video() {
    let uri = "file://videos/generated.mp4";
    let store = ReplyMediaStore::new(ReplyMediaRef::video(
        uri,
        42,
        "https://chud.example/videos/generated.mp4",
    ));

    let error = attach_asset(
        &store,
        ClientToolCall {
            id: ToolUseId::new("call-1"),
            name: ToolName::new(ATTACH_ASSET_TOOL),
            input: json!({ "uri": uri }),
        },
    )
    .await
    .expect_err("videos should not be attachable through attach");

    assert!(
        matches!(error, BotToolError::InvalidInput(message) if message.contains("only supports stored image assets"))
    );
}

#[tokio::test]
async fn explicit_attach_deduplicates_with_automatic_generated_attachment() {
    let uri = "file://images/generated.jpg";
    let public_url = "https://chud.example/images/generated.jpg";
    let store = ReplyMediaStore::new(ReplyMediaRef::image(uri, public_url));
    let traces = vec![generated_image_trace(uri, public_url), attach_trace(uri)];

    let media = generated_reply_media(&store, &traces).await;

    assert_eq!(media.attachments.len(), 1);
    assert_eq!(media.attachments[0].filename, "generated.jpg");
    assert!(media.public_urls.is_empty());
}

#[tokio::test]
async fn stat_asset_reports_missing_uri_without_error_result() {
    let store = ReplyMediaStore::new(ReplyMediaRef::image(
        "file://images/generated.jpg",
        "https://chud.example/images/generated.jpg",
    ));

    let output = stat_asset(
        &store,
        ClientToolCall {
            id: ToolUseId::new("call-1"),
            name: ToolName::new(STAT_ASSET_TOOL),
            input: json!({ "uri": "file://images/missing.jpg" }),
        },
    )
    .await
    .expect("stat should return a pass/fail payload");

    assert!(!output.is_error);
    assert!(output.media.is_empty());
    let ClientToolResultContent::Json { value } = &output.result else {
        panic!("expected json result");
    };
    assert_eq!(value["exists"], false);
    assert_eq!(value["uri"], "file://images/missing.jpg");
}

#[test_case(MediaCategory::Image, "image/png", true ; "png image")]
#[test_case(MediaCategory::Image, "image/jpeg; charset=binary", true ; "jpeg image with params")]
#[test_case(MediaCategory::Image, "IMAGE/WEBP", true ; "case insensitive webp")]
#[test_case(MediaCategory::Image, "image/gif", false ; "unsupported image mime")]
#[test_case(MediaCategory::Image, "video/mp4", false ; "image category with video mime")]
#[test_case(MediaCategory::Video, "video/mp4", false ; "video category")]
#[test_case(MediaCategory::Audio, "audio/ogg", false ; "audio category")]
fn model_transcript_media_support_matches_llm_image_inputs(
    category: MediaCategory,
    mime_type: &str,
    expected: bool,
) {
    let media = PromptMediaRef {
        metadata: MediaMetadata {
            category,
            name: "media.bin".to_string(),
            uri: MediaUri::new("file://media/generated.bin"),
            mime_type: mime_type.to_string(),
            size_bytes: 42,
        },
        public_url: PublicMediaUrl::new("https://chud.example/media/generated.bin"),
    };

    assert_eq!(model_transcript_supports_media(&media), expected);
}

// Audio tests cover attachment detection, injected context refs, wake-word
// gating, and the keyterm defaults passed to transcription providers.
#[test_case(None, "voice.dat", true, true ; "voice flag")]
#[test_case(Some("audio/ogg"), "voice.dat", false, true ; "audio content type")]
#[test_case(None, "voice.m4a", false, true ; "audio extension")]
#[test_case(Some("video/mp4"), "clip.mp4", false, false ; "video content type")]
#[test_case(None, "clip.mp4", false, false ; "ambiguous mp4 extension")]
fn detects_audio_attachments(
    content_type: Option<&str>,
    filename: &str,
    is_voice_message: bool,
    expected: bool,
) {
    assert_eq!(
        looks_like_audio(content_type, filename, is_voice_message),
        expected
    );
}

#[test]
fn injects_audio_refs_into_message_json() {
    let mut value = json!({
        "content": "<@111> voice note",
        "attachments": [
            { "filename": "image.png" },
            { "filename": "voice.ogg" }
        ]
    });
    let audio = PromptMediaRef::boxed_audio("file://audio/voice.ogg");
    let saved = StoredAttachmentMedia {
        attachment_index: 1,
        media: audio,
    };

    inject_audio_attachment_refs(&mut value, &[saved]);

    assert_eq!(value["audio_attachments"][0], "file://audio/voice.ogg");
    assert_eq!(
        value["attachments"][1]["audio_uri"],
        "file://audio/voice.ogg"
    );
    assert!(value["attachments"][0].get("audio_uri").is_none());
}

#[test]
fn spoken_wake_word_detection_uses_configured_term() {
    assert!(text_mentions_wake_word(
        "Hello, Chudbot, can you hear me?",
        "chudbot"
    ));
    assert!(text_mentions_wake_word("hello chud bot", "Chudbot"));
    assert!(text_mentions_wake_word(
        "Computer, status report.",
        "computer"
    ));
    assert!(!text_mentions_wake_word("hello chat bot", "Chudbot"));
    assert!(!text_mentions_wake_word("hello Chudbot", ""));
}

#[test]
fn audio_transcription_default_keyterms_use_configured_wake_word() {
    let binding = TranscriptionBinding {
        provider: ProviderName::new("grok_audio"),
        model: None,
        wake_word: Some(" Chudbot ".to_string()),
    };

    assert_eq!(
        audio_transcription_default_keyterms(&binding),
        vec!["Chudbot".to_string()]
    );
}

#[test]
fn default_audio_keyterms_preserve_and_deduplicate_explicit_terms() {
    let defaults = vec!["Chudbot".to_string(), " ".to_string()];
    let mut keyterms = vec!["Universe".to_string()];

    append_default_audio_keyterms(&mut keyterms, &defaults);

    assert_eq!(
        keyterms,
        vec!["Universe".to_string(), "Chudbot".to_string()]
    );

    append_default_audio_keyterms(&mut keyterms, &[" chudbot ".to_string()]);

    assert_eq!(
        keyterms,
        vec!["Universe".to_string(), "Chudbot".to_string()]
    );
}

#[test]
fn no_mention_audio_preflight_requires_wake_word_outside_conversation() {
    let without_wake_word = TranscriptionBinding {
        provider: ProviderName::new("grok_audio"),
        model: None,
        wake_word: None,
    };
    let with_wake_word = TranscriptionBinding {
        provider: ProviderName::new("grok_audio"),
        model: None,
        wake_word: Some("Chudbot".to_string()),
    };

    assert!(!no_mention_audio_preflight_enabled_for_binding(
        Some(&without_wake_word),
        false
    ));
    assert!(no_mention_audio_preflight_enabled_for_binding(
        Some(&without_wake_word),
        true
    ));
    assert!(no_mention_audio_preflight_enabled_for_binding(
        Some(&with_wake_word),
        false
    ));
    assert!(!no_mention_audio_preflight_enabled_for_binding(None, true));
}

#[test]
fn automatic_audio_context_omits_audio_uri_when_hidden_from_model() {
    let transcription = IncomingAudioTranscription {
        attachment_index: 0,
        audio_uri: None,
        text: "Hello, Chudbot.".to_string(),
        language: Some("en".to_string()),
        duration_seconds: 2.6,
        result: json!({ "text": "Hello, Chudbot." }),
        trace_response: json!({ "transcription": { "text": "Hello, Chudbot." } }),
        usage: Vec::new(),
    };

    let value = audio_transcription_context_json(&transcription);

    assert_eq!(value["text"], "Hello, Chudbot.");
    assert!(value.get("audio_uri").is_none());
}

// Transcript replay tests preserve provider continuations and client-tool
// history so follow-up turns can resume without re-running earlier tools.
#[tokio::test]
async fn transcript_media_refs_are_sanitized_from_replies() {
    let media = PromptMediaRef::boxed(
        "file://images/generated.jpg",
        "https://chud.example/images/generated.jpg",
    );
    let transcript = Transcript {
        id: None,
        instructions: None,
        turns: vec![TranscriptTurn {
            role: TurnRole::User,
            blocks: vec![
                ContentBlock::Media {
                    media: media.clone(),
                },
                ContentBlock::Media { media },
            ],
            metadata: serde_json::Value::Null,
        }],
    };

    let refs = media_reply_refs_from_transcript(&transcript).await;
    assert_eq!(
        refs,
        vec![
            "file://images/generated.jpg".to_string(),
            "https://chud.example/images/generated.jpg".to_string()
        ]
    );
    let reply = strip_generated_media_refs(
        "Done.\n\nhttps://chud.example/images/generated.jpg\nfile://images/generated.jpg",
        &refs,
    );

    assert_eq!(reply, "Done.");
}

#[tokio::test]
async fn transcript_reply_refs_include_replayed_tool_result_media() {
    let trace = generated_image_trace(
        "file://videos/generated.mp4",
        "https://chud.example/videos/generated.mp4",
    );
    let mut transcript = Transcript::new();
    append_client_tool_replay(&mut transcript, &[trace]);

    let refs = media_reply_refs_from_transcript(&transcript).await;

    assert_eq!(refs, vec!["file://videos/generated.mp4".to_string()]);
}

#[test]
fn client_tool_trace_replays_as_call_then_result() {
    let trace = generated_image_trace(
        "file://images/generated.jpg",
        "https://chud.example/images/generated.jpg",
    );
    let mut transcript = Transcript::new();

    append_client_tool_replay(&mut transcript, &[trace]);

    assert_eq!(transcript.turns.len(), 2);
    assert_eq!(transcript.turns[0].role, TurnRole::Assistant);
    assert_eq!(transcript.turns[1].role, TurnRole::User);
    let [ContentBlock::ClientToolCall(call)] = transcript.turns[0].blocks.as_slice() else {
        panic!("expected client tool call replay");
    };
    assert_eq!(call.name.as_str(), "generate_image");
    let [ContentBlock::ClientToolResult(result)] = transcript.turns[1].blocks.as_slice() else {
        panic!("expected client tool result replay");
    };
    assert_eq!(result.tool_use_id, call.id);
    let ClientToolResultContent::Json { value } = &result.content else {
        panic!("expected json result");
    };
    assert_eq!(value["uri"], "file://images/generated.jpg");
}

#[test]
fn model_step_replay_uses_provider_continuations_and_matching_tool_results() {
    let trace = generated_image_trace(
        "file://images/generated.jpg",
        "https://chud.example/images/generated.jpg",
    );
    let provider = ProviderName::new("xai");
    let model_steps = vec![
        ModelStepTrace {
            ordinal: 0,
            kind: ModelStepKind::ClientTools,
            provider: provider.clone(),
            model: ModelId::new("grok-4.3"),
            continuation: Some(chudbot_api::ProviderContinuation {
                provider: provider.clone(),
                data: json!([
                    {
                        "type": "reasoning",
                        "id": "rs_1",
                        "encrypted_content": "BLOB_1",
                        "summary": [{ "type": "summary_text", "text": "Need an image." }]
                    },
                    {
                        "type": "function_call",
                        "call_id": "call-1",
                        "name": "generate_image",
                        "arguments": "{\"prompt\":\"a worm\"}"
                    }
                ]),
            }),
        },
        ModelStepTrace {
            ordinal: 1,
            kind: ModelStepKind::Final,
            provider: provider.clone(),
            model: ModelId::new("grok-4.3"),
            continuation: Some(chudbot_api::ProviderContinuation {
                provider,
                data: json!([
                    { "type": "reasoning", "id": "rs_2", "encrypted_content": "BLOB_2" },
                    {
                        "type": "message",
                        "id": "msg_2",
                        "role": "assistant",
                        "content": [{ "type": "output_text", "text": "Done." }]
                    }
                ]),
            }),
        },
    ];
    let mut transcript = Transcript::new();

    let replayed = append_model_step_replay(&mut transcript, &model_steps, &[trace], Some("Done."));

    assert!(replayed);
    assert_eq!(transcript.turns.len(), 3);
    assert_eq!(transcript.turns[0].role, TurnRole::Assistant);
    assert_eq!(transcript.turns[1].role, TurnRole::User);
    assert_eq!(transcript.turns[2].role, TurnRole::Assistant);
    assert!(matches!(
        transcript.turns[0].blocks.as_slice(),
        [ContentBlock::Continuation(_)]
    ));
    let [ContentBlock::ClientToolResult(result)] = transcript.turns[1].blocks.as_slice() else {
        panic!("expected matching client tool result");
    };
    assert_eq!(result.tool_use_id.as_str(), "call-1");
    assert!(matches!(
        transcript.turns[2].blocks.as_slice(),
        [
            ContentBlock::Continuation(_),
            ContentBlock::Text { text }
        ] if text == "Done."
    ));
}

#[test]
fn model_step_replay_matches_anthropic_tool_use_results() {
    let trace = generated_image_trace(
        "file://images/generated.jpg",
        "https://chud.example/images/generated.jpg",
    );
    let provider = ProviderName::new("anthropic");
    let model_steps = vec![
        ModelStepTrace {
            ordinal: 0,
            kind: ModelStepKind::ClientTools,
            provider: provider.clone(),
            model: ModelId::new("claude-haiku-4-5"),
            continuation: Some(chudbot_api::ProviderContinuation {
                provider: provider.clone(),
                data: json!([
                    { "type": "text", "text": "Need an image." },
                    {
                        "type": "tool_use",
                        "id": "call-1",
                        "name": "generate_image",
                        "input": { "prompt": "a worm" }
                    }
                ]),
            }),
        },
        ModelStepTrace {
            ordinal: 1,
            kind: ModelStepKind::Final,
            provider,
            model: ModelId::new("claude-haiku-4-5"),
            continuation: None,
        },
    ];
    let mut transcript = Transcript::new();

    let replayed = append_model_step_replay(&mut transcript, &model_steps, &[trace], Some("Done."));

    assert!(replayed);
    assert_eq!(transcript.turns.len(), 3);
    assert_eq!(transcript.turns[0].role, TurnRole::Assistant);
    assert_eq!(transcript.turns[1].role, TurnRole::User);
    assert_eq!(transcript.turns[2].role, TurnRole::Assistant);
    let [ContentBlock::ClientToolResult(result)] = transcript.turns[1].blocks.as_slice() else {
        panic!("expected matching Anthropic client tool result");
    };
    assert_eq!(result.tool_use_id.as_str(), "call-1");
    assert!(matches!(
        transcript.turns[2].blocks.as_slice(),
        [ContentBlock::Text { text }] if text == "Done."
    ));
}

#[test]
fn generated_media_replays_after_assistant_message() {
    let turn_id = TurnId::new();
    let trace = generated_image_trace(
        "file://images/generated.jpg",
        "https://chud.example/images/generated.jpg",
    );
    let media = PromptMediaRef::boxed(
        "file://images/generated.jpg",
        "https://chud.example/images/generated.jpg",
    );
    let mut transcript = Transcript::new();
    transcript.push(TranscriptTurn::text(TurnRole::User, "draw an image"));
    append_client_tool_replay(&mut transcript, &[trace]);
    transcript.push(TranscriptTurn::text(
        TurnRole::Assistant,
        "Done. Image generated and attached.",
    ));

    append_generated_media_replay(
        &mut transcript,
        turn_id,
        vec!["file://images/generated.jpg".to_string()],
        vec![ContentBlock::Media { media }],
    );

    assert_eq!(transcript.turns.len(), 5);
    assert_eq!(transcript.turns[0].role, TurnRole::User);
    assert_eq!(transcript.turns[1].role, TurnRole::Assistant);
    assert_eq!(transcript.turns[2].role, TurnRole::User);
    assert_eq!(transcript.turns[3].role, TurnRole::Assistant);
    assert_eq!(transcript.turns[4].role, TurnRole::User);
    let expected_id = turn_transcript_message_id(turn_id, "assistant_media");
    assert_eq!(
        transcript.turns[4].metadata["id"].as_str(),
        Some(expected_id.as_str())
    );
    let [ContentBlock::Text { text }, ContentBlock::Media { .. }] =
        transcript.turns[4].blocks.as_slice()
    else {
        panic!("expected generated media replay note and media");
    };
    assert!(text.contains("file://images/generated.jpg"));
    assert!(text.contains("reference_images"));
}

// Replay ownership and reply-formatting tests pin Discord-visible text behavior:
// trace-link style, bare mention repair, and thread-threshold decisions.
#[test_case("platform:message:message-1:image:0", true ; "platform attachment")]
#[test_case("platform:quoted:message-1:image:0", true ; "quoted platform attachment")]
#[test_case("generate_image", false ; "generated image")]
fn replay_asset_user_turn_ownership(source: &str, expected: bool) {
    let asset = TurnAsset {
        uri: MediaUri::new("file://images/image.jpg"),
        turn_id: TurnId::new(),
        source: source.to_string(),
        mime_type: Some("image/jpeg".to_string()),
    };

    assert_eq!(replay_asset_belongs_to_user_turn(&asset), expected);
}

#[test]
fn client_tool_replay_skips_non_client_traces() {
    let mut transcript = Transcript::new();
    append_client_tool_replay(&mut transcript, &[]);

    assert!(transcript.turns.is_empty());
}

#[test]
fn formats_new_conversation_reply_with_legacy_trace_link() {
    let conversation_id = ConversationId::new();

    let reply = format_reply_content("answer", true, conversation_id, "https://chud.example/");

    assert_eq!(
        reply,
        format!("answer\n\n-# 🔎 [full trace](https://chud.example/c/{conversation_id})")
    );
}

#[test]
fn full_trace_link_markdown_uses_discord_reply_style() {
    let conversation_id = ConversationId::new();

    let link = full_trace_link_markdown("https://chud.example/", conversation_id);

    assert_eq!(
        link,
        format!("-# 🔎 [full trace](https://chud.example/c/{conversation_id})")
    );
}

#[test]
fn trace_link_prompt_guidance_includes_url_and_reply_style() {
    let conversation_id = ConversationId::new();
    let url = format!("https://chud.example/c/{conversation_id}");

    let guidance = trace_link_prompt_guidance("https://chud.example/", conversation_id);

    assert!(guidance.contains(&format!("Full trace URL for this conversation: {url}.")));
    assert!(guidance.contains("trace link"));
    assert!(guidance.contains("full trace link"));
    assert!(guidance.contains(&format!("-# 🔎 [full trace]({url})")));
}

#[test]
fn formats_continuation_reply_without_trace_link() {
    let reply = format_reply_content(
        "talk to @123456789012345678",
        false,
        ConversationId::new(),
        "https://chud.example",
    );

    assert_eq!(reply, "talk to <@123456789012345678>");
}

#[test]
fn rendered_lines_count_short_blank_and_wrapped_rows() {
    assert_eq!(rendered_line_count("a\nb\nc"), 3);
    assert_eq!(rendered_line_count("a\n\nb"), 3);

    let wrapped = "x".repeat(THREAD_REPLY_WRAP_WIDTH * 3);
    assert_eq!(rendered_line_count(&wrapped), 3);

    let mixed = format!("hi\n{}", "y".repeat(THREAD_REPLY_WRAP_WIDTH + 1));
    assert_eq!(rendered_line_count(&mixed), 3);
}

#[test]
fn should_thread_respects_char_and_visible_line_thresholds() {
    let big = "x".repeat(DEFAULT_THREAD_THRESHOLD_CHARS + 1);
    assert!(!should_thread(
        false,
        &big,
        DEFAULT_THREAD_THRESHOLD_CHARS,
        DEFAULT_THREAD_THRESHOLD_LINES,
    ));
    assert!(should_thread(
        true,
        &big,
        DEFAULT_THREAD_THRESHOLD_CHARS,
        DEFAULT_THREAD_THRESHOLD_LINES,
    ));

    let tall = (1..=24)
        .map(|index| format!("{index}. short line"))
        .collect::<Vec<_>>()
        .join("\n");
    assert!(tall.chars().count() < DEFAULT_THREAD_THRESHOLD_CHARS);
    assert!(rendered_line_count(&tall) > DEFAULT_THREAD_THRESHOLD_LINES);
    assert!(should_thread(
        true,
        &tall,
        DEFAULT_THREAD_THRESHOLD_CHARS,
        DEFAULT_THREAD_THRESHOLD_LINES,
    ));
    assert!(!should_thread(
        true,
        "hi",
        DEFAULT_THREAD_THRESHOLD_CHARS,
        DEFAULT_THREAD_THRESHOLD_LINES,
    ));
}

#[test]
fn rejects_video_duration_over_provider_limit() {
    let error =
        tool_optional_u8_bounded(&json!({ "duration_seconds": 16 }), "duration_seconds", 15)
            .expect_err("duration should be capped");

    assert!(error.to_string().contains("at most 15"));
}
