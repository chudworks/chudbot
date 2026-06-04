//! xAI image and video generation implementation.

use base64::Engine;
use base64::engine::general_purpose::STANDARD as B64;
use chudbot_api::{
    CostAmount, GeneratedImage, ImageGenerator, ImageRequest, MediaRef, ModelId, ProviderName,
    UsageRecord, UsageSubject, VideoGenerator, VideoJobId, VideoJobStatus, VideoMeta, VideoRequest,
};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::{XaiClient, XaiError, truncate_body};

const IMAGE_STANDARD_MODEL: &str = "grok-imagine-image";
const IMAGE_QUALITY_MODEL: &str = "grok-imagine-image-quality";
const VIDEO_DEFAULT_MODEL: &str = "grok-imagine-video";

impl ImageGenerator for XaiClient {
    type Error = XaiError;

    fn backend_name(&self) -> &ProviderName {
        self.provider_name()
    }

    #[tracing::instrument(name = "xai.generate_image", skip_all)]
    async fn generate_image(&self, request: ImageRequest) -> Result<GeneratedImage, Self::Error> {
        tracing::debug!(
            prompt_chars = request.prompt.chars().count(),
            references = request.references.len(),
            aspect_ratio = ?request.aspect_ratio.as_deref(),
            model = ?request.model.as_ref().map(ModelId::as_str),
            "building xAI image request"
        );
        let mut references = Vec::with_capacity(request.references.len());
        for reference in &request.references {
            references.push(media_provider_url(reference.as_ref()).await?);
        }
        let model = resolve_image_model(request.model.as_ref().map(ModelId::as_str));
        let body = build_image_body(
            model,
            &request.prompt,
            request.aspect_ratio.as_deref(),
            references,
        );
        let endpoint = if request.references.is_empty() {
            "/images/generations"
        } else {
            "/images/edits"
        };
        let parsed: ImagesResponse = self.post_json(endpoint, &body, "imagegen[xai]").await?;
        let first = parsed
            .data
            .into_iter()
            .next()
            .ok_or_else(|| XaiError::Decode("image response had no data".into()))?;
        let b64 = first
            .b64_json
            .ok_or_else(|| XaiError::Decode("image response item lacked b64_json".into()))?;
        let bytes = B64
            .decode(b64.as_bytes())
            .map_err(|e| XaiError::Decode(format!("base64: {e}")))?;
        let mime_type = first
            .mime_type
            .or(first.content_type)
            .unwrap_or_else(|| "image/jpeg".to_string());
        let model = parsed
            .model
            .as_deref()
            .map(ModelId::new)
            .unwrap_or_else(|| ModelId::new(model));
        let usage: Vec<UsageRecord> = usage_from_xai_media(
            self.provider_name(),
            Some(model.clone()),
            UsageSubject::ImageGeneration,
            parsed.usage.as_ref(),
        )
        .into_iter()
        .collect();
        tracing::info!(
            model = %model,
            mime_type = %mime_type,
            bytes = bytes.len(),
            revised_prompt = first.revised_prompt.is_some(),
            usage_records = usage.len(),
            "xAI image generated"
        );
        Ok(GeneratedImage {
            bytes,
            mime_type,
            model,
            revised_prompt: first.revised_prompt,
            usage,
        })
    }
}

impl VideoGenerator for XaiClient {
    type Error = XaiError;

    fn backend_name(&self) -> &ProviderName {
        self.provider_name()
    }

    #[tracing::instrument(name = "xai.submit_video", skip_all)]
    async fn submit_video(&self, request: VideoRequest) -> Result<VideoJobId, Self::Error> {
        let model = request
            .model
            .as_ref()
            .map(ModelId::as_str)
            .unwrap_or(VIDEO_DEFAULT_MODEL);
        tracing::debug!(
            prompt_chars = request.prompt.chars().count(),
            model = %model,
            has_image = request.image.is_some(),
            duration_seconds = ?request.duration_seconds,
            aspect_ratio = ?request.aspect_ratio.as_deref(),
            resolution = ?request.resolution.as_deref(),
            "building xAI video submit request"
        );
        let mut body = json!({
            "model": model,
            "prompt": request.prompt,
            "resolution": request.resolution.unwrap_or_else(|| "480p".to_string()),
        });
        if let Some(duration) = request.duration_seconds {
            body["duration"] = json!(duration);
        }
        if let Some(aspect_ratio) = request.aspect_ratio {
            body["aspect_ratio"] = json!(aspect_ratio);
        }
        if let Some(image) = request.image.as_ref() {
            body["image"] = json!({ "url": media_provider_url(image.as_ref()).await? });
        }
        let policy = chudbot_api::retry::RetryPolicy {
            retry_network: false,
            ..chudbot_api::retry::RetryPolicy::default()
        };
        let parsed: SubmitVideoResponse = self
            .post_json_with_policy("/videos/generations", &body, policy, "videogen[xai].submit")
            .await?;
        let job = VideoJobId::new(parsed.request_id);
        tracing::info!(job = %job, "xAI video submitted");
        Ok(job)
    }

    #[tracing::instrument(name = "xai.check_video", skip_all, fields(job = %job))]
    async fn check_video(&self, job: VideoJobId) -> Result<VideoJobStatus, Self::Error> {
        let endpoint = format!("/videos/{}", job.as_str());
        let parsed: PollVideoResponse = self.get_json(&endpoint, "videogen[xai].check").await?;
        tracing::debug!(status = %parsed.status, "xAI video status received");
        match parsed.status.as_str() {
            "done" => {
                let video = parsed.video.ok_or_else(|| {
                    XaiError::Decode("video poll returned done without video metadata".into())
                })?;
                Ok(VideoJobStatus::Done {
                    meta: VideoMeta {
                        url: video.url,
                        duration_seconds: video.duration,
                        usage: usage_from_xai_media(
                            self.provider_name(),
                            parsed
                                .model
                                .as_deref()
                                .or(video.model.as_deref())
                                .map(ModelId::new),
                            UsageSubject::VideoGeneration,
                            parsed.usage.as_ref().or(video.usage.as_ref()),
                        )
                        .into_iter()
                        .collect(),
                    },
                })
            }
            "failed" => Ok(VideoJobStatus::Failed {
                message: parsed
                    .error
                    .map(|e| e.message)
                    .unwrap_or_else(|| "(no message)".to_string()),
            }),
            "expired" => Ok(VideoJobStatus::Expired),
            _ => Ok(VideoJobStatus::Pending),
        }
    }

    #[tracing::instrument(name = "xai.download_video", skip_all)]
    async fn download_video(&self, url: String) -> Result<Vec<u8>, Self::Error> {
        let resp = chudbot_api::retry::with_retry(
            chudbot_api::retry::RetryPolicy::default(),
            "videogen[xai].download",
            || {
                let request = self.http().get(&url);
                async move {
                    let resp = request.send().await.map_err(|e| {
                        tracing::warn!(error = %e, "xAI video download transport error");
                        XaiError::Transport(e.to_string())
                    })?;
                    let status = resp.status();
                    if !status.is_success() {
                        let body = truncate_body(resp.text().await.unwrap_or_default(), 600);
                        tracing::warn!(
                            status = status.as_u16(),
                            body_chars = body.chars().count(),
                            "xAI video download returned non-success status"
                        );
                        return Err(XaiError::Api {
                            status: status.as_u16(),
                            body,
                        });
                    }
                    Ok(resp)
                }
            },
        )
        .await?;
        let bytes = resp.bytes().await.map(|b| b.to_vec()).map_err(|e| {
            tracing::warn!(error = %e, "failed to read xAI video download body");
            XaiError::Transport(e.to_string())
        })?;
        tracing::debug!(bytes = bytes.len(), "xAI video downloaded");
        Ok(bytes)
    }
}

fn build_image_body(
    model: &str,
    prompt: &str,
    aspect_ratio: Option<&str>,
    references: Vec<String>,
) -> Value {
    let mut body = json!({
        "model": model,
        "prompt": prompt,
        "response_format": "b64_json",
        "n": 1,
    });
    if let Some(aspect_ratio) = aspect_ratio {
        body["aspect_ratio"] = json!(aspect_ratio);
    }
    let ref_count = references.len();
    if ref_count > 0 {
        let mut refs = references
            .into_iter()
            .map(|url| json!({ "url": url, "type": "image_url" }));
        if ref_count == 1 {
            body["image"] = refs.next().unwrap_or(Value::Null);
        } else {
            body["images"] = Value::Array(refs.collect());
        }
    }
    body
}

fn resolve_image_model(model: Option<&str>) -> &str {
    let Some(model) = model else {
        return IMAGE_STANDARD_MODEL;
    };
    match model.to_ascii_lowercase().as_str() {
        "standard" => IMAGE_STANDARD_MODEL,
        "quality" => IMAGE_QUALITY_MODEL,
        _ => model,
    }
}

pub(crate) async fn media_provider_url(media: &dyn MediaRef) -> Result<String, XaiError> {
    match media.public_url().await {
        Ok(url) => {
            tracing::debug!(
                uri = %media.uri(),
                category = ?media.category(),
                "resolved media public URL for xAI"
            );
            Ok(url.to_string())
        }
        Err(public_error) => match media.load().await {
            Ok(loaded) => {
                tracing::debug!(
                    uri = %media.uri(),
                    category = ?media.category(),
                    bytes = loaded.bytes.len(),
                    mime_type = loaded.media.mime_type(),
                    "inlined media bytes for xAI"
                );
                Ok(data_uri(loaded.media.mime_type(), &loaded.bytes))
            }
            Err(load_error) => {
                tracing::warn!(
                    uri = %media.uri(),
                    category = ?media.category(),
                    public_error = %public_error,
                    load_error = %load_error,
                    "failed to resolve media for xAI"
                );
                Err(XaiError::Reference(format!(
                    "media `{}` has no public URL ({public_error}) and could not be loaded ({load_error})",
                    media.uri()
                )))
            }
        },
    }
}

fn data_uri(mime_type: &str, bytes: &[u8]) -> String {
    format!("data:{mime_type};base64,{}", B64.encode(bytes))
}

pub(crate) fn usage_from_xai_media(
    provider: &ProviderName,
    model: Option<ModelId>,
    subject: UsageSubject,
    usage: Option<&Value>,
) -> Option<UsageRecord> {
    let raw = usage?.clone();
    let parsed = serde_json::from_value::<Usage>(raw.clone()).ok()?;
    let cost = (parsed.cost_in_usd_ticks > 0).then(|| CostAmount {
        amount: parsed.cost_in_usd_ticks.to_string(),
        unit: "usd_ticks".to_string(),
        estimated: false,
    });
    Some(UsageRecord {
        provider: provider.clone(),
        model,
        subject,
        input_tokens: Some(parsed.input_tokens),
        cached_input_tokens: Some(parsed.input_tokens_details.cached_tokens),
        output_tokens: Some(parsed.output_tokens),
        reasoning_tokens: Some(parsed.output_tokens_details.reasoning_tokens),
        total_tokens: Some(parsed.total_tokens),
        cost,
        raw: Some(raw),
    })
}

#[derive(Deserialize)]
struct ImagesResponse {
    #[serde(default)]
    data: Vec<ImageResponseItem>,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    usage: Option<Value>,
}

#[derive(Deserialize)]
struct ImageResponseItem {
    #[serde(default)]
    b64_json: Option<String>,
    #[serde(default)]
    revised_prompt: Option<String>,
    #[serde(default)]
    mime_type: Option<String>,
    #[serde(default)]
    content_type: Option<String>,
}

#[derive(Deserialize)]
struct SubmitVideoResponse {
    request_id: String,
}

#[derive(Deserialize)]
struct PollVideoResponse {
    status: String,
    #[serde(default)]
    video: Option<RawVideoMeta>,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    usage: Option<Value>,
    #[serde(default)]
    error: Option<PollError>,
}

#[derive(Deserialize)]
struct RawVideoMeta {
    url: String,
    #[serde(default)]
    duration: Option<f32>,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    usage: Option<Value>,
}

#[derive(Deserialize)]
struct PollError {
    #[serde(default)]
    message: String,
}

#[derive(Deserialize, Debug, Default)]
struct Usage {
    #[serde(default)]
    input_tokens: u64,
    #[serde(default)]
    input_tokens_details: TokenDetails,
    #[serde(default)]
    output_tokens: u64,
    #[serde(default)]
    output_tokens_details: TokenDetails,
    #[serde(default)]
    total_tokens: u64,
    #[serde(default)]
    cost_in_usd_ticks: u64,
}

#[derive(Deserialize, Debug, Default)]
struct TokenDetails {
    #[serde(default)]
    cached_tokens: u64,
    #[serde(default)]
    reasoning_tokens: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn xai_media_usage_preserves_cost_ticks_and_token_details() {
        let provider = ProviderName::new("xai");
        let usage = usage_from_xai_media(
            &provider,
            Some(ModelId::new("grok-imagine-image")),
            UsageSubject::ImageGeneration,
            Some(&json!({
                "input_tokens": 10,
                "input_tokens_details": { "cached_tokens": 3 },
                "output_tokens": 20,
                "output_tokens_details": { "reasoning_tokens": 4 },
                "total_tokens": 30,
                "cost_in_usd_ticks": 123
            })),
        )
        .unwrap();

        assert_eq!(usage.provider, provider);
        assert_eq!(
            usage.model.as_ref().map(ModelId::as_str),
            Some("grok-imagine-image")
        );
        assert!(matches!(usage.subject, UsageSubject::ImageGeneration));
        assert_eq!(usage.input_tokens, Some(10));
        assert_eq!(usage.cached_input_tokens, Some(3));
        assert_eq!(usage.output_tokens, Some(20));
        assert_eq!(usage.reasoning_tokens, Some(4));
        assert_eq!(usage.total_tokens, Some(30));
        let cost = usage.cost.unwrap();
        assert_eq!(cost.amount, "123");
        assert_eq!(cost.unit, "usd_ticks");
        assert!(!cost.estimated);
        assert!(usage.raw.is_some());
    }
}
