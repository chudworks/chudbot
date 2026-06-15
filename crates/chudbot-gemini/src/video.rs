//! Google Gemini API Veo video generation implementation.

use chudbot_api::{
    MediaRef, ProviderName, VideoGenerator, VideoJobId, VideoJobStatus, VideoMeta, VideoRequest,
};
use serde_json::{Value, json};

use crate::{GeminiClient, GeminiError, get_field, inline_media, json_strip_nulls, truncate_body};

const DEFAULT_VIDEO_MODEL: &str = "veo-3.1-generate-preview";

impl VideoGenerator for GeminiClient {
    type Error = GeminiError;

    fn backend_name(&self) -> &ProviderName {
        self.provider_name()
    }

    #[tracing::instrument(name = "gemini.submit_video", skip_all)]
    async fn submit_video(&self, request: VideoRequest) -> Result<VideoJobId, Self::Error> {
        let model = request
            .model
            .as_ref()
            .map(chudbot_api::ModelId::as_str)
            .unwrap_or(DEFAULT_VIDEO_MODEL);
        tracing::debug!(
            prompt_chars = request.prompt.chars().count(),
            model = %model,
            has_image = request.image.is_some(),
            duration_seconds = ?request.duration_seconds,
            aspect_ratio = ?request.aspect_ratio.as_deref(),
            resolution = ?request.resolution.as_deref(),
            "building Gemini video submit request"
        );
        let instance = video_instance(&request).await?;
        let parameters = video_parameters(&request);
        let body = json_strip_nulls(json!({
            "instances": [instance],
            "parameters": parameters,
        }));
        let endpoint = format!("/models/{model}:predictLongRunning");
        let parsed: Value = self
            .post_json_with_policy(
                &endpoint,
                &body,
                chudbot_api::retry::RetryPolicy {
                    retry_network: false,
                    ..chudbot_api::retry::RetryPolicy::default()
                },
                "videogen[gemini].submit",
            )
            .await?;
        let name = parsed
            .get("name")
            .and_then(Value::as_str)
            .ok_or_else(|| GeminiError::Decode("video submit response lacked name".to_string()))?;
        let job = VideoJobId::new(name);
        tracing::info!(job = %job, "Gemini video submitted");
        Ok(job)
    }

    #[tracing::instrument(name = "gemini.check_video", skip_all, fields(job = %job))]
    async fn check_video(&self, job: VideoJobId) -> Result<VideoJobStatus, Self::Error> {
        let endpoint = format!("/{}", job.as_str());
        let parsed: Value = self.get_json(&endpoint, "videogen[gemini].check").await?;
        if !parsed.get("done").and_then(Value::as_bool).unwrap_or(false) {
            return Ok(VideoJobStatus::Pending);
        }
        if let Some(error) = parsed.get("error") {
            return Ok(VideoJobStatus::Failed {
                message: error_message(error),
            });
        }
        let response = parsed.get("response").ok_or_else(|| {
            GeminiError::Decode("done video operation lacked response".to_string())
        })?;
        let url = video_uri_from_response(response).ok_or_else(|| {
            GeminiError::Decode("done video operation lacked generated video URI".to_string())
        })?;
        Ok(VideoJobStatus::Done {
            meta: VideoMeta {
                url,
                duration_seconds: None,
                usage: Vec::new(),
            },
        })
    }

    #[tracing::instrument(name = "gemini.download_video", skip_all)]
    async fn download_video(&self, url: String) -> Result<Vec<u8>, Self::Error> {
        let resp = chudbot_api::retry::with_retry(
            chudbot_api::retry::RetryPolicy::default(),
            "videogen[gemini].download",
            || {
                let request = self
                    .http()
                    .get(&url)
                    .header("x-goog-api-key", self.api_key());
                async move {
                    let resp = request.send().await.map_err(|e| {
                        tracing::warn!(error = %e, "Gemini video download transport error");
                        GeminiError::Transport(e.to_string())
                    })?;
                    let status = resp.status();
                    if !status.is_success() {
                        let body = truncate_body(resp.text().await.unwrap_or_default(), 600);
                        tracing::warn!(
                            status = status.as_u16(),
                            body_chars = body.chars().count(),
                            "Gemini video download returned non-success status"
                        );
                        return Err(GeminiError::Api {
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
            tracing::warn!(error = %e, "failed to read Gemini video download body");
            GeminiError::Transport(e.to_string())
        })?;
        tracing::debug!(bytes = bytes.len(), "Gemini video downloaded");
        Ok(bytes)
    }
}

async fn video_instance(request: &VideoRequest) -> Result<Value, GeminiError> {
    let mut instance = serde_json::Map::new();
    instance.insert("prompt".to_string(), Value::String(request.prompt.clone()));
    if let Some(image) = request.image.as_ref() {
        instance.insert("image".to_string(), video_image(image.as_ref()).await?);
    }
    Ok(Value::Object(instance))
}

async fn video_image(media: &dyn MediaRef) -> Result<Value, GeminiError> {
    let mime_type = media.mime_type();
    if !mime_type.starts_with("image/") {
        return Err(GeminiError::Reference(format!(
            "media `{}` has MIME type `{mime_type}`, but Gemini video generation accepts an image input here",
            media.uri()
        )));
    }
    Ok(json!({
        "inlineData": inline_media(media).await?["inlineData"].clone(),
    }))
}

fn video_parameters(request: &VideoRequest) -> Option<Value> {
    let value = json_strip_nulls(json!({
        "aspectRatio": request.aspect_ratio.as_deref(),
        "durationSeconds": request.duration_seconds,
        "resolution": request.resolution.as_deref(),
    }));
    match &value {
        Value::Object(map) if map.is_empty() => None,
        _ => Some(value),
    }
}

fn error_message(error: &Value) -> String {
    error
        .get("message")
        .and_then(Value::as_str)
        .unwrap_or("(no message)")
        .to_string()
}

fn video_uri_from_response(response: &Value) -> Option<String> {
    for path in [
        &[
            "generateVideoResponse",
            "generatedSamples",
            "0",
            "video",
            "uri",
        ][..],
        &[
            "generate_video_response",
            "generated_samples",
            "0",
            "video",
            "uri",
        ][..],
        &["generatedVideos", "0", "video", "uri"][..],
        &["generated_videos", "0", "video", "uri"][..],
    ] {
        if let Some(uri) = value_path(response, path).and_then(Value::as_str) {
            return Some(uri.to_string());
        }
    }
    None
}

fn value_path<'a>(mut value: &'a Value, path: &[&str]) -> Option<&'a Value> {
    for segment in path {
        value = if let Ok(index) = segment.parse::<usize>() {
            value.as_array()?.get(index)?
        } else {
            get_field(value, segment, segment)?
        };
    }
    Some(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn video_parameters_preserve_shared_request_controls() {
        let request = VideoRequest {
            prompt: "test".to_string(),
            image: None,
            duration_seconds: Some(8),
            aspect_ratio: Some("9:16".to_string()),
            resolution: Some("1080p".to_string()),
            model: None,
        };

        let params = video_parameters(&request).unwrap();

        assert_eq!(params["durationSeconds"], 8);
        assert_eq!(params["aspectRatio"], "9:16");
        assert_eq!(params["resolution"], "1080p");
    }

    #[test]
    fn extracts_video_uri_from_operation_response() {
        let response = json!({
            "generateVideoResponse": {
                "generatedSamples": [
                    { "video": { "uri": "https://example.com/video.mp4" } }
                ]
            }
        });

        assert_eq!(
            video_uri_from_response(&response).as_deref(),
            Some("https://example.com/video.mp4")
        );
    }
}
