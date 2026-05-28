//! xAI Grok Imagine Video implementation of [`VideoProvider`].
//!
//! Video generation at xAI is asynchronous: POST to
//! `/v1/videos/generations` returns a `request_id`, then `GET
//! /v1/videos/{request_id}` polls until `status` is `done`, `failed`,
//! or `expired`. When done, the response carries a `video.url` to
//! download.

use serde::Deserialize;
use serde_json::json;

use super::{JobStatus, VideoGenError, VideoGenRequest, VideoMeta, VideoProvider};

const DEFAULT_BASE_URL: &str = "https://api.x.ai/v1";
const DEFAULT_MODEL: &str = "grok-imagine-video";

/// xAI Grok Imagine Video client.
#[derive(Debug, Clone)]
pub struct XaiVideoProvider {
    http: reqwest::Client,
    api_key: String,
    base_url: String,
}

impl XaiVideoProvider {
    /// Construct from an xAI API key.
    pub fn new(api_key: String) -> Self {
        Self {
            http: reqwest::Client::new(),
            api_key,
            base_url: DEFAULT_BASE_URL.to_string(),
        }
    }

    /// Override the base URL. Used by tests.
    pub fn with_base_url(mut self, base_url: String) -> Self {
        self.base_url = base_url;
        self
    }
}

impl VideoProvider for XaiVideoProvider {
    fn name(&self) -> &str {
        "xai"
    }

    async fn submit(&self, request: &VideoGenRequest) -> Result<String, VideoGenError> {
        let resolution = request.resolution.as_deref().unwrap_or("480p");
        let model = request.model.as_deref().unwrap_or(DEFAULT_MODEL);
        let mut body = json!({
            "model": model,
            "prompt": request.prompt,
            "resolution": resolution,
        });
        if let Some(d) = request.duration_seconds {
            body["duration"] = json!(d);
        }
        if let Some(ar) = &request.aspect_ratio {
            body["aspect_ratio"] = json!(ar);
        }
        if let Some(url) = &request.image_url {
            body["image"] = json!({ "url": url });
        }

        let resp = self
            .http
            .post(format!("{}/videos/generations", self.base_url))
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await
            .map_err(|e| VideoGenError::Transport(e.to_string()))?;
        let status = resp.status();
        if !status.is_success() {
            let body = truncate_body(resp.text().await.unwrap_or_default(), 400);
            return Err(VideoGenError::Api {
                status: status.as_u16(),
                body,
            });
        }
        let parsed: SubmitResponse = resp
            .json()
            .await
            .map_err(|e| VideoGenError::Decode(e.to_string()))?;
        Ok(parsed.request_id)
    }

    async fn check_once(&self, request_id: &str) -> Result<JobStatus, VideoGenError> {
        let endpoint = format!("{}/videos/{}", self.base_url, request_id);
        let resp = self
            .http
            .get(&endpoint)
            .bearer_auth(&self.api_key)
            .send()
            .await
            .map_err(|e| VideoGenError::Transport(e.to_string()))?;
        let status = resp.status();
        if !status.is_success() {
            let body = truncate_body(resp.text().await.unwrap_or_default(), 400);
            return Err(VideoGenError::Api {
                status: status.as_u16(),
                body,
            });
        }
        let parsed: PollResponse = resp
            .json()
            .await
            .map_err(|e| VideoGenError::Decode(e.to_string()))?;
        match parsed.status.as_str() {
            "done" => {
                let video = parsed.video.ok_or_else(|| {
                    VideoGenError::Decode("status=done but no video object".into())
                })?;
                Ok(JobStatus::Done(VideoMeta {
                    url: video.url,
                    duration: video.duration,
                }))
            }
            "failed" => Ok(JobStatus::Failed(
                parsed
                    .error
                    .map(|e| e.message)
                    .unwrap_or_else(|| "(no message)".into()),
            )),
            "expired" => Ok(JobStatus::Expired),
            _ => Ok(JobStatus::Pending),
        }
    }

    async fn download_bytes(&self, url: &str) -> Result<Vec<u8>, VideoGenError> {
        let resp = self
            .http
            .get(url)
            .send()
            .await
            .map_err(|e| VideoGenError::Transport(e.to_string()))?;
        let status = resp.status();
        if !status.is_success() {
            let body = truncate_body(resp.text().await.unwrap_or_default(), 400);
            return Err(VideoGenError::Api {
                status: status.as_u16(),
                body,
            });
        }
        resp.bytes()
            .await
            .map(|b| b.to_vec())
            .map_err(|e| VideoGenError::Transport(e.to_string()))
    }
}

/// Guess a MIME type from the video URL's extension. Public helper:
/// the bot uses it after [`VideoProvider::download_bytes`] to label
/// the saved file.
pub fn guess_mime(url: &str) -> String {
    let no_query = url.split('?').next().unwrap_or(url);
    let ext = no_query.rsplit('.').next().unwrap_or("").to_ascii_lowercase();
    match ext.as_str() {
        "mp4" => "video/mp4".to_string(),
        "webm" => "video/webm".to_string(),
        "mov" => "video/quicktime".to_string(),
        _ => "video/mp4".to_string(),
    }
}

fn truncate_body(mut s: String, max: usize) -> String {
    if s.len() > max {
        s.truncate(max);
    }
    s
}

#[derive(Deserialize)]
struct SubmitResponse {
    request_id: String,
}

#[derive(Deserialize)]
struct PollResponse {
    status: String,
    #[serde(default)]
    video: Option<RawVideoMeta>,
    #[serde(default)]
    error: Option<PollError>,
}

#[derive(Deserialize)]
struct RawVideoMeta {
    url: String,
    #[serde(default)]
    duration: Option<f32>,
}

#[derive(Deserialize, Debug, Clone)]
struct PollError {
    #[serde(default)]
    #[allow(dead_code)]
    code: Option<String>,
    #[serde(default)]
    message: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mime_from_url_extension() {
        assert_eq!(guess_mime("https://x/v.mp4"), "video/mp4");
        assert_eq!(guess_mime("https://x/v.WEBM?token=abc"), "video/webm");
        assert_eq!(guess_mime("https://x/v"), "video/mp4");
    }
}
