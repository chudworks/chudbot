//! xAI Grok Imagine video client.
//!
//! Video generation at xAI is asynchronous: you POST to
//! `/v1/videos/generations`, get back a `request_id`, then poll
//! `GET /v1/videos/{request_id}` until `status` becomes `done`,
//! `failed`, or `expired`. When done, the response carries a
//! `video.url` we download to bytes.
//!
//! This client handles the whole flow in one async call:
//!   1. POST the request, parse `request_id`.
//!   2. Sleep, then GET status. Repeat until done / timeout.
//!   3. Download bytes from the final `video.url`.
//!   4. Return `(bytes, mime, duration)`.
//!
//! Both text-to-video and image-to-video use the same endpoint; the
//! `image` field on the request switches modes.

use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use thiserror::Error;

const DEFAULT_BASE_URL: &str = "https://api.x.ai/v1";
const DEFAULT_MODEL: &str = "grok-imagine-video";
/// First poll happens after this much delay; subsequent polls reuse it
/// as the inter-poll interval.
const POLL_INTERVAL: Duration = Duration::from_secs(3);
/// Hard cap on how long we'll wait for one video. Max generation
/// length is 15s; xAI typically returns in well under 2 minutes.
const MAX_WAIT: Duration = Duration::from_secs(300);

/// Errors returned by [`VideoGenerator`].
#[derive(Debug, Error)]
pub enum VideoGenError {
    /// Network/transport failure.
    #[error("transport: {0}")]
    Transport(String),
    /// xAI returned a non-success status.
    #[error("api {status}: {body}")]
    Api {
        /// HTTP status code.
        status: u16,
        /// Truncated response body.
        body: String,
    },
    /// Response couldn't be decoded to the expected shape.
    #[error("decode: {0}")]
    Decode(String),
    /// Status came back `failed` or `expired` from xAI.
    #[error("upstream {status}: {message}")]
    Upstream {
        /// `failed` or `expired`.
        status: String,
        /// Error message body from xAI.
        message: String,
    },
    /// Hit the polling timeout without a terminal status.
    #[error("polling timed out after {0:?}")]
    Timeout(Duration),
}

/// Resolution tier — pricing per second depends on this.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum VideoResolution {
    /// `480p` — cheapest ($0.05/sec).
    #[default]
    P480,
    /// `720p` — higher fidelity ($0.07/sec).
    P720,
}

impl VideoResolution {
    fn as_str(&self) -> &'static str {
        match self {
            Self::P480 => "480p",
            Self::P720 => "720p",
        }
    }
}

/// Input to [`VideoGenerator::generate`].
#[derive(Debug, Clone)]
pub struct VideoGenRequest {
    /// Text prompt describing the desired video.
    pub prompt: String,
    /// Optional image URL/URI to animate from. `https://` URLs are
    /// passed through; future support for `file://` would need to
    /// upload the bytes first.
    pub image_url: Option<String>,
    /// Length in seconds (1-15). `None` lets xAI pick a default.
    pub duration_seconds: Option<u8>,
    /// Aspect ratio (e.g. `"16:9"`). `None` defaults to `16:9` server-side.
    pub aspect_ratio: Option<String>,
    /// 480p or 720p.
    pub resolution: VideoResolution,
}

/// Output of [`VideoGenerator::generate`].
#[derive(Debug, Clone)]
pub struct GeneratedVideo {
    /// Raw video bytes (typically MP4).
    pub bytes: Vec<u8>,
    /// MIME type, e.g. `video/mp4`.
    pub mime_type: String,
    /// Actual generated duration in seconds (xAI may round).
    pub duration_seconds: f32,
    /// Echoed back so caller can correlate with logs / pricing.
    pub request_id: String,
}

/// HTTP client for xAI's video endpoints.
#[derive(Debug, Clone)]
pub struct VideoGenerator {
    http: reqwest::Client,
    api_key: String,
    base_url: String,
}

impl VideoGenerator {
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

    /// Run the full submit → poll → download flow in one call. Suitable
    /// for direct callers; the agent-driven flow in `bot.rs` uses the
    /// lower-level `submit` / `check_once` / `download_bytes` methods
    /// instead so the model can post status messages between polls and
    /// so each step is durable in `video_jobs`.
    pub async fn generate(
        &self,
        request: VideoGenRequest,
    ) -> Result<GeneratedVideo, VideoGenError> {
        let request_id = self.submit(&request).await?;
        tracing::info!(
            request_id = %request_id,
            resolution = %request.resolution.as_str(),
            duration = ?request.duration_seconds,
            "videogen: submitted, polling for completion"
        );

        let video_meta = self.poll_until_done(&request_id).await?;
        let bytes = self.download_bytes(&video_meta.url).await?;
        let mime_type = guess_mime(&video_meta.url);

        Ok(GeneratedVideo {
            bytes,
            mime_type,
            duration_seconds: video_meta.duration.unwrap_or(0.0),
            request_id,
        })
    }

    /// Submit a generation request. Returns the xAI `request_id` to
    /// poll on. Fast (~hundreds of ms).
    pub async fn submit(&self, request: &VideoGenRequest) -> Result<String, VideoGenError> {
        let mut body = json!({
            "model": DEFAULT_MODEL,
            "prompt": request.prompt,
            "resolution": request.resolution.as_str(),
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

    /// Poll the job once. Returns the current status without blocking
    /// across multiple polls. Caller is responsible for spacing
    /// repeated calls — the agent loop uses this so the model can post
    /// status messages between polls.
    pub async fn check_once(&self, request_id: &str) -> Result<JobStatus, VideoGenError> {
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
                Ok(JobStatus::Done(video))
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

    /// Download the bytes at `url`. Public so the agent can call it
    /// once the polling step reports `JobStatus::Done`.
    pub async fn download_bytes(&self, url: &str) -> Result<Vec<u8>, VideoGenError> {
        self.download(url).await
    }

    async fn poll_until_done(&self, request_id: &str) -> Result<VideoMeta, VideoGenError> {
        let endpoint = format!("{}/videos/{}", self.base_url, request_id);
        let start = std::time::Instant::now();
        loop {
            tokio::time::sleep(POLL_INTERVAL).await;
            if start.elapsed() > MAX_WAIT {
                return Err(VideoGenError::Timeout(MAX_WAIT));
            }

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
                    tracing::info!(
                        request_id,
                        elapsed_secs = start.elapsed().as_secs(),
                        url = %video.url,
                        "videogen: completed"
                    );
                    return Ok(video);
                }
                "failed" => {
                    let msg = parsed
                        .error
                        .as_ref()
                        .map(|e| e.message.clone())
                        .unwrap_or_else(|| "(no message)".into());
                    return Err(VideoGenError::Upstream {
                        status: "failed".into(),
                        message: msg,
                    });
                }
                "expired" => {
                    return Err(VideoGenError::Upstream {
                        status: "expired".into(),
                        message: "video generation request expired".into(),
                    });
                }
                _other => {
                    tracing::debug!(
                        request_id,
                        elapsed_secs = start.elapsed().as_secs(),
                        status = %parsed.status,
                        "videogen: still pending"
                    );
                    continue;
                }
            }
        }
    }

    async fn download(&self, url: &str) -> Result<Vec<u8>, VideoGenError> {
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

fn truncate_body(mut s: String, max: usize) -> String {
    if s.len() > max {
        s.truncate(max);
    }
    s
}

fn guess_mime(url: &str) -> String {
    let no_query = url.split('?').next().unwrap_or(url);
    let ext = no_query.rsplit('.').next().unwrap_or("").to_ascii_lowercase();
    match ext.as_str() {
        "mp4" => "video/mp4".to_string(),
        "webm" => "video/webm".to_string(),
        "mov" => "video/quicktime".to_string(),
        _ => "video/mp4".to_string(),
    }
}

/// Polling outcome for [`VideoGenerator::check_once`].
#[derive(Debug, Clone)]
pub enum JobStatus {
    /// Not done yet; caller should sleep and try again.
    Pending,
    /// Generation complete; the contained `VideoMeta` carries the URL
    /// + duration.
    Done(VideoMeta),
    /// xAI's classifiers refused or the generation otherwise failed.
    /// The string is the upstream error message.
    Failed(String),
    /// Job expired before completion.
    Expired,
}

#[derive(Deserialize)]
struct SubmitResponse {
    request_id: String,
}

#[derive(Deserialize)]
struct PollResponse {
    status: String,
    #[serde(default)]
    video: Option<VideoMeta>,
    #[serde(default)]
    error: Option<PollError>,
}

/// Metadata returned by xAI when polling reports `status=done`.
#[derive(Deserialize, Debug, Clone)]
pub struct VideoMeta {
    /// Direct download URL.
    pub url: String,
    /// Duration in seconds (may be slightly off the requested duration).
    #[serde(default)]
    pub duration: Option<f32>,
}

#[derive(Deserialize, Debug, Clone)]
struct PollError {
    #[serde(default)]
    #[allow(dead_code)]
    code: Option<String>,
    #[serde(default)]
    message: String,
}

#[allow(dead_code)]
fn _force_value(_v: Value) {}

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
