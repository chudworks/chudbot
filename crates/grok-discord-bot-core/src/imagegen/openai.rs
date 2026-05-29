//! OpenAI image generation (`gpt-image-1` family) implementation of
//! [`ImageProvider`].
//!
//! Two endpoints, picked by whether the request carries reference images:
//!   - `POST /v1/images/generations` — pure text-to-image, JSON body.
//!   - `POST /v1/images/edits` — text + 1-N reference images.
//!     **`multipart/form-data`**: OpenAI's edits endpoint takes the
//!     reference images as uploaded file parts, NOT as URLs (this differs
//!     from xAI, which accepts URLs/data-URIs in JSON). So every
//!     reference is resolved to raw bytes here — `file://` from disk,
//!     `https://` fetched server-side, `data:` base64-decoded — and sent
//!     as repeated `image[]` parts.
//!
//! The free-form [`ImageGenRequest::model`] field is interpreted against
//! OpenAI's catalog: a `gpt-image-*` / `dall-e-*` value is used verbatim
//! as the model id; the quality words `low` / `medium` / `high` / `auto`
//! (plus xAI-style `standard` / `quality` aliases) select the `quality`
//! parameter on the default `gpt-image-1` model. [`ImageGenRequest::
//! aspect_ratio`] maps to OpenAI's discrete `size` values.
//!
//! `gpt-image-1` always returns images as `b64_json`, so the bytes come
//! back inline and we decode them directly.

use base64::Engine;
use base64::engine::general_purpose::STANDARD as B64;
use serde::Deserialize;
use serde_json::json;

use super::{GeneratedImage, ImageGenError, ImageGenRequest, ImageProvider};

const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";
const DEFAULT_MODEL: &str = "gpt-image-1";

/// OpenAI image client. Holds an API key and a `reqwest` handle — safe
/// to clone (reqwest's client is internally `Arc`-wrapped).
#[derive(Debug, Clone)]
pub struct OpenAiImageProvider {
    http: reqwest::Client,
    api_key: String,
    base_url: String,
}

impl OpenAiImageProvider {
    /// Construct from an OpenAI API key.
    pub fn new(api_key: String) -> Self {
        Self {
            http: reqwest::Client::new(),
            api_key,
            base_url: DEFAULT_BASE_URL.to_string(),
        }
    }

    /// Override the base URL. Used by tests and Azure-style gateways.
    pub fn with_base_url(mut self, base_url: String) -> Self {
        self.base_url = base_url;
        self
    }
}

impl ImageProvider for OpenAiImageProvider {
    fn name(&self) -> &str {
        "openai"
    }

    async fn generate(&self, request: ImageGenRequest) -> Result<GeneratedImage, ImageGenError> {
        let (model, quality) = resolve_model_and_quality(request.model.as_deref());
        let size = map_aspect_to_size(request.aspect_ratio.as_deref());

        if request.references.is_empty() {
            self.generate_from_text(&request, model, quality, size)
                .await
        } else {
            self.generate_from_edit(&request, model, quality, size)
                .await
        }
    }
}

impl OpenAiImageProvider {
    /// `POST /images/generations` — JSON body, no reference images.
    async fn generate_from_text(
        &self,
        request: &ImageGenRequest,
        model: &str,
        quality: Option<&str>,
        size: Option<&str>,
    ) -> Result<GeneratedImage, ImageGenError> {
        let mut body = json!({
            "model": model,
            "prompt": request.prompt,
            "n": 1,
        });
        if let Some(q) = quality {
            body["quality"] = json!(q);
        }
        if let Some(s) = size {
            body["size"] = json!(s);
        }

        let endpoint = format!("{}/images/generations", self.base_url);
        tracing::info!(
            endpoint = %endpoint,
            model = %model,
            quality = ?quality,
            size = ?size,
            "imagegen[openai]: requesting image (generations)"
        );

        let resp = crate::retry::with_retry(
            crate::retry::RetryPolicy::default(),
            "imagegen[openai]",
            || {
                let req = self
                    .http
                    .post(&endpoint)
                    .bearer_auth(&self.api_key)
                    .json(&body);
                async move { send_and_check(req).await }
            },
        )
        .await?;

        decode_image_response(resp, model).await
    }

    /// `POST /images/edits` — multipart body with reference image parts.
    async fn generate_from_edit(
        &self,
        request: &ImageGenRequest,
        model: &str,
        quality: Option<&str>,
        size: Option<&str>,
    ) -> Result<GeneratedImage, ImageGenError> {
        // Resolve every reference to raw bytes up front (so the retry
        // closure can clone them into a fresh multipart Form per attempt;
        // reqwest's Form isn't Clone).
        let mut refs: Vec<RefImage> = Vec::with_capacity(request.references.len());
        for r in &request.references {
            refs.push(self.resolve_reference_bytes(r, &request.images_dir).await?);
        }

        let endpoint = format!("{}/images/edits", self.base_url);
        tracing::info!(
            endpoint = %endpoint,
            model = %model,
            quality = ?quality,
            size = ?size,
            references = refs.len(),
            "imagegen[openai]: requesting image (edits)"
        );

        let resp = crate::retry::with_retry(
            crate::retry::RetryPolicy::default(),
            "imagegen[openai]",
            || {
                let mut form = reqwest::multipart::Form::new()
                    .text("model", model.to_string())
                    .text("prompt", request.prompt.clone())
                    .text("n", "1");
                if let Some(q) = quality {
                    form = form.text("quality", q.to_string());
                }
                if let Some(s) = size {
                    form = form.text("size", s.to_string());
                }
                for (i, r) in refs.iter().enumerate() {
                    let part = reqwest::multipart::Part::bytes(r.bytes.clone())
                        .file_name(format!("image_{i}.{}", r.ext))
                        .mime_str(r.mime)
                        .unwrap_or_else(|_| {
                            reqwest::multipart::Part::bytes(r.bytes.clone())
                                .file_name(format!("image_{i}"))
                        });
                    // gpt-image-1 accepts multiple references as repeated
                    // `image[]` parts.
                    form = form.part("image[]", part);
                }
                let req = self
                    .http
                    .post(&endpoint)
                    .bearer_auth(&self.api_key)
                    .multipart(form);
                async move { send_and_check(req).await }
            },
        )
        .await?;

        decode_image_response(resp, model).await
    }

    /// Resolve a reference URI to raw bytes (OpenAI's edits endpoint
    /// uploads files, not URLs). `https://` is fetched server-side;
    /// `file://images/<name>` is read from disk; `data:` is decoded.
    async fn resolve_reference_bytes(
        &self,
        reference: &str,
        images_dir: &std::path::Path,
    ) -> Result<RefImage, ImageGenError> {
        if reference.starts_with("http://") || reference.starts_with("https://") {
            let resp = self
                .http
                .get(reference)
                .send()
                .await
                .map_err(|e| ImageGenError::Reference(format!("fetch {reference}: {e}")))?;
            if !resp.status().is_success() {
                return Err(ImageGenError::Reference(format!(
                    "fetch {reference}: status {}",
                    resp.status().as_u16()
                )));
            }
            let mime = resp
                .headers()
                .get(reqwest::header::CONTENT_TYPE)
                .and_then(|v| v.to_str().ok())
                .map(str::to_string);
            let bytes = resp
                .bytes()
                .await
                .map_err(|e| ImageGenError::Reference(format!("read body {reference}: {e}")))?
                .to_vec();
            let mime = static_mime(mime.as_deref().unwrap_or("image/png"));
            return Ok(RefImage {
                ext: ext_for_mime(mime),
                mime,
                bytes,
            });
        }
        if let Some(rest) = reference.strip_prefix("data:") {
            // data:<mime>;base64,<payload>
            let (meta, payload) = rest
                .split_once(',')
                .ok_or_else(|| ImageGenError::Reference("malformed data URI".into()))?;
            let bytes = B64
                .decode(payload.as_bytes())
                .map_err(|e| ImageGenError::Reference(format!("data URI base64: {e}")))?;
            let mime = static_mime(meta.split(';').next().unwrap_or("image/png"));
            return Ok(RefImage {
                ext: ext_for_mime(mime),
                mime,
                bytes,
            });
        }
        if let Some(local_path) = crate::storage::file_uri_to_local_path(reference, images_dir) {
            let bytes = tokio::fs::read(&local_path)
                .await
                .map_err(|e| ImageGenError::Reference(format!("read {}: {e}", local_path.display())))?;
            let mime = mime_from_ext(local_path.extension().and_then(|s| s.to_str()));
            return Ok(RefImage {
                ext: ext_for_mime(mime),
                mime,
                bytes,
            });
        }
        Err(ImageGenError::Reference(format!(
            "unsupported reference scheme: {reference}"
        )))
    }
}

/// A reference image resolved to bytes for multipart upload.
struct RefImage {
    bytes: Vec<u8>,
    mime: &'static str,
    ext: &'static str,
}

/// Send a built request and map a non-success status into a (retryable
/// where appropriate) [`ImageGenError::Api`].
async fn send_and_check(req: reqwest::RequestBuilder) -> Result<reqwest::Response, ImageGenError> {
    let resp = req
        .send()
        .await
        .map_err(|e| ImageGenError::Transport(e.to_string()))?;
    let status = resp.status();
    if !status.is_success() {
        let mut text = resp.text().await.unwrap_or_default();
        if text.len() > 400 {
            text.truncate(400);
        }
        return Err(ImageGenError::Api {
            status: status.as_u16(),
            body: text,
        });
    }
    Ok(resp)
}

/// Decode the `b64_json` of the first item in an images response.
async fn decode_image_response(
    resp: reqwest::Response,
    model: &str,
) -> Result<GeneratedImage, ImageGenError> {
    let parsed: ImagesResponse = resp
        .json()
        .await
        .map_err(|e| ImageGenError::Decode(e.to_string()))?;

    let first = parsed
        .data
        .into_iter()
        .next()
        .ok_or_else(|| ImageGenError::Decode("response had no images".into()))?;

    let b64 = first
        .b64_json
        .ok_or_else(|| ImageGenError::Decode("response item lacked b64_json".into()))?;
    let bytes = B64
        .decode(b64.as_bytes())
        .map_err(|e| ImageGenError::Decode(format!("base64: {e}")))?;

    // gpt-image-1 returns PNG unless an output_format was requested; the
    // response doesn't echo a per-item mime, so default to PNG.
    let mime_type = first
        .output_format
        .map(|f| static_mime(&format!("image/{f}")).to_string())
        .unwrap_or_else(|| "image/png".to_string());

    Ok(GeneratedImage {
        bytes,
        mime_type,
        model: parsed.model.unwrap_or_else(|| model.to_string()),
        revised_prompt: first.revised_prompt,
    })
}

/// Interpret the agent-supplied model/quality string. Returns the model
/// id and an optional `quality` value.
///
/// - `None` → default model, no explicit quality (OpenAI picks `auto`).
/// - a `gpt-image-*` / `dall-e-*` id → used verbatim, no quality.
/// - `low` / `medium` / `high` / `auto` → default model + that quality.
/// - xAI-style `standard` / `quality` aliases → `medium` / `high`.
/// - anything else → forwarded verbatim as the model id (so OpenAI's own
///   "model not found" surfaces rather than a silent substitution).
fn resolve_model_and_quality(s: Option<&str>) -> (&str, Option<&'static str>) {
    let Some(raw) = s else {
        return (DEFAULT_MODEL, None);
    };
    match raw.to_ascii_lowercase().as_str() {
        "low" => (DEFAULT_MODEL, Some("low")),
        "medium" | "standard" => (DEFAULT_MODEL, Some("medium")),
        "high" | "quality" => (DEFAULT_MODEL, Some("high")),
        "auto" => (DEFAULT_MODEL, Some("auto")),
        _ => (raw, None),
    }
}

/// Map a free-form aspect ratio to one of OpenAI's discrete `size`
/// values. `None` (or an unrecognized ratio) lets the backend default by
/// returning `None` / `"auto"`.
fn map_aspect_to_size(aspect: Option<&str>) -> Option<&'static str> {
    match aspect?.trim() {
        "1:1" => Some("1024x1024"),
        "16:9" | "3:2" | "landscape" => Some("1536x1024"),
        "9:16" | "2:3" | "portrait" => Some("1024x1536"),
        "auto" => Some("auto"),
        _ => Some("auto"),
    }
}

fn mime_from_ext(ext: Option<&str>) -> &'static str {
    match ext.unwrap_or("").to_ascii_lowercase().as_str() {
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "webp" => "image/webp",
        _ => "image/png",
    }
}

/// Normalize a possibly-parametrized content type to one of the formats
/// OpenAI accepts for uploads, defaulting to PNG.
fn static_mime(mime: &str) -> &'static str {
    match mime.split(';').next().unwrap_or("").trim() {
        "image/jpeg" | "image/jpg" => "image/jpeg",
        "image/webp" => "image/webp",
        _ => "image/png",
    }
}

fn ext_for_mime(mime: &str) -> &'static str {
    match mime {
        "image/jpeg" => "jpg",
        "image/webp" => "webp",
        _ => "png",
    }
}

#[derive(Deserialize)]
struct ImagesResponse {
    #[serde(default)]
    data: Vec<ImagesResponseItem>,
    #[serde(default)]
    model: Option<String>,
}

#[derive(Deserialize)]
struct ImagesResponseItem {
    #[serde(default)]
    b64_json: Option<String>,
    #[serde(default)]
    revised_prompt: Option<String>,
    #[serde(default)]
    output_format: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_model_and_quality() {
        assert_eq!(resolve_model_and_quality(None), (DEFAULT_MODEL, None));
        assert_eq!(
            resolve_model_and_quality(Some("high")),
            (DEFAULT_MODEL, Some("high"))
        );
        assert_eq!(
            resolve_model_and_quality(Some("quality")),
            (DEFAULT_MODEL, Some("high"))
        );
        assert_eq!(
            resolve_model_and_quality(Some("standard")),
            (DEFAULT_MODEL, Some("medium"))
        );
        // Explicit model ids pass through with no quality override.
        assert_eq!(
            resolve_model_and_quality(Some("gpt-image-1.5")),
            ("gpt-image-1.5", None)
        );
        assert_eq!(
            resolve_model_and_quality(Some("dall-e-3")),
            ("dall-e-3", None)
        );
    }

    #[test]
    fn maps_aspect_to_size() {
        assert_eq!(map_aspect_to_size(None), None);
        assert_eq!(map_aspect_to_size(Some("1:1")), Some("1024x1024"));
        assert_eq!(map_aspect_to_size(Some("16:9")), Some("1536x1024"));
        assert_eq!(map_aspect_to_size(Some("9:16")), Some("1024x1536"));
        // Unknown ratios fall back to auto rather than erroring.
        assert_eq!(map_aspect_to_size(Some("7:5")), Some("auto"));
    }

    #[test]
    fn static_mime_normalizes() {
        assert_eq!(static_mime("image/jpeg; charset=binary"), "image/jpeg");
        assert_eq!(static_mime("image/webp"), "image/webp");
        assert_eq!(static_mime("application/octet-stream"), "image/png");
    }

    #[test]
    fn ext_matches_mime() {
        assert_eq!(ext_for_mime("image/jpeg"), "jpg");
        assert_eq!(ext_for_mime("image/webp"), "webp");
        assert_eq!(ext_for_mime("image/png"), "png");
    }

    #[test]
    fn decodes_b64_response_shape() {
        let body = json!({
            "model": "gpt-image-1",
            "data": [{ "b64_json": "QUJD", "revised_prompt": "a cat, refined" }],
        });
        let parsed: ImagesResponse = serde_json::from_value(body).unwrap();
        assert_eq!(parsed.model.as_deref(), Some("gpt-image-1"));
        let first = parsed.data.into_iter().next().unwrap();
        assert_eq!(first.b64_json.as_deref(), Some("QUJD"));
        assert_eq!(first.revised_prompt.as_deref(), Some("a cat, refined"));
    }
}
