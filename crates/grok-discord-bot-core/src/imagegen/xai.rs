//! xAI Grok Imagine implementation of [`ImageProvider`].
//!
//! Two endpoints, picked automatically by [`XaiImageProvider::generate`]
//! based on whether the request carries reference images:
//!   - `POST /v1/images/generations` — pure text-to-image.
//!   - `POST /v1/images/edits` — text + 1-3 reference images. A single
//!     reference goes in `image: {url, type:"image_url"}`; two or three
//!     go in an `images` array of the same objects (the two fields are
//!     mutually exclusive, and multi-image prompts reference them as
//!     `<IMAGE_0>`, `<IMAGE_1>`, …).
//!
//! References can be supplied as:
//!   - `https://…` URLs — passed through as the `url`; xAI fetches them.
//!   - `file://images/<name>` URIs — resolved to local disk, the bytes
//!     are read here, base64-encoded as a `data:` URI, and inlined as the
//!     `url`. This is what the bot hands the model for conversation
//!     images, so editing works without our own server being reachable.
//!   - `data:image/…;base64,…` — passed through unchanged.
//!
//! The response always uses `response_format = b64_json` so the bytes
//! come back inline (one fewer round-trip vs `url`, and immune to xAI's
//! signed-URL TTL). Decoded bytes are returned to the caller, which is
//! expected to persist them via `core::storage::save_image_bytes`.

use base64::Engine;
use base64::engine::general_purpose::STANDARD as B64;
use serde::Deserialize;
use serde_json::{Value, json};

use super::{GeneratedImage, ImageGenError, ImageGenRequest, ImageProvider};

const DEFAULT_BASE_URL: &str = "https://api.x.ai/v1";
const STANDARD_MODEL: &str = "grok-imagine-image";
const QUALITY_MODEL: &str = "grok-imagine-image-quality";

/// xAI Grok Imagine client. Holds an API key and a `reqwest` handle —
/// safe to clone (reqwest's client is internally `Arc`-wrapped).
#[derive(Debug, Clone)]
pub struct XaiImageProvider {
    http: reqwest::Client,
    api_key: String,
    base_url: String,
}

impl XaiImageProvider {
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

impl ImageProvider for XaiImageProvider {
    fn name(&self) -> &str {
        "xai"
    }

    async fn generate(&self, request: ImageGenRequest) -> Result<GeneratedImage, ImageGenError> {
        // Resolve any file:// references to inline data URIs before
        // building the request body.
        let mut resolved_refs: Vec<String> = Vec::with_capacity(request.references.len());
        for r in &request.references {
            resolved_refs.push(resolve_reference(r, &request.images_dir).await?);
        }

        let model = resolve_model(request.model.as_deref());
        let ref_count = resolved_refs.len();
        let is_edit = ref_count > 0;

        let body = build_request_body(
            model,
            &request.prompt,
            request.aspect_ratio.as_deref(),
            resolved_refs,
        );

        let endpoint = if is_edit {
            format!("{}/images/edits", self.base_url)
        } else {
            format!("{}/images/generations", self.base_url)
        };

        tracing::info!(
            endpoint = %endpoint,
            model = %model,
            aspect_ratio = ?request.aspect_ratio,
            references = ref_count,
            "imagegen[xai]: requesting image"
        );

        let resp = self
            .http
            .post(&endpoint)
            .bearer_auth(&self.api_key)
            .json(&body)
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

        let mime_type = first
            .mime_type
            .or(first.content_type)
            .unwrap_or_else(|| "image/jpeg".to_string());

        Ok(GeneratedImage {
            bytes,
            mime_type,
            model: parsed.model.unwrap_or_else(|| model.to_string()),
            revised_prompt: first.revised_prompt,
        })
    }
}

/// Build the JSON request body for `/images/generations` or
/// `/images/edits`. With no references it's a plain generation; with
/// references it becomes an edit, encoding them per xAI's schema:
/// exactly one reference → `image: {url, type:"image_url"}`; two or
/// three → an `images` array of the same objects (the two fields are
/// mutually exclusive). `resolved_refs` already hold the final `url`
/// strings (public https or base64 `data:` URIs).
fn build_request_body(
    model: &str,
    prompt: &str,
    aspect_ratio: Option<&str>,
    resolved_refs: Vec<String>,
) -> Value {
    let mut body = json!({
        "model": model,
        "prompt": prompt,
        "response_format": "b64_json",
        "n": 1,
    });
    if let Some(ar) = aspect_ratio {
        body["aspect_ratio"] = json!(ar);
    }
    let ref_count = resolved_refs.len();
    if ref_count > 0 {
        let mut refs = resolved_refs
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

/// Translate the agent-supplied model string into an xAI model id.
///
/// Accepts the short-hand quality tiers (`"standard"`, `"quality"`),
/// the full xAI model ids, or `None` (defaults to `standard`). Anything
/// else is forwarded verbatim — xAI returns a clear API error for
/// unknown model ids, which is more informative than us rejecting it
/// here.
fn resolve_model(s: Option<&str>) -> &str {
    let Some(raw) = s else { return STANDARD_MODEL };
    match raw.to_ascii_lowercase().as_str() {
        "standard" => STANDARD_MODEL,
        "quality" => QUALITY_MODEL,
        _ => raw,
    }
}

/// Turn a user-supplied reference into something xAI accepts.
async fn resolve_reference(
    reference: &str,
    images_dir: &std::path::Path,
) -> Result<String, ImageGenError> {
    if reference.starts_with("http://") || reference.starts_with("https://") {
        return Ok(reference.to_string());
    }
    if reference.starts_with("data:") {
        return Ok(reference.to_string());
    }
    if let Some(local_path) = crate::storage::file_uri_to_local_path(reference, images_dir) {
        let bytes = tokio::fs::read(&local_path)
            .await
            .map_err(|e| ImageGenError::Reference(format!("read {}: {e}", local_path.display())))?;
        let mime = mime_from_ext(local_path.extension().and_then(|s| s.to_str()));
        let encoded = B64.encode(&bytes);
        return Ok(format!("data:{mime};base64,{encoded}"));
    }
    Err(ImageGenError::Reference(format!(
        "unsupported reference scheme: {reference}"
    )))
}

fn mime_from_ext(ext: Option<&str>) -> &'static str {
    match ext.unwrap_or("").to_ascii_lowercase().as_str() {
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "gif" => "image/gif",
        "webp" => "image/webp",
        "heic" | "heif" => "image/heic",
        _ => "application/octet-stream",
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
    #[serde(default, rename = "mime_type")]
    mime_type: Option<String>,
    #[serde(default)]
    content_type: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_model_aliases() {
        assert_eq!(resolve_model(None), STANDARD_MODEL);
        assert_eq!(resolve_model(Some("standard")), STANDARD_MODEL);
        assert_eq!(resolve_model(Some("STANDARD")), STANDARD_MODEL);
        assert_eq!(resolve_model(Some("quality")), QUALITY_MODEL);
        assert_eq!(resolve_model(Some("grok-imagine-image")), STANDARD_MODEL);
        // Unknown values pass through so the caller sees xAI's own
        // "model not found" error rather than a silent substitution.
        assert_eq!(resolve_model(Some("future-model-x")), "future-model-x");
    }

    #[test]
    fn generation_body_has_no_image_fields() {
        let body = build_request_body("grok-imagine-image", "a cat", Some("16:9"), vec![]);
        assert_eq!(body["prompt"], "a cat");
        assert_eq!(body["aspect_ratio"], "16:9");
        assert!(body.get("image").is_none());
        assert!(body.get("images").is_none());
    }

    #[test]
    fn single_reference_uses_image_object() {
        let body = build_request_body(
            "grok-imagine-image-quality",
            "whiten the teeth, keep everything else identical",
            None,
            vec!["data:image/jpeg;base64,AAAA".to_string()],
        );
        // Singular `image` object, not an `images` array, not a bare string.
        assert_eq!(
            body["image"],
            json!({ "url": "data:image/jpeg;base64,AAAA", "type": "image_url" })
        );
        assert!(body.get("images").is_none());
    }

    #[test]
    fn multiple_references_use_images_array() {
        let body = build_request_body(
            "grok-imagine-image-quality",
            "combine <IMAGE_0> and <IMAGE_1>",
            None,
            vec![
                "https://x/a.png".to_string(),
                "https://x/b.png".to_string(),
            ],
        );
        assert!(body.get("image").is_none());
        assert_eq!(
            body["images"],
            json!([
                { "url": "https://x/a.png", "type": "image_url" },
                { "url": "https://x/b.png", "type": "image_url" },
            ])
        );
    }

    #[test]
    fn http_url_passes_through() {
        let tmp = std::env::temp_dir();
        let result = tokio_test_block_on(async {
            resolve_reference("https://cdn.discordapp.com/x.png", &tmp).await
        });
        assert_eq!(result.unwrap(), "https://cdn.discordapp.com/x.png");
    }

    #[test]
    fn data_uri_passes_through() {
        let tmp = std::env::temp_dir();
        let result = tokio_test_block_on(async {
            resolve_reference("data:image/png;base64,aGk=", &tmp).await
        });
        assert_eq!(result.unwrap(), "data:image/png;base64,aGk=");
    }

    #[test]
    fn unknown_scheme_errors() {
        let tmp = std::env::temp_dir();
        let result =
            tokio_test_block_on(async { resolve_reference("s3://bucket/key", &tmp).await });
        assert!(matches!(result, Err(ImageGenError::Reference(_))));
    }

    fn tokio_test_block_on<F: std::future::Future>(f: F) -> F::Output {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(f)
    }
}
