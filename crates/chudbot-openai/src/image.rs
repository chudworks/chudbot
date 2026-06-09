//! OpenAI image generation implementation.

use base64::Engine;
use base64::engine::general_purpose::STANDARD as B64;
use chudbot_api::{
    GeneratedImage, ImageGenerator, ImageRequest, MediaRef, ModelId, ProviderName, UsageRecord,
    UsageSubject,
    retry::{RetryPolicy, with_retry},
};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::pricing::{ImagePricingUsage, OpenAiPricing};
use crate::{OpenAiClient, OpenAiError, decode_response, log_json_request};

const DEFAULT_IMAGE_MODEL: &str = "gpt-image-1";

impl ImageGenerator for OpenAiClient {
    type Error = OpenAiError;

    fn backend_name(&self) -> &ProviderName {
        self.provider_name()
    }

    #[tracing::instrument(name = "openai.generate_image", skip_all)]
    async fn generate_image(&self, request: ImageRequest) -> Result<GeneratedImage, Self::Error> {
        let (model, quality) =
            resolve_model_and_quality(request.model.as_ref().map(ModelId::as_str));
        let size = map_aspect_to_size(request.aspect_ratio.as_deref());
        tracing::debug!(
            prompt_chars = request.prompt.chars().count(),
            references = request.references.len(),
            model = %model,
            quality = ?quality,
            size = ?size,
            "building OpenAI image request"
        );

        if request.references.is_empty() {
            self.generate_from_text(&request.prompt, model, quality, size)
                .await
        } else {
            self.generate_from_edit(&request, model, quality, size)
                .await
        }
    }
}

impl OpenAiClient {
    async fn generate_from_text(
        &self,
        prompt: &str,
        model: &str,
        quality: Option<&str>,
        size: Option<&str>,
    ) -> Result<GeneratedImage, OpenAiError> {
        tracing::debug!(
            model = %model,
            quality = ?quality,
            size = ?size,
            prompt_chars = prompt.chars().count(),
            "sending OpenAI text-to-image request"
        );
        let mut body = json!({
            "model": model,
            "prompt": prompt,
            "n": 1,
        });
        if let Some(quality) = quality {
            body["quality"] = json!(quality);
        }
        if let Some(size) = size {
            body["size"] = json!(size);
        }

        let parsed: ImagesResponse = self
            .post_json("/images/generations", &body, "imagegen[openai].generate")
            .await?;
        image_from_response(parsed, self.provider_name(), model, self.pricing())
    }

    async fn generate_from_edit(
        &self,
        request: &ImageRequest,
        model: &str,
        quality: Option<&str>,
        size: Option<&str>,
    ) -> Result<GeneratedImage, OpenAiError> {
        tracing::debug!(
            model = %model,
            quality = ?quality,
            size = ?size,
            references = request.references.len(),
            prompt_chars = request.prompt.chars().count(),
            "sending OpenAI image edit request"
        );
        let mut references = Vec::with_capacity(request.references.len());
        for reference in &request.references {
            references.push(reference_image(reference.as_ref(), self).await?);
        }

        let endpoint = "/images/edits";
        log_json_request(
            self.provider_name(),
            endpoint,
            &image_edit_log_body(request, model, quality, size, &references),
        );
        let url = format!("{}{}", self.base_url(), endpoint);
        let parsed: ImagesResponse =
            with_retry(RetryPolicy::default(), "imagegen[openai].edit", || async {
                let mut form = reqwest::multipart::Form::new()
                    .text("model", model.to_string())
                    .text("prompt", request.prompt.clone())
                    .text("n", "1");
                if let Some(quality) = quality {
                    form = form.text("quality", quality.to_string());
                }
                if let Some(size) = size {
                    form = form.text("size", size.to_string());
                }
                for (i, reference) in references.iter().enumerate() {
                    let file_name = format!("image_{i}.{}", reference.ext);
                    let part = reqwest::multipart::Part::bytes(reference.bytes.clone())
                        .file_name(file_name)
                        .mime_str(reference.mime)
                        .map_err(|e| OpenAiError::Reference(format!("reference MIME: {e}")))?;
                    form = form.part("image[]", part);
                }
                let request = self
                    .http()
                    .post(&url)
                    .bearer_auth(self.api_key())
                    .multipart(form);
                let resp = send_image_request(request, "OpenAI image edit").await?;
                decode_response(resp, self.provider_name(), endpoint).await
            })
            .await?;
        image_from_response(parsed, self.provider_name(), model, self.pricing())
    }
}

pub(crate) async fn media_bytes_or_url(media: &dyn MediaRef) -> Result<String, OpenAiError> {
    match media.public_url().await {
        Ok(url) => {
            tracing::debug!(
                uri = %media.uri(),
                category = ?media.category(),
                "resolved media public URL for OpenAI"
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
                    "inlined media bytes for OpenAI"
                );
                Ok(data_uri(loaded.media.mime_type(), &loaded.bytes))
            }
            Err(load_error) => {
                tracing::warn!(
                    uri = %media.uri(),
                    category = ?media.category(),
                    public_error = %public_error,
                    load_error = %load_error,
                    "failed to resolve media for OpenAI"
                );
                Err(OpenAiError::Reference(format!(
                    "media `{}` has no public URL ({public_error}) and could not be loaded ({load_error})",
                    media.uri()
                )))
            }
        },
    }
}

async fn reference_image(
    media: &dyn MediaRef,
    client: &OpenAiClient,
) -> Result<RefImage, OpenAiError> {
    match media.load().await {
        Ok(loaded) => {
            let mime = static_mime(loaded.media.mime_type());
            tracing::debug!(
                uri = %media.uri(),
                category = ?media.category(),
                bytes = loaded.bytes.len(),
                mime_type = mime,
                "loaded OpenAI image reference from media store"
            );
            Ok(RefImage {
                bytes: loaded.bytes,
                mime,
                ext: ext_for_mime(mime),
            })
        }
        Err(load_error) => match media.public_url().await {
            Ok(url) => {
                tracing::debug!(
                    uri = %media.uri(),
                    category = ?media.category(),
                    "loading OpenAI image reference from public URL"
                );
                reference_from_url_or_data(url.as_str(), client).await
            }
            Err(public_error) => {
                tracing::warn!(
                    uri = %media.uri(),
                    category = ?media.category(),
                    load_error = %load_error,
                    public_error = %public_error,
                    "failed to resolve OpenAI image reference"
                );
                Err(OpenAiError::Reference(format!(
                    "media `{}` could not be loaded ({load_error}) and has no public URL ({public_error})",
                    media.uri()
                )))
            }
        },
    }
}

async fn reference_from_url_or_data(
    reference: &str,
    client: &OpenAiClient,
) -> Result<RefImage, OpenAiError> {
    if reference.starts_with("data:") {
        let (mime, bytes) = decode_data_uri(reference)?;
        tracing::debug!(
            mime_type = mime,
            bytes = bytes.len(),
            "decoded OpenAI data URI reference"
        );
        return Ok(RefImage {
            bytes,
            mime,
            ext: ext_for_mime(mime),
        });
    }

    if reference.starts_with("http://") || reference.starts_with("https://") {
        let resp = client.http().get(reference).send().await.map_err(|e| {
            tracing::warn!(error = %e, "failed to fetch OpenAI image reference URL");
            OpenAiError::Reference(format!("fetch {reference}: {e}"))
        })?;
        if !resp.status().is_success() {
            tracing::warn!(
                status = resp.status().as_u16(),
                "OpenAI image reference URL returned non-success status"
            );
            return Err(OpenAiError::Reference(format!(
                "fetch {reference}: status {}",
                resp.status().as_u16()
            )));
        }
        let mime = resp
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .map(static_mime)
            .unwrap_or("image/png");
        let bytes = resp
            .bytes()
            .await
            .map_err(|e| {
                tracing::warn!(error = %e, "failed to read OpenAI image reference body");
                OpenAiError::Reference(format!("read body {reference}: {e}"))
            })?
            .to_vec();
        tracing::debug!(
            mime_type = mime,
            bytes = bytes.len(),
            "fetched OpenAI image reference URL"
        );
        return Ok(RefImage {
            bytes,
            mime,
            ext: ext_for_mime(mime),
        });
    }

    Err(OpenAiError::Reference(format!(
        "unsupported reference URL: {reference}"
    )))
}

fn image_edit_log_body(
    request: &ImageRequest,
    model: &str,
    quality: Option<&str>,
    size: Option<&str>,
    references: &[RefImage],
) -> Value {
    let images = references
        .iter()
        .enumerate()
        .map(|(i, reference)| {
            json!({
                "field": "image[]",
                "file_name": format!("image_{i}.{}", reference.ext),
                "mime_type": reference.mime,
                "bytes": reference.bytes.len(),
            })
        })
        .collect::<Vec<_>>();
    let mut body = json!({
        "model": model,
        "prompt": request.prompt.as_str(),
        "n": 1,
        "images": images,
    });
    if let Some(quality) = quality {
        body["quality"] = json!(quality);
    }
    if let Some(size) = size {
        body["size"] = json!(size);
    }
    body
}

fn image_from_response(
    parsed: ImagesResponse,
    provider: &ProviderName,
    requested_model: &str,
    pricing: &OpenAiPricing,
) -> Result<GeneratedImage, OpenAiError> {
    let model = parsed
        .model
        .as_deref()
        .map(ModelId::new)
        .unwrap_or_else(|| ModelId::new(requested_model));
    let first = parsed
        .data
        .into_iter()
        .next()
        .ok_or_else(|| OpenAiError::Decode("image response had no data".into()))?;
    let b64 = first
        .b64_json
        .ok_or_else(|| OpenAiError::Decode("image response item lacked b64_json".into()))?;
    let bytes = B64.decode(b64.as_bytes()).map_err(|e| {
        tracing::warn!(error = %e, "failed to decode OpenAI image base64 payload");
        OpenAiError::Decode(format!("base64: {e}"))
    })?;
    let mime_type = first
        .output_format
        .map(|format| static_mime(&format!("image/{format}")).to_string())
        .unwrap_or_else(|| "image/png".to_string());
    let usage = usage_from_openai_image(
        provider,
        Some(model.clone()),
        parsed.usage.as_ref(),
        pricing,
    );
    tracing::info!(
        model = %model,
        mime_type = %mime_type,
        bytes = bytes.len(),
        revised_prompt = first.revised_prompt.is_some(),
        usage_records = usage.iter().count(),
        "OpenAI image generated"
    );

    Ok(GeneratedImage {
        bytes,
        mime_type,
        model,
        revised_prompt: first.revised_prompt,
        usage: usage.into_iter().collect(),
    })
}

async fn send_image_request(
    request: reqwest::RequestBuilder,
    label: &str,
) -> Result<reqwest::Response, OpenAiError> {
    let resp = request.send().await.map_err(|e| {
        tracing::warn!(label, error = %e, "OpenAI image request transport error");
        OpenAiError::Transport(e.to_string())
    })?;
    let status = resp.status();
    tracing::debug!(label, status = %status, "received OpenAI image response");
    Ok(resp)
}

fn resolve_model_and_quality(model: Option<&str>) -> (&str, Option<&'static str>) {
    let Some(raw) = model else {
        return (DEFAULT_IMAGE_MODEL, None);
    };
    match raw.to_ascii_lowercase().as_str() {
        "low" => (DEFAULT_IMAGE_MODEL, Some("low")),
        "medium" | "standard" => (DEFAULT_IMAGE_MODEL, Some("medium")),
        "high" | "quality" => (DEFAULT_IMAGE_MODEL, Some("high")),
        "auto" => (DEFAULT_IMAGE_MODEL, Some("auto")),
        _ => (raw, None),
    }
}

fn map_aspect_to_size(aspect: Option<&str>) -> Option<&'static str> {
    match aspect?.trim() {
        "1:1" => Some("1024x1024"),
        "16:9" | "3:2" | "landscape" => Some("1536x1024"),
        "9:16" | "2:3" | "portrait" => Some("1024x1536"),
        "auto" => Some("auto"),
        _ => Some("auto"),
    }
}

fn data_uri(mime_type: &str, bytes: &[u8]) -> String {
    format!("data:{mime_type};base64,{}", B64.encode(bytes))
}

fn decode_data_uri(uri: &str) -> Result<(&'static str, Vec<u8>), OpenAiError> {
    let rest = uri
        .strip_prefix("data:")
        .ok_or_else(|| OpenAiError::Reference("malformed data URI".into()))?;
    let (meta, payload) = rest
        .split_once(',')
        .ok_or_else(|| OpenAiError::Reference("malformed data URI".into()))?;
    let bytes = B64
        .decode(payload.as_bytes())
        .map_err(|e| OpenAiError::Reference(format!("data URI base64: {e}")))?;
    Ok((
        static_mime(meta.split(';').next().unwrap_or("image/png")),
        bytes,
    ))
}

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

fn usage_from_openai_image(
    provider: &ProviderName,
    model: Option<ModelId>,
    usage: Option<&Value>,
    pricing: &OpenAiPricing,
) -> Option<UsageRecord> {
    let raw = usage?.clone();
    let parsed = serde_json::from_value::<ImageUsage>(raw.clone()).ok()?;
    let cost = pricing.estimate_image_cost(
        model.as_ref(),
        ImagePricingUsage {
            input_tokens: parsed.input_tokens,
            text_input_tokens: parsed.input_tokens_details.text_tokens,
            image_input_tokens: parsed.input_tokens_details.image_tokens,
            cached_text_input_tokens: parsed.input_tokens_details.cached_text_tokens,
            cached_image_input_tokens: parsed.input_tokens_details.cached_image_tokens,
            output_tokens: parsed.output_tokens,
            text_output_tokens: parsed.output_tokens_details.text_tokens,
            image_output_tokens: parsed.output_tokens_details.image_tokens,
        },
    );
    Some(UsageRecord {
        provider: provider.clone(),
        model,
        subject: UsageSubject::ImageGeneration,
        input_tokens: parsed.input_tokens,
        cached_input_tokens: None,
        output_tokens: parsed.output_tokens,
        reasoning_tokens: None,
        total_tokens: parsed.total_tokens,
        cost,
        raw: Some(raw),
    })
}

struct RefImage {
    bytes: Vec<u8>,
    mime: &'static str,
    ext: &'static str,
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
    output_format: Option<String>,
}

#[derive(Deserialize, Debug, Default)]
struct ImageUsage {
    #[serde(default)]
    input_tokens: Option<u64>,
    #[serde(default)]
    input_tokens_details: ImageTokenDetails,
    #[serde(default)]
    output_tokens: Option<u64>,
    #[serde(default)]
    output_tokens_details: ImageTokenDetails,
    #[serde(default)]
    total_tokens: Option<u64>,
}

#[derive(Deserialize, Debug, Default)]
struct ImageTokenDetails {
    #[serde(default)]
    image_tokens: Option<u64>,
    #[serde(default)]
    text_tokens: Option<u64>,
    #[serde(default)]
    cached_image_tokens: Option<u64>,
    #[serde(default)]
    cached_text_tokens: Option<u64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_model_and_quality() {
        assert_eq!(resolve_model_and_quality(None), (DEFAULT_IMAGE_MODEL, None));
        assert_eq!(
            resolve_model_and_quality(Some("high")),
            (DEFAULT_IMAGE_MODEL, Some("high"))
        );
        assert_eq!(
            resolve_model_and_quality(Some("quality")),
            (DEFAULT_IMAGE_MODEL, Some("high"))
        );
        assert_eq!(
            resolve_model_and_quality(Some("standard")),
            (DEFAULT_IMAGE_MODEL, Some("medium"))
        );
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
    fn decodes_data_uri() {
        let (mime, bytes) = decode_data_uri("data:image/webp;base64,QUJD").unwrap();
        assert_eq!(mime, "image/webp");
        assert_eq!(bytes, b"ABC");
    }

    #[test]
    fn parses_image_usage() {
        let provider = ProviderName::new("openai");
        let usage = json!({
            "input_tokens": 12,
            "output_tokens": 200,
            "total_tokens": 212,
        });
        let record = usage_from_openai_image(
            &provider,
            Some(ModelId::new("gpt-image-1")),
            Some(&usage),
            &OpenAiPricing::default(),
        )
        .unwrap();
        assert_eq!(record.input_tokens, Some(12));
        assert_eq!(record.output_tokens, Some(200));
        assert_eq!(record.total_tokens, Some(212));
        assert!(record.cost.is_some());
    }

    #[test]
    fn estimates_image_usage_cost_from_token_details() {
        let provider = ProviderName::new("openai");
        let usage = json!({
            "input_tokens": 25,
            "input_tokens_details": { "text_tokens": 10, "image_tokens": 15 },
            "output_tokens": 100,
            "output_tokens_details": { "image_tokens": 100 },
            "total_tokens": 125,
        });
        let record = usage_from_openai_image(
            &provider,
            Some(ModelId::new("gpt-image-1.5")),
            Some(&usage),
            &OpenAiPricing::default(),
        )
        .unwrap();

        let cost = record.cost.expect("estimated cost");
        assert_eq!(cost.unit, "usd_ticks");
        assert!(cost.estimated);
        assert_eq!(cost.amount, "33700000");
    }
}
