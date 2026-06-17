//! Gemini image generation through the `generateContent` API.
//!
//! Chudbot's provider-neutral image request is translated into a single Gemini
//! user turn with one text part and zero or more inline image parts. Gemini can
//! return both explanatory text and generated image bytes in the same candidate,
//! so response decoding preserves text as the optional revised prompt while
//! taking the first inline image payload as the generated image.

use chudbot_api::{
    GeneratedImage, ImageGenerator, ImageRequest, MediaRef, ModelId, ProviderName, UsageSubject,
};
use serde_json::{Value, json};

use crate::llm::usage_from_gemini;
use crate::{GeminiClient, GeminiError, get_field, inline_media, json_strip_nulls};

const DEFAULT_IMAGE_MODEL: &str = "gemini-3.1-flash-image";

impl ImageGenerator for GeminiClient {
    type Error = GeminiError;

    fn backend_name(&self) -> &ProviderName {
        self.provider_name()
    }

    #[tracing::instrument(name = "gemini.generate_image", skip_all)]
    async fn generate_image(&self, request: ImageRequest) -> Result<GeneratedImage, Self::Error> {
        // Gemini image generation uses the same generateContent endpoint as
        // chat, but constrains the generation config to request image output.
        let model = request
            .model
            .as_ref()
            .map(ModelId::as_str)
            .unwrap_or(DEFAULT_IMAGE_MODEL);
        tracing::debug!(
            prompt_chars = request.prompt.chars().count(),
            references = request.references.len(),
            aspect_ratio = ?request.aspect_ratio.as_deref(),
            model = %model,
            "building Gemini image request"
        );

        let mut parts = Vec::with_capacity(request.references.len() + 1);
        parts.push(json!({ "text": request.prompt }));
        for reference in &request.references {
            // References are inlined because the public Gemini endpoint accepts
            // image inputs as content parts, not as multipart upload fields.
            parts.push(reference_image(reference.as_ref()).await?);
        }

        let generation_config = image_generation_config(request.aspect_ratio.as_deref());
        let body = json_strip_nulls(json!({
            "contents": [{
                "role": "user",
                "parts": parts,
            }],
            "generationConfig": generation_config,
        }));
        let endpoint = format!("/models/{model}:generateContent");
        let parsed: Value = self.post_json(&endpoint, &body, "imagegen[gemini]").await?;
        image_from_response(parsed, self.provider_name(), model)
    }
}

/// Convert a provider-neutral media reference into a Gemini inline image part.
async fn reference_image(media: &dyn MediaRef) -> Result<Value, GeminiError> {
    let mime_type = media.mime_type();
    if !mime_type.starts_with("image/") {
        return Err(GeminiError::Reference(format!(
            "media `{}` has MIME type `{mime_type}`, but Gemini image generation accepts image references here",
            media.uri()
        )));
    }
    inline_media(media).await
}

/// Build Gemini's image generation controls from Chudbot's shared request knobs.
fn image_generation_config(aspect_ratio: Option<&str>) -> Option<Value> {
    // Ask Gemini for both text and image output: text is useful when the model
    // revises or explains the prompt, while inlineData carries the image bytes.
    let image_config = json_strip_nulls(json!({
        "aspectRatio": aspect_ratio,
    }));
    let value = json_strip_nulls(json!({
        "responseModalities": ["TEXT", "IMAGE"],
        "imageConfig": match &image_config {
            Value::Object(map) if map.is_empty() => Value::Null,
            _ => image_config,
        },
    }));
    match &value {
        Value::Object(map) if map.is_empty() => None,
        _ => Some(value),
    }
}

/// Decode Gemini's mixed text/image response into Chudbot's generated image.
fn image_from_response(
    response: Value,
    provider: &ProviderName,
    requested_model: &str,
) -> Result<GeneratedImage, GeminiError> {
    let candidate = response
        .get("candidates")
        .and_then(Value::as_array)
        .and_then(|items| items.first())
        .ok_or_else(|| GeminiError::Decode("image response had no candidates".to_string()))?;
    let parts = candidate
        .get("content")
        .and_then(|content| content.get("parts"))
        .and_then(Value::as_array)
        .ok_or_else(|| GeminiError::Decode("image response candidate had no parts".to_string()))?;
    let mut revised_prompt = String::new();
    for part in parts {
        if let Some(text) = part.get("text").and_then(Value::as_str) {
            revised_prompt.push_str(text);
        }
        if let Some(inline_data) = get_field(part, "inlineData", "inline_data") {
            // Gemini may use either camelCase or snake_case in examples and
            // responses; accept both so recorded raw payloads remain decodable.
            let data = get_field(inline_data, "data", "data")
                .and_then(Value::as_str)
                .ok_or_else(|| GeminiError::Decode("image inlineData lacked data".to_string()))?;
            let bytes =
                base64::Engine::decode(&base64::engine::general_purpose::STANDARD, data.as_bytes())
                    .map_err(|e| GeminiError::Decode(format!("base64: {e}")))?;
            let mime_type = get_field(inline_data, "mimeType", "mime_type")
                .and_then(Value::as_str)
                .unwrap_or("image/png")
                .to_string();
            let model = response
                .get("modelVersion")
                .or_else(|| response.get("model_version"))
                .and_then(Value::as_str)
                .map(ModelId::new)
                .unwrap_or_else(|| ModelId::new(requested_model));
            let usage = usage_from_gemini(
                provider,
                Some(model.clone()),
                UsageSubject::ImageGeneration,
                get_field(&response, "usageMetadata", "usage_metadata"),
            )
            .into_iter()
            .collect();
            tracing::info!(
                model = %model,
                mime_type = %mime_type,
                bytes = bytes.len(),
                revised_prompt = !revised_prompt.is_empty(),
                "Gemini image generated"
            );
            return Ok(GeneratedImage {
                bytes,
                mime_type,
                model,
                revised_prompt: (!revised_prompt.is_empty()).then_some(revised_prompt),
                usage,
            });
        }
    }

    Err(GeminiError::Decode(
        "image response had no inline image data".to_string(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn image_config_requests_image_modality() {
        let value = image_generation_config(Some("16:9")).unwrap();

        assert_eq!(value["responseModalities"], json!(["TEXT", "IMAGE"]));
        assert_eq!(value["imageConfig"]["aspectRatio"], "16:9");
    }
}
