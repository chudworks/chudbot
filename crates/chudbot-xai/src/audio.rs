//! xAI speech-to-text implementation.

use chudbot_api::{
    AudioTranscriber, AudioTranscriptChannel, AudioTranscriptWord, AudioTranscription,
    AudioTranscriptionRequest, CostAmount, MediaRef, ModelId, ProviderName, UsageRecord,
    UsageSubject,
};
use reqwest::multipart::{Form, Part};
use serde::Deserialize;
use serde_json::Value;

use crate::imagine::{media_provider_url, usage_from_xai_media};
use crate::{XaiClient, XaiError, decode_response};

const STT_ENDPOINT: &str = "/stt";
const XAI_STT_REST_USD_TICKS_PER_HOUR: f64 = 1_000_000_000.0;

impl AudioTranscriber for XaiClient {
    type Error = XaiError;

    fn backend_name(&self) -> &ProviderName {
        self.provider_name()
    }

    #[tracing::instrument(name = "xai.transcribe_audio", skip_all)]
    async fn transcribe_audio(
        &self,
        request: AudioTranscriptionRequest,
    ) -> Result<AudioTranscription, Self::Error> {
        let audio = resolve_stt_audio(request.audio.as_ref()).await?;
        let model = request.model.clone();
        tracing::debug!(
            audio_source = audio.kind(),
            language = ?request.language.as_deref(),
            keyterms = request.keyterms.len(),
            model = ?model.as_ref().map(ModelId::as_str),
            "building xAI STT request"
        );
        let url = format!("{}{}", self.base_url(), STT_ENDPOINT);
        let raw: Value = chudbot_api::retry::with_retry(
            chudbot_api::retry::RetryPolicy::default(),
            "stt[xai]",
            || {
                let form = build_stt_form(&audio, request.language.as_deref(), &request.keyterms);
                let request = self
                    .http()
                    .post(&url)
                    .bearer_auth(self.api_key())
                    .multipart(form);
                async move {
                    let resp = request.send().await.map_err(|e| {
                        tracing::warn!(error = %e, "xAI STT transport error");
                        XaiError::Transport(e.to_string())
                    })?;
                    tracing::debug!(status = resp.status().as_u16(), "received xAI STT response");
                    decode_response(resp, self.provider_name(), STT_ENDPOINT).await
                }
            },
        )
        .await?;
        let parsed: SttResponse = serde_json::from_value(raw.clone()).map_err(|error| {
            tracing::warn!(error = %error, "failed to decode xAI STT response shape");
            XaiError::Decode(error.to_string())
        })?;
        let duration_seconds = parsed.duration.unwrap_or(0.0);
        let language = parsed
            .language
            .filter(|language| !language.trim().is_empty());
        let usage = stt_usage_with_cost_estimate(
            self.provider_name(),
            model.clone(),
            parsed.usage.as_ref(),
            duration_seconds,
            raw.clone(),
        );
        tracing::info!(
            duration_seconds,
            text_chars = parsed.text.chars().count(),
            words = parsed.words.len(),
            channels = parsed.channels.len(),
            usage_records = usage.len(),
            "xAI audio transcribed"
        );
        Ok(AudioTranscription {
            text: parsed.text,
            language,
            duration_seconds,
            words: parsed.words.into_iter().map(Into::into).collect(),
            channels: parsed.channels.into_iter().map(Into::into).collect(),
            model,
            usage,
        })
    }
}

#[derive(Debug, Clone)]
enum SttAudio {
    File {
        file_name: String,
        mime_type: String,
        bytes: Vec<u8>,
    },
    Url {
        url: String,
    },
}

impl SttAudio {
    fn kind(&self) -> &'static str {
        match self {
            Self::File { .. } => "file",
            Self::Url { .. } => "url",
        }
    }
}

async fn resolve_stt_audio(audio: &dyn MediaRef) -> Result<SttAudio, XaiError> {
    match audio.load().await {
        Ok(loaded) => Ok(SttAudio::File {
            file_name: loaded.media.name().to_string(),
            mime_type: loaded.media.mime_type().to_string(),
            bytes: loaded.bytes,
        }),
        Err(load_error) => {
            tracing::debug!(
                uri = %audio.uri(),
                error = %load_error,
                "falling back to public URL for xAI STT audio"
            );
            let url = media_provider_url(audio).await?;
            Ok(SttAudio::Url { url })
        }
    }
}

fn build_stt_form(audio: &SttAudio, language: Option<&str>, keyterms: &[String]) -> Form {
    let mut form = Form::new();
    if let Some(language) = language {
        form = form
            .text("format", "true")
            .text("language", language.to_string());
    }
    for keyterm in keyterms {
        form = form.text("keyterm", keyterm.clone());
    }
    match audio {
        SttAudio::File {
            file_name,
            mime_type,
            bytes,
        } => {
            let part = Part::bytes(bytes.clone()).file_name(file_name.clone());
            let part = part.mime_str(mime_type).unwrap_or_else(|error| {
                tracing::warn!(
                    mime_type = %mime_type,
                    error = %error,
                    "failed to apply audio MIME type to STT multipart part"
                );
                Part::bytes(bytes.clone()).file_name(file_name.clone())
            });
            form = form.part("file", part);
        }
        SttAudio::Url { url } => {
            form = form.text("url", url.clone());
        }
    }
    form
}

fn stt_usage_with_cost_estimate(
    provider: &ProviderName,
    model: Option<ModelId>,
    usage: Option<&Value>,
    duration_seconds: f64,
    raw: Value,
) -> Vec<UsageRecord> {
    let mut records = usage_from_xai_media(
        provider,
        model.clone(),
        UsageSubject::AudioTranscription,
        usage,
    )
    .into_iter()
    .collect::<Vec<_>>();
    let estimate = estimated_stt_usage(provider, model, duration_seconds, raw);
    if records.is_empty() {
        return vec![estimate];
    }
    if records.iter().all(|record| record.cost.is_none())
        && let Some(cost) = estimate.cost
    {
        records[0].cost = Some(cost);
    }
    records
}

fn estimated_stt_usage(
    provider: &ProviderName,
    model: Option<ModelId>,
    duration_seconds: f64,
    raw: Value,
) -> UsageRecord {
    let ticks =
        ((duration_seconds.max(0.0) * XAI_STT_REST_USD_TICKS_PER_HOUR) / 3600.0).ceil() as u64;
    let cost = (ticks > 0).then(|| CostAmount {
        amount: ticks.to_string(),
        unit: "usd_ticks".to_string(),
        estimated: true,
    });
    UsageRecord {
        provider: provider.clone(),
        model,
        subject: UsageSubject::AudioTranscription,
        input_tokens: None,
        cached_input_tokens: None,
        output_tokens: None,
        reasoning_tokens: None,
        total_tokens: None,
        cost,
        raw: Some(raw),
    }
}

#[derive(Debug, Deserialize)]
struct SttResponse {
    text: String,
    #[serde(default)]
    language: Option<String>,
    #[serde(default)]
    duration: Option<f64>,
    #[serde(default)]
    words: Vec<SttWord>,
    #[serde(default)]
    channels: Vec<SttChannel>,
    #[serde(default)]
    usage: Option<Value>,
}

#[derive(Debug, Deserialize)]
struct SttWord {
    text: String,
    #[serde(default)]
    start: f64,
    #[serde(default)]
    end: f64,
    #[serde(default)]
    confidence: Option<f64>,
    #[serde(default)]
    speaker: Option<u32>,
}

impl From<SttWord> for AudioTranscriptWord {
    fn from(word: SttWord) -> Self {
        Self {
            text: word.text,
            start_seconds: word.start,
            end_seconds: word.end,
            confidence: word.confidence,
            speaker: word.speaker,
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn stt_usage_estimates_cost_when_provider_usage_lacks_cost() {
        let provider = ProviderName::new("xai");
        let usage = json!({
            "input_tokens": 10,
            "input_tokens_details": { "cached_tokens": 3 },
            "output_tokens": 20,
            "output_tokens_details": { "reasoning_tokens": 4 },
            "total_tokens": 30,
            "cost_in_usd_ticks": 0
        });

        let records = stt_usage_with_cost_estimate(
            &provider,
            Some(ModelId::new("grok-stt")),
            Some(&usage),
            360.0,
            json!({ "text": "hello", "duration": 360.0, "usage": usage }),
        );

        assert_eq!(records.len(), 1);
        assert_eq!(records[0].input_tokens, Some(10));
        assert_eq!(records[0].cached_input_tokens, Some(3));
        let cost = records[0].cost.as_ref().expect("estimated cost");
        assert_eq!(cost.amount, "100000000");
        assert_eq!(cost.unit, "usd_ticks");
        assert!(cost.estimated);
    }

    #[test]
    fn stt_usage_preserves_provider_cost_when_present() {
        let provider = ProviderName::new("xai");
        let usage = json!({
            "input_tokens": 0,
            "input_tokens_details": { "cached_tokens": 0 },
            "output_tokens": 0,
            "output_tokens_details": { "reasoning_tokens": 0 },
            "total_tokens": 0,
            "cost_in_usd_ticks": 123
        });

        let records = stt_usage_with_cost_estimate(
            &provider,
            None,
            Some(&usage),
            360.0,
            json!({ "text": "hello", "duration": 360.0, "usage": usage }),
        );

        let cost = records[0].cost.as_ref().expect("provider cost");
        assert_eq!(cost.amount, "123");
        assert!(!cost.estimated);
    }
}

#[derive(Debug, Deserialize)]
struct SttChannel {
    #[serde(default)]
    index: u32,
    #[serde(default)]
    text: String,
    #[serde(default)]
    words: Vec<SttWord>,
}

impl From<SttChannel> for AudioTranscriptChannel {
    fn from(channel: SttChannel) -> Self {
        Self {
            index: channel.index,
            text: channel.text,
            words: channel.words.into_iter().map(Into::into).collect(),
        }
    }
}
