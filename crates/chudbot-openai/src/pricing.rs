//! Built-in OpenAI price tables and local usage-cost estimation.
//!
//! The OpenAI APIs return token usage but not billing totals, so this module
//! converts reported usage into Chudbot's `usd_ticks` accounting unit. The
//! tables here are defaults only: deployment config can override or add model
//! entries through [`OpenAiClient::with_token_pricing`] and
//! [`OpenAiClient::with_image_pricing`].
//!
//! [`OpenAiClient::with_token_pricing`]: crate::OpenAiClient::with_token_pricing
//! [`OpenAiClient::with_image_pricing`]: crate::OpenAiClient::with_image_pricing

use std::collections::BTreeMap;

use chudbot_api::{CostAmount, ModelId};
use serde::{Deserialize, Serialize};

/// Token-count denominator used by OpenAI pricing pages.
const TOKENS_PER_MILLION: f64 = 1_000_000.0;
/// Fixed-point USD scale shared by usage reports.
const USD_TICKS_PER_DOLLAR: f64 = 10_000_000_000.0;

/// Runtime pricing registry for OpenAI text and image providers.
///
/// The maps are keyed by model id so exact deployment overrides can replace
/// built-ins. Lookup also falls back from dated snapshot ids to their base
/// model when there is no exact entry.
#[derive(Debug, Clone)]
pub(crate) struct OpenAiPricing {
    token: BTreeMap<ModelId, OpenAiTokenPricing>,
    image: BTreeMap<ModelId, OpenAiImagePricing>,
}

impl Default for OpenAiPricing {
    fn default() -> Self {
        Self {
            token: default_token_pricing(),
            image: default_image_pricing(),
        }
    }
}

impl OpenAiPricing {
    /// Merge configured text-token pricing into the built-in defaults.
    pub(crate) fn apply_token_overrides(&mut self, pricing: BTreeMap<ModelId, OpenAiTokenPricing>) {
        self.token.extend(pricing);
    }

    /// Merge configured image-token pricing into the built-in defaults.
    pub(crate) fn apply_image_overrides(&mut self, pricing: BTreeMap<ModelId, OpenAiImagePricing>) {
        self.image.extend(pricing);
    }

    /// Estimate Responses API usage cost from reported token buckets.
    ///
    /// Returns `None` when there is no model, no known pricing entry, or the
    /// billable estimate is zero ticks. Cached input tokens use the
    /// cached rate when one is configured and otherwise fall back to regular
    /// input pricing.
    pub(crate) fn estimate_token_cost(
        &self,
        model: Option<&ModelId>,
        input_tokens: u64,
        cached_input_tokens: u64,
        output_tokens: u64,
    ) -> Option<CostAmount> {
        let pricing = pricing_for_model(&self.token, model?)?;
        let uncached_input_tokens = input_tokens.saturating_sub(cached_input_tokens);
        let cached_price = pricing
            .cached_input_usd_per_million_tokens
            .unwrap_or(pricing.input_usd_per_million_tokens);
        let ticks =
            usd_ticks_for_tokens(uncached_input_tokens, pricing.input_usd_per_million_tokens)
                .saturating_add(usd_ticks_for_tokens(cached_input_tokens, cached_price))
                .saturating_add(usd_ticks_for_tokens(
                    output_tokens,
                    pricing.output_usd_per_million_tokens,
                ));
        cost_from_ticks(ticks)
    }

    /// Estimate image-generation usage cost from the token buckets OpenAI may report.
    ///
    /// Image endpoints have changed their usage shape over time. This accepts
    /// both aggregate and modality-specific buckets, then charges text input,
    /// image input, optional text output, and image output independently.
    pub(crate) fn estimate_image_cost(
        &self,
        model: Option<&ModelId>,
        usage: ImagePricingUsage,
    ) -> Option<CostAmount> {
        let pricing = pricing_for_model(&self.image, model?)?;
        // Older payloads can report only aggregate input tokens. Treat the
        // portion not attributed to image input as text input.
        let text_input_tokens = usage.text_input_tokens.unwrap_or_else(|| {
            usage.input_tokens.unwrap_or(0).saturating_sub(
                usage
                    .image_input_tokens
                    .or(usage.cached_image_input_tokens)
                    .unwrap_or(0),
            )
        });
        let image_input_tokens = usage.image_input_tokens.unwrap_or(0);
        let cached_text_input_tokens = usage.cached_text_input_tokens.unwrap_or(0);
        let cached_image_input_tokens = usage.cached_image_input_tokens.unwrap_or(0);
        // If only aggregate output is present, prefer image output because the
        // image endpoint's billable output is normally the generated image.
        let fallback_output_tokens = usage.output_tokens.unwrap_or(0).saturating_sub(
            usage
                .text_output_tokens
                .or(usage.image_output_tokens)
                .unwrap_or(0),
        );
        let text_output_tokens = usage.text_output_tokens.unwrap_or(0);
        let image_output_tokens = usage.image_output_tokens.unwrap_or(fallback_output_tokens);

        let mut ticks: u64 = 0;
        ticks = ticks.saturating_add(usd_ticks_for_tokens(
            text_input_tokens.saturating_sub(cached_text_input_tokens),
            pricing.text_input_usd_per_million_tokens,
        ));
        ticks = ticks.saturating_add(usd_ticks_for_tokens(
            cached_text_input_tokens,
            pricing
                .cached_text_input_usd_per_million_tokens
                .unwrap_or(pricing.text_input_usd_per_million_tokens),
        ));
        ticks = ticks.saturating_add(usd_ticks_for_tokens(
            image_input_tokens.saturating_sub(cached_image_input_tokens),
            pricing.image_input_usd_per_million_tokens,
        ));
        ticks = ticks.saturating_add(usd_ticks_for_tokens(
            cached_image_input_tokens,
            pricing
                .cached_image_input_usd_per_million_tokens
                .unwrap_or(pricing.image_input_usd_per_million_tokens),
        ));
        if let Some(price) = pricing.text_output_usd_per_million_tokens {
            ticks = ticks.saturating_add(usd_ticks_for_tokens(text_output_tokens, price));
        }
        ticks = ticks.saturating_add(usd_ticks_for_tokens(
            image_output_tokens,
            pricing.image_output_usd_per_million_tokens,
        ));
        cost_from_ticks(ticks)
    }
}

/// OpenAI text-token pricing for one model, in USD per 1M tokens.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct OpenAiTokenPricing {
    /// Uncached input token price in USD per 1M tokens.
    pub input_usd_per_million_tokens: f64,
    /// Cached input token price in USD per 1M tokens.
    ///
    /// When omitted, cached input is charged at the uncached input rate.
    #[serde(default)]
    pub cached_input_usd_per_million_tokens: Option<f64>,
    /// Output token price in USD per 1M tokens.
    pub output_usd_per_million_tokens: f64,
}

/// OpenAI image-token pricing for one model, in USD per 1M tokens.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct OpenAiImagePricing {
    /// Text input token price in USD per 1M tokens.
    pub text_input_usd_per_million_tokens: f64,
    /// Cached text input token price in USD per 1M tokens.
    ///
    /// When omitted, cached text input is charged at the uncached text rate.
    #[serde(default)]
    pub cached_text_input_usd_per_million_tokens: Option<f64>,
    /// Image input token price in USD per 1M tokens.
    pub image_input_usd_per_million_tokens: f64,
    /// Cached image input token price in USD per 1M tokens.
    ///
    /// When omitted, cached image input is charged at the uncached image rate.
    #[serde(default)]
    pub cached_image_input_usd_per_million_tokens: Option<f64>,
    /// Image output token price in USD per 1M tokens.
    pub image_output_usd_per_million_tokens: f64,
    /// Text output token price in USD per 1M tokens, when the model reports it.
    #[serde(default)]
    pub text_output_usd_per_million_tokens: Option<f64>,
}

/// Normalized image usage fields accepted by the image cost estimator.
///
/// Fields are optional because OpenAI image usage payloads can include only
/// aggregate buckets, only modality-specific buckets, or a mix of both.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct ImagePricingUsage {
    /// Aggregate input tokens across text and image inputs.
    pub(crate) input_tokens: Option<u64>,
    /// Input tokens attributable to text prompts.
    pub(crate) text_input_tokens: Option<u64>,
    /// Input tokens attributable to image references.
    pub(crate) image_input_tokens: Option<u64>,
    /// Cached text input tokens.
    pub(crate) cached_text_input_tokens: Option<u64>,
    /// Cached image input tokens.
    pub(crate) cached_image_input_tokens: Option<u64>,
    /// Aggregate output tokens across text and generated-image output.
    pub(crate) output_tokens: Option<u64>,
    /// Output tokens attributable to text emitted by the model.
    pub(crate) text_output_tokens: Option<u64>,
    /// Output tokens attributable to generated images.
    pub(crate) image_output_tokens: Option<u64>,
}

/// Built-in Responses API prices used when config does not override a model.
fn default_token_pricing() -> BTreeMap<ModelId, OpenAiTokenPricing> {
    let mut pricing = BTreeMap::new();
    pricing.insert(
        ModelId::new("gpt-5.5"),
        OpenAiTokenPricing {
            input_usd_per_million_tokens: 5.00,
            cached_input_usd_per_million_tokens: Some(0.50),
            output_usd_per_million_tokens: 30.00,
        },
    );
    pricing.insert(
        ModelId::new("gpt-5.5-pro"),
        OpenAiTokenPricing {
            input_usd_per_million_tokens: 30.00,
            cached_input_usd_per_million_tokens: None,
            output_usd_per_million_tokens: 180.00,
        },
    );
    pricing.insert(
        ModelId::new("gpt-5.4"),
        OpenAiTokenPricing {
            input_usd_per_million_tokens: 2.50,
            cached_input_usd_per_million_tokens: Some(0.25),
            output_usd_per_million_tokens: 15.00,
        },
    );
    pricing.insert(
        ModelId::new("gpt-5.4-mini"),
        OpenAiTokenPricing {
            input_usd_per_million_tokens: 0.75,
            cached_input_usd_per_million_tokens: Some(0.075),
            output_usd_per_million_tokens: 4.50,
        },
    );
    pricing.insert(
        ModelId::new("gpt-5.4-nano"),
        OpenAiTokenPricing {
            input_usd_per_million_tokens: 0.20,
            cached_input_usd_per_million_tokens: Some(0.02),
            output_usd_per_million_tokens: 1.25,
        },
    );
    pricing.insert(
        ModelId::new("gpt-5.4-pro"),
        OpenAiTokenPricing {
            input_usd_per_million_tokens: 30.00,
            cached_input_usd_per_million_tokens: None,
            output_usd_per_million_tokens: 180.00,
        },
    );
    pricing.insert(
        ModelId::new("chat-latest"),
        OpenAiTokenPricing {
            input_usd_per_million_tokens: 5.00,
            cached_input_usd_per_million_tokens: Some(0.50),
            output_usd_per_million_tokens: 30.00,
        },
    );
    pricing.insert(
        ModelId::new("gpt-5.3-codex"),
        OpenAiTokenPricing {
            input_usd_per_million_tokens: 1.75,
            cached_input_usd_per_million_tokens: Some(0.175),
            output_usd_per_million_tokens: 14.00,
        },
    );
    pricing
}

/// Built-in image-generation prices used when config does not override a model.
fn default_image_pricing() -> BTreeMap<ModelId, OpenAiImagePricing> {
    let mut pricing = BTreeMap::new();
    pricing.insert(
        ModelId::new("gpt-image-2"),
        OpenAiImagePricing {
            text_input_usd_per_million_tokens: 5.00,
            cached_text_input_usd_per_million_tokens: Some(1.25),
            image_input_usd_per_million_tokens: 8.00,
            cached_image_input_usd_per_million_tokens: Some(2.00),
            image_output_usd_per_million_tokens: 30.00,
            text_output_usd_per_million_tokens: None,
        },
    );
    pricing.insert(
        ModelId::new("gpt-image-1.5"),
        OpenAiImagePricing {
            text_input_usd_per_million_tokens: 5.00,
            cached_text_input_usd_per_million_tokens: Some(1.25),
            image_input_usd_per_million_tokens: 8.00,
            cached_image_input_usd_per_million_tokens: Some(2.00),
            image_output_usd_per_million_tokens: 32.00,
            text_output_usd_per_million_tokens: Some(10.00),
        },
    );
    pricing.insert(
        ModelId::new("gpt-image-1-mini"),
        OpenAiImagePricing {
            text_input_usd_per_million_tokens: 2.00,
            cached_text_input_usd_per_million_tokens: Some(0.20),
            image_input_usd_per_million_tokens: 2.50,
            cached_image_input_usd_per_million_tokens: Some(0.25),
            image_output_usd_per_million_tokens: 8.00,
            text_output_usd_per_million_tokens: None,
        },
    );
    pricing.insert(
        ModelId::new("gpt-image-1"),
        OpenAiImagePricing {
            text_input_usd_per_million_tokens: 5.00,
            cached_text_input_usd_per_million_tokens: None,
            image_input_usd_per_million_tokens: 10.00,
            cached_image_input_usd_per_million_tokens: None,
            image_output_usd_per_million_tokens: 40.00,
            text_output_usd_per_million_tokens: None,
        },
    );
    pricing
}

/// Convert a token count and USD-per-million rate into `usd_ticks`.
fn usd_ticks_for_tokens(tokens: u64, usd_per_million_tokens: f64) -> u64 {
    if tokens == 0 || !usd_per_million_tokens.is_finite() || usd_per_million_tokens <= 0.0 {
        return 0;
    }
    // Round up so tiny but nonzero billable usage is not silently dropped.
    let ticks = ((tokens as f64) * usd_per_million_tokens * USD_TICKS_PER_DOLLAR
        / TOKENS_PER_MILLION)
        .ceil();
    if ticks >= u64::MAX as f64 {
        u64::MAX
    } else {
        ticks as u64
    }
}

/// Build a usage cost value when there is a nonzero estimate.
fn cost_from_ticks(ticks: u64) -> Option<CostAmount> {
    (ticks > 0).then(|| CostAmount {
        amount: ticks.to_string(),
        unit: "usd_ticks".to_string(),
        estimated: true,
    })
}

/// Find exact pricing first, then fall back from dated snapshot ids.
fn pricing_for_model<'a, T>(pricing: &'a BTreeMap<ModelId, T>, model: &ModelId) -> Option<&'a T> {
    pricing.get(model).or_else(|| {
        strip_dated_model_suffix(model.as_str()).and_then(|base| pricing.get(&ModelId::new(base)))
    })
}

/// Strip `-YYYY-MM-DD` model suffixes used by OpenAI snapshot ids.
fn strip_dated_model_suffix(model: &str) -> Option<&str> {
    const SUFFIX_LEN: usize = "-YYYY-MM-DD".len();
    let suffix_start = model.len().checked_sub(SUFFIX_LEN)?;
    if !model.is_char_boundary(suffix_start) {
        return None;
    }
    let suffix = model.as_bytes().get(suffix_start..)?;
    let dated = suffix[0] == b'-'
        && suffix[1..5].iter().all(u8::is_ascii_digit)
        && suffix[5] == b'-'
        && suffix[6..8].iter().all(u8::is_ascii_digit)
        && suffix[8] == b'-'
        && suffix[9..11].iter().all(u8::is_ascii_digit);
    dated.then_some(&model[..suffix_start])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn estimates_text_tokens_with_cached_discount() {
        let pricing = OpenAiPricing::default();

        let cost = pricing
            .estimate_token_cost(Some(&ModelId::new("gpt-5.5")), 1_000_000, 250_000, 100_000)
            .expect("cost estimate");

        assert_eq!(cost.unit, "usd_ticks");
        assert!(cost.estimated);
        assert_eq!(cost.amount, "68750000000");
    }

    #[test]
    fn token_overrides_replace_builtin_prices() {
        let mut pricing = OpenAiPricing::default();
        pricing.apply_token_overrides(BTreeMap::from([(
            ModelId::new("gpt-5.5"),
            OpenAiTokenPricing {
                input_usd_per_million_tokens: 1.00,
                cached_input_usd_per_million_tokens: Some(0.10),
                output_usd_per_million_tokens: 2.00,
            },
        )]));

        let cost = pricing
            .estimate_token_cost(Some(&ModelId::new("gpt-5.5")), 100, 50, 10)
            .expect("cost estimate");

        assert_eq!(cost.amount, "750000");
    }

    #[test]
    fn dated_snapshot_models_use_base_model_pricing() {
        let pricing = OpenAiPricing::default();

        let cost = pricing
            .estimate_token_cost(Some(&ModelId::new("gpt-5.4-mini-2026-03-17")), 100, 40, 20)
            .expect("cost estimate");

        assert_eq!(cost.amount, "1380000");
    }

    #[test]
    fn estimates_image_tokens_by_modality() {
        let pricing = OpenAiPricing::default();

        let cost = pricing
            .estimate_image_cost(
                Some(&ModelId::new("gpt-image-1.5")),
                ImagePricingUsage {
                    input_tokens: Some(320),
                    text_input_tokens: Some(120),
                    image_input_tokens: Some(200),
                    output_tokens: Some(1_000),
                    image_output_tokens: Some(1_000),
                    ..ImagePricingUsage::default()
                },
            )
            .expect("cost estimate");

        assert_eq!(cost.amount, "342000000");
    }

    #[test]
    fn exact_pricing_overrides_snapshot_fallback() {
        let mut pricing = OpenAiPricing::default();
        pricing.apply_token_overrides(BTreeMap::from([(
            ModelId::new("gpt-5.4-mini-2026-03-17"),
            OpenAiTokenPricing {
                input_usd_per_million_tokens: 1.00,
                cached_input_usd_per_million_tokens: Some(0.10),
                output_usd_per_million_tokens: 2.00,
            },
        )]));

        let cost = pricing
            .estimate_token_cost(Some(&ModelId::new("gpt-5.4-mini-2026-03-17")), 100, 40, 20)
            .expect("cost estimate");

        assert_eq!(cost.amount, "1040000");
    }
}
