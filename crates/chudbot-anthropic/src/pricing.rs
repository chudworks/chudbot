use std::collections::BTreeMap;

use chudbot_api::{CostAmount, ModelId};
use serde::{Deserialize, Serialize};

const CACHE_CREATION_5M_MULTIPLIER: f64 = 1.25;
const CACHE_CREATION_1H_MULTIPLIER: f64 = 2.0;
const CACHE_READ_MULTIPLIER: f64 = 0.1;
const TOKENS_PER_MILLION: f64 = 1_000_000.0;
const USD_TICKS_PER_DOLLAR: f64 = 10_000_000_000.0;

#[derive(Debug, Clone)]
pub(crate) struct AnthropicPricing {
    token: BTreeMap<ModelId, AnthropicTokenPricing>,
}

impl Default for AnthropicPricing {
    fn default() -> Self {
        Self {
            token: default_token_pricing(),
        }
    }
}

impl AnthropicPricing {
    pub(crate) fn apply_token_overrides(
        &mut self,
        pricing: BTreeMap<ModelId, AnthropicTokenPricing>,
    ) {
        self.token.extend(pricing);
    }

    pub(crate) fn estimate_token_cost(
        &self,
        model: Option<&ModelId>,
        usage: AnthropicTokenUsage,
    ) -> Option<CostAmount> {
        let pricing = pricing_for_model(&self.token, model?)?;
        let multiplier = usage.inference_geo_price_multiplier();
        let ticks = usd_ticks_for_tokens(
            usage.input_tokens,
            pricing.input_usd_per_million_tokens * multiplier,
        )
        .saturating_add(usd_ticks_for_tokens(
            usage.cache_creation_5m_input_tokens,
            pricing.cache_creation_5m_price() * multiplier,
        ))
        .saturating_add(usd_ticks_for_tokens(
            usage.cache_creation_1h_input_tokens,
            pricing.cache_creation_1h_price() * multiplier,
        ))
        .saturating_add(usd_ticks_for_tokens(
            usage.cache_read_input_tokens,
            pricing.cache_read_price() * multiplier,
        ))
        .saturating_add(usd_ticks_for_tokens(
            usage.output_tokens,
            pricing.output_usd_per_million_tokens * multiplier,
        ));
        cost_from_ticks(ticks)
    }
}

/// Anthropic text-token pricing for one model, in USD per 1M tokens.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct AnthropicTokenPricing {
    /// Uncached input token price in USD per 1M tokens.
    pub input_usd_per_million_tokens: f64,
    /// 5-minute cache write token price in USD per 1M tokens.
    ///
    /// Defaults to 1.25x the uncached input price.
    #[serde(default)]
    pub cache_creation_5m_usd_per_million_tokens: Option<f64>,
    /// 1-hour cache write token price in USD per 1M tokens.
    ///
    /// Defaults to 2x the uncached input price.
    #[serde(default)]
    pub cache_creation_1h_usd_per_million_tokens: Option<f64>,
    /// Cache hit/read token price in USD per 1M tokens.
    ///
    /// Defaults to 0.1x the uncached input price.
    #[serde(default)]
    pub cache_read_usd_per_million_tokens: Option<f64>,
    /// Output token price in USD per 1M tokens.
    pub output_usd_per_million_tokens: f64,
}

impl AnthropicTokenPricing {
    fn cache_creation_5m_price(&self) -> f64 {
        self.cache_creation_5m_usd_per_million_tokens
            .unwrap_or(self.input_usd_per_million_tokens * CACHE_CREATION_5M_MULTIPLIER)
    }

    fn cache_creation_1h_price(&self) -> f64 {
        self.cache_creation_1h_usd_per_million_tokens
            .unwrap_or(self.input_usd_per_million_tokens * CACHE_CREATION_1H_MULTIPLIER)
    }

    fn cache_read_price(&self) -> f64 {
        self.cache_read_usd_per_million_tokens
            .unwrap_or(self.input_usd_per_million_tokens * CACHE_READ_MULTIPLIER)
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct AnthropicTokenUsage<'a> {
    pub(crate) input_tokens: u64,
    pub(crate) cache_creation_5m_input_tokens: u64,
    pub(crate) cache_creation_1h_input_tokens: u64,
    pub(crate) cache_read_input_tokens: u64,
    pub(crate) output_tokens: u64,
    pub(crate) inference_geo: Option<&'a str>,
}

impl AnthropicTokenUsage<'_> {
    fn inference_geo_price_multiplier(&self) -> f64 {
        match self.inference_geo {
            Some("us") => 1.1,
            _ => 1.0,
        }
    }
}

fn default_token_pricing() -> BTreeMap<ModelId, AnthropicTokenPricing> {
    let mut pricing = BTreeMap::new();
    insert_token_price(&mut pricing, "claude-fable-5", 10.00, 50.00);
    insert_token_price(&mut pricing, "claude-mythos-5", 10.00, 50.00);
    insert_token_price(&mut pricing, "claude-opus-4-8", 5.00, 25.00);
    insert_token_price(&mut pricing, "claude-opus-4-7", 5.00, 25.00);
    insert_token_price(&mut pricing, "claude-opus-4-6", 5.00, 25.00);
    insert_token_price(&mut pricing, "claude-opus-4-5", 5.00, 25.00);
    insert_token_price(&mut pricing, "claude-opus-4-1", 15.00, 75.00);
    insert_token_price(&mut pricing, "claude-opus-4", 15.00, 75.00);
    insert_token_price(&mut pricing, "claude-sonnet-4-6", 3.00, 15.00);
    insert_token_price(&mut pricing, "claude-sonnet-4-5", 3.00, 15.00);
    insert_token_price(&mut pricing, "claude-sonnet-4", 3.00, 15.00);
    insert_token_price(&mut pricing, "claude-haiku-4-5", 1.00, 5.00);
    insert_token_price(&mut pricing, "claude-3-5-haiku", 0.80, 4.00);
    insert_token_price(&mut pricing, "claude-3-5-haiku-latest", 0.80, 4.00);
    pricing
}

fn insert_token_price(
    pricing: &mut BTreeMap<ModelId, AnthropicTokenPricing>,
    model: &str,
    input_usd_per_million_tokens: f64,
    output_usd_per_million_tokens: f64,
) {
    pricing.insert(
        ModelId::new(model),
        AnthropicTokenPricing {
            input_usd_per_million_tokens,
            cache_creation_5m_usd_per_million_tokens: None,
            cache_creation_1h_usd_per_million_tokens: None,
            cache_read_usd_per_million_tokens: None,
            output_usd_per_million_tokens,
        },
    );
}

fn usd_ticks_for_tokens(tokens: u64, usd_per_million_tokens: f64) -> u64 {
    if tokens == 0 || !usd_per_million_tokens.is_finite() || usd_per_million_tokens <= 0.0 {
        return 0;
    }
    let ticks = ((tokens as f64) * usd_per_million_tokens * USD_TICKS_PER_DOLLAR
        / TOKENS_PER_MILLION)
        .ceil();
    if ticks >= u64::MAX as f64 {
        u64::MAX
    } else {
        ticks as u64
    }
}

fn cost_from_ticks(ticks: u64) -> Option<CostAmount> {
    (ticks > 0).then(|| CostAmount {
        amount: ticks.to_string(),
        unit: "usd_ticks".to_string(),
        estimated: true,
    })
}

fn pricing_for_model<'a>(
    pricing: &'a BTreeMap<ModelId, AnthropicTokenPricing>,
    model: &ModelId,
) -> Option<&'a AnthropicTokenPricing> {
    pricing.get(model).or_else(|| {
        strip_compact_date_suffix(model.as_str()).and_then(|base| pricing.get(&ModelId::new(base)))
    })
}

fn strip_compact_date_suffix(model: &str) -> Option<&str> {
    const SUFFIX_LEN: usize = "-YYYYMMDD".len();
    let suffix_start = model.len().checked_sub(SUFFIX_LEN)?;
    if !model.is_char_boundary(suffix_start) {
        return None;
    }
    let suffix = model.as_bytes().get(suffix_start..)?;
    let dated = suffix[0] == b'-' && suffix[1..].iter().all(u8::is_ascii_digit);
    dated.then_some(&model[..suffix_start])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn estimates_text_tokens_with_prompt_cache_rates() {
        let pricing = AnthropicPricing::default();

        let cost = pricing
            .estimate_token_cost(
                Some(&ModelId::new("claude-sonnet-4-6")),
                AnthropicTokenUsage {
                    input_tokens: 100,
                    cache_creation_5m_input_tokens: 12,
                    cache_creation_1h_input_tokens: 8,
                    cache_read_input_tokens: 40,
                    output_tokens: 10,
                    inference_geo: None,
                },
            )
            .expect("cost estimate");

        assert_eq!(cost.unit, "usd_ticks");
        assert!(cost.estimated);
        assert_eq!(cost.amount, "5550001");
    }

    #[test]
    fn token_overrides_replace_builtin_prices() {
        let mut pricing = AnthropicPricing::default();
        pricing.apply_token_overrides(BTreeMap::from([(
            ModelId::new("claude-sonnet-4-6"),
            AnthropicTokenPricing {
                input_usd_per_million_tokens: 1.00,
                cache_creation_5m_usd_per_million_tokens: Some(1.25),
                cache_creation_1h_usd_per_million_tokens: Some(2.00),
                cache_read_usd_per_million_tokens: Some(0.10),
                output_usd_per_million_tokens: 2.00,
            },
        )]));

        let cost = pricing
            .estimate_token_cost(
                Some(&ModelId::new("claude-sonnet-4-6")),
                AnthropicTokenUsage {
                    input_tokens: 100,
                    cache_creation_5m_input_tokens: 20,
                    cache_creation_1h_input_tokens: 10,
                    cache_read_input_tokens: 50,
                    output_tokens: 10,
                    inference_geo: None,
                },
            )
            .expect("cost estimate");

        assert_eq!(cost.amount, "1700000");
    }

    #[test]
    fn dated_snapshot_models_use_base_model_pricing() {
        let pricing = AnthropicPricing::default();

        let cost = pricing
            .estimate_token_cost(
                Some(&ModelId::new("claude-haiku-4-5-20251001")),
                AnthropicTokenUsage {
                    input_tokens: 100,
                    output_tokens: 20,
                    ..AnthropicTokenUsage::default()
                },
            )
            .expect("cost estimate");

        assert_eq!(cost.amount, "2000000");
    }

    #[test]
    fn us_inference_geo_applies_pricing_multiplier() {
        let pricing = AnthropicPricing::default();

        let cost = pricing
            .estimate_token_cost(
                Some(&ModelId::new("claude-haiku-4-5")),
                AnthropicTokenUsage {
                    input_tokens: 100,
                    output_tokens: 20,
                    inference_geo: Some("us"),
                    ..AnthropicTokenUsage::default()
                },
            )
            .expect("cost estimate");

        assert_eq!(cost.amount, "2200001");
    }
}
