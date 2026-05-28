//! Generic retry-with-backoff for transient upstream failures.
//!
//! All three external-service clients (LLM, image, video) hit the same
//! kinds of transient failure: a 5xx / 429 from an overloaded backend,
//! or a dropped connection. [`with_retry`] wraps a single HTTP attempt
//! and re-runs it a few times with exponential backoff when the error
//! classifies as retryable, turning a momentary blip into a slightly
//! slower success instead of a dead turn.
//!
//! Errors classify via [`ClassifyError`]; the policy ([`RetryPolicy`])
//! decides how many attempts, how long to back off, and whether ambiguous
//! network failures are safe to retry — they are NOT for a non-idempotent
//! call like "submit a paid video job", where the request might have
//! reached the server before the connection dropped.

use std::fmt::Display;
use std::future::Future;
use std::time::Duration;

/// How an error should be treated by [`with_retry`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorClass {
    /// The server received the request and failed to process it (HTTP
    /// 5xx / 429). Safe to retry: the operation didn't commit, so
    /// re-running it can't double up a side effect.
    ServerTransient,
    /// A network/transport failure (DNS, TCP, TLS, timeout). Retryable
    /// for idempotent reads, but ambiguous for a non-idempotent write —
    /// the request may have reached the server and committed before the
    /// connection dropped. Gated behind [`RetryPolicy::retry_network`].
    Network,
    /// Anything else (4xx other than 429, decode failures, config
    /// errors, …). Never retried — re-running won't change the outcome.
    Permanent,
}

/// Classifies a provider error so [`with_retry`] knows whether to retry it.
pub trait ClassifyError {
    /// Bucket this error into an [`ErrorClass`].
    fn error_class(&self) -> ErrorClass;
}

/// Tunables for [`with_retry`].
#[derive(Debug, Clone, Copy)]
pub struct RetryPolicy {
    /// Total attempts including the first (so `3` = one try + two retries).
    pub max_attempts: u32,
    /// Delay before the first retry; doubles each subsequent retry.
    pub base_delay: Duration,
    /// Upper bound on any single backoff sleep.
    pub max_delay: Duration,
    /// Whether [`ErrorClass::Network`] failures are retried. Leave `true`
    /// for idempotent calls; set `false` for a non-idempotent write where
    /// a retry could duplicate a committed side effect.
    pub retry_network: bool,
}

impl Default for RetryPolicy {
    /// 3 attempts, 500ms base backoff doubling to a 4s cap, network
    /// errors retried. Total added latency on a hard outage is ~1.5s
    /// (500ms + 1s) before giving up.
    fn default() -> Self {
        Self {
            max_attempts: 3,
            base_delay: Duration::from_millis(500),
            max_delay: Duration::from_secs(4),
            retry_network: true,
        }
    }
}

impl RetryPolicy {
    /// Whether an error of the given class should be retried under this policy.
    fn should_retry(&self, class: ErrorClass) -> bool {
        match class {
            ErrorClass::ServerTransient => true,
            ErrorClass::Network => self.retry_network,
            ErrorClass::Permanent => false,
        }
    }

    /// Backoff before the `retry`-th retry (1-based): `base * 2^(retry-1)`,
    /// capped at `max_delay`.
    fn backoff(&self, retry: u32) -> Duration {
        let factor = 2u32.saturating_pow(retry.saturating_sub(1));
        self.base_delay.saturating_mul(factor).min(self.max_delay)
    }
}

/// Run `op`, retrying transient failures per `policy`. `label` tags the
/// per-retry warning log line so a `tail -f` shows which call is flaking.
///
/// Returns the first `Ok`, or the last `Err` once attempts are exhausted
/// or the error is non-retryable.
pub async fn with_retry<F, Fut, T, E>(policy: RetryPolicy, label: &str, mut op: F) -> Result<T, E>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<T, E>>,
    E: ClassifyError + Display,
{
    let mut attempt: u32 = 1;
    loop {
        match op().await {
            Ok(value) => return Ok(value),
            Err(err) => {
                let class = err.error_class();
                if attempt >= policy.max_attempts || !policy.should_retry(class) {
                    return Err(err);
                }
                let delay = policy.backoff(attempt);
                tracing::warn!(
                    label,
                    attempt,
                    max_attempts = policy.max_attempts,
                    delay_ms = delay.as_millis() as u64,
                    error = %err,
                    "transient failure; retrying after backoff"
                );
                tokio::time::sleep(delay).await;
                attempt += 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};

    #[derive(Debug)]
    enum TestErr {
        Transient,
        Net,
        Perm,
    }

    impl Display for TestErr {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "{self:?}")
        }
    }

    impl ClassifyError for TestErr {
        fn error_class(&self) -> ErrorClass {
            match self {
                TestErr::Transient => ErrorClass::ServerTransient,
                TestErr::Net => ErrorClass::Network,
                TestErr::Perm => ErrorClass::Permanent,
            }
        }
    }

    /// Zero-delay policy so tests don't actually sleep.
    fn fast(retry_network: bool) -> RetryPolicy {
        RetryPolicy {
            max_attempts: 3,
            base_delay: Duration::ZERO,
            max_delay: Duration::ZERO,
            retry_network,
        }
    }

    fn block_on<F: Future>(f: F) -> F::Output {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(f)
    }

    #[test]
    fn retries_then_succeeds() {
        let calls = AtomicU32::new(0);
        let result: Result<u32, TestErr> = block_on(with_retry(fast(true), "t", || {
            let n = calls.fetch_add(1, Ordering::SeqCst) + 1;
            async move {
                if n < 3 {
                    Err(TestErr::Transient)
                } else {
                    Ok(n)
                }
            }
        }));
        assert_eq!(result.unwrap(), 3);
        assert_eq!(calls.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn gives_up_after_max_attempts() {
        let calls = AtomicU32::new(0);
        let result: Result<u32, TestErr> = block_on(with_retry(fast(true), "t", || {
            calls.fetch_add(1, Ordering::SeqCst);
            async { Err::<u32, _>(TestErr::Transient) }
        }));
        assert!(matches!(result, Err(TestErr::Transient)));
        assert_eq!(calls.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn permanent_is_not_retried() {
        let calls = AtomicU32::new(0);
        let result: Result<u32, TestErr> = block_on(with_retry(fast(true), "t", || {
            calls.fetch_add(1, Ordering::SeqCst);
            async { Err::<u32, _>(TestErr::Perm) }
        }));
        assert!(matches!(result, Err(TestErr::Perm)));
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn network_error_skipped_when_policy_disallows() {
        let calls = AtomicU32::new(0);
        let result: Result<u32, TestErr> = block_on(with_retry(fast(false), "t", || {
            calls.fetch_add(1, Ordering::SeqCst);
            async { Err::<u32, _>(TestErr::Net) }
        }));
        assert!(matches!(result, Err(TestErr::Net)));
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn network_error_retried_when_policy_allows() {
        let calls = AtomicU32::new(0);
        let result: Result<u32, TestErr> = block_on(with_retry(fast(true), "t", || {
            calls.fetch_add(1, Ordering::SeqCst);
            async { Err::<u32, _>(TestErr::Net) }
        }));
        assert!(matches!(result, Err(TestErr::Net)));
        assert_eq!(calls.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn backoff_grows_and_caps() {
        let p = RetryPolicy {
            max_attempts: 10,
            base_delay: Duration::from_millis(500),
            max_delay: Duration::from_secs(4),
            retry_network: true,
        };
        assert_eq!(p.backoff(1), Duration::from_millis(500));
        assert_eq!(p.backoff(2), Duration::from_secs(1));
        assert_eq!(p.backoff(3), Duration::from_secs(2));
        assert_eq!(p.backoff(4), Duration::from_secs(4));
        // Capped, not unbounded.
        assert_eq!(p.backoff(9), Duration::from_secs(4));
    }
}
