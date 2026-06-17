//! Generic retry-with-backoff for transient upstream failures.
//!
//! Provider clients for model, image, and video APIs all see the same broad
//! failure modes: HTTP 429/5xx when a platform is overloaded, and transport
//! failures when a connection drops. [`with_retry`] wraps one attempt and
//! repeats it with exponential backoff when the provider error classifies as
//! retryable.
//!
//! Network retries are policy-controlled because some calls are not idempotent.
//! In particular, submitting a paid async video job may have reached the server
//! before the connection failed, so that path should retry server-transient
//! statuses but not ambiguous transport errors.
//!
//! The flow has three parts:
//!
//! 1. Provider crates keep their concrete error types and implement
//!    [`ClassifyError`] to expose only the retry-relevant category.
//! 2. Callers choose a [`RetryPolicy`] for the operation they are about to run,
//!    especially whether ambiguous network failures are safe to repeat.
//! 3. [`with_retry`] runs the operation, classifies failures, logs handled
//!    transient errors, sleeps, and returns either the first success or the
//!    final error.
//!
//! This module is deliberately provider-neutral. It does not inspect HTTP
//! status codes, provider error bodies, or transport library types directly;
//! those details stay in the provider crate that owns the request.

use std::fmt::Display;
use std::future::Future;
use std::time::Duration;

/// Coarse retry decision returned by [`ClassifyError`].
///
/// This type is intentionally about retry behavior, not full error semantics.
/// Provider errors should keep their original status, source chain, and message
/// while exposing one of these buckets to the shared retry loop.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorClass {
    /// The upstream returned a response that the caller considers transient,
    /// such as HTTP 429 or 5xx.
    ServerTransient,
    /// DNS, TCP, TLS, timeout, connection reset, or another failure before a
    /// usable upstream response was available.
    ///
    /// A network failure can be ambiguous: the request may have reached the
    /// upstream before the client observed the error. [`RetryPolicy`] decides
    /// whether this class is safe for the current operation.
    Network,
    /// Anything that retrying the same request is not expected to fix, such as
    /// invalid input, authorization failure, or unsupported model selection.
    Permanent,
}

/// Classifies a concrete provider error for shared retry handling.
///
/// Implement this in provider crates rather than teaching `chudbot-api` about
/// provider-specific error enums, HTTP clients, or response formats. The
/// returned class only controls retry behavior; [`with_retry`] returns the
/// original error value unchanged when it gives up.
pub trait ClassifyError {
    /// Bucket this error into an [`ErrorClass`].
    ///
    /// Classifications should be conservative. If a repeated request could
    /// duplicate side effects, classify ambiguous failures as
    /// [`ErrorClass::Network`] and let the caller choose a policy with
    /// [`RetryPolicy::retry_network`] disabled.
    fn error_class(&self) -> ErrorClass;
}

/// Retry limits and backoff timing for [`with_retry`].
///
/// The policy is passed per call site so provider code can distinguish safe
/// read-like requests from operations where a retry might duplicate work
/// upstream.
#[derive(Debug, Clone, Copy)]
pub struct RetryPolicy {
    /// Total attempts including the first.
    ///
    /// [`with_retry`] always performs the initial attempt. A value of `0` or
    /// `1` therefore means "try once and do not retry".
    pub max_attempts: u32,
    /// Delay before the first retry; doubles on each subsequent retry.
    pub base_delay: Duration,
    /// Upper bound for any single backoff sleep.
    pub max_delay: Duration,
    /// Whether [`ErrorClass::Network`] failures are retried.
    pub retry_network: bool,
}

impl Default for RetryPolicy {
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

    /// Backoff before the `retry`-th retry, where retry is 1-based.
    fn backoff(&self, retry: u32) -> Duration {
        // Use saturating arithmetic so very large retry indexes cannot panic or
        // wrap before the policy cap is applied.
        let factor = 2u32.saturating_pow(retry.saturating_sub(1));
        self.base_delay.saturating_mul(factor).min(self.max_delay)
    }
}

/// Run `op`, retrying transient failures per `policy`.
///
/// Returns the first success, or the last error once attempts are exhausted or
/// the error class is not retryable.
///
/// `op` is a factory for one attempt, not a reusable future. It is called again
/// for every retry, so provider code must create a fresh request/future each
/// time and clone any retry-owned inputs before moving them into the attempt.
///
/// `label` is emitted as a structured tracing field on retry warnings. It is
/// not included in the returned error.
pub async fn with_retry<F, Fut, T, E>(policy: RetryPolicy, label: &str, mut op: F) -> Result<T, E>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<T, E>>,
    E: ClassifyError + Display,
{
    let mut attempt: u32 = 1;
    loop {
        // Step 1: run exactly one operation attempt. The closure is invoked
        // inside the loop so retries construct fresh futures and request state.
        match op().await {
            Ok(value) => return Ok(value),
            Err(err) => {
                // Step 2: reduce the concrete provider error to retry policy.
                let class = err.error_class();

                // Step 3: stop on exhausted attempts or non-retryable classes,
                // returning the original provider error unchanged.
                if attempt >= policy.max_attempts || !policy.should_retry(class) {
                    return Err(err);
                }

                // Step 4: log the handled transient failure and wait before the
                // next fresh attempt.
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
    enum TestError {
        ServerTransient,
        Network,
        Permanent,
    }

    impl Display for TestError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "{self:?}")
        }
    }

    impl ClassifyError for TestError {
        fn error_class(&self) -> ErrorClass {
            match self {
                TestError::ServerTransient => ErrorClass::ServerTransient,
                TestError::Network => ErrorClass::Network,
                TestError::Permanent => ErrorClass::Permanent,
            }
        }
    }

    fn fast(retry_network: bool) -> RetryPolicy {
        RetryPolicy {
            max_attempts: 3,
            base_delay: Duration::ZERO,
            max_delay: Duration::ZERO,
            retry_network,
        }
    }

    fn block_on<F>(future: F) -> F::Output
    where
        F: Future,
    {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(future)
    }

    #[test]
    fn retries_then_succeeds() {
        let calls = AtomicU32::new(0);
        let result: Result<u32, TestError> = block_on(with_retry(fast(true), "test", || {
            let attempt = calls.fetch_add(1, Ordering::SeqCst) + 1;
            async move {
                if attempt < 3 {
                    Err(TestError::ServerTransient)
                } else {
                    Ok(attempt)
                }
            }
        }));

        assert_eq!(result.unwrap(), 3);
        assert_eq!(calls.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn gives_up_after_max_attempts() {
        let calls = AtomicU32::new(0);
        let result: Result<u32, TestError> = block_on(with_retry(fast(true), "test", || {
            calls.fetch_add(1, Ordering::SeqCst);
            async { Err(TestError::ServerTransient) }
        }));

        assert!(matches!(result, Err(TestError::ServerTransient)));
        assert_eq!(calls.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn permanent_errors_are_not_retried() {
        let calls = AtomicU32::new(0);
        let result: Result<u32, TestError> = block_on(with_retry(fast(true), "test", || {
            calls.fetch_add(1, Ordering::SeqCst);
            async { Err(TestError::Permanent) }
        }));

        assert!(matches!(result, Err(TestError::Permanent)));
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn network_errors_follow_policy() {
        let calls = AtomicU32::new(0);
        let result: Result<u32, TestError> = block_on(with_retry(fast(false), "test", || {
            calls.fetch_add(1, Ordering::SeqCst);
            async { Err(TestError::Network) }
        }));

        assert!(matches!(result, Err(TestError::Network)));
        assert_eq!(calls.load(Ordering::SeqCst), 1);

        let calls = AtomicU32::new(0);
        let result: Result<u32, TestError> = block_on(with_retry(fast(true), "test", || {
            calls.fetch_add(1, Ordering::SeqCst);
            async { Err(TestError::Network) }
        }));

        assert!(matches!(result, Err(TestError::Network)));
        assert_eq!(calls.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn backoff_grows_and_caps() {
        let policy = RetryPolicy {
            max_attempts: 10,
            base_delay: Duration::from_millis(500),
            max_delay: Duration::from_secs(4),
            retry_network: true,
        };

        assert_eq!(policy.backoff(1), Duration::from_millis(500));
        assert_eq!(policy.backoff(2), Duration::from_secs(1));
        assert_eq!(policy.backoff(3), Duration::from_secs(2));
        assert_eq!(policy.backoff(4), Duration::from_secs(4));
        assert_eq!(policy.backoff(9), Duration::from_secs(4));
    }
}
