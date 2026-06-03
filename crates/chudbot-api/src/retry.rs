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

use std::fmt::Display;
use std::future::Future;
use std::time::Duration;

/// How an error should be treated by [`with_retry`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorClass {
    /// The server received the request and reported a transient status, such as
    /// HTTP 429 or 5xx.
    ServerTransient,
    /// DNS, TCP, TLS, timeout, or another transport failure.
    Network,
    /// Anything that retrying the same request is not expected to fix.
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
    /// Total attempts including the first.
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
        let factor = 2u32.saturating_pow(retry.saturating_sub(1));
        self.base_delay.saturating_mul(factor).min(self.max_delay)
    }
}

/// Run `op`, retrying transient failures per `policy`.
///
/// Returns the first success, or the last error once attempts are exhausted or
/// the error class is not retryable.
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
