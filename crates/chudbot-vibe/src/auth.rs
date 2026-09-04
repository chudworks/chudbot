//! Shared secret-token helpers for Vibe browser authentication.

use sha2::{Digest, Sha256};
use uuid::Uuid;

/// Lifetime of a single-use login link delivered through a platform DM.
pub const LOGIN_LINK_TTL_MINUTES: i64 = 10;

/// Generate an opaque 256-bit token suitable for a cookie or one-time link.
pub fn random_token() -> String {
    format!("{}{}", Uuid::new_v4().simple(), Uuid::new_v4().simple())
}

/// Hash an opaque token before it crosses the durable-storage boundary.
pub fn token_hash(token: &str) -> Vec<u8> {
    Sha256::digest(token.as_bytes()).to_vec()
}

/// Build the apex-host URL that redeems a single-use direct-login token.
pub fn direct_login_url(base_domain: &str, token: &str) -> String {
    format!("https://{base_domain}/login/direct?token={token}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokens_are_url_safe_and_hash_to_sha256() {
        let token = random_token();

        assert_eq!(token.len(), 64);
        assert!(token.bytes().all(|byte| byte.is_ascii_hexdigit()));
        assert_eq!(token_hash(&token).len(), 32);
        assert_eq!(
            direct_login_url("vibe.example", &token),
            format!("https://vibe.example/login/direct?token={token}")
        );
    }
}
