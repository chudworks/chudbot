use std::net::SocketAddr;
use std::time::Instant;

use axum::extract::{ConnectInfo, Request, State};
use axum::http::{HeaderName, HeaderValue, StatusCode, header};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use http_body::Body as _;
use tower_http::set_header::SetResponseHeaderLayer;

pub(crate) async fn access_log(
    State(trust_forwarded_for): State<bool>,
    req: Request,
    next: Next,
) -> Response {
    let method = req.method().clone();
    let path = req.uri().path().to_owned();
    let remote = client_ip(&req, trust_forwarded_for);
    let user_agent = req
        .headers()
        .get(header::USER_AGENT)
        .and_then(|value| value.to_str().ok())
        .map(short_user_agent)
        .unwrap_or_else(|| "-".to_string());
    let input_bytes = req.body().size_hint().exact().unwrap_or(0);

    let start = Instant::now();
    let response = next.run(req).await;
    let duration = start.elapsed();
    let output_bytes = response.body().size_hint().exact().unwrap_or(0);

    tracing::info!(
        target: "web::access",
        %method,
        path,
        remote,
        status = response.status().as_u16(),
        duration_ms = duration.as_millis(),
        input_bytes,
        output_bytes,
        user_agent,
        "request"
    );

    response
}

pub(crate) async fn block_crawlers(req: Request, next: Next) -> Response {
    if req.uri().path() != "/robots.txt"
        && let Some(ua) = req
            .headers()
            .get(header::USER_AGENT)
            .and_then(|value| value.to_str().ok())
        && is_blocked_crawler(ua)
    {
        return (
            StatusCode::FORBIDDEN,
            [(
                HeaderName::from_static("x-robots-tag"),
                HeaderValue::from_static(X_ROBOTS_TAG),
            )],
            "crawling and indexing of this host are not permitted\n",
        )
            .into_response();
    }
    next.run(req).await
}

pub(crate) fn x_robots_layer() -> SetResponseHeaderLayer<HeaderValue> {
    SetResponseHeaderLayer::overriding(
        HeaderName::from_static("x-robots-tag"),
        HeaderValue::from_static(X_ROBOTS_TAG),
    )
}

pub(crate) fn default_trust_forwarded_for() -> bool {
    true
}

const X_ROBOTS_TAG: &str = "noindex, nofollow, noarchive, nosnippet";

fn client_ip(req: &Request, trust_forwarded_for: bool) -> String {
    if trust_forwarded_for && let Some(ip) = forwarded_client_ip(req) {
        return ip;
    }
    req.extensions()
        .get::<ConnectInfo<SocketAddr>>()
        .map(|ConnectInfo(addr)| addr.ip().to_string())
        .unwrap_or_else(|| "-".to_string())
}

fn forwarded_client_ip(req: &Request) -> Option<String> {
    header_value(req, "cf-connecting-ip")
        .or_else(|| header_value(req, "true-client-ip"))
        .or_else(|| x_forwarded_for(req))
        .or_else(|| forwarded_for(req))
}

fn header_value(req: &Request, name: &'static str) -> Option<String> {
    req.headers()
        .get(HeaderName::from_static(name))
        .and_then(|value| value.to_str().ok())
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
}

fn x_forwarded_for(req: &Request) -> Option<String> {
    req.headers()
        .get(HeaderName::from_static("x-forwarded-for"))
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.split(',').next())
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
}

fn forwarded_for(req: &Request) -> Option<String> {
    let forwarded = req
        .headers()
        .get(HeaderName::from_static("forwarded"))?
        .to_str()
        .ok()?;
    let first = forwarded.split(',').next()?;
    first.split(';').find_map(|field| {
        let (name, value) = field.split_once('=')?;
        if !name.trim().eq_ignore_ascii_case("for") {
            return None;
        }
        let value = value.trim().trim_matches('"').trim();
        (!value.is_empty()).then(|| value.to_string())
    })
}

const UA_MAX_LEN: usize = 48;

fn short_user_agent(ua: &str) -> String {
    let token = ua.split_whitespace().next().unwrap_or(ua);
    token.chars().take(UA_MAX_LEN).collect()
}

const CRAWLER_UA_TOKENS: &[&str] = &[
    // Major search engines.
    "googlebot",
    "google-inspectiontool",
    "storebot-google",
    "bingbot",
    "bingpreview",
    "msnbot",
    "slurp",
    "duckduckbot",
    "duckassistbot",
    "baiduspider",
    "yandex",
    "sogou",
    "exabot",
    "seznambot",
    "petalbot",
    "applebot",
    "ia_archiver",
    "archive.org_bot",
    // AI / answer-engine crawlers.
    "gptbot",
    "oai-searchbot",
    "chatgpt-user",
    "ccbot",
    "claudebot",
    "claude-web",
    "anthropic-ai",
    "perplexitybot",
    "perplexity-user",
    "amazonbot",
    "bytespider",
    "meta-externalagent",
    "cohere-ai",
    "diffbot",
    "google-extended",
    // Aggressive SEO / backlink scrapers.
    "semrushbot",
    "ahrefsbot",
    "mj12bot",
    "dotbot",
    "dataforseobot",
    "blexbot",
];

fn is_blocked_crawler(user_agent: &str) -> bool {
    let ua = user_agent.to_ascii_lowercase();
    CRAWLER_UA_TOKENS.iter().any(|token| ua.contains(token))
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;

    fn request_with_peer(peer: SocketAddr) -> Request {
        let mut req = Request::builder().body(Body::empty()).unwrap();
        req.extensions_mut().insert(ConnectInfo(peer));
        req
    }

    #[test]
    fn client_ip_prefers_cloudflare_header_when_trusted() {
        let mut req = request_with_peer(SocketAddr::from(([10, 0, 0, 2], 443)));
        req.headers_mut().insert(
            HeaderName::from_static("cf-connecting-ip"),
            HeaderValue::from_static("203.0.113.42"),
        );
        req.headers_mut().insert(
            HeaderName::from_static("x-forwarded-for"),
            HeaderValue::from_static("198.51.100.7, 10.0.0.1"),
        );

        assert_eq!(client_ip(&req, true), "203.0.113.42");
    }

    #[test]
    fn client_ip_uses_first_x_forwarded_for_when_trusted() {
        let mut req = request_with_peer(SocketAddr::from(([10, 0, 0, 2], 443)));
        req.headers_mut().insert(
            HeaderName::from_static("x-forwarded-for"),
            HeaderValue::from_static("198.51.100.7, 10.0.0.1"),
        );

        assert_eq!(client_ip(&req, true), "198.51.100.7");
    }

    #[test]
    fn client_ip_uses_standard_forwarded_header_when_trusted() {
        let mut req = request_with_peer(SocketAddr::from(([10, 0, 0, 2], 443)));
        req.headers_mut().insert(
            HeaderName::from_static("forwarded"),
            HeaderValue::from_static("for=198.51.100.8;proto=https"),
        );

        assert_eq!(client_ip(&req, true), "198.51.100.8");
    }

    #[test]
    fn client_ip_ignores_forwarded_headers_when_untrusted() {
        let mut req = request_with_peer(SocketAddr::from(([10, 0, 0, 2], 443)));
        req.headers_mut().insert(
            HeaderName::from_static("cf-connecting-ip"),
            HeaderValue::from_static("203.0.113.42"),
        );

        assert_eq!(client_ip(&req, false), "10.0.0.2");
    }

    #[test]
    fn client_ip_returns_dash_without_peer_or_forwarded_header() {
        let req = Request::builder().body(Body::empty()).unwrap();

        assert_eq!(client_ip(&req, true), "-");
    }

    #[test]
    fn short_user_agent_keeps_first_token() {
        let ua = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:151.0)";

        assert_eq!(short_user_agent(ua), "Mozilla/5.0");
    }

    #[test]
    fn short_user_agent_caps_long_tokens() {
        let ua = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ/1.0";

        assert_eq!(
            short_user_agent(ua),
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUV"
        );
    }
}
