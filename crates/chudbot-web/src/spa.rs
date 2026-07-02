use std::path::Path as FsPath;

use axum::extract::State;
use axum::http::{HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response};
use bytes::Bytes;
use chudbot_api::{BotStorage, Conversation, ConversationId, ConversationLookup};
use uuid::Uuid;

use crate::server::{WebConfig, WebRuntimeTypes, WebState};
use crate::static_files::{
    CACHE_NO_CACHE, StaticFileCache, read_static_file, static_file_response,
};

/// Accent color picked up by link-preview embeds; matches `--accent` in
/// `frontend/src/styles/main.scss`.
const EMBED_THEME_COLOR: &str = "#5b6cff";

#[tracing::instrument(name = "web.spa_index", skip_all, fields(path = %uri.path()))]
pub(crate) async fn spa_index<R>(State(state): State<WebState<R>>, uri: axum::http::Uri) -> Response
where
    R: WebRuntimeTypes,
{
    let preview_meta = match conversation_path_id(uri.path()) {
        Some(id) => conversation_preview_meta(&state, id).await,
        None => None,
    };
    render_spa(
        &state.static_files,
        &state.config.frontend_dir,
        uri.path(),
        preview_meta.as_deref(),
    )
    .await
}

#[tracing::instrument(
    name = "web.render_spa",
    skip_all,
    fields(frontend_dir = %frontend_dir.display(), request_path = %request_path)
)]
async fn render_spa(
    static_files: &StaticFileCache,
    frontend_dir: &FsPath,
    request_path: &str,
    head_meta: Option<&str>,
) -> Response {
    let last_segment = request_path.rsplit('/').next().unwrap_or("");
    if last_segment.contains('.') {
        tracing::debug!("asset-looking SPA fallback path not found");
        return (StatusCode::NOT_FOUND, "not found").into_response();
    }
    let index_path = frontend_dir.join("index.html");
    let index = if cache_spa_index() {
        static_files.load(index_path.clone()).await
    } else {
        read_static_file(index_path.clone()).await
    };
    match index {
        Some(bytes) => {
            let bytes = match head_meta {
                Some(meta) => inject_head_meta(&bytes, meta),
                None => bytes,
            };
            tracing::debug!(bytes = bytes.len(), "serving SPA index");
            let mut response = static_file_response(&index_path, bytes, CACHE_NO_CACHE);
            response.headers_mut().insert(
                header::CONTENT_TYPE,
                HeaderValue::from_static("text/html; charset=utf-8"),
            );
            response
        }
        None => {
            tracing::warn!("SPA index is missing or unreadable");
            (
                StatusCode::NOT_FOUND,
                "frontend not built (index.html missing)",
            )
                .into_response()
        }
    }
}

/// Splice extra `<head>` markup in just before `</head>`. Serves the index
/// unmodified when the marker is missing.
fn inject_head_meta(index: &Bytes, meta: &str) -> Bytes {
    let needle = b"</head>";
    match index
        .windows(needle.len())
        .position(|window| window == needle)
    {
        Some(pos) => {
            let mut out = Vec::with_capacity(index.len() + meta.len());
            out.extend_from_slice(&index[..pos]);
            out.extend_from_slice(meta.as_bytes());
            out.extend_from_slice(&index[pos..]);
            Bytes::from(out)
        }
        None => {
            tracing::warn!("SPA index has no </head>; skipping link-preview metadata");
            index.clone()
        }
    }
}

fn cache_spa_index() -> bool {
    !cfg!(debug_assertions)
}

/// Extract the conversation id from `/c/<uuid>` viewer paths.
fn conversation_path_id(path: &str) -> Option<Uuid> {
    let id = path.strip_prefix("/c/")?.trim_end_matches('/');
    Uuid::try_parse(id).ok()
}

/// Load conversation metadata and render the `<head>` OpenGraph tags that
/// Discord-style link unfurlers read. Returns `None` (plain SPA index) when
/// the conversation is unknown or storage fails.
#[tracing::instrument(
    name = "web.conversation_preview_meta",
    skip_all,
    fields(conversation = %id)
)]
async fn conversation_preview_meta<R>(state: &WebState<R>, id: Uuid) -> Option<String>
where
    R: WebRuntimeTypes,
{
    let snapshot = state
        .storage
        .load_conversation(ConversationLookup::Id {
            id: ConversationId(id),
        })
        .await
        .unwrap_or_else(|error| {
            tracing::warn!(error = %error, "failed to load conversation for link preview");
            None
        })?;
    tracing::debug!(
        titled = snapshot.conversation.title.is_some(),
        turns = snapshot.turns.len(),
        "rendering link-preview metadata"
    );
    Some(preview_meta_html(
        &state.config,
        &snapshot.conversation,
        snapshot.turns.len(),
    ))
}

fn preview_meta_html(config: &WebConfig, conversation: &Conversation, turns: usize) -> String {
    let title = conversation
        .title
        .as_deref()
        .unwrap_or("Untitled conversation");
    let turn_word = if turns == 1 { "turn" } else { "turns" };
    let description = format!(
        "{turns} {turn_word} · started {}",
        conversation.created_at.date()
    );
    let mut meta = String::from("\n");
    push_meta(
        &mut meta,
        "property",
        "og:site_name",
        site_name(&config.title_prefix),
    );
    push_meta(&mut meta, "property", "og:type", "website");
    push_meta(&mut meta, "property", "og:title", title);
    push_meta(&mut meta, "property", "og:description", &description);
    if let Some(base) = trimmed_public_base_url(config) {
        push_meta(
            &mut meta,
            "property",
            "og:url",
            &format!("{base}/c/{}", conversation.id),
        );
        if config.og_image_path.is_some() {
            push_meta(
                &mut meta,
                "property",
                "og:image",
                &format!("{base}/og-image"),
            );
        }
    }
    push_meta(&mut meta, "name", "theme-color", EMBED_THEME_COLOR);
    meta
}

/// Derive the embed site name from the tab-title prefix, e.g.
/// "Chudbot QA - " -> "Chudbot QA".
fn site_name(title_prefix: &str) -> &str {
    let name = title_prefix
        .trim()
        .trim_end_matches(['-', '|', ':', '·', '—', '–'])
        .trim_end();
    if name.is_empty() { "Chudbot" } else { name }
}

fn trimmed_public_base_url(config: &WebConfig) -> Option<&str> {
    let base = config
        .public_base_url
        .as_deref()?
        .trim()
        .trim_end_matches('/');
    (!base.is_empty()).then_some(base)
}

fn push_meta(out: &mut String, key_attr: &str, key: &str, content: &str) {
    out.push_str("<meta ");
    out.push_str(key_attr);
    out.push_str("=\"");
    out.push_str(key);
    out.push_str("\" content=\"");
    out.push_str(&escape_attr(content));
    out.push_str("\">\n");
}

/// Minimal HTML attribute escaping for injected meta content.
fn escape_attr(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len());
    for ch in value.chars() {
        match ch {
            '&' => escaped.push_str("&amp;"),
            '<' => escaped.push_str("&lt;"),
            '>' => escaped.push_str("&gt;"),
            '"' => escaped.push_str("&quot;"),
            '\'' => escaped.push_str("&#39;"),
            _ => escaped.push(ch),
        }
    }
    escaped
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    use axum::body::to_bytes;
    use chudbot_api::{ChannelRef, MessageRef, UserRef};
    use test_case::test_case;

    async fn temp_dir() -> PathBuf {
        let path = std::env::temp_dir().join(format!("chudbot-web-{}", Uuid::new_v4()));
        tokio::fs::create_dir_all(&path)
            .await
            .expect("create temp dir");
        path
    }

    async fn response_bytes(response: Response) -> Bytes {
        to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("read response body")
    }

    fn test_web_config() -> WebConfig {
        WebConfig {
            title_prefix: "Chudbot QA - ".to_string(),
            version: "test".to_string(),
            frontend_dir: PathBuf::from("frontend-build"),
            favicon_path: None,
            public_base_url: Some("https://chudbot.example.com/".to_string()),
            og_image_path: Some(PathBuf::from("chudbot.png")),
            trust_forwarded_for: true,
        }
    }

    fn test_conversation(title: Option<&str>) -> Conversation {
        let conversation_id = Uuid::try_parse("0626e4d2-d41f-44ea-aa7f-45489b48955c").unwrap();
        Conversation {
            id: ConversationId(conversation_id),
            created_at: time::OffsetDateTime::UNIX_EPOCH,
            channel: ChannelRef {
                platform: "discord".into(),
                guild_id: Some("1".into()),
                channel_id: "2".into(),
            },
            created_by: UserRef {
                platform: "discord".into(),
                guild_id: Some("1".into()),
                user_id: "3".into(),
            },
            root_message: MessageRef {
                platform: "discord".into(),
                guild_id: Some("1".into()),
                channel_id: "2".into(),
                message_id: "4".into(),
            },
            initial_model: "grok-4".into(),
            agent_name: "chud".to_string(),
            provider: "xai".into(),
            system_instructions: String::new(),
            title: title.map(str::to_string),
            stopped_at: None,
            stopped_by: None,
        }
    }

    #[test_case("/c/0626e4d2-d41f-44ea-aa7f-45489b48955c", true ; "conversation path")]
    #[test_case("/c/0626e4d2-d41f-44ea-aa7f-45489b48955c/", true ; "trailing slash")]
    #[test_case("/c/not-a-uuid", false ; "invalid uuid")]
    #[test_case("/c/", false ; "missing id")]
    #[test_case("/", false ; "root")]
    #[test_case("/c/0626e4d2-d41f-44ea-aa7f-45489b48955c/extra", false ; "nested path")]
    fn conversation_path_id_matches_viewer_paths(path: &str, expected: bool) {
        assert_eq!(conversation_path_id(path).is_some(), expected);
    }

    #[test_case("Chudbot QA - ", "Chudbot QA" ; "dash separator")]
    #[test_case("Chudbot | ", "Chudbot" ; "pipe separator")]
    #[test_case("Chudbot", "Chudbot" ; "no separator")]
    #[test_case("   ", "Chudbot" ; "blank prefix falls back")]
    fn site_name_strips_title_separators(prefix: &str, expected: &str) {
        assert_eq!(site_name(prefix), expected);
    }

    #[test]
    fn preview_meta_escapes_title_and_builds_absolute_urls() {
        let conversation = test_conversation(Some(r#"Tom & "Jerry" <3"#));

        let meta = preview_meta_html(&test_web_config(), &conversation, 3);

        assert!(meta.contains(r#"<meta property="og:site_name" content="Chudbot QA">"#));
        assert!(
            meta.contains(
                r#"<meta property="og:title" content="Tom &amp; &quot;Jerry&quot; &lt;3">"#
            )
        );
        assert!(meta.contains(
            r#"<meta property="og:description" content="3 turns · started 1970-01-01">"#
        ));
        assert!(meta.contains(
            r#"<meta property="og:url" content="https://chudbot.example.com/c/0626e4d2-d41f-44ea-aa7f-45489b48955c">"#
        ));
        assert!(
            meta.contains(
                r#"<meta property="og:image" content="https://chudbot.example.com/og-image">"#
            )
        );
        assert!(meta.contains(r##"<meta name="theme-color" content="#5b6cff">"##));
    }

    #[test]
    fn preview_meta_omits_urls_without_public_base() {
        let config = WebConfig {
            public_base_url: None,
            ..test_web_config()
        };
        let conversation = test_conversation(None);

        let meta = preview_meta_html(&config, &conversation, 1);

        assert!(meta.contains(r#"<meta property="og:title" content="Untitled conversation">"#));
        assert!(meta.contains(r#"content="1 turn · started 1970-01-01""#));
        assert!(!meta.contains("og:url"));
        assert!(!meta.contains("og:image"));
    }

    #[test]
    fn preview_meta_omits_image_without_configured_path() {
        let config = WebConfig {
            og_image_path: None,
            ..test_web_config()
        };

        let meta = preview_meta_html(&config, &test_conversation(None), 1);

        assert!(meta.contains("og:url"));
        assert!(!meta.contains("og:image"));
    }

    #[test]
    fn inject_head_meta_inserts_before_head_close() {
        let index = Bytes::from_static(b"<html><head><title>x</title></head><body></body></html>");

        let injected = inject_head_meta(&index, "<meta name=\"a\" content=\"b\">\n");

        let html = String::from_utf8(injected.to_vec()).expect("utf8 index");
        assert!(html.contains("<meta name=\"a\" content=\"b\">\n</head>"));
        assert!(html.ends_with("</html>"));
    }

    #[test]
    fn inject_head_meta_keeps_index_without_head() {
        let index = Bytes::from_static(b"no head here");

        assert_eq!(inject_head_meta(&index, "<meta>"), index);
    }

    #[cfg(debug_assertions)]
    #[tokio::test]
    async fn render_spa_reads_index_from_disk_in_debug_builds() {
        let dir = temp_dir().await;
        let path = dir.join("index.html");
        tokio::fs::write(&path, "first")
            .await
            .expect("write first index");

        let cache = StaticFileCache::new();
        let first = render_spa(&cache, &dir, "/c/test", None).await;
        tokio::fs::write(&path, "second")
            .await
            .expect("write second index");
        let second = render_spa(&cache, &dir, "/c/test", None).await;

        assert_eq!(&response_bytes(first).await[..], b"first");
        assert_eq!(&response_bytes(second).await[..], b"second");

        tokio::fs::remove_dir_all(dir)
            .await
            .expect("remove temp dir");
    }
}
