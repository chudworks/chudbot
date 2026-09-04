use std::hash::{Hash, Hasher};
use std::path::{Component, Path, PathBuf};
use std::time::Duration as StdDuration;

use axum::extract::ws::WebSocketUpgrade;
use axum::extract::{FromRequestParts, Request, State};
use axum::http::{HeaderMap, HeaderValue, Method, StatusCode, Uri, header};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use chudbot_api::vibe::{NewVibeOauthState, NewVibeSession, VibeIdentityGuild, VibeIdentitySite};
use chudbot_api::{
    ExternalId, PlatformName, VibeIdentity, VibeIdentityProvider, VibeMembership, VibeSession,
    VibeSite, VibeSiteAccess, VibeSiteStatus, VibeStorage,
};
use chudbot_vibe::{SDK_V1_JAVASCRIPT, SDK_V1_TYPESCRIPT, VibeConfig, VibeDiskStore, VibeNames};
use moka::future::Cache;
use percent_encoding::percent_decode_str;
use sha2::{Digest, Sha256};
use time::{Duration, OffsetDateTime};
use uuid::Uuid;

use crate::middleware;
use crate::server::{VibeWebParts, WebRuntimeTypes, WebState};
use crate::vibe_rooms::{RoomHub, serve_room};

const SESSION_COOKIE: &str = "vibe_session";
const SITE_CACHE_CONTROL: &str = "private, no-store";
#[derive(Debug, Clone, Eq)]
struct MembershipKey {
    platform: PlatformName,
    guild_id: ExternalId,
    user_id: ExternalId,
}

impl PartialEq for MembershipKey {
    fn eq(&self, other: &Self) -> bool {
        self.platform == other.platform
            && self.guild_id == other.guild_id
            && self.user_id == other.user_id
    }
}

impl Hash for MembershipKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.platform.hash(state);
        self.guild_id.hash(state);
        self.user_id.hash(state);
    }
}

#[derive(Debug, Clone)]
pub(crate) struct VibeWebState {
    config: VibeConfig,
    disk: VibeDiskStore,
    member_cache: Cache<MembershipKey, VibeMembership>,
    nonmember_cache: Cache<MembershipKey, ()>,
    rooms: RoomHub,
}

impl VibeWebState {
    pub(crate) fn new(parts: VibeWebParts) -> Self {
        let rooms = RoomHub::new(&parts.config.limits);
        Self {
            config: parts.config,
            disk: parts.disk,
            member_cache: Cache::builder()
                .time_to_live(StdDuration::from_secs(5 * 60))
                .build(),
            nonmember_cache: Cache::builder()
                .time_to_live(StdDuration::from_secs(30))
                .build(),
            rooms,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum VibeHost {
    Apex,
    Source,
    Site(String),
    Reserved,
    NotVibe,
    Invalid,
}

pub(crate) async fn host_router<R>(
    State(state): State<WebState<R>>,
    request: Request,
    next: Next,
) -> Response
where
    R: WebRuntimeTypes,
{
    let Some(vibe) = state.vibe.as_ref() else {
        return next.run(request).await;
    };
    let host = request
        .headers()
        .get(header::HOST)
        .and_then(|value| value.to_str().ok());
    match classify_host(host, &vibe.config.base_domain, &vibe.config.reserved_names) {
        VibeHost::NotVibe => next.run(request).await,
        VibeHost::Invalid | VibeHost::Reserved => StatusCode::NOT_FOUND.into_response(),
        VibeHost::Apex => apex_request(&state, vibe, request).await,
        VibeHost::Source => source_request(&state, vibe, request).await,
        VibeHost::Site(name) => site_request(&state, vibe, &name, request).await,
    }
}

fn classify_host(host: Option<&str>, base_domain: &str, reserved: &[String]) -> VibeHost {
    let Some(host) = host else {
        return VibeHost::NotVibe;
    };
    if host.is_empty()
        || host.ends_with('.')
        || host.contains('/')
        || host.contains(char::is_whitespace)
    {
        return VibeHost::Invalid;
    }
    if host.starts_with('[') {
        return VibeHost::NotVibe;
    }
    let Some(host) = strip_port(host) else {
        return VibeHost::Invalid;
    };
    let host = host.to_ascii_lowercase();
    let base = base_domain.to_ascii_lowercase();
    if host == base {
        return VibeHost::Apex;
    }
    let Some(label) = host.strip_suffix(&format!(".{base}")) else {
        return VibeHost::NotVibe;
    };
    if label.is_empty() || label.contains('.') {
        return VibeHost::Invalid;
    }
    if label == "src" {
        return VibeHost::Source;
    }
    let names = VibeNames::new(reserved.iter().cloned());
    if label != "src" && names.validate(label).is_err() {
        return VibeHost::Reserved;
    }
    VibeHost::Site(label.to_string())
}

fn strip_port(host: &str) -> Option<&str> {
    let Some((name, port)) = host.rsplit_once(':') else {
        return Some(host);
    };
    if !name.is_empty()
        && !name.contains(':')
        && port.parse::<u16>().ok().filter(|port| *port > 0).is_some()
    {
        Some(name)
    } else {
        None
    }
}

async fn apex_request<R>(state: &WebState<R>, vibe: &VibeWebState, request: Request) -> Response
where
    R: WebRuntimeTypes,
{
    match (request.method(), request.uri().path()) {
        (&Method::GET, "/") => html(
            StatusCode::OK,
            "<main><h1>Vibe</h1><p>Small websites made with friends in Discord.</p></main>",
        ),
        (&Method::GET, "/login") => login(state, vibe, request.uri()).await,
        (&Method::GET, "/oauth/callback") => oauth_callback(state, vibe, request.uri()).await,
        (&Method::GET, "/logout") | (&Method::POST, "/logout") => {
            logout(state, vibe, request.headers()).await
        }
        _ => StatusCode::NOT_FOUND.into_response(),
    }
}

async fn login<R>(state: &WebState<R>, vibe: &VibeWebState, uri: &Uri) -> Response
where
    R: WebRuntimeTypes,
{
    let query = query_map(uri);
    let return_url = query
        .get("return")
        .cloned()
        .unwrap_or_else(|| format!("https://{}/", vibe.config.base_domain));
    if !valid_return_url(&return_url, &vibe.config.base_domain) {
        return error_page(StatusCode::BAD_REQUEST, "invalid return URL");
    }
    let state_token = random_token();
    let now = OffsetDateTime::now_utc();
    if let Err(error) = state
        .storage
        .create_oauth_state(NewVibeOauthState {
            state_hash: token_hash(&state_token),
            return_url,
            expires_at: now + Duration::minutes(10),
        })
        .await
    {
        tracing::error!(error=%error, "failed to create Vibe OAuth state");
        return error_page(
            StatusCode::SERVICE_UNAVAILABLE,
            "sign-in is temporarily unavailable",
        );
    }
    let callback = format!("https://{}/oauth/callback", vibe.config.base_domain);
    match state.identity.authorization_url(&state_token, &callback) {
        Ok(location) => redirect(&location),
        Err(error) => {
            tracing::error!(error=%error, "failed to build Discord authorization URL");
            error_page(
                StatusCode::SERVICE_UNAVAILABLE,
                "sign-in is temporarily unavailable",
            )
        }
    }
}

async fn oauth_callback<R>(state: &WebState<R>, vibe: &VibeWebState, uri: &Uri) -> Response
where
    R: WebRuntimeTypes,
{
    let query = query_map(uri);
    let (Some(code), Some(state_token)) = (query.get("code"), query.get("state")) else {
        return error_page(StatusCode::BAD_REQUEST, "missing OAuth response");
    };
    let now = OffsetDateTime::now_utc();
    let oauth_state = match state
        .storage
        .consume_oauth_state(&token_hash(state_token), now)
        .await
    {
        Ok(Some(value)) => value,
        Ok(None) => {
            return error_page(
                StatusCode::BAD_REQUEST,
                "this sign-in link expired or was already used",
            );
        }
        Err(error) => {
            tracing::error!(error=%error,"failed to consume Vibe OAuth state");
            return error_page(
                StatusCode::SERVICE_UNAVAILABLE,
                "sign-in is temporarily unavailable",
            );
        }
    };
    let callback = format!("https://{}/oauth/callback", vibe.config.base_domain);
    let login = match state.identity.exchange_code(code, &callback).await {
        Ok(login) => login,
        Err(error) => {
            tracing::warn!(error=%error,"Discord OAuth exchange failed");
            return error_page(StatusCode::BAD_GATEWAY, "Discord sign-in failed");
        }
    };
    let token = random_token();
    if let Err(error) = state
        .storage
        .create_session(NewVibeSession {
            token_hash: token_hash(&token),
            platform: login.platform,
            user_id: login.user_id,
            expires_at: now + Duration::days(i64::from(vibe.config.auth.session_days)),
        })
        .await
    {
        tracing::error!(error=%error,"failed to create Vibe session");
        return error_page(
            StatusCode::SERVICE_UNAVAILABLE,
            "sign-in is temporarily unavailable",
        );
    }
    let mut response = redirect(&oauth_state.return_url);
    let cookie = session_cookie(
        &vibe.config.base_domain,
        &token,
        vibe.config.auth.session_days,
    );
    response.headers_mut().insert(
        header::SET_COOKIE,
        HeaderValue::from_str(&cookie).expect("generated cookie is valid"),
    );
    response
}

async fn logout<R>(state: &WebState<R>, vibe: &VibeWebState, headers: &HeaderMap) -> Response
where
    R: WebRuntimeTypes,
{
    if let Some(token) = cookie_value(headers, SESSION_COOKIE)
        && let Err(error) = state.storage.revoke_session(&token_hash(token)).await
    {
        tracing::warn!(error=%error,"failed to revoke Vibe session");
    }
    let mut response = redirect(&format!("https://{}/", vibe.config.base_domain));
    let cookie = clear_session_cookie(&vibe.config.base_domain);
    response.headers_mut().insert(
        header::SET_COOKIE,
        HeaderValue::from_str(&cookie).expect("generated cookie is valid"),
    );
    response
}

async fn site_request<R>(
    state: &WebState<R>,
    vibe: &VibeWebState,
    name: &str,
    request: Request,
) -> Response
where
    R: WebRuntimeTypes,
{
    let site = match state.storage.find_site_by_name(name).await {
        Ok(Some(site)) => site,
        Ok(None) => return StatusCode::NOT_FOUND.into_response(),
        Err(error) => {
            tracing::error!(site=name,error=%error,"Vibe site lookup failed");
            return secured(error_page(
                StatusCode::SERVICE_UNAVAILABLE,
                "site unavailable",
            ));
        }
    };
    if !guild_is_allowed(&vibe.config, &site.platform, &site.guild_id) {
        return StatusCode::NOT_FOUND.into_response();
    }
    let membership = match site.access {
        VibeSiteAccess::Public => None,
        VibeSiteAccess::Protected => {
            match authorize(state, vibe, &site, request.headers(), request.uri()).await {
                Ok((_, membership)) => Some(membership),
                Err(response) => {
                    let response = *response;
                    return secured(
                        if site.status == VibeSiteStatus::Archived
                            && response.status() != StatusCode::SERVICE_UNAVAILABLE
                        {
                            StatusCode::NOT_FOUND.into_response()
                        } else {
                            response
                        },
                    );
                }
            }
        }
    };
    if let Some(membership) = membership.as_ref() {
        middleware::record_access_identity(
            &request,
            site.platform.as_str(),
            membership.user_id.as_str(),
        );
    }
    if site.status != VibeSiteStatus::Active {
        return secured(error_page(StatusCode::GONE, "This site is not available."));
    }
    let path = request.uri().path();
    let response = if path.starts_with("/__vibe/") {
        vibe_api(state, vibe, request, site, membership).await
    } else {
        serve_site_file(vibe, &site, path, is_document(request.headers())).await
    };
    secured(response)
}

async fn vibe_api<R>(
    state: &WebState<R>,
    vibe: &VibeWebState,
    request: Request,
    site: VibeSite,
    membership: Option<VibeMembership>,
) -> Response
where
    R: WebRuntimeTypes,
{
    if request.method() == Method::GET
        && let Some(response) = static_vibe_api(request.uri().path())
    {
        return response;
    }
    match (request.method(), request.uri().path()) {
        (&Method::GET, "/__vibe/api/v1/identity") => identity_response(&site, membership.as_ref()),
        (&Method::GET, "/__vibe/api/v1/rooms") => {
            room_upgrade(state, vibe, request, site, membership).await
        }
        _ => StatusCode::NOT_FOUND.into_response(),
    }
}

fn static_vibe_api(path: &str) -> Option<Response> {
    match path {
        "/__vibe/sdk/v1/vibe.js" => Some(
            (
                StatusCode::OK,
                [(header::CONTENT_TYPE, "text/javascript; charset=utf-8")],
                SDK_V1_JAVASCRIPT,
            )
                .into_response(),
        ),
        "/__vibe/sdk/v1/vibe.d.ts" => Some(
            (
                StatusCode::OK,
                [(header::CONTENT_TYPE, "text/plain; charset=utf-8")],
                SDK_V1_TYPESCRIPT,
            )
                .into_response(),
        ),
        _ => None,
    }
}

fn identity_response(site: &VibeSite, membership: Option<&VibeMembership>) -> Response {
    axum::Json(
        membership
            .filter(|_| site.access == VibeSiteAccess::Protected)
            .map(|membership| VibeIdentity {
                id: membership.user_id.as_str().to_string(),
                username: membership.username.clone(),
                display_name: membership.display_name.clone(),
                avatar_url: membership.avatar_url.clone(),
                guild: VibeIdentityGuild {
                    id: membership.guild_id.as_str().to_string(),
                    display_name: membership.guild_display_name.clone(),
                },
                site: VibeIdentitySite {
                    name: site.name.clone(),
                },
            }),
    )
    .into_response()
}

async fn room_upgrade<R>(
    state: &WebState<R>,
    vibe: &VibeWebState,
    request: Request,
    site: VibeSite,
    membership: Option<VibeMembership>,
) -> Response
where
    R: WebRuntimeTypes,
{
    if !room_origin_matches(request.headers(), &site.name, &vibe.config.base_domain) {
        tracing::warn!(site=%site.name, origin=?request.headers().get(header::ORIGIN), "rejected cross-origin Vibe room upgrade");
        return json_error(
            StatusCode::FORBIDDEN,
            "invalid_origin",
            "The WebSocket origin does not match this Vibe site.",
        );
    }
    let (mut parts, _body) = request.into_parts();
    let upgrade = match WebSocketUpgrade::from_request_parts(&mut parts, &()).await {
        Ok(upgrade) => upgrade,
        Err(error) => return error.into_response(),
    };
    let rooms = vibe.rooms.clone();
    let shutdown = state.shutdown_token();
    let max_message_bytes = vibe.config.limits.max_room_message_bytes;
    let max_commands_per_second = vibe.config.limits.max_room_commands_per_second;
    upgrade
        .max_message_size(max_message_bytes)
        .max_frame_size(max_message_bytes)
        .on_upgrade(move |socket| {
            serve_room(
                socket,
                rooms,
                site,
                membership,
                shutdown,
                max_message_bytes,
                max_commands_per_second,
            )
        })
}

fn room_origin_matches(headers: &HeaderMap, site: &str, base_domain: &str) -> bool {
    let Some(origin) = headers
        .get(header::ORIGIN)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| url::Url::parse(value).ok())
    else {
        return false;
    };
    origin.scheme() == "https"
        && origin.username().is_empty()
        && origin.password().is_none()
        && origin.port().is_none()
        && origin.path() == "/"
        && origin.query().is_none()
        && origin.fragment().is_none()
        && origin
            .host_str()
            .is_some_and(|host| host.eq_ignore_ascii_case(&format!("{site}.{base_domain}")))
}

async fn serve_site_file(
    vibe: &VibeWebState,
    site: &VibeSite,
    path: &str,
    document: bool,
) -> Response {
    let Some(revision) = site.active_revision_id else {
        return error_page(StatusCode::SERVICE_UNAVAILABLE, "site artifact missing");
    };
    let root = vibe.disk.artifact_path(site.id, revision);
    let requested = match safe_url_path(path) {
        Some(path) => path,
        None => return StatusCode::NOT_FOUND.into_response(),
    };
    let exact = if requested.as_os_str().is_empty() {
        root.join("index.html")
    } else {
        root.join(&requested)
    };
    if let Ok(bytes) = tokio::fs::read(&exact).await {
        return file_response(&exact, bytes);
    }
    if document && let Ok(bytes) = tokio::fs::read(root.join("index.html")).await {
        return file_response(&root.join("index.html"), bytes);
    }
    StatusCode::NOT_FOUND.into_response()
}

async fn source_request<R>(state: &WebState<R>, vibe: &VibeWebState, request: Request) -> Response
where
    R: WebRuntimeTypes,
{
    if request.method() != Method::GET {
        return StatusCode::METHOD_NOT_ALLOWED.into_response();
    }
    let path = request.uri().path();
    if let Some(asset) = path.strip_prefix("/assets/") {
        let Some(relative) = safe_url_path(asset) else {
            return StatusCode::NOT_FOUND.into_response();
        };
        let file = state.config.frontend_dir.join("assets").join(relative);
        return match tokio::fs::read(&file).await {
            Ok(bytes) => secured(file_response(&file, bytes)),
            Err(_) => StatusCode::NOT_FOUND.into_response(),
        };
    }
    let api = path.strip_prefix("/api/vibe/source/");
    let page = path.strip_prefix("/sites/");
    let Some(name) = api
        .or(page)
        .filter(|name| !name.is_empty() && !name.contains('/'))
    else {
        return html(
            StatusCode::OK,
            "<main><h1>Vibe source</h1><p>Open a source link shared in Discord.</p></main>",
        );
    };
    let site = match state.storage.find_site_by_name(name).await {
        Ok(Some(site)) => site,
        _ => return StatusCode::NOT_FOUND.into_response(),
    };
    if !guild_is_allowed(&vibe.config, &site.platform, &site.guild_id) {
        return StatusCode::NOT_FOUND.into_response();
    }
    let membership = match authorize(state, vibe, &site, request.headers(), request.uri()).await {
        Ok((_, membership)) => membership,
        Err(response) => return *response,
    };
    middleware::record_access_identity(
        &request,
        site.platform.as_str(),
        membership.user_id.as_str(),
    );
    if site.status != VibeSiteStatus::Active {
        return error_page(StatusCode::GONE, "This site is not available.");
    }
    if api.is_none() {
        return match tokio::fs::read(state.config.frontend_dir.join("index.html")).await {
            Ok(bytes) => secured(file_response(
                &state.config.frontend_dir.join("index.html"),
                bytes,
            )),
            Err(error) => {
                tracing::error!(error=%error,"Vibe source browser bundle missing");
                error_page(
                    StatusCode::SERVICE_UNAVAILABLE,
                    "source browser unavailable",
                )
            }
        };
    }
    let revisions = match state.storage.list_revisions(site.id).await {
        Ok(r) => r,
        Err(error) => {
            tracing::error!(error=%error,"Vibe revisions lookup failed");
            return error_page(StatusCode::SERVICE_UNAVAILABLE, "source unavailable");
        }
    };
    let query = query_map(request.uri());
    if let (Some(from), Some(to)) = (
        query.get("from").and_then(|v| v.parse::<i32>().ok()),
        query.get("to").and_then(|v| v.parse::<i32>().ok()),
    ) {
        let pair = (
            revisions.iter().find(|r| r.ordinal == from),
            revisions.iter().find(|r| r.ordinal == to),
        );
        if let (Some(from), Some(to)) = pair {
            return match vibe
                .disk
                .diff(site.id, &from.commit_oid, &to.commit_oid)
                .await
            {
                Ok(diff) => secured(axum::Json(serde_json::json!({"site":{"name":site.name,"url":format!("https://{}.{}",site.name,vibe.config.base_domain)},"revisions":revisions.iter().map(revision_json).collect::<Vec<_>>(),"view":{"kind":"diff","from":from.ordinal,"to":to.ordinal,"text":diff}})).into_response()),
                Err(_) => StatusCode::NOT_FOUND.into_response(),
            };
        }
    }
    let selected = query
        .get("revision")
        .and_then(|v| v.parse::<i32>().ok())
        .and_then(|ordinal| revisions.iter().find(|r| r.ordinal == ordinal))
        .or_else(|| revisions.first());
    let Some(selected) = selected else {
        return error_page(StatusCode::SERVICE_UNAVAILABLE, "source history is empty");
    };
    if let Some(path) = query.get("path") {
        let path = Path::new(path);
        return match vibe
            .disk
            .read_file(site.id, &selected.commit_oid, path)
            .await
        {
            Ok(content) => secured(axum::Json(serde_json::json!({"site":{"name":site.name,"url":format!("https://{}.{}",site.name,vibe.config.base_domain)},"revisions":revisions.iter().map(revision_json).collect::<Vec<_>>(),"view":{"kind":"file","revision":selected.ordinal,"path":path.display().to_string(),"text":content}})).into_response()),
            Err(_) => StatusCode::NOT_FOUND.into_response(),
        };
    }
    let files = vibe
        .disk
        .list_files(site.id, &selected.commit_oid)
        .await
        .unwrap_or_default();
    secured(axum::Json(serde_json::json!({"site":{"name":site.name,"url":format!("https://{}.{}",site.name,vibe.config.base_domain)},"revisions":revisions.iter().map(revision_json).collect::<Vec<_>>(),"view":{"kind":"tree","revision":selected.ordinal,"files":files}})).into_response())
}

fn revision_json(revision: &chudbot_api::VibeRevision) -> serde_json::Value {
    serde_json::json!({"ordinal":revision.ordinal,"message":revision.message,"authorId":revision.actor_user_id,"createdAt":revision.created_at})
}

fn guild_is_allowed(config: &VibeConfig, platform: &PlatformName, guild: &ExternalId) -> bool {
    config.access.allowed_guilds.is_empty()
        || config
            .access
            .allowed_guilds
            .iter()
            .any(|allowed| &allowed.platform == platform && &allowed.guild_id == guild)
}

async fn authorize<R>(
    state: &WebState<R>,
    vibe: &VibeWebState,
    site: &VibeSite,
    headers: &HeaderMap,
    uri: &Uri,
) -> Result<(VibeSession, VibeMembership), Box<Response>>
where
    R: WebRuntimeTypes,
{
    let Some(token) = cookie_value(headers, SESSION_COOKIE) else {
        return Err(Box::new(unauthenticated(
            headers,
            uri,
            &vibe.config.base_domain,
        )));
    };
    let session = match state
        .storage
        .find_session(&token_hash(token), OffsetDateTime::now_utc())
        .await
    {
        Ok(Some(s)) => s,
        Ok(None) => {
            return Err(Box::new(unauthenticated(
                headers,
                uri,
                &vibe.config.base_domain,
            )));
        }
        Err(error) => {
            tracing::error!(error=%error,"Vibe session lookup failed");
            return Err(Box::new(error_page(
                StatusCode::SERVICE_UNAVAILABLE,
                "authorization unavailable",
            )));
        }
    };
    if session.platform != site.platform {
        return Err(Box::new(StatusCode::NOT_FOUND.into_response()));
    }
    let key = MembershipKey {
        platform: site.platform.clone(),
        guild_id: site.guild_id.clone(),
        user_id: session.user_id.clone(),
    };
    if let Some(member) = vibe.member_cache.get(&key).await {
        return Ok((session, member));
    }
    if vibe.nonmember_cache.contains_key(&key) {
        return Err(Box::new(StatusCode::NOT_FOUND.into_response()));
    }
    match state
        .identity
        .guild_membership(&key.platform, &key.guild_id, &key.user_id)
        .await
    {
        Ok(Some(member)) => {
            vibe.nonmember_cache.invalidate(&key).await;
            vibe.member_cache.insert(key, member.clone()).await;
            Ok((session, member))
        }
        Ok(None) => {
            vibe.nonmember_cache.insert(key, ()).await;
            Err(Box::new(StatusCode::NOT_FOUND.into_response()))
        }
        Err(error) => {
            tracing::warn!(error=%error,"Discord membership check failed");
            Err(Box::new(error_page(
                StatusCode::SERVICE_UNAVAILABLE,
                "Discord membership could not be checked",
            )))
        }
    }
}

fn unauthenticated(headers: &HeaderMap, uri: &Uri, base: &str) -> Response {
    if is_document(headers) {
        let host = headers
            .get(header::HOST)
            .and_then(|v| v.to_str().ok())
            .unwrap_or(base);
        let return_url = format!("https://{host}{uri}");
        redirect(&format!(
            "https://{base}/login?return={}",
            percent_encoding::utf8_percent_encode(&return_url, percent_encoding::NON_ALPHANUMERIC)
        ))
    } else {
        json_error(
            StatusCode::UNAUTHORIZED,
            "not_authenticated",
            "Sign in with Discord to continue.",
        )
    }
}
fn is_document(headers: &HeaderMap) -> bool {
    headers.get("sec-fetch-dest").and_then(|v| v.to_str().ok()) == Some("document")
        || headers
            .get(header::ACCEPT)
            .and_then(|v| v.to_str().ok())
            .is_some_and(|v| v.split(',').any(|v| v.trim().starts_with("text/html")))
}
fn cookie_value<'a>(headers: &'a HeaderMap, name: &str) -> Option<&'a str> {
    headers
        .get(header::COOKIE)?
        .to_str()
        .ok()?
        .split(';')
        .filter_map(|p| p.trim().split_once('='))
        .find_map(|(key, value)| {
            (key == name && !value.is_empty() && value.len() <= 128).then_some(value)
        })
}
fn random_token() -> String {
    format!("{}{}", Uuid::new_v4().simple(), Uuid::new_v4().simple())
}
fn session_cookie(base: &str, token: &str, days: u16) -> String {
    format!(
        "{SESSION_COOKIE}={token}; Domain=.{base}; Secure; HttpOnly; SameSite=Lax; Path=/; Max-Age={}",
        u64::from(days) * 86_400
    )
}
fn clear_session_cookie(base: &str) -> String {
    format!("{SESSION_COOKIE}=; Domain=.{base}; Secure; HttpOnly; SameSite=Lax; Path=/; Max-Age=0")
}
fn token_hash(token: &str) -> Vec<u8> {
    Sha256::digest(token.as_bytes()).to_vec()
}
fn query_map(uri: &Uri) -> std::collections::BTreeMap<String, String> {
    url::form_urlencoded::parse(uri.query().unwrap_or_default().as_bytes())
        .into_owned()
        .collect()
}
fn valid_return_url(value: &str, base: &str) -> bool {
    let Ok(url) = url::Url::parse(value) else {
        return false;
    };
    if url.scheme() != "https"
        || !url.username().is_empty()
        || url.password().is_some()
        || url.port().is_some()
    {
        return false;
    }
    let Some(host) = url.host_str() else {
        return false;
    };
    host == base
        || host
            .strip_suffix(&format!(".{base}"))
            .is_some_and(|label| !label.is_empty() && !label.contains('.'))
}
fn redirect(location: &str) -> Response {
    let mut response = StatusCode::SEE_OTHER.into_response();
    match HeaderValue::from_str(location) {
        Ok(value) => {
            response.headers_mut().insert(header::LOCATION, value);
            response
        }
        Err(_) => error_page(StatusCode::BAD_REQUEST, "invalid redirect"),
    }
}
fn safe_url_path(path: &str) -> Option<PathBuf> {
    let decoded = percent_decode_str(path.trim_start_matches('/'))
        .decode_utf8()
        .ok()?;
    if decoded.contains('\\') {
        return None;
    }
    let path = Path::new(decoded.as_ref());
    let mut result = PathBuf::new();
    for part in path.components() {
        match part {
            Component::Normal(v) => result.push(v),
            Component::CurDir => {}
            _ => return None,
        }
    }
    Some(result)
}
fn file_response(path: &Path, bytes: Vec<u8>) -> Response {
    let mut response = (StatusCode::OK, bytes).into_response();
    response.headers_mut().insert(
        header::CONTENT_TYPE,
        HeaderValue::from_str(
            mime_guess::from_path(path)
                .first_or_octet_stream()
                .essence_str(),
        )
        .expect("MIME is valid"),
    );
    response
}
fn secured(mut response: Response) -> Response {
    let headers = response.headers_mut();
    headers.insert(
        header::CACHE_CONTROL,
        HeaderValue::from_static(SITE_CACHE_CONTROL),
    );
    headers.insert(
        header::X_CONTENT_TYPE_OPTIONS,
        HeaderValue::from_static("nosniff"),
    );
    headers.insert(
        header::REFERRER_POLICY,
        HeaderValue::from_static("no-referrer"),
    );
    headers.insert(header::X_FRAME_OPTIONS, HeaderValue::from_static("DENY"));
    response
}
fn json_error(status: StatusCode, code: &str, message: &str) -> Response {
    (
        status,
        [(header::CONTENT_TYPE, "application/json")],
        serde_json::json!({"error":{"code":code,"message":message}}).to_string(),
    )
        .into_response()
}
fn html(status: StatusCode, body: &str) -> Response {
    (status,[(header::CONTENT_TYPE,"text/html; charset=utf-8"),(header::CACHE_CONTROL,SITE_CACHE_CONTROL)],format!("<!doctype html><html><meta name=viewport content=\"width=device-width\"><title>Vibe</title><body>{body}</body></html>")).into_response()
}
fn error_page(status: StatusCode, message: &str) -> Response {
    html(
        status,
        &format!("<main><h1>Vibe</h1><p>{}</p></main>", escape_html(message)),
    )
}
fn escape_html(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#39;")
}

#[cfg(test)]
mod tests {
    use super::*;
    use chudbot_api::{VibeRevisionId, VibeSiteAccess, VibeSiteId};
    use chudbot_vibe::config::{
        VibeAccessConfig, VibeAuthConfig, VibeLimitsConfig, VibeSandboxConfig,
    };
    use http_body_util::BodyExt;
    use test_case::test_case;
    #[test_case(Some("vibe.example"), VibeHost::Apex)]
    #[test_case(Some("VIBE.EXAMPLE:1860"), VibeHost::Apex)]
    #[test_case(Some("src.vibe.example"), VibeHost::Source)]
    #[test_case(Some("site.vibe.example"),VibeHost::Site("site".into()))]
    #[test_case(Some("unknown.example"), VibeHost::NotVibe)]
    #[test_case(Some("a.b.vibe.example"), VibeHost::Invalid)]
    #[test_case(Some("vibe.example."), VibeHost::Invalid)]
    #[test_case(Some("www.vibe.example"), VibeHost::Reserved)]
    #[test_case(Some("vibe.example:bad"), VibeHost::Invalid)]
    #[test_case(Some("vibe.example:99999"), VibeHost::Invalid)]
    #[test_case(Some("[::1]:1860"), VibeHost::NotVibe)]
    fn host_routes(host: Option<&str>, expected: VibeHost) {
        assert_eq!(classify_host(host, "vibe.example", &[]), expected);
    }
    #[test_case("https://vibe.example/", true)]
    #[test_case("https://x.vibe.example/a", true)]
    #[test_case("https://a.b.vibe.example/", false)]
    #[test_case("http://vibe.example/", false)]
    #[test_case("https://evil.example/", false)]
    fn return_urls(value: &str, valid: bool) {
        assert_eq!(valid_return_url(value, "vibe.example"), valid);
    }

    #[test_case("https://site.vibe.example", true ; "exact origin")]
    #[test_case("https://SITE.VIBE.EXAMPLE", true ; "origin host case is ignored")]
    #[test_case("https://other.vibe.example", false)]
    #[test_case("https://site.vibe.example.evil.example", false)]
    #[test_case("http://site.vibe.example", false)]
    fn room_origins(value: &str, valid: bool) {
        let mut headers = HeaderMap::new();
        headers.insert(header::ORIGIN, HeaderValue::from_str(value).unwrap());
        assert_eq!(room_origin_matches(&headers, "site", "vibe.example"), valid);
    }

    #[test]
    fn session_cookie_has_exact_security_flags() {
        let cookie = session_cookie("vibe.example", "secret", 7);
        for expected in [
            "Domain=.vibe.example",
            "Secure",
            "HttpOnly",
            "SameSite=Lax",
            "Path=/",
            "Max-Age=604800",
        ] {
            assert!(cookie.contains(expected), "{expected}");
        }
        assert!(clear_session_cookie("vibe.example").ends_with("Max-Age=0"));
    }

    #[test]
    fn secured_site_responses_have_exact_private_headers() {
        let response = secured(StatusCode::OK.into_response());
        assert_eq!(
            response.headers()[header::CACHE_CONTROL],
            "private, no-store"
        );
        assert_eq!(
            response.headers()[header::X_CONTENT_TYPE_OPTIONS],
            "nosniff"
        );
        assert_eq!(response.headers()[header::REFERRER_POLICY], "no-referrer");
        assert_eq!(response.headers()[header::X_FRAME_OPTIONS], "DENY");
    }

    #[test_case("/assets/app.js",Some("assets/app.js");"ordinary file")]
    #[test_case("/route.with.dots",Some("route.with.dots");"dots are safe")]
    #[test_case("/../secret",None;"parent escape")]
    #[test_case("/a%5Cb",None;"encoded backslash")]
    fn safe_site_paths(value: &str, expected: Option<&str>) {
        assert_eq!(safe_url_path(value).as_deref(), expected.map(Path::new));
    }

    fn test_vibe(root: PathBuf) -> VibeWebState {
        VibeWebState::new(VibeWebParts {
            config: VibeConfig {
                enabled: true,
                base_domain: "vibe.example".into(),
                root_dir: root.clone(),
                reserved_names: Vec::new(),
                access: VibeAccessConfig::default(),
                auth: VibeAuthConfig {
                    client_id: "1".into(),
                    client_secret: "secret".into(),
                    session_days: 7,
                },
                sandbox: VibeSandboxConfig::default(),
                limits: VibeLimitsConfig::default(),
            },
            disk: VibeDiskStore::new(root.clone(), root.join("template")),
        })
    }

    #[tokio::test]
    async fn serving_exact_files_document_fallback_and_other_404() {
        let root = std::env::temp_dir().join(format!("vibe-serving-test-{}", Uuid::new_v4()));
        let vibe = test_vibe(root.clone());
        let site_id = VibeSiteId::new();
        let revision_id = VibeRevisionId::new();
        let artifact = vibe.disk.artifact_path(site_id, revision_id);
        tokio::fs::create_dir_all(&artifact).await.unwrap();
        tokio::fs::write(artifact.join("index.html"), "INDEX")
            .await
            .unwrap();
        tokio::fs::write(artifact.join("app.js"), "EXACT")
            .await
            .unwrap();
        let site = VibeSite {
            id: site_id,
            name: "site".into(),
            platform: PlatformName::new("discord"),
            guild_id: ExternalId::new("1"),
            owner_user_id: ExternalId::new("2"),
            description: String::new(),
            status: VibeSiteStatus::Active,
            access: VibeSiteAccess::Protected,
            active_revision_id: Some(revision_id),
            running_job_id: None,
            created_at: OffsetDateTime::UNIX_EPOCH,
            updated_at: OffsetDateTime::UNIX_EPOCH,
        };
        let exact = serve_site_file(&vibe, &site, "/app.js", false)
            .await
            .into_body()
            .collect()
            .await
            .unwrap()
            .to_bytes();
        assert_eq!(exact.as_ref(), b"EXACT");
        let fallback = serve_site_file(&vibe, &site, "/route.with.dots", true)
            .await
            .into_body()
            .collect()
            .await
            .unwrap()
            .to_bytes();
        assert_eq!(fallback.as_ref(), b"INDEX");
        assert_eq!(
            serve_site_file(&vibe, &site, "/missing.png", false)
                .await
                .status(),
            StatusCode::NOT_FOUND
        );
        let _ = tokio::fs::remove_dir_all(root).await;
    }

    #[test]
    fn vibe_api_reserves_sdk_prefix_before_static_files() {
        assert_eq!(
            static_vibe_api("/__vibe/sdk/v1/vibe.js").unwrap().status(),
            StatusCode::OK
        );
        assert!(static_vibe_api("/__vibe/unknown").is_none());
    }

    #[tokio::test]
    async fn public_site_identity_is_json_null() {
        let membership = VibeMembership {
            user_id: ExternalId::new("1"),
            username: "u".into(),
            display_name: "User".into(),
            avatar_url: None,
            guild_id: ExternalId::new("2"),
            guild_display_name: "Guild".into(),
        };
        let site = VibeSite {
            id: VibeSiteId::new(),
            name: "site".into(),
            platform: PlatformName::new("discord"),
            guild_id: ExternalId::new("2"),
            owner_user_id: ExternalId::new("1"),
            description: String::new(),
            status: VibeSiteStatus::Active,
            access: VibeSiteAccess::Public,
            active_revision_id: None,
            running_job_id: None,
            created_at: OffsetDateTime::UNIX_EPOCH,
            updated_at: OffsetDateTime::UNIX_EPOCH,
        };
        let response = identity_response(&site, Some(&membership));
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response
                .into_body()
                .collect()
                .await
                .unwrap()
                .to_bytes()
                .as_ref(),
            b"null"
        );
    }
}
