use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use axum::extract::ws::{Message, WebSocket};
use chudbot_api::{VibeMembership, VibeSite, VibeSiteId};
use chudbot_vibe::VibeLimitsConfig;
use dashmap::DashMap;
use dashmap::mapref::entry::Entry;
use futures::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

const JOIN_TIMEOUT: Duration = Duration::from_secs(10);
const PING_INTERVAL: Duration = Duration::from_secs(30);
const MAX_NAME_BYTES: usize = 64;

#[derive(Debug, Clone)]
pub(crate) struct RoomHub {
    sites: Arc<DashMap<VibeSiteId, Arc<SiteRooms>>>,
    connections: Arc<AtomicUsize>,
    limits: RoomLimits,
}

#[derive(Debug, Default)]
struct SiteRooms {
    rooms: DashMap<String, Arc<Mutex<RoomState>>>,
    room_count: AtomicUsize,
    connections: AtomicUsize,
}

#[derive(Debug, Default)]
struct RoomState {
    users: HashMap<String, RoomUserEntry>,
    clients: HashMap<Uuid, RoomClient>,
    closed: bool,
}

#[derive(Debug)]
struct RoomUserEntry {
    user: RoomUser,
    connections: HashSet<Uuid>,
}

#[derive(Debug)]
struct RoomClient {
    user_id: String,
    outbound: mpsc::Sender<Arc<str>>,
    cancel: CancellationToken,
}

#[derive(Debug, Clone, Copy)]
struct RoomLimits {
    max_rooms_per_site: usize,
    max_connections_per_room: usize,
    max_connections_per_site: usize,
    max_connections: usize,
    max_user_states: usize,
    max_user_state_bytes: usize,
    outbound_queue: usize,
}

impl From<&VibeLimitsConfig> for RoomLimits {
    fn from(value: &VibeLimitsConfig) -> Self {
        Self {
            max_rooms_per_site: usize::from(value.max_rooms_per_site),
            max_connections_per_room: usize::from(value.max_room_connections),
            max_connections_per_site: usize::from(value.max_room_connections_per_site),
            max_connections: value.max_room_connections_total,
            max_user_states: usize::from(value.max_room_user_states),
            max_user_state_bytes: value.max_room_user_state_bytes,
            outbound_queue: usize::from(value.room_outbound_queue),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct RoomUser {
    id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    username: Option<String>,
    display_name: String,
    avatar_url: Option<String>,
    anonymous: bool,
    states: BTreeMap<String, Value>,
}

#[derive(Debug, Clone)]
enum PendingIdentity {
    Protected(VibeMembership),
    Anonymous,
}

#[derive(Debug)]
struct JoinedRoom {
    connection_id: Uuid,
    user_id: String,
    outbound: mpsc::Receiver<Arc<str>>,
    outbound_sender: mpsc::Sender<Arc<str>>,
    cancel: CancellationToken,
    welcome: String,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
enum ClientMessage {
    #[serde(rename = "join")]
    Join {
        room: String,
        #[serde(rename = "anonymousId")]
        anonymous_id: Option<String>,
    },
    #[serde(rename = "broadcast")]
    Broadcast {
        id: String,
        label: String,
        event: Value,
    },
    #[serde(rename = "set_state")]
    SetState {
        id: String,
        key: String,
        value: Value,
    },
}

#[derive(Debug)]
struct RoomError {
    code: &'static str,
    message: &'static str,
}

#[derive(Debug)]
struct CommandRate {
    window_started: Instant,
    commands: u16,
    maximum: u16,
}

impl CommandRate {
    fn new(maximum: u16) -> Self {
        Self {
            window_started: Instant::now(),
            commands: 0,
            maximum,
        }
    }

    fn admit(&mut self) -> bool {
        if self.window_started.elapsed() >= Duration::from_secs(1) {
            self.window_started = Instant::now();
            self.commands = 0;
        }
        if self.commands >= self.maximum {
            return false;
        }
        self.commands += 1;
        true
    }
}

impl RoomError {
    const fn new(code: &'static str, message: &'static str) -> Self {
        Self { code, message }
    }

    fn message(&self, id: Option<&str>) -> String {
        json!({"type":"error","id":id,"code":self.code,"message":self.message}).to_string()
    }
}

impl RoomHub {
    pub(crate) fn new(limits: &VibeLimitsConfig) -> Self {
        Self {
            sites: Arc::new(DashMap::new()),
            connections: Arc::new(AtomicUsize::new(0)),
            limits: limits.into(),
        }
    }

    fn join(
        &self,
        site: VibeSiteId,
        room_name: &str,
        user: RoomUser,
    ) -> Result<JoinedRoom, RoomError> {
        if !reserve_capacity(&self.connections, self.limits.max_connections) {
            return Err(RoomError::new(
                "rooms_busy",
                "Vibe rooms are at their deployment-wide connection limit.",
            ));
        }
        let site_rooms = match self.sites.entry(site) {
            Entry::Occupied(entry) => {
                if !reserve_capacity(
                    &entry.get().connections,
                    self.limits.max_connections_per_site,
                ) {
                    release_capacity(&self.connections);
                    return Err(RoomError::new(
                        "site_rooms_busy",
                        "This site has reached its Vibe room connection limit.",
                    ));
                }
                Arc::clone(entry.get())
            }
            Entry::Vacant(entry) => {
                let site_rooms = Arc::new(SiteRooms::default());
                let reserved = reserve_capacity(
                    &site_rooms.connections,
                    self.limits.max_connections_per_site,
                );
                debug_assert!(reserved);
                entry.insert(Arc::clone(&site_rooms));
                site_rooms
            }
        };

        loop {
            let room = match site_rooms.rooms.entry(room_name.to_string()) {
                Entry::Occupied(entry) => Arc::clone(entry.get()),
                Entry::Vacant(entry) => {
                    if !reserve_capacity(&site_rooms.room_count, self.limits.max_rooms_per_site) {
                        release_capacity(&site_rooms.connections);
                        release_capacity(&self.connections);
                        return Err(RoomError::new(
                            "too_many_rooms",
                            "This site has reached its active room limit.",
                        ));
                    }
                    let room = Arc::new(Mutex::new(RoomState::default()));
                    entry.insert(Arc::clone(&room));
                    room
                }
            };
            let mut room = lock_room(&room);
            if room.closed {
                drop(room);
                std::thread::yield_now();
                continue;
            }
            if room.clients.len() >= self.limits.max_connections_per_room {
                release_capacity(&site_rooms.connections);
                release_capacity(&self.connections);
                return Err(RoomError::new(
                    "room_full",
                    "This Vibe room has reached its connection limit.",
                ));
            }

            let connection_id = Uuid::new_v4();
            let cancel = CancellationToken::new();
            let (outbound_sender, outbound) = mpsc::channel(self.limits.outbound_queue);
            let is_new_user = !room.users.contains_key(&user.id);
            let user_id = user.id.clone();
            room.users
                .entry(user_id.clone())
                .and_modify(|entry| {
                    entry.connections.insert(connection_id);
                })
                .or_insert_with(|| RoomUserEntry {
                    user: user.clone(),
                    connections: HashSet::from([connection_id]),
                });
            let welcome = json!({
                "type":"welcome",
                "selfId":user_id,
                "users":room.users.values().map(|entry| &entry.user).collect::<Vec<_>>(),
            })
            .to_string();
            if is_new_user {
                send_room(
                    &room,
                    &json!({"type":"user:join","user":user}).to_string(),
                    None,
                );
            }
            room.clients.insert(
                connection_id,
                RoomClient {
                    user_id: user_id.clone(),
                    outbound: outbound_sender.clone(),
                    cancel: cancel.clone(),
                },
            );
            return Ok(JoinedRoom {
                connection_id,
                user_id,
                outbound,
                outbound_sender,
                cancel,
                welcome,
            });
        }
    }

    fn broadcast(
        &self,
        site: VibeSiteId,
        room_name: &str,
        connection: Uuid,
        label: &str,
        event: Value,
    ) -> Result<(), RoomError> {
        validate_event_label(label)?;
        let room = self.room(site, room_name)?;
        let room = lock_room(&room);
        let client = room.clients.get(&connection).ok_or_else(not_joined)?;
        let user = room
            .users
            .get(&client.user_id)
            .map(|entry| entry.user.clone())
            .ok_or_else(not_joined)?;
        let message = json!({"type":"event","label":label,"event":event,"user":user}).to_string();
        tracing::trace!(site=%site, room=room_name, label, recipients=room.clients.len(), "broadcasting Vibe room event");
        send_room(&room, &message, None);
        Ok(())
    }

    fn set_state(
        &self,
        site: VibeSiteId,
        room_name: &str,
        connection: Uuid,
        key: &str,
        value: Value,
    ) -> Result<(), RoomError> {
        validate_token(key, "invalid_state_key", "The user-state key is invalid.")?;
        let room = self.room(site, room_name)?;
        let mut room = lock_room(&room);
        let user_id = room
            .clients
            .get(&connection)
            .map(|client| client.user_id.clone())
            .ok_or_else(not_joined)?;
        let entry = room.users.get_mut(&user_id).ok_or_else(not_joined)?;
        if !entry.user.states.contains_key(key)
            && entry.user.states.len() >= self.limits.max_user_states
        {
            return Err(RoomError::new(
                "too_many_user_states",
                "This user has reached the state-key limit for the room.",
            ));
        }
        let state_bytes = entry
            .user
            .states
            .iter()
            .filter(|(existing, _)| existing.as_str() != key)
            .map(|(existing, value)| existing.len() + value.to_string().len())
            .sum::<usize>()
            .saturating_add(key.len())
            .saturating_add(value.to_string().len());
        if state_bytes > self.limits.max_user_state_bytes {
            return Err(RoomError::new(
                "user_state_too_large",
                "This user's room state exceeds the byte limit.",
            ));
        }
        let before = entry.user.states.insert(key.to_string(), value.clone());
        let user = entry.user.clone();
        let message = json!({
            "type":"user:state",
            "key":key,
            "user":user,
            "beforePresent":before.is_some(),
            "before":before,
            "after":value,
        })
        .to_string();
        tracing::trace!(site=%site, room=room_name, user=%user_id, key, recipients=room.clients.len(), "updating Vibe room user state");
        send_room(&room, &message, None);
        Ok(())
    }

    fn leave(&self, site: VibeSiteId, room_name: &str, connection: Uuid) {
        let Some(site_rooms) = self.sites.get(&site).map(|entry| Arc::clone(entry.value())) else {
            return;
        };
        let Some(room) = site_rooms
            .rooms
            .get(room_name)
            .map(|entry| Arc::clone(entry.value()))
        else {
            return;
        };
        let removed = {
            let mut room = lock_room(&room);
            let Some(client) = room.clients.remove(&connection) else {
                return;
            };
            let quit_user = room.users.get_mut(&client.user_id).and_then(|entry| {
                entry.connections.remove(&connection);
                entry.connections.is_empty().then(|| entry.user.clone())
            });
            if let Some(user) = quit_user {
                room.users.remove(&client.user_id);
                send_room(
                    &room,
                    &json!({"type":"user:quit","user":user}).to_string(),
                    None,
                );
            }
            true
        };
        if removed {
            release_capacity(&site_rooms.connections);
            release_capacity(&self.connections);
            self.remove_empty_room(&site_rooms, room_name, &room);
            self.remove_empty_site(site, &site_rooms);
        }
    }

    fn room(&self, site: VibeSiteId, room_name: &str) -> Result<Arc<Mutex<RoomState>>, RoomError> {
        self.sites
            .get(&site)
            .and_then(|site| {
                site.rooms
                    .get(room_name)
                    .map(|room| Arc::clone(room.value()))
            })
            .ok_or_else(not_joined)
    }

    fn remove_empty_room(
        &self,
        site: &SiteRooms,
        room_name: &str,
        expected: &Arc<Mutex<RoomState>>,
    ) {
        let Entry::Occupied(entry) = site.rooms.entry(room_name.to_string()) else {
            return;
        };
        if !Arc::ptr_eq(entry.get(), expected) {
            return;
        }
        let mut room = lock_room(entry.get());
        if !room.clients.is_empty() {
            return;
        }
        room.closed = true;
        drop(room);
        entry.remove();
        release_capacity(&site.room_count);
    }

    fn remove_empty_site(&self, site: VibeSiteId, expected: &Arc<SiteRooms>) {
        let Entry::Occupied(entry) = self.sites.entry(site) else {
            return;
        };
        if Arc::ptr_eq(entry.get(), expected)
            && expected.connections.load(Ordering::Acquire) == 0
            && expected.room_count.load(Ordering::Acquire) == 0
        {
            entry.remove();
        }
    }
}

fn reserve_capacity(counter: &AtomicUsize, maximum: usize) -> bool {
    counter
        .try_update(Ordering::AcqRel, Ordering::Acquire, |current| {
            (current < maximum).then_some(current + 1)
        })
        .is_ok()
}

fn release_capacity(counter: &AtomicUsize) {
    let previous = counter.fetch_sub(1, Ordering::AcqRel);
    debug_assert!(previous > 0);
}

fn lock_room(room: &Mutex<RoomState>) -> MutexGuard<'_, RoomState> {
    room.lock().unwrap_or_else(|poison| poison.into_inner())
}

fn not_joined() -> RoomError {
    RoomError::new(
        "not_connected",
        "The Vibe room connection is no longer active.",
    )
}

fn send_room(room: &RoomState, message: &str, except: Option<Uuid>) {
    let message = Arc::<str>::from(message);
    for (id, client) in &room.clients {
        if Some(*id) == except {
            continue;
        }
        if client.outbound.try_send(Arc::clone(&message)).is_err() {
            tracing::warn!(connection=%id, user=%client.user_id, "closing slow Vibe room connection");
            client.cancel.cancel();
        }
    }
}

fn validate_token(value: &str, code: &'static str, message: &'static str) -> Result<(), RoomError> {
    if value.is_empty()
        || value.len() > MAX_NAME_BYTES
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b':' | b'_' | b'-'))
    {
        return Err(RoomError::new(code, message));
    }
    Ok(())
}

fn validate_event_label(label: &str) -> Result<(), RoomError> {
    validate_token(label, "invalid_event_label", "The event label is invalid.")?;
    if label.starts_with("user:") {
        return Err(RoomError::new(
            "reserved_event_label",
            "Event labels beginning with user: are reserved.",
        ));
    }
    Ok(())
}

fn protected_user(membership: &VibeMembership) -> RoomUser {
    RoomUser {
        id: membership.user_id.as_str().to_string(),
        username: Some(membership.username.clone()),
        display_name: membership.display_name.clone(),
        avatar_url: membership.avatar_url.clone(),
        anonymous: false,
        states: BTreeMap::new(),
    }
}

fn anonymous_user(id: Option<&str>) -> Result<RoomUser, RoomError> {
    let id = id
        .and_then(|value| Uuid::try_parse(value).ok())
        .filter(|value| value.get_version_num() == 4)
        .ok_or_else(|| {
            RoomError::new(
                "invalid_anonymous_id",
                "The anonymous browser identifier is invalid.",
            )
        })?;
    let suffix = id.simple().to_string();
    Ok(RoomUser {
        id: format!("anonymous:{id}"),
        username: None,
        display_name: format!("Guest {}", &suffix[..6]),
        avatar_url: None,
        anonymous: true,
        states: BTreeMap::new(),
    })
}

pub(crate) async fn serve_room(
    socket: WebSocket,
    hub: RoomHub,
    site: VibeSite,
    membership: Option<VibeMembership>,
    shutdown: CancellationToken,
    max_message_bytes: usize,
    max_commands_per_second: u16,
) {
    let identity = membership.map_or(PendingIdentity::Anonymous, PendingIdentity::Protected);
    let mut socket = socket;
    let join = match receive_join(&mut socket, identity, max_message_bytes).await {
        Ok(value) => value,
        Err(error) => {
            let _ = socket.send(Message::Text(error.message(None).into())).await;
            let _ = socket.close().await;
            tracing::debug!(site=%site.name, code=error.code, "rejected Vibe room join");
            return;
        }
    };
    let room_name = join.0;
    let mut joined = match hub.join(site.id, &room_name, join.1) {
        Ok(joined) => joined,
        Err(error) => {
            let _ = socket.send(Message::Text(error.message(None).into())).await;
            let _ = socket.close().await;
            tracing::warn!(site=%site.name, room=room_name, code=error.code, "Vibe room capacity rejected a join");
            return;
        }
    };
    if socket
        .send(Message::Text(joined.welcome.clone().into()))
        .await
        .is_err()
    {
        hub.leave(site.id, &room_name, joined.connection_id);
        return;
    }
    tracing::info!(site=%site.name, site_id=%site.id, room=room_name, user=%joined.user_id, connection=%joined.connection_id, "Vibe room connection opened");

    let (mut sink, mut stream) = socket.split();
    let mut command_rate = CommandRate::new(max_commands_per_second);
    let mut ping = tokio::time::interval(PING_INTERVAL);
    ping.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    ping.tick().await;
    loop {
        tokio::select! {
            _ = shutdown.cancelled() => break,
            _ = joined.cancel.cancelled() => break,
            outbound = joined.outbound.recv() => {
                let Some(outbound) = outbound else { break };
                if sink.send(Message::Text(outbound.to_string().into())).await.is_err() { break; }
            }
            incoming = stream.next() => {
                let Some(Ok(message)) = incoming else { break };
                match message {
                    Message::Text(text) => {
                        let response = handle_message(
                            &hub,
                            site.id,
                            &room_name,
                            joined.connection_id,
                            text.as_str(),
                            max_message_bytes,
                            &mut command_rate,
                        );
                        if let Some(response) = response
                            && joined.outbound_sender.try_send(response.into()).is_err()
                        {
                            tracing::warn!(site=%site.name, room=room_name, connection=%joined.connection_id, "closing Vibe room connection with a full response queue");
                            break;
                        }
                    }
                    Message::Close(_) => break,
                    Message::Ping(value) => {
                        if sink.send(Message::Pong(value)).await.is_err() { break; }
                    }
                    Message::Pong(_) => {}
                    Message::Binary(_) => {
                        let error = RoomError::new("unsupported_message", "Vibe rooms accept JSON text messages only.");
                        let _ = sink.send(Message::Text(error.message(None).into())).await;
                        break;
                    }
                }
            }
            _ = ping.tick() => {
                if sink.send(Message::Ping(Vec::new().into())).await.is_err() { break; }
            }
        }
    }
    hub.leave(site.id, &room_name, joined.connection_id);
    tracing::info!(site=%site.name, site_id=%site.id, room=room_name, user=%joined.user_id, connection=%joined.connection_id, "Vibe room connection closed");
}

async fn receive_join(
    socket: &mut WebSocket,
    identity: PendingIdentity,
    max_message_bytes: usize,
) -> Result<(String, RoomUser), RoomError> {
    let received = tokio::time::timeout(JOIN_TIMEOUT, socket.recv()).await;
    let message = match received {
        Ok(Some(Ok(Message::Text(text)))) if text.len() <= max_message_bytes => text,
        Ok(Some(Ok(_))) => {
            return Err(RoomError::new(
                "invalid_join",
                "The first room message must be a JSON join request.",
            ));
        }
        Ok(Some(Err(_)) | None) => {
            return Err(RoomError::new(
                "connection_closed",
                "The Vibe room connection closed before joining.",
            ));
        }
        Err(_) => {
            return Err(RoomError::new(
                "join_timeout",
                "The Vibe room join timed out.",
            ));
        }
    };
    let parsed = match serde_json::from_str::<ClientMessage>(message.as_str()) {
        Ok(parsed) => parsed,
        Err(_) => {
            return Err(RoomError::new(
                "invalid_join",
                "The first room message must be a valid join request.",
            ));
        }
    };
    let ClientMessage::Join { room, anonymous_id } = parsed else {
        return Err(RoomError::new(
            "invalid_join",
            "The first room message must be a join request.",
        ));
    };
    validate_token(&room, "invalid_room", "The Vibe room name is invalid.")?;
    let user = match identity {
        PendingIdentity::Protected(membership) => protected_user(&membership),
        PendingIdentity::Anonymous => match anonymous_user(anonymous_id.as_deref()) {
            Ok(user) => user,
            Err(error) => return Err(error),
        },
    };
    Ok((room, user))
}

fn handle_message(
    hub: &RoomHub,
    site: VibeSiteId,
    room: &str,
    connection: Uuid,
    text: &str,
    max_message_bytes: usize,
    command_rate: &mut CommandRate,
) -> Option<String> {
    if text.len() > max_message_bytes {
        return Some(
            RoomError::new("message_too_large", "The Vibe room message is too large.")
                .message(None),
        );
    }
    let parsed = match serde_json::from_str::<ClientMessage>(text) {
        Ok(parsed) => parsed,
        Err(_) => {
            return Some(
                RoomError::new("invalid_message", "The Vibe room message is invalid.")
                    .message(None),
            );
        }
    };
    let id = match &parsed {
        ClientMessage::Join { .. } => None,
        ClientMessage::Broadcast { id, .. } | ClientMessage::SetState { id, .. } => {
            Some(id.as_str())
        }
    };
    if !command_rate.admit() {
        return Some(
            RoomError::new(
                "rate_limited",
                "This connection is sending Vibe room commands too quickly.",
            )
            .message(id),
        );
    }
    let (id, result) = match parsed {
        ClientMessage::Join { .. } => (
            None,
            Err(RoomError::new(
                "already_joined",
                "This connection already joined a Vibe room.",
            )),
        ),
        ClientMessage::Broadcast { id, label, event } => {
            let result = hub.broadcast(site, room, connection, &label, event);
            (Some(id), result)
        }
        ClientMessage::SetState { id, key, value } => {
            let result = hub.set_state(site, room, connection, &key, value);
            (Some(id), result)
        }
    };
    Some(match result {
        Ok(()) => json!({"type":"ack","id":id.expect("commands have ids")}).to_string(),
        Err(error) => error.message(id.as_deref()),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn limits() -> VibeLimitsConfig {
        VibeLimitsConfig::default()
    }

    fn user(id: &str) -> RoomUser {
        RoomUser {
            id: id.to_string(),
            username: None,
            display_name: id.to_string(),
            avatar_url: None,
            anonymous: true,
            states: BTreeMap::new(),
        }
    }

    #[test]
    fn room_membership_state_and_last_connection_quit_are_consistent() {
        let hub = RoomHub::new(&limits());
        let site = VibeSiteId::new();
        let first = hub.join(site, "topic", user("one")).unwrap();
        let second = hub.join(site, "topic", user("one")).unwrap();
        let other = hub.join(site, "topic", user("two")).unwrap();
        hub.set_state(
            site,
            "topic",
            first.connection_id,
            "status",
            json!("online"),
        )
        .unwrap();
        {
            let room = hub.room(site, "topic").unwrap();
            let room = lock_room(&room);
            assert_eq!(room.users.len(), 2);
            assert_eq!(room.users["one"].connections.len(), 2);
            assert_eq!(room.users["one"].user.states["status"], "online");
        }
        hub.leave(site, "topic", first.connection_id);
        let room = hub.room(site, "topic").unwrap();
        assert!(lock_room(&room).users.contains_key("one"));
        hub.leave(site, "topic", second.connection_id);
        let room = hub.room(site, "topic").unwrap();
        assert!(!lock_room(&room).users.contains_key("one"));
        hub.leave(site, "topic", other.connection_id);
        assert!(!hub.sites.contains_key(&site));
    }

    #[test]
    fn room_labels_reserve_user_namespace() {
        assert!(validate_event_label("event").is_ok());
        assert!(validate_event_label("user:join").is_err());
        assert!(validate_event_label("spaces are bad").is_err());
    }

    #[test]
    fn anonymous_ids_are_namespaced_and_stable() {
        let id = Uuid::new_v4();
        let first = anonymous_user(Some(&id.to_string())).unwrap();
        let second = anonymous_user(Some(&id.to_string())).unwrap();
        assert_eq!(first.id, second.id);
        assert!(first.id.starts_with("anonymous:"));
        assert!(anonymous_user(Some("not-a-uuid")).is_err());
    }

    #[test]
    fn rooms_with_the_same_name_are_isolated_between_sites() {
        let hub = RoomHub::new(&limits());
        let first_site = VibeSiteId::new();
        let second_site = VibeSiteId::new();
        let first = hub.join(first_site, "topic", user("one")).unwrap();
        let second = hub.join(second_site, "topic", user("two")).unwrap();
        hub.set_state(
            first_site,
            "topic",
            first.connection_id,
            "site",
            json!("first"),
        )
        .unwrap();
        let first_room = hub.room(first_site, "topic").unwrap();
        let second_room = hub.room(second_site, "topic").unwrap();
        let first_room = lock_room(&first_room);
        assert_eq!(first_room.users["one"].user.states["site"], "first");
        drop(first_room);
        assert!(lock_room(&second_room).users["two"].user.states.is_empty());
        drop(second);
    }

    #[test]
    fn room_capacity_is_released_when_a_connection_leaves() {
        let mut limits = limits();
        limits.max_room_connections = 1;
        let hub = RoomHub::new(&limits);
        let site = VibeSiteId::new();
        let first = hub.join(site, "topic", user("one")).unwrap();
        assert_eq!(
            hub.join(site, "topic", user("two")).unwrap_err().code,
            "room_full"
        );
        hub.leave(site, "topic", first.connection_id);
        assert!(hub.join(site, "topic", user("two")).is_ok());
    }

    #[test]
    fn user_state_and_command_rate_limits_fail_locally() {
        let mut limits = limits();
        limits.max_room_user_state_bytes = 8;
        let hub = RoomHub::new(&limits);
        let site = VibeSiteId::new();
        let joined = hub.join(site, "topic", user("one")).unwrap();
        assert_eq!(
            hub.set_state(
                site,
                "topic",
                joined.connection_id,
                "status",
                json!("far too large"),
            )
            .unwrap_err()
            .code,
            "user_state_too_large"
        );

        let mut rate = CommandRate::new(2);
        assert!(rate.admit());
        assert!(rate.admit());
        assert!(!rate.admit());
    }

    #[test]
    fn primitive_and_object_values_round_trip_through_the_protocol() {
        let hub = RoomHub::new(&limits());
        let site = VibeSiteId::new();
        let mut joined = hub.join(site, "topic", user("one")).unwrap();
        let mut rate = CommandRate::new(10);
        let ack = handle_message(
            &hub,
            site,
            "topic",
            joined.connection_id,
            r#"{"type":"broadcast","id":"1","label":"event","event":"abc"}"#,
            32 * 1024,
            &mut rate,
        )
        .unwrap();
        assert_eq!(serde_json::from_str::<Value>(&ack).unwrap()["type"], "ack");
        let event = joined.outbound.try_recv().unwrap();
        assert_eq!(
            serde_json::from_str::<Value>(&event).unwrap()["event"],
            "abc"
        );

        let ack = handle_message(
            &hub,
            site,
            "topic",
            joined.connection_id,
            r#"{"type":"set_state","id":"2","key":"details","value":{"data":123}}"#,
            32 * 1024,
            &mut rate,
        )
        .unwrap();
        assert_eq!(serde_json::from_str::<Value>(&ack).unwrap()["type"], "ack");
        let state = joined.outbound.try_recv().unwrap();
        assert_eq!(
            serde_json::from_str::<Value>(&state).unwrap()["after"],
            json!({"data":123})
        );
    }
}
