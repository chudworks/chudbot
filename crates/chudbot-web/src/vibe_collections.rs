use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

use axum::extract::ws::{Message, WebSocket};
use chudbot_api::{VibeCollectionChange, VibeSite, VibeSiteId};
use chudbot_vibe::VibeLimitsConfig;
use dashmap::DashMap;
use dashmap::mapref::entry::Entry;
use futures::{SinkExt, StreamExt};
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

const SUBSCRIBE_TIMEOUT: Duration = Duration::from_secs(10);
const PING_INTERVAL: Duration = Duration::from_secs(30);
const MAX_FILTERS: usize = 64;

#[derive(Debug, Clone)]
pub(crate) struct CollectionWatchHub {
    sites: Arc<DashMap<VibeSiteId, Arc<SiteWatches>>>,
    connections: Arc<AtomicUsize>,
    max_connections: usize,
    max_connections_per_site: usize,
    outbound_queue: usize,
}

#[derive(Debug, Default)]
struct SiteWatches {
    collections: DashMap<String, Arc<Mutex<CollectionWatches>>>,
    connections: AtomicUsize,
}

#[derive(Debug, Default)]
struct CollectionWatches {
    clients: HashMap<Uuid, WatchClient>,
    closed: bool,
}

#[derive(Debug)]
struct WatchClient {
    filters: BTreeMap<String, Value>,
    outbound: mpsc::Sender<Arc<str>>,
    cancel: CancellationToken,
}

#[derive(Debug)]
struct JoinedWatch {
    id: Uuid,
    outbound: mpsc::Receiver<Arc<str>>,
    cancel: CancellationToken,
}

#[derive(Debug, Deserialize)]
struct SubscribeMessage {
    #[serde(rename = "type")]
    kind: String,
    #[serde(default, rename = "where")]
    filters: BTreeMap<String, Value>,
}

#[derive(Debug)]
struct WatchError {
    code: &'static str,
    message: &'static str,
}

impl WatchError {
    const fn new(code: &'static str, message: &'static str) -> Self {
        Self { code, message }
    }

    fn json(&self) -> String {
        json!({"type":"error","code":self.code,"message":self.message}).to_string()
    }
}

impl CollectionWatchHub {
    pub(crate) fn new(limits: &VibeLimitsConfig) -> Self {
        Self {
            sites: Arc::new(DashMap::new()),
            connections: Arc::new(AtomicUsize::new(0)),
            max_connections: limits.max_room_connections_total,
            max_connections_per_site: usize::from(limits.max_room_connections_per_site),
            outbound_queue: usize::from(limits.room_outbound_queue),
        }
    }

    fn join(
        &self,
        site: VibeSiteId,
        collection: &str,
        filters: BTreeMap<String, Value>,
    ) -> Result<JoinedWatch, WatchError> {
        if !reserve_capacity(&self.connections, self.max_connections) {
            return Err(WatchError::new(
                "watches_busy",
                "Vibe collection watches are at their deployment-wide connection limit.",
            ));
        }
        let site_watches = match self.sites.entry(site) {
            Entry::Occupied(entry) => {
                if !reserve_capacity(&entry.get().connections, self.max_connections_per_site) {
                    release_capacity(&self.connections);
                    return Err(WatchError::new(
                        "site_watches_busy",
                        "This site has reached its Vibe collection watch limit.",
                    ));
                }
                Arc::clone(entry.get())
            }
            Entry::Vacant(entry) => {
                let site_watches = Arc::new(SiteWatches::default());
                let reserved =
                    reserve_capacity(&site_watches.connections, self.max_connections_per_site);
                debug_assert!(reserved);
                entry.insert(Arc::clone(&site_watches));
                site_watches
            }
        };

        loop {
            let watches = match site_watches.collections.entry(collection.to_string()) {
                Entry::Occupied(entry) => Arc::clone(entry.get()),
                Entry::Vacant(entry) => {
                    let watches = Arc::new(Mutex::new(CollectionWatches::default()));
                    entry.insert(Arc::clone(&watches));
                    watches
                }
            };
            let mut watches = lock_watches(&watches);
            if watches.closed {
                drop(watches);
                std::thread::yield_now();
                continue;
            }
            let id = Uuid::new_v4();
            let cancel = CancellationToken::new();
            let (sender, outbound) = mpsc::channel(self.outbound_queue);
            watches.clients.insert(
                id,
                WatchClient {
                    filters,
                    outbound: sender,
                    cancel: cancel.clone(),
                },
            );
            return Ok(JoinedWatch {
                id,
                outbound,
                cancel,
            });
        }
    }

    fn leave(&self, site: VibeSiteId, collection: &str, id: Uuid) {
        let Some(site_watches) = self.sites.get(&site).map(|entry| Arc::clone(entry.value()))
        else {
            return;
        };
        let Some(watches) = site_watches
            .collections
            .get(collection)
            .map(|entry| Arc::clone(entry.value()))
        else {
            return;
        };
        if lock_watches(&watches).clients.remove(&id).is_none() {
            return;
        }
        release_capacity(&site_watches.connections);
        release_capacity(&self.connections);
        self.remove_empty_collection(&site_watches, collection, &watches);
        self.remove_empty_site(site, &site_watches);
    }

    pub(crate) fn publish(
        &self,
        site: VibeSiteId,
        collection: &str,
        changes: &[VibeCollectionChange],
    ) {
        if changes.is_empty() {
            return;
        }
        let Some(watches) = self.sites.get(&site).and_then(|site| {
            site.collections
                .get(collection)
                .map(|entry| Arc::clone(entry.value()))
        }) else {
            return;
        };
        let watches = lock_watches(&watches);
        for (id, client) in &watches.clients {
            for change in changes {
                let matches_before = document_matches(change.before.as_ref(), &client.filters);
                let matches_after = document_matches(change.after.as_ref(), &client.filters);
                if !matches_before && !matches_after {
                    continue;
                }
                let message = Arc::<str>::from(
                    json!({
                        "type":"change",
                        "changeType":change.kind,
                        "document":change.after.as_ref().or(change.before.as_ref()),
                        "before":change.before,
                        "after":change.after,
                        "matchesBefore":matches_before,
                        "matchesAfter":matches_after,
                    })
                    .to_string(),
                );
                if client.outbound.try_send(message).is_err() {
                    tracing::warn!(connection=%id, site=%site, collection, "closing slow Vibe collection watch");
                    client.cancel.cancel();
                    break;
                }
            }
        }
    }

    fn remove_empty_collection(
        &self,
        site: &SiteWatches,
        collection: &str,
        expected: &Arc<Mutex<CollectionWatches>>,
    ) {
        let Entry::Occupied(entry) = site.collections.entry(collection.to_string()) else {
            return;
        };
        if !Arc::ptr_eq(entry.get(), expected) {
            return;
        }
        let mut watches = lock_watches(entry.get());
        if !watches.clients.is_empty() {
            return;
        }
        watches.closed = true;
        drop(watches);
        entry.remove();
    }

    fn remove_empty_site(&self, site: VibeSiteId, expected: &Arc<SiteWatches>) {
        let Entry::Occupied(entry) = self.sites.entry(site) else {
            return;
        };
        if Arc::ptr_eq(entry.get(), expected)
            && expected.connections.load(Ordering::Acquire) == 0
            && expected.collections.is_empty()
        {
            entry.remove();
        }
    }
}

pub(crate) async fn serve_collection_watch(
    mut socket: WebSocket,
    hub: CollectionWatchHub,
    site: VibeSite,
    collection: String,
    shutdown: CancellationToken,
    max_message_bytes: usize,
) {
    let filters = match receive_subscription(&mut socket, max_message_bytes).await {
        Ok(filters) => filters,
        Err(error) => {
            let _ = socket.send(Message::Text(error.json().into())).await;
            let _ = socket.close().await;
            return;
        }
    };
    let mut joined = match hub.join(site.id, &collection, filters) {
        Ok(joined) => joined,
        Err(error) => {
            let _ = socket.send(Message::Text(error.json().into())).await;
            let _ = socket.close().await;
            return;
        }
    };
    if socket
        .send(Message::Text(json!({"type":"ready"}).to_string().into()))
        .await
        .is_err()
    {
        hub.leave(site.id, &collection, joined.id);
        return;
    }
    tracing::info!(site=%site.name, site_id=%site.id, collection, connection=%joined.id, "Vibe collection watch opened");
    let (mut sink, mut stream) = socket.split();
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
                    Message::Close(_) => break,
                    Message::Ping(value) => {
                        if sink.send(Message::Pong(value)).await.is_err() { break; }
                    }
                    Message::Pong(_) => {}
                    Message::Text(_) | Message::Binary(_) => {
                        let error = WatchError::new("already_subscribed", "This collection watch is already subscribed.");
                        if sink.send(Message::Text(error.json().into())).await.is_err() { break; }
                    }
                }
            }
            _ = ping.tick() => {
                if sink.send(Message::Ping(Vec::new().into())).await.is_err() { break; }
            }
        }
    }
    hub.leave(site.id, &collection, joined.id);
    tracing::info!(site=%site.name, site_id=%site.id, collection, connection=%joined.id, "Vibe collection watch closed");
}

async fn receive_subscription(
    socket: &mut WebSocket,
    max_message_bytes: usize,
) -> Result<BTreeMap<String, Value>, WatchError> {
    let received = tokio::time::timeout(SUBSCRIBE_TIMEOUT, socket.recv()).await;
    let text = match received {
        Ok(Some(Ok(Message::Text(text)))) if text.len() <= max_message_bytes => text,
        Ok(Some(Ok(_))) => {
            return Err(WatchError::new(
                "invalid_watch",
                "The first watch message must be a JSON subscription.",
            ));
        }
        Ok(Some(Err(_)) | None) => {
            return Err(WatchError::new(
                "connection_closed",
                "The collection watch closed before subscribing.",
            ));
        }
        Err(_) => {
            return Err(WatchError::new(
                "subscribe_timeout",
                "The collection watch timed out before subscribing.",
            ));
        }
    };
    let request = serde_json::from_str::<SubscribeMessage>(text.as_str()).map_err(|_| {
        WatchError::new(
            "invalid_watch",
            "The collection watch subscription is invalid.",
        )
    })?;
    if request.kind != "watch"
        || request.filters.len() > MAX_FILTERS
        || !request.filters.values().all(valid_filter)
    {
        return Err(WatchError::new(
            "invalid_watch",
            "The collection watch subscription is invalid.",
        ));
    }
    Ok(request.filters)
}

fn valid_filter(value: &Value) -> bool {
    match value {
        Value::Array(values) => values.iter().all(is_json_scalar),
        value => is_json_scalar(value),
    }
}

fn document_matches(document: Option<&Value>, filters: &BTreeMap<String, Value>) -> bool {
    let Some(document) = document.and_then(Value::as_object) else {
        return false;
    };
    filters.iter().all(|(column, expected)| {
        let Some(actual) = document.get(column).filter(|value| is_json_scalar(value)) else {
            return false;
        };
        match expected {
            Value::Array(values) => values.iter().any(|value| value == actual),
            value => value == actual,
        }
    })
}

fn is_json_scalar(value: &Value) -> bool {
    !matches!(value, Value::Array(_) | Value::Object(_))
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

fn lock_watches(watches: &Mutex<CollectionWatches>) -> MutexGuard<'_, CollectionWatches> {
    watches.lock().unwrap_or_else(|poison| poison.into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;
    use chudbot_api::VibeCollectionChangeKind;

    #[tokio::test]
    async fn publishes_updates_entering_and_leaving_a_filter() {
        let hub = CollectionWatchHub::new(&VibeLimitsConfig::default());
        let site = VibeSiteId::new();
        let filters = BTreeMap::from([("user".into(), json!("123"))]);
        let mut joined = hub.join(site, "scores", filters).unwrap();
        hub.publish(
            site,
            "scores",
            &[VibeCollectionChange {
                kind: VibeCollectionChangeKind::Update,
                before: Some(json!({"id":"1","user":"123"})),
                after: Some(json!({"id":"1","user":"456"})),
            }],
        );
        let message = joined.outbound.recv().await.unwrap();
        let value = serde_json::from_str::<Value>(&message).unwrap();
        assert_eq!(value["matchesBefore"], true);
        assert_eq!(value["matchesAfter"], false);
        hub.leave(site, "scores", joined.id);
    }

    #[test]
    fn json_columns_do_not_match_filters() {
        assert!(!document_matches(
            Some(&json!({"value":{"nested":true}})),
            &BTreeMap::from([("value".into(), json!("anything"))]),
        ));
    }
}
