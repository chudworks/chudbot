import { useEffect } from 'react';
import { useParams } from 'react-router-dom';
import { useConversation } from '../store';
import { usePageTitle } from '../title';
import Turn from './Turn';
import RelativeTime from './RelativeTime';

export default function ConversationView() {
  const { id } = useParams<{ id: string }>();
  const state = useConversation((s) => s.state);
  const storeId = useConversation((s) => s.id);
  const load = useConversation((s) => s.load);
  const refresh = useConversation((s) => s.refresh);

  // Load on mount or whenever the URL conversation id changes.
  useEffect(() => {
    if (!id) return;
    void load(id);
  }, [id, load]);

  // Subscribe to SSE while this conversation is mounted. Any event for
  // this conversation (or a global event like a user avatar update)
  // triggers a refetch. Refetches are debounced via the store's own
  // single-flight semantics — overlapping calls just replace state.
  useEffect(() => {
    if (!id) return;
    const source = new EventSource(`/api/conversations/${id}/events`);
    const onAny = () => {
      if (useConversation.getState().id === id) {
        void refresh();
      }
    };
    // Listen to every named event the backend emits. `onmessage`
    // alone wouldn't catch them because EventSource only fires
    // `message` for unnamed events.
    [
      'created',
      'turn_started',
      'turn_updated',
      'tool_trace_recorded',
      'context_recorded',
      'title_updated',
      'conversation_updated',
      'user_profile_updated',
      'lag',
    ].forEach((name) => source.addEventListener(name, onAny));
    source.onerror = () => {
      // EventSource auto-reconnects with backoff. No-op on error;
      // logging here is too noisy because every reconnect attempt
      // emits one.
    };
    return () => source.close();
  }, [id, refresh]);

  // Tab title tracks the load state: the conversation's own title once
  // it's ready, an interim label otherwise. Called unconditionally
  // (rules of hooks) before any early return below.
  const pageTitle =
    !id
      ? 'Conversation'
      : state.kind === 'ready'
        ? (state.view.conversation.title ?? 'Untitled conversation')
        : state.kind === 'error'
          ? state.status === 404
            ? 'Not found'
            : 'Error'
          : 'Loading…';
  usePageTitle(pageTitle);

  if (!id) return <main className="center"><p>missing conversation id</p></main>;
  if (state.kind === 'idle' || (state.kind === 'loading' && storeId !== id)) {
    return <main className="center"><p>Loading…</p></main>;
  }
  if (state.kind === 'loading') {
    return <main className="center"><p>Loading…</p></main>;
  }
  if (state.kind === 'error') {
    return (
      <main className="center">
        <h1>{state.status === 404 ? '404' : 'Error'}</h1>
        <p>{state.status === 404 ? 'No conversation here. The link may be wrong or the row was deleted.' : state.message}</p>
      </main>
    );
  }

  const { conversation, turns, users } = state.view;
  const title = conversation.title ?? 'Untitled conversation';
  const model = `${conversation.provider}/${conversation.initial_model}`;
  const starter = users[userKey(conversation.created_by)]?.label;
  const stopper = conversation.stopped_by
    ? users[userKey(conversation.stopped_by)]?.label
    : null;

  return (
    <>
      <header className="conv-header">
        <h1>{title}</h1>
        <p className="meta">
          <RelativeTime iso={conversation.created_at} prefix="Started " />
          {starter && (
            <>
              {' by '}
              <strong>{starter}</strong>
            </>
          )}
          {' · model '}
          <code>{model}</code>
          {' · agent '}
          <code>{conversation.agent_name}</code>
        </p>
        {conversation.stopped_at && (
          <p className="stopped-banner" role="status">
            🛑 {stopper ?? 'An admin'} paused the bot in this conversation
            {' '}
            <RelativeTime iso={conversation.stopped_at} prefix="" />.
            {' '}It won’t reply until the 🛑 reaction is removed.
          </p>
        )}
      </header>
      <main className="conv">
        {turns.map((tv) => (
          <Turn key={tv.turn.id} turnView={tv} users={users} />
        ))}
        {turns.length === 0 && <p className="empty">No turns yet.</p>}
      </main>
    </>
  );
}

function userKey(user: { platform: string; guild_id: string | null; user_id: string }): string {
  return `${user.platform}:${user.guild_id ?? 'global'}:${user.user_id}`;
}
