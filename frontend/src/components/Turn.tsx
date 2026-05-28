import type { ContextItem, DiscordUser, TurnView } from '../types';
import Avatar from './Avatar';
import RelativeTime from './RelativeTime';
import ToolCall from './ToolCall';
import { displayNameFor } from '../users';

interface Props {
  turnView: TurnView;
  users: Record<string, DiscordUser>;
}

export default function Turn({ turnView, users }: Props) {
  const { turn, system_prompt, context, tool_calls } = turnView;
  // discord_user_id is a snowflake string and the `users` map is keyed
  // by the same string, so this lookup is exact (see types.ts on why
  // IDs are strings, not numbers).
  const user =
    turn.discord_user_id != null ? users[turn.discord_user_id] : undefined;
  const userLabel =
    displayNameFor(user) ?? turn.discord_user_name ?? 'user';

  return (
    <section className="turn">
      <header className="turn__header">
        <span className="turn__index">Turn {turn.turn_index + 1}</span>
        <StatusBadge status={turn.status} />
        {turn.persona_name && (
          <span className="turn__persona">
            · persona <code>{turn.persona_name}</code>
          </span>
        )}
        <span className="turn__time">
          · <RelativeTime iso={turn.created_at} />
        </span>
      </header>

      <div className="turn__user">
        <div className="turn__user-row">
          <Avatar user={user} fallbackName={userLabel} size={32} />
          <div className="turn__user-meta">
            <strong>{userLabel}</strong>
          </div>
        </div>
        <pre className="turn__content">{turn.user_content}</pre>
      </div>

      {system_prompt && (
        <details className="context">
          <summary>System prompt</summary>
          <pre className="turn__content">{system_prompt}</pre>
        </details>
      )}

      {context.length > 0 && (
        <details className="context">
          <summary>Context fed to model ({context.length} items)</summary>
          {context.map((item, i) => (
            <ContextItemView key={i} item={item} />
          ))}
        </details>
      )}

      {tool_calls.length > 0 && (
        <section className="tools">
          <h3>Tool calls ({tool_calls.length})</h3>
          {tool_calls.map((tc, i) => (
            <ToolCall key={i} call={tc} />
          ))}
        </section>
      )}

      <div className="turn__assistant">
        <h3>Assistant</h3>
        {turn.assistant_content ? (
          <pre className="turn__content">{turn.assistant_content}</pre>
        ) : turn.status === 'failed' ? (
          <pre className="turn__content turn__content--err">
            {turn.error ?? '(no error message)'}
          </pre>
        ) : (
          <em>(no response yet)</em>
        )}
      </div>
    </section>
  );
}

function StatusBadge({ status }: { status: string }) {
  const cls =
    status === 'completed'
      ? 'badge badge--ok'
      : status === 'failed'
      ? 'badge badge--err'
      : 'badge';
  return <span className={cls}>{status}</span>;
}

function ContextItemView({ item }: { item: ContextItem }) {
  const isImage = isImageUri(item.content);
  const isVideo = isVideoUri(item.content);
  return (
    <article className="context-item">
      <header>
        <span className="context-item__role">{item.role}</span>
        {' · '}
        <span className="context-item__source">{item.source}</span>
      </header>
      {isImage ? (
        <img
          className="context-image"
          src={toWebPath(item.content)}
          alt="user attachment"
        />
      ) : isVideo ? (
        <video className="context-video" controls src={toWebPath(item.content)} />
      ) : (
        <pre>{item.content}</pre>
      )}
    </article>
  );
}

// These mirror the helpers in `core::storage`. Duplicated here because
// the frontend is the only place that needs them in JS-land.
function isImageUri(s: string): boolean {
  return s.startsWith('file://images/');
}
function isVideoUri(s: string): boolean {
  return s.startsWith('file://videos/');
}
function toWebPath(s: string): string {
  return '/' + s.slice('file://'.length);
}
