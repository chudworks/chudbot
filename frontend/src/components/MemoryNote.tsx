import type { ReactNode } from 'react';

/** Model-facing memory payload shared by injected notes and lookup_user_memory. */
interface MemoryPayload {
  message_provider?: string;
  target_user_id?: string;
  scope_key?: string;
  profile_found?: boolean;
  profile_revision?: number | null;
  profile?: string;
  recent_events?: MemoryEvent[];
  recent_diary_entries?: MemoryDiaryEntry[];
}

interface MemoryEvent {
  id?: string;
  kind?: string;
  body?: string;
  tags?: string[];
  confidence?: number | null;
  created_at?: string;
}

interface MemoryDiaryEntry {
  id?: string;
  window_start?: string;
  window_end?: string;
  created_at?: string;
  markdown?: string;
}

interface ParsedMemoryNote {
  userId: string;
  displayName: string | null;
  reason: string | null;
  preamble: string;
  payload: MemoryPayload;
}

interface Props {
  source: string;
  content: string;
  /** Shown when the content is not a parseable memory payload. */
  fallback?: ReactNode;
}

/** Detect `memory:user:<id>` context items and parse preamble + JSON payload. */
function parseMemoryNote(
  source: string,
  content: string
): ParsedMemoryNote | null {
  if (!source.startsWith('memory:user:')) {
    return null;
  }
  const userId = source.slice('memory:user:'.length);
  const jsonStart = content.indexOf('{');
  if (jsonStart === -1) {
    return null;
  }
  let payload: unknown;
  try {
    payload = JSON.parse(content.slice(jsonStart));
  } catch {
    return null;
  }
  if (!payload || typeof payload !== 'object' || Array.isArray(payload)) {
    return null;
  }
  const preamble = content.slice(0, jsonStart).trim();
  const meta = parsePreamble(preamble);
  return {
    userId,
    displayName: meta.displayName,
    reason: meta.reason,
    preamble,
    payload: payload as MemoryPayload,
  };
}

/**
 * Styled viewer for automatically injected user-memory system notes.
 * Falls back when the content is not a parseable memory payload so the caller
 * can still show the raw system item.
 */
export default function MemoryNote({ source, content, fallback = null }: Props) {
  const parsed = parseMemoryNote(source, content);
  if (!parsed) {
    return <>{fallback}</>;
  }

  const { userId, displayName, reason, preamble, payload } = parsed;
  const profile = payload.profile ?? '';
  const profileFound = payload.profile_found === true;
  const emptyProfile = !profileFound || profile === '(no stored memory)' || !profile.trim();
  const diary = Array.isArray(payload.recent_diary_entries)
    ? payload.recent_diary_entries
    : [];
  const events = Array.isArray(payload.recent_events) ? payload.recent_events : [];
  const title = displayName ?? userId;

  return (
    <article className="memory-note">
      <header className="memory-note__header">
        <div className="memory-note__title-row">
          <span className="memory-note__badge">memory</span>
          <strong className="memory-note__name">{title}</strong>
          {displayName && (
            <code className="memory-note__user-id">{userId}</code>
          )}
        </div>
        {reason && (
          <p className="memory-note__reason">Loaded because {reason}.</p>
        )}
        <div className="memory-note__meta">
          {payload.profile_revision != null && (
            <span className="memory-note__chip">
              rev <code>{payload.profile_revision}</code>
            </span>
          )}
          {payload.message_provider && (
            <span className="memory-note__chip">
              <code>{payload.message_provider}</code>
            </span>
          )}
          {payload.scope_key && (
            <span className="memory-note__chip">
              <code>{payload.scope_key}</code>
            </span>
          )}
          <span
            className={
              profileFound
                ? 'memory-note__chip memory-note__chip--ok'
                : 'memory-note__chip memory-note__chip--muted'
            }
          >
            {profileFound ? 'profile found' : 'no profile'}
          </span>
        </div>
      </header>

      <section className="memory-note__section">
        <h4 className="memory-note__section-title">Profile</h4>
        {emptyProfile ? (
          <p className="memory-note__empty">(no stored memory)</p>
        ) : (
          <div className="memory-note__markdown">
            <SimpleMarkdown text={profile} />
          </div>
        )}
      </section>

      {diary.length > 0 && (
        <details className="memory-note__fold" open>
          <summary>
            Recent diary ({diary.length})
          </summary>
          <div className="memory-note__diary-list">
            {diary.map((entry, index) => (
              <DiaryEntryView
                key={entry.id ?? `diary-${index}`}
                entry={entry}
              />
            ))}
          </div>
        </details>
      )}

      {events.length > 0 && (
        <details className="memory-note__fold">
          <summary>
            Pending events ({events.length})
          </summary>
          <div className="memory-note__event-list">
            {events.map((event, index) => (
              <EventView key={event.id ?? `event-${index}`} event={event} />
            ))}
          </div>
        </details>
      )}

      <details className="memory-note__fold memory-note__fold--raw">
        <summary>Raw note</summary>
        <pre className="memory-note__raw">{preamble}</pre>
        <pre className="memory-note__raw">
          {JSON.stringify(payload, null, 2)}
        </pre>
      </details>
    </article>
  );
}

function DiaryEntryView({ entry }: { entry: MemoryDiaryEntry }) {
  const windowLabel = formatWindow(entry.window_start, entry.window_end);
  const createdLabel = formatTimestamp(entry.created_at);
  return (
    <article className="memory-note__diary">
      <header>
        {windowLabel && <span>{windowLabel}</span>}
        {createdLabel && (
          <>
            {windowLabel ? ' · ' : null}
            <span>created {createdLabel}</span>
          </>
        )}
        {entry.id && (
          <>
            {' · '}
            <code className="memory-note__id">{shortId(entry.id)}</code>
          </>
        )}
      </header>
      {entry.markdown ? (
        <div className="memory-note__markdown memory-note__markdown--compact">
          <SimpleMarkdown text={entry.markdown} />
        </div>
      ) : (
        <p className="memory-note__empty">(empty diary entry)</p>
      )}
    </article>
  );
}

function EventView({ event }: { event: MemoryEvent }) {
  return (
    <article className="memory-note__event">
      <header>
        {event.kind && (
          <span className="memory-note__event-kind">{event.kind}</span>
        )}
        {event.created_at && (
          <>
            {' · '}
            <span>{formatTimestamp(event.created_at)}</span>
          </>
        )}
        {event.confidence != null && (
          <>
            {' · '}
            <span>confidence {event.confidence}</span>
          </>
        )}
      </header>
      {event.body && <p className="memory-note__event-body">{event.body}</p>}
      {Array.isArray(event.tags) && event.tags.length > 0 && (
        <div className="memory-note__tags">
          {event.tags.map((tag) => (
            <span className="memory-note__chip" key={tag}>
              {tag}
            </span>
          ))}
        </div>
      )}
    </article>
  );
}

/** Lightweight markdown renderer for memory profiles (headings, lists, inline). */
function SimpleMarkdown({ text }: { text: string }) {
  const lines = text.replace(/\r\n/g, '\n').split('\n');
  const blocks: ReactNode[] = [];
  let listItems: string[] = [];
  let paragraph: string[] = [];
  let key = 0;

  const flushList = () => {
    if (listItems.length === 0) return;
    blocks.push(
      <ul key={`ul-${key++}`}>
        {listItems.map((item, i) => (
          <li key={i}>{renderInline(item)}</li>
        ))}
      </ul>
    );
    listItems = [];
  };

  const flushParagraph = () => {
    if (paragraph.length === 0) return;
    blocks.push(
      <p key={`p-${key++}`}>{renderInline(paragraph.join(' '))}</p>
    );
    paragraph = [];
  };

  for (const raw of lines) {
    const line = raw.trimEnd();
    const trimmed = line.trim();

    if (trimmed === '') {
      flushList();
      flushParagraph();
      continue;
    }

    const heading = trimmed.match(/^(#{1,3})\s+(.+)$/);
    if (heading) {
      flushList();
      flushParagraph();
      const level = heading[1].length;
      const content = renderInline(heading[2]);
      if (level === 1) {
        blocks.push(<h1 key={`h-${key++}`}>{content}</h1>);
      } else if (level === 2) {
        blocks.push(<h2 key={`h-${key++}`}>{content}</h2>);
      } else {
        blocks.push(<h3 key={`h-${key++}`}>{content}</h3>);
      }
      continue;
    }

    const listMatch = trimmed.match(/^[-*]\s+(.+)$/);
    if (listMatch) {
      flushParagraph();
      listItems.push(listMatch[1]);
      continue;
    }

    flushList();
    paragraph.push(trimmed);
  }

  flushList();
  flushParagraph();
  return <>{blocks}</>;
}

function renderInline(text: string): ReactNode {
  // Split on **bold**, `code`, and leave the rest as text.
  const parts: ReactNode[] = [];
  const pattern = /(\*\*[^*]+\*\*|`[^`]+`)/g;
  let last = 0;
  let match: RegExpExecArray | null;
  let i = 0;
  while ((match = pattern.exec(text)) !== null) {
    if (match.index > last) {
      parts.push(text.slice(last, match.index));
    }
    const token = match[0];
    if (token.startsWith('**')) {
      parts.push(<strong key={i++}>{token.slice(2, -2)}</strong>);
    } else {
      parts.push(<code key={i++}>{token.slice(1, -1)}</code>);
    }
    last = match.index + token.length;
  }
  if (last < text.length) {
    parts.push(text.slice(last));
  }
  return parts.length === 1 ? parts[0] : parts;
}

function parsePreamble(preamble: string): {
  displayName: string | null;
  reason: string | null;
} {
  const nameMatch = preamble.match(
    /^User memory note for (.+?) \(user id [^)]+\)/i
  );
  const reasonMatch = preamble.match(/loaded automatically because ([^.]+)\./i);
  return {
    displayName: nameMatch?.[1]?.trim() || null,
    reason: reasonMatch?.[1]?.trim() || null,
  };
}

function formatTimestamp(iso: string | undefined): string | null {
  if (!iso) return null;
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return iso;
  return new Intl.DateTimeFormat(undefined, {
    dateStyle: 'medium',
    timeStyle: 'short',
  }).format(date);
}

function formatWindow(
  start: string | undefined,
  end: string | undefined
): string | null {
  const startLabel = formatTimestamp(start);
  const endLabel = formatTimestamp(end);
  if (startLabel && endLabel) return `${startLabel} → ${endLabel}`;
  return startLabel ?? endLabel;
}

function shortId(id: string): string {
  return id.length > 12 ? `${id.slice(0, 8)}…` : id;
}
