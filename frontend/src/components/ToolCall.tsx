import type { ToolCallRecord } from '../types';

interface Props {
  call: ToolCallRecord;
}

/** Renders one tool call: its name, any media it produced (scanned out
 *  of the response JSON), and collapsible request + response panels. */
export default function ToolCall({ call }: Props) {
  const media = collectMediaUris(call.response);
  return (
    <article className="tool-call">
      <header>
        <span className="tool-call__name">{call.tool_name}</span>
      </header>
      {media.length > 0 && (
        <div className="tool-call__media">
          {media.map((m, i) =>
            m.kind === 'image' ? (
              <img key={i} className="context-image" src={m.path} alt={call.tool_name} />
            ) : (
              <video key={i} className="context-video" controls src={m.path} />
            )
          )}
        </div>
      )}
      <details>
        <summary>Request</summary>
        <pre>{prettyJson(call.request)}</pre>
      </details>
      <details>
        <summary>Response</summary>
        <pre>{prettyJson(call.response)}</pre>
      </details>
    </article>
  );
}

type MediaRef = { kind: 'image' | 'video'; path: string };

/** Walk the value, collecting any string that looks like a
 *  `file://images/…` or `file://videos/…` URI. Mirrors the Rust
 *  `walk_for_media_uris` helper in the old maud renderer. */
function collectMediaUris(value: unknown): MediaRef[] {
  const out: MediaRef[] = [];
  walk(value, out);
  return out;
}

function walk(value: unknown, out: MediaRef[]) {
  if (typeof value === 'string') {
    if (value.startsWith('file://images/')) {
      out.push({ kind: 'image', path: '/' + value.slice('file://'.length) });
    } else if (value.startsWith('file://videos/')) {
      out.push({ kind: 'video', path: '/' + value.slice('file://'.length) });
    }
    return;
  }
  if (Array.isArray(value)) {
    value.forEach((v) => walk(v, out));
    return;
  }
  if (value && typeof value === 'object') {
    Object.values(value).forEach((v) => walk(v, out));
  }
}

function prettyJson(value: unknown): string {
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}
