import type { ToolTrace } from '../types';

interface Props {
  trace: ToolTrace;
}

/** Renders one v2 trace event: client tool call/result, provider-side
 *  tool use, or provider grounding metadata. */
export default function ToolCall({ trace }: Props) {
  const view = traceView(trace);
  const media = collectMediaUris(view.response);
  return (
    <article className="tool-call">
      <header>
        <span className="tool-call__name">{view.name}</span>
      </header>
      {media.length > 0 && (
        <div className="tool-call__media">
          {media.map((m, i) =>
            m.kind === 'image' ? (
              <img key={i} className="context-image" src={m.path} alt={view.name} />
            ) : (
              <video key={i} className="context-video" controls src={m.path} />
            )
          )}
        </div>
      )}
      <details>
        <summary>{view.requestLabel}</summary>
        <pre>{prettyJson(view.request)}</pre>
      </details>
      <details>
        <summary>{view.responseLabel}</summary>
        <pre>{prettyJson(view.response)}</pre>
      </details>
    </article>
  );
}

function traceView(trace: ToolTrace) {
  switch (trace.kind) {
    case 'client':
      return {
        name: trace.trace.call.name,
        requestLabel: 'Request',
        request: trace.trace.call.input,
        responseLabel: trace.trace.result.is_error ? 'Error result' : 'Result',
        response: {
          result: trace.trace.result.content,
          trace_response: trace.trace.trace_response,
        },
      };
    case 'server':
      return {
        name: `${trace.tool.provider}/${trace.tool.name}`,
        requestLabel: 'Provider event',
        request: {
          id: trace.tool.id,
          status: trace.tool.status,
        },
        responseLabel: 'Raw payload',
        response: trace.tool.raw,
      };
    case 'grounding':
      return {
        name: `${trace.metadata.provider}/grounding`,
        requestLabel: 'Metadata',
        request: { provider: trace.metadata.provider },
        responseLabel: 'Raw payload',
        response: trace.metadata.raw,
      };
  }
}

type MediaRef = { kind: 'image' | 'video'; path: string };

/** Walk the value, collecting any string that looks like a
 *  `file://images/...` or `file://videos/...` URI. */
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
