import type { ClientToolResultContent, ToolTrace } from '../types';

interface Props {
  trace: ToolTrace;
}

/** Renders one trace event: client tool call/result, provider-side tool use,
 *  or provider grounding metadata. */
export default function ToolCall({ trace }: Props) {
  const view = traceView(trace);
  const media = collectMediaUris([view.response, view.tracePayload]);
  const showTracePayload =
    view.tracePayload !== undefined && !jsonEqual(view.response, view.tracePayload);
  return (
    <article className="tool-call">
      <header>
        <span className="tool-call__name">{view.name}</span>
      </header>
      {media.length > 0 && (
        <div className="tool-call__media">
          {media.map((m) =>
            m.kind === 'image' ? (
              <img key={m.uri} className="context-image" src={m.path} alt={view.name} />
            ) : m.kind === 'video' ? (
              <video key={m.uri} className="context-video" controls src={m.path} />
            ) : (
              <audio key={m.uri} className="context-audio" controls src={m.path} />
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
      {showTracePayload && (
        <details>
          <summary>Trace payload</summary>
          <pre>{prettyJson(view.tracePayload)}</pre>
        </details>
      )}
    </article>
  );
}

interface TraceView {
  name: string;
  requestLabel: string;
  request: unknown;
  responseLabel: string;
  response: unknown;
  tracePayload?: unknown;
}

function traceView(trace: ToolTrace): TraceView {
  switch (trace.kind) {
    case 'client':
      return {
        name: trace.trace.call.name,
        requestLabel: 'Request',
        request: trace.trace.call.input,
        responseLabel: trace.trace.result.is_error ? 'Error result' : 'Result',
        response: resultContentValue(trace.trace.result.content),
        tracePayload: trace.trace.trace_payload,
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
        name: groundingName(trace.metadata.provider, trace.metadata.raw),
        requestLabel: 'Metadata',
        request: { provider: trace.metadata.provider },
        responseLabel: 'Raw payload',
        response: trace.metadata.raw,
      };
  }
}

function groundingName(provider: string, raw: unknown): string {
  return `${provider}/${isCitationMetadata(raw) ? 'citations' : 'grounding'}`;
}

function isCitationMetadata(raw: unknown): boolean {
  const values = Array.isArray(raw) ? raw : [raw];
  return values.some((value) => {
    if (!value || typeof value !== 'object') return false;
    const type = (value as { type?: unknown }).type;
    return typeof type === 'string' && type.endsWith('_location');
  });
}

function resultContentValue(content: ClientToolResultContent): unknown {
  switch (content.kind) {
    case 'json':
      return content.value;
    case 'text':
      return content.text;
  }
}

type MediaRef = { kind: 'image' | 'video' | 'audio'; uri: string; path: string };

/** Walk the value, collecting any string that looks like a
 *  `file://images/...`, `file://videos/...`, or `file://audio/...` URI. */
function collectMediaUris(value: unknown): MediaRef[] {
  const out: MediaRef[] = [];
  const seen = new Set<string>();
  walk(value, out, seen);
  return out;
}

function walk(value: unknown, out: MediaRef[], seen: Set<string>) {
  if (typeof value === 'string') {
    if (value.startsWith('file://images/')) {
      pushMediaRef(out, seen, 'image', value);
    } else if (value.startsWith('file://videos/')) {
      pushMediaRef(out, seen, 'video', value);
    } else if (value.startsWith('file://audio/')) {
      pushMediaRef(out, seen, 'audio', value);
    }
    return;
  }
  if (Array.isArray(value)) {
    value.forEach((v) => walk(v, out, seen));
    return;
  }
  if (value && typeof value === 'object') {
    Object.values(value).forEach((v) => walk(v, out, seen));
  }
}

function pushMediaRef(
  out: MediaRef[],
  seen: Set<string>,
  kind: 'image' | 'video' | 'audio',
  uri: string
) {
  if (seen.has(uri)) return;
  seen.add(uri);
  out.push({ kind, uri, path: '/' + uri.slice('file://'.length) });
}

function prettyJson(value: unknown): string {
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

function jsonEqual(left: unknown, right: unknown): boolean {
  return prettyJson(left) === prettyJson(right);
}
