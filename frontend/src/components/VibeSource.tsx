import { useEffect, useState, type CSSProperties } from 'react';
import { Link, useParams, useSearchParams } from 'react-router-dom';
import type { HighlightedToken } from '../syntaxHighlight';
import RelativeTime from './RelativeTime';

type Revision = {
  ordinal: number;
  message: string;
  authorId: string;
  createdAt: string;
};

type SourceResponse = {
  site: { name: string; url: string };
  revisions: Revision[];
  view:
    | { kind: 'tree'; revision: number; files: string[] }
    | { kind: 'file'; revision: number; path: string; text: string }
    | { kind: 'diff'; from: number; to: number; text: string };
};

function fileLabel(path: string): string {
  const name = path.split('/').pop() ?? path;
  if (name.startsWith('.') && !name.includes('.', 1)) return name.slice(1, 4).toUpperCase();
  const extension = name.split('.').pop()?.toLowerCase();
  const labels: Record<string, string> = {
    css: 'CSS',
    html: 'HTML',
    js: 'JS',
    json: 'JSON',
    jsx: 'JSX',
    lock: 'LOCK',
    md: 'MD',
    scss: 'SCSS',
    ts: 'TS',
    tsx: 'TSX',
    toml: 'TOML',
    yaml: 'YAML',
    yml: 'YAML',
  };
  return labels[extension ?? ''] ?? 'FILE';
}

function splitPath(path: string) {
  const slash = path.lastIndexOf('/');
  return slash < 0
    ? { directory: '', name: path }
    : { directory: path.slice(0, slash + 1), name: path.slice(slash + 1) };
}

function diffLineKind(line: string): string {
  if (line.startsWith('+++') || line.startsWith('---')) return 'meta';
  if (line.startsWith('+')) return 'added';
  if (line.startsWith('-')) return 'removed';
  if (line.startsWith('@@')) return 'hunk';
  if (line.startsWith('diff ') || line.startsWith('index ')) return 'meta';
  return 'context';
}

type SyntaxTokenStyle = CSSProperties & {
  '--shiki-dark'?: string;
  '--shiki-light'?: string;
};

function syntaxTokenStyle(token: HighlightedToken): SyntaxTokenStyle {
  const fontStyle = token.fontStyle ?? 0;
  return {
    '--shiki-dark': token.darkColor,
    '--shiki-light': token.lightColor,
    fontStyle: fontStyle & 1 ? 'italic' : undefined,
    fontWeight: fontStyle & 2 ? 700 : undefined,
    textDecoration: fontStyle & 4 ? 'underline' : undefined,
  };
}

function SourceLines({ text, path, diff = false }: { text: string; path?: string; diff?: boolean }) {
  const normalized = text.endsWith('\n') ? text.slice(0, -1) : text;
  const plainLines = normalized ? normalized.split('\n') : [''];
  const highlightKey = `${diff ? 'diff' : path ?? ''}\0${normalized}`;
  const [highlightResult, setHighlightResult] = useState<{
    key: string;
    lines: HighlightedToken[][];
  } | null>(null);
  const highlightedLines = highlightResult?.key === highlightKey ? highlightResult.lines : null;

  useEffect(() => {
    let active = true;

    void import('../syntaxHighlight')
      .then(async ({ highlightSource: highlight, languageForPath }) => {
        const language = diff ? 'diff' : languageForPath(path ?? '');
        if (!language) return null;
        return highlight(normalized, language);
      })
      .then((lines) => {
        if (active && lines && lines.length === plainLines.length) {
          setHighlightResult({ key: highlightKey, lines });
        }
      })
      .catch(() => {
        // Highlighting is an enhancement. Source remains readable if a grammar fails.
      });

    return () => { active = false; };
  }, [diff, highlightKey, normalized, path, plainLines.length]);

  const lines = highlightedLines ?? plainLines.map((line) => [{ content: line }]);

  return (
    <pre className={`vibe-source__code${diff ? ' vibe-source__code--diff' : ''}`}>
      <code>
        {lines.map((line, index) => (
          <span
            className={`vibe-source__code-line${diff ? ` vibe-source__code-line--${diffLineKind(plainLines[index] ?? '')}` : ''}`}
            key={index}
          >
            <span className="vibe-source__line-number" aria-hidden="true">{index + 1}</span>
            <span className="vibe-source__line-text">
              {line.length > 0
                ? line.map((token, tokenIndex) => (
                    <span className="vibe-source__syntax-token" style={syntaxTokenStyle(token)} key={tokenIndex}>
                      {token.content}
                    </span>
                  ))
                : '\u200b'}
            </span>
          </span>
        ))}
      </code>
    </pre>
  );
}

export default function VibeSource() {
  const { name = '' } = useParams();
  const [search] = useSearchParams();
  const query = search.toString();
  const requestKey = `${name}?${query}`;
  const [result, setResult] = useState<{
    key: string;
    data: SourceResponse | null;
    error: string | null;
  }>({ key: '', data: null, error: null });
  const data = result.key === requestKey ? result.data : null;
  const error = result.key === requestKey ? result.error : null;

  useEffect(() => {
    const controller = new AbortController();
    fetch(`/api/vibe/source/${encodeURIComponent(name)}${query ? `?${query}` : ''}`, {
      signal: controller.signal,
      credentials: 'same-origin',
      headers: { Accept: 'application/json' },
    })
      .then(async (response) => {
        if (!response.ok) throw new Error(`Source request failed (${response.status})`);
        return (await response.json()) as SourceResponse;
      })
      .then((value) => {
        document.title = `${value.site.name} source — Vibe`;
        setResult({ key: requestKey, data: value, error: null });
      })
      .catch((reason: unknown) => {
        if (!controller.signal.aborted) {
          setResult({
            key: requestKey,
            data: null,
            error: reason instanceof Error ? reason.message : 'Source request failed',
          });
        }
      });
    return () => controller.abort();
  }, [name, query, requestKey]);

  if (error) {
    return (
      <main className="vibe-source-state">
        <span className="vibe-source-state__mark">!</span>
        <h1>Source unavailable</h1>
        <p>{error}</p>
      </main>
    );
  }
  if (!data) {
    return (
      <main className="vibe-source-state" aria-live="polite">
        <span className="vibe-source-state__loader" aria-hidden="true" />
        <p>Loading source…</p>
      </main>
    );
  }

  const view = data.view;
  const selectedRevision = view.kind === 'diff' ? view.to : view.revision;

  return (
    <main className="vibe-source">
      <header className="vibe-source__header">
        <div className="vibe-source__identity">
          <span className="vibe-source__logo" aria-hidden="true">V</span>
          <div>
            <p className="vibe-source__eyebrow">Vibe source</p>
            <h1>{data.site.name}</h1>
          </div>
        </div>
        <div className="vibe-source__header-actions">
          <span className="vibe-source__revision-chip">Revision {selectedRevision}</span>
          <a className="vibe-source__site-link" href={data.site.url} target="_blank" rel="noreferrer">
            Open site
            <svg viewBox="0 0 16 16" aria-hidden="true">
              <path d="M5 3h8v8M13 3 3 13" />
            </svg>
          </a>
        </div>
      </header>

      <div className="vibe-source__layout">
        <aside className="vibe-source__history-panel" aria-label="Revision history">
          <div className="vibe-source__panel-heading">
            <h2>History</h2>
            <span>{data.revisions.length}</span>
          </div>
          <ol className="vibe-source__history">
            {data.revisions.map((revision, index) => {
              const older = data.revisions[index + 1];
              const active = revision.ordinal === selectedRevision;
              return (
                <li className={active ? 'is-active' : undefined} key={revision.ordinal}>
                  <div className="vibe-source__history-topline">
                    <Link
                      className="vibe-source__revision-link"
                      to={`?revision=${revision.ordinal}`}
                      aria-current={active ? 'page' : undefined}
                    >
                      r{revision.ordinal}
                    </Link>
                    <RelativeTime iso={revision.createdAt} />
                  </div>
                  <p title={revision.message}>{revision.message}</p>
                  {revision.message.length > 180 && (
                    <details className="vibe-source__history-message">
                      <summary>Show full message</summary>
                      <p>{revision.message}</p>
                    </details>
                  )}
                  <div className="vibe-source__history-meta">
                    <span title={`Discord user ${revision.authorId}`}>
                      by {revision.authorId.length > 12
                        ? `${revision.authorId.slice(0, 6)}…${revision.authorId.slice(-4)}`
                        : revision.authorId}
                    </span>
                    {older && (
                      <Link to={`?from=${older.ordinal}&to=${revision.ordinal}`}>
                        Compare to r{older.ordinal}
                      </Link>
                    )}
                  </div>
                </li>
              );
            })}
          </ol>
        </aside>

        <section className="vibe-source__content">
          {view.kind === 'tree' ? (
            <>
              <div className="vibe-source__content-heading">
                <div>
                  <p>Revision {view.revision}</p>
                  <h2>Files</h2>
                </div>
                <span>{view.files.length} {view.files.length === 1 ? 'file' : 'files'}</span>
              </div>
              {view.files.length > 0 ? (
                <ul className="vibe-source__files">
                  {view.files.map((path) => {
                    const parts = splitPath(path);
                    return (
                      <li key={path}>
                        <Link to={`?revision=${view.revision}&path=${encodeURIComponent(path)}`}>
                          <span className="vibe-source__file-kind">{fileLabel(path)}</span>
                          <span className="vibe-source__file-path">
                            {parts.directory && <span>{parts.directory}</span>}
                            <strong>{parts.name}</strong>
                          </span>
                          <svg viewBox="0 0 16 16" aria-hidden="true">
                            <path d="m6 3 5 5-5 5" />
                          </svg>
                        </Link>
                      </li>
                    );
                  })}
                </ul>
              ) : (
                <p className="vibe-source__empty">This revision has no source files.</p>
              )}
            </>
          ) : view.kind === 'file' ? (
            <>
              <div className="vibe-source__content-heading vibe-source__content-heading--file">
                <div>
                  <Link className="vibe-source__back-link" to={`?revision=${view.revision}`}>
                    <svg viewBox="0 0 16 16" aria-hidden="true"><path d="m10 3-5 5 5 5" /></svg>
                    All files
                  </Link>
                  <p>Revision {view.revision}</p>
                  <h2>{view.path}</h2>
                </div>
                <span>{view.text.split('\n').length} lines</span>
              </div>
              <SourceLines text={view.text} path={view.path} />
            </>
          ) : (
            <>
              <div className="vibe-source__content-heading vibe-source__content-heading--diff">
                <div>
                  <Link className="vibe-source__back-link" to={`?revision=${view.to}`}>
                    <svg viewBox="0 0 16 16" aria-hidden="true"><path d="m10 3-5 5 5 5" /></svg>
                    Revision {view.to}
                  </Link>
                  <p>Changes</p>
                  <h2>r{view.from} <span>→</span> r{view.to}</h2>
                </div>
                <div className="vibe-source__diff-key" aria-label="Diff legend">
                  <span className="is-added">Added</span>
                  <span className="is-removed">Removed</span>
                </div>
              </div>
              <SourceLines text={view.text} diff />
            </>
          )}
        </section>
      </div>
    </main>
  );
}
