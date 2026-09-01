import { useEffect, useState } from 'react';
import { Link, useParams, useSearchParams } from 'react-router-dom';

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

  if (error) return <main className="vibe-source-state"><h1>Source unavailable</h1><p>{error}</p></main>;
  if (!data) return <main className="vibe-source-state"><p>Loading source…</p></main>;
  const view = data.view;

  return (
    <main className="vibe-source">
      <header className="vibe-source__header">
        <div>
          <p className="vibe-source__eyebrow">Vibe source</p>
          <h1>{data.site.name}</h1>
        </div>
        <a href={data.site.url}>Open site ↗</a>
      </header>
      <div className="vibe-source__layout">
        <aside>
          <h2>History</h2>
          <ol className="vibe-source__history">
            {data.revisions.map((revision, index) => {
              const older = data.revisions[index + 1];
              return (
                <li key={revision.ordinal}>
                  <Link to={`?revision=${revision.ordinal}`}>r{revision.ordinal}</Link>
                  <span>{revision.message}</span>
                  <small>Discord user {revision.authorId}</small>
                  {older && <Link to={`?from=${older.ordinal}&to=${revision.ordinal}`}>Diff from r{older.ordinal}</Link>}
                </li>
              );
            })}
          </ol>
        </aside>
        <section className="vibe-source__content">
          {view.kind === 'tree' ? (
            <>
              <h2>Revision {view.revision}</h2>
              <ul className="vibe-source__files">
                {view.files.map((path) => (
                  <li key={path}>
                    <Link to={`?revision=${view.revision}&path=${encodeURIComponent(path)}`}>{path}</Link>
                  </li>
                ))}
              </ul>
            </>
          ) : (
            <>
              <h2>{view.kind === 'file' ? `r${view.revision} / ${view.path}` : `Diff r${view.from} → r${view.to}`}</h2>
              <pre><code>{view.text}</code></pre>
            </>
          )}
        </section>
      </div>
    </main>
  );
}
