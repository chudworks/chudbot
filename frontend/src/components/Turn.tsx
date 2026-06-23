import type {
  ContextItem,
  ReasoningItem,
  ReasoningSummary,
  TurnAsset,
  TurnReasoning,
  TurnView,
  UserMetadata,
} from '../types';
import Avatar from './Avatar';
import RelativeTime from './RelativeTime';
import ToolCall from './ToolCall';

interface Props {
  turnView: TurnView;
  users: Record<string, UserMetadata>;
}

export default function Turn({ turnView, users }: Props) {
  const { turn, system_instructions, context, tool_trace, replay_assets, reasoning } = turnView;
  const user = users[userKey(turn.user)];
  const userLabel = user?.label || turn.user_display_name || 'user';
  const avatarPath = avatarPathFromUri(user?.avatar_media_uri);
  const modelLabel = turn.provider && turn.model ? `${turn.provider}/${turn.model}` : turn.model;

  return (
    <section className="turn">
      <header className="turn__header">
        <span className="turn__index">Turn {turn.ordinal + 1}</span>
        <StatusBadge status={turn.status} />
        {turn.agent_name && (
          <span className="turn__agent">
            · agent <code>{turn.agent_name}</code>
          </span>
        )}
        {modelLabel && (
          <span className="turn__model">
            · model <code>{modelLabel}</code>
          </span>
        )}
        {turn.app_version_id != null && (
          <span className="turn__version">
            · build <code>v{turn.app_version_id}</code>
          </span>
        )}
        <span className="turn__time">
          · <RelativeTime iso={turn.created_at} />
        </span>
      </header>

      <div className="turn__user">
        <div className="turn__user-row">
          <Avatar name={userLabel} avatarPath={avatarPath} size={32} />
          <div className="turn__user-meta">
            <strong>{userLabel}</strong>
            {user?.username && user.username !== userLabel && (
              <span className="turn__username">@{user.username}</span>
            )}
          </div>
        </div>
        <pre className="turn__content">{turn.user_content}</pre>
      </div>

      {system_instructions && (
        <details className="context">
          <summary>System instructions</summary>
          <pre className="turn__content">{system_instructions}</pre>
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

      {tool_trace.length > 0 && (
        <section className="tools">
          <h3>Tool trace ({tool_trace.length})</h3>
          {tool_trace.map((tc, i) => (
            <ToolCall key={i} trace={tc} />
          ))}
        </section>
      )}

      <TurnAssets assets={replay_assets} />

      <ReasoningPanel reasoning={reasoning} />

      <div className="turn__assistant">
        <h3>Assistant</h3>
        {/* A failed turn shows its error in red AND any partial content
            the model managed to produce (e.g. an image-gen failure where
            the model still wrote text). "(no response yet)" only when
            there's genuinely nothing — no content, no error, not failed. */}
        {turn.error && (
          <pre className="turn__content turn__content--err">{turn.error}</pre>
        )}
        {turn.assistant_content ? (
          <pre className="turn__content">{turn.assistant_content}</pre>
        ) : turn.error ? null : turn.status === 'failed' ? (
          <pre className="turn__content turn__content--err">
            (no error message)
          </pre>
        ) : (
          <em>(no response yet)</em>
        )}
      </div>
    </section>
  );
}

function ReasoningPanel({ reasoning }: { reasoning: TurnReasoning }) {
  if (reasoning.items.length === 0 && reasoning.usage.length === 0) return null;

  const summaryCount = reasoning.items.reduce(
    (count, item) => count + item.summary.length,
    0
  );
  const tokens = reasoning.usage.reduce(
    (count, usage) => count + usage.reasoning_tokens,
    0
  );
  const details = [
    tokens > 0 ? `${formatNumber(tokens)} tokens` : null,
    summaryCount > 0
      ? `${formatNumber(summaryCount)} ${summaryCount === 1 ? 'summary' : 'summaries'}`
      : null,
  ].filter(Boolean);

  return (
    <details className="reasoning">
      <summary>Reasoning{details.length > 0 ? ` (${details.join(' · ')})` : ''}</summary>
      {reasoning.usage.length > 0 && (
        <div className="reasoning__usage">
          {reasoning.usage.map((usage) => (
            <span
              className="reasoning__usage-item"
              key={`${usage.provider}:${usage.model ?? ''}`}
            >
              <code>{providerModelLabel(usage.provider, usage.model)}</code>
              {' · '}
              {formatNumber(usage.reasoning_tokens)} tokens
            </span>
          ))}
        </div>
      )}
      {reasoning.items.map((item, index) => (
        <ReasoningItemView
          key={item.id ?? `${item.provider}:${item.model ?? ''}:${index}`}
          item={item}
        />
      ))}
    </details>
  );
}

function ReasoningItemView({ item }: { item: ReasoningItem }) {
  return (
    <article className="reasoning__item">
      <header>
        <code>{providerModelLabel(item.provider, item.model)}</code>
        {item.status && (
          <>
            {' · '}
            <span>{item.status}</span>
          </>
        )}
        {item.id && (
          <>
            {' · '}
            <span>{item.id}</span>
          </>
        )}
      </header>
      {item.summary.map((summary, index) => (
        <ReasoningSummaryView
          key={`${summary.kind ?? 'summary'}:${index}`}
          summary={summary}
        />
      ))}
    </article>
  );
}

function ReasoningSummaryView({ summary }: { summary: ReasoningSummary }) {
  return (
    <section className="reasoning__summary">
      {summary.kind && <div className="reasoning__summary-kind">{summary.kind}</div>}
      <pre className="reasoning__summary-text">{summary.text}</pre>
    </section>
  );
}

function TurnAssets({ assets }: { assets: TurnAsset[] }) {
  const media = collectMediaAssets(assets);
  if (media.length === 0) return null;

  return (
    <section className="turn-assets">
      <h3>Turn media ({media.length})</h3>
      <div className="turn-assets__grid">
        {media.map((asset) => (
          <figure className="turn-assets__item" key={asset.uri}>
            {asset.kind === 'image' ? (
              <img className="context-image" src={asset.path} alt={asset.source} />
            ) : asset.kind === 'video' ? (
              <video className="context-video" controls src={asset.path} />
            ) : (
              <audio className="context-audio" controls src={asset.path} />
            )}
            <figcaption>{asset.source}</figcaption>
          </figure>
        ))}
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
  const isAudio = isAudioUri(item.content);
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
      ) : isAudio ? (
        <audio className="context-audio" controls src={toWebPath(item.content)} />
      ) : (
        <pre>{item.content}</pre>
      )}
    </article>
  );
}

type MediaAsset = TurnAsset & { kind: 'image' | 'video' | 'audio'; path: string };

function collectMediaAssets(assets: TurnAsset[]): MediaAsset[] {
  const seen = new Set<string>();
  const media: MediaAsset[] = [];
  for (const asset of assets) {
    if (seen.has(asset.uri)) continue;
    const kind = assetKind(asset);
    if (!kind) continue;
    seen.add(asset.uri);
    media.push({ ...asset, kind, path: toWebPath(asset.uri) });
  }
  return media;
}

function assetKind(asset: TurnAsset): 'image' | 'video' | 'audio' | null {
  const mimeType = asset.mime_type ?? '';
  if (mimeType.startsWith('image/') || isImageUri(asset.uri)) return 'image';
  if (mimeType.startsWith('video/') || isVideoUri(asset.uri)) return 'video';
  if (mimeType.startsWith('audio/') || isAudioUri(asset.uri)) return 'audio';
  return null;
}

// These mirror the stored media URI shape exposed by `chudbot-api`.
// Duplicated here because the frontend is the only place that needs them in JS.
function isImageUri(s: string): boolean {
  return storedMediaPath(s)?.startsWith('images/') ?? false;
}
function isVideoUri(s: string): boolean {
  return storedMediaPath(s)?.startsWith('videos/') ?? false;
}
function isAudioUri(s: string): boolean {
  return storedMediaPath(s)?.startsWith('audio/') ?? false;
}
function toWebPath(s: string): string {
  const path = storedMediaPath(s);
  return path ? '/' + path : s;
}

function avatarPathFromUri(uri: string | null | undefined): string | null {
  const path = uri ? storedMediaPath(uri) : null;
  return path?.startsWith('avatars/') ? path.slice('avatars/'.length) : null;
}

function storedMediaPath(uri: string): string | null {
  if (uri.startsWith('media://')) return uri.slice('media://'.length);
  if (uri.startsWith('file://')) return uri.slice('file://'.length);
  return null;
}

function userKey(user: { platform: string; guild_id: string | null; user_id: string }): string {
  return `${user.platform}:${user.guild_id ?? 'global'}:${user.user_id}`;
}

function providerModelLabel(provider: string, model: string | null): string {
  return model ? `${provider}/${model}` : provider;
}

function formatNumber(value: number): string {
  return new Intl.NumberFormat().format(value);
}
