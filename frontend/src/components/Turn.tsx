import type { ContextItem, TurnAsset, TurnSnapshot, UserMetadata } from '../types';
import Avatar from './Avatar';
import RelativeTime from './RelativeTime';
import ToolCall from './ToolCall';

interface Props {
  turnView: TurnSnapshot;
  users: Record<string, UserMetadata>;
}

export default function Turn({ turnView, users }: Props) {
  const { turn, system_instructions, context, tool_trace, replay_assets } = turnView;
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
            ) : (
              <video className="context-video" controls src={asset.path} />
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

type MediaAsset = TurnAsset & { kind: 'image' | 'video'; path: string };

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

function assetKind(asset: TurnAsset): 'image' | 'video' | null {
  const mimeType = asset.mime_type ?? '';
  if (mimeType.startsWith('image/') || isImageUri(asset.uri)) return 'image';
  if (mimeType.startsWith('video/') || isVideoUri(asset.uri)) return 'video';
  return null;
}

// These mirror the file-backed media URI shape exposed by `chudbot-api`.
// Duplicated here because the frontend is the only place that needs them in JS.
function isImageUri(s: string): boolean {
  return s.startsWith('file://images/');
}
function isVideoUri(s: string): boolean {
  return s.startsWith('file://videos/');
}
function toWebPath(s: string): string {
  return s.startsWith('file://') ? '/' + s.slice('file://'.length) : s;
}

function avatarPathFromUri(uri: string | null | undefined): string | null {
  return uri?.startsWith('file://avatars/')
    ? uri.slice('file://avatars/'.length)
    : null;
}

function userKey(user: { platform: string; guild_id: string | null; user_id: string }): string {
  return `${user.platform}:${user.guild_id ?? 'global'}:${user.user_id}`;
}
