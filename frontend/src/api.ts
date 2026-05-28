import type { ConversationView } from './types';

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

/** Front-end config served from `/api/config`. */
export interface SiteConfig {
  /** Prefix prepended to every browser-tab title. */
  title_prefix: string;
  /** Ordered "vN" build number of the running server. */
  version_number: number;
  /** Full `git describe` string of the running server. */
  git_version: string;
}

/** Fetch the static site config. Soft-fails to a sensible default so a
 * transient hiccup never leaves the tab title broken. */
export async function fetchSiteConfig(): Promise<SiteConfig> {
  const resp = await fetch('/api/config', {
    headers: { Accept: 'application/json' },
  });
  if (!resp.ok) throw new ApiError(resp.status, resp.statusText);
  return (await resp.json()) as SiteConfig;
}

/** Fetch a conversation's full read-model from the backend. */
export async function fetchConversation(id: string): Promise<ConversationView> {
  const resp = await fetch(`/api/conversations/${id}`, {
    headers: { Accept: 'application/json' },
  });
  if (!resp.ok) {
    const body = await resp.text().catch(() => '');
    throw new ApiError(
      resp.status,
      body || resp.statusText || `request failed (${resp.status})`
    );
  }
  return (await resp.json()) as ConversationView;
}
