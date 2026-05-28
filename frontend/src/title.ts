import { useEffect } from 'react';
import { create } from 'zustand';
import { fetchSiteConfig } from './api';

interface SiteConfigStore {
  /** Tab-title prefix from `/api/config`. Defaults to what
   * `index.html` ships with so there's no flash before the fetch
   * resolves, and so the title stays sane if the fetch fails. */
  titlePrefix: string;
  loaded: boolean;
  load: () => Promise<void>;
}

export const useSiteConfig = create<SiteConfigStore>((set, get) => ({
  titlePrefix: 'grok · ',
  loaded: false,
  load: async () => {
    if (get().loaded) return;
    try {
      const cfg = await fetchSiteConfig();
      set({ titlePrefix: cfg.title_prefix, loaded: true });
    } catch (err) {
      console.warn('failed to load site config', err);
    }
  },
}));

/**
 * Set the browser-tab title to `<prefix><page>` for the lifetime of
 * the calling component. The prefix comes from the server config; the
 * page-specific part is supplied per route (e.g. the conversation
 * title, "Viewer", "Not found"). Re-runs when either changes so the
 * title stays correct as a conversation loads or its title updates.
 */
export function usePageTitle(page: string): void {
  const prefix = useSiteConfig((s) => s.titlePrefix);
  useEffect(() => {
    document.title = `${prefix}${page}`;
  }, [prefix, page]);
}
