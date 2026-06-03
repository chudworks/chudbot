import { create } from 'zustand';
import type { ConversationSnapshot } from './types';
import { fetchConversation } from './api';

type LoadState =
  | { kind: 'idle' }
  | { kind: 'loading' }
  | { kind: 'ready'; view: ConversationSnapshot }
  | { kind: 'error'; status: number; message: string };

interface ConversationStore {
  /** ID of the conversation currently in the store. Used by SSE
   * handlers to decide whether an incoming event is relevant. */
  id: string | null;
  state: LoadState;
  /** Bumped whenever a refetch completes. Lets components animate
   * "something changed" UI without a structural diff. */
  revision: number;
  load: (id: string) => Promise<void>;
  refresh: () => Promise<void>;
  clear: () => void;
}

export const useConversation = create<ConversationStore>((set, get) => ({
  id: null,
  state: { kind: 'idle' },
  revision: 0,
  load: async (id) => {
    set({ id, state: { kind: 'loading' } });
    try {
      const view = await fetchConversation(id);
      // Bail out if the user navigated away mid-fetch.
      if (get().id !== id) return;
      set((s) => ({
        state: { kind: 'ready', view },
        revision: s.revision + 1,
      }));
    } catch (err: unknown) {
      if (get().id !== id) return;
      const status = err instanceof Error && 'status' in err ? (err as { status: number }).status : 0;
      const message = err instanceof Error ? err.message : String(err);
      set({ state: { kind: 'error', status, message } });
    }
  },
  refresh: async () => {
    const id = get().id;
    if (!id) return;
    try {
      const view = await fetchConversation(id);
      if (get().id !== id) return;
      set((s) => ({
        state: { kind: 'ready', view },
        revision: s.revision + 1,
      }));
    } catch (err) {
      // Soft-fail refresh — keep the last-known state on screen rather
      // than wiping it out on a transient network hiccup.
      console.warn('refresh failed', err);
    }
  },
  clear: () => set({ id: null, state: { kind: 'idle' } }),
}));
