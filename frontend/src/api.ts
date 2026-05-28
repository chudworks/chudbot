import type { ConversationView } from './types';

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
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
