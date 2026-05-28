import type { DiscordUser } from './types';

/** Pick the best display name for a Discord user: their per-guild
 *  display name if set, otherwise their username. Returns `null` when
 *  we have no record of the user (e.g. legacy turns predating
 *  identity tracking). */
export function displayNameFor(user: DiscordUser | undefined): string | null {
  if (!user) return null;
  return user.display_name ?? user.username;
}
