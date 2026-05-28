import type { DiscordUser } from '../types';
import { displayNameFor } from '../users';

interface Props {
  user: DiscordUser | undefined;
  /** Fallback text to render as initials when we have no user record
   *  (e.g. historical turns before identity tracking shipped). */
  fallbackName?: string | null;
  /** Pixel size. Defaults to 32 (matches the turn header avatar). */
  size?: number;
}

/** Round avatar tile. Uses the cached avatar at `/avatars/<file>` when
 *  available; otherwise renders the first letter of the user's name
 *  on a colored background. */
export default function Avatar({ user, fallbackName, size = 32 }: Props) {
  const name = displayNameFor(user) ?? fallbackName ?? 'user';
  const initial = name.trim()[0]?.toUpperCase() ?? '?';
  // The avatar fetcher writes files to <avatars_dir>/<file> and the
  // backend serves them at /avatars/<file>.
  const src = user?.avatar_local_path
    ? `/avatars/${user.avatar_local_path}`
    : null;

  if (src) {
    return (
      <img
        className="avatar"
        src={src}
        width={size}
        height={size}
        alt={name}
        loading="lazy"
      />
    );
  }
  return (
    <div
      className="avatar avatar--fallback"
      style={{
        width: size,
        height: size,
        background: colorFor(name),
      }}
      aria-label={name}
    >
      {initial}
    </div>
  );
}

/** Pick a stable color for an initials avatar based on the name. */
function colorFor(name: string): string {
  let h = 0;
  for (let i = 0; i < name.length; i++) {
    h = (h * 31 + name.charCodeAt(i)) | 0;
  }
  // hsl with reasonable saturation/lightness for readability on dark + light.
  const hue = ((h % 360) + 360) % 360;
  return `hsl(${hue}deg 45% 55%)`;
}
