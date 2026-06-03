interface Props {
  name: string;
  avatarPath?: string | null;
  /** Pixel size. Defaults to 32 (matches the turn header avatar). */
  size?: number;
}

/** Round avatar tile. Uses a cached avatar path when supplied; otherwise
 *  renders stable initials from the display name. */
export default function Avatar({ name, avatarPath, size = 32 }: Props) {
  const label = name || 'user';
  const initial = label.trim()[0]?.toUpperCase() ?? '?';
  const src = avatarPath ? `/avatars/${avatarPath}` : null;

  if (src) {
    return (
      <img
        className="avatar"
        src={src}
        width={size}
        height={size}
        alt={label}
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
        background: colorFor(label),
      }}
      aria-label={label}
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
  const hue = ((h % 360) + 360) % 360;
  return `hsl(${hue}deg 45% 55%)`;
}
