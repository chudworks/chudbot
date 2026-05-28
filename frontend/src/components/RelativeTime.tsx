import { useEffect, useState } from 'react';

interface Props {
  /** ISO 8601 timestamp (matches what the backend serializes from
   *  `time::OffsetDateTime`). */
  iso: string;
  /** Optional prefix, e.g. "Started ". Renders inline with the
   *  relative form. */
  prefix?: string;
}

/** Renders a relative-time string ("just now", "2m ago", "yesterday")
 *  with the absolute timestamp on hover. Re-renders every 30s while
 *  mounted so durations don't go stale.
 *
 *  Uses `Intl.RelativeTimeFormat` for localization; falls back to a
 *  simple "<n>m ago" form if the browser is too old (it isn't, but
 *  this keeps the function total). */
export default function RelativeTime({ iso, prefix }: Props) {
  const [, setTick] = useState(0);
  useEffect(() => {
    const handle = setInterval(() => setTick((t) => t + 1), 30_000);
    return () => clearInterval(handle);
  }, []);
  const date = new Date(iso);
  // Guard against an unparseable timestamp. A non-finite Date fed to
  // Intl.RelativeTimeFormat.format() throws a RangeError, which —
  // thrown during render with no error boundary — blanks the whole
  // app. Degrade to showing the raw value instead of crashing.
  if (Number.isNaN(date.getTime())) {
    return (
      <time title={iso}>
        {prefix}
        {iso || 'unknown'}
      </time>
    );
  }
  const absolute = date.toLocaleString();
  const relative = relativeFormat(date);
  return (
    <time dateTime={iso} title={absolute}>
      {prefix}
      {relative}
    </time>
  );
}

const RTF =
  typeof Intl !== 'undefined' && 'RelativeTimeFormat' in Intl
    ? new Intl.RelativeTimeFormat('en', { numeric: 'auto' })
    : null;

function relativeFormat(d: Date): string {
  const seconds = Math.round((d.getTime() - Date.now()) / 1000);
  const abs = Math.abs(seconds);

  if (abs < 5) return 'just now';
  if (abs < 60) return RTF ? RTF.format(seconds, 'second') : `${abs}s ago`;
  const minutes = Math.round(seconds / 60);
  if (Math.abs(minutes) < 60)
    return RTF ? RTF.format(minutes, 'minute') : `${Math.abs(minutes)}m ago`;
  const hours = Math.round(minutes / 60);
  if (Math.abs(hours) < 24)
    return RTF ? RTF.format(hours, 'hour') : `${Math.abs(hours)}h ago`;
  const days = Math.round(hours / 24);
  if (Math.abs(days) < 30)
    return RTF ? RTF.format(days, 'day') : `${Math.abs(days)}d ago`;
  const months = Math.round(days / 30);
  if (Math.abs(months) < 12)
    return RTF ? RTF.format(months, 'month') : `${Math.abs(months)}mo ago`;
  const years = Math.round(days / 365);
  return RTF ? RTF.format(years, 'year') : `${Math.abs(years)}y ago`;
}
