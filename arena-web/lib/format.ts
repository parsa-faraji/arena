export function fmtDate(raw: string): string {
  const d = new Date(raw);
  if (Number.isNaN(d.getTime())) return raw;
  return d.toLocaleString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export function fmtRelative(raw: string): string {
  const d = new Date(raw);
  if (Number.isNaN(d.getTime())) return raw;
  const diff = Date.now() - d.getTime();
  const mins = Math.floor(diff / 60000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins}m ago`;
  const hours = Math.floor(mins / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  if (days < 30) return `${days}d ago`;
  return d.toLocaleDateString();
}

export function fmtCost(cents: number): string {
  if (cents < 0.01) return `$${cents.toFixed(4)}`;
  return `$${cents.toFixed(3)}`;
}

export function fmtTokens(n: number): string {
  if (n < 1000) return String(n);
  if (n < 1_000_000) return `${(n / 1000).toFixed(1)}k`;
  return `${(n / 1_000_000).toFixed(1)}M`;
}

export function scoreColour(score: number): "good" | "warn" | "bad" {
  if (score >= 0.8) return "good";
  if (score >= 0.5) return "warn";
  return "bad";
}

export function statusColour(status: string): "good" | "warn" | "bad" | "mute" {
  switch (status) {
    case "done":
      return "good";
    case "running":
      return "warn";
    case "error":
      return "bad";
    default:
      return "mute";
  }
}
