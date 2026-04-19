import clsx from "clsx";
import type { ReactNode } from "react";

type Tone = "good" | "warn" | "bad" | "mute";

export function Chip({ tone = "mute", children }: { tone?: Tone; children: ReactNode }) {
  return (
    <span
      className={clsx(
        "chip",
        tone === "good" && "chip-good",
        tone === "warn" && "chip-warn",
        tone === "bad" && "chip-bad",
        tone === "mute" && "chip-mute"
      )}
    >
      {children}
    </span>
  );
}
