"use client";

import clsx from "clsx";
import { scoreColour } from "@/lib/format";

export function ScoreBar({
  label,
  score,
  width = "md",
}: {
  label: string;
  score: number;
  width?: "sm" | "md" | "lg";
}) {
  const tone = scoreColour(score);
  const pct = Math.max(0, Math.min(1, score)) * 100;
  return (
    <div
      className={clsx("grid items-center gap-3", {
        "grid-cols-[1fr_60px_160px]": width === "md",
        "grid-cols-[1fr_60px_120px]": width === "sm",
        "grid-cols-[1fr_80px_240px]": width === "lg",
      })}
    >
      <span className="text-sm text-ink-200 mono truncate">{label}</span>
      <span
        className={clsx("text-right mono text-sm tabular-nums", {
          "text-good": tone === "good",
          "text-warn": tone === "warn",
          "text-bad": tone === "bad",
        })}
      >
        {score.toFixed(3)}
      </span>
      <div className="h-2 bg-ink-700 rounded-full overflow-hidden">
        <div
          className={clsx("h-full rounded-full transition-all", {
            "bg-good": tone === "good",
            "bg-warn": tone === "warn",
            "bg-bad": tone === "bad",
          })}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}
