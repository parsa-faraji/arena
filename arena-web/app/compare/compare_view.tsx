import Link from "next/link";
import { getResults, getRun, getVariant, runScoreSummary } from "@/lib/db";
import { ScoreBar } from "@/components/ScoreBar";
import { Chip } from "@/components/Chip";

export default function CompareView({
  aId,
  bId,
}: {
  aId: string;
  bId: string;
}) {
  const runA = getRun(aId);
  const runB = getRun(bId);
  if (!runA || !runB) {
    return (
      <div className="card p-8 text-center text-ink-400">
        <p>Couldn&apos;t resolve one of the run ids.</p>
        <Link href="/compare" className="text-accent">
          ← back to picker
        </Link>
      </div>
    );
  }

  const varA = getVariant(runA.variant_id);
  const varB = getVariant(runB.variant_id);
  const scoresA = runScoreSummary(runA.id);
  const scoresB = runScoreSummary(runB.id);

  // Merge judge names so we can align side-by-side.
  const judges = Array.from(
    new Set([...scoresA.map((s) => s.judge), ...scoresB.map((s) => s.judge)])
  ).sort();
  const byJudgeA = new Map(scoresA.map((s) => [s.judge, s.mean] as const));
  const byJudgeB = new Map(scoresB.map((s) => [s.judge, s.mean] as const));

  // Per-case shared ids for win/loss hint.
  const resultsA = getResults(runA.id);
  const resultsB = getResults(runB.id);
  const shared = resultsA.filter((a) =>
    resultsB.some((b) => b.case_id === a.case_id)
  );

  return (
    <div className="space-y-10">
      <header className="space-y-3">
        <Link href="/compare" className="text-ink-400 text-sm hover:text-ink-200">
          ← pick different runs
        </Link>
        <div className="flex items-baseline gap-3 flex-wrap">
          <h1 className="text-3xl">
            <span className="text-accent">{varA?.name ?? "A"}</span>
            <span className="text-ink-500 mx-3">vs</span>
            <span className="text-warn">{varB?.name ?? "B"}</span>
          </h1>
        </div>
        <p className="text-ink-300 text-sm max-w-3xl">
          {shared.length} shared cases. Scores below are averaged across each
          run; the bar colour shows whether the judge score passed (green),
          was borderline (yellow), or regressed (red).
        </p>
      </header>

      <section className="grid grid-cols-2 gap-6">
        <RunCard
          label="A"
          tone="accent"
          runId={runA.id}
          variantName={varA?.name ?? "A"}
          model={varA?.model ?? ""}
          cases={`${runA.completed_cases}/${runA.total_cases}`}
        />
        <RunCard
          label="B"
          tone="warn"
          runId={runB.id}
          variantName={varB?.name ?? "B"}
          model={varB?.model ?? ""}
          cases={`${runB.completed_cases}/${runB.total_cases}`}
        />
      </section>

      <section className="card p-6 space-y-5">
        <h2 className="text-xl">Score deltas</h2>
        <div className="space-y-5">
          {judges.map((judge) => {
            const a = byJudgeA.get(judge) ?? 0;
            const b = byJudgeB.get(judge) ?? 0;
            const delta = a - b;
            return (
              <div key={judge} className="space-y-2">
                <div className="flex items-baseline justify-between">
                  <span className="mono text-sm text-ink-200">{judge}</span>
                  <span
                    className={`mono text-xs tabular-nums ${
                      delta > 0.01
                        ? "text-good"
                        : delta < -0.01
                          ? "text-bad"
                          : "text-ink-400"
                    }`}
                  >
                    {delta > 0 ? "+" : ""}
                    {delta.toFixed(3)}
                  </span>
                </div>
                <div className="grid grid-cols-2 gap-4">
                  <ScoreBar label="A" score={a} width="sm" />
                  <ScoreBar label="B" score={b} width="sm" />
                </div>
              </div>
            );
          })}
        </div>
      </section>

      <section className="card p-6 space-y-3">
        <h2 className="text-lg">Pairwise verdicts</h2>
        <p className="text-sm text-ink-300">
          To run a fresh pairwise shoot-out with an LLM judge, use:
        </p>
        <pre className="mono text-xs bg-ink-900 border border-ink-700 rounded p-3 overflow-x-auto">
          arena judge {runA.id.slice(0, 8)} --pairwise --vs {runB.id.slice(0, 8)}
        </pre>
        <p className="text-xs text-ink-400">
          The CLI prints win rates and per-case verdicts; this dashboard reads
          the resulting <code className="mono">judgescore</code> rows on refresh.
        </p>
      </section>
    </div>
  );
}

function RunCard({
  label,
  tone,
  runId,
  variantName,
  model,
  cases,
}: {
  label: string;
  tone: "accent" | "warn";
  runId: string;
  variantName: string;
  model: string;
  cases: string;
}) {
  return (
    <div className="card p-5 space-y-3">
      <div className="flex items-center gap-3">
        <Chip tone={tone === "accent" ? "good" : "warn"}>{label}</Chip>
        <Link href={`/runs/${runId}`} className="mono text-xs text-ink-400">
          {runId.slice(0, 10)}
        </Link>
      </div>
      <div>
        <div className="text-xl text-ink-50">{variantName}</div>
        <div className="mono text-xs text-ink-400 mt-1">{model}</div>
      </div>
      <div className="text-xs text-ink-400">cases {cases}</div>
    </div>
  );
}
