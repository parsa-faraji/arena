import Link from "next/link";
import { listRuns, runCost, runScoreSummary } from "@/lib/db";
import { fmtCost, fmtRelative } from "@/lib/format";

export const dynamic = "force-dynamic";

export default async function ComparePicker({
  searchParams,
}: {
  searchParams: Promise<{ a?: string; b?: string }>;
}) {
  const { a, b } = await searchParams;
  const runs = listRuns(20);

  if (a && b) {
    // eslint-disable-next-line react/jsx-pascal-case
    const { default: CompareView } = await import("./compare_view");
    return <CompareView aId={a} bId={b} />;
  }

  return (
    <div className="space-y-8">
      <header className="space-y-2">
        <h1 className="text-3xl">Compare runs</h1>
        <p className="text-ink-300 text-sm max-w-2xl">
          Pick a challenger (A) and a baseline (B). Arena joins them on
          shared case IDs, replays the pairwise judge, and reports the
          win rate plus per-case verdicts.
        </p>
      </header>

      <div className="card overflow-hidden">
        <table className="w-full text-sm">
          <thead className="bg-ink-700/40 text-ink-300 text-xs uppercase tracking-wide">
            <tr>
              <th className="table-cell-pad text-left">run</th>
              <th className="table-cell-pad text-left">variant</th>
              <th className="table-cell-pad text-left">top score</th>
              <th className="table-cell-pad text-right">cost</th>
              <th className="table-cell-pad text-left">started</th>
              <th className="table-cell-pad"></th>
            </tr>
          </thead>
          <tbody>
            {runs.map((run) => {
              const top = [...runScoreSummary(run.id)].sort(
                (x, y) => y.mean - x.mean
              )[0];
              return (
                <tr key={run.id} className="border-t border-ink-700">
                  <td className="table-cell-pad mono text-xs">
                    <Link href={`/runs/${run.id}`}>{run.id.slice(0, 10)}</Link>
                  </td>
                  <td className="table-cell-pad">{run.variant_name}</td>
                  <td className="table-cell-pad">
                    {top ? (
                      <span className="mono text-xs text-ink-300">
                        {top.judge}{" "}
                        <span className="text-ink-100">{top.mean.toFixed(3)}</span>
                      </span>
                    ) : (
                      <span className="text-ink-500 text-xs">—</span>
                    )}
                  </td>
                  <td className="table-cell-pad text-right mono text-xs text-ink-400">
                    {fmtCost(runCost(run.id))}
                  </td>
                  <td className="table-cell-pad text-ink-400 text-xs">
                    {fmtRelative(run.started_at)}
                  </td>
                  <td className="table-cell-pad">
                    <div className="flex gap-2">
                      <Link
                        href={`/compare?a=${run.id}${b ? `&b=${b}` : ""}`}
                        className="chip chip-mute hover:chip-good"
                      >
                        pick A
                      </Link>
                      <Link
                        href={`/compare?b=${run.id}${a ? `&a=${a}` : ""}`}
                        className="chip chip-mute hover:chip-good"
                      >
                        pick B
                      </Link>
                    </div>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {(a || b) && (
        <div className="card p-4 flex items-center justify-between">
          <div className="text-sm text-ink-300">
            {a && <span>A = <code className="mono text-ink-100">{a.slice(0, 10)}</code></span>}
            {a && b && <span className="mx-3 text-ink-500">vs</span>}
            {b && <span>B = <code className="mono text-ink-100">{b.slice(0, 10)}</code></span>}
          </div>
          <Link
            href="/compare"
            className="text-xs text-ink-400 hover:text-ink-200"
          >
            clear
          </Link>
        </div>
      )}
    </div>
  );
}
