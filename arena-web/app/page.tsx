import Link from "next/link";
import { listRuns, runCost, runScoreSummary } from "@/lib/db";
import { Chip } from "@/components/Chip";
import { fmtCost, fmtRelative, statusColour } from "@/lib/format";

export const dynamic = "force-dynamic";

export default function RunsPage() {
  const runs = listRuns(30);

  return (
    <div className="space-y-8">
      <header className="space-y-2">
        <h1 className="text-3xl">Runs</h1>
        <p className="text-ink-300 text-sm max-w-2xl">
          Every invocation of <code>arena run</code> lands here. Each row is
          one variant executed over a dataset via Respan Gateway, with
          scores from the configured evaluators and judges.
        </p>
      </header>

      <div className="card overflow-hidden">
        <table className="w-full text-sm">
          <thead className="bg-ink-700/40 text-ink-300 text-xs uppercase tracking-wide">
            <tr>
              <th className="table-cell-pad text-left">run</th>
              <th className="table-cell-pad text-left">variant</th>
              <th className="table-cell-pad text-left">model</th>
              <th className="table-cell-pad text-left">status</th>
              <th className="table-cell-pad text-right">cases</th>
              <th className="table-cell-pad text-left">top score</th>
              <th className="table-cell-pad text-right">cost</th>
              <th className="table-cell-pad text-left">started</th>
            </tr>
          </thead>
          <tbody>
            {runs.length === 0 && (
              <tr>
                <td colSpan={8} className="table-cell-pad text-center text-ink-400 py-10">
                  No runs yet. Run{" "}
                  <code className="mono text-ink-200">arena run v0 --cases 20</code>{" "}
                  to seed the dashboard.
                </td>
              </tr>
            )}
            {runs.map((run) => {
              const summary = runScoreSummary(run.id);
              const top = [...summary].sort((a, b) => b.mean - a.mean)[0];
              const cost = runCost(run.id);
              return (
                <tr
                  key={run.id}
                  className="border-t border-ink-700 hover:bg-ink-800/60 transition-colors"
                >
                  <td className="table-cell-pad">
                    <Link href={`/runs/${run.id}`} className="mono">
                      {run.id.slice(0, 10)}
                    </Link>
                  </td>
                  <td className="table-cell-pad text-ink-100">{run.variant_name}</td>
                  <td className="table-cell-pad text-ink-300 mono text-xs">
                    {run.variant_model}
                  </td>
                  <td className="table-cell-pad">
                    <Chip tone={statusColour(run.status)}>{run.status}</Chip>
                  </td>
                  <td className="table-cell-pad text-right tabular-nums text-ink-300">
                    {run.completed_cases}/{run.total_cases}
                  </td>
                  <td className="table-cell-pad">
                    {top ? (
                      <div className="flex items-center gap-2">
                        <span className="text-xs text-ink-400 mono">
                          {top.judge}
                        </span>
                        <span className="tabular-nums text-ink-100">
                          {top.mean.toFixed(3)}
                        </span>
                      </div>
                    ) : (
                      <span className="text-ink-500 text-xs">no scores</span>
                    )}
                  </td>
                  <td className="table-cell-pad text-right tabular-nums text-ink-300 mono">
                    {fmtCost(cost)}
                  </td>
                  <td className="table-cell-pad text-ink-400 text-xs">
                    {fmtRelative(run.started_at)}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
