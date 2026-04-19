/**
 * SQLite access for the dashboard. Reads directly from the same
 * `arena.db` that the CLI writes, so there's zero sync concern.
 *
 * Path resolution: `ARENA_DB` env var -> relative path in the parent
 * directory (the typical dev layout where the dashboard lives under
 * the repo). We open the DB read-only so stray concurrent writes can
 * never corrupt it from the UI side.
 */
import Database from "better-sqlite3";
import path from "node:path";
import fs from "node:fs";

export type Run = {
  id: string;
  variant_id: string;
  dataset: string;
  status: string;
  total_cases: number;
  completed_cases: number;
  error: string | null;
  started_at: string;
  finished_at: string | null;
};

export type Variant = {
  id: string;
  name: string;
  prompt: string;
  model: string;
  temperature: number;
  max_tokens: number | null;
  created_at: string;
};

export type CaseResult = {
  id: string;
  run_id: string;
  case_id: string;
  output: string;
  input_tokens: number;
  output_tokens: number;
  latency_ms: number;
  model: string;
  cache_hit: number;
  error: string | null;
};

export type JudgeScore = {
  id: string;
  result_id: string;
  judge: string;
  score: number;
  rationale: string | null;
  raw_json: string | null;
};

export type Case = {
  id: string;
  dataset: string;
  inputs_json: string;
  expected_json: string | null;
};

let _db: Database.Database | null = null;

export function db(): Database.Database {
  if (_db) return _db;
  const candidates = [
    process.env.ARENA_DB,
    "../arena.db",
    "../examples/support-triage/arena.db",
    "./arena.db",
  ].filter(Boolean) as string[];

  for (const rel of candidates) {
    const resolved = path.resolve(process.cwd(), rel);
    if (fs.existsSync(resolved)) {
      _db = new Database(resolved, { readonly: true, fileMustExist: true });
      return _db;
    }
  }
  throw new Error(
    `arena.db not found. Set ARENA_DB or run the CLI first. Tried: ${candidates.join(", ")}`
  );
}

/** Return most-recent runs joined with their variant. */
export function listRuns(limit = 50): (Run & { variant_name: string; variant_model: string })[] {
  const stmt = db().prepare(`
    SELECT r.*, v.name as variant_name, v.model as variant_model
    FROM run r
    LEFT JOIN variant v ON v.id = r.variant_id
    ORDER BY r.started_at DESC
    LIMIT ?
  `);
  return stmt.all(limit) as any;
}

export function getRun(id: string): Run | null {
  const byId = db().prepare("SELECT * FROM run WHERE id = ?").get(id) as Run | undefined;
  if (byId) return byId;
  const byPrefix = db()
    .prepare("SELECT * FROM run WHERE id LIKE ? LIMIT 2")
    .all(`${id}%`) as Run[];
  return byPrefix.length === 1 ? byPrefix[0] : null;
}

export function getVariant(id: string): Variant | null {
  return (db().prepare("SELECT * FROM variant WHERE id = ?").get(id) as Variant) || null;
}

export function getResults(runId: string): CaseResult[] {
  return db()
    .prepare("SELECT * FROM caseresult WHERE run_id = ? ORDER BY case_id")
    .all(runId) as CaseResult[];
}

export function getJudgeScores(resultIds: string[]): JudgeScore[] {
  if (resultIds.length === 0) return [];
  const placeholders = resultIds.map(() => "?").join(",");
  return db()
    .prepare(`SELECT * FROM judgescore WHERE result_id IN (${placeholders})`)
    .all(...resultIds) as JudgeScore[];
}

export function getCase(id: string): Case | null {
  // `case` is a SQL keyword; quote the table name.
  return (db().prepare(`SELECT * FROM "case" WHERE id = ?`).get(id) as Case) || null;
}

/** Aggregate scores by judge name for a run. */
export function runScoreSummary(runId: string): { judge: string; mean: number; n: number }[] {
  return db()
    .prepare(
      `SELECT js.judge as judge,
              AVG(js.score) as mean,
              COUNT(*) as n
       FROM judgescore js
       JOIN caseresult cr ON cr.id = js.result_id
       WHERE cr.run_id = ?
       GROUP BY js.judge
       ORDER BY js.judge`
    )
    .all(runId) as any;
}

/** Cost in USD. We keep the price table in sync with Python's arena/gateway/pricing.py. */
const PRICES: Record<string, { in: number; out: number }> = {
  "gpt-4o-mini": { in: 0.15, out: 0.6 },
  "gpt-4o": { in: 2.5, out: 10 },
  "gpt-4.1": { in: 2.0, out: 8.0 },
  "gpt-4.1-mini": { in: 0.4, out: 1.6 },
  "claude-haiku-4-5": { in: 0.8, out: 4 },
  "claude-sonnet-4-5": { in: 3, out: 15 },
  "claude-sonnet-4-6": { in: 3, out: 15 },
  "claude-opus-4-7": { in: 15, out: 75 },
};

export function costUsd(model: string, inputTokens: number, outputTokens: number): number {
  const key = model.toLowerCase().split("/").pop() ?? model;
  let price: { in: number; out: number } | undefined = PRICES[key];
  if (!price) {
    const match = Object.entries(PRICES).find(([p]) => key.startsWith(p));
    price = match?.[1];
  }
  if (!price) return 0;
  return (inputTokens / 1_000_000) * price.in + (outputTokens / 1_000_000) * price.out;
}

export function runCost(runId: string): number {
  const rows = db()
    .prepare("SELECT model, input_tokens, output_tokens, cache_hit FROM caseresult WHERE run_id = ?")
    .all(runId) as { model: string; input_tokens: number; output_tokens: number; cache_hit: number }[];
  return rows
    .filter((r) => !r.cache_hit)
    .reduce((sum, r) => sum + costUsd(r.model, r.input_tokens, r.output_tokens), 0);
}

export function runTotals(runId: string): { inputTokens: number; outputTokens: number; cacheHits: number } {
  const row = db()
    .prepare(
      `SELECT
         COALESCE(SUM(input_tokens), 0) as inputTokens,
         COALESCE(SUM(output_tokens), 0) as outputTokens,
         COALESCE(SUM(cache_hit), 0) as cacheHits
       FROM caseresult WHERE run_id = ?`
    )
    .get(runId) as { inputTokens: number; outputTokens: number; cacheHits: number };
  return row;
}
