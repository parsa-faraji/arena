/**
 * Capture dashboard screenshots for the README.
 *
 * Prereqs:
 *   1. Seed the DB: `cd examples/support-triage && python demo_drive.py`
 *      (produces a seeded arena.db in that folder).
 *   2. Start the dev server pointing at that DB:
 *      `ARENA_DB=../examples/support-triage/arena.db npm run dev`
 *
 * Then run: `npm run screenshots`
 *
 * Writes to ../docs/screenshots/ so the README can embed them directly.
 */
import { chromium } from "@playwright/test";
import Database from "better-sqlite3";
import path from "node:path";
import fs from "node:fs";

const BASE = process.env.BASE_URL ?? "http://localhost:3000";
const DB_PATH = process.env.ARENA_DB ?? path.resolve(
  __dirname,
  "../../examples/support-triage/arena.db"
);
const OUT = path.resolve(__dirname, "../../docs/screenshots");

const VIEWPORT = { width: 1440, height: 900 };

async function main() {
  fs.mkdirSync(OUT, { recursive: true });

  if (!fs.existsSync(DB_PATH)) {
    console.error(
      `[screenshots] arena.db not found at ${DB_PATH}\n` +
      `  run: cd examples/support-triage && python demo_drive.py`
    );
    process.exit(1);
  }

  // Prefer the headline story (v0-baseline vs v1-optimized) when those
  // variants exist; otherwise fall back to the two most recent runs.
  const db = new Database(DB_PATH, { readonly: true });
  const pickNamed = (name: string) =>
    db
      .prepare(
        "SELECT r.id, v.name FROM run r JOIN variant v ON v.id = r.variant_id " +
          "WHERE v.name = ? ORDER BY r.started_at DESC LIMIT 1"
      )
      .get(name) as { id: string; name: string } | undefined;

  const headliner = pickNamed("v1-optimized");
  const baseline = pickNamed("v0-baseline");

  let latest: { id: string; name: string };
  let previous: { id: string; name: string };
  if (headliner && baseline) {
    latest = headliner;
    previous = baseline;
  } else {
    const recent = db
      .prepare(
        "SELECT r.id, v.name FROM run r JOIN variant v ON v.id = r.variant_id " +
          "ORDER BY r.started_at DESC LIMIT 2"
      )
      .all() as { id: string; name: string }[];
    if (recent.length < 2) {
      console.error(`[screenshots] need at least 2 runs, found ${recent.length}`);
      process.exit(1);
    }
    [latest, previous] = recent;
  }
  db.close();
  console.log(`[screenshots] using runs:`);
  console.log(`  latest:   ${latest.id.slice(0, 10)}  (${latest.name})`);
  console.log(`  previous: ${previous.id.slice(0, 10)}  (${previous.name})`);

  const browser = await chromium.launch();
  const context = await browser.newContext({ viewport: VIEWPORT, deviceScaleFactor: 2 });
  const page = await context.newPage();

  const shots: [string, string, () => Promise<void>][] = [
    [
      "01-runs-list.png",
      `${BASE}/`,
      async () => { await page.waitForSelector("table"); },
    ],
    [
      "02-run-detail.png",
      `${BASE}/runs/${latest.id}`,
      async () => { await page.waitForSelector("h1"); },
    ],
    [
      "03-compare.png",
      `${BASE}/compare?a=${latest.id}&b=${previous.id}`,
      async () => { await page.waitForSelector("h1"); },
    ],
  ];

  for (const [name, url, waitFn] of shots) {
    console.log(`[screenshots] -> ${name}  ${url}`);
    await page.goto(url, { waitUntil: "networkidle" });
    await waitFn();
    await page.waitForTimeout(300); // let fonts settle
    const out = path.join(OUT, name);
    await page.screenshot({ path: out, fullPage: true });
    console.log(`[screenshots]    saved ${out}`);
  }

  await browser.close();
  console.log(`[screenshots] done -> ${OUT}`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
