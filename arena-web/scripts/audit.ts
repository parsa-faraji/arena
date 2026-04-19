/**
 * Pre-launch dashboard audit.
 *
 * Navigates every route, asserts expected content, and records:
 *   - HTTP status per request
 *   - console errors
 *   - unhandled page exceptions
 *   - broken in-page links (clicks through and verifies the target)
 *
 * Exits 1 on any critical finding so this can run in CI.
 */
import { chromium, type ConsoleMessage, type Page } from "@playwright/test";
import Database from "better-sqlite3";
import path from "node:path";

const BASE = process.env.BASE_URL ?? "http://localhost:3000";
const DB_PATH = process.env.ARENA_DB ?? path.resolve(__dirname, "../../arena.db");

type Finding = { severity: "error" | "warn" | "info"; where: string; message: string };

const findings: Finding[] = [];

function record(sev: Finding["severity"], where: string, message: string) {
  findings.push({ severity: sev, where, message });
  const tag = sev === "error" ? "✗" : sev === "warn" ? "!" : "·";
  console.log(`  ${tag} ${where}  ${message}`);
}

function banner(title: string) {
  console.log("\n" + "=".repeat(72));
  console.log("  " + title);
  console.log("=".repeat(72));
}

let currentSection = "init";
let listenersAttached = false;

function attachLogging(page: Page) {
  if (listenersAttached) return;
  listenersAttached = true;

  page.on("console", (msg: ConsoleMessage) => {
    const txt = msg.text();
    if (msg.type() === "error") {
      // Chromium emits "Failed to load resource ... 404" for every
      // intentional 404 navigation; treat that as expected on the 404 probe.
      if (/Failed to load resource.*status of 404/.test(txt) && currentSection === "run-404") return;
      record("error", currentSection, `console.error: ${txt}`);
    } else if (msg.type() === "warning") {
      if (!/Download the React DevTools/.test(txt)) {
        record("warn", currentSection, `console.warn: ${txt.slice(0, 120)}`);
      }
    }
  });
  page.on("pageerror", (err) => {
    record("error", currentSection, `uncaught: ${err.message}`);
  });
  page.on("requestfailed", (req) => {
    const url = req.url();
    const err = req.failure()?.errorText ?? "";
    // Next.js prefetches RSC payloads on Link hover/visible; Playwright
    // aborts those on nav. Not a real failure.
    if (url.includes("_rsc=") && err.includes("ERR_ABORTED")) return;
    if (url.includes("/_next/") && err.includes("ERR_ABORTED")) return;
    record("error", currentSection, `request failed: ${url} (${err})`);
  });
  page.on("response", (resp) => {
    const status = resp.status();
    const url = resp.url();
    // Expected 404 on our explicit negative test.
    if (status === 404 && url.includes("/runs/doesnotexist")) return;
    // Chromium reports the synthetic 404 the dev overlay requests too.
    if (status >= 400 && !url.startsWith("data:")) {
      record("error", currentSection, `${status} ${url}`);
    }
  });
}

async function expectVisible(page: Page, where: string, selector: string, label: string) {
  try {
    await page.waitForSelector(selector, { timeout: 5000 });
    record("info", where, `✓ ${label}`);
  } catch {
    record("error", where, `missing: ${label} (${selector})`);
  }
}

async function expectText(page: Page, where: string, text: string) {
  const body = await page.content();
  if (body.includes(text)) {
    record("info", where, `✓ contains "${text}"`);
  } else {
    record("error", where, `missing text: "${text}"`);
  }
}

async function main() {
  const db = new Database(DB_PATH, { readonly: true });
  const runs = db
    .prepare(
      `SELECT r.id, v.name, r.completed_cases
       FROM run r LEFT JOIN variant v ON v.id = r.variant_id
       WHERE r.completed_cases > 0
       ORDER BY r.started_at DESC LIMIT 2`
    )
    .all() as { id: string; name: string; completed_cases: number }[];
  db.close();

  if (runs.length < 2) {
    console.error(`[audit] need at least 2 runs with results; found ${runs.length}`);
    process.exit(1);
  }
  const [latest, previous] = runs;
  console.log(`[audit] base: ${BASE}`);
  console.log(`[audit] runs: latest=${latest.id.slice(0, 10)} (${latest.name}),` +
    ` previous=${previous.id.slice(0, 10)} (${previous.name})`);

  const browser = await chromium.launch();
  const context = await browser.newContext({
    viewport: { width: 1440, height: 900 },
  });
  const page = await context.newPage();

  attachLogging(page);

  banner("1. Runs list (/)");
  currentSection = "runs-list";
  await page.goto(`${BASE}/`, { waitUntil: "networkidle" });
  await expectVisible(page, "runs-list", "table", "runs table rendered");
  await expectText(page, "runs-list", latest.name);
  await expectText(page, "runs-list", previous.name);
  await expectText(page, "runs-list", "done");
  // Click into the latest run.
  const runLink = page.locator(`a[href="/runs/${latest.id}"]`);
  if (await runLink.count()) {
    await runLink.first().click();
    await page.waitForURL(`**/runs/${latest.id}`, { timeout: 5000 });
    record("info", "runs-list", `✓ click-through to /runs/${latest.id.slice(0, 8)}`);
  } else {
    record("error", "runs-list", `no link for run ${latest.id}`);
  }

  banner("2. Run detail (/runs/[id])");
  currentSection = "run-detail";
  // Already navigated above, but re-load for clean logging.
  await page.goto(`${BASE}/runs/${latest.id}`, { waitUntil: "networkidle" });
  await expectText(page, "run-detail", latest.name);
  await expectText(page, "run-detail", latest.id); // full id visible
  await expectText(page, "run-detail", "Scores");
  await expectText(page, "run-detail", "Cases");
  await expectVisible(page, "run-detail", "table", "cases table rendered");
  // Prompt section visible
  await expectText(page, "run-detail", "Prompt");
  // "← all runs" back link
  const backLink = page.locator('a:has-text("all runs")').first();
  if (await backLink.count()) {
    record("info", "run-detail", "✓ back link present");
  } else {
    record("warn", "run-detail", "back link not found");
  }

  banner("3. Run detail with 8-char prefix");
  currentSection = "run-prefix";
  await page.goto(`${BASE}/runs/${latest.id.slice(0, 10)}`, { waitUntil: "networkidle" });
  await expectText(page, "run-prefix", latest.id); // should still resolve to full id

  banner("4. Run detail: unknown id returns 404");
  currentSection = "run-404";
  const resp = await page.goto(`${BASE}/runs/doesnotexist1234`, { waitUntil: "networkidle" });
  if (resp && resp.status() === 404) {
    record("info", "run-404", "✓ unknown run id -> 404");
  } else {
    record("warn", "run-404", `expected 404, got ${resp?.status()}`);
  }

  banner("5. Compare picker (/compare)");
  currentSection = "compare-picker";
  await page.goto(`${BASE}/compare`, { waitUntil: "networkidle" });
  await expectText(page, "compare-picker", "Compare runs");
  await expectVisible(page, "compare-picker", "table", "compare picker table");
  await expectText(page, "compare-picker", "pick A");
  await expectText(page, "compare-picker", "pick B");

  banner("6. Compare view (/compare?a=...&b=...)");
  currentSection = "compare-view";
  await page.goto(
    `${BASE}/compare?a=${latest.id}&b=${previous.id}`,
    { waitUntil: "networkidle" }
  );
  await expectText(page, "compare-view", latest.name);
  await expectText(page, "compare-view", previous.name);
  await expectText(page, "compare-view", "Score deltas");
  await expectText(page, "compare-view", "Pairwise verdicts");

  banner("7. Navigation + footer");
  currentSection = "nav";
  await page.goto(`${BASE}/`);
  await expectVisible(page, "nav", "nav", "top nav rendered");
  await expectText(page, "nav", "Runs");
  await expectText(page, "nav", "Compare");
  await expectText(page, "nav", "github");

  await browser.close();

  banner("summary");
  const errors = findings.filter((f) => f.severity === "error");
  const warns = findings.filter((f) => f.severity === "warn");
  const infos = findings.filter((f) => f.severity === "info");
  console.log(`  ✓ ${infos.length} checks passed`);
  console.log(`  ! ${warns.length} warnings`);
  console.log(`  ✗ ${errors.length} errors`);
  if (errors.length) {
    console.log("\nerrors:");
    for (const f of errors) console.log(`  - ${f.where}: ${f.message}`);
  }
  process.exit(errors.length ? 1 : 0);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
