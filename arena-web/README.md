# arena-web

The read-only dashboard for Arena. Reads the same SQLite `arena.db` the CLI
writes to, so runs appear the moment `arena run` finishes.

## Pages

- `/` — recent runs with status, cost, and top score per run.
- `/runs/[id]` — per-case breakdown, judge scores, and the full prompt text.
- `/compare?a=<run>&b=<run>` — side-by-side score deltas between two runs.

## Development

```bash
# from this directory, with Python demo data seeded
cd ../examples/support-triage && python demo_drive.py
cd ../../arena-web
ARENA_DB=../examples/support-triage/arena.db npm run dev
```

## Screenshots

Run the capture script after the dev server is up:

```bash
npm run screenshots
```

Writes to `../docs/screenshots/`.

## Stack

- Next.js 16 App Router + React 19, TypeScript strict.
- Tailwind v3 for styling; hand-rolled components (no shadcn CLI).
- `better-sqlite3` for direct DB reads (no Python runtime on the UI side).
- `recharts` wired up in case future pages need graphs; current pages use CSS bars.
- Playwright for the screenshot capture script.
