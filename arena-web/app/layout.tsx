import "./globals.css";
import type { Metadata } from "next";
import Link from "next/link";

export const metadata: Metadata = {
  title: "Arena — LLM regression gym for Respan",
  description:
    "Runs, scores, and pairwise comparisons for your Respan-powered eval suite.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>
        <div className="min-h-screen">
          <nav className="border-b border-ink-700 bg-ink-900/80 backdrop-blur sticky top-0 z-10">
            <div className="max-w-6xl mx-auto flex items-center justify-between px-6 h-14">
              <Link href="/" className="flex items-center gap-2 font-semibold text-ink-50">
                <span className="mono bg-accent/10 border border-accent/30 text-accent px-2 py-0.5 rounded">
                  arena
                </span>
                <span className="text-ink-300 text-sm">LLM regression gym</span>
              </Link>
              <div className="flex items-center gap-5 text-sm">
                <Link href="/">Runs</Link>
                <Link href="/compare">Compare</Link>
                <a href="https://github.com/parsa-faraji/arena" className="text-ink-300">
                  github
                </a>
                <a href="https://respan.ai" className="text-ink-300">
                  respan.ai
                </a>
              </div>
            </div>
          </nav>
          <main className="max-w-6xl mx-auto px-6 py-10">{children}</main>
          <footer className="border-t border-ink-700 mt-16">
            <div className="max-w-6xl mx-auto px-6 py-6 text-xs text-ink-400">
              Arena is a portfolio project showcasing eval frameworks,
              LLM-as-judge pipelines, and prompt optimization for Respan.
              Every call routes through Respan Gateway.
            </div>
          </footer>
        </div>
      </body>
    </html>
  );
}
