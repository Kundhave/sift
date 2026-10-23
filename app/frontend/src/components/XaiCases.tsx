import { useMemo, useState } from "react";
import clsx from "clsx";
import { useApi } from "../api";
import { DiffView } from "./DiffView";
import { Badge, ErrorBox, SkeletonBlock } from "./ui";
import { driftName, pct } from "../format";

interface Case {
  sample_id: string; repo: string; p: number; true: string; drift_true: string; intent: string;
  ui: { intent_words: { word: string; score: number }[]; hunk_lines: { line: string; score: number }[]; hunk: string; file_path: string };
}
const ORDER = ["aligned", "collateral", "scope_expansion", "purpose_drift"];

function tint(s: number, max: number) {
  const a = Math.min(0.55, Math.abs(s) / (max || 1) * 0.55);
  return s >= 0 ? `rgb(var(--bad) / ${a})` : `rgb(var(--accent) / ${a})`;
}

/** Worked Integrated-Gradients examples (one per class, test split) from results/xai_cases_ui.json. */
export function XaiCases({ path = "xai_cases_ui.json" }: { path?: string }) {
  const { data, error, loading, reload } = useApi<Record<string, Case>>(`/api/results/${path}`);
  const keys = useMemo(() => ORDER.filter((k) => data?.[k]), [data]);
  const [sel, setSel] = useState<string | null>(null);
  if (loading) return <SkeletonBlock rows={4} />;
  if (error) return <ErrorBox message={error} onRetry={reload} />;
  const k = sel && data?.[sel] ? sel : keys[0];
  const c = k ? data![k] : null;
  if (!c) return null;
  const maxW = Math.max(...c.ui.intent_words.map((w) => Math.abs(w.score)), 1e-9);
  const maxL = Math.max(...c.ui.hunk_lines.map((w) => Math.abs(w.score)), 1e-9);
  const lineScores = new Map(c.ui.hunk_lines.map((l) => [l.line, l.score / maxL] as [string, number]));
  return (
    <div className="card p-4" data-testid="xai-cases">
      <div className="mb-3 flex flex-wrap gap-1.5" role="tablist" aria-label="Worked examples">
        {keys.map((x) => (
          <button key={x} role="tab" aria-selected={x === k} onClick={() => setSel(x)}
            className={clsx("btn", x === k && "border-accent text-accent")}>{x === "aligned" ? "Aligned" : driftName(x)}</button>
        ))}
      </div>
      <div className="mb-2 flex flex-wrap items-center gap-1.5 text-[12.5px]">
        <span className="font-mono">{c.sample_id}</span><span className="text-muted">· {c.repo} · test split</span>
        <Badge tone={c.p >= 0.5 ? "bad" : "ok"}>P(misaligned) {pct(c.p, 1)}</Badge>
        <Badge>truth: {c.true}{c.drift_true && c.drift_true !== "none" ? ` / ${driftName(c.drift_true)}` : ""}</Badge>
      </div>
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.4fr)] [&>*]:min-w-0">
        <div>
          <div className="label">Intent words</div>
          <p className="rounded-lg bg-raised p-2 text-[13px] leading-7">
            {c.ui.intent_words.map((w, i) => <span key={i} className="rounded px-0.5" style={{ background: tint(w.score, maxW) }} title={w.score.toFixed(4)}>{w.word} </span>)}
          </p>
          <p className="mt-2 text-[11.5px] text-muted">Red pushes the score toward misaligned, blue toward aligned. Attribution is evidence about the model, not proof of the cause.</p>
        </div>
        <div>
          <div className="label">Hunk lines</div>
          <DiffView text={c.ui.hunk} file={c.ui.file_path} lineScores={lineScores} maxLines={30} />
        </div>
      </div>
    </div>
  );
}
