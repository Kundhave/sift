import { useMemo, useState } from "react";
import clsx from "clsx";
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { matchFiles, useResultIndex } from "../api";
import { Caption, ErrorBox, LIMITS, Limits, NotAvailable, PageHeader, Section, Segmented, SkeletonBlock, Stat } from "../components/ui";
import { Highlights } from "../components/Highlights";
import { COLORS, ModelRow, axisProps, buildRows, gridStroke, tooltipStyle, useMultiJson } from "../components/charts";
import { Gallery } from "../components/Gallery";
import { f3 } from "../format";

const SUBSETS = [
  { v: "full", label: "All test" }, { v: "tier_random", label: "random tier" }, { v: "tier_hard", label: "hard tier" },
  { v: "python", label: "Python" }, { v: "js_ts", label: "JS/TS" },
];

export const PLANNED_GRAPHS: { name: string; re: RegExp }[] = [
  { name: "Training and validation loss curves", re: /loss|train_curve|learning/i },
  { name: "Precision-recall curves, all models", re: /pr_?curve|precision_?recall/i },
  { name: "ROC curves per tier", re: /roc/i },
  { name: "Confusion matrix", re: /confusion/i },
  { name: "NDCG@K versus K, all models", re: /ndcg/i },
  { name: "Per-tier pairwise AUC bars", re: /pairwise|tier_auc|auc_tier/i },
  { name: "Calibration reliability diagram", re: /reliab|calib/i },
  { name: "Noise-robustness curve", re: /noise|robust/i },
  { name: "Confident-learning detection vs known flips", re: /confident|cl_detect|flip/i },
  { name: "K-means clusters, elbow and silhouette", re: /kmeans|k_means|cluster|umap|tsne|elbow|silhouette/i },
  { name: "Ablation bars", re: /abl/i },
  { name: "Latency versus size scatter", re: /latency|size/i },
  { name: "GAN baseline versus main model", re: /gan/i },
  { name: "XAI faithfulness curve", re: /faith|xai/i },
];

type SortKey = keyof ModelRow["m"] | "ndcg5" | "aucRandom" | "aucHard" | "name";
const COLS: { k: SortKey; label: string; lower?: boolean }[] = [
  { k: "macro_f1", label: "Macro-F1" }, { k: "precision", label: "Prec." }, { k: "recall", label: "Recall" }, { k: "specificity", label: "Spec." },
  { k: "roc_auc", label: "ROC-AUC" }, { k: "ndcg5", label: "NDCG@5" }, { k: "aucRandom", label: "AUC random" }, { k: "aucHard", label: "AUC hard" }, { k: "ece", label: "ECE", lower: true },
];
const PRIMARY = new Set<SortKey>(["macro_f1", "ndcg5", "aucRandom", "aucHard", "ece"]);
const val = (r: ModelRow, k: SortKey): number | null => (k === "name" ? null : k === "ndcg5" ? r.ndcg5 : k === "aucRandom" ? r.aucRandom : k === "aucHard" ? r.aucHard : r.m[k] ?? null);

export default function ResultsPage() {
  const { files, error, loading } = useResultIndex();
  const evalPaths = useMemo(() => matchFiles(files, /(^|\/)eval_[^/]*\.json$/).map((f) => f.path), [files]);
  const hasBase = !!files?.some((f) => f.path === "baselines.json");
  const hasGan = !!files?.some((f) => f.path === "gan.json");
  const paths = useMemo(() => [...evalPaths, ...(hasBase ? ["baselines.json"] : []), ...(hasGan ? ["gan.json"] : [])], [evalPaths, hasBase, hasGan]);
  const { data, errors, loading: jl } = useMultiJson(paths);
  const [subset, setSubset] = useState("full");
  const [more, setMore] = useState(false);
  const [sort, setSort] = useState<{ k: SortKey; asc: boolean }>({ k: "ndcg5", asc: false });
  const [cmSel, setCmSel] = useState<string>("");

  const evals = useMemo(() => Object.fromEntries(Object.entries(data).filter(([p]) => /eval_/.test(p))), [data]);
  const rows = useMemo(() => buildRows(evals, data["baselines.json"] ?? null, data["gan.json"] ?? null, subset), [evals, data, subset]);
  const sorted = useMemo(() => [...rows].sort((a, b) => {
    const x = val(a, sort.k), y = val(b, sort.k);
    if (sort.k === "name") return sort.asc ? a.name.localeCompare(b.name) : b.name.localeCompare(a.name);
    if (x == null && y == null) return 0; if (x == null) return 1; if (y == null) return -1;
    return sort.asc ? x - y : y - x;
  }), [rows, sort]);

  const mainRow = rows.find((r) => r.group === "Main");
  const b4 = rows.find((r) => r.name.startsWith("B4"));
  const evalRows = rows.filter((r) => r.full);
  const cm = evalRows.find((r) => r.id === cmSel) ?? mainRow ?? evalRows[0];
  const png = matchFiles(files, /\.png$/i);

  if (loading || (jl && !rows.length)) return <><PageHeader title="Results" /><SkeletonBlock rows={8} /></>;
  if (error) return <><PageHeader title="Results" /><ErrorBox message={error} /></>;

  return (
    <div className="space-y-8">
      <PageHeader title="Results" sub="Every number here is read from a results file written by a script. Test split only (repository-disjoint). Thresholds were tuned on validation." />
      <Highlights page="results" />

      <Section title="Headline metrics" hint={mainRow ? "Main model (ModernBERT cross-encoder), all test hunks." : undefined}>
        {mainRow?.full ? (
          <div className="grid grid-cols-2 gap-3 md:grid-cols-3 xl:grid-cols-6">
            <Stat label="Macro-F1" value={f3(mainRow.full.macro_f1)} sub={b4 ? undefined : undefined} />
            <Stat label="NDCG@5" value={f3(mainRow.full["ndcg@5"])} sub={b4?.ndcg5 != null ? `B4 TF-IDF: ${f3(b4.ndcg5)}` : undefined} />
            <Stat label="AUC random tier" value={f3(mainRow.full.pairwise_auc?.random?.auc)} sub="within-session pairwise" />
            <Stat label="AUC hard tier" value={f3(mainRow.full.pairwise_auc?.hard?.auc)} sub="within-session pairwise" />
            <Stat label="ECE" value={f3(mainRow.full.ece)} sub="lower is better" />
            <Stat label="MRR" value={f3(mainRow.full.mrr)} sub={`${mainRow.full.n_rankable_sessions} rankable sessions`} />
          </div>
        ) : <NotAvailable what="Main model evaluation" expects="eval_full.json" />}
      </Section>

      <Section title="Master model comparison" hint="Click a column header to sort."
        right={<div className="flex flex-wrap items-center gap-2"><button className="btn !py-1 text-[12px]" onClick={() => setMore((m) => !m)} aria-pressed={more}>{more ? "Fewer columns" : "More columns"}</button><Segmented value={subset} onChange={setSubset} options={SUBSETS} /></div>}>
        {Object.keys(errors).length > 0 && <ErrorBox message={`Could not read: ${Object.keys(errors).join(", ")}`} />}
        {sorted.length === 0 ? <NotAvailable what="Model comparison" expects="eval_<model>.json" /> : (
          <div className="card max-h-[34rem] overflow-auto">
            <table className="w-full min-w-[560px] text-[12.5px] tabular-nums">
              <thead className="sticky top-0 bg-surface text-left text-muted">
                <tr className="border-b border-line">
                  <th className="px-3 py-2 font-medium"><button onClick={() => setSort({ k: "name", asc: sort.k === "name" ? !sort.asc : true })}>Model</button></th>
                  {COLS.filter((c) => more || PRIMARY.has(c.k)).map((c) => (
                    <th key={c.k} aria-sort={sort.k === c.k ? (sort.asc ? "ascending" : "descending") : "none"} className="px-2 py-2 text-right font-medium">
                      <button onClick={() => setSort({ k: c.k, asc: sort.k === c.k ? !sort.asc : !!c.lower })} className="hover:text-fg">
                        {c.label}{sort.k === c.k ? (sort.asc ? " ▲" : " ▼") : ""}
                      </button>
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {sorted.map((r) => (
                  <tr key={r.id} className={clsx("border-b border-line/60 last:border-0", r.group === "Main" && "bg-accent/5")}>
                    <td className="px-3 py-2"><div className="font-medium">{r.name}</div><div className="text-[11px] text-muted">{r.group}</div></td>
                    {COLS.filter((c) => more || PRIMARY.has(c.k)).map((c) => <td key={c.k} className="px-2 py-2 text-right">{f3(val(r, c.k))}</td>)}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        <p className="text-[12px] text-muted">"-" means that metric is not in the source file for this subset. Non-neural and GAN rows carry classification columns only on "All test". NDCG@5 and AUC for those rows come from their own result files and are not subset-specific.</p>
      </Section>

      {rows.length > 0 && (
        <div className="grid gap-6 lg:grid-cols-2">
          <Section title="NDCG@5 by model">
            <div className="card p-3">
              <ResponsiveContainer width="100%" height={Math.max(220, rows.length * 30)}>
                <BarChart data={[...rows].filter((r) => r.ndcg5 != null).sort((a, b) => b.ndcg5! - a.ndcg5!)} layout="vertical" margin={{ left: 10, right: 20 }}>
                  <CartesianGrid horizontal={false} stroke={gridStroke} /><XAxis type="number" domain={[0.6, 1]} {...axisProps} />
                  <YAxis type="category" dataKey="name" width={150} {...axisProps} />
                  <Tooltip {...tooltipStyle} formatter={(v: number) => v.toFixed(4)} />
                  <Bar dataKey="ndcg5" name="NDCG@5" fill={COLORS[0]} radius={[0, 4, 4, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
            <Caption>NDCG@5 has a floor near 0.72 because the median session has 2 hunks, so random order already scores well. Judge models by the per-tier AUC next to it.</Caption>
          </Section>
          <Section title="Within-session pairwise AUC by tier">
            <div className="card p-3">
              <ResponsiveContainer width="100%" height={Math.max(220, rows.length * 30)}>
                <BarChart data={rows.filter((r) => r.aucRandom != null || r.aucHard != null)} layout="vertical" margin={{ left: 10, right: 20 }}>
                  <CartesianGrid horizontal={false} stroke={gridStroke} /><XAxis type="number" domain={[0.4, 1]} {...axisProps} />
                  <YAxis type="category" dataKey="name" width={150} {...axisProps} />
                  <Tooltip {...tooltipStyle} formatter={(v: number) => v.toFixed(4)} /><Legend wrapperStyle={{ fontSize: 12 }} />
                  <Bar dataKey="aucRandom" name="random tier" fill={COLORS[1]} radius={[0, 4, 4, 0]} />
                  <Bar dataKey="aucHard" name="hard tier" fill={COLORS[2]} radius={[0, 4, 4, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
            <Caption>An AUC of 0.5 is chance. The hard tier mines donor hunks with similar identifiers, so it separates models that only match surface words from those that read intent.</Caption>
          </Section>
        </div>
      )}

      {cm?.full && (
        <Section title="Confusion matrix" hint="All test hunks at the validation-tuned threshold."
          right={evalRows.length > 1 ? (
            <select className="input !w-auto" value={cm.id} onChange={(e) => setCmSel(e.target.value)} aria-label="Model for confusion matrix">
              {evalRows.map((r) => <option key={r.id} value={r.id}>{r.name}</option>)}
            </select>) : undefined}>
          <div className="card inline-block p-4">
            <div className="grid grid-cols-[auto_repeat(2,minmax(96px,1fr))] gap-1.5 text-center text-[12px]">
              <div /><div className="text-muted">pred aligned</div><div className="text-muted">pred misaligned</div>
              <div className="self-center text-right text-muted">true aligned</div><Cell v={cm.full.tn} good /><Cell v={cm.full.fp} />
              <div className="self-center text-right text-muted">true misaligned</div><Cell v={cm.full.fn} /><Cell v={cm.full.tp} good />
            </div>
            <p className="mt-2 text-[12px] text-muted">{cm.name}, threshold {Number(cm.full.threshold).toFixed(2)}</p>
          </div>
          <Caption>False negatives (bottom left) are misaligned hunks the model let through; false positives cost reviewer time.</Caption>
        </Section>
      )}

      <Section title="Graphs" hint="Every PNG written to results/ appears below. The checklist shows which of the planned graphs exist yet.">
        <ul className="grid gap-1.5 sm:grid-cols-2 xl:grid-cols-3">
          {PLANNED_GRAPHS.map((g) => {
            const found = png.filter((f) => g.re.test(f.path));
            return (
              <li key={g.name} className="flex items-center gap-2 rounded-lg border border-line px-2.5 py-1.5 text-[12.5px]" data-testid="graph-status">
                <span className={clsx("h-2 w-2 rounded-full", found.length ? "bg-ok" : "bg-warn")} />
                <span className="flex-1">{g.name}</span>
                <span className="text-[11px] text-muted">{found.length ? `${found.length} file${found.length > 1 ? "s" : ""}` : "not yet available"}</span>
              </li>
            );
          })}
        </ul>
        {png.length > 0 && <Gallery files={png} />}
      </Section>

      <Limits items={LIMITS} />
      <p className="text-[12px] text-muted">All values are single-run unless a file states several seeds.</p>
    </div>
  );
}

function Cell({ v, good }: { v: number; good?: boolean }) {
  return <div className={clsx("rounded-lg p-3 text-lg font-semibold tabular-nums", good ? "bg-ok/15 text-ok" : "bg-bad/10 text-bad")}>{v.toLocaleString()}</div>;
}
