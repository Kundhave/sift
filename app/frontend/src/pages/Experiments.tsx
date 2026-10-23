import { type ReactNode, useMemo } from "react";
import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { matchFiles, useResultIndex } from "../api";
import type { ResultFile } from "../types";
import { DataTree } from "../components/DataTree";
import { Deployment } from "../components/Deployment";
import { XaiCases } from "../components/XaiCases";
import { Gallery } from "../components/Gallery";
import { JsonResult } from "../components/JsonResult";
import { COLORS, axisProps, buildRows, gridStroke, tooltipStyle, useMultiJson } from "../components/charts";
import { Caption, ErrorBox, LIMITS, Limits, NotAvailable, PageHeader, Section, SkeletonBlock } from "../components/ui";
import { Highlights } from "../components/Highlights";
import { f3 } from "../format";

const isJson = (f: ResultFile) => f.path.toLowerCase().endsWith(".json");
const RE_GAN = /gan/i, RE_NOISE = /denoise|noise/i, RE_KM = /kmeans|k_means|cluster/i, RE_XAI = /xai|faith|attrib/i, RE_ABL = /abl/i;
const RE_OTHER = /(^|\/)(eval_|baselines|consolidation|data_verification|testing_|test_summary|coverage)/i;
const RE_TRAIN = /train|config|hpo|optuna|distill|student/i;

/** Line chart for any `history: [{epoch, ...numbers}]` array found in a result file. */
function HistoryChart({ rows, title }: { rows: Record<string, any>[]; title: string }) {
  const keys = Object.keys(rows[0] ?? {}).filter((k) => !/^(epoch|step|elapsed|epoch_s)/.test(k) && typeof rows[0][k] === "number").slice(0, 5);
  const xk = "epoch" in (rows[0] ?? {}) ? "epoch" : "step";
  if (!keys.length) return null;
  return (
    <div className="card p-3">
      <div className="mb-1 text-[13px] font-semibold">{title}</div>
      <ResponsiveContainer width="100%" height={230}>
        <LineChart data={rows} margin={{ left: -10, right: 12 }}>
          <CartesianGrid stroke={gridStroke} vertical={false} /><XAxis dataKey={xk} {...axisProps} /><YAxis {...axisProps} domain={["auto", "auto"]} />
          <Tooltip {...tooltipStyle} formatter={(v: number) => v.toFixed(4)} /><Legend wrapperStyle={{ fontSize: 12 }} />
          {keys.map((k, i) => <Line key={k} type="monotone" dataKey={k} stroke={COLORS[i % COLORS.length]} strokeWidth={2} dot={rows.length < 25} />)}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

function Block({ children }: { children: ReactNode }) { return <div className="space-y-3">{children}</div>; }

export default function ExperimentsPage() {
  const { files, error, loading } = useResultIndex();
  const evalPaths = useMemo(() => matchFiles(files, /(^|\/)eval_[^/]*\.json$/).map((f) => f.path), [files]);
  const js = useMemo(() => (files ?? []).filter(isJson), [files]);
  const pick = (re: RegExp) => js.filter((f) => re.test(f.path) && !/(^|\/)eval_/.test(f.path));
  const gan = pick(RE_GAN), noise = pick(RE_NOISE).filter((f) => !RE_GAN.test(f.path)), km = pick(RE_KM), xai = pick(RE_XAI);
  const hasDeploy = js.some((f) => f.path === "latency_size_onnx.json");
  const abl = pick(RE_ABL).filter((f) => !RE_GAN.test(f.path));
  const train = js.filter((f) => RE_TRAIN.test(f.path) && !RE_OTHER.test(f.path) && !RE_GAN.test(f.path) && !RE_NOISE.test(f.path) && !RE_KM.test(f.path) && !RE_XAI.test(f.path) && !RE_ABL.test(f.path));
  const pngs = (re: RegExp) => matchFiles(files, re, ".png");

  const paths = useMemo(() => [...evalPaths, ...train.map((f) => f.path), ...gan.map((f) => f.path)], [evalPaths, train, gan]);
  const { data, loading: jl } = useMultiJson(paths);
  const ablRows = useMemo(() => buildRows(Object.fromEntries(Object.entries(data).filter(([p]) => /eval_/.test(p))), null, null, "full")
    .filter((r) => ["Main", "Ablation", "Student"].includes(r.group)), [data]);
  const tiered = useMemo(() => ({
    random: buildRows(Object.fromEntries(Object.entries(data).filter(([p]) => /eval_/.test(p))), null, null, "tier_random"),
    hard: buildRows(Object.fromEntries(Object.entries(data).filter(([p]) => /eval_/.test(p))), null, null, "tier_hard"),
  }), [data]);

  if (loading) return <><PageHeader title="Experiments" /><SkeletonBlock rows={8} /></>;
  if (error) return <><PageHeader title="Experiments" /><ErrorBox message={error} /></>;

  return (
    <div className="space-y-9">
      <PageHeader title="Experiments" sub="Raw experiment outputs, rendered exactly as the scripts wrote them. A section stays empty until its result file exists." />
      <Highlights page="experiments" />

      <Section title="Training runs and settings" hint="Hardware, hyper-parameters, epochs and timing from run configuration files.">
        {jl && !train.length ? <SkeletonBlock rows={3} /> : train.length === 0 && evalPaths.length === 0 ? <NotAvailable what="Training run info" expects="a *_train*.json / *_config.json file" /> : (
          <Block>
            {train.map((f) => (
              <div key={f.path} className="space-y-3">
                {/smoke/i.test(f.path) && <p className="rounded-lg border border-warn/30 bg-warn/5 p-2.5 text-[12.5px]">This is a smoke run on a tiny subset to prove the loop works. It is not the reported model.</p>}
                {Array.isArray(data[f.path]?.history) && data[f.path].history.length > 0 && <HistoryChart rows={data[f.path].history} title={`Training history: ${f.path}`} />}
                <JsonResult path={f.path} />
              </div>
            ))}
            {evalPaths.map((p) => data[p]?.info && Object.keys(data[p].info).length > 1 && (
              <div key={p} className="card p-4"><div className="mb-2 text-[13px] font-semibold">Evaluation environment: {data[p].name ?? p}</div><DataTree value={data[p].info} /></div>
            ))}
          </Block>
        )}
      </Section>

      <Section title="Deployment: size and CPU latency" hint="Distilled MiniLM student exported to ONNX (fp32 and INT8), measured on this machine.">
        {hasDeploy ? <Deployment /> : <NotAvailable what="Latency and size benchmark" expects="latency_size_onnx.json" />}
      </Section>

      <Section title="Ablation" hint="Same test split. Configurations appear as their eval_*.json files are written (hunk-only, intent-only, no consolidation, no focal loss, no multi-task, no denoising, student, INT8).">
        {ablRows.length <= 1 && abl.length === 0 ? <NotAvailable what="Ablation study" expects="eval_abl_*.json" /> : (
          <Block>
            <div className="card overflow-x-auto">
              <table className="w-full min-w-[640px] text-[12.5px] tabular-nums">
                <thead className="text-left text-muted"><tr className="border-b border-line">{["Configuration", "Macro-F1", "Precision", "Recall", "Spec.", "NDCG@5", "AUC random", "AUC hard"].map((h, i) => <th key={h} className={"px-3 py-2 font-medium " + (i ? "text-right" : "")}>{h}</th>)}</tr></thead>
                <tbody>
                  {ablRows.map((r) => {
                    const rr = tiered.random.find((x) => x.id === r.id), hh = tiered.hard.find((x) => x.id === r.id);
                    return (
                      <tr key={r.id} className="border-b border-line/60 last:border-0">
                        <td className="px-3 py-2"><div className="font-medium">{r.name}</div><div className="text-[11px] text-muted">{r.group}</div></td>
                        <td className="px-3 py-2 text-right">{f3(r.m.macro_f1)}</td><td className="px-3 py-2 text-right">{f3(r.m.precision)}</td><td className="px-3 py-2 text-right">{f3(r.m.recall)}</td>
                        <td className="px-3 py-2 text-right">{f3(r.m.specificity)}</td><td className="px-3 py-2 text-right">{f3(r.ndcg5)}</td>
                        <td className="px-3 py-2 text-right">{f3(rr?.aucRandom ?? r.aucRandom)}</td><td className="px-3 py-2 text-right">{f3(hh?.aucHard ?? r.aucHard)}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
            <Caption>Each row differs from the full model in one respect. Read the drop in hard-tier AUC to see what that component contributes.</Caption>
            {abl.map((f) => <JsonResult key={f.path} path={f.path} />)}
            {pngs(RE_ABL).length > 0 && <Gallery files={pngs(RE_ABL)} />}
          </Block>
        )}
      </Section>

      <Section title="Noise and denoising study" hint="Rule filters, confident learning scored against the known 569 label flips, and noise-robustness runs.">
        {noise.length === 0 ? <NotAvailable what="Denoising study" expects="denoise.json" /> : <Block>{noise.map((f) => <JsonResult key={f.path} path={f.path} />)}{pngs(RE_NOISE).length > 0 && <Gallery files={pngs(RE_NOISE)} />}</Block>}
      </Section>

      <Section title="K-means analysis" hint="Clusters of pooled (intent, hunk) embeddings: elbow, silhouette, purity, NMI, hybrid features.">
        {km.length === 0 && pngs(RE_KM).length === 0 ? <NotAvailable what="K-means analysis" expects="kmeans*.json" /> : <Block>{km.map((f) => <JsonResult key={f.path} path={f.path} />)}{pngs(RE_KM).length > 0 && <Gallery files={pngs(RE_KM)} />}</Block>}
      </Section>

      <Section title="GAN comparison" hint="Adversarially trained baseline on frozen (intent, hunk) embeddings. This is the interpretation taken for the course requirement; Sift has no natural GAN.">
        {gan.length === 0 ? <NotAvailable what="GAN comparison" expects="gan.json" /> : (
          <Block>
            {gan.map((f) => {
              const d = data[f.path];
              return (
                <div key={f.path} className="space-y-3">
                  {d?.variants && (
                    <div className="grid gap-4 lg:grid-cols-2">
                      {Object.entries<any>(d.variants).filter(([, v]) => Array.isArray(v.history) && v.history.length).map(([k, v]) => (
                        <HistoryChart key={k} rows={v.history} title={`${k.replace(/_/g, " ")}: losses and val AUC per epoch`} />
                      ))}
                    </div>
                  )}
                  <JsonResult path={f.path} />
                </div>
              );
            })}
            <Caption>Look for instability: a generator diversity value near 1.0 or erratic losses indicate mode collapse. The Results page puts these variants in the master comparison next to the cross-encoder.</Caption>
          </Block>
        )}
      </Section>

      <Section title="Explainability: faithfulness" hint="Integrated Gradients; masking the top-attributed tokens versus random tokens.">
        {xai.length === 0 && pngs(RE_XAI).length === 0 ? <NotAvailable what="XAI faithfulness" expects="xai*.json" /> : <Block>{xai.some((f) => f.path === "xai_cases_ui.json") && <XaiCases />}{xai.filter((f) => f.path !== "xai_cases_ui.json").map((f) => <JsonResult key={f.path} path={f.path} />)}{pngs(RE_XAI).length > 0 && <Gallery files={pngs(RE_XAI)} />}</Block>}
      </Section>

      <Limits items={LIMITS} />
    </div>
  );
}
