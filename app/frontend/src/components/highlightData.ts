/* Computes the "What worked best" items from the real result files. Nothing here is typed in by hand:
   every number is derived from JSON served by /api/results/*. A missing file yields no item. */
/* eslint-disable @typescript-eslint/no-explicit-any */
import { f3 } from "../format";

export type Tag = "helped" | "no clear effect" | "did not help";
export interface Item {
  id: string; label: string; value: string; detail?: string; criterion: string;
  lines?: { tag: Tag; text: string }[]; note?: string;
}
export type Data = Record<string, any | null>;
export type PageKey = "review" | "results" | "experiments" | "dataset" | "testing" | "related" | "about";

const NOISE = 0.02; // macro-F1 differences below this are treated as within single-seed noise (stated on the page)
export const FILES: Record<PageKey, string[]> = {
  review: ["master_comparison.json", "latency_full.json", "latency_size_onnx.json", "ablation.json"],
  results: ["master_comparison.json", "ablation.json", "calibration.json", "latency_size_onnx.json"],
  experiments: ["ablation.json", "noise_robustness.json", "kmeans.json", "xai.json", "confident_learning.json"],
  dataset: ["/api/dataset/stats"],
  testing: ["test_summary.json"],
  related: ["master_comparison.json"],
  about: ["master_comparison.json", "latency_size_onnx.json"],
};
export const MACHINE_FILE = "latency_full.json";

const isNum = (x: unknown): x is number => typeof x === "number" && Number.isFinite(x);
const rowsOf = (d: any): any[] => (Array.isArray(d?.rows) ? d.rows : []);
const mainOf = (rows: any[]) => rows.find((r) => r.group === "main");
const strip = (s: string) => s.replace(/^\d+[a-z]?\.\s*/, "").replace(/^Main:\s*/, "");
const signed = (x: number) => (x >= 0 ? "+" : "") + x.toFixed(3);

function best(rows: any[], key: string, dir: "max" | "min" = "max") {
  const c = rows.filter((r) => isNum(r[key]));
  if (!c.length) return null;
  return c.reduce((a, b) => ((dir === "max" ? b[key] > a[key] : b[key] < a[key]) ? b : a));
}

const ONNX_NAME: Record<string, string> = { student_onnx_int8: "INT8 student (ONNX)", student_onnx_fp32: "fp32 student (ONNX)" };

function deployment(d: Data) {
  const o = d["latency_size_onnx.json"], f = d["latency_full.json"], t = o?.targets;
  if (!o || !t || !isNum(t.size_mb_lt) || !isNum(t.p95_ms_lt)) return null;
  const cands: { name: string; size: number; p95: number; ndcg?: number; f1?: number }[] = [];
  for (const k of Object.keys(ONNX_NAME)) {
    const m = o[k];
    if (m && isNum(m.size_mb) && isNum(m.latency_ms?.p95_ms)) cands.push({ name: ONNX_NAME[k], size: m.size_mb, p95: m.latency_ms.p95_ms, ndcg: m.test?.["ndcg@5"], f1: m.test?.macro_f1 });
  }
  if (f && isNum(f.size_mb) && isNum(f.cpu?.p95_ms)) cands.push({ name: "ModernBERT-base (torch)", size: f.size_mb, p95: f.cpu.p95_ms });
  const ok = cands.filter((c) => c.size < t.size_mb_lt && c.p95 < t.p95_ms_lt).sort((a, b) => a.size - b.size);
  return { ok, t, full: f && isNum(f.cpu?.p95_ms) ? { p95: f.cpu.p95_ms as number, size: f.size_mb as number } : null };
}

function reviewItems(d: Data): Item[] {
  const rows = rowsOf(d["master_comparison.json"]);
  const out: Item[] = [];
  const trained = rows.filter((r) => ["main", "dl_baseline", "student", "gan"].includes(r.group));
  const bm = best(trained, "auc_hard");
  const main = mainOf(rows);
  if (bm) {
    const frozen = rows.find((r) => r.group === "frozen_embedding" && isNum(r.auc_hard));
    const abl = rowsOf(d["ablation.json"]).filter((r) => r.group === "ablation" && isNum(r.auc_hard) && r.auc_hard > bm.auc_hard);
    const bits: string[] = [];
    if (frozen) bits.push(`untrained frozen cosine: ${f3(frozen.auc_hard)}`);
    if (abl.length) bits.push(`${abl.length} single-seed ablation variant(s) score higher, see Results`);
    out.push({
      id: "best-model", label: "Best model to use", value: strip(bm.model),
      detail: `hard-tier pairwise AUC ${f3(bm.auc_hard)}, NDCG@5 ${f3(bm["ndcg@5"])}`,
      criterion: "highest hard-tier pairwise AUC among the trained neural models (test split)", note: bits.join("; ") || undefined,
    });
  }
  const dep = deployment(d);
  if (dep) {
    const w = dep.ok[0];
    out.push(w ? {
      id: "best-deploy", label: "Best deployment option", value: w.name,
      detail: `${w.size.toFixed(1)} MB, CPU p95 ${w.p95.toFixed(1)} ms${isNum(w.f1) ? `, macro-F1 ${f3(w.f1)}` : ""}`,
      criterion: `smallest model meeting size < ${dep.t.size_mb_lt} MB and p95 < ${dep.t.p95_ms_lt} ms`,
    } : { id: "best-deploy", label: "Best deployment option", value: "No model meets both targets", criterion: `size < ${dep.t.size_mb_lt} MB and p95 < ${dep.t.p95_ms_lt} ms` });
    if (bm && w) {
      const mainNdcg = main?.["ndcg@5"];
      out.push({
        id: "when", label: "When to use which", value: `${strip(bm.model)} or ${w.name}`,
        detail: `Use ${strip(bm.model)} when accuracy on hard cases matters most${dep.full ? ` (${dep.full.size.toFixed(0)} MB, CPU p95 ${dep.full.p95.toFixed(0)} ms)` : ""}. Use the ${w.name} for everyday local review${isNum(w.ndcg) && isNum(mainNdcg) ? `: NDCG@5 ${f3(w.ndcg)} vs ${f3(mainNdcg)}` : ""}.`,
        criterion: "derived from the two items to the left",
      });
    }
  }
  return out;
}

function resultsItems(d: Data): Item[] {
  const master = rowsOf(d["master_comparison.json"]);
  const abl = rowsOf(d["ablation.json"]).filter((r) => r.group === "ablation");
  const all = [...master, ...abl];
  const main = mainOf(master);
  if (!all.length) return [];
  const mk = (id: string, label: string, key: string, tag: string): Item | null => {
    const w = best(all, key);
    if (!w) return null;
    const mv = main?.[key];
    const isMain = w === main;
    const isAbl = abl.includes(w) && !master.includes(w);
    let note = isMain ? "The main model wins this metric." : isNum(mv) ? `Main model: ${f3(mv)}; the winner is ${signed(w[key] - mv)} ahead` : undefined;
    if (!isMain && isNum(mv) && isAbl) note += "; a single-seed ablation variant";
    return { id, label, value: strip(w.model), detail: `${tag} ${f3(w[key])}`, criterion: `best ${tag} over master comparison and ablation rows (test)`, note };
  };
  const out = [
    mk("ndcg", "Best NDCG@5", "ndcg@5", "NDCG@5"), mk("auc", "Best hard-tier AUC", "auc_hard", "hard-tier AUC"), mk("f1", "Best macro-F1", "macro_f1", "macro-F1"),
  ].filter(Boolean) as Item[];
  const c = d["calibration.json"], o = d["latency_size_onnx.json"];
  const eces: { name: string; v: number }[] = [];
  if (isNum(c?.ece_test_after)) eces.push({ name: "Main: ModernBERT-base", v: c.ece_test_after });
  for (const k of Object.keys(ONNX_NAME)) if (isNum(o?.[k]?.calibration?.ece_test_after)) eces.push({ name: ONNX_NAME[k], v: o[k].calibration.ece_test_after });
  if (eces.length) {
    const w = eces.reduce((a, b) => (b.v < a.v ? b : a));
    const mainE = eces.find((e) => e.name.startsWith("Main"));
    out.push({ id: "ece", label: "Best calibration", value: w.name, detail: `ECE ${f3(w.v)} after temperature scaling`, criterion: "lowest test ECE after temperature scaling (lower is better)",
      note: w.name.startsWith("Main") ? "The main model wins this metric." : mainE ? `Main model: ${f3(mainE.v)}` : undefined });
  }
  return out;
}

const COMP: Record<string, string> = { "hunk-only": "intent input", "intent-only": "hunk input", "alignment head only": "multi-task heads", "no intent consolidation": "intent consolidation", "no focal loss (CE)": "focal loss", "no denoising": "denoising" };
const delta = (n: number) => `${n >= 0 ? "+" : ""}${n.toFixed(3)}`;

function experimentsItems(d: Data): Item[] {
  const out: Item[] = [];
  const rows = rowsOf(d["ablation.json"]);
  const full = rows.find((r) => r.run === "full" && r.group === "ablation");
  if (full && isNum(full.macro_f1)) {
    const lines: Item["lines"] = [];
    const cmp = rows.filter((r) => r.group === "ablation" && /^[2-7]\./.test(r.model) && isNum(r.macro_f1));
    const grp: Record<Tag, string[]> = { helped: [], "no clear effect": [], "did not help": [] };
    for (const r of cmp) {
      const dl = r.macro_f1 - full.macro_f1;
      const t: Tag = dl < -NOISE ? "helped" : dl > NOISE ? "did not help" : "no clear effect";
      grp[t].push(`${COMP[strip(r.model)] ?? strip(r.model)} (${delta(dl)})`);
    }
    (Object.keys(grp) as Tag[]).forEach((t) => grp[t].length && lines.push({ tag: t, text: grp[t].join(", ") }));
    if (lines.length) out.push({ id: "ablation", label: "Ablation: component by component", value: `${cmp.length} variants vs full model`,
      criterion: `change in macro-F1 on removing the named component; |change| < ${NOISE} counts as no clear effect (single seed)`, lines });
  }
  const nr = d["noise_robustness.json"]?.results;
  if (Array.isArray(nr)) {
    const rates = [...new Set(nr.map((r: any) => r.noise_rate))].sort() as number[];
    const g: Record<Tag, string[]> = { helped: [], "no clear effect": [], "did not help": [] };
    for (const q of rates) {
      const a = nr.find((r: any) => r.noise_rate === q && r.arm === "no_denoise")?.test?.macro_f1;
      const b = nr.find((r: any) => r.noise_rate === q && r.arm === "denoise")?.test?.macro_f1;
      if (!isNum(a) || !isNum(b)) continue;
      const dl = b - a;
      g[dl > NOISE ? "helped" : dl < -NOISE ? "did not help" : "no clear effect"].push(`${q.toFixed(1)} (${delta(dl)})`);
    }
    const lines: Item["lines"] = [];
    (Object.keys(g) as Tag[]).forEach((t) => g[t].length && lines.push({ tag: t, text: `at label-noise rate ${g[t].join(", ")}` }));
    if (lines.length) out.push({ id: "noise", label: "Denoising in the noise sweep", value: "macro-F1, denoise vs no denoise",
      criterion: `change in macro-F1 per injected noise rate; |change| < ${NOISE} is no clear effect (reduced setting, single seed)`, lines });
  }
  const km = d["kmeans.json"]?.clustering_quality?.test;
  if (km && isNum(km.purity_vs_alignment) && isNum(km.majority_class_rate_alignment)) {
    const gap = km.purity_vs_alignment - km.majority_class_rate_alignment;
    const hy = d["kmeans.json"]?.hybrid?.delta_cluster_minus_baseline?.macro_f1;
    out.push({ id: "kmeans", label: "k-means clusters", value: `purity ${f3(km.purity_vs_alignment)} vs majority rate ${f3(km.majority_class_rate_alignment)}`,
      criterion: "cluster purity against the rate of always predicting the majority class (test)",
      lines: [{ tag: gap > 0.05 ? "helped" : "no clear effect", text: `purity is ${delta(gap)} above the majority rate${isNum(hy) ? `; cluster features change macro-F1 by ${delta(hy)}` : ""}` }] });
  }
  const ig = d["xai.json"]?.faithfulness?.ig;
  if (ig) {
    const ks = Object.keys(ig).sort((a, b) => +a - +b).filter((k) => isNum(ig[k].mean_drop_top) && isNum(ig[k].mean_drop_random));
    if (ks.length) {
      const all = ks.every((k) => ig[k].mean_drop_top > ig[k].mean_drop_random);
      out.push({ id: "xai", label: "XAI faithfulness (Integrated Gradients)", value: `masking top-k vs random-k tokens`,
        criterion: "mean drop in P(misaligned) when top-k attributed tokens are masked vs k random tokens",
        lines: [{ tag: all ? "helped" : "no clear effect", text: ks.map((k) => `k=${k}: ${f3(ig[k].mean_drop_top)} vs ${f3(ig[k].mean_drop_random)}`).join("; ") }] });
    }
  }
  const cl = d["confident_learning.json"]?.detection?.confident_learning, rp = d["confident_learning.json"]?.random_precision;
  if (cl && isNum(cl.precision) && isNum(cl.recall)) {
    const ratio = isNum(rp) && rp > 0 ? cl.precision / rp : null;
    out.push({ id: "cl", label: "Confident learning (flip detection)", value: `precision ${f3(cl.precision)}, recall ${f3(cl.recall)}`,
      criterion: "precision against the random-flagging precision, scored on the known label flips (train)",
      lines: [{ tag: ratio != null && ratio > 2 ? "helped" : "no clear effect", text: `${ratio != null ? `${ratio.toFixed(1)}x random precision (${f3(rp)}), ` : ""}${((1 - cl.precision) * 100).toFixed(0)}% of flagged rows are not flips` }] });
  }
  return out;
}

function datasetItems(d: Data): Item[] {
  const s = d["/api/dataset/stats"];
  if (!s?.train || !s?.val || !s?.test) return [];
  const sp = [s.train, s.val, s.test];
  const hunks = sp.reduce((a, x) => a + x.hunks, 0), sessions = sp.reduce((a, x) => a + x.sessions, 0);
  const repos = sp.map((x) => (x.repos ?? []) as string[]);
  const seen = new Set<string>(); let shared = 0;
  repos.forEach((r) => r.forEach((n) => (seen.has(n) ? shared++ : seen.add(n))));
  const out: Item[] = [
    { id: "scale", label: "Key facts", value: `${hunks.toLocaleString()} hunks, ${seen.size} repositories`, detail: `${sessions.toLocaleString()} sessions; train ${s.train.hunks.toLocaleString()} / val ${s.val.hunks.toLocaleString()} / test ${s.test.hunks.toLocaleString()}`, criterion: "counts from the dataset statistics file" },
    { id: "split", label: "Strongest design property", value: `${shared} repositories shared across splits`, detail: "Split by repository; labels follow from construction, not manual judgement.", criterion: "overlap of repository lists between train, validation and test" },
  ];
  if (isNum(s.rows_with_any_noise) && hunks) out.push({ id: "noise", label: "Noise", value: `${((s.rows_with_any_noise / hunks) * 100).toFixed(1)}% of rows carry noise`, detail: `${s.rows_with_any_noise.toLocaleString()} rows, shipped with ground truth`, criterion: "rows with at least one noise type over all hunks" });
  out.push({ id: "limit", label: "Main limitation", value: "No human labels", detail: "Misaligned hunks are constructed donors from other commits, so a model can learn construction artefacts.", criterion: "stated in the dataset README" });
  return out;
}

const CAT: Record<string, string> = { redteam: "red-team", blackbox: "black-box", data_pipeline: "data pipeline", whitebox: "white-box", metamorphic: "metamorphic", api: "API", smoke: "smoke" };
const SEV: Record<string, number> = { high: 3, medium: 2, low: 1 };

function testingItems(d: Data): Item[] {
  const t = d["test_summary.json"];
  if (!t) return [];
  const out: Item[] = [];
  const defects: any[] = Array.isArray(t.defects) ? t.defects : [];
  if (defects.length) {
    const by: Record<string, { n: number; top: number; topName: string }> = {};
    for (const x of defects) {
      const stem = String(x.test ?? "").split("::")[0].replace(/^test_/, "").replace(/\.py$/, "");
      const c = CAT[stem] ?? stem.replace(/_/g, " ");
      const sv = SEV[String(x.severity).toLowerCase()] ?? 0;
      const e = (by[c] ??= { n: 0, top: 0, topName: "" });
      e.n++; if (sv > e.top) { e.top = sv; e.topName = x.severity; }
    }
    const [name, e] = Object.entries(by).sort((a, b) => b[1].top - a[1].top || b[1].n - a[1].n)[0];
    out.push({ id: "practice", label: "Best testing practice", value: `${name} testing`, detail: `found ${e.n} of ${defects.length} registered defects, including the highest severity (${e.topName})`,
      criterion: "test category that found the highest-severity defect, ties broken by defect count" });
  }
  const tt = t.totals;
  if (tt && isNum(tt.passed)) out.push({ id: "totals", label: "Test results", value: `${tt.passed} passed, ${tt.xfailed ?? 0} xfailed, ${tt.failed ?? 0} failed`, detail: `${tt.total} tests; xfailed are known defects`, criterion: "totals in test_summary.json" });
  const cov = t.coverage?.percent_covered;
  if (isNum(cov)) out.push({ id: "cov", label: "Coverage", value: `${cov.toFixed(1)}%`, detail: String(t.coverage?.scope ?? ""), criterion: "line coverage from pytest-cov" });
  return out;
}

function relatedItems(d: Data): Item[] {
  const n = rowsOf(d["master_comparison.json"]).length;
  const out: Item[] = [{ id: "reuse", label: "Reused vs new", value: "Reused: encoder and training methods. New: task framing, pipeline, dataset.",
    detail: "ModernBERT, focal loss, temperature scaling, distillation, confident learning and Integrated Gradients are existing. No new architecture is claimed.", criterion: "project brief (no numbers)" }];
  if (n) out.push({ id: "compared", label: "Compared in this repo", value: `${n} models and baselines`, detail: "same split and metrics; published papers are not comparable", criterion: "rows in master_comparison.json" });
  return out;
}

function aboutItems(d: Data): Item[] {
  const rows = rowsOf(d["master_comparison.json"]);
  const bm = best(rows.filter((r) => ["main", "dl_baseline", "student", "gan"].includes(r.group)), "auc_hard");
  const dep = deployment(d)?.ok[0];
  const out: Item[] = [{ id: "what", label: "What it is", value: "A local verification layer for AI coding agents", detail: "Classifies each hunk as Aligned or Misaligned against the stated intent and ranks hunks for review.", criterion: "project brief (no numbers)" }];
  if (bm) out.push({ id: "result", label: "Best result", value: `${strip(bm.model)}: hard-tier AUC ${f3(bm.auc_hard)}`, detail: dep ? `deployable as ${dep.name}: ${dep.size.toFixed(1)} MB, p95 ${dep.p95.toFixed(1)} ms` : undefined, criterion: "highest hard-tier pairwise AUC among trained neural models (test)" });
  return out;
}

export function buildItems(page: PageKey, d: Data): Item[] {
  return { review: reviewItems, results: resultsItems, experiments: experimentsItems, dataset: datasetItems, testing: testingItems, related: relatedItems, about: aboutItems }[page](d);
}
